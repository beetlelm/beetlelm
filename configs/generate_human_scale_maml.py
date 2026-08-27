#!/usr/bin/env python3
"""
configs/generate_human_scale_maml.py

Human-scale trilingual continual-learning sweep (EWC + MAML + ExposureLR).

Configurable language triple (default: eng → nld → zho).
Uses smooth curriculum by default (--legacy for discrete phases).

Datasets          : Beetle-HumanScale/{lang}-33M-{tok_triple}  (33M per language, ~100M total)
Vocab size        : 50,000
Steps             : 10,000
Schedule          : Phase-aware log₂ (L2 at 3333, L3 at 6666)

Conditions
  1  baseline               (no EWC, no MAML)
  4  EWC sweep              (λ ∈ {5, 20, 50, 150})
  4  MAML sweep             (ratio ∈ {0.25, 0.5, 0.75, 1.0})
 16  EWC + MAML grid
  4  ExposureLR sweep       (plasticity_boost ∈ {1.5, 2.0, 3.0, 5.0})
  4  EWC + ExposureLR       (λ=20 × boost ∈ {1.5, 2.0, 3.0, 5.0})
──────────────────────────────────
 33  total
"""

import itertools
import sys
from pathlib import Path

import yaml

# ── Shared constants ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config._human_scale_defaults import (
    HF_USER_MODELS, MODEL_BASE, TRAINING_DEFAULTS,
    CHECKPOINTING_DEFAULTS, MAX_STEPS, VOCAB_SIZE, TOKENS_TRILINGUAL,
    get_trilingual_phase_starts,
    babybabel_tri_dataset, babybabel_trilingual_tokenizer,
)
from configs.ckpt_schedule import phase_log2_schedule

# ── Default experimental choices (configurable via CLI) ──────────────────────

DEFAULT_TRIPLE = ("eng", "nld", "zho")
MIX            = "sequential"

OUT_DIR = Path("configs/human_scale_maml")

# ── Sweep grids ───────────────────────────────────────────────────────────────

EWC_LAMBDAS            = [5.0, 20.0, 50.0, 150.0]
MAML_HYBRID_RATIOS     = [0.25, 0.5, 0.75, 1.0]
EXPOSURE_BOOSTS        = [1.5, 2.0, 3.0, 5.0]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _setup_triple(triple):
    """Configure datasets and tokenizer for a language triple."""
    l1, l2, l3 = triple
    datasets = {lang: babybabel_tri_dataset(lang, l1, l2, l3) for lang in triple}
    tokenizer_repo = babybabel_trilingual_tokenizer(l1, l2, l3)
    pk = f"{l1}_L1-{l2}_L2-{l3}_L3"
    phase_starts = get_trilingual_phase_starts(MIX, n_langs=3)
    ckpt_steps = phase_log2_schedule(phase_starts, max_steps=MAX_STEPS)
    return l1, l2, l3, datasets, tokenizer_repo, pk, phase_starts, ckpt_steps


def _base_cfg(run_name: str, l1, l2, l3, datasets, tokenizer_repo,
              ckpt_steps, *, use_exposure_lr: bool = False,
              use_smooth: bool = True) -> dict:
    total_tokens = TOKENS_TRILINGUAL * 3  # 3 languages × 33M

    if use_exposure_lr:
        opt = {
            **TRAINING_DEFAULTS["optimization"],
            "lr_scheduler":  "exposure_based",
            "exposure_lr": {
                "warmup_tokens":    5_000_000,
                "total_tokens":     total_tokens,
                "plasticity_boost": 2.0,          # overridden per-config
                "min_lr_frac":      0.1,
                "decay_shape":      "cosine",
            },
        }
    else:
        opt = dict(TRAINING_DEFAULTS["optimization"])

    training = {
        **TRAINING_DEFAULTS,
        "max_steps":    MAX_STEPS,
        "optimization": opt,
    }
    ckpt = {**CHECKPOINTING_DEFAULTS, "checkpoint_steps": ckpt_steps}

    cfg = {
        "run_name": run_name,
        "hf_user":  HF_USER_MODELS,
        "model": {
            "architecture": MODEL_BASE["architecture"],
            "params": dict(MODEL_BASE["params"]),
        },
        "data": {
            "data_mix":               MIX,
            "lang_sources":           [datasets[l1]],
            "l2_streams":             [datasets[l2], datasets[l3]],
            "pretokenized":           True,
            "pretokenized_cache_dir": f"./pretok_cache/{run_name}",
        },
        "tokenizer": {
            "pretrained_tokenizer_path": tokenizer_repo,
            "dataset_source": "babybabel",
            "vocab_size":     VOCAB_SIZE,
            "overwrite":      False,
            "push_to_hub":    True,
        },
        "training":      training,
        "checkpointing": ckpt,
    }

    if use_smooth:
        cfg["curriculum"] = {
            "type": "smooth",
            "condition": MIX,
            "lang_keys": [l1, l2, l3],
        }

    return cfg


def _ewc_block(lam: float) -> dict:
    return {
        "enabled":          True,
        "lambda_":          lam,
        "n_fisher_samples": 10,
        "mu":               0.0,
        "trigger":          "phase_change",
    }


def _maml_block(ratio: float) -> dict:
    return {
        "enabled":      True,
        "hybrid_ratio": ratio,
        "n_way":        5,
        "k_shot":       1,
        "inner_steps":  3,
        "inner_lr":     0.1,
        "head": {
            "num_layers":  2,
            "hidden_dim":  256,
            "init_method": "xavier",
        },
    }


def _write(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


# ── Generator ─────────────────────────────────────────────────────────────────

def generate(triple=DEFAULT_TRIPLE, use_smooth: bool = True):
    l1, l2, l3, datasets, tokenizer_repo, PK, phase_starts, ckpt_steps = _setup_triple(triple)

    counts = {k: 0 for k in ("baseline", "ewc", "maml", "ewc_maml",
                               "exposure_lr", "ewc_exposure_lr")}

    def _cfg(run_name, use_exposure_lr=False):
        return _base_cfg(run_name, l1, l2, l3, datasets, tokenizer_repo,
                         ckpt_steps, use_exposure_lr=use_exposure_lr,
                         use_smooth=use_smooth)

    # 1. Baseline
    cfg = _cfg(f"beetlelm_{PK}_{MIX}_baseline")
    _write(cfg, OUT_DIR / "baseline" / f"train_{PK}.yaml")
    counts["baseline"] += 1

    # 2. EWC sweep
    for lam in EWC_LAMBDAS:
        tag = f"l{int(lam)}"
        cfg = _cfg(f"beetlelm_{PK}_{MIX}_ewc_{tag}")
        cfg["ewc"] = _ewc_block(lam)
        _write(cfg, OUT_DIR / "ewc" / f"train_{PK}_{tag}.yaml")
        counts["ewc"] += 1

    # 3. MAML sweep
    for ratio in MAML_HYBRID_RATIOS:
        tag = f"h{int(ratio * 100)}"
        cfg = _cfg(f"beetlelm_{PK}_{MIX}_maml_{tag}")
        cfg["maml"] = _maml_block(ratio)
        _write(cfg, OUT_DIR / "maml" / f"train_{PK}_{tag}.yaml")
        counts["maml"] += 1

    # 4. EWC + MAML combined
    for lam, ratio in itertools.product(EWC_LAMBDAS, MAML_HYBRID_RATIOS):
        ltag = f"l{int(lam)}"
        rtag = f"h{int(ratio * 100)}"
        cfg  = _cfg(f"beetlelm_{PK}_{MIX}_ewc_{ltag}_maml_{rtag}")
        cfg["ewc"]  = _ewc_block(lam)
        cfg["maml"] = _maml_block(ratio)
        _write(cfg, OUT_DIR / "ewc_maml" / f"train_{PK}_{ltag}_{rtag}.yaml")
        counts["ewc_maml"] += 1

    # 5. ExposureLR sweep (plasticity_boost only)
    for boost in EXPOSURE_BOOSTS:
        tag = f"b{int(boost * 10)}"
        cfg = _cfg(f"beetlelm_{PK}_{MIX}_exposurelr_{tag}", use_exposure_lr=True)
        cfg["training"]["optimization"]["exposure_lr"]["plasticity_boost"] = boost
        _write(cfg, OUT_DIR / "exposure_lr" / f"train_{PK}_{tag}.yaml")
        counts["exposure_lr"] += 1

    # 6. EWC + ExposureLR (λ=20, varying boost)
    for boost in EXPOSURE_BOOSTS:
        tag = f"l20_b{int(boost * 10)}"
        cfg = _cfg(f"beetlelm_{PK}_{MIX}_ewc_l20_exposurelr_{tag}",
                   use_exposure_lr=True)
        cfg["training"]["optimization"]["exposure_lr"]["plasticity_boost"] = boost
        cfg["ewc"] = _ewc_block(20.0)
        _write(cfg, OUT_DIR / "ewc_exposure_lr" / f"train_{PK}_{tag}.yaml")
        counts["ewc_exposure_lr"] += 1

    total = sum(counts.values())
    print(f"permutation   : {l1} → {l2} → {l3}")
    print(f"mix           : {MIX}")
    print(f"smooth        : {use_smooth}")
    print(f"max_steps     : {MAX_STEPS:,}")
    print(f"vocab_size    : {VOCAB_SIZE:,}")
    print(f"phase_starts  : {phase_starts}")
    print(f"checkpoints   : {len(ckpt_steps)}")
    print(f"\nConditions    : {total}")
    for k, v in counts.items():
        print(f"  {k:<20}: {v}")
    print(f"\nOutput → {OUT_DIR.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--triple", nargs=3, default=list(DEFAULT_TRIPLE),
                        help="Language triple (L1 L2 L3), e.g. eng nld zho")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy discrete phase transitions instead of smooth")
    args = parser.parse_args()
    generate(triple=tuple(args.triple), use_smooth=not args.legacy)
