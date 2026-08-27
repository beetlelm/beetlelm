#!/usr/bin/env python3
"""
configs/generate_human_scale_bilingual.py

Human-scale bilingual configs with smooth curricula and combinatorial sweeps.

Datasets  : Beetle-Data/BabyBabel-{lang}-50M  (50M tokens per language = 100M total)
Vocab size: 50,000 BPE tokens
Steps     : 10,000 (fixed)
Schedule  : Phase-aware log2 checkpoints with extra offsets at phase transitions

Modes:
  Default    : Generate base configs for all conditions + combinatorial sweeps
  --l1/--l2  : Restrict to a single language pair (default: nld/eng)
  --pilot    : Generate sweep for pilot pairs
  --best-config PATH : Read best settings JSON, generate configs for remaining langs

Sweep: MAML x LR x (EWC or LAMOL) combinations on sweep conditions.

Output: configs/human_scale_bilingual/{shared,per_language}/{condition}/train_{pk}.yaml
        configs/human_scale_bilingual/shared/sweep_{condition}/train_{pk}_{combo}.yaml
        configs/human_scale_bilingual/monolingual/train_{lang}.yaml
"""

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import yaml

# -- Shared constants ----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config._human_scale_defaults import (
    HF_ORG, DATASET_SCALE, MODEL_BASE, TRAINING_DEFAULTS,
    CHECKPOINTING_DEFAULTS, LANGS, BILINGUAL_CONDITIONS, MAX_STEPS, VOCAB_SIZE,
    TOKENIZER_TYPE, TOKENIZER_TOKENS, PILOT_L1_LANGS,
    babybabel_l1_dataset, babybabel_l2_dataset, babybabel_mono_dataset,
    babybabel_bilingual_tokenizer, babybabel_mono_tokenizer,
)
from src.config._repo_naming import build_model_run_name
from configs.ckpt_schedule import phase_log2_schedule

# -- Output --------------------------------------------------------------------

OUT_DIR = Path("configs/human_scale_bilingual")

# -- Sweep configuration (combinatorial) ---------------------------------------

SWEEP_CONDITIONS = ["B1", "B2", "B3"]

# 3 values each -> 3x3x3 = 27 combos per set (54 total per pair per condition)
MAML_RATIOS  = [0, 0.2, 0.4]
LR_BOOSTS    = [1.0, 2.0, 3.0]
EWC_LAMBDAS  = [0, 10, 30]        # Set A: MAML x LR x EWC
LAMOL_LAMBDAS = [0, 0.2, 0.5]     # Set B: MAML x LR x LAMOL

# Pilot language pairs for Phase 3a
PILOT_PAIRS = [(l1, "eng") for l1 in PILOT_L1_LANGS]

# -- Pair registry -------------------------------------------------------------

def build_pair_registry():
    valid_pairs, tokenizer_lookup = [], {}
    for l1, l2 in itertools.permutations(LANGS, 2):
        tok_repo = babybabel_bilingual_tokenizer(l1, l2)
        tokenizer_lookup[(l1, l2)] = tok_repo
        valid_pairs.append((l1, l2))
    return sorted(valid_pairs), tokenizer_lookup


PAIRS, TOKENIZER_LOOKUP = build_pair_registry()


def _make_cfg(
    run_name:        str,
    l1_ds:           str,
    l2_ds:           str | None,
    condition:       str,
    tokenizer_repo:  str,
    vocab_sharing:   str,
    pk:              str,
    phase_starts:    list,
    *,
    compression:     str = "equal",
    ewc_lambda=None,
    maml_ratio=None,
    lr_boost=None,
    lamol_lambda=None,
) -> dict:
    extra_offsets = [2, 4, 6, 8] if condition in SWEEP_CONDITIONS else None
    ckpt_steps = phase_log2_schedule(phase_starts, max_steps=MAX_STEPS,
                                      extra_offsets=extra_offsets)
    training = copy.deepcopy(TRAINING_DEFAULTS)
    training["max_steps"] = MAX_STEPS
    ckpt = {**CHECKPOINTING_DEFAULTS, "checkpoint_steps": ckpt_steps}

    cfg = {
        "run_name": run_name,
        "hf_user":  HF_ORG,
        "model": {
            "architecture": MODEL_BASE["architecture"],
            "params": dict(MODEL_BASE["params"]),
        },
        "data": {
            "data_mix":               condition,
            "lang_sources":           [l1_ds],
            "pretokenized":           True,
            "pretokenized_cache_dir": f"./pretok_cache/{run_name}",
        },
        "tokenizer": {
            "pretrained_tokenizer_path": tokenizer_repo,
            "dataset_source": "babybabel",
            "vocab_size":     VOCAB_SIZE,
            "vocab_sharing":  vocab_sharing,
            "type":           TOKENIZER_TYPE,
            "target_tokens":  TOKENIZER_TOKENS,
            "compression":    compression,
            "output_dir":     f"./tokenizers/{pk}_{vocab_sharing}",
            "overwrite":      False,
            "push_to_hub":    True,
        },
        "training":      training,
        "checkpointing": ckpt,
    }

    # L2 streams (bilingual) or omit (monolingual)
    if l2_ds is not None:
        cfg["data"]["l2_streams"] = [l2_ds]

    # Smooth curriculum
    if l2_ds is not None:
        cfg["curriculum"] = {"type": "smooth", "condition": condition}

    # EWC
    if ewc_lambda is not None and ewc_lambda > 0:
        cfg["ewc"] = {"enabled": True, "lambda_": float(ewc_lambda)}
    else:
        cfg["ewc"] = {"enabled": False}

    # MAML
    if maml_ratio is not None and maml_ratio > 0:
        cfg["maml"] = {"enabled": True, "hybrid_ratio": float(maml_ratio)}
    else:
        cfg["maml"] = {"enabled": False}

    # LR boost
    if lr_boost is not None and lr_boost > 1.0:
        cfg["training"]["optimization"]["exposure_lr"] = {
            "plasticity_boost": float(lr_boost),
        }

    # LAMOL (generative replay)
    if lamol_lambda is not None and lamol_lambda > 0:
        cfg["generative_replay"] = {
            "enabled": True,
            "lambda_replay": float(lamol_lambda),
            "trigger": "phase_change",
        }
    else:
        cfg["generative_replay"] = {"enabled": False}

    return cfg


def _write_yaml(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def _combo_tag(maml, lr, ewc=None, lamol=None):
    parts = [f"maml{maml}", f"lr{lr}"]
    if ewc is not None:
        parts.append(f"ewc{ewc}")
    if lamol is not None:
        parts.append(f"lamol{lamol}")
    return "_".join(parts)


# -- Generators ----------------------------------------------------------------

def generate_monolingual(compression: str):
    """Generate monolingual baseline configs for each language."""
    n = 0
    mono_dir = OUT_DIR / "monolingual"
    for lang in LANGS:
        ds = babybabel_mono_dataset(lang)
        tok_repo = babybabel_mono_tokenizer(lang)

        run_name = build_model_run_name(
            dataset=DATASET_SCALE, condition="monolingual", l1=lang, l2=None,
        )
        cfg = _make_cfg(
            run_name, ds, None, "monolingual", tok_repo,
            "shared", lang, [0], compression=compression,
        )
        _write_yaml(cfg, mono_dir / f"train_{lang}.yaml")
        n += 1
    return n


def generate_base(pairs, compression: str):
    """Generate base configs for all conditions (shared + per_language)."""
    n = 0
    for l1, l2 in pairs:
        l1_ds = babybabel_l1_dataset(l1, l2)
        l2_ds = babybabel_l2_dataset(l1, l2)
        tok_repo = TOKENIZER_LOOKUP[(l1, l2)]
        pk = f"{l1}_L1-{l2}_L2"

        for condition in BILINGUAL_CONDITIONS:
            ps = [0]  # smooth curriculum handles phase transitions internally
            for sharing in ("shared", "per_language"):
                run_name = build_model_run_name(
                    dataset=DATASET_SCALE, condition=condition,
                    l1=l1, l2=l2, vocab_sharing=sharing,
                )
                cfg = _make_cfg(run_name, l1_ds, l2_ds, condition, tok_repo,
                                sharing, pk, ps, compression=compression)
                _write_yaml(cfg, OUT_DIR / sharing / condition / f"train_{pk}.yaml")
                n += 1
    return n


def generate_combo_sweep(pairs, compression: str):
    """Generate combinatorial sweep: MAML x LR x (EWC or LAMOL)."""
    n = 0
    for l1, l2 in pairs:
        l1_ds = babybabel_l1_dataset(l1, l2)
        l2_ds = babybabel_l2_dataset(l1, l2)
        tok_repo = TOKENIZER_LOOKUP[(l1, l2)]
        pk = f"{l1}_L1-{l2}_L2"

        for condition in SWEEP_CONDITIONS:
            ps = [0]
            sweep_dir = OUT_DIR / "shared" / f"sweep_{condition}"

            # Set A: MAML x LR x EWC
            for maml, lr, ewc in itertools.product(MAML_RATIOS, LR_BOOSTS, EWC_LAMBDAS):
                tag = _combo_tag(maml, lr, ewc=ewc)
                run_name = build_model_run_name(
                    dataset=DATASET_SCALE, condition=condition,
                    l1=l1, l2=l2, vocab_sharing="shared",
                    ewc_lambda=ewc, maml_ratio=maml, lr_boost=lr,
                )
                cfg = _make_cfg(
                    run_name, l1_ds, l2_ds, condition, tok_repo, "shared", pk, ps,
                    compression=compression,
                    ewc_lambda=ewc, maml_ratio=maml, lr_boost=lr,
                )
                _write_yaml(cfg, sweep_dir / f"train_{pk}_{tag}.yaml")
                n += 1

            # Set B: MAML x LR x LAMOL
            for maml, lr, lamol in itertools.product(MAML_RATIOS, LR_BOOSTS, LAMOL_LAMBDAS):
                tag = _combo_tag(maml, lr, lamol=lamol)
                run_name = build_model_run_name(
                    dataset=DATASET_SCALE, condition=condition,
                    l1=l1, l2=l2, vocab_sharing="shared",
                    maml_ratio=maml, lr_boost=lr, lamol_lambda=lamol,
                )
                cfg = _make_cfg(
                    run_name, l1_ds, l2_ds, condition, tok_repo, "shared", pk, ps,
                    compression=compression,
                    maml_ratio=maml, lr_boost=lr, lamol_lambda=lamol,
                )
                _write_yaml(cfg, sweep_dir / f"train_{pk}_{tag}.yaml")
                n += 1

    return n


def generate_best(best_path: str, compression: str):
    """Generate configs for remaining languages using best settings from sweep."""
    with open(best_path) as f:
        best = json.load(f)

    pilot_langs = {l for pair in PILOT_PAIRS for l in pair}
    remaining_langs = [l for l in LANGS if l not in pilot_langs]
    remaining_pairs = [(l1, l2) for l1, l2 in PAIRS
                       if l1 in remaining_langs or l2 in remaining_langs]

    n = 0
    best_dir = OUT_DIR / "shared" / "best"
    for l1, l2 in remaining_pairs:
        l1_ds = babybabel_l1_dataset(l1, l2)
        l2_ds = babybabel_l2_dataset(l1, l2)
        tok_repo = TOKENIZER_LOOKUP[(l1, l2)]
        pk = f"{l1}_L1-{l2}_L2"

        for condition in SWEEP_CONDITIONS:
            ps = [0]
            run_name = build_model_run_name(
                dataset=DATASET_SCALE, condition=condition,
                l1=l1, l2=l2, vocab_sharing="shared",
                ewc_lambda=best.get("ewc_lambda"),
                maml_ratio=best.get("maml_ratio"),
                lr_boost=best.get("lr_boost"),
                lamol_lambda=best.get("lamol_lambda"),
            )
            cfg = _make_cfg(
                run_name, l1_ds, l2_ds, condition, tok_repo, "shared", pk, ps,
                compression=compression,
                ewc_lambda=best.get("ewc_lambda"),
                maml_ratio=best.get("maml_ratio"),
                lr_boost=best.get("lr_boost"),
                lamol_lambda=best.get("lamol_lambda"),
            )
            _write_yaml(cfg, best_dir / f"train_{pk}_{condition}.yaml")
            n += 1

    # Monolingual baselines for remaining languages
    mono_dir = OUT_DIR / "monolingual"
    for lang in remaining_langs:
        ds = babybabel_mono_dataset(lang)
        tok_repo = babybabel_mono_tokenizer(lang)
        run_name = build_model_run_name(
            dataset=DATASET_SCALE, condition="monolingual", l1=lang, l2=None,
        )
        cfg = _make_cfg(
            run_name, ds, None, "monolingual", tok_repo,
            "shared", lang, [0], compression=compression,
        )
        _write_yaml(cfg, mono_dir / f"train_{lang}.yaml")
        n += 1

    return n


# -- Main ----------------------------------------------------------------------

def generate(args):
    compression = args.compression

    print(f"max_steps   : {MAX_STEPS:,}")
    print(f"vocab_size  : {VOCAB_SIZE:,}")
    print(f"compression : {compression}")

    if args.best_config:
        print(f"\n=== Best-config mode (remaining languages) ===")
        n = generate_best(args.best_config, compression)
        print(f"Generated {n} best configs -> {OUT_DIR.resolve()}")
        return

    # Determine which pairs to generate
    if args.l1 and args.l2:
        target_pairs = [(args.l1, args.l2)]
        if (args.l2, args.l1) in TOKENIZER_LOOKUP:
            target_pairs.append((args.l2, args.l1))
        print(f"Single pair : {args.l1}-{args.l2}")
    elif args.pilot:
        target_pairs = []
        for l1, l2 in PILOT_PAIRS:
            if (l1, l2) in TOKENIZER_LOOKUP:
                target_pairs.append((l1, l2))
            if (l2, l1) in TOKENIZER_LOOKUP:
                target_pairs.append((l2, l1))
        print(f"Pilot pairs : {PILOT_PAIRS}")
    else:
        target_pairs = PAIRS
        print(f"All pairs   : {len(PAIRS)}")

    print(f"conditions  : {BILINGUAL_CONDITIONS}")
    print(f"sweep conds : {SWEEP_CONDITIONS}")
    print(f"combos/pair : {len(MAML_RATIOS)*len(LR_BOOSTS)*len(EWC_LAMBDAS)} (EWC) + "
          f"{len(MAML_RATIOS)*len(LR_BOOSTS)*len(LAMOL_LAMBDAS)} (LAMOL) = "
          f"{len(MAML_RATIOS)*len(LR_BOOSTS)*(len(EWC_LAMBDAS)+len(LAMOL_LAMBDAS))} per pair/condition\n")

    # Generate monolingual baselines
    mono_langs = set()
    for l1, l2 in target_pairs:
        mono_langs.add(l1)
        mono_langs.add(l2)
    # Filter LANGS to only those in target pairs
    n_mono = 0
    mono_dir = OUT_DIR / "monolingual"
    for lang in LANGS:
        if lang in mono_langs:
            ds = babybabel_mono_dataset(lang)
            tok_repo = babybabel_mono_tokenizer(lang)
            run_name = build_model_run_name(
                dataset=DATASET_SCALE, condition="monolingual", l1=lang, l2=None,
            )
            cfg = _make_cfg(
                run_name, ds, None, "monolingual", tok_repo,
                "shared", lang, [0], compression=compression,
            )
            _write_yaml(cfg, mono_dir / f"train_{lang}.yaml")
            n_mono += 1

    # Generate bilingual base configs
    n_base = generate_base(target_pairs, compression)

    # Generate combinatorial sweep
    n_sweep = generate_combo_sweep(target_pairs, compression)

    total = n_mono + n_base + n_sweep
    print(f"Generated {total} configs -> {OUT_DIR.resolve()}")
    print(f"  monolingual : {n_mono}")
    print(f"  base configs: {n_base}")
    print(f"  sweep combos: {n_sweep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate human-scale bilingual training configs"
    )
    parser.add_argument("--compression", choices=["equal", "proportional", "weighted"],
                        default="equal", help="Tokenizer compression strategy")
    parser.add_argument("--l1", type=str, default=None,
                        help="L1 language code (default: generate all pairs)")
    parser.add_argument("--l2", type=str, default=None,
                        help="L2 language code (default: generate all pairs)")
    parser.add_argument("--pilot", action="store_true",
                        help="Generate for pilot pairs")
    parser.add_argument("--best-config", type=str, default=None,
                        help="Path to best_sweep_settings.json -- generate remaining langs")
    args = parser.parse_args()

    # Default to nld/eng if only --l1 or --l2 is specified
    if args.l1 and not args.l2:
        args.l2 = "eng"
    if args.l2 and not args.l1:
        args.l1 = "nld"

    generate(args)
