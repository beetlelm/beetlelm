#!/usr/bin/env python3
"""
Generate all 26 YAML configs for Plan P1 of the Wilkes3 sweep:

    18  pretraining-strategy ablations  (6 strategies x 3 pairs x 142M @ B3)
     4  trilingual                       (T1..T4 x 1 ordering of nld/deu/eng @ 142M)
     1  multilingual                     (eng,nld,deu,zho balanced @ 142M)
     3  400M scale anchor                (B3 x 3 pairs L1->eng)
    ───
    26  total

Output tree:
    configs/p1_sweep/strategies/B3/<pair>/<strategy>.yaml
    configs/p1_sweep/trilingual/T<n>/train_<ordering>.yaml
    configs/p1_sweep/multilingual/train_eng_nld_deu_zho_balanced.yaml
    configs/p1_sweep/400m/B3/train_<pair>.yaml

All configs target 24B tokens on a 16-GPU layout (4 nodes x 4 GPUs) so they
work with script/slurm_fineweb_16gpu.sh and the 64-GPU SL2 cap (4 jobs run
concurrently).

Usage:
    python configs/generate_p1_sweep.py
"""

from __future__ import annotations

from pathlib import Path
import yaml

OUT_ROOT = Path(__file__).resolve().parent / "p1_sweep"

# ── Language pair metadata ──────────────────────────────────────────────────
# 3-letter -> (hf 2-letter, friendly_l1_token)
PAIRS = {
    "nld": "nl",
    "deu": "de",
    "zho": "zh",
}

HF_USER_BILINGUAL = "Beetle-FineWeb2-24B"
HF_USER_STRATEGY = "Beetle-Variants-24B"     # separate org so the strategy
                                              # ablations don't collide with
                                              # the already-trained base runs
HF_USER_TRILINGUAL = "Beetle-Trilingual-24B"
HF_USER_MULTILINGUAL = "Beetle-Multilingual-24B"

# ── Shared blocks ────────────────────────────────────────────────────────────
MODEL_142M = {
    "architecture": "pico",
    "params": {
        "n_layers": 14,
        "d_model": 768,
        "vocab_size": 50000,
        "attention_n_heads": 12,
        "attention_n_kv_heads": 1,
        "max_seq_len": 512,
        "batch_size": 32,                # per-gpu; 16 GPUs x 32 = 512 effective
        "position_emb_theta": 10000.0,
        "activation_hidden_dim": 3072,
        "norm_eps": 1.0e-05,
        "dropout": 0.1,
    },
}

MODEL_400M = {
    "architecture": "pico",
    "params": {
        "n_layers": 22,
        "d_model": 1024,
        "vocab_size": 50000,
        "attention_n_heads": 16,
        "attention_n_kv_heads": 4,        # GQA, easier to fit at 400M
        "max_seq_len": 512,
        "batch_size": 16,                 # per-gpu; 16 GPUs x 16 = 256 / step,
                                          # grad_accum=2 -> effective 512
        "position_emb_theta": 10000.0,
        "activation_hidden_dim": 4096,
        "norm_eps": 1.0e-05,
        "dropout": 0.1,
    },
}

# 16-GPU layout: nodes=4, devices=4, target_effective_batch=512
def training_block(max_steps: int, lr_warmup_steps: int = 625, lr: float = 5e-4):
    return {
        "max_steps": max_steps,
        "batch_size": None,               # filled per-yaml below
        "seq_len": 512,
        "target_effective_batch": 512,
        "fabric": {
            "accelerator": "cuda",
            "devices": 4,                 # per node
            "strategy": "ddp",
            "precision": "bf16-mixed",
        },
        "optimization": {
            "optimizer": "adamw",
            "lr": lr,
            "lr_warmup_steps": lr_warmup_steps,
            "lr_scheduler": "linear_with_warmup",
        },
    }

# 24B tokens = 91552 steps * 512 seq * 512 effective_batch  (matches hpc64)
MAX_STEPS_24B = 91552
PHASE_MIDPOINT = MAX_STEPS_24B // 2       # 45776 — B3 sequential L1->L2 boundary

# Dense around phase boundary + sparse log-spaced.
CHECKPOINT_STEPS_24B = [
    0, 1, 2, 4, 6, 8, 16, 64, 256, 1024, 4096, 16384,
    PHASE_MIDPOINT, PHASE_MIDPOINT + 1, PHASE_MIDPOINT + 2,
    PHASE_MIDPOINT + 4, PHASE_MIDPOINT + 8, PHASE_MIDPOINT + 16,
    PHASE_MIDPOINT + 64, PHASE_MIDPOINT + 256, PHASE_MIDPOINT + 1024,
    PHASE_MIDPOINT + 4096, PHASE_MIDPOINT + 16384,
    MAX_STEPS_24B,
]

def checkpointing_block(steps=None, every_n=None):
    blk = {
        "runs_dir": "./runs",
        "auto_resume": True,
        "save_to_hf": True,
        "export_final_only": False,
        "keep_local_checkpoints": 1,
        "checkpoint_steps": steps if steps is not None else CHECKPOINT_STEPS_24B,
    }
    if every_n is not None:
        blk["checkpoint_every_n_steps"] = every_n
    return blk

POLCA_BLOCK = {
    "enabled": True,
    "every_n_steps": 500,
    "k_eigenvectors": 32,
    "basis_rank": 32,
    "holdout_batch_size": 8,
    "holdout_seq_len": 512,
    "artifact_dir": "polca",
    "push_to_hf": True,
    "precision": "fp32",
}

# ── Strategy variant blocks ─────────────────────────────────────────────────
# Each entry: (top-level yaml keys it injects, dict of those keys)
STRATEGIES = {
    "ewc": {
        "ewc": {
            "enabled": True,
            "lambda_": 20.0,
            "n_fisher_samples": 10,
            "mu": 0.0,
            "trigger": "phase_change",
        },
    },
    "maml": {
        "maml": {
            "enabled": True,
            "hybrid_ratio": 0.5,
            "n_way": 5,
            "k_shot": 1,
            "inner_steps": 3,
            "inner_lr": 0.1,
            "head": {"num_layers": 2, "hidden_dim": 256, "init_method": "xavier"},
        },
    },
    "lamol": {
        "generative_replay": {
            "enabled": True,
            "lambda_replay": 0.3,
            "replay_ratio": 0.3,
            "temperature": 0.9,
            "top_p": 0.92,
            "max_gen_len": 128,
            "prompt_style": "lang_tag",
            "buffer_size": 1000,
            "trigger": "phase_change",
        },
    },
    "sim-replay": {
        "similarity_replay": {
            "enabled": True,
            "lambda_replay": 0.3,
            "similarity_metric": "cosine",
            "centroid_layer": "last_hidden",
            "n_centroid_samples": 100,
            "replay_buffer_size": 2000,
            "min_similarity_threshold": 0.0,
            "trigger": "phase_change",
        },
    },
    "lang-id": {
        "lang_id": {
            "enabled": True,
            "layer": 6,
            "hidden_dim": 256,
            "weight": 0.1,
            "num_languages": 2,
            "per_token": False,
        },
    },
    "pre-pretrain": {
        "pre_pretraining": {
            "enabled": True,
            "formal_lang": "shuffle_dyck",
            "steps": 500,
            "num_symbols": 20,
            "max_depth": 8,
            "p_open": 0.5,
            "vocab_offset": 0,
            "reset_optimizer": True,
            "save_stage1_checkpoint": True,
        },
    },
}

# Off-blocks: every strategy yaml must explicitly disable the ones it isn't
OFF_BLOCKS = {
    "ewc":               {"ewc": {"enabled": False}},
    "maml":              {"maml": {"enabled": False}},
    "generative_replay": {"generative_replay": {"enabled": False}},
    "similarity_replay": {"similarity_replay": {"enabled": False}},
    "lang_id":           {"lang_id": {"enabled": False}},
    "pre_pretraining":   {"pre_pretraining": {"enabled": False}},
}

# Map strategy name -> top-level keys it sets (so we know which OFFs to skip)
STRATEGY_KEYS = {
    "ewc":           ["ewc"],
    "maml":          ["maml"],
    "lamol":         ["generative_replay"],
    "sim-replay":    ["similarity_replay"],
    "lang-id":       ["lang_id"],
    "pre-pretrain":  ["pre_pretraining"],
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def bilingual_data_block(l1_iso3: str):
    """L1 = l1_iso3 (nld|deu|zho), L2 = eng. Matches hpc64 dataset naming."""
    code2 = PAIRS[l1_iso3]
    return {
        "data_mix": "B3",
        "lang_sources": [f"Beetle-Data/{code2}-28B"],
        "l2_streams":   [f"Beetle-Data/en-for-{code2}-28B"],
        "pretokenized": True,
        "pretokenized_cache_dir":
            f"./pretok_cache/p1-24b/{l1_iso3}_L1-eng_L2",
    }


def bilingual_tokenizer_block(l1_iso3: str):
    code2 = PAIRS[l1_iso3]
    return {
        "pretrained_tokenizer_path": f"Beetle-Data/tokenizer-{code2}-en",
        "dataset_source": "fineweb",
        "vocab_size": 50000,
        "vocab_sharing": "shared",
        "type": "bpe",
        "target_tokens": 1_000_000_000,
        "output_dir": f"./tokenizers/fineweb_{l1_iso3}_L1-eng_L2_shared",
        "overwrite": False,
        "push_to_hub": True,
    }


def base_bilingual_142m(l1_iso3: str, hf_user: str, run_name: str) -> dict:
    """Skeleton with strategy blocks all OFF; caller flips the one it wants."""
    train = training_block(MAX_STEPS_24B, lr_warmup_steps=625)
    train["batch_size"] = MODEL_142M["params"]["batch_size"]

    cfg = {
        "run_name": run_name,
        "hf_user": hf_user,
        "model": MODEL_142M,
        "data": bilingual_data_block(l1_iso3),
        "tokenizer": bilingual_tokenizer_block(l1_iso3),
        "training": train,
        "checkpointing": checkpointing_block(every_n=4577),
        "curriculum": {"type": "smooth", "condition": "B3", "mode": "tokens"},
        "ewc":                {"enabled": False},
        "maml":               {"enabled": False},
        "generative_replay":  {"enabled": False},
        "similarity_replay":  {"enabled": False},
        "lang_id":            {"enabled": False},
        "pre_pretraining":    {"enabled": False},
        "polca": POLCA_BLOCK,
    }
    return cfg


# ── Tier 1: Strategies (18 jobs) ────────────────────────────────────────────
def gen_strategies():
    n = 0
    for strategy, payload in STRATEGIES.items():
        for l1 in PAIRS:
            run_name = (
                f"beetle-bilingual-b3-fineweb-24b-{l1}-eng-{strategy}"
            )
            cfg = base_bilingual_142m(l1, HF_USER_STRATEGY, run_name)

            # Flip the chosen strategy on, overriding the default-off entry.
            for key, val in payload.items():
                cfg[key] = val

            # lang-id needs lang_keys in data block
            if strategy == "lang-id":
                cfg["data"]["lang_keys"] = [l1, "eng"]

            out = OUT_ROOT / "strategies" / "B3" / f"{l1}_L1-eng_L2" / f"{strategy}.yaml"
            write_yaml(out, cfg)
            n += 1
    return n


# ── Tier 2: Trilingual T1..T4 x 1 ordering (4 jobs) ─────────────────────────
# We pick ordering (nld, deu, eng): L1=nld, L2=deu, L3=eng — the canonical
# Germanic-then-English curriculum. T-condition controls phase split.
def gen_trilingual():
    triple = ("nld", "deu", "eng")
    ordering_tag = "nld_L1-deu_L2-eng_L3"
    code = {"nld": "nl", "deu": "de", "eng": "en"}

    # Per-language 24B-tier source datasets (8B each x 3 = 24B effective)
    # Naming follows the Beetle-Data convention used for bilingual datasets.
    lang_sources = {
        lang: f"Beetle-Data/{code[lang]}-trilingual-8B-pretok"
        for lang in triple
    }

    # For trilingual at 24B tokens, total steps roughly the same (24B / 512 / 512)
    max_steps = MAX_STEPS_24B
    n = 0
    for t_idx in range(1, 5):
        t_cond = f"T{t_idx}"
        run_name = f"beetle-trilingual-{t_cond.lower()}-fineweb-24b-{ordering_tag}"

        train = training_block(max_steps, lr_warmup_steps=625)
        train["batch_size"] = MODEL_142M["params"]["batch_size"]

        cfg = {
            "run_name": run_name,
            "hf_user": HF_USER_TRILINGUAL,
            "model": MODEL_142M,
            "data": {
                "data_mix": t_cond,
                "lang_sources": lang_sources,
                "pretokenized": True,
                "pretokenized_cache_dir":
                    f"./pretok_cache/p1-24b-tri/{ordering_tag}_{t_cond}",
            },
            "tokenizer": {
                "pretrained_tokenizer_path":
                    "Beetle-Data/tokenizer-nl-de-en-trilingual",
                "dataset_source": "fineweb",
                "vocab_size": 50000,
                "vocab_sharing": "shared",
                "type": "bpe",
                "target_tokens": 1_000_000_000,
                "output_dir": f"./tokenizers/fineweb_{ordering_tag}_shared",
                "overwrite": False,
                "push_to_hub": True,
            },
            "training": train,
            "checkpointing": checkpointing_block(every_n=4577),
            "curriculum": {
                "type": "smooth",
                "condition": t_cond,
                "lang_keys": list(triple),
                "mode": "tokens",
            },
            "ewc":               {"enabled": False},
            "maml":              {"enabled": False},
            "generative_replay": {"enabled": False},
            "polca": POLCA_BLOCK,
        }

        out = OUT_ROOT / "trilingual" / t_cond / f"train_{ordering_tag}.yaml"
        write_yaml(out, cfg)
        n += 1
    return n


# ── Tier 3: Multilingual balanced (1 job) ───────────────────────────────────
def gen_multilingual():
    triple = ("eng", "nld", "deu", "zho")
    run_name = "beetle-multilingual-fineweb-24b-eng-nld-deu-zho-balanced"
    train = training_block(MAX_STEPS_24B, lr_warmup_steps=625)
    train["batch_size"] = MODEL_142M["params"]["batch_size"]

    cfg = {
        "run_name": run_name,
        "hf_user": HF_USER_MULTILINGUAL,
        "model": MODEL_142M,
        "data": {
            "data_mix": "multilingual_balanced",
            "lang_sources": [
                "Beetle-Data/en-for-multi-24B-pretok",
                "Beetle-Data/nl-multi-24B-pretok",
                "Beetle-Data/de-multi-24B-pretok",
                "Beetle-Data/zh-multi-24B-pretok",
            ],
            "pretokenized": True,
            "pretokenized_cache_dir":
                "./pretok_cache/p1-24b-multi/eng_nld_deu_zho",
        },
        "tokenizer": {
            "pretrained_tokenizer_path":
                "Beetle-Data/tokenizer-en-nl-de-zh-multi",
            "dataset_source": "fineweb",
            "vocab_size": 50000,
            "vocab_sharing": "shared",
            "type": "unigram",
            "target_tokens": 5_000_000_000,
            "output_dir": "./tokenizers/fineweb_24b_multi_eng_nld_deu_zho",
            "overwrite": False,
            "push_to_hub": True,
        },
        "training": train,
        "checkpointing": checkpointing_block(
            steps=[0, 1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384,
                   MAX_STEPS_24B // 2, MAX_STEPS_24B],
            every_n=4577,
        ),
        "curriculum": {"type": "smooth", "condition": "B1", "mode": "tokens"},
        "ewc":               {"enabled": False},
        "maml":              {"enabled": False},
        "generative_replay": {"enabled": False},
        "polca": POLCA_BLOCK,
    }
    out = OUT_ROOT / "multilingual" / f"train_eng_nld_deu_zho_balanced.yaml"
    write_yaml(out, cfg)
    return 1


# ── Tier 4: 400M B3 x 3 pairs (3 jobs) ──────────────────────────────────────
def gen_400m():
    n = 0
    for l1 in PAIRS:
        run_name = f"beetle-bilingual-b3-fineweb-24b-400m-{l1}-eng"
        train = training_block(MAX_STEPS_24B, lr_warmup_steps=625, lr=4.5e-4)
        train["batch_size"] = MODEL_400M["params"]["batch_size"]

        cfg = {
            "run_name": run_name,
            "hf_user": HF_USER_BILINGUAL,
            "model": MODEL_400M,
            "data": bilingual_data_block(l1),
            "tokenizer": bilingual_tokenizer_block(l1),
            "training": train,
            "checkpointing": checkpointing_block(every_n=4577),
            "curriculum": {"type": "smooth", "condition": "B3", "mode": "tokens"},
            "ewc":               {"enabled": False},
            "maml":              {"enabled": False},
            "generative_replay": {"enabled": False},
            "polca": POLCA_BLOCK,
        }

        out = OUT_ROOT / "400m" / "B3" / f"train_{l1}_L1-eng_L2.yaml"
        write_yaml(out, cfg)
        n += 1
    return n


def main():
    counts = {
        "strategies":   gen_strategies(),
        "trilingual":   gen_trilingual(),
        "multilingual": gen_multilingual(),
        "400m":         gen_400m(),
    }
    total = sum(counts.values())
    print(f"Generated under {OUT_ROOT}:")
    for tier, c in counts.items():
        print(f"  {tier:14s} {c:3d}")
    print(f"  {'TOTAL':14s} {total:3d}")
    if total != 26:
        raise SystemExit(f"Expected 26 yamls, got {total}")


if __name__ == "__main__":
    main()
