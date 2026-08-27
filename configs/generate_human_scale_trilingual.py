#!/usr/bin/env python3
"""
configs/generate_human_scale_trilingual.py

Human-scale trilingual configs with smooth curricula and vocabulary-sharing investigation.

Datasets  : Beetle-Data/BabyBabel-{lang}-33M  (33M tokens per language = ~100M total)
Vocab size: 50,000 BPE tokens
Steps     : 10,000 (fixed)
Schedule  : Phase-aware log₂ checkpoints (restarts at each language introduction)
Curriculum: Smooth phase transitions by default (--legacy for discrete phases)

Two config variants per (permutation × mix):
  shared/      — one shared BPE tokenizer over all three languages
  per_language/ — separate BPE tokenizer per language (vocab-sharing ablation)

Languages: eng, nld, zho (all 6 ordered permutations)
Mixes    : balanced, late, sequential, heritage, part_time, simultaneous

Output: configs/human_scale_trilingual/{shared,per_language}/{mix}/train_{pk}.yaml
"""

import itertools
import sys
from pathlib import Path

import yaml

# ── Shared constants ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config._human_scale_defaults import (
    HF_ORG, MODEL_BASE, TRAINING_DEFAULTS,
    CHECKPOINTING_DEFAULTS, TRILINGUAL_CONDITIONS, MAX_STEPS, VOCAB_SIZE,
    TOKENIZER_TYPE, TOKENIZER_TOKENS,
    babybabel_tri_dataset, babybabel_trilingual_tokenizer,
)
from configs.ckpt_schedule import phase_log2_schedule

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("configs/human_scale_trilingual")

TRI_LANGS = ["eng", "nld", "zho"]
PERMS     = list(itertools.permutations(TRI_LANGS, 3))

# Datasets and tokenizer are now computed dynamically per triple


def _make_cfg(
    run_name:      str,
    l1:            str,
    l2:            str,
    l3:            str,
    condition:     str,
    vocab_sharing: str,
    phase_starts:  list,
) -> dict:
    ckpt_steps = phase_log2_schedule(phase_starts, max_steps=MAX_STEPS)
    training   = {**TRAINING_DEFAULTS, "max_steps": MAX_STEPS}
    ckpt       = {**CHECKPOINTING_DEFAULTS, "checkpoint_steps": ckpt_steps}
    pk         = f"{l1}_L1-{l2}_L2-{l3}_L3"

    cfg = {
        "run_name": run_name,
        "hf_user":  HF_ORG,
        "model": {
            "architecture": MODEL_BASE["architecture"],
            "params": dict(MODEL_BASE["params"]),
        },
        "data": {
            "data_mix":               condition,
            "lang_sources":           [babybabel_tri_dataset(l1, l1, l2, l3)],
            "l2_streams":             [babybabel_tri_dataset(l2, l1, l2, l3),
                                       babybabel_tri_dataset(l3, l1, l2, l3)],
            "pretokenized":           True,
            "pretokenized_cache_dir": f"./pretok_cache/{run_name}",
        },
        "tokenizer": {
            "pretrained_tokenizer_path": babybabel_trilingual_tokenizer(l1, l2, l3),
            "dataset_source": "babybabel",
            "vocab_size":     VOCAB_SIZE,
            "vocab_sharing":  vocab_sharing,
            "type":           TOKENIZER_TYPE,
            "target_tokens":  TOKENIZER_TOKENS,
            "output_dir":     f"./tokenizers/{pk}_{vocab_sharing}",
            "overwrite":      False,
            "push_to_hub":    True,
        },
        "training":      training,
        "checkpointing": ckpt,
        "curriculum": {
            "type": "smooth",
            "condition": condition,
            "lang_keys": [l1, l2, l3],
        },
    }

    return cfg


# ── Generator ─────────────────────────────────────────────────────────────────

def generate():
    print(f"max_steps    : {MAX_STEPS:,}")
    print(f"vocab_size   : {VOCAB_SIZE:,}")
    print(f"permutations : {len(PERMS)}")
    print(f"conditions   : {TRILINGUAL_CONDITIONS}")
    print(f"variants     : shared / per_language\n")

    n_shared = n_perlang = 0

    for l1, l2, l3 in PERMS:
        pk = f"{l1}_L1-{l2}_L2-{l3}_L3"

        for condition in TRILINGUAL_CONDITIONS:
            ps = [0]  # smooth handles phases

            for vocab_sharing, variant_dir in [("shared", "shared"),
                                                ("per_language", "per_language")]:
                run_name = f"beetlelm_{pk}_{condition}_{variant_dir}"
                cfg      = _make_cfg(run_name, l1, l2, l3, condition, vocab_sharing, ps)

                out_path = OUT_DIR / variant_dir / condition
                out_path.mkdir(parents=True, exist_ok=True)
                with open(out_path / f"train_{pk}.yaml", "w") as f:
                    yaml.dump(cfg, f, sort_keys=False)

                if vocab_sharing == "shared":
                    n_shared += 1
                else:
                    n_perlang += 1

    total = n_shared + n_perlang
    print(f"Generated {total} configs -> {OUT_DIR.resolve()}")
    print(f"  shared      : {n_shared}")
    print(f"  per_language: {n_perlang}")
    print(f"  ({len(PERMS)} perms x {len(TRILINGUAL_CONDITIONS)} conditions x 2 variants)")


if __name__ == "__main__":
    generate()
