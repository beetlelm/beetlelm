#!/usr/bin/env python3
"""
configs/generate_conditions_trilingual.py

Generate T1-T4 trilingual condition configs using best continual learning
settings from the Experiment 1 sweep.

For each condition, the config uses:
  - Condition-specific curriculum from condition_presets.py
  - Condition-specific EWC/MAML/LR recommendations as defaults
  - Best sweep settings from Experiment 1 to override where applicable

Usage:
  python configs/generate_conditions_trilingual.py
  python configs/generate_conditions_trilingual.py --best-config results/best_sweep_settings.json
  python configs/generate_conditions_trilingual.py --dry-run

Output: configs/conditions/trilingual/{T1,T2,T3,T4}/train_{pk}.yaml
"""

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config._human_scale_defaults import (
    HF_ORG, MODEL_BASE, TRAINING_DEFAULTS,
    CHECKPOINTING_DEFAULTS, MAX_STEPS, VOCAB_SIZE,
    TOKENIZER_TYPE, TOKENIZER_TOKENS,
    TRILINGUAL_CONDITIONS,
    babybabel_tri_dataset, babybabel_trilingual_tokenizer,
)
from src.bilingual.data.condition_presets import (
    CONDITION_DEFAULTS,
    get_condition_defaults,
)
from configs.ckpt_schedule import phase_log2_schedule

OUT_DIR = Path("configs/conditions/trilingual")

CONDITIONS = TRILINGUAL_CONDITIONS  # T1, T2, T3, T4

# Trilingual triples to generate configs for
# Default: MECO-based triples (eng always included)
TRI_LANGS = ["eng", "nld", "zho", "deu"]
DEFAULT_TRIPLES = [
    ("eng", "nld", "zho"),
    ("eng", "nld", "deu"),
    ("eng", "zho", "deu"),
]


def _make_condition_cfg(
    l1: str,
    l2: str,
    l3: str,
    condition: str,
    *,
    best_settings: dict | None = None,
) -> dict:
    pk = f"{l1}_L1-{l2}_L2-{l3}_L3"
    run_name = f"beetlelm_{pk}_{condition}"

    l1_ds = babybabel_tri_dataset(l1, l1, l2, l3)
    l2_ds = babybabel_tri_dataset(l2, l1, l2, l3)
    l3_ds = babybabel_tri_dataset(l3, l1, l2, l3)
    tok_repo = babybabel_trilingual_tokenizer(l1, l2, l3)

    # Smooth curriculum handles phases internally
    ckpt_steps = phase_log2_schedule([0], max_steps=MAX_STEPS,
                                      extra_offsets=[2, 4, 6, 8])

    training = copy.deepcopy(TRAINING_DEFAULTS)
    training["max_steps"] = MAX_STEPS

    cond_defaults = get_condition_defaults(condition)

    cfg = {
        "run_name": run_name,
        "hf_user": HF_ORG,
        "model": {
            "architecture": MODEL_BASE["architecture"],
            "params": dict(MODEL_BASE["params"]),
        },
        "data": {
            "data_mix": condition,
            "lang_sources": {l1: l1_ds, l2: l2_ds, l3: l3_ds},
            "pretokenized": True,
            "pretokenized_cache_dir": f"./pretok_cache/{run_name}",
        },
        "tokenizer": {
            "pretrained_tokenizer_path": tok_repo,
            "dataset_source": "babybabel",
            "vocab_size": VOCAB_SIZE,
            "vocab_sharing": "shared",
            "type": TOKENIZER_TYPE,
            "target_tokens": TOKENIZER_TOKENS,
            "output_dir": f"./tokenizers/{pk}_shared",
            "overwrite": False,
            "push_to_hub": True,
        },
        "training": training,
        "checkpointing": {**CHECKPOINTING_DEFAULTS, "checkpoint_steps": ckpt_steps},
        "curriculum": {
            "type": "smooth",
            "condition": condition,
            "lang_keys": [l1, l2, l3],
            "mode": "steps",
        },
    }

    # Apply condition-specific EWC defaults
    cfg["ewc"] = dict(cond_defaults["ewc"])

    # Apply condition-specific MAML defaults
    cfg["maml"] = dict(cond_defaults["maml"])

    # Apply condition-specific LR/optimization defaults
    if "exposure_lr" in cond_defaults.get("optimization", {}):
        cfg["training"]["optimization"]["exposure_lr"] = dict(
            cond_defaults["optimization"]["exposure_lr"]
        )

    # Override with best sweep settings where provided
    if best_settings:
        ewc_lambda = best_settings.get("ewc_lambda")
        if ewc_lambda is not None and cfg["ewc"].get("enabled"):
            cfg["ewc"]["lambda_"] = float(ewc_lambda)

        maml_ratio = best_settings.get("maml_ratio")
        if maml_ratio is not None and cfg["maml"].get("enabled"):
            cfg["maml"]["hybrid_ratio"] = float(maml_ratio)

        lr_boost = best_settings.get("lr_boost")
        if lr_boost is not None and "exposure_lr" in cfg["training"]["optimization"]:
            cfg["training"]["optimization"]["exposure_lr"]["plasticity_boost"] = float(lr_boost)

        lamol_lambda = best_settings.get("lamol_lambda")
        if lamol_lambda is not None and lamol_lambda > 0:
            cfg["generative_replay"] = {
                "enabled": True,
                "lambda_replay": float(lamol_lambda),
                "trigger": "phase_change",
            }
        else:
            cfg["generative_replay"] = {"enabled": False}
    else:
        cfg["generative_replay"] = {"enabled": False}

    return cfg


def _write_yaml(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def generate(args):
    best_settings = None
    if args.best_config:
        with open(args.best_config) as f:
            best_settings = json.load(f)
        print(f"Loaded best settings: {best_settings}")

    triples = DEFAULT_TRIPLES
    print(f"Conditions : {CONDITIONS}")
    print(f"Triples    : {triples}")
    print(f"max_steps  : {MAX_STEPS:,}")

    n = 0
    for l1, l2, l3 in triples:
        # Generate all 6 permutations of each triple
        for perm in itertools.permutations([l1, l2, l3]):
            pl1, pl2, pl3 = perm
            pk = f"{pl1}_L1-{pl2}_L2-{pl3}_L3"

            for condition in CONDITIONS:
                cfg = _make_condition_cfg(
                    pl1, pl2, pl3, condition,
                    best_settings=best_settings,
                )

                if args.dry_run:
                    print(f"  [DRY RUN] {condition}/train_{pk}.yaml")
                else:
                    _write_yaml(cfg, OUT_DIR / condition / f"train_{pk}.yaml")

                n += 1

    print(f"\nGenerated {n} configs -> {OUT_DIR.resolve()}")
    for cond in CONDITIONS:
        if cond in CONDITION_DEFAULTS:
            desc = CONDITION_DEFAULTS[cond]["description"]
            print(f"  {cond}: {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate T1-T4 trilingual condition configs"
    )
    parser.add_argument("--best-config", type=str, default=None,
                        help="Path to best_sweep_settings.json from Experiment 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated without writing files")
    args = parser.parse_args()

    generate(args)
