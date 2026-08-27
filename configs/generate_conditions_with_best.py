#!/usr/bin/env python3
"""
configs/generate_conditions_with_best.py

Generate B1-B6 condition configs using best continual learning settings
from the Experiment 1 sweep.

For each condition, the config uses:
  - Condition-specific curriculum from condition_presets.py
  - Condition-specific EWC/MAML/LR recommendations as defaults
  - Best sweep settings from Experiment 1 to override where applicable

Usage:
  python configs/generate_conditions_with_best.py --best-config results/best_sweep_settings.json
  python configs/generate_conditions_with_best.py --best-config results/best_sweep_settings.json --l1 nld --l2 eng
  python configs/generate_conditions_with_best.py --ablations
  python configs/generate_conditions_with_best.py --dry-run

Output: configs/conditions/bilingual/{B1,..,B6}/train_{pk}.yaml
"""

import argparse
import copy
import json
import itertools
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config._human_scale_defaults import (
    HF_ORG, DATASET_SCALE, MODEL_BASE, TRAINING_DEFAULTS,
    CHECKPOINTING_DEFAULTS, LANGS, MAX_STEPS, VOCAB_SIZE,
    TOKENIZER_TYPE, TOKENIZER_TOKENS,
    BILINGUAL_CONDITIONS,
    babybabel_l1_dataset, babybabel_l2_dataset,
    babybabel_bilingual_tokenizer,
    TOKENS_CLASSROOM_L1, TOKENS_CLASSROOM_L2,
)
from src.config._repo_naming import build_model_run_name
from src.bilingual.data.condition_presets import (
    CONDITION_DEFAULTS,
    get_condition_defaults,
)
from configs.ckpt_schedule import phase_log2_schedule

OUT_DIR = Path("configs/conditions/bilingual")

CONDITIONS = ["B1", "B2", "B3", "B4", "B5", "B6"]

ABLATION_VARIANTS = [
    "B1a", "B1b",
    "B2_later", "B2_smooth", "B2_burst", "B2_ewc", "B2_lr",
    "B3_full", "B3_25_75",
    "B4a", "B4b",
    "B5_ewc",
]

# Phase starts per condition (approximate, used for checkpoint scheduling)
CONDITION_PHASE_STARTS = {
    "B1": [0],
    "B2": [0, int(MAX_STEPS * 0.5)],
    "B3": [0, int(MAX_STEPS * 0.5)],
    "B4": [0, int(MAX_STEPS * 0.5)],
    "B5": [0, int(MAX_STEPS * 0.8)],
    "B6": [0, int(MAX_STEPS * 0.3), int(MAX_STEPS * 0.5)],
}

# Reuse pair registry from human-scale generator
from configs.generate_human_scale_bilingual import (
    TOKENIZER_LOOKUP, PAIRS,
    build_pair_registry,
)


def _make_condition_cfg(
    l1: str,
    l2: str,
    condition: str,
    tok_repo: str,
    *,
    best_settings: dict | None = None,
) -> dict:
    pk = f"{l1}_L1-{l2}_L2"

    # B4 uses asymmetric token budgets (80M L1 + 20M L2)
    if condition.startswith("B4"):
        l1_target = f"{TOKENS_CLASSROOM_L1 // 1_000_000}M"
        l2_target = f"{TOKENS_CLASSROOM_L2 // 1_000_000}M"
    else:
        l1_target = "50M"
        l2_target = "50M"
    l1_ds = babybabel_l1_dataset(l1, l2, target=l1_target)
    l2_ds = babybabel_l2_dataset(l1, l2, target=l2_target)

    # Resolve phase_starts for ablation variants
    base_cond = condition.split("_")[0] if "_" in condition else condition.rstrip("ab")
    if base_cond not in CONDITION_PHASE_STARTS:
        base_cond = condition[:2]
    phase_starts = CONDITION_PHASE_STARTS.get(base_cond, [0])

    ckpt_steps = phase_log2_schedule(phase_starts, max_steps=MAX_STEPS,
                                      extra_offsets=[2, 4, 6, 8])

    training = copy.deepcopy(TRAINING_DEFAULTS)
    training["max_steps"] = MAX_STEPS

    # Start with condition-specific defaults
    cond_defaults = get_condition_defaults(condition)

    # Placeholder run_name; finalized after all overrides so feature suffix matches config
    cfg = {
        "run_name": "",
        "hf_user": HF_ORG,
        "model": {
            "architecture": MODEL_BASE["architecture"],
            "params": dict(MODEL_BASE["params"]),
        },
        "data": {
            "data_mix": condition,
            "lang_sources": [l1_ds],
            "l2_streams": [l2_ds],
            "pretokenized": True,
            "pretokenized_cache_dir": f"./pretok_cache/{pk}_{condition}",
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

    # Build final run_name from the resolved config so the HF repo name
    # accurately reflects the active EWC/MAML/LR/LAMOL values.
    ewc_cfg = cfg.get("ewc", {})
    maml_cfg = cfg.get("maml", {})
    exposure_lr = cfg["training"]["optimization"].get("exposure_lr", {})
    gen_replay = cfg.get("generative_replay", {})

    run_name_kwargs = dict(
        dataset=DATASET_SCALE,
        condition=condition,
        l1=l1,
        l2=l2,
        vocab_sharing="shared",
        ewc_lambda=ewc_cfg.get("lambda_") if ewc_cfg.get("enabled") else None,
        maml_ratio=maml_cfg.get("hybrid_ratio") if maml_cfg.get("enabled") else None,
        lr_boost=exposure_lr.get("plasticity_boost"),
        lamol_lambda=(
            gen_replay.get("lambda_replay") if gen_replay.get("enabled") else None
        ),
    )
    # B4 asymmetric proportion (L2 = 20% of combined 100M budget by default)
    if condition.startswith("B4"):
        run_name_kwargs["b4_proportion_pct"] = int(
            TOKENS_CLASSROOM_L2 * 100 // (TOKENS_CLASSROOM_L1 + TOKENS_CLASSROOM_L2)
        )

    cfg["run_name"] = build_model_run_name(**run_name_kwargs)
    cfg["data"]["pretokenized_cache_dir"] = f"./pretok_cache/{cfg['run_name']}"

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

    # Determine target pairs
    if args.l1 and args.l2:
        target_pairs = [(args.l1, args.l2)]
        if (args.l2, args.l1) in TOKENIZER_LOOKUP:
            target_pairs.append((args.l2, args.l1))
        print(f"Single pair: {args.l1}-{args.l2}")
    else:
        target_pairs = PAIRS
        print(f"All pairs: {len(PAIRS)}")

    # Include ablation variants if requested
    conditions_to_generate = list(CONDITIONS)
    if args.ablations:
        conditions_to_generate.extend(ABLATION_VARIANTS)

    print(f"Conditions : {conditions_to_generate}")
    print(f"max_steps  : {MAX_STEPS:,}")

    n = 0
    for l1, l2 in target_pairs:
        tok_repo = TOKENIZER_LOOKUP[(l1, l2)]
        pk = f"{l1}_L1-{l2}_L2"

        for condition in conditions_to_generate:
            cfg = _make_condition_cfg(
                l1, l2, condition, tok_repo,
                best_settings=best_settings,
            )

            if args.dry_run:
                print(f"  [DRY RUN] {condition}/train_{pk}.yaml")
            else:
                _write_yaml(cfg, OUT_DIR / condition / f"train_{pk}.yaml")

            n += 1

    print(f"\nGenerated {n} configs -> {OUT_DIR.resolve()}")
    for cond in conditions_to_generate:
        if cond in CONDITION_DEFAULTS:
            desc = CONDITION_DEFAULTS[cond]["description"]
            print(f"  {cond}: {desc}")
        else:
            print(f"  {cond}: (ablation variant)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate B1-B6 condition configs with best sweep settings"
    )
    parser.add_argument("--best-config", type=str, default=None,
                        help="Path to best_sweep_settings.json from Experiment 1")
    parser.add_argument("--l1", type=str, default=None,
                        help="L1 language code (default: all pairs)")
    parser.add_argument("--l2", type=str, default=None,
                        help="L2 language code (default: all pairs)")
    parser.add_argument("--ablations", action="store_true",
                        help="Also generate configs for ablation variants")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated without writing files")
    args = parser.parse_args()

    if args.l1 and not args.l2:
        args.l2 = "eng"
    if args.l2 and not args.l1:
        args.l1 = "nld"

    generate(args)
