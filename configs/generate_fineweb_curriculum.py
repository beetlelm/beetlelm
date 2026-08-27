#!/usr/bin/env python3
"""
generate_fineweb_curriculum.py — Generate BeetleLM training configs for
BeetleStream v2 curriculum mode.

Generates one YAML per condition × language pair at:
  configs/fineweb_curriculum/{condition}/{lang}/config.yaml

Each config specifies:
  - stream_mode: "curriculum"
  - indexed_data_dir or HF repo for indexed shards
  - condition-specific quality table (auto-linked)
  - topic alignment settings

Usage:
    # Generate all conditions × all core languages
    python configs/generate_fineweb_curriculum.py

    # Specific condition and languages
    python configs/generate_fineweb_curriculum.py --conditions B1 B2 --langs fr de zh

    # Custom output directory
    python configs/generate_fineweb_curriculum.py --output-dir configs/custom/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config._fineweb_defaults import (
    EFFECTIVE_BATCH, MAX_STEPS, SEQ_LEN, fineweb_per_gpu_batch,
)

# Core bilingual language pairs (L1 → EN)
CORE_PAIRS = [
    "pl", "nl", "es", "el", "ja", "fr", "zh", "de", "it", "eu",
    "tr", "id", "tl", "fa", "hi", "ta", "sv", "ru", "ca", "ar",
]

EXTENSION_PAIRS = ["ur", "bn", "cs", "gu", "th", "vi", "ko", "da"]

BILINGUAL_CONDITIONS = ["B1", "B2", "B3", "B4", "B5"]
TRILINGUAL_CONDITIONS = ["T1", "T2", "T3", "T4"]

# Best EWC/MAML/LR settings from human-scale sweep (update after Phase 1)
BEST_EWC_LAMBDA = 20.0
BEST_MAML_RATIO = 0.3
BEST_LR_BOOST = 2.0


def generate_bilingual_config(
    condition: str,
    lang: str,
    hf_user: str = "Beetle-Data",
    max_steps: int = MAX_STEPS,
    seq_len: int = SEQ_LEN,
    batch_size: Optional[int] = None,
    devices: int = 8,
    world_size: Optional[int] = None,
) -> dict:
    """Generate a single bilingual curriculum training config.

    If batch_size is None it is derived from world_size (or devices, if
    world_size is unset) so that effective batch = EFFECTIVE_BATCH.
    """
    ws = world_size if world_size is not None else devices
    if batch_size is None:
        batch_size = fineweb_per_gpu_batch(ws)
    config = {
        "hf_user": hf_user,
        "run_name": f"beetlelm_{lang}-en_{condition}_curriculum",

        "model": {
            "architecture": "pico",
            "params": {
                "n_layers": 12,
                "d_model": 768,
                "vocab_size": 50000,
                "attention_n_heads": 12,
                "attention_n_kv_heads": 1,
                "max_seq_len": seq_len,
                "batch_size": batch_size,
                "position_emb_theta": 10000.0,
                "activation_hidden_dim": 3072,
                "norm_eps": 1e-5,
                "dropout": 0.1,
            },
        },

        "data": {
            "stream_mode": "curriculum",
            "indexed_hf_repos": {
                "l1": f"{hf_user}/{lang}-curriculum-24B",
                "l2": f"{hf_user}/en-curriculum-24B",
            },
            "topic_aligned_batching": True,
            "n_topics": 200,
            "splits": {"train": 0.98, "val": 0.02},
        },

        "curriculum": {
            "type": "smooth",
            "condition": condition,
        },

        "tokenizer": {
            "repo": f"{hf_user}/tokenizer-{lang}-en",
            "type": "bpe",
            "vocab_size": 50000,
        },

        "training": {
            "max_steps": max_steps,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "target_effective_batch": EFFECTIVE_BATCH,
            "fabric": {
                "accelerator": "cuda",
                "strategy": "auto",
                "precision": "bf16-mixed",
                "devices": devices,
            },
            "optimization": {
                "optimizer": "adamw",
                "lr": 5e-4,
                "weight_decay": 0.01,
                "lr_scheduler": "cosine",
                "lr_warmup_steps": max(max_steps // 20, 1),
            },
        },

        "monitoring": {
            "save_to_wandb": True,
            "logging": {"log_every_n_steps": 50},
        },

        "checkpointing": {
            "save_every_n_steps": max_steps // 20,
            "checkpoint_every_n_steps": max_steps // 20,
            "auto_resume": True,
            "save_to_hf": True,
            "learning_dynamics": {
                "enabled": True,
                "log_every_n_steps": max(max_steps // 100, 1),
                "batch_size": 64,
                "sequence_idx": -1,
                "layer_suffixes": ["attention", "swiglu", "norm"],
                "compute_gradients": False,
                "learning_dynamics_dir": "learning_dynamics",
                "save_training_batch": True,
                "cleanup_local": True,
                "keep_n_local": 1,
            },
        },
    }

    return config


def generate_trilingual_config(
    condition: str,
    l1: str,
    l2: str,
    l3: str,
    hf_user: str = "Beetle-Data",
    max_steps: int = MAX_STEPS,
    seq_len: int = SEQ_LEN,
    batch_size: Optional[int] = None,
    devices: int = 8,
    world_size: Optional[int] = None,
) -> dict:
    """Generate a single trilingual curriculum training config."""
    config = generate_bilingual_config(
        condition, l1, hf_user, max_steps, seq_len, batch_size, devices,
        world_size=world_size,
    )

    # Update for trilingual
    config["run_name"] = f"beetlelm_{l1}-{l2}-{l3}_{condition}_curriculum"
    config["data"]["indexed_hf_repos"] = {
        "l1": f"{hf_user}/{l1}-curriculum-24B",
        "l2": f"{hf_user}/{l2}-curriculum-24B",
        "l3": f"{hf_user}/{l3}-curriculum-24B",
    }
    config["curriculum"]["lang_keys"] = [l1, l2, l3]

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate BeetleLM FineWeb curriculum training configs",
    )
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=BILINGUAL_CONDITIONS,
                        help="Conditions to generate (default: B1-B5)")
    parser.add_argument("--langs", type=str, nargs="+",
                        default=CORE_PAIRS,
                        help="Languages to generate (default: 20 core)")
    parser.add_argument("--output-dir", type=str,
                        default="configs/fineweb_curriculum")
    parser.add_argument("--hf-user", type=str, default="Beetle-Data")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--devices", type=int, default=8,
                        help="GPUs per node (fabric.devices). Default 8.")
    parser.add_argument("--world-size", type=int, default=None,
                        help="Total GPU count; drives per-GPU batch=512/world_size. "
                             "Defaults to --devices.")
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    total = 0

    for condition in args.conditions:
        for lang in args.langs:
            config = generate_bilingual_config(
                condition=condition,
                lang=lang,
                hf_user=args.hf_user,
                max_steps=args.max_steps,
                devices=args.devices,
                world_size=args.world_size,
            )

            out_dir = output_base / condition / lang
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "config.yaml"

            with open(out_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            total += 1

    print(f"Generated {total} configs in {output_base}/")
    print(f"Conditions: {args.conditions}")
    print(f"Languages: {args.langs}")
    print(f"\nTo train:")
    print(f"  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python script/train.py \\")
    print(f"    --config_path {output_base}/B1/fr/config.yaml")


if __name__ == "__main__":
    main()
