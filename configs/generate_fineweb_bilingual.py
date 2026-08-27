#!/usr/bin/env python3
"""
configs/generate_fineweb_bilingual.py

FineWeb-scale bilingual configs for Experiment 2.

Token budget: 12B per language (50:50 split, 24B total per pair), 1 epoch
Vocab size  : 50,000 BPE tokens
Steps       : 91,552  (= 24B / (512 effective batch * 512 seq_len))
Schedule    : Phase-aware log2 (subsampled for FineWeb scale)
Batching    : effective batch = 512 samples (constant); per-GPU batch = 512 / world_size

Two conditions (no sequential):
  - balanced:     100% L1 first 50%, then 50:50 L1:L2
  - simultaneous: 50:50 throughout, alternating L1:L2 epochs

Bidirectional: generates both (L1=non-English, L2=English) AND
               (L1=English, L2=non-English) when --bidirectional is set.

EWC/MAML/LR/LAMOL: Uses best settings from human-scale sweep (Phase 3).

Output: configs/fineweb_bilingual/{core,monolingual}/{condition}/train_{pk}.yaml

Usage:
  python configs/generate_fineweb_bilingual.py
  python configs/generate_fineweb_bilingual.py --bidirectional
  python configs/generate_fineweb_bilingual.py --bidirectional --hpc
  python configs/generate_fineweb_bilingual.py --best-config results/best_sweep_settings.json
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config._fineweb_defaults import (
    HF_ORG_DATA, HF_USER_MODELS, MODEL_BASE, TRAINING_DEFAULTS,
    CHECKPOINTING_DEFAULTS, CORE_LANGS, MAX_STEPS_BILINGUAL,
    MAX_STEPS_MONOLINGUAL, TOKENS_BILINGUAL, VOCAB_SIZE,
    TOKENIZER_TYPE, TOKENIZER_TOKENS, DATASET_SOURCE, DATASET_SCALE,
    FINEWEB_CONDITIONS, FINEWEB_TIER_1, FINEWEB_TIER_2,
    EFFECTIVE_BATCH, fineweb_per_gpu_batch,
    fineweb_l1_dataset, fineweb_en_dataset, fineweb_tokenizer_repo,
)
from src.config._repo_naming import build_model_run_name
from configs.ckpt_schedule import fineweb_checkpoint_schedule

OUT_DIR = Path("configs/fineweb_bilingual")

# Use condition codes from defaults (B1, B2, B3)
FINEWEB_MIXES = FINEWEB_CONDITIONS

# Language sets: use tier 1 + tier 2 from defaults
EXP2_LANGS = FINEWEB_TIER_1 + FINEWEB_TIER_2

# Best continual learning settings from human-scale sweep (update after Exp 1)
BEST_EWC_LAMBDA = None       # e.g., 10.0
BEST_MAML_RATIO = None       # e.g., 0.2
BEST_LR_BOOST = None         # e.g., 2.0
BEST_LAMOL_LAMBDA = None     # e.g., 0.2


def _make_cfg(
    l1: str,
    l2: str,
    condition: str,
    *,
    max_steps: int = MAX_STEPS_BILINGUAL,
    hpc: bool = False,
    world_size: int = 8,
    devices_per_node: int = 8,
    ewc_lambda=None,
    maml_ratio=None,
    lr_boost=None,
    lamol_lambda=None,
) -> dict:
    pk = f"{l1}_L1-{l2}_L2"
    run_name = build_model_run_name(
        dataset=DATASET_SCALE, condition=condition,
        l1=l1, l2=l2, vocab_sharing="shared",
        ewc_lambda=ewc_lambda, maml_ratio=maml_ratio,
        lr_boost=lr_boost, lamol_lambda=lamol_lambda,
    )

    # Determine datasets based on direction
    if l2 == "eng":
        l1_ds = fineweb_l1_dataset(l1)
        l2_ds = fineweb_en_dataset(l1)
        tok_repo = fineweb_tokenizer_repo(l1)
    elif l1 == "eng":
        # Reverse direction: L1=English, L2=non-English
        l1_ds = fineweb_en_dataset(l2)   # English data paired with L2
        l2_ds = fineweb_l1_dataset(l2)   # non-English data
        tok_repo = fineweb_tokenizer_repo(l2)  # same bilingual tokenizer
    else:
        raise ValueError(f"One of l1/l2 must be 'eng', got {l1}/{l2}")

    # Phase starts for checkpoint scheduling
    _PHASE_STARTS = {
        "B1": [0],
        "B2": [0, max_steps // 2],
        "B3": [0, max_steps // 2],
        "B4": [0, max_steps // 2],
        "B5": [0, int(max_steps * 0.8)],
    }
    phase_starts = _PHASE_STARTS.get(condition, [0])
    ckpt_steps = fineweb_checkpoint_schedule(
        phase_starts, max_steps=max_steps,
        extra_offsets=[2, 4, 6, 8],
    )

    training = copy.deepcopy(TRAINING_DEFAULTS)
    training["max_steps"] = max_steps

    # Per-GPU batch scales inversely with world_size so effective batch = 512.
    per_gpu = fineweb_per_gpu_batch(world_size)
    training["batch_size"] = per_gpu
    training["target_effective_batch"] = EFFECTIVE_BATCH
    training["fabric"]["devices"] = devices_per_node
    training["fabric"]["strategy"] = "ddp"

    # Keep model.params.batch_size in sync with training.batch_size
    model_params = dict(MODEL_BASE["params"])
    model_params["batch_size"] = per_gpu

    cfg = {
        "run_name": run_name,
        "hf_user": HF_USER_MODELS,
        "model": {
            "architecture": MODEL_BASE["architecture"],
            "params": model_params,
        },
        "data": {
            "data_mix": condition,
            "lang_sources": [l1_ds],
            "l2_streams": [l2_ds],
            "pretokenized": True,
            "pretokenized_cache_dir": f"./pretok_cache/{run_name}",
        },
        "tokenizer": {
            "pretrained_tokenizer_path": tok_repo,
            "dataset_source": DATASET_SOURCE,
            "vocab_size": VOCAB_SIZE,
            "vocab_sharing": "shared",
            "type": TOKENIZER_TYPE,
            "target_tokens": TOKENIZER_TOKENS,
            "output_dir": f"./tokenizers/fineweb_{pk}_shared",
            "overwrite": False,
            "push_to_hub": True,
        },
        "training": training,
        "checkpointing": {
            **CHECKPOINTING_DEFAULTS,
            "checkpoint_steps": ckpt_steps,
        },
        "curriculum": {
            "type": "smooth",
            "condition": condition,
            "mode": "tokens",
        },
    }

    # Continual learning settings
    if ewc_lambda is not None and ewc_lambda > 0:
        cfg["ewc"] = {"enabled": True, "lambda_": float(ewc_lambda)}
    else:
        cfg["ewc"] = {"enabled": False}

    if maml_ratio is not None and maml_ratio > 0:
        cfg["maml"] = {"enabled": True, "hybrid_ratio": float(maml_ratio)}
    else:
        cfg["maml"] = {"enabled": False}

    if lr_boost is not None and lr_boost > 1.0:
        cfg["training"]["optimization"]["exposure_lr"] = {
            "plasticity_boost": float(lr_boost),
        }

    if lamol_lambda is not None and lamol_lambda > 0:
        cfg["generative_replay"] = {
            "enabled": True,
            "lambda_replay": float(lamol_lambda),
            "trigger": "phase_change",
        }
    else:
        cfg["generative_replay"] = {"enabled": False}

    return cfg


def _make_mono_cfg(
    lang: str,
    *,
    max_steps: int = MAX_STEPS_MONOLINGUAL,
    hpc: bool = False,
    world_size: int = 8,
    devices_per_node: int = 8,
) -> dict:
    run_name = build_model_run_name(
        dataset=DATASET_SCALE, condition="monolingual", l1=lang, l2=None,
    )

    if lang == "eng":
        # Use any paired English dataset (e.g., en-for-nl)
        ds = fineweb_en_dataset("nld")
        tok_repo = fineweb_tokenizer_repo("nld")
    else:
        ds = fineweb_l1_dataset(lang)
        tok_repo = fineweb_tokenizer_repo(lang)

    ckpt_steps = fineweb_checkpoint_schedule([0], max_steps=max_steps)

    training = copy.deepcopy(TRAINING_DEFAULTS)
    training["max_steps"] = max_steps

    per_gpu = fineweb_per_gpu_batch(world_size)
    training["batch_size"] = per_gpu
    training["target_effective_batch"] = EFFECTIVE_BATCH
    training["fabric"]["devices"] = devices_per_node
    training["fabric"]["strategy"] = "ddp"

    model_params = dict(MODEL_BASE["params"])
    model_params["batch_size"] = per_gpu

    return {
        "run_name": run_name,
        "hf_user": HF_USER_MODELS,
        "model": {
            "architecture": MODEL_BASE["architecture"],
            "params": model_params,
        },
        "data": {
            "data_mix": "monolingual",
            "lang_sources": [ds],
            "pretokenized": True,
            "pretokenized_cache_dir": f"./pretok_cache/{run_name}",
        },
        "tokenizer": {
            "pretrained_tokenizer_path": tok_repo,
            "dataset_source": DATASET_SOURCE,
            "vocab_size": VOCAB_SIZE,
            "vocab_sharing": "shared",
            "type": TOKENIZER_TYPE,
            "target_tokens": TOKENIZER_TOKENS,
            "output_dir": f"./tokenizers/fineweb_{lang}_shared",
            "overwrite": False,
            "push_to_hub": True,
        },
        "training": training,
        "checkpointing": {
            **CHECKPOINTING_DEFAULTS,
            "checkpoint_steps": ckpt_steps,
        },
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "generative_replay": {"enabled": False},
    }


def _write_yaml(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def generate(args):
    hpc = args.hpc
    bidirectional = args.bidirectional
    langs = EXP2_LANGS

    # World size (total GPUs) drives per-GPU batch.
    # Defaults: HPC = 64 GPUs (16 nodes x 4), local = 8 GPUs (1 node x 8).
    if args.world_size is not None:
        world_size = args.world_size
    else:
        world_size = 64 if hpc else 8

    if args.devices_per_node is not None:
        devices_per_node = args.devices_per_node
    else:
        devices_per_node = 4 if hpc else 8

    per_gpu = fineweb_per_gpu_batch(world_size)

    # Load best settings from file if provided
    ewc_lambda = BEST_EWC_LAMBDA
    maml_ratio = BEST_MAML_RATIO
    lr_boost = BEST_LR_BOOST
    lamol_lambda = BEST_LAMOL_LAMBDA

    if args.best_config:
        with open(args.best_config) as f:
            best = json.load(f)
        ewc_lambda = best.get("ewc_lambda", ewc_lambda)
        maml_ratio = best.get("maml_ratio", maml_ratio)
        lr_boost = best.get("lr_boost", lr_boost)
        lamol_lambda = best.get("lamol_lambda", lamol_lambda)
        print(f"Loaded best settings: ewc={ewc_lambda}, maml={maml_ratio}, "
              f"lr={lr_boost}, lamol={lamol_lambda}")

    print(f"max_steps       : {MAX_STEPS_BILINGUAL:,}")
    print(f"world_size      : {world_size}  (per-GPU batch={per_gpu}, effective={per_gpu*world_size})")
    print(f"devices/node    : {devices_per_node}")
    print(f"vocab_size      : {VOCAB_SIZE:,}")
    print(f"conditions      : {FINEWEB_CONDITIONS}")
    print(f"bidirectional   : {bidirectional}")
    print(f"hpc             : {hpc}")
    print(f"languages       : {len(langs)} ({langs})")

    n_bi = 0
    n_mono = 0

    non_eng = [l for l in langs if l != "eng"]

    # Bilingual: L1=non-English, L2=English
    for l1 in non_eng:
        for condition in FINEWEB_MIXES:
            cfg = _make_cfg(
                l1, "eng", condition, hpc=hpc,
                world_size=world_size, devices_per_node=devices_per_node,
                ewc_lambda=ewc_lambda, maml_ratio=maml_ratio,
                lr_boost=lr_boost, lamol_lambda=lamol_lambda,
            )
            _write_yaml(cfg, OUT_DIR / "core" / condition / f"train_{l1}_L1-eng_L2.yaml")
            n_bi += 1

    # Bidirectional: L1=English, L2=non-English
    if bidirectional:
        for l2 in non_eng:
            for condition in FINEWEB_MIXES:
                cfg = _make_cfg(
                    "eng", l2, condition, hpc=hpc,
                    world_size=world_size, devices_per_node=devices_per_node,
                    ewc_lambda=ewc_lambda, maml_ratio=maml_ratio,
                    lr_boost=lr_boost, lamol_lambda=lamol_lambda,
                )
                _write_yaml(cfg, OUT_DIR / "core" / condition / f"train_eng_L1-{l2}_L2.yaml")
                n_bi += 1

    # Monolingual baselines
    mono_langs = set(non_eng)
    mono_langs.add("eng")
    for lang in sorted(mono_langs):
        cfg = _make_mono_cfg(
            lang, hpc=hpc,
            world_size=world_size, devices_per_node=devices_per_node,
        )
        _write_yaml(cfg, OUT_DIR / "monolingual" / f"train_{lang}.yaml")
        n_mono += 1

    total = n_bi + n_mono
    print(f"\nGenerated {total} configs -> {OUT_DIR.resolve()}")
    print(f"  bilingual   : {n_bi}")
    print(f"  monolingual : {n_mono}")
    if bidirectional:
        print(f"  (includes both L1->L2 and L2->L1 directions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FineWeb-scale bilingual training configs (Experiment 2)"
    )
    parser.add_argument("--bidirectional", action="store_true",
                        help="Generate both L1->L2 and L2->L1 directions")
    parser.add_argument("--hpc", action="store_true",
                        help="Cambridge HPC preset: world_size=64 (16 nodes x 4 GPUs/node)")
    parser.add_argument("--world-size", type=int, default=None,
                        help="Total GPU count; drives per-GPU batch=512/world_size. "
                             "Default: 64 if --hpc else 8.")
    parser.add_argument("--devices-per-node", type=int, default=None,
                        help="GPUs per node (fabric.devices in YAML). "
                             "Default: 4 if --hpc else 8.")
    parser.add_argument("--best-config", type=str, default=None,
                        help="Path to best_sweep_settings.json from Experiment 1")
    args = parser.parse_args()

    generate(args)
