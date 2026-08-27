#!/usr/bin/env python3
"""
configs/generate_fineweb_2b.py

FineWeb bilingual configs at the 2B-token horizon (Experiment 2B).

Same data pipeline, model architecture, and effective-batch invariants as the
24B FineWeb runs (see ``configs/generate_fineweb_bilingual.py``); only the
total token budget (and downstream knobs derived from it) change.

Token budget : 2B total per pair, curriculum sets L1/L2 split.
Steps        : 7,629 (= 2B / (512 effective batch * 512 seq_len)).
Conditions   : B1-B5
Languages    : nld, deu, zho, spa, jpn, ita, rus, pol, tur, fil, hin, eus, arb
               (each paired with English; English is always L1 or L2)
Bidirectional: emits both ``train_{lang}_L1-eng_L2.yaml`` and
               ``train_eng_L1-{lang}_L2.yaml`` when --bidirectional is set.

Output       : configs/fineweb_bilingual_2b/{core,monolingual}/{condition}/...

Usage:
  python configs/generate_fineweb_2b.py
  python configs/generate_fineweb_2b.py --bidirectional
  python configs/generate_fineweb_2b.py --bidirectional --hpc
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config._fineweb_2b_defaults import (
    HF_USER_MODELS, MODEL_BASE, TRAINING_DEFAULTS,
    CHECKPOINTING_DEFAULTS, MAX_STEPS_BILINGUAL,
    MAX_STEPS_MONOLINGUAL, VOCAB_SIZE,
    TOKENIZER_TYPE, TOKENIZER_TOKENS, DATASET_SOURCE, DATASET_SCALE,
    FINEWEB_CONDITIONS, EXP_LANGS,
    EFFECTIVE_BATCH, fineweb_per_gpu_batch,
    fineweb_l1_dataset, fineweb_en_dataset, fineweb_tokenizer_repo,
)
from src.config._repo_naming import build_model_run_name
from configs.ckpt_schedule import fineweb_checkpoint_schedule

OUT_DIR = Path("configs/fineweb_bilingual_2b")

FINEWEB_MIXES = FINEWEB_CONDITIONS  # B1..B5

# Pre-staged subset shards (see script/data/stage_pretok_subset.py and
# the Stage-0 step in script/launch/launch_fineweb_2b_groupA_safe.sh).
# Replaces the 28B-token sources with 2B-budget subsets, so the loader
# pulls a few GB once per language pair instead of tens to hundreds of
# GB per run. The cache directory is shared across all tiers B1-B5 so
# tier B2..B5 hit the cache-hit branch in pretokenize.py.
PRETOK_SUFFIX = "2B-pretok"
SHARED_CACHE_ROOT = "./pretok_cache/fineweb-2b"


def _to_subset(repo: str, suffix: str = PRETOK_SUFFIX) -> str:
    """``Beetle-Data/nl-28B`` -> ``Beetle-Data/nl-{suffix}`` (idempotent)."""
    if repo.endswith(f"-{suffix}"):
        return repo
    if repo.endswith("-28B"):
        return repo[: -len("-28B")] + f"-{suffix}"
    return repo


def _phase_starts(condition: str, max_steps: int) -> list:
    """Phase boundaries for checkpoint scheduling, per condition."""
    if condition == "B1":
        return [0]
    if condition in ("B2", "B3", "B4"):
        return [0, max_steps // 2]
    if condition == "B5":
        return [0, int(max_steps * 0.8)]
    raise ValueError(f"unknown condition: {condition!r}")


def _make_cfg(
    l1: str,
    l2: str,
    condition: str,
    *,
    max_steps: int = MAX_STEPS_BILINGUAL,
    world_size: int = 8,
    devices_per_node: int = 8,
) -> dict:
    pk = f"{l1}_L1-{l2}_L2"
    run_name = build_model_run_name(
        dataset=DATASET_SCALE, condition=condition,
        l1=l1, l2=l2, vocab_sharing="shared",
    )

    # Determine datasets based on direction
    if l2 == "eng":
        l1_ds = _to_subset(fineweb_l1_dataset(l1))
        l2_ds = _to_subset(fineweb_en_dataset(l1))
        tok_repo = fineweb_tokenizer_repo(l1)
    elif l1 == "eng":
        l1_ds = _to_subset(fineweb_en_dataset(l2))
        l2_ds = _to_subset(fineweb_l1_dataset(l2))
        tok_repo = fineweb_tokenizer_repo(l2)
    else:
        raise ValueError(f"One of l1/l2 must be 'eng', got {l1}/{l2}")

    phase_starts = _phase_starts(condition, max_steps)
    ckpt_steps = fineweb_checkpoint_schedule(
        phase_starts, max_steps=max_steps,
        extra_offsets=[2, 4, 6, 8],
    )

    training = copy.deepcopy(TRAINING_DEFAULTS)
    training["max_steps"] = max_steps

    per_gpu = fineweb_per_gpu_batch(world_size)
    training["batch_size"] = per_gpu
    training["target_effective_batch"] = EFFECTIVE_BATCH
    training["fabric"]["devices"] = devices_per_node
    training["fabric"]["strategy"] = "ddp"

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
            "pretokenized_cache_dir": f"{SHARED_CACHE_ROOT}/{pk}",
        },
        "tokenizer": {
            "pretrained_tokenizer_path": tok_repo,
            "dataset_source": DATASET_SOURCE,
            "vocab_size": VOCAB_SIZE,
            "vocab_sharing": "shared",
            "type": TOKENIZER_TYPE,
            "target_tokens": TOKENIZER_TOKENS,
            "output_dir": f"./tokenizers/fineweb_2b_{pk}_shared",
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
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "generative_replay": {"enabled": False},
    }

    return cfg


def _make_mono_cfg(
    lang: str,
    *,
    max_steps: int = MAX_STEPS_MONOLINGUAL,
    world_size: int = 8,
    devices_per_node: int = 8,
) -> dict:
    run_name = build_model_run_name(
        dataset=DATASET_SCALE, condition="monolingual", l1=lang, l2=None,
    )

    if lang == "eng":
        ds = _to_subset(fineweb_en_dataset("nld"))
        tok_repo = fineweb_tokenizer_repo("nld")
    else:
        ds = _to_subset(fineweb_l1_dataset(lang))
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
            "pretokenized_cache_dir": f"{SHARED_CACHE_ROOT}/mono-{lang}",
        },
        "tokenizer": {
            "pretrained_tokenizer_path": tok_repo,
            "dataset_source": DATASET_SOURCE,
            "vocab_size": VOCAB_SIZE,
            "vocab_sharing": "shared",
            "type": TOKENIZER_TYPE,
            "target_tokens": TOKENIZER_TOKENS,
            "output_dir": f"./tokenizers/fineweb_2b_{lang}_shared",
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
    bidirectional = args.bidirectional
    langs = EXP_LANGS

    if args.world_size is not None:
        world_size = args.world_size
    else:
        world_size = 64 if args.hpc else 8

    if args.devices_per_node is not None:
        devices_per_node = args.devices_per_node
    else:
        devices_per_node = 4 if args.hpc else 8

    per_gpu = fineweb_per_gpu_batch(world_size)

    print(f"max_steps       : {MAX_STEPS_BILINGUAL:,}")
    print(f"world_size      : {world_size}  (per-GPU batch={per_gpu}, effective={per_gpu*world_size})")
    print(f"devices/node    : {devices_per_node}")
    print(f"vocab_size      : {VOCAB_SIZE:,}")
    print(f"conditions      : {FINEWEB_CONDITIONS}")
    print(f"bidirectional   : {bidirectional}")
    print(f"hpc             : {args.hpc}")
    print(f"languages       : {len(langs)} ({langs})")

    n_bi = 0
    n_mono = 0

    non_eng = [l for l in langs if l != "eng"]

    # Bilingual: L1=non-English, L2=English
    for l1 in non_eng:
        for condition in FINEWEB_MIXES:
            cfg = _make_cfg(
                l1, "eng", condition,
                world_size=world_size, devices_per_node=devices_per_node,
            )
            _write_yaml(cfg, OUT_DIR / "core" / condition / f"train_{l1}_L1-eng_L2.yaml")
            n_bi += 1

    if bidirectional:
        for l2 in non_eng:
            for condition in FINEWEB_MIXES:
                cfg = _make_cfg(
                    "eng", l2, condition,
                    world_size=world_size, devices_per_node=devices_per_node,
                )
                _write_yaml(cfg, OUT_DIR / "core" / condition / f"train_eng_L1-{l2}_L2.yaml")
                n_bi += 1

    mono_langs = set(non_eng)
    mono_langs.add("eng")
    for lang in sorted(mono_langs):
        cfg = _make_mono_cfg(
            lang, world_size=world_size, devices_per_node=devices_per_node,
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
        description="Generate FineWeb 2B-token bilingual training configs (Experiment 2B)"
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
    args = parser.parse_args()

    generate(args)
