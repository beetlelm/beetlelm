#!/usr/bin/env python3
"""
configs/generate_humanscale_fineweb.py

100M FineWeb bilingual configs.

FineWeb data + tokenizers (matching 24B FineWeb pipeline) at a Human-Scale
token budget (100M total per pair) and compute footprint (max_steps=10_000,
effective batch=64, devices=4).

Models push to ``Beetle-FineWeb-100M``; ``run_name`` leads with ``fineweb-100m-``
so the data origin is unambiguous on the HF model list.

Token budget : 100M total per pair, curriculum sets L1/L2 split.
Steps        : 10,000 (matches existing Human-Scale step counts).
Conditions   : B1-B5
Languages    : nld, deu, zho, spa, jpn, ita, rus, pol, tur, fil, hin, eus, arb
               (each paired with English; English is L1 or L2).
Bidirectional: emits both directions when --bidirectional is set.

Output       : configs/humanscale_fineweb_bilingual/{core,monolingual}/{condition}/...

Usage:
  python configs/generate_humanscale_fineweb.py
  python configs/generate_humanscale_fineweb.py --bidirectional
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config._humanscale_fineweb_defaults import (
    HF_USER_MODELS, MODEL_BASE, TRAINING_DEFAULTS,
    CHECKPOINTING_DEFAULTS, MAX_STEPS_BILINGUAL,
    MAX_STEPS_MONOLINGUAL, VOCAB_SIZE,
    TOKENIZER_TYPE, TOKENIZER_TOKENS, DATASET_SOURCE, DATASET_SCALE,
    CONDITIONS, EXP_LANGS,
    EFFECTIVE_BATCH, BATCH_SIZE,
    fineweb_l1_dataset, fineweb_en_dataset, fineweb_tokenizer_repo,
)
from src.config._repo_naming import build_model_run_name
from configs.ckpt_schedule import phase_log2_schedule

OUT_DIR = Path("configs/humanscale_fineweb_bilingual")

# Pre-staged subset shards (see script/data/stage_pretok_subset.py and
# the Stage-0 step in script/launch/launch_humanscale_b1_b5_all_safe.sh).
# Replaces the 28B-token sources with 100M-budget subsets, so the loader
# pulls hundreds of MB once per language pair instead of tens to hundreds
# of GB per run. The cache directory is shared across all tiers B1-B5 so
# tier B2..B5 hit the cache-hit branch in pretokenize.py.
PRETOK_SUFFIX = "100M-pretok"
SHARED_CACHE_ROOT = "./pretok_cache/humanscale-100m"


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
    devices: int = 8,
) -> dict:
    pk = f"{l1}_L1-{l2}_L2"
    run_name = build_model_run_name(
        dataset=DATASET_SCALE, condition=condition,
        l1=l1, l2=l2, vocab_sharing="shared",
    )

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
    # Human-scale uses the full log2 schedule (not the subsampled FineWeb one).
    ckpt_steps = phase_log2_schedule(phase_starts, max_steps=max_steps)

    training = copy.deepcopy(TRAINING_DEFAULTS)
    training["max_steps"] = max_steps
    training["fabric"]["devices"] = devices

    model_params = dict(MODEL_BASE["params"])
    model_params["batch_size"] = BATCH_SIZE

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
            "output_dir": f"./tokenizers/fineweb_humanscale_{pk}_shared",
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
    devices: int = 8,
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

    ckpt_steps = phase_log2_schedule([0], max_steps=max_steps)

    training = copy.deepcopy(TRAINING_DEFAULTS)
    training["max_steps"] = max_steps
    training["fabric"]["devices"] = devices

    model_params = dict(MODEL_BASE["params"])
    model_params["batch_size"] = BATCH_SIZE

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
            "output_dir": f"./tokenizers/fineweb_humanscale_{lang}_shared",
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
    devices = args.devices if args.devices is not None else 8
    langs = EXP_LANGS

    print(f"max_steps       : {MAX_STEPS_BILINGUAL:,}")
    print(f"effective_batch : {EFFECTIVE_BATCH}")
    print(f"devices         : {devices}")
    print(f"vocab_size      : {VOCAB_SIZE:,}")
    print(f"conditions      : {CONDITIONS}")
    print(f"bidirectional   : {bidirectional}")
    print(f"languages       : {len(langs)} ({langs})")

    n_bi = 0
    n_mono = 0

    non_eng = [l for l in langs if l != "eng"]

    for l1 in non_eng:
        for condition in CONDITIONS:
            cfg = _make_cfg(l1, "eng", condition, devices=devices)
            _write_yaml(cfg, OUT_DIR / "core" / condition / f"train_{l1}_L1-eng_L2.yaml")
            n_bi += 1

    if bidirectional:
        for l2 in non_eng:
            for condition in CONDITIONS:
                cfg = _make_cfg("eng", l2, condition, devices=devices)
                _write_yaml(cfg, OUT_DIR / "core" / condition / f"train_eng_L1-{l2}_L2.yaml")
                n_bi += 1

    mono_langs = set(non_eng)
    mono_langs.add("eng")
    for lang in sorted(mono_langs):
        cfg = _make_mono_cfg(lang, devices=devices)
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
        description="Generate Human-Scale-on-FineWeb bilingual training configs"
    )
    parser.add_argument("--bidirectional", action="store_true",
                        help="Generate both L1->L2 and L2->L1 directions")
    parser.add_argument("--devices", type=int, default=None,
                        help="GPUs per node (fabric.devices in YAML). Default: 8.")
    args = parser.parse_args()

    generate(args)
