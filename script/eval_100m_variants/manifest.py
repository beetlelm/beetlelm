"""
manifest.py — single source of truth for the 100M nld-eng B3 variant sweep.

Each entry resolves to:
  variant_name : short label used in plots and CSV headers
  hf_repo      : HuggingFace model repo (variant pushes go to *-Variants org)
  yaml_path    : training config (so we can read the canonical step schedule)

Imported by both the per-checkpoint eval driver and the plot generator so
they agree on what to score and what to plot.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple

import yaml


class Variant(NamedTuple):
    name: str          # short label
    hf_repo: str       # owner/name on HF Hub
    yaml_path: Path    # absolute path to training YAML


_REPO = Path("/root/beetlelm")
_CORE = _REPO / "configs/humanscale_fineweb_bilingual/core/B3"
_VAR  = _CORE / "variants_nld_eng_100m"


def _v(name: str, yaml_rel: str, hf_user: str | None = None) -> Variant:
    yaml_path = _REPO / yaml_rel if not yaml_rel.startswith("/") else Path(yaml_rel)
    cfg = yaml.safe_load(yaml_path.read_text())
    user = hf_user or cfg["hf_user"]
    return Variant(name=name, hf_repo=f"{user}/{cfg['run_name']}", yaml_path=yaml_path)


def all_variants() -> List[Variant]:
    return [
        _v("baseline",     "configs/humanscale_fineweb_bilingual/core/B3/train_nld_L1-eng_L2.safe.yaml"),
        _v("ewc",          "configs/humanscale_fineweb_bilingual/core/B3/variants_nld_eng_100m/train_nld_L1-eng_L2.ewc.yaml"),
        _v("lamol",        "configs/humanscale_fineweb_bilingual/core/B3/variants_nld_eng_100m/train_nld_L1-eng_L2.lamol.yaml"),
        _v("maml",         "configs/humanscale_fineweb_bilingual/core/B3/variants_nld_eng_100m/train_nld_L1-eng_L2.maml.yaml"),
        _v("pre-pretrain", "configs/humanscale_fineweb_bilingual/core/B3/variants_nld_eng_100m/train_nld_L1-eng_L2.pre-pretrain.yaml"),
        _v("alibi",        "configs/humanscale_fineweb_bilingual/core/B3/variants_nld_eng_100m/train_nld_L1-eng_L2.alibi.yaml"),
        _v("layer-growth", "configs/humanscale_fineweb_bilingual/core/B3/variants_nld_eng_100m/train_nld_L1-eng_L2.layer-growth.yaml"),
    ]


def checkpoint_steps(v: Variant) -> List[int]:
    cfg = yaml.safe_load(v.yaml_path.read_text())
    return [int(s) for s in cfg["checkpointing"]["checkpoint_steps"]]


if __name__ == "__main__":
    for v in all_variants():
        steps = checkpoint_steps(v)
        print(f"{v.name:14s} {v.hf_repo:80s}  {len(steps)} ckpts (max {steps[-1]})")
