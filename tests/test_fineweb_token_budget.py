"""Regression test: every FineWeb YAML must consume ~24B tokens in 1 epoch.

Guards against the 8× drift that happened when ``target_effective_batch``
was set to 64 (per-GPU-equivalent) while per-GPU batch was also 64 on 8
GPUs — silently producing 191B tokens / 8 epochs instead of 24B / 1 epoch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.config._fineweb_defaults import (
    EFFECTIVE_BATCH,
    MAX_STEPS,
    SEQ_LEN,
    TOKENS_TOTAL,
    fineweb_per_gpu_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

FINEWEB_CONFIG_ROOTS = [
    REPO_ROOT / "configs" / "fineweb_bilingual",
    REPO_ROOT / "configs" / "fineweb_bilingual_hpc64",
    REPO_ROOT / "configs" / "fineweb_curriculum",
]

QUICKSTART = REPO_ROOT / "configs" / "quickstart_fineweb.yaml"


def _discover_configs() -> list[Path]:
    paths: list[Path] = []
    for root in FINEWEB_CONFIG_ROOTS:
        if root.exists():
            paths.extend(sorted(root.rglob("*.yaml")))
    if QUICKSTART.exists():
        paths.append(QUICKSTART)
    return paths


ALL_CONFIGS = _discover_configs()


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_fineweb_token_budget(path: Path):
    """max_steps × effective_batch × seq_len must equal 24B tokens (±0.1%)."""
    cfg = yaml.safe_load(path.read_text())
    t = cfg["training"]

    max_steps = t["max_steps"]
    effective = t["target_effective_batch"]
    seq_len   = t["seq_len"]

    tokens = max_steps * effective * seq_len
    assert abs(tokens - TOKENS_TOTAL) / TOKENS_TOTAL < 0.001, (
        f"{path.relative_to(REPO_ROOT)}: "
        f"max_steps={max_steps} * effective={effective} * seq_len={seq_len} "
        f"= {tokens:,} tokens (expected ~{TOKENS_TOTAL:,})"
    )


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_fineweb_effective_batch_is_honest(path: Path):
    """per_gpu_batch × fabric.devices must equal target_effective_batch.

    If this fails, the trainer's grad_accum computation will either over- or
    under-shoot the intended effective batch, producing the wrong token budget.
    """
    cfg = yaml.safe_load(path.read_text())
    t = cfg["training"]

    per_gpu  = t["batch_size"]
    devices  = t["fabric"]["devices"]
    target   = t["target_effective_batch"]

    # For multi-node HPC configs the YAML records devices/node (e.g. 4) while
    # the actual world_size is larger (e.g. 64). In that case we only require
    # (per_gpu * devices) divides target_effective_batch evenly.
    if per_gpu * devices == target:
        return  # single-node case

    assert target % (per_gpu * devices) == 0, (
        f"{path.relative_to(REPO_ROOT)}: "
        f"per_gpu_batch={per_gpu} * devices/node={devices} = {per_gpu*devices} "
        f"does not divide target_effective_batch={target} evenly"
    )


def test_fineweb_per_gpu_helper():
    """Helper must keep effective batch constant at EFFECTIVE_BATCH."""
    for devices in (8, 16, 32, 64, 128):
        per_gpu = fineweb_per_gpu_batch(devices)
        assert per_gpu * devices == EFFECTIVE_BATCH, (
            f"devices={devices}: per_gpu={per_gpu}, product={per_gpu*devices} "
            f"≠ {EFFECTIVE_BATCH}"
        )


def test_fineweb_defaults_invariants():
    """Central invariant: MAX_STEPS × EFFECTIVE_BATCH × SEQ_LEN ≈ 24B."""
    tokens = MAX_STEPS * EFFECTIVE_BATCH * SEQ_LEN
    assert abs(tokens - TOKENS_TOTAL) / TOKENS_TOTAL < 0.001
    assert MAX_STEPS == 91_552
    assert EFFECTIVE_BATCH == 512
