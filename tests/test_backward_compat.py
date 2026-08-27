"""Backward-compatibility regression: every shipped YAML still parses and
declares an architecture that is present in ``MODEL_REGISTRY``.

This catches typos, dropped architectures, and registry refactor regressions
without paying the cost of constructing a full ``BeetleLMTrainer`` (which
pulls in distributed-training, tokenizer downloads, and HF-Hub auth).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.training import _arch_imports  # noqa: F401  (registers builtins)
from beetlelm.training.registry import MODEL_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOTS = [REPO_ROOT / "configs"]


def _discover_yamls() -> list[Path]:
    paths: list[Path] = []
    for root in CONFIG_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.yaml"):
            # Skip generator-output samples or anything explicitly opt-out.
            paths.append(p)
    return sorted(paths)


CONFIG_PATHS = _discover_yamls()


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=[
    str(p.relative_to(REPO_ROOT)) for p in CONFIG_PATHS
])
def test_yaml_parses(config_path: Path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg is not None, f"empty YAML: {config_path}"


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=[
    str(p.relative_to(REPO_ROOT)) for p in CONFIG_PATHS
])
def test_architecture_is_registered(config_path: Path):
    """If a YAML declares a model.architecture, it must be in MODEL_REGISTRY."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        pytest.skip(f"no model block in {config_path.name}")
    arch = model_cfg.get("architecture")
    if arch is None:
        pytest.skip(f"no model.architecture in {config_path.name}")
    assert arch in MODEL_REGISTRY, (
        f"{config_path.relative_to(REPO_ROOT)}: architecture {arch!r} "
        f"not in MODEL_REGISTRY (known: {sorted(MODEL_REGISTRY)})"
    )
