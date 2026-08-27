"""tests/test_polca_block_present.py

Asserts that every training YAML in the canonical sweep directories
declares a top-level ``polca:`` block.  This guards against new YAMLs
being added to the sweeps without the POLCA-compatibility block that
``script/add_polca_block.py`` is meant to maintain.

If you add a new sweep directory that should track POLCA, add it to
``SWEEP_ROOTS`` below.  If you intentionally have a YAML in one of these
directories that should not carry a ``polca:`` block, add it to
``ALLOWLIST``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent

SWEEP_ROOTS = [
    REPO_ROOT / "configs" / "b3_condition_sweep_2b",
    REPO_ROOT / "configs" / "humanscale_fineweb_bilingual",
    REPO_ROOT / "configs" / "fineweb_bilingual_2b",
    REPO_ROOT / "configs" / "fineweb_bilingual",
    REPO_ROOT / "configs" / "fineweb_bilingual_hpc64",
    REPO_ROOT / "configs" / "fineweb_curriculum",
]

# Paths (relative to REPO_ROOT) that are intentionally exempt.
ALLOWLIST: set[str] = set()


def _yaml_files() -> list[Path]:
    out: list[Path] = []
    for root in SWEEP_ROOTS:
        if not root.exists():
            continue
        out.extend(sorted(root.rglob("*.yaml")))
    return out


@pytest.mark.parametrize("path", _yaml_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_polca_block_present(path: Path) -> None:
    rel = str(path.relative_to(REPO_ROOT))
    if rel in ALLOWLIST:
        pytest.skip(f"explicitly allow-listed: {rel}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg is not None, f"empty YAML: {rel}"
    assert "polca" in cfg, (
        f"missing top-level 'polca:' block in {rel}.  "
        f"Run: python script/add_polca_block.py"
    )
    block = cfg["polca"]
    assert isinstance(block, dict), f"polca block must be a mapping: {rel}"
    assert "enabled" in block, f"polca.enabled must be set in {rel}"
    assert isinstance(block["enabled"], bool), (
        f"polca.enabled must be a bool in {rel}"
    )


def test_at_least_one_yaml_found() -> None:
    """Sanity: the test discovery itself works."""
    files = _yaml_files()
    assert len(files) > 0, (
        "no YAMLs discovered in sweep roots — has the configs layout moved?"
    )
