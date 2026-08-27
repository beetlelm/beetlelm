"""Tests for beetlelm.checkpointing.training.prune_local_checkpoints.

The prune helper used to take just ``ckpt_dir`` + ``keep_last_n`` and
keep the highest-step entries.  That broke whenever a fresh run landed
in a ``runs/<name>/checkpoints/`` directory that still held leftover
``step_N/`` dirs from a prior, longer run: the just-written ``step_0/``
ranked below the stale ``step_4/`` and was the one chosen for deletion,
making the post-save HF round-trip smoke test fail with
"Unrecognized model in step_0" — config.json had been wiped seconds
after save_pretrained wrote it.

The fix: pass ``current_step`` to prune, and treat any on-disk
``step_N`` with ``N > current_step`` as a leftover that this run did
not produce — leave it alone *and* exclude it from the keep_last_n
quota.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from beetlelm.checkpointing.training import prune_local_checkpoints


def _populate(root: Path, step_nums: list[int]) -> None:
    for n in step_nums:
        (root / f"step_{n}").mkdir()
        (root / f"step_{n}" / "config.json").write_text("{}")


def test_prune_without_current_step_keeps_highest(tmp_path: Path) -> None:
    """Backwards-compatible behaviour when current_step is omitted."""
    _populate(tmp_path, [0, 1, 2])
    prune_local_checkpoints(str(tmp_path), keep_last_n=1)
    survivors = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert survivors == ["step_2"], survivors


def test_prune_with_current_step_ignores_higher_leftovers(tmp_path: Path) -> None:
    """The just-written checkpoint must always survive even when stale
    higher-step dirs from a previous run are sitting in the same tree."""
    # Simulate the production failure: this run wrote step_0; a prior
    # longer run left step_1 and step_4 behind.
    _populate(tmp_path, [0, 1, 4])
    prune_local_checkpoints(str(tmp_path), keep_last_n=1, current_step=0)
    survivors = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    # step_0 must be preserved (it's the just-written one).
    # step_1 and step_4 are leftovers: NOT this run's, so we don't touch
    # them either — preserves user/operator data the run didn't create.
    assert "step_0" in survivors, survivors
    assert "step_1" in survivors, survivors
    assert "step_4" in survivors, survivors


def test_prune_with_current_step_still_prunes_old_dirs(tmp_path: Path) -> None:
    """When this run *itself* has saved enough older checkpoints, prune
    should still delete the surplus old ones — the bug fix only protects
    against future-step leftovers, not past-step ones."""
    _populate(tmp_path, [0, 100, 200, 300])
    prune_local_checkpoints(str(tmp_path), keep_last_n=2, current_step=300)
    survivors = sorted(
        (int(p.name.removeprefix("step_")), p.name)
        for p in tmp_path.iterdir() if p.is_dir()
    )
    assert [s[1] for s in survivors] == ["step_200", "step_300"], survivors


def test_prune_keep_zero_is_noop(tmp_path: Path) -> None:
    _populate(tmp_path, [0, 1, 2])
    prune_local_checkpoints(str(tmp_path), keep_last_n=0)
    survivors = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert survivors == ["step_0", "step_1", "step_2"], survivors
