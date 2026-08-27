#!/usr/bin/env python3
"""
CurriculumCheckpointScheduler

Handles phase- and step-based checkpointing decisions during curriculum
training. Produces a compressed ``.pt.gz`` resume artifact on disk.

HF Hub upload is performed exclusively by
``BeetleTrainer._upload_step_checkpoint_to_hf`` (dispatched asynchronously
after ``BeetleTrainer._save_step_checkpoint_local``); this class only decides
*when* a checkpoint should happen and writes the compressed resume blob.
DDP-safe via Fabric.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Any, Dict

import torch
from lightning.fabric import Fabric
from transformers import PreTrainedTokenizerBase

from beetlelm.checkpointing.training import (
    save_evaluation_results,
    save_learning_dynamics_states,
)
from beetlelm.bilingual.data.pipeline_state import DataPipelineState
from configs.ckpt_schedule import phase_log2_schedule, fineweb_checkpoint_schedule


class CurriculumCheckpointScheduler:

    def __init__(
        self,
        configs: Dict[str, Any],
        pipeline_state: DataPipelineState,
        fabric: Fabric,
        model: Any,
        optimizer: Any,
        lr_scheduler: Any,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.configs        = configs
        self.pipeline_state = pipeline_state
        self.fabric         = fabric
        self.model          = model
        self.optimizer      = optimizer
        self.lr_scheduler   = lr_scheduler
        self.tokenizer      = tokenizer

        self._checkpoint_steps_set = self._compute_checkpoint_schedule()

    # ── Schedule computation ─────────────────────────────────────────────────

    def _compute_checkpoint_schedule(self) -> set:
        """Compute checkpoint steps from curriculum phase boundaries.

        Uses explicit checkpoint_steps from config if provided (backward compat
        for generated configs).  Otherwise, dynamically computes a phase-aware
        log₂ schedule from ``pipeline_state.data_mix.step_boundaries()``.

        DDP-safe: deterministic computation from config values identical on all
        ranks.
        """
        ckpt_cfg = self.configs.get("checkpointing", {})
        max_steps = self.configs.get("training", {}).get("max_steps", 10_000)

        # Backward compat: explicit list in YAML takes precedence
        explicit = ckpt_cfg.get("checkpoint_steps")
        if explicit is not None:
            schedule = set(int(s) for s in explicit)
            schedule.add(max_steps)
            if self.fabric.is_global_zero:
                print(f"[Checkpointing] Using explicit schedule from config: "
                      f"{len(schedule)} steps")
            return schedule

        # Dynamic computation from curriculum phase boundaries
        phase_starts = self.pipeline_state.data_mix.step_boundaries()

        FINEWEB_THRESHOLD = 100_000
        if max_steps > FINEWEB_THRESHOLD:
            steps = fineweb_checkpoint_schedule(
                phase_starts, max_steps=max_steps, extra_offsets=[2, 4, 6, 8],
            )
        else:
            steps = phase_log2_schedule(
                phase_starts, max_steps=max_steps, extra_offsets=[2, 4, 6, 8],
            )

        steps_set = set(steps)
        steps_set.add(max_steps)

        if self.fabric.is_global_zero:
            sorted_steps = sorted(steps_set)
            print(f"[Checkpointing] Dynamic log₂ schedule: "
                  f"{len(sorted_steps)} checkpoints")
            print(f"[Checkpointing] Phase starts: {phase_starts}")
            print(f"[Checkpointing] Schedule: {sorted_steps}")

        return steps_set

    # ── Checkpoint decision ───────────────────────────────────────────────────

    @property
    def checkpoint_steps(self) -> frozenset:
        """Log₂-spaced checkpoint step set (phase-aware), frozen for external use."""
        return frozenset(self._checkpoint_steps_set)

    def should_checkpoint(self, step: int, phase_changed: bool = False) -> bool:
        """Return True on phase transitions or at scheduled log₂ steps only.

        Periodic (linear) fallbacks have been removed: every checkpoint write
        is gated by either a curriculum phase boundary or membership in the
        precomputed log₂ schedule (``phase_log2_schedule`` /
        ``fineweb_checkpoint_schedule``). This eliminates the disk bloat that
        plagued earlier 24B runs.
        """
        if phase_changed:
            return True
        return step in self._checkpoint_steps_set

    def should_checkpoint_for_hf(self, step: int, phase_changed: bool = False) -> bool:
        """Alias used by the trainer to decide whether to invoke the canonical
        HF uploader on this step. Pass-through to ``should_checkpoint``."""
        return self.should_checkpoint(step, phase_changed)

    # ── Main entry ────────────────────────────────────────────────────────────

    def checkpoint(self, step: int, phase_changed: bool = False) -> None:
        """Write a compressed ``.pt.gz`` resume artifact on disk when scheduled.

        HF Hub upload is handled separately by the asynchronous
        ``BeetleTrainer._upload_step_checkpoint_to_hf`` dispatch path.
        """
        if not self.should_checkpoint(step, phase_changed):
            return

        if self.fabric.is_global_zero:
            pt_path = self._save_training_checkpoint(step)
            self._compress_checkpoint(pt_path)

        self.fabric.barrier()

    # ── Fast checkpoint save ──────────────────────────────────────────────────

    def _save_training_checkpoint(self, step: int) -> Path:
        run_name = self.configs.get("run_name", "default_run")
        runs_dir = Path(self.configs["checkpointing"].get("runs_dir", "./runs"))
        ckpt_dir = runs_dir / run_name / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        pt_path = ckpt_dir / f"step_{step}.pt"
        state = {
            "model":         self.model.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "scheduler":     self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "pipeline_state": self.pipeline_state.state_dict(),
        }
        torch.save(state, pt_path, pickle_protocol=4, _use_new_zipfile_serialization=True)
        return pt_path

    # ── Compression ───────────────────────────────────────────────────────────

    def _compress_checkpoint(self, pt_path: Path) -> Path:
        gz_path = pt_path.with_suffix(".pt.gz")
        with (
            open(pt_path, "rb") as f_in,
            gzip.open(gz_path, "wb", compresslevel=3) as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)
        pt_path.unlink(missing_ok=True)
        return gz_path
