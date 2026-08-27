#src/checkpointing/curriculum_checkpoint.py

"""
Pico-style curriculum checkpoint callback and config.

Works with both the pico trainer (dataclass configs) and the beetle trainer
(dict configs) by delegating all I/O to beetlelm.checkpointing.training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import pytorch_lightning as pl

from beetlelm.checkpointing.training import (
    compute_learning_dynamics_states,
    load_checkpoint,
    save_checkpoint,
    save_evaluation_results,
    save_learning_dynamics_states,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckpointingConfig:
    """
    Checkpointing config for the curriculum / pico trainer.

    The beetle trainer supplies an equivalent plain dict; all fields here
    match the keys expected by _CkptCfg._DEFAULTS in training.py.
    """

    # Directory layout
    runs_dir:                   str = "runs"
    run_name:                   str = "run"
    checkpoints_dir:            str = "checkpoints"
    fabric_checkpoint_dir:      str = "fabric_checkpoint"
    fabric_checkpoint_filename: str = "checkpoint.pt"
    logs_dir:                   str = "logs"

    # Save behaviour
    save_every_n_steps:     int  = 1000
    keep_local_checkpoints: int  = 1       # 0 = keep all
    save_to_hf:             bool = False

    # Learning dynamics
    learning_dynamics: dict = field(default_factory=dict)
    compute_gradients: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumCheckpointCallback(pl.Callback):
    """
    Lightning callback for Pico-style curriculum checkpointing.

    Automatically saves at steps determined by the CurriculumCheckpointScheduler:
      - Model + optimiser + scheduler + RNG states
      - Pipeline state
      - Learning dynamics states  (when configured)
      - Evaluation results        (when available on the scheduler)
      - HuggingFace model + tokenizer + optional HF Hub upload
    """

    def __init__(self, curriculum_scheduler: Any, prefix: str = "train") -> None:
        super().__init__()
        self.curriculum_scheduler = curriculum_scheduler
        self.prefix = prefix

    # ── Lightning hooks ───────────────────────────────────────────────────────

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Trigger a checkpoint whenever the scheduler says it's time."""
        if self.curriculum_scheduler.should_checkpoint(trainer.global_step):
            self._do_checkpoint(trainer, pl_module)

    # Keep backward-compat with older Lightning versions that call on_batch_end
    def on_batch_end(self, trainer, pl_module):
        if self.curriculum_scheduler.should_checkpoint(trainer.global_step):
            self._do_checkpoint(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        """Restore a checkpoint at the beginning of training if one exists."""
        ckpt_path: Optional[str] = getattr(trainer, "ckpt_path", None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            return

        sched = self.curriculum_scheduler
        result = load_checkpoint(
            checkpointing_config=sched.configs["checkpointing"],
            checkpoint_step=ckpt_path,
            fabric=getattr(sched, "fabric", None),
            model=getattr(sched, "model", None),
            optimizer=getattr(sched, "optimizer", None),
            lr_scheduler=getattr(sched, "lr_scheduler", None),
            pipeline_state=getattr(sched, "pipeline_state", None),
        )

        if result is not None:
            _, _, _, extra = result
            step = extra.get("_checkpoint_step", 0)
            if hasattr(getattr(sched, "pipeline_state", None), "step_counter"):
                sched.pipeline_state.step_counter = step
            print(
                f"[CurriculumCheckpointCallback] Restored checkpoint from {ckpt_path}"
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _do_checkpoint(self, trainer, pl_module) -> None:
        step  = trainer.global_step
        sched = self.curriculum_scheduler
        cfgs  = sched.configs

        fabric        = getattr(sched, "fabric",        None)
        model         = getattr(sched, "model",         None)
        optimizer     = getattr(sched, "optimizer",     None)
        lr_scheduler  = getattr(sched, "lr_scheduler",  None)
        tokenizer     = getattr(sched, "tokenizer",     None)
        pipeline_state = getattr(sched, "pipeline_state", None)

        # 1. Main checkpoint
        save_checkpoint(
            configs=cfgs,
            checkpoint_step=step,
            fabric=fabric,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tokenizer=tokenizer,
            pipeline_state=pipeline_state,
            upload_logs=False,
        )

        # 2. Learning dynamics (optional)
        ckpt_cfg = cfgs.get("checkpointing", {}) if isinstance(cfgs, dict) else cfgs
        ld_cfg   = (
            ckpt_cfg.get("learning_dynamics", {})
            if isinstance(ckpt_cfg, dict)
            else getattr(ckpt_cfg, "learning_dynamics", {})
        )
        if ld_cfg:
            dataset = getattr(sched, "learning_dynamics_dataset", None)
            if dataset is not None:
                states = compute_learning_dynamics_states(
                    checkpointing_config=ckpt_cfg,
                    checkpoint_step=step,
                    fabric=fabric,
                    model=model,
                    dataset=dataset,
                    compute_gradients=(
                        ckpt_cfg.get("compute_gradients", False)
                        if isinstance(ckpt_cfg, dict)
                        else getattr(ckpt_cfg, "compute_gradients", False)
                    ),
                )
                save_learning_dynamics_states(
                    checkpointing_config=ckpt_cfg,
                    checkpoint_step=step,
                    prefix=self.prefix,
                    fabric=fabric,
                    learning_dynamics_states=states,
                    learning_dynamics_dataset=dataset,
                    tokenizer=tokenizer,
                )

        # 3. Evaluation results (optional)
        eval_results = getattr(sched, "evaluation_results", None)
        if eval_results:
            save_evaluation_results(
                checkpointing_config=ckpt_cfg,
                checkpoint_step=step,
                fabric=fabric,
                evaluation_results=eval_results,
            )

        # 4. Advance pipeline step counter
        if pipeline_state is not None and hasattr(pipeline_state, "step_counter"):
            pipeline_state.step_counter = step

        print(
            f"[CurriculumCheckpointCallback] Checkpoint saved at step {step}"
        )
