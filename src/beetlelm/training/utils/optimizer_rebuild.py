"""
src/training/utils/optimizer_rebuild.py
─────────────────────────────────────────
Shared helper for rebuilding the optimizer after the model's trainable
parameter set changes at runtime (used by LoRA and layer-growth modules).

Calling ``rebuild_optimizer_after_param_change(trainer)`` will:
1. Build a new optimizer from the current ``requires_grad`` parameter mask.
2. Re-run ``fabric.setup`` on the model + new optimizer.
3. Re-attach the existing LR-scheduler to the new optimizer and preserve its
   step count.

This avoids the need to track which parameters are new vs. old — the
scheduler's step count is preserved so the LR trajectory is not disrupted.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass   # avoid circular import; type-hint BeetleLMTrainer as Any

logger = logging.getLogger(__name__)


def rebuild_optimizer_after_param_change(trainer) -> None:
    """
    Rebuild optimizer and re-run fabric.setup after parameter changes.

    Parameters
    ----------
    trainer : BeetleLMTrainer
        The live trainer instance.  Must have:
        ``trainer.model``, ``trainer.optimizer``,
        ``trainer.lr_scheduler``, ``trainer.fabric``,
        ``trainer.training_cfg``, ``trainer.log``.
    """
    from beetlelm.training.utils.initialization import initialize_optimizer

    # Save LR-scheduler state (step count, last_epoch, etc.)
    sched_state: dict = {}
    if trainer.lr_scheduler is not None:
        try:
            sched_state = trainer.lr_scheduler.state_dict()
        except Exception:
            pass

    # Build a new optimizer from the current requires_grad mask.
    new_opt = initialize_optimizer(
        training_config=trainer.training_cfg, model=trainer.model
    )

    # Re-register with Fabric (wraps for DDP/FSDP and moves to device).
    trainer.model, trainer.optimizer = trainer.fabric.setup(trainer.model, new_opt)

    # Restore LR-scheduler pointing at the new optimizer.
    if trainer.lr_scheduler is not None and sched_state:
        try:
            trainer.lr_scheduler.optimizer = trainer.optimizer
            trainer.lr_scheduler.load_state_dict(sched_state)
        except Exception as exc:
            logger.warning(
                "[optimizer_rebuild] Could not restore LR-scheduler state: %s", exc
            )

    n_trainable = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    trainer.log(
        f"[optimizer_rebuild] New optimizer built | "
        f"trainable params: {n_trainable:,}"
    )
