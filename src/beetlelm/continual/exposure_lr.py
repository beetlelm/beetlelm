"""
src/continual/exposure_lr.py

Exposure-Based Learning Rate Scheduler for continual / curriculum language learning.

Motivation
----------
In standard training, LR is a function of *step count*.  For human-scale multilingual
pretraining where each new language is introduced at a curriculum phase boundary, it is
more principled to tie LR to *token exposure* — the amount of data the model has seen —
and to explicitly boost plasticity at each new-language introduction point.

The key ideas:
  1. LR is computed from tokens_seen rather than step count.
  2. At each phase transition (L2 / L3 introduction), LR is multiplied by
     ``plasticity_boost`` and then decays from that boosted value over the
     remaining token budget of the current phase.
  3. This creates a curriculum-aligned sawtooth curve: high plasticity at the
     onset of each new language, lower plasticity as the model stabilises.

YAML configuration (under optimization:)
-----------------------------------------
    optimization:
      lr: 5.0e-4
      lr_scheduler: "exposure_based"
      exposure_lr:
        warmup_tokens: 5_000_000      # linear warmup over first 5M tokens
        total_tokens: 100_000_000     # full training token budget
        plasticity_boost: 2.0         # LR multiplier at new-language introduction
        min_lr_frac: 0.1              # floor = base_lr * min_lr_frac
        decay_shape: "cosine"         # "cosine" | "linear"
        # Optional: per-phase LR reset factors (override plasticity_boost)
        phase_lr_resets:
          l2_intro: 0.5               # reset to 50% of base_lr at L2 introduction
          l3_intro: 0.3

Integration with beetle_trainer.py
------------------------------------
beetle_trainer._on_phase_change() calls notify_phase_change().
The training loop calls step() instead of lr_scheduler.step().
ExposureLRScheduler.state_dict() / load_state_dict() are used for checkpointing.

Factory
-------
    from beetlelm.continual.exposure_lr import build_exposure_lr
    scheduler = build_exposure_lr(cfg=opt_cfg, optimizer=optimizer, training_config=training_config)
    # Returns None if lr_scheduler != "exposure_based" or exposure_lr block is absent.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, Optional

import torch
import torch.optim as optim

logger = logging.getLogger(__name__)


class ExposureLRScheduler:
    """Token-exposure-based LR scheduler with per-phase plasticity boosts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    base_lr : float
        Peak learning rate (should match optimizer's initial LR).
    warmup_tokens : int
        Number of tokens over which LR is linearly warmed up from 0 → base_lr.
    total_tokens : int
        Total training token budget for the decay schedule.
    plasticity_boost : float
        Multiplier applied to current LR at each new-language phase boundary.
        E.g. 2.0 doubles the LR when L2/L3 is introduced.
        Default 1.0 = no boost.
    min_lr_frac : float
        LR floor as a fraction of base_lr.  Prevents LR from dropping to zero.
    decay_shape : str
        "cosine" or "linear" for the post-warmup decay curve.
    phase_lr_resets : dict, optional
        {phase_label: lr_factor} — if a phase label matches, LR is reset to
        base_lr * lr_factor instead of applying plasticity_boost.
        Useful for forcing specific LR values at known phase boundaries.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_lr: float,
        warmup_tokens: int,
        total_tokens: int,
        plasticity_boost: float = 1.0,
        min_lr_frac: float = 0.0,
        decay_shape: str = "cosine",
        phase_lr_resets: Optional[Dict[str, float]] = None,
    ) -> None:
        self.optimizer        = optimizer
        self.base_lr          = base_lr
        self.warmup_tokens    = warmup_tokens
        self.total_tokens     = total_tokens
        self.plasticity_boost = plasticity_boost
        self.min_lr_frac      = min_lr_frac
        self.decay_shape      = decay_shape
        self.phase_lr_resets  = phase_lr_resets or {}

        # Internal state
        self._tokens_seen:   int   = 0
        self._current_phase: str   = "initial"
        # When a phase starts, we record the base from which decay restarts
        self._phase_start_tokens: int   = 0
        self._phase_start_lr:     float = base_lr
        # Track how many optimizer steps have been taken (for LambdaLR compat)
        self._step_count: int = 0

        # Wrap in a LambdaLR so that external code can call .get_last_lr()
        self._lambda_lr = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: self._current_factor()
        )

    # ── LR computation ────────────────────────────────────────────────────

    def _current_factor(self) -> float:
        """Compute LR multiplier relative to optimizer's initial LR (base_lr)."""
        t = self._tokens_seen
        if t < self.warmup_tokens:
            factor = t / max(1, self.warmup_tokens)
        else:
            # Decay from phase_start_lr over remaining tokens in total budget
            elapsed_in_phase = t - self._phase_start_tokens
            phase_budget      = max(1, self.total_tokens - self._phase_start_tokens)
            progress          = min(elapsed_in_phase / phase_budget, 1.0)

            peak_factor = self._phase_start_lr / self.base_lr

            if self.decay_shape == "cosine":
                decayed = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:  # linear
                decayed = 1.0 - progress

            floor_factor = self.min_lr_frac
            factor = peak_factor * (decayed * (1.0 - floor_factor) + floor_factor)

        return max(self.min_lr_frac, factor)

    def _apply_lr(self) -> None:
        """Set optimizer LR groups to current value."""
        factor = self._current_factor()
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * factor

    # ── Public API ────────────────────────────────────────────────────────

    def get_lr_factor(self, tokens_seen: int, current_phase: str) -> float:
        """Return the LR scaling factor for the given exposure and phase.
        Does not modify internal state — useful for logging / plotting.
        """
        saved = (self._tokens_seen, self._current_phase)
        self._tokens_seen   = tokens_seen
        self._current_phase = current_phase
        factor = self._current_factor()
        self._tokens_seen, self._current_phase = saved
        return factor

    def notify_phase_change(self, new_phase: str, tokens_seen: int) -> None:
        """Called from beetle_trainer._on_phase_change() at language introduction.

        Applies plasticity boost (or per-phase reset) and records the new
        phase start so the decay curve restarts from the boosted LR.
        """
        self._tokens_seen   = tokens_seen
        self._current_phase = new_phase

        if new_phase in self.phase_lr_resets:
            # Explicit per-phase reset overrides plasticity_boost
            reset_factor = self.phase_lr_resets[new_phase]
            new_lr = self.base_lr * reset_factor
            logger.info(
                "[ExposureLR] Phase '%s' at %d tokens → LR reset to %.2e (factor %.2f)",
                new_phase, tokens_seen, new_lr, reset_factor,
            )
        else:
            # Apply plasticity boost to *current* LR
            current_lr  = self.base_lr * self._current_factor()
            new_lr      = min(current_lr * self.plasticity_boost, self.base_lr)
            logger.info(
                "[ExposureLR] Phase '%s' at %d tokens → LR boosted %.2e → %.2e (×%.1f)",
                new_phase, tokens_seen, current_lr, new_lr, self.plasticity_boost,
            )

        # Record phase start so decay restarts from new_lr
        self._phase_start_tokens = tokens_seen
        self._phase_start_lr     = new_lr
        self._apply_lr()

    def step(self, tokens_seen: int, current_phase: str) -> None:
        """Update the optimizer LR based on current token exposure.

        Call once per gradient accumulation cycle from the training loop,
        replacing the standard ``lr_scheduler.step()`` call.
        """
        self._tokens_seen   = tokens_seen
        self._current_phase = current_phase
        self._apply_lr()
        # Advance the underlying LambdaLR so get_last_lr() is accurate
        self._lambda_lr.step()
        self._step_count += 1

    def get_last_lr(self):
        """Compatibility with torch LR scheduler interface."""
        return [g["lr"] for g in self.optimizer.param_groups]

    # ── Checkpointing ─────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        return {
            "tokens_seen":        self._tokens_seen,
            "current_phase":      self._current_phase,
            "phase_start_tokens": self._phase_start_tokens,
            "phase_start_lr":     self._phase_start_lr,
            "step_count":         self._step_count,
            # Hyperparams (for reference / sanity checks on resume)
            "base_lr":            self.base_lr,
            "warmup_tokens":      self.warmup_tokens,
            "total_tokens":       self.total_tokens,
            "plasticity_boost":   self.plasticity_boost,
            "min_lr_frac":        self.min_lr_frac,
            "decay_shape":        self.decay_shape,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._tokens_seen        = state["tokens_seen"]
        self._current_phase      = state["current_phase"]
        self._phase_start_tokens = state["phase_start_tokens"]
        self._phase_start_lr     = state["phase_start_lr"]
        self._step_count         = state.get("step_count", 0)
        self._apply_lr()
        logger.info(
            "[ExposureLR] Restored: phase='%s', tokens=%d, phase_start_lr=%.2e",
            self._current_phase, self._tokens_seen, self._phase_start_lr,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def build_exposure_lr(
    cfg: dict,
    optimizer: optim.Optimizer,
    training_config,
) -> Optional[ExposureLRScheduler]:
    """Factory function mirroring build_ewc / build_maml patterns.

    Returns None if lr_scheduler != "exposure_based" or the exposure_lr
    sub-block is absent — preserving the existing linear_with_warmup path.

    Parameters
    ----------
    cfg : dict
        The optimization sub-config (opt_cfg from initialize_lr_scheduler).
    optimizer : torch.optim.Optimizer
    training_config : dict or TrainingConfig dataclass
    """
    lr_scheduler_type = (
        cfg.get("lr_scheduler")
        if isinstance(cfg, dict)
        else getattr(cfg, "lr_scheduler", None)
    )
    if lr_scheduler_type != "exposure_based":
        return None

    exposure_cfg = cfg.get("exposure_lr", {}) if isinstance(cfg, dict) else {}
    if not exposure_cfg:
        logger.warning(
            "[ExposureLR] lr_scheduler=exposure_based but no 'exposure_lr' block found "
            "— falling back to no scheduling beyond constant LR."
        )

    base_lr = (
        cfg.get("lr", 5e-4)
        if isinstance(cfg, dict)
        else getattr(cfg, "lr", 5e-4)
    )

    if isinstance(training_config, dict):
        total_tokens = training_config.get("total_tokens", exposure_cfg.get("total_tokens", 100_000_000))
    else:
        total_tokens = exposure_cfg.get("total_tokens", 100_000_000)

    return ExposureLRScheduler(
        optimizer        = optimizer,
        base_lr          = base_lr,
        warmup_tokens    = exposure_cfg.get("warmup_tokens", 5_000_000),
        total_tokens     = total_tokens,
        plasticity_boost = exposure_cfg.get("plasticity_boost", 1.0),
        min_lr_frac      = exposure_cfg.get("min_lr_frac", 0.0),
        decay_shape      = exposure_cfg.get("decay_shape", "cosine"),
        phase_lr_resets  = exposure_cfg.get("phase_lr_resets"),
    )
