"""
src/continual/gaussian_noise.py
───────────────────────────────
Gaussian Noise Injection (GNI) baseline used in Aoyama et al. (2024).

Adds Gaussian noise N(0, σ²) directly to parameter gradients before the
optimizer step.  This is the *only* continual-learning / regularization
baseline from Aoyama et al. that was not yet implemented in BeetleLM; it is
required to make the Paper-A RQ-A5 sweep an apples-to-apples comparison.

Usage (YAML):

    gaussian_noise:
      enabled: true
      sigma: 1.0e-3
      activate_step: 0          # start step (default 0 = whole training)
      deactivate_step: null     # optional turn-off step
      seed: 0
      target: gradients         # 'gradients' (default) or 'parameters'

When target == 'gradients' (Aoyama default), noise is added to .grad
post-backward and pre-optimizer-step.

When target == 'parameters', noise is added directly to parameter tensors
after the optimizer step (the "weight noise" variant from
Welling & Teh 2011).  Both are wired by the trainer through
`apply(model, step)`.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


class GaussianNoiseInjection:
    """Adds N(0, σ²) noise to gradients (or parameters) per training step."""

    def __init__(
        self,
        sigma: float = 1e-3,
        activate_step: int = 0,
        deactivate_step: Optional[int] = None,
        seed: int = 0,
        target: str = "gradients",
    ):
        if target not in ("gradients", "parameters"):
            raise ValueError(f"Invalid target '{target}'.")
        self.sigma = float(sigma)
        self.activate_step = int(activate_step)
        self.deactivate_step = (int(deactivate_step)
                                if deactivate_step is not None else None)
        self.seed = int(seed)
        self.target = target

        self._gen: Optional[torch.Generator] = None
        self._n_steps_applied = 0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self.sigma > 0.0

    def is_active(self, step: int) -> bool:
        if not self.is_ready:
            return False
        if step < self.activate_step:
            return False
        if self.deactivate_step is not None and step >= self.deactivate_step:
            return False
        return True

    def _generator(self, device: torch.device) -> torch.Generator:
        if self._gen is None or self._gen.device != device:
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self.seed)
        return self._gen

    def apply(self, model: nn.Module, step: int) -> None:
        """Inject noise according to `target`.  No-op if not active."""
        if not self.is_active(step):
            return
        raw = _unwrap(model)
        any_param: Optional[torch.Tensor] = None
        for p in raw.parameters():
            if not p.requires_grad:
                continue
            any_param = p
            tensor = p.grad if self.target == "gradients" else p.data
            if tensor is None:
                continue
            gen = self._generator(tensor.device)
            noise = torch.randn(tensor.shape, generator=gen,
                                device=tensor.device, dtype=tensor.dtype)
            tensor.add_(self.sigma * noise)
        if any_param is not None:
            self._n_steps_applied += 1

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics / serialisation
    # ──────────────────────────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, float]:
        return {
            "gni/sigma": self.sigma,
            "gni/n_steps_applied": float(self._n_steps_applied),
            "gni/is_ready": float(self.is_ready),
        }

    def state_dict(self) -> dict:
        return {
            "sigma": self.sigma,
            "activate_step": self.activate_step,
            "deactivate_step": self.deactivate_step,
            "seed": self.seed,
            "target": self.target,
            "n_steps_applied": self._n_steps_applied,
        }

    def load_state_dict(self, state: dict) -> None:
        self.sigma = float(state.get("sigma", self.sigma))
        self.activate_step = int(state.get("activate_step", self.activate_step))
        self.deactivate_step = state.get("deactivate_step", self.deactivate_step)
        self.seed = int(state.get("seed", self.seed))
        self.target = state.get("target", self.target)
        self._n_steps_applied = int(state.get("n_steps_applied",
                                              self._n_steps_applied))


# ──────────────────────────────────────────────────────────────────────────────
# Factory — builds a GaussianNoiseInjection from a YAML config block
# ──────────────────────────────────────────────────────────────────────────────

def build_gaussian_noise(cfg: dict) -> Optional["GaussianNoiseInjection"]:
    """Build a GaussianNoiseInjection from the `gaussian_noise:` YAML block."""
    if not cfg or not cfg.get("enabled", False):
        return None
    return GaussianNoiseInjection(
        sigma=float(cfg.get("sigma", 1e-3)),
        activate_step=int(cfg.get("activate_step", 0)),
        deactivate_step=cfg.get("deactivate_step"),
        seed=int(cfg.get("seed", 0)),
        target=cfg.get("target", "gradients"),
    )
