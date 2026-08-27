"""
src/continual/layer_growth.py
───────────────────────────────
Layer-growth at phase boundaries.

Appends ``add_n_layers`` new transformer blocks to ``model.layers`` at the
L1→L2 phase boundary.  Three init modes:

* ``identity``   — zero-init the output projection and the FFN down-proj so
                   new layers act as identity at initialisation (no logit spike).
* ``clone_last`` — deep-copy the last existing layer.
* ``random``     — standard PyTorch random init.

v1 strategy: ``stack`` only (width growth / LiGO deferred to v2).

YAML:

    layer_growth:
      enabled: false
      strategy: stack
      add_n_layers: 2
      init_mode: identity    # identity | clone_last | random
      freeze_old_layers: false
      trigger: phase_change
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LayerGrowthModule:
    """Manages stack-based layer growth for supported model architectures."""

    def __init__(
        self,
        add_n_layers: int = 2,
        init_mode: str = "identity",
        freeze_old_layers: bool = False,
        strategy: str = "stack",
    ):
        if strategy != "stack":
            raise ValueError(f"[LayerGrowth] Only strategy='stack' is supported in v1; got {strategy!r}.")
        self.add_n_layers     = int(add_n_layers)
        self.init_mode        = init_mode
        self.freeze_old_layers = bool(freeze_old_layers)
        self._is_applied      = False
        self._n_layers_before = 0
        self._n_layers_after  = 0

    @property
    def is_ready(self) -> bool:
        return self._is_applied

    def apply(self, model: nn.Module) -> None:
        """Append new layers to ``model.layers`` in-place."""
        if self._is_applied:
            logger.warning("[LayerGrowth] apply() called twice — skipping.")
            return

        raw = getattr(model, "module", model)
        if not hasattr(raw, "layers"):
            raise RuntimeError(
                "[LayerGrowth] model has no 'layers' attribute — "
                "only PicoDecoder / BeetleMoEDecoder supported in v1."
            )

        self._n_layers_before = len(raw.layers)
        existing_layers = list(raw.layers)

        new_layers = []
        for _ in range(self.add_n_layers):
            # We always start from a structural deep-copy of the last layer so
            # the new block has the same submodule shape (Attention, SwiGLU,
            # RMSNorm, …) as the existing stack — this is what makes the growth
            # portable across architectures (PicoDecoder, BeetleMoE, …) without
            # needing to know each block's constructor signature.
            new_layer = copy.deepcopy(existing_layers[-1])
            if self.init_mode == "clone_last":
                pass  # keep cloned weights as-is
            elif self.init_mode == "identity":
                # Zero the output projections so the residual connection
                # makes the new block an identity function at init.
                _zero_output_projections(new_layer)
            elif self.init_mode == "random":
                # Re-initialise every parameter of the new block from
                # scratch — equivalent to instantiating a fresh block, but
                # without needing the block class's constructor signature.
                _reinitialize_parameters(new_layer)
            else:
                raise ValueError(
                    f"[LayerGrowth] unknown init_mode={self.init_mode!r}; "
                    f"expected one of 'identity', 'clone_last', 'random'."
                )
            new_layers.append(new_layer)

        raw.layers = nn.ModuleList(existing_layers + new_layers)
        self._n_layers_after = len(raw.layers)
        self._is_applied = True

        if self.freeze_old_layers:
            for i, layer in enumerate(raw.layers):
                if i < self._n_layers_before:
                    for p in layer.parameters():
                        p.requires_grad_(False)

        n_params = sum(p.numel() for p in raw.parameters())
        logger.info(
            "[LayerGrowth] Added %d layers (%s → %s total) | init=%s | params=%d",
            self.add_n_layers,
            self._n_layers_before,
            self._n_layers_after,
            self.init_mode,
            n_params,
        )

    # ─── Persistence ──────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "add_n_layers": self.add_n_layers,
            "init_mode": self.init_mode,
            "freeze_old_layers": self.freeze_old_layers,
            "is_applied": self._is_applied,
            "n_layers_before": self._n_layers_before,
            "n_layers_after": self._n_layers_after,
        }

    def load_state_dict(self, state: dict) -> None:
        self.add_n_layers      = int(state.get("add_n_layers", self.add_n_layers))
        self.init_mode         = state.get("init_mode", self.init_mode)
        self.freeze_old_layers = bool(state.get("freeze_old_layers", self.freeze_old_layers))
        self._is_applied       = bool(state.get("is_applied", False))
        self._n_layers_before  = int(state.get("n_layers_before", 0))
        self._n_layers_after   = int(state.get("n_layers_after", 0))

    # ─── Diagnostics ──────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        return {
            "layer_growth/is_applied":     1.0 if self._is_applied else 0.0,
            "layer_growth/n_layers_before": float(self._n_layers_before),
            "layer_growth/n_layers_after":  float(self._n_layers_after),
        }


# ── Helper ────────────────────────────────────────────────────────────────────

def _zero_output_projections(layer: nn.Module) -> None:
    """Zero-init output projections so the new layer starts as identity."""
    # PicoDecoderBlock: attention.o_proj, swiglu.w_2
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            if any(n in name for n in ("o_proj", "w_2", "c_proj")):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def _reinitialize_parameters(layer: nn.Module) -> None:
    """Re-initialise every parameter of a transformer block as if it had just
    been instantiated.

    Strategy: walk the submodule tree and, for each module whose class exposes
    a ``reset_parameters`` method (the standard PyTorch convention for
    ``nn.Linear``, ``nn.Embedding``, ``nn.LayerNorm``, …), call it.  For
    custom modules without ``reset_parameters`` (e.g. RMSNorm with a learned
    weight, RoPE with no learnable weights), fall back to:
        - ``ones_`` for any parameter named ``weight`` whose shape suggests a
          norm-style scale (1-D)
        - Kaiming-uniform for 2-D ``weight`` tensors
        - zeros for biases / 1-D non-scale parameters
    This recovers a true random init for ``init_mode='random'`` portably
    across PicoDecoder, BeetleMoE, BGPT, etc.
    """
    for module in layer.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            try:
                reset()
                continue
            except Exception:
                # Fall through to manual init if reset_parameters raises.
                pass
        # Manual init for parameters owned directly by ``module`` (not by
        # children — the recursion above already visited them).
        for pname, param in module.named_parameters(recurse=False):
            if param is None:
                continue
            if param.dim() == 1:
                # Norm-style scales conventionally start at 1; biases at 0.
                if "weight" in pname:
                    nn.init.ones_(param)
                else:
                    nn.init.zeros_(param)
            elif param.dim() >= 2:
                nn.init.kaiming_uniform_(param, a=5 ** 0.5)
            else:
                nn.init.zeros_(param)


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_layer_growth(cfg: dict) -> Optional["LayerGrowthModule"]:
    """
    Build a LayerGrowthModule from the ``layer_growth:`` YAML block.

    Returns None if absent or ``enabled: false``.
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    return LayerGrowthModule(
        add_n_layers=int(cfg.get("add_n_layers", 2)),
        init_mode=str(cfg.get("init_mode", "identity")),
        freeze_old_layers=bool(cfg.get("freeze_old_layers", False)),
        strategy=str(cfg.get("strategy", "stack")),
    )
