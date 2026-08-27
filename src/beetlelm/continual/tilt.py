"""
src/continual/tilt.py
─────────────────────
TILT — Transformer Inter-Layer Transfer (Papadimitriou & Jurafsky, 2020;
transformer-LM implementation from Yadavalli et al., 2023) — for BeetleLM's
bilingual continual-learning experiments.

Recipe
------
1. Train an LM on the L1 normally (no TILT logic active).
2. At the L1→L2 curriculum phase boundary, freeze every transformer block.
   Only the input embedding (`embedding_proj`) and the LM head
   (`de_embedding_proj`) remain trainable.
3. Continue pretraining on L2 with the frozen backbone, forcing the model to
   acquire the new language primarily via embedding learning while reusing the
   L1 grammar stored in the frozen intermediate layers.

TILT contributes **no loss penalty** — it only mutates `requires_grad` on the
model parameters (and optionally re-initialises the input embeddings).

Usage (controlled entirely via YAML — see example at bottom of file):

    from beetlelm.continual.tilt import TILTModule

    tilt = TILTModule(trainable_modules=["embedding_proj", "de_embedding_proj"])
    tilt.apply(model)   # call at L1→L2 phase boundary
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TILTModule:
    """
    Layer-freezing controller for TILT-style continued pretraining.

    Parameters
    ----------
    trainable_modules : list[str]
        Parameter-name prefixes that remain trainable after `apply()`. Every
        other parameter has `requires_grad` set to False. Defaults to the
        PicoDecoder's input embedding and LM head.
    reset_embeddings : bool
        If True, reinitialise `embedding_proj.weight` at `apply()` time using
        the model's own `_init_weights` if available, otherwise N(0, 0.02²) to
        match PicoDecoder's init scheme.
    """

    def __init__(
        self,
        trainable_modules: Optional[List[str]] = None,
        reset_embeddings: bool = False,
    ):
        self.trainable_modules: List[str] = list(
            trainable_modules
            if trainable_modules is not None
            else ["embedding_proj", "de_embedding_proj"]
        )
        self.reset_embeddings = bool(reset_embeddings)

        self._is_ready = False
        self._trainable_params = 0
        self._frozen_params = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once the freeze mask has been applied to a model."""
        return self._is_ready

    def apply(self, model: torch.nn.Module, _resuming: bool = False) -> None:
        """
        Mutate `model` in place: freeze every parameter whose name does not
        start with one of `self.trainable_modules`. Optionally reinitialise the
        input embedding (skipped when `_resuming=True` so that resume does not
        clobber L2 progress).
        """
        raw = _unwrap(model)

        trainable = 0
        frozen = 0
        for name, param in raw.named_parameters():
            keep = any(name.startswith(prefix) for prefix in self.trainable_modules)
            param.requires_grad_(bool(keep))
            n = param.numel()
            if keep:
                trainable += n
            else:
                frozen += n

        if self.reset_embeddings and not _resuming:
            self._reinit_embeddings(raw)

        self._trainable_params = trainable
        self._frozen_params = frozen
        self._is_ready = True

        total = trainable + frozen
        frac = (trainable / total) if total > 0 else 0.0
        logger.info(
            "[TILT] Freeze applied | trainable=%.2fM (%.1f%%) | frozen=%.2fM "
            "| trainable_modules=%s | reset_embeddings=%s",
            trainable / 1e6, frac * 100.0, frozen / 1e6,
            self.trainable_modules, self.reset_embeddings and not _resuming,
        )

    def state_dict(self) -> dict:
        """Serialisable state for checkpointing."""
        return {
            "trainable_modules": list(self.trainable_modules),
            "reset_embeddings": self.reset_embeddings,
            "is_ready": self._is_ready,
            "trainable_params": self._trainable_params,
            "frozen_params": self._frozen_params,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a previously saved state_dict."""
        self.trainable_modules = list(
            state.get("trainable_modules", self.trainable_modules)
        )
        self.reset_embeddings = bool(
            state.get("reset_embeddings", self.reset_embeddings)
        )
        self._is_ready = bool(state.get("is_ready", False))
        self._trainable_params = int(state.get("trainable_params", 0))
        self._frozen_params = int(state.get("frozen_params", 0))
        logger.info(
            "[TILT] State restored from checkpoint (is_ready=%s).", self._is_ready,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        """Return a small dict of scalar statistics useful for W&B logging."""
        if not self._is_ready:
            return {"tilt/is_ready": 0.0}
        total = self._trainable_params + self._frozen_params
        frac = (self._trainable_params / total) if total > 0 else 0.0
        return {
            "tilt/is_ready": 1.0,
            "tilt/trainable_params": float(self._trainable_params),
            "tilt/frozen_params": float(self._frozen_params),
            "tilt/trainable_frac": float(frac),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _reinit_embeddings(self, raw: torch.nn.Module) -> None:
        emb = getattr(raw, "embedding_proj", None)
        if emb is None or not hasattr(emb, "weight"):
            logger.warning(
                "[TILT] reset_embeddings=True but no `embedding_proj` found on "
                "the model; skipping reinitialisation."
            )
            return
        init_fn = getattr(raw, "_init_weights", None)
        if callable(init_fn):
            try:
                init_fn(emb)
                logger.info("[TILT] embedding_proj reinitialised via model._init_weights.")
                return
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[TILT] model._init_weights(emb) failed (%s); falling back "
                    "to N(0, 0.02²).", e,
                )
        nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        logger.info("[TILT] embedding_proj reinitialised via N(0, 0.02²).")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip Lightning/DDP wrappers to reach the raw nn.Module."""
    return getattr(model, "module", model)


# ──────────────────────────────────────────────────────────────────────────────
# Factory — builds a TILTModule from a YAML config block
# ──────────────────────────────────────────────────────────────────────────────

def build_tilt(cfg: dict) -> Optional["TILTModule"]:
    """
    Build a TILTModule from the `tilt:` block of a training YAML.

    Returns None if the block is absent or  enabled: false.

    Example YAML:

        tilt:
          enabled: true
          # 'phase_change' = freeze at the L1→L2 curriculum boundary (recommended)
          # 'manual'       = caller calls tilt.apply(model) directly
          trigger: phase_change
          trainable_modules:
            - embedding_proj
            - de_embedding_proj
          reset_embeddings: false
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    trainable = cfg.get("trainable_modules", None)
    if trainable is not None and not isinstance(trainable, (list, tuple)):
        raise TypeError(
            "tilt.trainable_modules must be a list of parameter-name prefixes "
            f"(got {type(trainable).__name__})."
        )
    return TILTModule(
        trainable_modules=list(trainable) if trainable is not None else None,
        reset_embeddings=bool(cfg.get("reset_embeddings", False)),
    )
