"""
src/continual/alignment_loss.py
─────────────────────────────────
Cross-lingual alignment auxiliary loss (InfoNCE or linear CKA).

Supports two parallel-data sources, gated by ``parallel_source``:
  * ``topic``    — topic-aligned curriculum batches yielding
                   ``{l1_input_ids, l2_input_ids}`` (Option B, default).
  * ``explicit`` — explicit Parquet-based ParallelPairDataset (Option A).

The module attaches a ``HiddenStateExtractor`` forward hook to layer K of
the model.  In ``_compute_loss``, the trainer calls
``alignment_loss_module.compute(model, l1_ids, l2_ids, step)`` when the
batch contains ``l1_input_ids`` / ``l2_input_ids`` keys.

YAML:

    alignment_loss:
      enabled: false
      mode: infonce            # infonce | cka
      layer: 6
      pooling: mean            # mean | first | last
      weight: 0.1
      temperature: 0.07        # infonce only
      parallel_source: topic   # topic | explicit
      parallel_path: null      # required for explicit
      pair_batch_fraction: 0.25
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from beetlelm.continual._alignment_extractor import HiddenStateExtractor

logger = logging.getLogger(__name__)


# ── CKA helper ───────────────────────────────────────────────────────────────

def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute linear CKA between two (B, D) representation matrices.

    Feature-space formula (invariant to orthogonal rotations):
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic_xy = (Y.t() @ X).norm("fro").pow(2)
    hsic_xx = (X.t() @ X).norm("fro")
    hsic_yy = (Y.t() @ Y).norm("fro")
    return hsic_xy / (hsic_xx * hsic_yy + 1e-10)


# ── InfoNCE helper ────────────────────────────────────────────────────────────

def _infonce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE loss between two (B, D) normalised embeddings."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.shape[0]
    logits = torch.cat([
        z1 @ z2.t() / temperature,
        z2 @ z1.t() / temperature,
    ], dim=0)                                          # (2B, B)
    labels = torch.arange(B, device=z1.device).repeat(2)
    return F.cross_entropy(logits, labels)


# ── AlignmentLossModule ───────────────────────────────────────────────────────

class AlignmentLossModule:
    """
    Cross-lingual alignment auxiliary loss.

    Two extractors are used (one per language); ``compute`` must be called
    AFTER a forward pass on each language's input.
    """

    def __init__(
        self,
        mode: str = "infonce",
        layer: int = 6,
        pooling: str = "mean",
        weight: float = 0.1,
        temperature: float = 0.07,
        parallel_source: str = "topic",
        pair_batch_fraction: float = 0.25,
    ):
        self.mode = mode
        self.layer = layer
        self.pooling = pooling
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.parallel_source = parallel_source
        self.pair_batch_fraction = float(pair_batch_fraction)

        self._extractor = HiddenStateExtractor(pooling=pooling)
        self._attached = False
        self._last_score = 0.0

    def attach(self, model: nn.Module) -> None:
        """Attach forward hook to model.layers[layer]."""
        if self._attached:
            return
        self._extractor.attach(model, self.layer)
        self._attached = True
        logger.info(
            "[AlignmentLoss] Attached at layer %d | mode=%s | pooling=%s",
            self.layer, self.mode, self.pooling,
        )

    def detach(self) -> None:
        self._extractor.detach()
        self._attached = False

    def compute(
        self,
        model: nn.Module,
        l1_input_ids: torch.Tensor,
        l2_input_ids: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Run forward passes for both language inputs, extract representations,
        and compute the alignment loss.

        Returns a scalar loss tensor.  Returns 0 if the extractor is not
        attached or inputs are empty.
        """
        raw = getattr(model, "module", model)
        dev = next(raw.parameters()).device

        if not self._attached:
            return torch.tensor(0.0, device=dev)

        V = int(getattr(raw.model_config, "vocab_size", 32000))

        # Forward L1 — captures hidden at layer K via hook.
        x1 = l1_input_ids[:, :-1].clamp(0, V - 1).to(dev)
        with torch.no_grad():
            raw(x1)
        h1 = self._extractor.get_and_reset()

        # Forward L2.
        x2 = l2_input_ids[:, :-1].clamp(0, V - 1).to(dev)
        with torch.no_grad():
            raw(x2)
        h2 = self._extractor.get_and_reset()

        if h1 is None or h2 is None:
            return torch.tensor(0.0, device=dev)

        # Crop to same batch size (may differ if padding differs).
        B = min(h1.shape[0], h2.shape[0])
        h1, h2 = h1[:B], h2[:B]

        if self.mode == "infonce":
            loss = _infonce(h1.float(), h2.float(), self.temperature)
        elif self.mode == "cka":
            cka_score = _linear_cka(h1.float(), h2.float())
            self._last_score = float(cka_score.item())
            loss = 1.0 - cka_score
        else:
            raise ValueError(f"[AlignmentLoss] Unknown mode {self.mode!r}.")

        return self.weight * loss

    # ─── Persistence ──────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "layer": self.layer,
            "pooling": self.pooling,
            "weight": self.weight,
            "temperature": self.temperature,
            "parallel_source": self.parallel_source,
            "pair_batch_fraction": self.pair_batch_fraction,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mode                = state.get("mode", self.mode)
        self.layer               = int(state.get("layer", self.layer))
        self.pooling             = state.get("pooling", self.pooling)
        self.weight              = float(state.get("weight", self.weight))
        self.temperature         = float(state.get("temperature", self.temperature))
        self.parallel_source     = state.get("parallel_source", self.parallel_source)
        self.pair_batch_fraction = float(state.get("pair_batch_fraction", self.pair_batch_fraction))

    # ─── Diagnostics ──────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        return {
            "alignment/mode": 0.0,          # categorical; logged as metadata
            "alignment/cka_score": float(self._last_score),
            "alignment/weight": float(self.weight),
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def build_alignment_loss(cfg: dict) -> Optional["AlignmentLossModule"]:
    """
    Build an AlignmentLossModule from the ``alignment_loss:`` YAML block.

    Returns None if absent or ``enabled: false``.
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    return AlignmentLossModule(
        mode=str(cfg.get("mode", "infonce")),
        layer=int(cfg.get("layer", 6)),
        pooling=str(cfg.get("pooling", "mean")),
        weight=float(cfg.get("weight", 0.1)),
        temperature=float(cfg.get("temperature", 0.07)),
        parallel_source=str(cfg.get("parallel_source", "topic")),
        pair_batch_fraction=float(cfg.get("pair_batch_fraction", 0.25)),
    )
