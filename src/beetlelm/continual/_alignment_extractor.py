"""
src/continual/_alignment_extractor.py
────────────────────────────────────────
Forward-hook based hidden-state extractor for the alignment-loss module.

Attaches to ``model.layers[layer_idx]`` and captures the layer output,
then pools it to (B, D) with the specified strategy.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class HiddenStateExtractor:
    """
    Registers a forward hook on a transformer layer; captures and pools
    the hidden state to (B, d_model) after each forward pass.

    Parameters
    ----------
    pooling : str
        One of ``"mean"``, ``"first"``, ``"last"``.
    """

    def __init__(self, pooling: str = "mean"):
        if pooling not in ("mean", "first", "last"):
            raise ValueError(f"pooling must be 'mean'/'first'/'last', got {pooling!r}")
        self.pooling = pooling
        self._hidden: Optional[torch.Tensor] = None
        self._handle = None

    def attach(self, model: nn.Module, layer_idx: int) -> None:
        raw = getattr(model, "module", model)
        layers = getattr(raw, "layers", None)
        if layers is None or layer_idx >= len(layers):
            raise RuntimeError(
                f"[AlignmentExtractor] model.layers[{layer_idx}] does not exist "
                f"(n_layers={len(layers) if layers is not None else 0})."
            )

        def _hook(_m, _inp, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if self.pooling == "mean":
                self._hidden = h.mean(dim=1)
            elif self.pooling == "first":
                self._hidden = h[:, 0, :]
            else:
                self._hidden = h[:, -1, :]

        self._handle = layers[layer_idx].register_forward_hook(_hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get_and_reset(self) -> Optional[torch.Tensor]:
        """Return the last captured hidden state and reset the buffer."""
        h = self._hidden
        self._hidden = None
        return h
