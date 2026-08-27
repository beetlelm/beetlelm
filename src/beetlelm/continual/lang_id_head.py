"""
src/continual/lang_id_head.py
──────────────────────────────
Language-ID auxiliary head for BeetleLM.

A 2-layer MLP probe attached via forward hook to a chosen transformer layer.
At each training step the hidden-state output of that layer is mean-pooled
to (B, d_model) and used to predict the per-batch dominant language label
with a cross-entropy loss added to the main LM objective.

v1 label provenance: the trainer passes the integer index of the current
dominant language (derived from data_mix.current_phase_label()).
Per-token labelling is deferred to v2.

Usage:

    from beetlelm.continual.lang_id_head import build_lang_id_head

    head = build_lang_id_head(cfg, raw_model, lang_keys=["eng", "deu"])
    # head is attached to raw_model._lang_id_head and registers a forward hook.
    # Inside _compute_loss:
    if head:
        lang_loss = head.compute_loss(phase_label_idx)
        loss = loss + cfg["weight"] * lang_loss
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LangIDHead(nn.Module):
    """
    2-layer MLP language-ID auxiliary head.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the base transformer.
    hidden_dim : int
        MLP hidden size (typically 256).
    num_languages : int
        Number of language classes.
    weight : float
        Loss multiplier (applied in ``compute_loss``).
    lang_keys : list of str
        Language key list in role order; used to map label strings to ints.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_languages: int,
        weight: float = 0.1,
        lang_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.weight = float(weight)
        self.num_languages = num_languages
        self.lang_keys = lang_keys or []
        # label_map: lang_key string → int class index
        self.label_map: Dict[str, int] = {k: i for i, k in enumerate(self.lang_keys)}

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_languages),
        )
        # Populated by the forward hook registered in attach().
        self._hidden: Optional[torch.Tensor] = None
        self._hook_handle = None

    # ─── Attachment ───────────────────────────────────────────────────────

    def attach(self, model: nn.Module, layer_idx: int) -> None:
        """Register a forward hook on ``model.layers[layer_idx]``.

        Also attaches ``model._lang_id_head = self`` so the head's parameters
        are picked up by ``model.parameters()`` and thus included in the
        optimizer.
        """
        layers = getattr(model, "layers", None)
        if layers is None or layer_idx >= len(layers):
            logger.warning(
                "[LangID] Could not attach hook — model.layers[%d] does not exist "
                "(n_layers=%d). Head will be a no-op.",
                layer_idx, len(layers) if layers is not None else 0,
            )
            return

        def _hook(_module, _input, output):
            # Transformer layers return (hidden, cache) or just hidden.
            h = output[0] if isinstance(output, (tuple, list)) else output
            # Mean-pool over sequence dimension → (B, d_model)
            self._hidden = h.mean(dim=1)

        self._hook_handle = layers[layer_idx].register_forward_hook(_hook)
        # Make head a sub-module of the model so its parameters enter the
        # optimizer automatically (called BEFORE initialize_optimizer).
        model._lang_id_head = self
        logger.info(
            "[LangID] Head attached at layer %d | d_model→%d→%d | num_langs=%d",
            layer_idx, self.mlp[0].in_features,
            self.mlp[0].out_features, self.num_languages,
        )

    def detach(self) -> None:
        """Remove the forward hook (does not remove the submodule ref)."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ─── Loss computation ─────────────────────────────────────────────────

    def compute_loss(self, label_idx: int) -> torch.Tensor:
        """
        Compute cross-entropy of the MLP output against ``label_idx``.

        Must be called **after** a forward pass has populated ``_hidden``.
        Returns a zero tensor if ``_hidden`` is not set.

        Parameters
        ----------
        label_idx : int
            Integer class index of the dominant language for this batch.
        """
        if self._hidden is None:
            # No forward pass has occurred yet.
            dev = next(self.parameters()).device
            return torch.tensor(0.0, device=dev)

        logits = self.mlp(self._hidden)               # (B, num_languages)
        B = logits.shape[0]
        labels = torch.full((B,), label_idx, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        self._hidden = None     # consume; reset for next step
        return loss

    def label_for_phase(self, phase_label: str) -> int:
        """Map a phase/language string to an integer class index.

        Falls back to 0 for unknown strings.
        """
        # Direct key match (e.g. "eng").
        if phase_label in self.label_map:
            return self.label_map[phase_label]
        # Try matching substrings (e.g. phase label may include step info).
        for key, idx in self.label_map.items():
            if key in phase_label:
                return idx
        return 0

    def accuracy(self) -> Optional[float]:
        """Return cached last-step accuracy (set by compute_loss if called with
        return_acc=True). v1 always returns None — diagnostics come from W&B."""
        return None

    # ─── Persistence ──────────────────────────────────────────────────────

    def state_dict_head(self) -> dict:
        return {
            "mlp": {k: v.cpu() for k, v in self.mlp.state_dict().items()},
            "weight": self.weight,
            "num_languages": self.num_languages,
            "lang_keys": self.lang_keys,
        }

    def load_state_dict_head(self, state: dict) -> None:
        self.weight = float(state.get("weight", self.weight))
        self.num_languages = int(state.get("num_languages", self.num_languages))
        self.lang_keys = state.get("lang_keys", self.lang_keys)
        self.label_map = {k: i for i, k in enumerate(self.lang_keys)}
        self.mlp.load_state_dict(state["mlp"])

    # ─── Diagnostics ──────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        n_params = sum(p.numel() for p in self.mlp.parameters())
        return {
            "lang_id/head_params": float(n_params),
            "lang_id/weight": float(self.weight),
            "lang_id/num_languages": float(self.num_languages),
        }


# ── Internal helper ─────────────────────────────────────────────────────────

def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


# ── Factory ─────────────────────────────────────────────────────────────────

def build_lang_id_head(
    cfg: dict,
    model: nn.Module,
    lang_keys: Optional[List[str]] = None,
) -> Optional["LangIDHead"]:
    """
    Build a LangIDHead from the ``lang_id:`` block and attach it to ``model``.

    Returns None if the block is absent or ``enabled: false``.

    The head is registered as ``model._lang_id_head`` so its parameters are
    visible to any subsequent ``model.parameters()`` call (important: call this
    BEFORE ``initialize_optimizer``).

    Example YAML:

        lang_id:
          enabled: true
          layer: 6
          hidden_dim: 256
          weight: 0.1
          num_languages: 2
          per_token: true   # v1: flag accepted but sequence-level label used
    """
    if not cfg or not cfg.get("enabled", False):
        return None

    raw = _unwrap(model)
    d_model = int(
        getattr(raw.model_config, "d_model",
                getattr(raw.model_config, "hidden_size", 768))
    )
    num_languages = int(cfg.get("num_languages", 2))
    if lang_keys is not None:
        num_languages = max(num_languages, len(lang_keys))

    head = LangIDHead(
        d_model=d_model,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_languages=num_languages,
        weight=float(cfg.get("weight", 0.1)),
        lang_keys=lang_keys or [],
    )
    layer_idx = int(cfg.get("layer", 6))
    head.attach(raw, layer_idx)
    return head
