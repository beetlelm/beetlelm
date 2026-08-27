"""
src/bilingual/geometry/extract_embeddings.py
──────────────────────────────────────────────
Utilities for extracting embedding matrices from trained checkpoints.
Used by the geo_bilingual supervisor to compute Procrustes alignment.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


def extract_token_embeddings(model: nn.Module) -> torch.Tensor:
    """Return the input embedding matrix (vocab_size, d_model) from model."""
    raw = getattr(model, "module", model)
    emb = getattr(raw, "embedding_proj", None)
    if emb is None:
        raise RuntimeError(
            "[extract_embeddings] model has no 'embedding_proj' attribute."
        )
    return emb.weight.detach().float()


def shared_token_embeddings(
    E_a: torch.Tensor,
    E_b: torch.Tensor,
    shared_token_ids: Optional[torch.Tensor] = None,
) -> tuple:
    """
    Return (E_a_shared, E_b_shared) for tokens present in both vocabularies.

    If ``shared_token_ids`` is None, assumes a shared BPE vocabulary and
    returns both full matrices unchanged (Procrustes operates on all rows).
    """
    if shared_token_ids is None:
        V = min(E_a.shape[0], E_b.shape[0])
        return E_a[:V], E_b[:V]
    ids = shared_token_ids.long()
    return E_a[ids], E_b[ids]
