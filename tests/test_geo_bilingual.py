"""Tests for geo_bilingual geometry and architecture."""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.bilingual.geometry.alignment import procrustes, linear_cka, vec2vec_align


def test_procrustes_recovers_rotation():
    torch.manual_seed(0)
    N, D = 40, 16
    E_a = torch.randn(N, D)
    Q, _ = torch.linalg.qr(torch.randn(D, D))
    R_true = Q
    E_b = E_a @ R_true
    R_hat = procrustes(E_a, E_b)
    err = (E_a @ R_hat - E_b).norm() / (E_b.norm() + 1e-9)
    assert err.item() < 1e-4, f"Procrustes recovery error {err.item()}"


def test_cka_invariant_to_rotation():
    torch.manual_seed(1)
    X = torch.randn(30, 8)
    Q, _ = torch.linalg.qr(torch.randn(8, 8))
    cka = linear_cka(X, X @ Q)
    assert abs(cka.item() - 1.0) < 1e-4


def test_vec2vec_reduces_residual():
    torch.manual_seed(2)
    X = torch.randn(50, 8)
    W_true = torch.randn(8, 12)
    Y = X @ W_true
    W_hat = vec2vec_align(X, Y)
    residual = (X @ W_hat - Y).norm().item()
    assert residual < 0.1, f"vec2vec residual too large: {residual}"


def test_geo_bilingual_registry():
    from beetlelm.training import _arch_imports  # noqa: F401
    from beetlelm.training.registry import MODEL_REGISTRY
    assert "geo_bilingual" in MODEL_REGISTRY


def test_geo_bilingual_forward_no_checkpoints():
    """With no checkpoint paths, GeoBilingualDecoder falls back to random init."""
    from beetlelm.geo_bilingual.geo_bilingual import GeoBilingualDecoder
    cfg = SimpleNamespace(
        n_layers=2, d_model=16, vocab_size=64,
        attention_n_heads=2, attention_n_kv_heads=1,
        max_seq_len=16, batch_size=2, position_emb_theta=10000.0,
        activation_hidden_dim=32, norm_eps=1e-5, dropout=0.0,
        mono_a_checkpoint=None, mono_b_checkpoint=None,
        rotation_path=None, block_init="average",
    )
    model = GeoBilingualDecoder(cfg)
    model.eval()
    ids = torch.randint(0, 64, (2, 8))
    logits, _ = model(ids)
    assert logits.shape == (2, 8, 64)
