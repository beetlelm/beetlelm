"""Tests for the cross-lingual alignment loss module."""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.continual.alignment_loss import AlignmentLossModule, build_alignment_loss, _linear_cka
from beetlelm.bilingual.geometry.alignment import procrustes, linear_cka, vec2vec_align


# ─── Geometry utilities ────────────────────────────────────────────────────────

def test_procrustes_recovers_rotation():
    """E_b = E_a @ R_true → procrustes should recover R̂ with error < 1e-4."""
    torch.manual_seed(0)
    N, D = 50, 16
    E_a = torch.randn(N, D)
    # Generate a true orthogonal rotation via QR
    Q, _ = torch.linalg.qr(torch.randn(D, D))
    R_true = Q
    E_b = E_a @ R_true
    R_hat = procrustes(E_a, E_b)
    err = (E_a @ R_hat - E_b).norm() / E_b.norm()
    assert err.item() < 1e-4, f"Procrustes error too large: {err.item()}"


def test_cka_identity():
    """CKA(X, X) should be 1."""
    X = torch.randn(20, 8)
    cka = linear_cka(X, X)
    assert abs(cka.item() - 1.0) < 1e-4


def test_cka_invariant_to_orthogonal_rotation():
    """CKA is invariant to orthogonal transformations."""
    torch.manual_seed(1)
    X = torch.randn(20, 8)
    Q, _ = torch.linalg.qr(torch.randn(8, 8))
    cka_before = linear_cka(X, X @ Q)
    assert abs(cka_before.item() - 1.0) < 1e-4


def test_vec2vec_align_shape():
    X = torch.randn(30, 8)
    Y = torch.randn(30, 12)
    W = vec2vec_align(X, Y)
    assert W.shape == (8, 12)


# ─── AlignmentLossModule ──────────────────────────────────────────────────────

class _TinyLM(nn.Module):
    def __init__(self, vocab_size=64, d_model=16, n_layers=4):
        super().__init__()
        self.model_config = SimpleNamespace(vocab_size=vocab_size, d_model=d_model)
        self.embedding_proj = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, *args, **kwargs):
        h = self.embedding_proj(input_ids)
        for layer in self.layers:
            h = torch.relu(layer(h))
        return self.head(h), None


def test_build_returns_none_when_disabled():
    assert build_alignment_loss({}) is None
    assert build_alignment_loss({"enabled": False}) is None


def test_build_creates_module():
    m = build_alignment_loss({
        "enabled": True, "mode": "cka", "layer": 2, "weight": 0.05
    })
    assert isinstance(m, AlignmentLossModule)


def test_cka_loss_zero_identical_inputs():
    """CKA loss = 1 - cka_score. For identical L1/L2 inputs cka_score → 1, loss → 0."""
    torch.manual_seed(2)
    model = _TinyLM(n_layers=4)
    m = AlignmentLossModule(mode="cka", layer=2, weight=1.0)
    m.attach(model)
    ids = torch.randint(0, 64, (4, 16))
    with torch.no_grad():
        loss = m.compute(model, ids, ids)
    assert loss.item() < 1e-4, f"CKA loss for identical inputs should be ~0, got {loss.item()}"


def test_infonce_loss_finite():
    torch.manual_seed(3)
    model = _TinyLM(n_layers=4)
    m = AlignmentLossModule(mode="infonce", layer=2, weight=0.1, temperature=0.07)
    m.attach(model)
    l1 = torch.randint(0, 64, (4, 16))
    l2 = torch.randint(0, 64, (4, 16))
    with torch.no_grad():
        loss = m.compute(model, l1, l2)
    assert torch.isfinite(loss)


def test_state_dict_roundtrip():
    m = AlignmentLossModule(mode="cka", layer=3, weight=0.2)
    state = m.state_dict()
    m2 = AlignmentLossModule()
    m2.load_state_dict(state)
    assert m2.mode == "cka"
    assert m2.layer == 3
    assert m2.weight == 0.2


def test_diagnostics_keys():
    m = AlignmentLossModule()
    d = m.diagnostics()
    assert "alignment/cka_score" in d
    assert "alignment/weight" in d
