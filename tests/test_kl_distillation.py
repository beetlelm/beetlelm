"""Tests for the KL-distillation continual module.

Verifies:
- ``build_kl_distillation`` returns None when disabled.
- ``kl_term`` returns a zero scalar before snapshot.
- After snapshotting the student onto itself, the KL term is ~0
  (identical distributions ⇒ KL = 0).
- After perturbing the student's parameters, the KL term becomes > 0.
- ``state_dict`` / ``load_state_dict`` round-trip on a fresh module.
- Diagnostics keys are present.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.continual.kl_distillation import (
    KLDistillationModule,
    build_kl_distillation,
)


# ─── Tiny model fixture ────────────────────────────────────────────────────

class _TinyLM(nn.Module):
    """Minimal LM that mimics the trainer's contract: returns ``(logits, _)``
    and exposes ``model_config.vocab_size``."""

    def __init__(self, vocab_size: int = 32, d_model: int = 16):
        super().__init__()
        self.model_config = SimpleNamespace(vocab_size=vocab_size, d_model=d_model)
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, *args, **kwargs):
        h = self.emb(input_ids)
        return self.head(h), None


# ─── Factory ───────────────────────────────────────────────────────────────

def test_build_returns_none_when_disabled():
    assert build_kl_distillation({}) is None
    assert build_kl_distillation({"enabled": False}) is None


def test_build_returns_module_when_enabled():
    m = build_kl_distillation({"enabled": True, "weight": 0.5, "temperature": 3.0})
    assert isinstance(m, KLDistillationModule)
    assert m.weight == 0.5
    assert m.temperature == 3.0


def test_invalid_temperature_raises():
    with pytest.raises(ValueError, match="temperature"):
        KLDistillationModule(temperature=0.0)


# ─── Behaviour ─────────────────────────────────────────────────────────────

def test_kl_term_zero_before_snapshot():
    model = _TinyLM()
    kl = KLDistillationModule(weight=1.0, temperature=2.0, teacher_dtype="float32")
    ids = torch.randint(0, 32, (2, 8))
    term = kl.kl_term(model, ids)
    assert torch.allclose(term, torch.tensor(0.0))


def test_zero_kl_when_identical():
    """After snapshotting the student onto itself, KL(s||t) on the same input
    must be ~0 (the teacher is a frozen copy with identical parameters)."""
    torch.manual_seed(0)
    model = _TinyLM()
    kl = KLDistillationModule(weight=1.0, temperature=2.0, teacher_dtype="float32")
    kl.snapshot(model)
    assert kl.is_ready

    ids = torch.randint(0, 32, (4, 16))
    term = kl.kl_term(model, ids)
    assert term.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_positive_after_student_perturbation():
    torch.manual_seed(0)
    model = _TinyLM()
    kl = KLDistillationModule(weight=1.0, temperature=2.0, teacher_dtype="float32")
    kl.snapshot(model)

    # Drift the student away from its snapshot.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    ids = torch.randint(0, 32, (4, 16))
    term = kl.kl_term(model, ids)
    assert term.item() > 1e-3, f"expected non-trivial KL, got {term.item()}"


def test_kl_term_scales_with_weight():
    torch.manual_seed(1)
    model = _TinyLM()
    ids = torch.randint(0, 32, (2, 8))

    kl1 = KLDistillationModule(weight=1.0, temperature=2.0, teacher_dtype="float32")
    kl1.snapshot(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    t1 = kl1.kl_term(model, ids).item()

    # Same drifted student, fresh module with weight=2.0 — but new snapshot
    # would set teacher = drifted student, giving zero. So we monkey-set the
    # teacher reference directly to compare term scaling.
    kl2 = KLDistillationModule(weight=2.0, temperature=2.0, teacher_dtype="float32")
    kl2._teacher = kl1._teacher
    kl2._is_ready = True
    t2 = kl2.kl_term(model, ids).item()

    assert t2 == pytest.approx(2.0 * t1, rel=1e-4)


# ─── Persistence ───────────────────────────────────────────────────────────

def test_state_dict_roundtrip():
    torch.manual_seed(2)
    model = _TinyLM()
    kl = KLDistillationModule(weight=0.7, temperature=1.5, teacher_dtype="float32")
    kl.snapshot(model)

    state = kl.state_dict()
    assert state["is_ready"] is True
    assert state["teacher_state"] is not None

    fresh_model = _TinyLM()
    kl2 = KLDistillationModule(weight=99.0, temperature=99.0, teacher_dtype="float32")
    kl2.load_state_dict(state, model=fresh_model)

    assert kl2.is_ready
    assert kl2.weight == 0.7
    assert kl2.temperature == 1.5

    # Restored teacher should match original teacher's outputs.
    ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        a, _ = kl._teacher(ids)
        b, _ = kl2._teacher(ids)
    assert torch.allclose(a, b, atol=1e-6)


def test_state_dict_roundtrip_when_not_ready():
    kl = KLDistillationModule()
    state = kl.state_dict()
    assert state["is_ready"] is False
    assert state["teacher_state"] is None

    kl2 = KLDistillationModule()
    kl2.load_state_dict(state)  # no model needed when not ready
    assert not kl2.is_ready


# ─── Diagnostics ───────────────────────────────────────────────────────────

def test_diagnostics_keys():
    kl = KLDistillationModule(weight=1.0, temperature=2.0, teacher_dtype="float32")
    d = kl.diagnostics()
    assert d["kl_distill/is_ready"] == 0.0
    assert "kl_distill/term" in d
    assert "kl_distill/weight" in d
    assert "kl_distill/temperature" in d

    kl.snapshot(_TinyLM())
    d2 = kl.diagnostics()
    assert d2["kl_distill/is_ready"] == 1.0
