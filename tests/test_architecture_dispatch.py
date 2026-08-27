"""Tests for the architecture registry and dispatch path.

Covers:
- ``MODEL_REGISTRY`` is populated by ``_arch_imports``.
- ``build_model`` constructs every registered architecture from a
  ``SimpleNamespace`` config.
- The ``pico`` ``input_ids`` clamp wrapper is applied idempotently.
- Unknown architectures raise a clear ``ValueError`` listing known keys.
- The historical ``from beetlelm.bgpt.bgpt import BGPT`` import works (regression
  test for the precondition fix).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.training import _arch_imports  # noqa: F401  (registers builtins)
from beetlelm.training.registry import (
    MODEL_REGISTRY,
    build_model,
    register_model,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────

def _tiny_pico_config(vocab_size: int = 256) -> SimpleNamespace:
    return SimpleNamespace(
        n_layers=2,
        d_model=32,
        vocab_size=vocab_size,
        attention_n_heads=2,
        attention_n_kv_heads=1,
        max_seq_len=16,
        batch_size=2,
        position_emb_theta=10000.0,
        activation_hidden_dim=64,
        norm_eps=1e-5,
        dropout=0.0,
    )


def _tiny_bgpt_config(vocab_size: int = 256) -> SimpleNamespace:
    return SimpleNamespace(
        n_layers=2,
        d_model=32,
        vocab_size=vocab_size,
        attention_n_heads=2,
        max_seq_len=16,
        batch_size=2,
        activation_hidden_dim=64,
        layer_norm_epsilon=1e-5,
    )


# ─── Registry population ───────────────────────────────────────────────────

def test_registry_populated_with_builtins():
    assert "pico" in MODEL_REGISTRY
    assert "bgpt" in MODEL_REGISTRY
    assert MODEL_REGISTRY["pico"]["clamp_input_ids"] is True
    assert MODEL_REGISTRY["bgpt"]["clamp_input_ids"] is False


def test_bgpt_alias_importable():
    """Regression: prior to the fix this import raised ImportError."""
    from beetlelm.bgpt.bgpt import BGPT, GPT2Decoder
    assert BGPT is GPT2Decoder


# ─── Construction ──────────────────────────────────────────────────────────

def test_build_model_pico():
    cfg = _tiny_pico_config()
    model = build_model("pico", cfg)
    assert model.model_config.vocab_size == cfg.vocab_size
    # smoke forward
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _ = model(ids)
    assert logits.shape == (1, 8, cfg.vocab_size)


def test_build_model_bgpt():
    cfg = _tiny_bgpt_config()
    model = build_model("bgpt", cfg)
    assert model.model_config.vocab_size == cfg.vocab_size
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _ = model(ids)
    assert logits.shape == (1, 8, cfg.vocab_size)


# ─── Clamp wrapper ─────────────────────────────────────────────────────────

def test_pico_input_clamp_handles_oob_ids():
    """OOB input_ids should be clamped, not crash the embedding lookup."""
    cfg = _tiny_pico_config(vocab_size=64)
    model = build_model("pico", cfg)
    ids = torch.tensor([[0, 10, 999]])  # 999 is OOB
    logits, _ = model(ids)
    assert logits.shape == (1, 3, cfg.vocab_size)


def test_clamp_wrapper_idempotent():
    """Calling build_model('pico', ...) twice must not stack wrappers."""
    cfg = _tiny_pico_config()
    _ = build_model("pico", cfg)
    _ = build_model("pico", cfg)
    cls = MODEL_REGISTRY["pico"]["cls"]
    assert getattr(cls, "_input_clamp_wrapped", False) is True


# ─── Error path ────────────────────────────────────────────────────────────

def test_unknown_architecture_raises():
    cfg = _tiny_pico_config()
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("does_not_exist", cfg)


# ─── Imperative registration of a custom architecture ─────────────────────

class _StubModel(torch.nn.Module):
    """Minimal model satisfying the registry contract.

    Accepts the config object as the first positional arg (any name is fine —
    the registry passes it positionally).
    """

    def __init__(self, cfg):
        super().__init__()
        self.model_config = cfg
        self.lin = torch.nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        emb = torch.zeros(*input_ids.shape, self.model_config.d_model)
        return self.lin(emb), None


def test_register_model_imperative():
    register_model("_stub_arch", _StubModel, clamp_input_ids=False)
    cfg = SimpleNamespace(d_model=8, vocab_size=16)
    model = build_model("_stub_arch", cfg)
    assert isinstance(model, _StubModel)
    # cleanup so other tests aren't polluted
    MODEL_REGISTRY.pop("_stub_arch", None)
