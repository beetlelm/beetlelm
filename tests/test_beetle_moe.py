"""Tests for the BeetleMoE architecture."""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.beetle_moe.beetle_moe import BeetleMoEDecoder


def _tiny_moe_config(vocab_size=64):
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
        moe_num_experts=4,
        moe_top_k=2,
        moe_capacity_factor=1.25,
        moe_router_jitter=0.0,
        moe_load_balance_weight=0.01,
        moe_router_z_loss_weight=0.001,
    )


def test_forward_shape():
    cfg = _tiny_moe_config(vocab_size=64)
    model = BeetleMoEDecoder(cfg)
    model.eval()
    ids = torch.randint(0, 64, (2, 8))
    logits, cache = model(ids)
    assert logits.shape == (2, 8, 64)
    assert cache is None or cache == ()


def test_aux_losses_finite():
    cfg = _tiny_moe_config()
    model = BeetleMoEDecoder(cfg)
    ids = torch.randint(0, 64, (2, 8))
    _ = model(ids)
    assert len(model._last_aux_losses) == 2
    for k, v in model._last_aux_losses.items():
        assert torch.isfinite(v).all(), f"{k} is not finite"


def test_aux_losses_positive():
    cfg = _tiny_moe_config()
    model = BeetleMoEDecoder(cfg)
    ids = torch.randint(0, 64, (2, 8))
    model.train()
    _ = model(ids)
    for k, v in model._last_aux_losses.items():
        assert v.item() >= 0.0, f"{k} should be non-negative"


def test_registry_registers_beetle_moe():
    from beetlelm.training import _arch_imports  # noqa: F401
    from beetlelm.training.registry import MODEL_REGISTRY
    assert "beetle_moe" in MODEL_REGISTRY


def test_model_config_accessible():
    cfg = _tiny_moe_config(vocab_size=128)
    model = BeetleMoEDecoder(cfg)
    assert model.model_config.vocab_size == 128
