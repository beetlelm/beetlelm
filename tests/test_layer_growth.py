"""Tests for the layer-growth continual module."""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.continual.layer_growth import LayerGrowthModule, build_layer_growth


class _ResBlock(nn.Module):
    """Simple residual block with named w_2 output projection for identity init."""
    def __init__(self, d):
        super().__init__()
        self.w_1 = nn.Linear(d, d, bias=False)
        self.w_2 = nn.Linear(d, d, bias=False)   # named 'w_2' → zeroed by identity init

    def forward(self, x):
        return x + self.w_2(torch.relu(self.w_1(x)))


class _TinyDecoder(nn.Module):
    """Minimal model with residual .layers for testing growth."""
    def __init__(self, n_layers=4, d=16):
        super().__init__()
        self.model_config = SimpleNamespace(d_model=d)
        self.layers = nn.ModuleList([_ResBlock(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, 8)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.head(x)


def test_build_returns_none_when_disabled():
    assert build_layer_growth({}) is None
    assert build_layer_growth({"enabled": False}) is None


def test_build_creates_module():
    m = build_layer_growth({"enabled": True, "add_n_layers": 2, "init_mode": "random"})
    assert isinstance(m, LayerGrowthModule)
    assert m.add_n_layers == 2


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        LayerGrowthModule(add_n_layers=2, strategy="ligo")


def test_grow_adds_layers():
    model = _TinyDecoder(n_layers=4)
    lg = LayerGrowthModule(add_n_layers=2, init_mode="random")
    assert len(model.layers) == 4
    lg.apply(model)
    assert len(model.layers) == 6


def test_identity_growth_preserves_logits():
    """With identity init, logits before and after growth should be ~identical."""
    torch.manual_seed(42)
    model = _TinyDecoder(n_layers=4, d=16)
    model.eval()
    x = torch.randn(2, 16)
    with torch.no_grad():
        before = model(x)

    lg = LayerGrowthModule(add_n_layers=2, init_mode="identity")
    lg.apply(model)
    model.eval()
    with torch.no_grad():
        after = model(x)
    # New layers with zero output projections pass input unchanged (identity).
    assert torch.allclose(before, after, atol=1e-5), (
        f"max diff = {(before - after).abs().max().item()}"
    )


def test_grow_is_idempotent():
    model = _TinyDecoder(n_layers=4)
    lg = LayerGrowthModule(add_n_layers=2, init_mode="random")
    lg.apply(model)
    lg.apply(model)  # second call should be no-op
    assert len(model.layers) == 6


def test_state_dict_roundtrip():
    lg = LayerGrowthModule(add_n_layers=3, init_mode="clone_last")
    lg._is_applied = True
    lg._n_layers_before = 4
    lg._n_layers_after  = 7
    state = lg.state_dict()
    fresh = LayerGrowthModule(add_n_layers=1)
    fresh.load_state_dict(state)
    assert fresh.add_n_layers == 3
    assert fresh._n_layers_after == 7


def test_diagnostics_keys():
    lg = LayerGrowthModule(add_n_layers=2)
    d = lg.diagnostics()
    assert "layer_growth/is_applied" in d
    assert "layer_growth/n_layers_before" in d
