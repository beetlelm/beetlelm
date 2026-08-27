"""Tests for the LoRA adapter continual module."""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.continual.lora_adapter import LoRAModule, LoRALinear, build_lora


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.out    = nn.Linear(16, 4,  bias=False)

    def forward(self, x):
        return self.out(torch.relu(self.q_proj(x) + self.v_proj(x)))


def test_build_returns_none_when_disabled():
    assert build_lora({}) is None
    assert build_lora({"enabled": False}) is None


def test_build_creates_module():
    m = build_lora({"enabled": True, "rank": 4, "alpha": 8.0, "target_modules": ["q_proj"]})
    assert isinstance(m, LoRAModule)
    assert m.rank == 4


def test_apply_replaces_target_modules():
    net = _TinyNet()
    lora = LoRAModule(rank=4, alpha=8, target_modules=["q_proj", "v_proj"],
                      freeze_backbone=False, use_peft="never")
    lora.apply(net)
    assert isinstance(net.q_proj, LoRALinear)
    assert isinstance(net.v_proj, LoRALinear)
    assert isinstance(net.out, nn.Linear)  # not a target, unchanged


def test_apply_freezes_backbone():
    net = _TinyNet()
    lora = LoRAModule(rank=4, alpha=8, target_modules=["q_proj"],
                      freeze_backbone=True, use_peft="never")
    lora.apply(net)
    trainable_names = [n for n, p in net.named_parameters() if p.requires_grad]
    # Only lora_A and lora_B of q_proj should be trainable.
    assert all("lora_A" in n or "lora_B" in n for n in trainable_names), trainable_names


def test_lora_linear_forward_shape():
    base = nn.Linear(16, 8, bias=False)
    ll = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 16)
    out = ll(x)
    assert out.shape == (3, 8)


def test_apply_idempotent():
    net = _TinyNet()
    lora = LoRAModule(rank=4, alpha=8, target_modules=["q_proj"],
                      freeze_backbone=False, use_peft="never")
    lora.apply(net)
    lora.apply(net)   # second call should be a no-op (warning)
    assert lora._n_replaced == 1


def test_state_dict_roundtrip():
    lora = LoRAModule(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    lora._is_applied = True
    lora._n_replaced = 2
    state = lora.state_dict()
    fresh = LoRAModule(rank=1, alpha=1)
    fresh.load_state_dict(state)
    assert fresh.rank == 8
    assert fresh._is_applied is True
    assert fresh._n_replaced == 2


def test_diagnostics_keys():
    lora = LoRAModule(rank=4, alpha=8, target_modules=["q_proj"])
    d = lora.diagnostics()
    assert "lora/is_applied" in d
    assert "lora/rank" in d
