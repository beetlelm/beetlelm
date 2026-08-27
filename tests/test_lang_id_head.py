"""Tests for the Language-ID auxiliary head."""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.continual.lang_id_head import LangIDHead, build_lang_id_head


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
    assert build_lang_id_head({}, nn.Linear(8, 8)) is None
    assert build_lang_id_head({"enabled": False}, nn.Linear(8, 8)) is None


def test_build_attaches_to_model():
    model = _TinyLM()
    head = build_lang_id_head(
        {"enabled": True, "layer": 2, "hidden_dim": 8, "num_languages": 2},
        model,
        lang_keys=["eng", "deu"],
    )
    assert head is not None
    assert hasattr(model, "_lang_id_head")


def test_hook_captures_hidden_after_forward():
    model = _TinyLM()
    head = build_lang_id_head(
        {"enabled": True, "layer": 2, "hidden_dim": 8, "num_languages": 2},
        model,
        lang_keys=["eng", "deu"],
    )
    ids = torch.randint(0, 64, (2, 8))
    model(ids)
    # _hidden should be (B, d_model)
    assert head._hidden is not None
    assert head._hidden.shape == (2, 16)


def test_compute_loss_returns_scalar():
    model = _TinyLM()
    head = build_lang_id_head(
        {"enabled": True, "layer": 1, "hidden_dim": 8, "num_languages": 2},
        model,
        lang_keys=["eng", "deu"],
    )
    ids = torch.randint(0, 64, (2, 8))
    model(ids)
    loss = head.compute_loss(0)
    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_compute_loss_zero_before_forward():
    model = _TinyLM()
    head = build_lang_id_head(
        {"enabled": True, "layer": 1, "hidden_dim": 8, "num_languages": 2},
        model,
        lang_keys=["eng", "deu"],
    )
    # No forward pass yet → should return 0.
    loss = head.compute_loss(0)
    assert loss.item() == 0.0


def test_label_for_phase_mapping():
    model = _TinyLM()
    head = build_lang_id_head(
        {"enabled": True, "layer": 1, "hidden_dim": 8, "num_languages": 2},
        model,
        lang_keys=["eng", "deu"],
    )
    assert head.label_for_phase("eng") == 0
    assert head.label_for_phase("deu") == 1
    assert head.label_for_phase("unknown") == 0   # fallback


def test_state_dict_roundtrip():
    model = _TinyLM()
    head = build_lang_id_head(
        {"enabled": True, "layer": 1, "hidden_dim": 8, "num_languages": 2},
        model,
        lang_keys=["eng", "deu"],
    )
    state = head.state_dict_head()
    assert "mlp" in state

    model2 = _TinyLM()
    head2 = build_lang_id_head(
        {"enabled": True, "layer": 1, "hidden_dim": 8, "num_languages": 2},
        model2,
        lang_keys=["eng", "deu"],
    )
    head2.load_state_dict_head(state)
    # Weights should match.
    for p1, p2 in zip(head.mlp.parameters(), head2.mlp.parameters()):
        assert torch.allclose(p1, p2)


def test_diagnostics_keys():
    model = _TinyLM()
    head = build_lang_id_head(
        {"enabled": True, "layer": 1, "hidden_dim": 8, "num_languages": 2},
        model,
        lang_keys=["eng", "deu"],
    )
    d = head.diagnostics()
    assert "lang_id/head_params" in d
    assert "lang_id/weight" in d
