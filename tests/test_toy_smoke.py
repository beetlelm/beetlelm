"""
Fast CPU smoke tests for the download-free ``beetlelm.toy`` path.

These exercise the real PicoDecoder forward/backward for a handful of steps on
the in-repo toy corpus and assert the loss is finite, the model learns, runs are
deterministic, and a loadable checkpoint is written. Total runtime is a few
seconds and needs no downloads and no heavy dependencies.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from beetlelm.pico_decoder.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig
from beetlelm.toy import train

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = str(REPO_ROOT / "configs" / "smoke.yaml")


def test_toy_model_forward_computes_finite_loss():
    cfg = PicoDecoderHFConfig(
        n_layers=2, d_model=64, vocab_size=256,
        attention_n_heads=4, attention_n_kv_heads=4,
        max_seq_len=64, activation_hidden_dim=128,
    )
    model = PicoDecoderHF(cfg)
    x = torch.randint(0, 256, (2, 32))
    out = model(input_ids=x, labels=x)
    assert out.loss is not None
    assert math.isfinite(out.loss.item())


def test_smoke_training_runs_and_writes_checkpoint(tmp_path):
    out_dir = tmp_path / "run"
    results = train(SMOKE_CONFIG, overrides={"training": {"output_dir": str(out_dir)}})
    assert math.isfinite(results["final_loss"])
    assert math.isfinite(results["final_val_loss"])
    assert (out_dir / "config.json").exists()
    assert (out_dir / "metrics.json").exists()
    reloaded = PicoDecoderHF.from_pretrained(out_dir)
    assert reloaded.config.vocab_size == 256


def test_training_reduces_loss():
    results = train(SMOKE_CONFIG, overrides={"training": {"max_steps": 60, "seed": 1}})
    # Random loss over the 256-way byte vocab is ~ln(256) = 5.545.
    assert results["final_loss"] < 5.0


def test_determinism_same_seed():
    r1 = train(SMOKE_CONFIG, overrides={"training": {"seed": 3}})
    r2 = train(SMOKE_CONFIG, overrides={"training": {"seed": 3}})
    assert abs(r1["final_loss"] - r2["final_loss"]) < 1e-5


def test_seed_changes_result():
    r1 = train(SMOKE_CONFIG, overrides={"training": {"seed": 10}})
    r2 = train(SMOKE_CONFIG, overrides={"training": {"seed": 11}})
    assert r1["final_loss"] != r2["final_loss"]
