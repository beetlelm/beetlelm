"""Tests for the BeetleSSM architecture.

If the real ``mamba_ssm`` CUDA package is installed it is used as-is.
Otherwise a lightweight CPU stub is injected into sys.modules so the shape
and registry tests can run on macOS / CPU-only CI boxes.
"""
from __future__ import annotations
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Inject CPU stub if real mamba_ssm is absent ───────────────────────────────

def _install_mamba_stub():
    """Register a minimal CPU Mamba stub in sys.modules."""

    class _MambaStub(nn.Module):
        """CPU stand-in: identity-like linear projection."""
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                     dt_min=0.001, dt_max=0.1):
            super().__init__()
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            return self.proj(x)

    # Build the package hierarchy that BeetleSSMBlock imports from.
    root      = types.ModuleType("mamba_ssm")
    modules   = types.ModuleType("mamba_ssm.modules")
    simple    = types.ModuleType("mamba_ssm.modules.mamba_simple")
    simple.Mamba = _MambaStub

    root.modules = modules
    modules.mamba_simple = simple

    sys.modules["mamba_ssm"]                       = root
    sys.modules["mamba_ssm.modules"]               = modules
    sys.modules["mamba_ssm.modules.mamba_simple"]  = simple


try:
    import mamba_ssm  # noqa: F401  — real package (CUDA env)
    _USING_STUB = False
except ImportError:
    _install_mamba_stub()
    _USING_STUB = True


from beetlelm.beetle_ssm.beetle_ssm import BeetleSSMDecoder  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_ssm_config(vocab_size=64):
    return SimpleNamespace(
        n_layers=2,
        d_model=32,
        vocab_size=vocab_size,
        d_state=8,
        d_conv=2,
        expand=1,
        dt_min=0.001,
        dt_max=0.1,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_forward_shape():
    cfg = _tiny_ssm_config()
    model = BeetleSSMDecoder(cfg)
    model.eval()
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        logits, cache = model(ids)
    assert logits.shape == (2, 8, 64), f"unexpected logits shape {logits.shape}"
    assert cache is None


def test_registry_registers_beetle_ssm():
    from beetlelm.training import _arch_imports  # noqa: F401
    from beetlelm.training.registry import MODEL_REGISTRY
    assert "beetle_ssm" in MODEL_REGISTRY


def test_working_memory_stubs_are_noop():
    """set_working_memory / update_progress must not raise."""
    model = BeetleSSMDecoder(_tiny_ssm_config())
    model.set_working_memory(None)
    model.update_progress(100, 1000)


def test_stub_vs_real_flag():
    """Sanity: the stub flag is set correctly so we know which path ran."""
    assert isinstance(_USING_STUB, bool)
    if _USING_STUB:
        pytest.skip("Using CPU stub — real mamba_ssm not installed (expected on macOS)")
    else:
        pass   # real package present; all tests already ran above
