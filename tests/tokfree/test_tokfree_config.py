"""Sanity tests for the tokenizer-free YAML configs.

Asserts: every config under ``configs/tokfree/examples/`` parses, declares a
registered architecture, declares a registered tokenizer-free representation,
and propagates vocab_size correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from beetlelm.bytes import TOKFREE_TOKENIZERS

CONFIGS_ROOT = Path(__file__).resolve().parents[2] / "configs" / "tokfree" / "examples"


@pytest.fixture(scope="module")
def example_configs():
    files = sorted(CONFIGS_ROOT.glob("*.yaml"))
    assert files, f"No example tokfree configs found under {CONFIGS_ROOT}"
    return files


def test_examples_directory_populated(example_configs):
    names = {p.name for p in example_configs}
    assert "train_hnet_bytes_eng_mono.yaml" in names
    assert "train_hnet_bytes_bilingual_eng_nld.yaml" in names
    assert "train_mambabyte_bytes_eng_mono.yaml" in names


def test_every_config_parses(example_configs):
    for f in example_configs:
        with f.open() as fh:
            cfg = yaml.safe_load(fh)
        assert isinstance(cfg, dict), f"{f.name} did not load as a mapping"


def test_every_config_has_tokfree_tokenizer(example_configs):
    for f in example_configs:
        cfg = yaml.safe_load(f.open())
        tok = cfg.get("tokenizer", {})
        tok_type = tok.get("type")
        assert tok_type in TOKFREE_TOKENIZERS, (
            f"{f.name}: tokenizer.type={tok_type!r} not registered "
            f"(expected one of {sorted(TOKFREE_TOKENIZERS)})"
        )


def test_byte_configs_set_vocab_256(example_configs):
    for f in example_configs:
        cfg = yaml.safe_load(f.open())
        if cfg["tokenizer"]["type"] != "byte":
            continue
        assert cfg["tokenizer"].get("vocab_size") == 256, (
            f"{f.name}: byte configs must declare vocab_size=256 in tokenizer block"
        )
        model_vocab = cfg["model"]["params"].get("vocab_size")
        assert model_vocab == 256, (
            f"{f.name}: byte configs must set model.params.vocab_size=256 "
            f"(found {model_vocab})"
        )


def test_byte_configs_use_streaming(example_configs):
    """The byte path requires pretokenized: false (Arrow pretokenization is BPE-only)."""
    for f in example_configs:
        cfg = yaml.safe_load(f.open())
        if cfg["tokenizer"]["type"] not in TOKFREE_TOKENIZERS:
            continue
        assert cfg["data"].get("pretokenized") is False, (
            f"{f.name}: tokenizer-free configs require data.pretokenized: false"
        )


def test_architectures_are_registered(example_configs):
    """Lazy: read MODEL_REGISTRY without triggering CUDA imports.

    Importing beetlelm.training._arch_imports triggers the lazy registration. The
    HNetDecoder / MambaByteDecoder constructors lazy-import CUDA libs, so the
    *registration* itself works on macOS.
    """
    from beetlelm.training._arch_imports import register_model  # noqa: F401
    from beetlelm.training.registry import MODEL_REGISTRY

    for f in example_configs:
        cfg = yaml.safe_load(f.open())
        arch = cfg["model"]["architecture"]
        assert arch in MODEL_REGISTRY, (
            f"{f.name}: architecture {arch!r} not in MODEL_REGISTRY "
            f"({sorted(MODEL_REGISTRY)})"
        )
