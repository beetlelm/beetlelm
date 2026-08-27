"""Unit tests for the tokenizer-free representation layer.

Runs on macOS without ``mamba_ssm`` / ``flash_attn`` / ``causal_conv1d`` — the
byte/codepoint managers are pure-Python.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from beetlelm.bytes import TOKFREE_TOKENIZERS, ByteTokenizerManager, CodepointTokenizerManager


def test_registry_lists_both():
    assert set(TOKFREE_TOKENIZERS) == {"byte", "codepoint"}


def test_byte_round_trip_ascii():
    tok = ByteTokenizerManager(max_length=64, add_bos=False, add_eos=False)
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"
    assert tok.vocab_size == 256
    assert len(tok) == 256


def test_byte_round_trip_multibyte():
    """UTF-8 multibyte (Chinese 中文, German umlaut ä) must round-trip."""
    tok = ByteTokenizerManager(max_length=64, add_bos=False, add_eos=False)
    for text in ["中文测试", "schöne grüße", "🚀 rocket", "العربية"]:
        ids = tok.encode(text)
        # Real bytes are in [0, 254); decode skips control ids (>=254)
        assert tok.decode(ids) == text


def test_byte_call_shapes_and_padding():
    tok = ByteTokenizerManager(max_length=16, add_bos=True, add_eos=False)
    out = tok(["hi", "a longer text here"], padding="max_length", max_length=16)
    assert out["input_ids"].shape == (2, 16)
    assert out["attention_mask"].shape == (2, 16)
    # First row: BOS + 2 chars + padding → 13 zeros at the tail
    assert out["attention_mask"][0].sum().item() == 1 + 2
    # Padded ids are pad_token_id (0 by default)
    assert int(out["input_ids"][0, -1]) == tok.pad_token_id


def test_byte_call_truncation():
    tok = ByteTokenizerManager(max_length=8, add_bos=False, add_eos=False)
    out = tok("the quick brown fox jumps over", padding="max_length", max_length=8, truncation=True)
    assert out["input_ids"].shape == (1, 8)
    assert out["attention_mask"].sum().item() == 8


def test_byte_save_and_load(tmp_path: Path):
    tok = ByteTokenizerManager(max_length=128, add_bos=True, add_eos=True)
    tok.save_pretrained(tmp_path)
    cfg = json.loads((tmp_path / "tokfree_tokenizer.json").read_text())
    assert cfg["type"] == "byte"
    assert cfg["vocab_size"] == 256
    restored = ByteTokenizerManager.from_pretrained(tmp_path)
    assert restored.max_length == 128
    assert restored.add_bos is True
    assert restored.add_eos is True


def test_codepoint_round_trip():
    tok = CodepointTokenizerManager(max_length=64, add_bos=False, add_eos=False)
    for text in ["hello", "中文", "схема", "🌟"]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text


def test_codepoint_vocab_clip():
    """Codepoints above vocab_size must collapse to pad_id, not crash."""
    tok = CodepointTokenizerManager(
        max_length=8, vocab_size=128, add_bos=False, add_eos=False,
    )
    ids = tok.encode("中")  # ord(中)=0x4E2D, well past 128
    assert all(0 <= i < 128 for i in ids)


def test_codepoint_call_returns_tensor_dict():
    tok = CodepointTokenizerManager(max_length=16, add_bos=True)
    out = tok(["abc"], padding="max_length", max_length=16)
    assert isinstance(out["input_ids"], torch.Tensor)
    assert out["input_ids"].dtype == torch.long
    assert out["input_ids"].shape == (1, 16)


def test_call_matches_pipeline_state_signature():
    """The trainer's pipeline_state calls tokenizer(texts, padding='max_length',
    max_length=seq_len, truncation=True, return_tensors='pt'). The manager
    must accept all five kwargs and return the same dict shape as PreTrainedTokenizerFast."""
    tok = ByteTokenizerManager(max_length=32)
    out = tok(
        ["alpha", "beta"],
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt",
    )
    assert set(out.keys()) == {"input_ids", "attention_mask"}
    assert out["input_ids"].dtype == torch.long
