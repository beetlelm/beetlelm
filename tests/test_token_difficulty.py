"""Tests for the token-level difficulty curriculum scorer."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.bilingual.data.token_difficulty import (
    LengthScorer,
    build_token_difficulty,
)


def _make_doc(length):
    return {"input_ids": list(range(length))}


def test_build_returns_none_when_disabled():
    assert build_token_difficulty({}) is None
    assert build_token_difficulty({"enabled": False}) is None


def test_build_length_scorer():
    s = build_token_difficulty({"enabled": True, "mode": "length"})
    assert isinstance(s, LengthScorer)


def test_length_scorer_score():
    s = LengthScorer()
    assert s.score(_make_doc(10)) == 10.0
    assert s.score(_make_doc(200)) == 200.0


def test_length_easy_first():
    """At progress 0.0 short doc picked more often; bias grows as temperature anneals."""
    s = LengthScorer()
    # Use lengths 5 vs 15 so the score difference (10) is visible at T=4.0
    # (exp(-10/4) ≈ 0.082 ≠ 0), allowing the weight to increase as T drops to 0.5.
    short = _make_doc(5)
    long  = _make_doc(15)
    docs  = [short, long]

    # At progress 0.0, temperature is high (4.0) — distribution closer to uniform
    # but still biased toward short (low score = easy when direction=easy_to_hard).
    weights_start = s.sample_weights(
        docs, progress=0.0,
        temperature_start=4.0, temperature_end=0.5,
        target_progress_end=0.6, direction="easy_to_hard",
    )
    # At progress 1.0, temperature is low (0.5) — distribution sharply biased short.
    weights_end = s.sample_weights(
        docs, progress=1.0,
        temperature_start=4.0, temperature_end=0.5,
        target_progress_end=0.6, direction="easy_to_hard",
    )
    # At high temperature (start) the short doc still gets higher weight.
    assert weights_start[0] > weights_start[1], (
        "short doc should have higher weight (easy_to_hard, progress=0)"
    )
    # At low temperature (end) the bias toward short is stronger.
    assert weights_end[0] > weights_start[0], (
        "short doc weight should increase as temperature anneals"
    )


def test_weights_sum_to_one():
    s = LengthScorer()
    docs = [_make_doc(n) for n in [5, 20, 100]]
    for progress in [0.0, 0.3, 1.0]:
        w = s.sample_weights(docs, progress=progress)
        assert abs(sum(w) - 1.0) < 1e-6


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown mode"):
        build_token_difficulty({"enabled": True, "mode": "unknown"})
