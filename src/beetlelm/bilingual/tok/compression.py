# src/bilingual/tok/compression.py
"""
Compression strategies for bilingual/multilingual tokenizer training.

Three modes:
  - **equal**        (default) — each language gets the same token budget.
  - **proportional** — budget proportional to raw corpus size.
  - **weighted**     — budget set by explicit per-language weights.
"""

import random
from typing import Dict, List, Optional


def _estimate_avg_tokens(texts: List[str], sample_size: int = 1000) -> float:
    """Estimate average whitespace-token count from a sample of lines."""
    sample = texts[:sample_size]
    if not sample:
        return 1.0
    return sum(len(t.split()) for t in sample) / len(sample) or 1.0


def _sample_to_budget(
    texts: List[str], target_lines: int
) -> List[str]:
    """Repeat or truncate *texts* to exactly *target_lines* lines."""
    texts = texts.copy()
    random.shuffle(texts)
    repeated: List[str] = []
    while len(repeated) < target_lines:
        repeated.extend(texts)
    return repeated[:target_lines]


# ──────────────────────────────────────────────────────────────────────────────
# Equal compression (original)
# ──────────────────────────────────────────────────────────────────────────────

def equal_compression_sample(
    lang_texts: Dict[str, List[str]], target_tokens: int
) -> Dict[str, List[str]]:
    """
    Downsample or repeat sequences from each language so that each language
    contributes roughly *target_tokens* tokens for tokenizer training.

    Args:
        lang_texts: dict mapping language -> list of raw text lines
        target_tokens: target number of tokens per language

    Returns:
        dict mapping language -> balanced list of lines
    """
    balanced_texts = {}
    for lang, texts in lang_texts.items():
        avg = _estimate_avg_tokens(texts)
        n_lines = max(1, int(target_tokens / avg))
        balanced_texts[lang] = _sample_to_budget(texts, n_lines)
    return balanced_texts


# ──────────────────────────────────────────────────────────────────────────────
# Proportional compression
# ──────────────────────────────────────────────────────────────────────────────

def proportional_compression_sample(
    lang_texts: Dict[str, List[str]],
    total_tokens: int,
) -> Dict[str, List[str]]:
    """
    Each language contributes tokens proportional to its raw corpus size.

    The overall token budget is *total_tokens* (summed across all languages).
    A language whose corpus is 60 % of the combined size receives 60 % of
    the budget.

    Args:
        lang_texts: dict mapping language -> list of raw text lines
        total_tokens: total token budget across all languages

    Returns:
        dict mapping language -> sampled list of lines
    """
    raw_sizes = {
        lang: sum(len(t.split()) for t in texts)
        for lang, texts in lang_texts.items()
    }
    grand_total = sum(raw_sizes.values()) or 1

    balanced_texts = {}
    for lang, texts in lang_texts.items():
        weight = raw_sizes[lang] / grand_total
        budget = max(1, int(total_tokens * weight))
        avg = _estimate_avg_tokens(texts)
        n_lines = max(1, int(budget / avg))
        balanced_texts[lang] = _sample_to_budget(texts, n_lines)
    return balanced_texts


# ──────────────────────────────────────────────────────────────────────────────
# Weighted compression
# ──────────────────────────────────────────────────────────────────────────────

def weighted_compression_sample(
    lang_texts: Dict[str, List[str]],
    lang_weights: Dict[str, float],
    total_tokens: int,
) -> Dict[str, List[str]]:
    """
    Each language contributes tokens according to explicit user-supplied weights.

    Args:
        lang_texts:   dict mapping language -> list of raw text lines
        lang_weights: dict mapping language -> float weight (need not sum to 1;
                      they are normalised internally)
        total_tokens: total token budget across all languages

    Returns:
        dict mapping language -> sampled list of lines
    """
    weight_sum = sum(lang_weights.get(lang, 1.0) for lang in lang_texts) or 1.0

    balanced_texts = {}
    for lang, texts in lang_texts.items():
        w = lang_weights.get(lang, 1.0) / weight_sum
        budget = max(1, int(total_tokens * w))
        avg = _estimate_avg_tokens(texts)
        n_lines = max(1, int(budget / avg))
        balanced_texts[lang] = _sample_to_budget(texts, n_lines)
    return balanced_texts
