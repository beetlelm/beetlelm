"""
src/pre_pretraining/formal_lang.py
──────────────────────────────────
Shuffle-Dyck formal-language generator for stage-1 pre-pretraining.

Ported verbatim (with light cleanup) from ``michahu/pre-pretraining/src/grammar.py``:
``generate_shuff_dyck`` produces a list of integers in ``[0, 2k)`` where
``[0, k)`` are opening brackets and ``[k, 2k)`` are the matching closers.
Symbol ``i + k`` is the close for opener ``i``.

This module also exposes ``ShuffleDyckIterableDataset``, a ``torch.utils.data
.IterableDataset`` that yields ``(seq_len + 1,)`` int64 tensors whose integer
symbols have been offset into the main tokenizer's vocab space — so the
resulting stage-1 checkpoint is drop-in loadable for the natural-language
stage-2 training (no vocab mismatch).
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset


def generate_shuff_dyck(
    k: int,
    max_length: int = 2048,
    p_open: float = 0.5,
    min_depth: int = 1,
    max_depth: int = 8,
) -> list[int]:
    """
    Generate a k-shuffle Dyck sequence, truncated at ``max_length``.

    When max depth is reached we force-close one bracket and continue.

    Args:
        k: Number of distinct bracket pairs.
        max_length: Target maximum length of the output sequence.
        p_open: Probability of opening a new bracket at each step.
        min_depth: Minimum required initial depth of nesting.
        max_depth: Maximum nesting depth allowed.

    Returns:
        List of ints in ``[0, 2k)``.  ``i`` is an opener, ``i + k`` its closer.
    """
    if min_depth < 1:
        raise ValueError("min_depth must be at least 1.")

    sequence: list[int] = []
    counts = [0] * k

    # Seed with minimum depth
    for _ in range(min_depth):
        bracket = random.randint(0, k - 1)
        sequence.append(bracket)
        counts[bracket] += 1

    while len(sequence) < max_length:
        depth = sum(counts)

        if depth == 0:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
            continue

        if depth >= max_depth:
            open_brackets = [i for i, c in enumerate(counts) if c > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1
            continue

        if random.random() < p_open and depth < max_depth:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            open_brackets = [i for i, c in enumerate(counts) if c > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1

    return sequence


class ShuffleDyckIterableDataset(IterableDataset):
    """
    IterableDataset that streams Shuffle-Dyck sequences shaped for causal LM
    training.

    Each sample is a ``torch.int64`` tensor of shape ``(seq_len + 1,)`` — the
    +1 accommodates the standard ``input = x[:-1]`` / ``target = x[1:]``
    shift used inside ``Trainer._compute_loss``.

    Vocab alignment
    ---------------
    The raw Dyck symbols live in ``[0, 2 * num_symbols)``.  We offset them by
    ``vocab_offset`` (default 0) so they fit inside the main tokenizer's
    vocabulary.  For a standard 50k-vocab BPE tokenizer, leaving ``vocab_offset
    = 0`` is safe (Dyck occupies the first ``2k`` IDs, which are normally used
    for tokens like ``<pad>``, ``<s>``, etc. — this is fine because stage 1
    is operating in a disjoint formal-language regime and the embedding table
    is simply re-used as a capacity-shared scratchpad).  Advanced users can
    pass ``vocab_offset`` to shift symbols into a reserved span.
    """

    def __init__(
        self,
        num_symbols: int = 20,
        seq_len: int = 512,
        batch_size: int = 64,
        max_depth: int = 8,
        p_open: float = 0.5,
        vocab_size: int = 50000,
        vocab_offset: int = 0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if 2 * num_symbols + vocab_offset > vocab_size:
            raise ValueError(
                f"Shuffle-Dyck uses {2 * num_symbols} IDs "
                f"starting at offset {vocab_offset}, which would exceed "
                f"vocab_size={vocab_size}."
            )
        self.num_symbols = int(num_symbols)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.max_depth = int(max_depth)
        self.p_open = float(p_open)
        self.vocab_size = int(vocab_size)
        self.vocab_offset = int(vocab_offset)
        self._seed = seed
        if seed is not None:
            random.seed(seed)

    def _one_sequence(self) -> list[int]:
        # Generate and pack until we have at least seq_len + 1 tokens.
        buf: list[int] = []
        while len(buf) < self.seq_len + 1:
            buf.extend(
                generate_shuff_dyck(
                    k=self.num_symbols,
                    max_length=self.seq_len + 1 - len(buf),
                    p_open=self.p_open,
                    min_depth=1,
                    max_depth=self.max_depth,
                )
            )
        return buf[: self.seq_len + 1]

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            # Yield one batch of (batch_size, seq_len + 1) int64 tensors so the
            # output shape matches what the existing trainer consumes from the
            # bilingual dataset.
            ids = torch.empty(
                (self.batch_size, self.seq_len + 1), dtype=torch.long
            )
            for b in range(self.batch_size):
                seq = self._one_sequence()
                ids[b] = torch.tensor(seq, dtype=torch.long) + self.vocab_offset
            yield ids
