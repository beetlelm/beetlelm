"""
Minimal, download-free data pipeline for ``beetlelm.toy``.

Reads a plain-text corpus from disk, encodes it to one long stream of token ids
with a tokenizer-free manager, and serves fixed-length contiguous windows. No
HuggingFace datasets, no network, no cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch


def resolve_path(train_file: str, config_dir: str) -> Path:
    """Resolve ``train_file`` against the config dir, then the CWD."""
    p = Path(train_file)
    if p.is_absolute() and p.exists():
        return p
    candidates = [Path(config_dir) / p, Path(config_dir).parent / p, Path.cwd() / p, p]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find training corpus '{train_file}'. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


class TokenStream:
    """A corpus encoded to one flat id tensor, sliced into (input, target) windows.

    Language-model targets are the inputs shifted by one, so each window of
    length ``seq_len + 1`` yields an ``input`` and a ``target`` of ``seq_len``.
    """

    def __init__(self, ids: torch.Tensor, seq_len: int):
        self.ids = ids
        self.seq_len = seq_len
        self.n_windows = max(0, (len(ids) - 1) // seq_len)
        if self.n_windows == 0:
            raise ValueError(
                f"Corpus too small: {len(ids)} tokens for seq_len={seq_len}."
            )

    def batch(self, batch_size: int, generator: torch.Generator):
        """Return a random ``(input_ids, target_ids)`` batch."""
        starts = torch.randint(
            0, self.n_windows, (batch_size,), generator=generator
        ) * self.seq_len
        inp = torch.stack([self.ids[s : s + self.seq_len] for s in starts])
        tgt = torch.stack([self.ids[s + 1 : s + 1 + self.seq_len] for s in starts])
        return inp, tgt


def build_streams(text: str, tokenizer, seq_len: int, val_fraction: float):
    """Encode ``text`` and split it into train / val :class:`TokenStream`s."""
    ids: List[int] = tokenizer.encode(text)
    ids_t = torch.tensor(ids, dtype=torch.long)
    n_val = int(len(ids_t) * val_fraction)
    if n_val < seq_len + 1:
        n_val = 0
    train_ids = ids_t[: len(ids_t) - n_val]
    val_ids = ids_t[len(ids_t) - n_val :] if n_val else train_ids
    train = TokenStream(train_ids, seq_len)
    try:
        val = TokenStream(val_ids, seq_len)
    except ValueError:
        val = train
    return train, val


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
