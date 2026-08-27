"""
src/bilingual/data/parallel_pair_dataset.py
────────────────────────────────────────────
Explicit parallel-pair dataset for cross-lingual alignment (Option A).

Reads ``{input_ids_l1, input_ids_l2, pair_id}`` columns from a
Parquet / Arrow file and yields batches as:
    ``{"l1_input_ids": tensor, "l2_input_ids": tensor}``.

Used when ``alignment_loss.parallel_source: explicit``.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class ParallelPairDataset(IterableDataset):
    """
    Iterable dataset over explicit translation-parallel pairs.

    Parameters
    ----------
    path : str
        Path to a Parquet file (or HF datasets arrow folder) with columns:
        ``input_ids_l1``, ``input_ids_l2``, optionally ``pair_id``.
    seq_len : int
        Target sequence length (sequences are truncated / padded to this).
    pad_token_id : int
        Token used for right-padding.
    """

    def __init__(self, path: str, seq_len: int = 512, pad_token_id: int = 0):
        self.path = path
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self._dataset = None

    def _load(self):
        if self._dataset is not None:
            return
        try:
            from datasets import load_from_disk
            self._dataset = load_from_disk(self.path)
        except Exception:
            try:
                import pandas as pd
                from datasets import Dataset
                df = pd.read_parquet(self.path)
                self._dataset = Dataset.from_pandas(df)
            except Exception as exc:
                raise RuntimeError(
                    f"[ParallelPairDataset] Could not load {self.path}: {exc}"
                ) from exc
        logger.info(
            "[ParallelPairDataset] Loaded %d pairs from %s.", len(self._dataset), self.path
        )

    def _pad_or_truncate(self, ids) -> torch.Tensor:
        t = torch.tensor(ids, dtype=torch.long)
        if t.shape[0] >= self.seq_len:
            return t[:self.seq_len]
        pad = torch.full((self.seq_len - t.shape[0],), self.pad_token_id, dtype=torch.long)
        return torch.cat([t, pad])

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._load()
        ds = self._dataset
        idx = 0
        while True:
            row = ds[idx % len(ds)]
            l1 = self._pad_or_truncate(row["input_ids_l1"])
            l2 = self._pad_or_truncate(row["input_ids_l2"])
            yield {"l1_input_ids": l1, "l2_input_ids": l2}
            idx += 1


class ParallelPairCollator:
    """Collate a list of pair dicts into a batched dict of tensors."""

    def __call__(self, samples):
        l1 = torch.stack([s["l1_input_ids"] for s in samples])
        l2 = torch.stack([s["l2_input_ids"] for s in samples])
        return {"l1_input_ids": l1, "l2_input_ids": l2}
