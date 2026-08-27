# beetlelm/src/bilingual/data/datamodule.py
"""
PyTorch Lightning DataModule for multilingual streaming datasets.

Polyglot API
------------
Pass ``lang_streams={"eng": [...], "fra": [...], ...}`` (preferred).

Legacy bilingual API
--------------------
Pass ``l1_streams=[...]`` and ``l2_streams=[...]`` — fully supported.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, List, Optional


class BilingualDataModule(pl.LightningDataModule):
    """
    DataModule for monolingual, bilingual, or polyglot streaming corpora.

    The name is preserved for import compatibility.
    """

    def __init__(
        self,
        # Polyglot API
        lang_streams: Optional[Dict[str, List]] = None,
        # Legacy bilingual API
        l1_streams: Optional[List] = None,
        l2_streams: Optional[List] = None,
        # Shared
        pipeline_state=None,
        tokenizer=None,
        batch_size: int = 8,
        seq_len:    int = 128,
        return_raw: bool = False,
    ):
        super().__init__()

        # Normalise to lang_streams dict
        if lang_streams is not None:
            self.lang_streams = lang_streams
        else:
            self.lang_streams = {}
            if l1_streams:
                self.lang_streams["l1"] = l1_streams
            if l2_streams:
                self.lang_streams["l2"] = l2_streams

        if not self.lang_streams:
            raise ValueError(
                "BilingualDataModule requires lang_streams or l1_streams/l2_streams."
            )

        self.pipeline_state = pipeline_state
        self.tokenizer      = tokenizer
        self.batch_size     = batch_size
        self.seq_len        = seq_len
        self.return_raw     = return_raw

        # Legacy attribute aliases
        self.l1_streams = self.lang_streams.get("l1", [])
        self.l2_streams = self.lang_streams.get("l2", [])

        self.train_dataset = None

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self, stage=None):
        from .bilingual_dataset import BilingualDataset

        self.train_dataset = BilingualDataset(
            lang_streams=self.lang_streams,
            pipeline_state=self.pipeline_state,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            tokenizer=self.tokenizer,
            return_raw=self.return_raw,
        )

    # ── DataLoader ────────────────────────────────────────────────────────

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,   # IterableDataset batches internally
            num_workers=0,
        )

    # ── Checkpoint support ────────────────────────────────────────────────

    def state_dict(self):
        return self.pipeline_state.state_dict()

    def load_state_dict(self, state):
        self.pipeline_state.load_state_dict(state)
