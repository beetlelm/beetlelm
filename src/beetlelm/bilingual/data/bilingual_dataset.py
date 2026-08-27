# beetlelm/src/bilingual/data/bilingual_dataset.py
"""
IterableDataset that yields tokenised batches for any number of languages.

Polyglot API
------------
Pass ``lang_streams={"eng": [...], "fra": [...], ...}`` (preferred).

Legacy bilingual API
--------------------
Pass ``l1_streams=[...]`` and ``l2_streams=[...]`` (still fully supported).
The two are normalised internally to a lang_streams dict keyed "l1"/"l2".
"""

import torch
from torch.utils.data import IterableDataset

from .pipeline_state import DataPipelineState
from .stream_state import DeterministicStreamState
from .curriculum import DataMixConfig


class BilingualDataset(IterableDataset):
    """
    IterableDataset for multilingual (≥1 language) streaming corpora.

    The name is preserved for import compatibility; this class now handles
    monolingual, bilingual, and polyglot settings transparently.
    """

    def __init__(
        self,
        # Polyglot API
        lang_streams=None,              # Dict[str, List[Iterator]]
        # Legacy bilingual API
        l1_streams=None,
        l2_streams=None,
        # Shared
        pipeline_state=None,
        batch_size: int = 8,
        seq_len: int = 128,
        tokenizer=None,
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
                "BilingualDataset requires lang_streams or l1_streams/l2_streams."
            )

        self.pipeline_state = pipeline_state
        self.batch_size     = batch_size
        self.seq_len        = seq_len
        self.tokenizer      = tokenizer
        self.return_raw     = return_raw

        # Legacy attribute aliases so code that reads .l1_streams / .l2_streams still works
        self.l1_streams = self.lang_streams.get("l1", [])
        self.l2_streams = self.lang_streams.get("l2", [])

    def __iter__(self):
        while True:
            batch = self.pipeline_state.get_next_batch(
                lang_streams=self.lang_streams,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                tokenizer=self.tokenizer,
            )
            if batch is None:
                break
            yield batch


# ─────────────────────────────────────────────────────────────────────────────
# Convenience initialiser
# ─────────────────────────────────────────────────────────────────────────────

def initialize_dataset(
    # Polyglot API
    lang_streams=None,                  # Dict[str, List]
    # Legacy bilingual API
    l1_streams=None,
    l2_streams=None,
    # Shared
    tokenizer=None,
    batch_size: int = 32,
    seq_len:    int = 128,
    total_steps: int = 1_000_000,
):
    """
    Returns a fully initialised BilingualDataModule for training.

    Accepts either ``lang_streams`` (polyglot) or ``l1_streams`` /
    ``l2_streams`` (legacy bilingual).
    """
    from .datamodule import BilingualDataModule
    from .condition_presets import get_condition

    # Normalise
    if lang_streams is None:
        lang_streams = {}
        if l1_streams:
            lang_streams["l1"] = l1_streams
        if l2_streams:
            lang_streams["l2"] = l2_streams

    lang_keys = list(lang_streams.keys())

    # Stream counts per language
    lang_counts = {k: len(v) for k, v in lang_streams.items()}
    stream_state = DeterministicStreamState(lang_counts=lang_counts)

    # Pick curriculum
    if len(lang_keys) <= 2:
        data_mix = get_condition("B1", total_steps=total_steps)
    elif len(lang_keys) == 3:
        data_mix = get_condition("T1", total_steps=total_steps, lang_keys=lang_keys)
    else:
        from .curriculum import ConstantExposure
        w = 1.0 / len(lang_keys)
        data_mix = DataMixConfig(
            schedule=ConstantExposure({k: w for k in lang_keys}),
            mode="steps", total_steps=total_steps,
            curriculum_type="polyglot_equal",
        )

    pipeline_state = DataPipelineState(stream_state, data_mix)

    # Wrap all streams as iterators
    lang_iters = {k: [iter(s) for s in streams] for k, streams in lang_streams.items()}

    dm = BilingualDataModule(
        lang_streams=lang_iters,
        pipeline_state=pipeline_state,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    dm.setup()
    return dm
