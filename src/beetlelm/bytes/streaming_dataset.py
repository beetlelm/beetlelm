"""
Streaming tokenizer-free dataset.

The pretokenized-Arrow path (``PretokenizedMultilingualDataset``) assumes
text was already encoded to fixed-length integer chunks by an external
pipeline. For tokenizer-free training we encode raw text to bytes (or
codepoints) on the fly, so we need a streaming counterpart.

``StreamingTokFreeDataset`` reads raw-text HF datasets (one per language),
asks the curriculum which language to sample at each step, encodes the next
document via the tokenizer-free manager, and yields ``{"input_ids",
"attention_mask"}`` tensor dicts of fixed length.

The trainer's ``_setup_data_pipeline`` branches to this dataset when
``tokenizer.type`` is in ``TOKFREE_TOKENIZERS`` and ``data.pretokenized`` is
``false``. Existing BPE/pretokenized paths remain untouched.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


def _infer_text_column(features: Any) -> str:
    """Pick a string column from an HF dataset's features dict."""
    for name, feat in features.items():
        dtype = getattr(feat, "dtype", None)
        if dtype == "string":
            return name
    raise ValueError(
        f"No string column found in features {list(features.keys())}; "
        "specify text_column explicitly in data config."
    )


class StreamingTokFreeDataset(IterableDataset):
    """Stream raw text from HF datasets, encode on-the-fly, yield fixed-length chunks.

    Parameters
    ----------
    streams
        ``{lang_key: hf_iterable_dataset}``. Each entry is expected to be an HF
        dataset (streaming or not) yielding row dicts containing a text column.
    tokenizer
        A tokenizer-free manager (``ByteTokenizerManager`` /
        ``CodepointTokenizerManager``) — anything with ``__call__(text,
        padding, max_length, truncation, return_tensors)``.
    pipeline_state
        Curriculum state (provides ``data_mix.next_source()`` to select the
        active language for each step).
    seq_len
        Output sequence length in tokens (bytes/codepoints).
    text_columns
        Optional ``{lang_key: column_name}`` override. Auto-detected if absent.
    """

    def __init__(
        self,
        streams: Dict[str, Any],
        tokenizer: Any,
        pipeline_state: Any,
        seq_len: int,
        text_columns: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self.streams = streams
        self.tokenizer = tokenizer
        self.pipeline_state = pipeline_state
        self.seq_len = int(seq_len)
        self._text_cols: Dict[str, str] = {}
        for lang_key, ds in streams.items():
            if text_columns and lang_key in text_columns:
                self._text_cols[lang_key] = text_columns[lang_key]
            else:
                self._text_cols[lang_key] = _infer_text_column(ds.features)
        self._iters: Dict[str, Iterator] = {k: iter(v) for k, v in streams.items()}

    def state_dict(self) -> Dict[str, Any]:
        return {}  # streaming HF datasets have no resumable cursor

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None

    def _next_text(self, lang_key: str) -> str:
        col = self._text_cols[lang_key]
        try:
            row = next(self._iters[lang_key])
        except StopIteration:
            self._iters[lang_key] = iter(self.streams[lang_key])
            row = next(self._iters[lang_key])
        return str(row[col])

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            lang_key = self.pipeline_state.data_mix.next_source()
            if lang_key not in self.streams:
                raise KeyError(
                    f"Curriculum requested lang={lang_key!r} but streaming "
                    f"pools only contain {sorted(self.streams)}. Check that "
                    f"the curriculum condition matches data.lang_sources."
                )
            text = self._next_text(lang_key)
            enc = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.seq_len + 1,  # +1 for next-token shift in trainer
                truncation=True,
                return_tensors="pt",
            )
            yield {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }
