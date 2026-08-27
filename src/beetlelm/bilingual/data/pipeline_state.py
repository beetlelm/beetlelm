"""
beetlelm/src/bilingual/data/pipeline_state.py

Single source of truth for training progress.

Polyglot changes
----------------
get_next_batch() now accepts either:

  • lang_streams: Dict[str, List[Iterator]]   ← preferred (polyglot)
  • l1_streams + l2_streams                   ← legacy bilingual (still works)

Both are routed through the same unified code path internally.

Step ownership
--------------
step() is called exclusively by get_next_batch().
The trainer must NOT call pipeline_state.step() separately.
"""

from __future__ import annotations
from typing import Dict, Iterator, List, Optional


class DataPipelineState:
    """
    Owns all mutable training-progress state.

    Attributes
    ----------
    stream_state : DeterministicStreamState
    data_mix : DataMixConfig
    step_counter : int   (read-only outside this class)
    token_counter : int  (read-only outside this class)
    """

    def __init__(self, stream_state, data_mix):
        self.stream_state  = stream_state
        self.data_mix      = data_mix
        self.step_counter  = 0
        self.token_counter = 0
        # Optional hook: a dataset (e.g. PretokenizedMultilingualDataset)
        # may register itself here so its per-language cyclic _indices
        # ride along with checkpoint save/restore. Keeping the hook here
        # avoids needing a separate top-level checkpoint section while
        # still letting the dataset opt in.
        self._dataset_ref = None

    # ── Batch fetching ─────────────────────────────────────────────────────

    def get_next_batch(
        self,
        # Polyglot API
        lang_streams: Optional[Dict[str, List]] = None,
        # Legacy bilingual API
        l1_streams:   Optional[List] = None,
        l2_streams:   Optional[List] = None,
        # Shared
        batch_size: int = 8,
        seq_len:    int = 128,
        tokenizer   = None,
    ):
        """
        Decide language → fetch text → tokenise → advance all state.

        Accepts either:
          ``lang_streams={"eng": [...], "fra": [...]}``  (polyglot)
        or:
          ``l1_streams=[...], l2_streams=[...]``          (legacy bilingual)

        Returns a HuggingFace tokeniser output dict, or None if the chosen
        stream is exhausted with no text collected.
        """
        # ── Normalise to lang_streams dict ────────────────────────────────
        if lang_streams is None:
            # Legacy bilingual: map to standard keys
            lang_streams = {}
            if l1_streams:
                lang_streams["l1"] = l1_streams
            if l2_streams:
                lang_streams["l2"] = l2_streams

        if not lang_streams:
            raise ValueError(
                "get_next_batch() requires either lang_streams or l1_streams/l2_streams."
            )

        # ── Decide language ───────────────────────────────────────────────
        lang_choice = self.data_mix.next_source()

        # Fallback: if the chosen language has no streams, try any other
        if lang_choice not in lang_streams or not lang_streams[lang_choice]:
            fallback = next(
                (k for k, v in lang_streams.items() if v and k != lang_choice),
                None,
            )
            if fallback is None:
                raise ValueError(
                    f"No streams available for language {lang_choice!r} and "
                    f"no fallback found among: {list(lang_streams.keys())}"
                )
            print(
                f"[DataPipelineState] WARNING: no streams for {lang_choice!r}, "
                f"falling back to {fallback!r}."
            )
            lang_choice = fallback

        streams = lang_streams[lang_choice]

        # ── Round-robin within the language's pool ────────────────────────
        stream_idx     = self.stream_state.get_next_stream_index(lang_choice)
        current_stream = streams[stream_idx % len(streams)]

        # ── Collect text ──────────────────────────────────────────────────
        batch_text = []
        for _ in range(batch_size):
            try:
                line = next(current_stream)
                batch_text.append(line)
            except StopIteration:
                if not batch_text:
                    return None
                break

        # ── Tokenise ──────────────────────────────────────────────────────
        tokens = tokenizer(
            batch_text,
            padding="max_length",
            max_length=seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # ── Advance all state (single call site) ──────────────────────────
        n_tokens = tokens["input_ids"].numel()
        self.step(tokens=n_tokens, lang=lang_choice, stream_idx=stream_idx)

        return tokens

    # ── State advancement ──────────────────────────────────────────────────

    def step(
        self,
        tokens:     int,
        lang:       Optional[str] = None,
        stream_idx: Optional[int] = None,
    ):
        """
        Advance all progress counters by one step.

        Called exclusively by get_next_batch().
        Do NOT call this from the trainer loop.
        """
        self.step_counter  += 1
        self.token_counter += tokens

        self.data_mix.step(tokens=tokens)

        if lang is not None and stream_idx is not None:
            self.stream_state.step(
                stream_idx=stream_idx,
                lang=lang,
                n_tokens=tokens,
            )
        elif lang is not None:
            # update aggregate counts only
            self.stream_state.update_lang_tokens(**{lang: tokens})

    # ── Checkpoint support ────────────────────────────────────────────────

    def register_dataset(self, dataset) -> None:
        """Attach a dataset whose state_dict() should be saved alongside ours.

        The dataset must expose state_dict()/load_state_dict() returning a
        plain dict (json/pickle-safe). Used by PretokenizedMultilingualDataset
        to persist its per-language cyclic _indices across resume so multi-seed
        ablations stay deterministic.
        """
        self._dataset_ref = dataset

    def state_dict(self) -> dict:
        out = {
            "stream_state":  self.stream_state.state_dict(),
            "data_mix":      self.data_mix.state_dict(),
            "step_counter":  self.step_counter,
            "token_counter": self.token_counter,
        }
        if self._dataset_ref is not None and hasattr(self._dataset_ref, "state_dict"):
            try:
                out["dataset_state"] = self._dataset_ref.state_dict()
            except Exception:
                # A dataset state-dict failure must not block checkpointing.
                out["dataset_state"] = None
        return out

    def load_state_dict(self, state: dict):
        self.stream_state.load_state_dict(state["stream_state"])
        self.data_mix.load_state_dict(state["data_mix"])
        self.step_counter  = state.get("step_counter",  0)
        self.token_counter = state.get("token_counter", 0)
        ds_state = state.get("dataset_state")
        if (
            ds_state is not None
            and self._dataset_ref is not None
            and hasattr(self._dataset_ref, "load_state_dict")
        ):
            try:
                self._dataset_ref.load_state_dict(ds_state)
            except Exception:
                # Loading a dataset state-dict from a checkpoint that pre-dates
                # this hook (or whose schema has drifted) must not abort resume.
                pass
