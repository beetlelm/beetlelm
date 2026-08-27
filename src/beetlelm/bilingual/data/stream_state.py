"""
beetlelm/src/bilingual/data/stream_state.py

N-language generalisation of DeterministicStreamState.

Backward compatibility
----------------------
The old two-arg constructor ``DeterministicStreamState(n_l1_streams,
n_l2_streams)`` and the ``total_streams`` legacy kwarg both still work.
Internally everything is stored in language-keyed dicts.

Changes from bilingual version
-------------------------------
1. Constructor accepts ``lang_counts: Dict[str, int]`` for N languages.
   Old n_l1_streams / n_l2_streams / total_streams still accepted.
2. get_next_stream_index(lang) routes to the correct per-language counter
   for any language key, not just "l1"/"l2".
3. step() and update_lang_tokens() generalised to arbitrary lang keys.
4. state_dict / load_state_dict are complete and symmetric.
"""

from __future__ import annotations
from typing import Dict, List, Optional


class DeterministicStreamState:
    """
    Tracks per-stream consumption positions for deterministic resume.

    Works for any number of languages; each language's streams are indexed
    independently so round-robin doesn't bleed across languages.
    """

    def __init__(
        self,
        # ── Polyglot API ─────────────────────────────────────────────────
        lang_counts: Optional[Dict[str, int]] = None,
        # ── Legacy bilingual API (still accepted) ─────────────────────────
        n_l1_streams: int = 0,
        n_l2_streams: int = 0,
        *,
        total_streams: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        lang_counts : Dict[str, int], optional
            Maps language key → number of stream sources.
            E.g. {"eng": 2, "fra": 1, "deu": 1}.
            If supplied, takes precedence over the legacy kwargs.
        n_l1_streams, n_l2_streams : int
            Legacy bilingual API.  Stored as lang_counts={"l1": n, "l2": m}.
        total_streams : int, optional
            Legacy monolingual/bilingual helper.  Split evenly between l1/l2
            when n_l1_streams and n_l2_streams are both 0.
        """
        if lang_counts is not None:
            # Polyglot path
            counts = lang_counts
        else:
            # Legacy path
            if total_streams is not None and n_l1_streams == 0 and n_l2_streams == 0:
                n_l1_streams = total_streams // 2
                n_l2_streams = total_streams - n_l1_streams
            counts = {"l1": n_l1_streams, "l2": n_l2_streams}

        assert all(v >= 0 for v in counts.values()), (
            "All stream counts must be non-negative"
        )

        self._lang_counts: Dict[str, int]       = dict(counts)
        self._positions:   Dict[str, List[int]] = {
            lang: [0] * n for lang, n in counts.items()
        }
        self._steps:       Dict[str, int]       = {lang: 0 for lang in counts}
        self._tokens_seen: Dict[str, int]       = {lang: 0 for lang in counts}

    # ── Backward-compatible properties ───────────────────────────────────

    @property
    def n_l1_streams(self) -> int:
        return self._lang_counts.get("l1", 0)

    @property
    def n_l2_streams(self) -> int:
        return self._lang_counts.get("l2", 0)

    @property
    def l1_step(self) -> int:
        return self._steps.get("l1", 0)

    @property
    def l2_step(self) -> int:
        return self._steps.get("l2", 0)

    @property
    def l1_tokens_seen(self) -> int:
        return self._tokens_seen.get("l1", 0)

    @property
    def l2_tokens_seen(self) -> int:
        return self._tokens_seen.get("l2", 0)

    @property
    def l1_positions(self) -> List[int]:
        return self._positions.get("l1", [])

    @property
    def l2_positions(self) -> List[int]:
        return self._positions.get("l2", [])

    # ── Ensure lang is registered ─────────────────────────────────────────

    def _ensure_lang(self, lang: str, n_streams: int = 1):
        """Lazily register a language that wasn't known at construction time."""
        if lang not in self._lang_counts:
            self._lang_counts[lang]  = n_streams
            self._positions[lang]    = [0] * n_streams
            self._steps[lang]        = 0
            self._tokens_seen[lang]  = 0

    # ── Stream index selection ────────────────────────────────────────────

    def get_next_stream_index(self, lang: str) -> int:
        """
        Return the next round-robin index for the given language's stream pool.

        Parameters
        ----------
        lang : str
            Language key, e.g. "l1", "l2", "eng", "fra".
        """
        self._ensure_lang(lang)
        n = self._lang_counts[lang]
        if n == 0:
            raise ValueError(f"No streams registered for language {lang!r}.")
        return self._steps[lang] % n

    # ── State advancement ─────────────────────────────────────────────────

    def step(self, stream_idx: int, lang: str, n_tokens: int = 1):
        """
        Record consumption of n_tokens from stream_idx of the given language.
        """
        self._ensure_lang(lang)
        positions = self._positions[lang]
        if stream_idx < len(positions):
            positions[stream_idx] += n_tokens
        self._steps[lang]       += 1
        self._tokens_seen[lang] += n_tokens

    def update_lang_tokens(self, **lang_token_kwargs: int):
        """
        Additive update to aggregate language token counts.

        Usage (bilingual)::

            state.update_lang_tokens(l1_tokens=512, l2_tokens=0)

        Usage (polyglot)::

            state.update_lang_tokens(eng=512, fra=256, deu=128)

        The legacy ``l1_tokens`` / ``l2_tokens`` kwargs are remapped to the
        "l1" / "l2" language keys automatically.
        """
        # Remap legacy kwarg names
        remap = {"l1_tokens": "l1", "l2_tokens": "l2"}
        for old, new in remap.items():
            if old in lang_token_kwargs:
                lang_token_kwargs[new] = lang_token_kwargs.pop(old)

        for lang, n in lang_token_kwargs.items():
            self._ensure_lang(lang)
            self._tokens_seen[lang] += n

    # ── Checkpoint support ────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "lang_counts":   dict(self._lang_counts),
            "positions":     {k: list(v) for k, v in self._positions.items()},
            "steps":         dict(self._steps),
            "tokens_seen":   dict(self._tokens_seen),
        }

    def load_state_dict(self, state: dict):
        # Support old bilingual checkpoint format
        if "lang_counts" in state:
            self._lang_counts  = dict(state["lang_counts"])
            self._positions    = {k: list(v) for k, v in state["positions"].items()}
            self._steps        = dict(state["steps"])
            self._tokens_seen  = dict(state["tokens_seen"])
        else:
            # Legacy bilingual format
            n_l1 = state.get("n_l1_streams", self._lang_counts.get("l1", 0))
            n_l2 = state.get("n_l2_streams", self._lang_counts.get("l2", 0))
            self._lang_counts  = {"l1": n_l1, "l2": n_l2}
            self._positions    = {
                "l1": list(state.get("l1_positions", [0] * n_l1)),
                "l2": list(state.get("l2_positions", [0] * n_l2)),
            }
            self._steps        = {
                "l1": state.get("l1_step", 0),
                "l2": state.get("l2_step", 0),
            }
            self._tokens_seen  = {
                "l1": state.get("l1_tokens_seen", 0),
                "l2": state.get("l2_tokens_seen", 0),
            }

    def __repr__(self) -> str:
        parts = " | ".join(
            f"{lang}: {self._lang_counts[lang]} streams, step={self._steps[lang]}"
            for lang in self._lang_counts
        )
        return f"DeterministicStreamState({parts})"
