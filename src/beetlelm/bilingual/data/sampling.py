# beetlelm/src/bilingual/data/sampling.py
"""
CurriculumSampler — N-language generalisation.

Backward compatible: old code passing l1_streams / l2_streams still works.
New polyglot code passes lang_streams={"eng": [...], "fra": [...], ...}.
"""

from typing import Dict, List, Optional


class CurriculumSampler:
    """
    Determines which language stream to sample from next, respecting the
    curriculum in DataMixConfig, and updates deterministic stream positions.

    Supports any number of languages.
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
    ):
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
                "CurriculumSampler requires lang_streams or l1_streams/l2_streams."
            )

        self.pipeline_state = pipeline_state

        # Per-language round-robin indices
        self._lang_idx: Dict[str, int] = {lang: 0 for lang in self.lang_streams}

        # Legacy aliases
        self.l1_streams = self.lang_streams.get("l1", [])
        self.l2_streams = self.lang_streams.get("l2", [])

    # ── Legacy index properties ───────────────────────────────────────────

    @property
    def l1_idx(self) -> int:
        return self._lang_idx.get("l1", 0)

    @l1_idx.setter
    def l1_idx(self, v: int):
        self._lang_idx["l1"] = v

    @property
    def l2_idx(self) -> int:
        return self._lang_idx.get("l2", 0)

    @l2_idx.setter
    def l2_idx(self, v: int):
        self._lang_idx["l2"] = v

    # ── Main sampling ─────────────────────────────────────────────────────

    def sample(self):
        """
        Sample one item from the next language according to the curriculum.

        Returns
        -------
        The next item from the chosen stream.
        """
        data_mix     = self.pipeline_state.data_mix
        stream_state = self.pipeline_state.stream_state

        # Decide language
        lang = data_mix.next_source()

        # Fallback if this language has no streams
        if lang not in self.lang_streams or not self.lang_streams[lang]:
            fallback = next(
                (k for k, v in self.lang_streams.items() if v and k != lang),
                None,
            )
            if fallback is None:
                raise StopIteration(
                    f"No streams available for {lang!r} and no fallback."
                )
            lang = fallback

        streams    = self.lang_streams[lang]
        idx        = self._lang_idx.get(lang, 0)
        stream_idx = idx % len(streams)
        stream     = streams[stream_idx]

        item = next(stream)

        # Update stream state
        stream_state.step(stream_idx=stream_idx, lang=lang, n_tokens=1)

        # Advance round-robin counter
        self._lang_idx[lang] = idx + 1

        return item

    # ── Checkpoint support ────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {"lang_idx": dict(self._lang_idx)}

    def load_state_dict(self, state: dict):
        if "lang_idx" in state:
            self._lang_idx = dict(state["lang_idx"])
        else:
            # Legacy bilingual format
            self._lang_idx = {
                "l1": state.get("l1_idx", 0),
                "l2": state.get("l2_idx", 0),
            }


