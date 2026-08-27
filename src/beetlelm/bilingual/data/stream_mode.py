# beetlelm/src/bilingual/data/stream_mode.py
"""
StreamMode enum for BeetleStream v2.

Controls how the data pipeline selects and sequences training documents:

- STATIC: Fixed language proportions, no quality gating or topic alignment.
  Equivalent to the original PretokenizedMultilingualDataset behaviour.
  Useful for baselines and ablation studies.

- RANDOM: Language selection follows the curriculum schedule, but documents
  are drawn uniformly at random (no topic alignment, no quality filtering).
  Isolates the effect of the exposure schedule from data curation.

- CURRICULUM: Full BeetleStream v2 pipeline with quality-gated sampling,
  per-condition quality tables, and cross-lingual topic alignment.
  This is the default for production training runs.

Usage
-----
    from beetlelm.bilingual.data.stream_mode import StreamMode, create_stream_mode

    mode = StreamMode.CURRICULUM
    mode = create_stream_mode("curriculum")
    mode = create_stream_mode("static")
"""

from __future__ import annotations

import enum
from typing import Union


class StreamMode(enum.Enum):
    """Streaming strategy for the BeetleStream data pipeline."""

    STATIC = "static"
    """Fixed language proportions with no quality gating or topic alignment."""

    RANDOM = "random"
    """Curriculum-driven language selection, random document sampling."""

    CURRICULUM = "curriculum"
    """Full quality-gated, topic-aligned curriculum sampling (default)."""

    def __str__(self) -> str:
        return self.value

    @property
    def uses_quality_filter(self) -> bool:
        """Whether this mode applies per-condition quality filtering."""
        return self == StreamMode.CURRICULUM

    @property
    def uses_topic_alignment(self) -> bool:
        """Whether this mode performs cross-lingual topic alignment."""
        return self == StreamMode.CURRICULUM

    @property
    def uses_curriculum_schedule(self) -> bool:
        """Whether this mode follows a dynamic language exposure schedule."""
        return self in (StreamMode.RANDOM, StreamMode.CURRICULUM)


def create_stream_mode(name: Union[str, StreamMode]) -> StreamMode:
    """Create a StreamMode from a string or pass through an existing enum value.

    Parameters
    ----------
    name : str or StreamMode
        One of "static", "random", "curriculum" (case-insensitive),
        or a StreamMode enum member.

    Returns
    -------
    StreamMode

    Raises
    ------
    ValueError
        If the name is not a recognised stream mode.

    Examples
    --------
    >>> create_stream_mode("curriculum")
    <StreamMode.CURRICULUM: 'curriculum'>
    >>> create_stream_mode(StreamMode.STATIC)
    <StreamMode.STATIC: 'static'>
    """
    if isinstance(name, StreamMode):
        return name

    lookup = {m.value: m for m in StreamMode}
    key = str(name).strip().lower()

    if key not in lookup:
        raise ValueError(
            f"Unknown stream mode {name!r}. "
            f"Available: {[m.value for m in StreamMode]}"
        )
    return lookup[key]
