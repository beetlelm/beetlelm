# beetlelm/src/bilingual/data/topic_aligner.py
"""
Cross-lingual topic alignment for BeetleStream v2.

Ensures that training batches draw documents about the *same topic* across
multiple languages, producing aligned multilingual mini-batches. This
encourages the model to learn cross-lingual semantic correspondences rather
than treating each language as an independent distribution.

Design
------
Topic IDs come from beetle-data's Stage D BERTopic clustering. The same
topic_id (e.g. topic 42 = "animals/pets") maps to semantically related
documents across languages, even though the documents are not exact
translations.

The TopicAligner works with an IndexedShardReader and produces batches
where every document shares the same topic_id. If a language lacks
sufficient documents for the current topic, it falls back to any available
topic to avoid starving the pipeline.

Usage
-----
    from beetlelm.bilingual.data.indexed_shard_reader import IndexedShardReader
    from beetlelm.bilingual.data.topic_aligner import TopicAligner

    reader = IndexedShardReader("./beetle-data/stage_d/")
    aligner = TopicAligner(reader, n_topics=100)

    batch = aligner.aligned_batch(
        lang_schedule={"eng": 4, "nld": 4},
        quality_ranges={"eng": (2.0, 5.0), "nld": (3.0, 5.0)},
        batch_size=8,
    )
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from beetlelm.bilingual.data.indexed_shard_reader import IndexedShardReader

logger = logging.getLogger(__name__)


class TopicAligner:
    """Cross-lingual topic-aligned batch sampler.

    Produces mini-batches where all documents share the same topic_id,
    drawn from potentially different languages according to the provided
    language schedule and quality ranges.

    Parameters
    ----------
    shard_reader : IndexedShardReader
        Pre-initialised shard reader with manifest loaded.
    n_topics : int
        Number of topic clusters to rotate through. The aligner cycles
        through available topic IDs in round-robin order to ensure
        balanced topic coverage over training.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        shard_reader: IndexedShardReader,
        n_topics: int = 100,
        seed: int = 0,
    ) -> None:
        self._reader = shard_reader
        self._n_topics = n_topics
        self._rng = random.Random(seed)

        # Build the shared topic inventory: topics that exist in at least
        # one language (ideally in all languages for best alignment)
        self._shared_topics = self._build_shared_topics()
        self._topic_cursor: int = 0  # round-robin pointer

        if not self._shared_topics:
            logger.warning(
                "TopicAligner: no shared topics found across languages. "
                "Topic alignment will fall back to random sampling."
            )

    def _build_shared_topics(self) -> List[int]:
        """Identify topics that appear across the most languages.

        Returns a sorted list of topic_ids, prioritised by how many
        languages contain documents for that topic (descending), then
        by total document count.
        """
        languages = self._reader.get_languages()
        if not languages:
            return []

        # Count how many languages have each topic
        topic_lang_count: Dict[int, int] = {}
        topic_total_docs: Dict[int, int] = {}

        for lang in languages:
            for topic_id in self._reader.get_topics(lang):
                topic_lang_count[topic_id] = topic_lang_count.get(topic_id, 0) + 1
                topic_total_docs[topic_id] = (
                    topic_total_docs.get(topic_id, 0)
                    + self._reader.get_topic_doc_count(lang, topic_id)
                )

        # Sort by (n_languages DESC, n_docs DESC)
        sorted_topics = sorted(
            topic_lang_count.keys(),
            key=lambda t: (topic_lang_count[t], topic_total_docs.get(t, 0)),
            reverse=True,
        )

        # Keep up to n_topics
        return sorted_topics[: self._n_topics]

    def _next_topic(self) -> Optional[int]:
        """Advance the round-robin topic cursor and return the next topic_id."""
        if not self._shared_topics:
            return None
        topic_id = self._shared_topics[self._topic_cursor % len(self._shared_topics)]
        self._topic_cursor += 1
        return topic_id

    def _sample_for_lang(
        self,
        lang: str,
        topic_id: Optional[int],
        min_quality: float,
        max_quality: float,
    ) -> Optional[Dict[str, Any]]:
        """Sample one document for a language, with topic fallback.

        Parameters
        ----------
        lang : str
            Language code.
        topic_id : int or None
            Preferred topic. If sampling fails, falls back to any topic.
        min_quality, max_quality : float
            Quality bounds.

        Returns
        -------
        dict or None
            Document dict from the shard reader.
        """
        if min_quality == 0 and max_quality == 0:
            # N/A sentinel from quality tables -- skip this language
            return None

        # Try the preferred topic first
        if topic_id is not None:
            doc = self._reader.sample_doc(
                lang=lang,
                min_quality=min_quality,
                max_quality=max_quality,
                topic_id=topic_id,
            )
            if doc is not None:
                return doc

        # Fallback: any topic
        doc = self._reader.sample_doc(
            lang=lang,
            min_quality=min_quality,
            max_quality=max_quality,
            topic_id=None,
        )
        if doc is not None:
            logger.debug(
                "TopicAligner: fallback to any-topic for lang=%s "
                "(preferred topic=%s had insufficient docs).",
                lang, topic_id,
            )
        return doc

    def aligned_batch(
        self,
        lang_schedule: Dict[str, int],
        quality_ranges: Dict[str, Tuple[float, float]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Sample a topic-aligned batch across languages.

        Parameters
        ----------
        lang_schedule : dict
            Maps language code to the number of documents to draw from
            that language in this batch. The sum of values should equal
            ``batch_size``.
        quality_ranges : dict
            Maps language code to ``(min_quality, max_quality)`` tuple.
            Languages with ``(0, 0)`` are skipped (N/A).
        batch_size : int
            Target batch size. The actual batch may be smaller if
            insufficient documents are available.

        Returns
        -------
        list of dict
            Each dict contains ``input_ids``, ``quality``, ``difficulty``,
            ``topic_id``, and an additional ``lang`` key indicating source
            language.
        """
        topic_id = self._next_topic()
        batch: List[Dict[str, Any]] = []

        for lang, n_samples in lang_schedule.items():
            min_q, max_q = quality_ranges.get(lang, (1.0, 5.0))

            for _ in range(n_samples):
                doc = self._sample_for_lang(lang, topic_id, min_q, max_q)
                if doc is not None:
                    doc["lang"] = lang
                    batch.append(doc)

        # If batch is under-filled (due to missing docs), pad with
        # any-topic samples from languages that had room
        if len(batch) < batch_size:
            deficit = batch_size - len(batch)
            available_langs = [
                lang for lang, (mq, xq) in quality_ranges.items()
                if not (mq == 0 and xq == 0)
            ]
            for _ in range(deficit):
                if not available_langs:
                    break
                lang = self._rng.choice(available_langs)
                min_q, max_q = quality_ranges[lang]
                doc = self._reader.sample_doc(
                    lang=lang,
                    min_quality=min_q,
                    max_quality=max_q,
                    topic_id=None,
                )
                if doc is not None:
                    doc["lang"] = lang
                    batch.append(doc)

        # Shuffle to avoid all same-language samples being contiguous
        self._rng.shuffle(batch)
        return batch

    # -- Introspection -------------------------------------------------------

    @property
    def shared_topics(self) -> List[int]:
        """The ordered list of topic IDs used for round-robin alignment."""
        return list(self._shared_topics)

    @property
    def topic_cursor(self) -> int:
        """Current position in the round-robin topic cycle."""
        return self._topic_cursor

    # -- Checkpoint support --------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serialize aligner state for checkpointing."""
        return {
            "topic_cursor": self._topic_cursor,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore aligner state from a checkpoint."""
        self._topic_cursor = state.get("topic_cursor", 0)
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])

    def __repr__(self) -> str:
        return (
            f"TopicAligner(n_shared_topics={len(self._shared_topics)}, "
            f"cursor={self._topic_cursor})"
        )
