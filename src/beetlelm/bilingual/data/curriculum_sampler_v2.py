# beetlelm/src/bilingual/data/curriculum_sampler_v2.py
"""
BeetleStream v2 core sampler and dataset.

Extends the existing DataMixConfig / DataMixConfig curriculum engine
with per-condition quality filtering (from quality_tables.py) and cross-
lingual topic alignment (from topic_aligner.py).

Architecture
------------

::

    CurriculumSamplerV2
      |
      +-- inner data_mix (DataMixConfig or DataMixConfig)
      |     |-- next_source()  -> language key
      |     |-- step()         -> advance curriculum
      |
      +-- quality_tables.get_quality_range(condition, lang_role, progress)
      |     |-- returns (min_quality, max_quality) per language per step
      |
      +-- TopicAligner (optional)
            |-- aligned_batch() -> topic-coherent cross-lingual batch

    CurriculumDataset (torch IterableDataset)
      |
      +-- Loads pretokenized Arrow datasets with {input_ids, quality, difficulty, topic_id}
      +-- Filters rows by quality range from per-condition table
      +-- Topic alignment via shared topic_id across languages
      +-- Yields {"input_ids": tensor} (same interface as PretokenizedMultilingualDataset)

Compatibility
-------------
CurriculumSamplerV2 implements the same ``next_source()`` / ``step()``
interface as DataMixConfig, so it is a drop-in replacement in existing
training loops. The CurriculumDataset yields the same dict format as
PretokenizedMultilingualDataset.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import IterableDataset

from beetlelm.bilingual.data.curriculum import DataMixConfig
from beetlelm.bilingual.data.quality_tables import (
    QUALITY_TABLES,
    get_quality_range,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language role mapping
# ---------------------------------------------------------------------------

def _build_role_map(
    condition: str,
    lang_keys: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Map concrete language keys to abstract roles (L1, L2, L3).

    For bilingual conditions (B1-B5), maps l1->L1, l2->L2.
    For trilingual conditions (T1-T4), maps the first three lang_keys
    to L1, L2, L3 respectively.

    Parameters
    ----------
    condition : str
        Condition code, e.g. "B2", "T3".
    lang_keys : list of str, optional
        Concrete language keys. If None, defaults to ["l1", "l2"] for
        bilingual or ["l1", "l2", "l3"] for trilingual.

    Returns
    -------
    dict
        Maps language key -> role string, e.g. {"eng": "L1", "nld": "L2"}.
    """
    is_trilingual = condition.upper().startswith("T")

    if lang_keys is None:
        if is_trilingual:
            lang_keys = ["l1", "l2", "l3"]
        else:
            lang_keys = ["l1", "l2"]

    roles = ["L1", "L2", "L3"] if is_trilingual else ["L1", "L2"]
    role_map = {}
    for i, role in enumerate(roles):
        if i < len(lang_keys):
            role_map[lang_keys[i]] = role
    return role_map


# ---------------------------------------------------------------------------
# CurriculumSamplerV2
# ---------------------------------------------------------------------------

class CurriculumSamplerV2:
    """BeetleStream v2 sampler wrapping an existing curriculum engine.

    Delegates language selection to the inner ``data_mix`` (DataMixConfig or
    DataMixConfig), and adds per-step quality range lookup via the
    condition quality tables.

    Parameters
    ----------
    data_mix : DataMixConfig or DataMixConfig
        Inner curriculum engine that controls language exposure schedule.
    condition : str
        Condition code (B1-B5, T1-T4) for quality table lookup.
    lang_keys : list of str, optional
        Concrete language keys in role order [L1, L2, ...].
        Defaults to ["l1", "l2"] (bilingual) or ["l1", "l2", "l3"] (trilingual).
    seed : int
        Random seed for any stochastic behaviour.
    """

    def __init__(
        self,
        data_mix: Union[DataMixConfig, DataMixConfig],
        condition: str,
        lang_keys: Optional[List[str]] = None,
        seed: int = 0,
    ) -> None:
        self.data_mix = data_mix
        self.condition = condition.upper()
        self._role_map = _build_role_map(self.condition, lang_keys)
        self._lang_keys = lang_keys or list(self._role_map.keys())
        self._rng = random.Random(seed)

        if self.condition not in QUALITY_TABLES:
            raise ValueError(
                f"Condition {self.condition!r} not found in quality tables. "
                f"Available: {sorted(QUALITY_TABLES.keys())}"
            )

    # -- DataMixConfig-compatible interface -----------------------------------

    @property
    def step_count(self) -> int:
        """Current step count (delegated to inner data_mix)."""
        return self.data_mix.step_count

    @property
    def token_count(self) -> int:
        """Current token count (delegated to inner data_mix)."""
        return self.data_mix.token_count

    def _progress(self) -> float:
        """Current training progress in [0, 1]."""
        return self.data_mix._progress()

    def current_phase_label(self) -> str:
        """Current phase label (delegated to inner data_mix)."""
        return self.data_mix.current_phase_label()

    def phase_just_changed(self) -> bool:
        """Whether a phase transition just occurred."""
        return self.data_mix.phase_just_changed()

    def next_source(self) -> str:
        """Select the next language to sample from.

        Delegates to the inner data_mix for curriculum-driven language
        selection.

        Returns
        -------
        str
            Language key, e.g. "l1", "eng", "fra".
        """
        return self.data_mix.next_source()

    def step(self, *, tokens: Optional[int] = None) -> None:
        """Advance curriculum state by one training step.

        Parameters
        ----------
        tokens : int, optional
            Number of tokens consumed (required for token-based mode).
        """
        self.data_mix.step(tokens=tokens)

    # -- Quality range API ---------------------------------------------------

    def get_quality_range(
        self,
        lang: str,
        step: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Return the quality range for a language at the current or given step.

        Parameters
        ----------
        lang : str
            Language key, e.g. "l1", "eng".
        step : int, optional
            Training step for progress computation. If None, uses the
            current step from the inner data_mix.

        Returns
        -------
        tuple of (float, float)
            ``(min_quality, max_quality)``. Returns ``(0, 0)`` if the
            language has no exposure at this point (N/A).
        """
        if step is not None:
            if self.data_mix.mode == "steps" and self.data_mix.total_steps:
                progress = min(1.0, step / self.data_mix.total_steps)
            elif self.data_mix.mode == "tokens" and self.data_mix.total_tokens:
                progress = min(1.0, step / self.data_mix.total_tokens)
            else:
                progress = self._progress()
        else:
            progress = self._progress()

        role = self._role_map.get(lang)
        if role is None:
            # Unknown language -- return permissive defaults
            logger.debug(
                "Language %r not in role map for condition %s; "
                "using permissive quality range.",
                lang, self.condition,
            )
            return (1.0, 5.0)

        return get_quality_range(self.condition, role, progress)

    def get_all_quality_ranges(
        self,
        step: Optional[int] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Return quality ranges for all languages at the current or given step.

        Returns
        -------
        dict
            Maps language key -> (min_quality, max_quality).
        """
        return {
            lang: self.get_quality_range(lang, step=step)
            for lang in self._lang_keys
        }

    # -- Convenience properties ----------------------------------------------

    @property
    def role_map(self) -> Dict[str, str]:
        """Mapping of language key -> abstract role (L1, L2, L3)."""
        return dict(self._role_map)

    @property
    def curriculum_type(self) -> str:
        """Curriculum type string from the inner data_mix."""
        return self.data_mix.curriculum_type

    # -- Checkpoint support --------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serialize sampler state for checkpointing."""
        return {
            "data_mix": self.data_mix.state_dict(),
            "condition": self.condition,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore sampler state from a checkpoint."""
        self.data_mix.load_state_dict(state["data_mix"])
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])

    def __repr__(self) -> str:
        return (
            f"CurriculumSamplerV2(condition={self.condition!r}, "
            f"curriculum={self.curriculum_type!r}, "
            f"progress={self._progress():.3f}, "
            f"roles={self._role_map})"
        )


# ---------------------------------------------------------------------------
# CurriculumDataset
# ---------------------------------------------------------------------------

class CurriculumDataset(IterableDataset):
    """IterableDataset for BeetleStream v2 with quality filtering and topic alignment.

    Loads pretokenized Arrow datasets that have columns:
    ``{input_ids, quality, difficulty, topic_id}`` and filters rows based
    on per-condition quality ranges. Optionally aligns topics across
    languages within each mini-batch.

    Yields ``{"input_ids": tensor}`` in the same format as
    ``PretokenizedMultilingualDataset`` for drop-in compatibility.

    Parameters
    ----------
    arrows : dict
        Maps language key -> path to Arrow dataset directory.
        Each Arrow dataset must contain columns: ``input_ids`` (required),
        ``quality`` (optional), ``topic_id`` (optional).
    sampler : CurriculumSamplerV2
        Controls language selection and quality range lookup.
    seq_len : int
        Sequence length (for validation; Arrow data should already be
        packed to seq_len+1).
    topic_aligned : bool
        If True, batches are assembled with shared topic_id across
        languages. Requires topic_id column in Arrow datasets.
    batch_size : int
        Number of samples per aligned batch (only used when
        topic_aligned=True).
    seed : int
        Random seed for reproducible sampling.

    Notes
    -----
    - ``sampler.step()`` must be called by the training loop, not by
      this dataset, to avoid double-stepping.
    - When ``topic_aligned=True``, the dataset pre-fetches aligned
      batches and yields individual samples from them.
    """

    def __init__(
        self,
        arrows: Dict[str, str],
        sampler: CurriculumSamplerV2,
        seq_len: int,
        topic_aligned: bool = False,
        batch_size: int = 8,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.seq_len = seq_len
        self.topic_aligned = topic_aligned
        self.batch_size = batch_size
        self._rng = random.Random(seed)

        # Load Arrow datasets
        from datasets import Dataset, load_from_disk
        self._pools: Dict[str, Dataset] = {}
        for lang_key, path in arrows.items():
            ds = load_from_disk(path)
            self._pools[lang_key] = ds
            logger.info(
                "CurriculumDataset: loaded %s with %d rows.",
                lang_key, len(ds),
            )

        # Per-language cyclic index
        self._indices: Dict[str, int] = {k: 0 for k in self._pools}

        # Build per-language quality and topic indices for efficient filtering
        self._quality_col_available: Dict[str, bool] = {}
        self._topic_col_available: Dict[str, bool] = {}
        for lang_key, ds in self._pools.items():
            self._quality_col_available[lang_key] = "quality" in ds.column_names
            self._topic_col_available[lang_key] = "topic_id" in ds.column_names

    def _sample_from_pool(
        self,
        lang_key: str,
        min_quality: float = 1.0,
        max_quality: float = 5.0,
        topic_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Sample one document from a language pool with quality filtering.

        Parameters
        ----------
        lang_key : str
            Language key.
        min_quality, max_quality : float
            Quality bounds.
        topic_id : int, optional
            If set, prefer documents with this topic_id.

        Returns
        -------
        dict or None
            Document dict with at least ``input_ids``.
        """
        if lang_key not in self._pools:
            return None

        pool = self._pools[lang_key]
        pool_size = len(pool)
        if pool_size == 0:
            return None

        has_quality = self._quality_col_available[lang_key]
        has_topic = self._topic_col_available[lang_key]

        # Try up to `max_attempts` random positions to find a matching doc.
        # For large pools with broad quality ranges this rarely needs more
        # than 1-2 attempts.
        max_attempts = min(50, pool_size)
        start_idx = self._indices[lang_key]

        for attempt in range(max_attempts):
            idx = (start_idx + attempt) % pool_size
            row = pool[idx]

            # Quality filter
            if has_quality and (min_quality > 1.0 or max_quality < 5.0):
                q = row.get("quality", 3.0)
                if q < min_quality or q > max_quality:
                    continue

            # Topic filter (soft preference, not hard requirement)
            if topic_id is not None and has_topic:
                row_topic = row.get("topic_id", -1)
                # Accept on first pass only if topic matches; on later
                # attempts accept any to avoid starvation
                if attempt < max_attempts // 2 and row_topic != topic_id:
                    continue

            # Found a matching document
            self._indices[lang_key] = (idx + 1) % pool_size
            return {
                "input_ids": row["input_ids"],
                "quality": row.get("quality", 0.0) if has_quality else 0.0,
                "topic_id": row.get("topic_id", -1) if has_topic else -1,
                "lang": lang_key,
            }

        # Exhausted attempts -- return whatever is at the current index
        idx = start_idx % pool_size
        self._indices[lang_key] = (idx + 1) % pool_size
        row = pool[idx]
        return {
            "input_ids": row["input_ids"],
            "quality": row.get("quality", 0.0) if has_quality else 0.0,
            "topic_id": row.get("topic_id", -1) if has_topic else -1,
            "lang": lang_key,
        }

    def _select_shared_topic(self, languages: List[str]) -> Optional[int]:
        """Select a topic_id that has documents in multiple languages.

        Uses a simple heuristic: pick the topic_id from the most recently
        sampled document, then check if other languages have docs with
        that topic.
        """
        # Sample a candidate from the first language
        if not languages:
            return None

        lang = languages[0]
        pool = self._pools.get(lang)
        if pool is None or len(pool) == 0:
            return None

        has_topic = self._topic_col_available.get(lang, False)
        if not has_topic:
            return None

        idx = self._indices[lang] % len(pool)
        row = pool[idx]
        return row.get("topic_id", None)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield training samples as {"input_ids": tensor}.

        Each iteration:
        1. The sampler selects a language via next_source().
        2. Quality range is looked up for that language at current progress.
        3. A document matching the quality range (and optionally topic) is
           fetched from the Arrow pool.
        4. The result is yielded as a tensor dict.
        """
        if self.topic_aligned:
            yield from self._iter_topic_aligned()
        else:
            yield from self._iter_simple()

    def _iter_simple(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Simple iteration: one document per step, quality-filtered."""
        while True:
            lang_key = self.sampler.next_source()
            min_q, max_q = self.sampler.get_quality_range(lang_key)

            # Skip N/A languages (no exposure at this point)
            if min_q == 0 and max_q == 0:
                continue

            # Fallback if language not in our pools
            if lang_key not in self._pools:
                fallback = next(
                    (k for k in self._pools if k != lang_key), None,
                )
                if fallback is None:
                    raise StopIteration("No language pools available.")
                lang_key = fallback
                min_q, max_q = self.sampler.get_quality_range(lang_key)

            doc = self._sample_from_pool(lang_key, min_q, max_q)
            if doc is None:
                continue

            yield {"input_ids": torch.tensor(doc["input_ids"], dtype=torch.long)}

    def _iter_topic_aligned(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Topic-aligned iteration: batches share topic_id across languages."""
        buffer: List[Dict[str, Any]] = []

        while True:
            # Refill buffer with an aligned batch
            if not buffer:
                buffer = self._fetch_aligned_batch()
                if not buffer:
                    # Could not produce any samples; yield from simple mode
                    # as fallback for one step
                    lang_key = self.sampler.next_source()
                    min_q, max_q = self.sampler.get_quality_range(lang_key)
                    if min_q == 0 and max_q == 0:
                        continue
                    if lang_key not in self._pools:
                        lang_key = next(iter(self._pools), None)
                        if lang_key is None:
                            raise StopIteration("No pools available.")
                    doc = self._sample_from_pool(lang_key, min_q, max_q)
                    if doc is not None:
                        yield {
                            "input_ids": torch.tensor(
                                doc["input_ids"], dtype=torch.long
                            )
                        }
                    continue

            doc = buffer.pop(0)
            yield {"input_ids": torch.tensor(doc["input_ids"], dtype=torch.long)}

    def _fetch_aligned_batch(self) -> List[Dict[str, Any]]:
        """Produce one batch of topic-aligned documents.

        Returns
        -------
        list of dict
            Documents sharing a common topic_id.
        """
        # Determine language schedule for this batch
        lang_counts: Dict[str, int] = {}
        for _ in range(self.batch_size):
            lang = self.sampler.next_source()
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Pick a shared topic
        active_langs = [
            lang for lang in lang_counts
            if lang in self._pools
            and self.sampler.get_quality_range(lang) != (0, 0)
        ]
        topic_id = self._select_shared_topic(active_langs)

        # Sample documents
        batch: List[Dict[str, Any]] = []
        for lang, n in lang_counts.items():
            min_q, max_q = self.sampler.get_quality_range(lang)
            if min_q == 0 and max_q == 0:
                continue
            if lang not in self._pools:
                continue

            for _ in range(n):
                doc = self._sample_from_pool(lang, min_q, max_q, topic_id)
                if doc is not None:
                    batch.append(doc)

        self._rng.shuffle(batch)
        return batch

    # -- Checkpoint support --------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serialize dataset state for checkpointing."""
        return {
            "indices": dict(self._indices),
            "sampler": self.sampler.state_dict(),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore dataset state from a checkpoint."""
        if "indices" in state:
            self._indices = dict(state["indices"])
        if "sampler" in state:
            self.sampler.load_state_dict(state["sampler"])
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])

    def __repr__(self) -> str:
        pools_info = ", ".join(
            f"{k}: {len(v)}" for k, v in self._pools.items()
        )
        return (
            f"CurriculumDataset(pools=[{pools_info}], "
            f"topic_aligned={self.topic_aligned}, "
            f"sampler={self.sampler!r})"
        )
