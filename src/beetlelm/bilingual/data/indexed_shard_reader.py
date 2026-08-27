# beetlelm/src/bilingual/data/indexed_shard_reader.py
"""
Indexed Parquet shard reader for beetle-data Stage D output.

Reads pretokenized, quality-annotated, topic-tagged Parquet shards produced
by beetle-data's Stage D pipeline. Each shard contains columns:

    input_ids   : list<int64>    -- pretokenized token IDs (seq_len+1)
    quality     : float32        -- document quality score [1.0, 5.0]
    difficulty  : float32        -- estimated reading difficulty
    topic_id    : int32          -- BERTopic cluster ID
    lang        : string         -- ISO 639-3 language code

Shards are organised on disk as::

    <data_root>/
        manifest.json          -- index mapping lang -> topic -> shard files
        eng/
            shard_000.parquet
            shard_001.parquet
            ...
        nld/
            shard_000.parquet
            ...

The manifest.json schema::

    {
        "version": 2,
        "languages": {
            "eng": {
                "topics": {
                    "42": [
                        {"path": "eng/shard_003.parquet", "n_docs": 1847},
                        ...
                    ]
                }
            }
        }
    }

Key design decisions
--------------------
- LRU cache of ~20 most-recently-used shard DataFrames (~200 MB RAM budget).
- PyArrow predicate pushdown on the ``quality`` column avoids scanning
  full shards when only a quality band is needed.
- Thread-safe: shard reads are guarded by a lock so multiple DataLoader
  workers can share one reader via fork.
- Deterministic: sample_doc uses a seeded RNG for reproducibility, with
  state_dict / load_state_dict for mid-training checkpointing.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import pyarrow.compute as pc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LRU shard cache
# ---------------------------------------------------------------------------

class _ShardLRU:
    """Bounded LRU cache for PyArrow Tables read from Parquet files.

    Parameters
    ----------
    max_shards : int
        Maximum number of shard tables to keep in memory.
    """

    def __init__(self, max_shards: int = 20):
        self._max = max_shards
        self._cache: OrderedDict[str, "pyarrow.Table"] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str) -> Optional["pyarrow.Table"]:
        """Retrieve a cached shard, promoting it to most-recently-used."""
        with self._lock:
            if path in self._cache:
                self._cache.move_to_end(path)
                return self._cache[path]
        return None

    def put(self, path: str, table: "pyarrow.Table") -> None:
        """Insert a shard into the cache, evicting the LRU entry if full."""
        with self._lock:
            if path in self._cache:
                self._cache.move_to_end(path)
                return
            if len(self._cache) >= self._max:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("Shard cache evicting: %s", evicted_key)
            self._cache[path] = table

    def clear(self) -> None:
        """Drop all cached shards."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Manifest index types
# ---------------------------------------------------------------------------

# {lang: {topic_id: [(absolute_path, n_docs)]}}
ShardIndex = Dict[str, Dict[int, List[Tuple[str, int]]]]


# ---------------------------------------------------------------------------
# IndexedShardReader
# ---------------------------------------------------------------------------

class IndexedShardReader:
    """Reads indexed Parquet shards with LRU caching and predicate pushdown.

    Parameters
    ----------
    data_root : str or Path
        Root directory containing ``manifest.json`` and language subdirs.
    max_cached_shards : int
        Maximum number of Parquet shard tables to keep in the LRU cache.
        Each shard is typically ~10 MB, so 20 shards ~ 200 MB.
    seed : int
        Random seed for reproducible document sampling.
    """

    def __init__(
        self,
        data_root: str,
        max_cached_shards: int = 20,
        seed: int = 0,
    ) -> None:
        self._root = Path(data_root)
        self._manifest_path = self._root / "manifest.json"

        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {self._manifest_path}. "
                f"Run beetle-data Stage D to generate indexed shards."
            )

        self._cache = _ShardLRU(max_shards=max_cached_shards)
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

        # Build in-memory index from manifest
        self._index: ShardIndex = {}
        self._load_manifest()

        # Sampling counters for checkpointing
        self._sample_count: int = 0

    def _load_manifest(self) -> None:
        """Parse manifest.json and build the in-memory shard index."""
        with open(self._manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        version = manifest.get("version", 1)
        if version < 2:
            logger.warning(
                "Manifest version %d; expected >= 2. "
                "Older manifests may lack topic-level granularity.",
                version,
            )

        languages = manifest.get("languages", {})
        for lang, lang_info in languages.items():
            self._index[lang] = {}
            topics = lang_info.get("topics", {})
            for topic_str, shard_list in topics.items():
                topic_id = int(topic_str)
                entries = []
                for shard_entry in shard_list:
                    abs_path = str(self._root / shard_entry["path"])
                    n_docs = shard_entry["n_docs"]
                    entries.append((abs_path, n_docs))
                self._index[lang][topic_id] = entries

        n_langs = len(self._index)
        n_topics = sum(len(topics) for topics in self._index.values())
        n_shards = sum(
            len(entries)
            for topics in self._index.values()
            for entries in topics.values()
        )
        logger.info(
            "IndexedShardReader: loaded manifest with %d languages, "
            "%d topic groups, %d shard files.",
            n_langs, n_topics, n_shards,
        )

    # -- Shard I/O with caching and predicate pushdown -----------------------

    def _read_shard(
        self,
        path: str,
        min_quality: Optional[float] = None,
        max_quality: Optional[float] = None,
    ) -> "pyarrow.Table":
        """Read a Parquet shard, applying quality predicate pushdown.

        If no quality filter is requested, the full shard is returned from
        cache (or read once and cached). When a quality filter is active,
        we still cache the *full* shard but apply the filter on retrieval
        to avoid redundant disk reads for different quality bands.

        Parameters
        ----------
        path : str
            Absolute path to the Parquet shard file.
        min_quality, max_quality : float, optional
            Quality bounds for predicate pushdown filtering.

        Returns
        -------
        pyarrow.Table
            Filtered (or full) shard table.
        """
        table = self._cache.get(path)

        if table is None:
            # Read from disk -- PyArrow predicate pushdown filters at scan time
            # when no cache entry exists and a quality filter is requested.
            if min_quality is not None or max_quality is not None:
                filters = []
                if min_quality is not None:
                    filters.append(("quality", ">=", min_quality))
                if max_quality is not None:
                    filters.append(("quality", "<=", max_quality))
                try:
                    table = pq.read_table(path, filters=filters)
                except Exception:
                    # Fallback: read without filters if schema mismatch
                    table = pq.read_table(path)
            else:
                table = pq.read_table(path)

            # Cache the *unfiltered* version for future different-range lookups
            try:
                full_table = pq.read_table(path)
                self._cache.put(path, full_table)
            except Exception:
                self._cache.put(path, table)

            return table

        # Table was cached -- apply filter in-memory
        if min_quality is not None or max_quality is not None:
            mask = None
            if min_quality is not None and "quality" in table.column_names:
                ge_mask = pc.greater_equal(table["quality"], min_quality)
                mask = ge_mask
            if max_quality is not None and "quality" in table.column_names:
                le_mask = pc.less_equal(table["quality"], max_quality)
                mask = le_mask if mask is None else pc.and_(mask, le_mask)
            if mask is not None:
                table = table.filter(mask)

        return table

    # -- Public API ----------------------------------------------------------

    def get_languages(self) -> List[str]:
        """Return all language codes in the index."""
        return sorted(self._index.keys())

    def get_topics(self, lang: str) -> List[int]:
        """Return all topic IDs available for a language.

        Parameters
        ----------
        lang : str
            Language code, e.g. "eng".

        Returns
        -------
        list of int
            Sorted topic IDs.
        """
        if lang not in self._index:
            return []
        return sorted(self._index[lang].keys())

    def get_topic_doc_count(self, lang: str, topic_id: int) -> int:
        """Return the total document count for a (lang, topic) pair.

        Parameters
        ----------
        lang : str
        topic_id : int

        Returns
        -------
        int
            Sum of n_docs across all shards for this (lang, topic).
        """
        if lang not in self._index or topic_id not in self._index[lang]:
            return 0
        return sum(n for _, n in self._index[lang][topic_id])

    def sample_doc(
        self,
        lang: str,
        min_quality: float = 1.0,
        max_quality: float = 5.0,
        topic_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Sample a single document matching the quality and topic constraints.

        Sampling is two-level: first a shard is chosen (weighted by n_docs),
        then a row is sampled uniformly from the quality-filtered shard.

        Parameters
        ----------
        lang : str
            Language code.
        min_quality : float
            Minimum quality score (inclusive).
        max_quality : float
            Maximum quality score (inclusive).
        topic_id : int, optional
            If provided, restrict to shards for this topic. If None, sample
            from any topic.

        Returns
        -------
        dict or None
            A document dict with keys ``input_ids``, ``quality``,
            ``difficulty``, ``topic_id``. Returns None if no matching
            documents are found.
        """
        if lang not in self._index:
            logger.warning("Language %r not in shard index.", lang)
            return None

        lang_topics = self._index[lang]

        # Determine candidate shards
        if topic_id is not None:
            if topic_id not in lang_topics:
                return None
            candidate_shards = lang_topics[topic_id]
        else:
            # Flatten all shards across topics
            candidate_shards = [
                entry
                for entries in lang_topics.values()
                for entry in entries
            ]

        if not candidate_shards:
            return None

        # Weighted shard selection by document count
        with self._lock:
            weights = [n for _, n in candidate_shards]
            total_weight = sum(weights)
            if total_weight == 0:
                return None

            r = self._rng.random() * total_weight
            cumulative = 0
            chosen_path = candidate_shards[0][0]
            for path, n in candidate_shards:
                cumulative += n
                if r <= cumulative:
                    chosen_path = path
                    break

        # Read shard with quality filtering
        table = self._read_shard(chosen_path, min_quality, max_quality)
        n_rows = table.num_rows

        if n_rows == 0:
            # Quality filter eliminated all rows in this shard; try others
            # with a limited retry
            for path, _ in candidate_shards:
                if path == chosen_path:
                    continue
                table = self._read_shard(path, min_quality, max_quality)
                if table.num_rows > 0:
                    n_rows = table.num_rows
                    break
            else:
                return None

        with self._lock:
            row_idx = self._rng.randint(0, n_rows - 1)
            self._sample_count += 1

        # Convert PyArrow row to dict
        row = table.slice(row_idx, 1).to_pydict()
        return {
            "input_ids": row["input_ids"][0],
            "quality": row["quality"][0] if "quality" in row else 0.0,
            "difficulty": row["difficulty"][0] if "difficulty" in row else 0.0,
            "topic_id": row["topic_id"][0] if "topic_id" in row else -1,
        }

    # -- Checkpoint support --------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serialize reader state for checkpointing.

        Returns
        -------
        dict
            Contains the RNG state and sample counter so that sampling
            can be resumed deterministically.
        """
        return {
            "sample_count": self._sample_count,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore reader state from a checkpoint.

        Parameters
        ----------
        state : dict
            State dict produced by a prior ``state_dict()`` call.
        """
        self._sample_count = state.get("sample_count", 0)
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])

    # -- Misc ----------------------------------------------------------------

    def clear_cache(self) -> None:
        """Drop all cached shard tables to free memory."""
        self._cache.clear()

    def __repr__(self) -> str:
        n_langs = len(self._index)
        n_topics = sum(len(t) for t in self._index.values())
        return (
            f"IndexedShardReader(root={str(self._root)!r}, "
            f"languages={n_langs}, topics={n_topics}, "
            f"cached_shards={len(self._cache)}, "
            f"samples_drawn={self._sample_count})"
        )
