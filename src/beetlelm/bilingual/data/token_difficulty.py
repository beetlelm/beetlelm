"""
src/bilingual/data/token_difficulty.py
────────────────────────────────────────
Token-level difficulty scoring for the BeetleLM curriculum sampler.

Three scorer modes are supported:

* ``length``    — document length in tokens (short = easy, long = hard).
* ``aoa``       — Age-of-Acquisition score from a lookup table
                  (low AoA = early-acquired = easy).
* ``surprisal`` — mean unigram surprisal from a pre-built reference table
                  (low surprisal = easy).

The ``TokenDifficultyScorer`` ABC defines the shared interface; concrete
subclasses implement ``score(doc) -> float``.  The scheduler calls
``sample_weights(docs, progress) -> list[float]`` to get per-document
softmax-temperature-annealed weights for the ``_sample_from_pool`` caller.

YAML example (under ``curriculum:``):

    curriculum:
      type: smooth
      condition: B2
      token_difficulty:
        enabled: true
        mode: length           # length | aoa | surprisal
        schedule: ramp         # ramp | step
        temperature_start: 4.0
        temperature_end: 0.5
        target_progress_end: 0.6
        direction: easy_to_hard
        aoa_table_path: null   # required if mode == aoa
        surprisal_ref_path: null  # required if mode == surprisal
"""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── ABC ──────────────────────────────────────────────────────────────────────

class TokenDifficultyScorer(ABC):
    """Abstract base class for document difficulty scorers."""

    @abstractmethod
    def score(self, doc: Dict[str, Any]) -> float:
        """Return a difficulty scalar for ``doc``.

        Higher = harder. Implementations should normalise to a comparable
        range (but exact magnitude doesn't matter since only relative order
        is used inside the softmax weighting).
        """

    def sample_weights(
        self,
        docs: List[Dict[str, Any]],
        progress: float,
        temperature_start: float = 4.0,
        temperature_end: float = 0.5,
        target_progress_end: float = 0.6,
        direction: str = "easy_to_hard",
    ) -> List[float]:
        """Return per-document softmax weights for curriculum sampling.

        Temperature is annealed linearly from ``temperature_start`` to
        ``temperature_end`` as ``progress`` goes from 0 to
        ``target_progress_end``.  After that it is held at
        ``temperature_end``.

        Parameters
        ----------
        docs     : list of document dicts from the Arrow pool.
        progress : float in [0, 1] — training progress.
        direction: ``"easy_to_hard"`` or ``"hard_to_easy"``.
        """
        t_progress = min(progress / max(target_progress_end, 1e-8), 1.0)
        temperature = temperature_start + (temperature_end - temperature_start) * t_progress

        scores = [float(self.score(d)) for d in docs]
        # easy_to_hard: low-score (easy) docs should have highest weight at start.
        # Negate so exp(score/T) rewards low original score.
        if direction == "easy_to_hard":
            scores = [-s for s in scores]

        # Softmax-temperature weighting: weight ∝ exp(score / T)
        max_s = max(scores) if scores else 0.0
        exp_s = [math.exp((s - max_s) / max(temperature, 1e-8)) for s in scores]
        total = sum(exp_s) or 1.0
        return [e / total for e in exp_s]


# ── Concrete scorers ──────────────────────────────────────────────────────────

class LengthScorer(TokenDifficultyScorer):
    """Difficulty ∝ number of tokens in the document."""

    def score(self, doc: Dict[str, Any]) -> float:
        ids = doc.get("input_ids", [])
        return float(len(ids))


class AoAScorer(TokenDifficultyScorer):
    """Difficulty based on mean Age-of-Acquisition score of the tokens.

    Parameters
    ----------
    aoa_table_path : str
        Path to a JSON file mapping token_id (int) → AoA score (float).
        Lower AoA = acquired earlier = easier.
    default_aoa : float
        AoA score for tokens not in the table (defaults to mid-range 5.0).
    """

    def __init__(self, aoa_table_path: str, default_aoa: float = 5.0):
        with open(aoa_table_path, "r") as f:
            raw = json.load(f)
        self._table: Dict[int, float] = {int(k): float(v) for k, v in raw.items()}
        self._default = float(default_aoa)
        logger.info("[AoAScorer] Loaded %d entries from %s.", len(self._table), aoa_table_path)

    def score(self, doc: Dict[str, Any]) -> float:
        ids = doc.get("input_ids", [])
        if not ids:
            return self._default
        return sum(self._table.get(int(i), self._default) for i in ids) / len(ids)


class SurprisalScorer(TokenDifficultyScorer):
    """Difficulty based on mean unigram surprisal from a reference table.

    Parameters
    ----------
    surprisal_ref_path : str
        Path to a JSON file mapping token_id (int) → surprisal (float).
    default_surprisal : float
        Fallback surprisal for unknown tokens.
    """

    def __init__(self, surprisal_ref_path: str, default_surprisal: float = 10.0):
        with open(surprisal_ref_path, "r") as f:
            raw = json.load(f)
        self._table: Dict[int, float] = {int(k): float(v) for k, v in raw.items()}
        self._default = float(default_surprisal)
        logger.info(
            "[SurprisalScorer] Loaded %d entries from %s.",
            len(self._table), surprisal_ref_path,
        )

    def score(self, doc: Dict[str, Any]) -> float:
        ids = doc.get("input_ids", [])
        if not ids:
            return self._default
        return sum(self._table.get(int(i), self._default) for i in ids) / len(ids)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_token_difficulty(cfg: dict) -> Optional["TokenDifficultyScorer"]:
    """Build a difficulty scorer from the ``curriculum.token_difficulty:`` YAML block.

    Returns None if ``enabled: false`` or the block is absent.
    """
    if not cfg or not cfg.get("enabled", False):
        return None

    mode = cfg.get("mode", "length").lower()

    if mode == "length":
        return LengthScorer()
    elif mode == "aoa":
        path = cfg.get("aoa_table_path")
        if not path:
            raise ValueError("[token_difficulty] mode='aoa' requires aoa_table_path.")
        return AoAScorer(aoa_table_path=path)
    elif mode == "surprisal":
        path = cfg.get("surprisal_ref_path")
        if not path:
            raise ValueError(
                "[token_difficulty] mode='surprisal' requires surprisal_ref_path."
            )
        return SurprisalScorer(surprisal_ref_path=path)
    else:
        raise ValueError(
            f"[token_difficulty] Unknown mode {mode!r}. "
            f"Allowed: 'length', 'aoa', 'surprisal'."
        )
