# beetlelm/src/bilingual/data/quality_tables.py
"""
Per-condition quality filtering tables for BeetleStream v2.

Encodes the minimum quality thresholds for each language role (L1, L2, L3)
across training progress for all bilingual (B1-B5) and trilingual (T1-T4)
conditions. Quality values are on a 1-5 scale matching beetle-data Stage D
quality annotations.

The tables reflect the principle that data quality requirements should vary
with acquisition stage: early monolingual input can be lower-quality (the
learner tolerates noise), but newly introduced languages need high-quality
exemplars for initial grammar induction, relaxing as exposure accumulates.

N/A entries (encoded as ``None``) indicate that the language has 0% exposure
at that training point and should not be sampled at all.

Usage
-----
    from beetlelm.bilingual.data.quality_tables import QUALITY_TABLES, get_quality_range

    # Get the quality range for L2 at 40% through B2 training
    min_q, max_q = get_quality_range("B2", "L2", progress=0.40)
    # -> (4, 5.0) -- high quality required for early L2 exposure

    # N/A language returns (0, 0) sentinel
    min_q, max_q = get_quality_range("B2", "L2", progress=0.10)
    # -> (0, 0) -- L2 has no exposure at this point
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A single threshold entry: (upper_bound_of_range, {lang_role: min_quality_or_None})
# The threshold is the *upper bound* of the progress range that this row covers.
# For example, (0.15, {"L1": 3, "L2": 4}) means: for t in [previous_upper, 0.15).
# The last entry uses 1.0 as the upper bound.
QualityRow = Tuple[float, Dict[str, Optional[int]]]

# Condition name -> ordered list of quality rows
QualityTable = Dict[str, List[QualityRow]]

# Maximum quality score in the annotation scheme
MAX_QUALITY: float = 5.0


# ---------------------------------------------------------------------------
# Quality tables
# ---------------------------------------------------------------------------

QUALITY_TABLES: QualityTable = {
    # ── Bilingual conditions ─────────────────────────────────────────────

    "B1": [
        # Simultaneous Balanced (BFLA)
        # Both languages present from t=0; L2 starts higher quality, relaxes
        (0.15, {"L1": 3, "L2": 4}),
        (0.50, {"L1": 3, "L2": 3}),
        (1.00, {"L1": 2, "L2": 2}),
    ],
    "B2": [
        # Sequential L2 Learner
        # L1 only initially, L2 introduced mid-training at high quality
        (0.25, {"L1": 3, "L2": None}),
        (0.50, {"L1": 3, "L2": 4}),
        (0.70, {"L1": 2, "L2": 3}),
        (1.00, {"L1": 2, "L2": 2}),
    ],
    "B3": [
        # Heritage Speaker (Attrition + Re-exposure)
        # L1 dominant early, L2 dominates later, L1 maintained
        (0.20, {"L1": 3, "L2": 4}),
        (0.50, {"L1": 3, "L2": 3}),
        (1.00, {"L1": 3, "L2": 2}),
    ],
    "B4": [
        # Classroom L2 Learner
        # L1 dominant throughout, L2 always requires high quality
        (1.00, {"L1": 3, "L2": 4}),
    ],
    "B5": [
        # Late Low-Exposure L2
        # Long monolingual phase, L2 introduced late at high quality
        (0.70, {"L1": 3, "L2": None}),
        (1.00, {"L1": 3, "L2": 4}),
    ],

    # ── Trilingual conditions ────────────────────────────────────────────

    "T1": [
        # Simultaneous Trilingual
        # All three from t=0; L2/L3 start higher quality, relax together
        (0.15, {"L1": 3, "L2": 4, "L3": 4}),
        (0.50, {"L1": 3, "L2": 3, "L3": 3}),
        (1.00, {"L1": 2, "L2": 2, "L3": 2}),
    ],
    "T2": [
        # Sequential L1 -> L2 -> L3
        # Staged introduction: L1 first, then L2, then L3
        (0.25, {"L1": 3, "L2": None, "L3": None}),
        (0.50, {"L1": 3, "L2": 4,    "L3": None}),
        (0.65, {"L1": 2, "L2": 3,    "L3": None}),
        (0.80, {"L1": 2, "L2": 2,    "L3": 4}),
        (1.00, {"L1": 2, "L2": 2,    "L3": 3}),
    ],
    "T3": [
        # Heritage + L3 Learner
        # Early L1 -> L2 dominance, later L3 introduction
        (0.20, {"L1": 3, "L2": 4, "L3": None}),
        (0.50, {"L1": 3, "L2": 3, "L3": 4}),
        (0.65, {"L1": 3, "L2": 2, "L3": 4}),
        (1.00, {"L1": 3, "L2": 2, "L3": 3}),
    ],
    "T4": [
        # Dominant L2 + Late L3
        # L2 dominant, L1 persistent, L3 introduced later
        (0.40, {"L1": 3, "L2": 3, "L3": None}),
        (0.55, {"L1": 3, "L2": 2, "L3": 4}),
        (1.00, {"L1": 2, "L2": 2, "L3": 3}),
    ],
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _find_row(condition: str, progress: float) -> Dict[str, Optional[int]]:
    """Return the quality requirements row for the given condition and progress.

    Parameters
    ----------
    condition : str
        Condition code, e.g. "B1", "T3".
    progress : float
        Training progress in [0, 1].

    Returns
    -------
    dict
        Maps language role (e.g. "L1", "L2") to minimum quality (int) or
        None if the language has no exposure at this point.

    Raises
    ------
    KeyError
        If the condition is not in QUALITY_TABLES.
    """
    if condition not in QUALITY_TABLES:
        raise KeyError(
            f"Unknown condition {condition!r}. "
            f"Available: {sorted(QUALITY_TABLES.keys())}"
        )

    table = QUALITY_TABLES[condition]
    progress = max(0.0, min(1.0, progress))

    for upper_bound, quality_row in table:
        if progress < upper_bound or upper_bound >= 1.0:
            return quality_row

    # Fallback to the last row (should not be reached due to 1.0 sentinel)
    return table[-1][1]


def get_quality_range(
    condition: str,
    lang_role: str,
    progress: float,
) -> Tuple[float, float]:
    """Return the (min_quality, max_quality) for a language role at a given progress.

    Parameters
    ----------
    condition : str
        Condition code, e.g. "B1", "T2".
    lang_role : str
        Language role, e.g. "L1", "L2", "L3". Case-insensitive.
    progress : float
        Training progress in [0, 1].

    Returns
    -------
    tuple of (float, float)
        ``(min_quality, max_quality)``. If the language has no exposure at
        this training point (N/A), returns ``(0, 0)`` as a sentinel.

    Examples
    --------
    >>> get_quality_range("B2", "L2", 0.10)
    (0, 0)
    >>> get_quality_range("B2", "L2", 0.35)
    (4, 5.0)
    >>> get_quality_range("B1", "L1", 0.80)
    (2, 5.0)
    """
    row = _find_row(condition, progress)
    role = lang_role.upper()

    if role not in row:
        raise KeyError(
            f"Language role {lang_role!r} not defined for condition {condition!r}. "
            f"Available roles: {sorted(row.keys())}"
        )

    min_quality = row[role]

    if min_quality is None:
        # N/A -- language has 0% exposure at this point
        return (0, 0)

    return (float(min_quality), MAX_QUALITY)


def list_conditions() -> List[str]:
    """Return all condition codes that have quality tables defined."""
    return sorted(QUALITY_TABLES.keys())


def get_roles(condition: str) -> List[str]:
    """Return the language roles defined for a condition (e.g. ['L1', 'L2'])."""
    if condition not in QUALITY_TABLES:
        raise KeyError(
            f"Unknown condition {condition!r}. "
            f"Available: {sorted(QUALITY_TABLES.keys())}"
        )
    # Collect all roles across all rows
    roles: set = set()
    for _, row in QUALITY_TABLES[condition]:
        roles.update(row.keys())
    return sorted(roles)
