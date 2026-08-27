# beetlelm/src/bilingual/data/smooth_curriculum.py
"""Backward-compatibility shim — all code moved to curriculum.py."""

# Re-export everything from curriculum.py so existing imports still work.
from beetlelm.bilingual.data.curriculum import *  # noqa: F401,F403
from beetlelm.bilingual.data.curriculum import DataMixConfig

# Legacy alias for code that imported SmoothDataMixConfig
SmoothDataMixConfig = DataMixConfig
