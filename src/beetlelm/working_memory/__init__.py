"""
src/working_memory
──────────────────
Optional "working memory" attention constraint for BeetleLM.

Ported from CPLM (osekilab/CPLM) — implements a time-varying ALiBi slope
schedule that starts with strong locality bias and decays as training
progresses, simulating the developmental expansion of working memory.

    from beetlelm.working_memory import build_working_memory, WorkingMemoryScheduler

Returns None if the YAML block is absent or ``enabled: false``, so the
default beetle_trainer.py path is completely unaffected.
"""

from beetlelm.working_memory.alibi import (  # noqa: F401
    WorkingMemoryScheduler,
    build_working_memory,
)

__all__ = ["WorkingMemoryScheduler", "build_working_memory"]
