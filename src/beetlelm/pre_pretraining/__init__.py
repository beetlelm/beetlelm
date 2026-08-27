"""
src/pre_pretraining
───────────────────
Optional formal-language pre-pretraining stage for BeetleLM.

Ported from the `michahu/pre-pretraining` recipe:
pre-train the model on a formal language (Shuffle-Dyck by default) for a
short number of steps, THEN begin the standard natural-language pre-training
from those weights.

    from beetlelm.pre_pretraining import build_pre_pretraining, run_pre_pretraining_stage

``build_pre_pretraining`` returns ``None`` if the YAML block is absent or
``enabled: false``, so the default beetle_trainer.py path is completely
unaffected.
"""

from beetlelm.pre_pretraining.formal_lang import (  # noqa: F401
    generate_shuff_dyck,
    ShuffleDyckIterableDataset,
)
from beetlelm.pre_pretraining.stage import (  # noqa: F401
    PrePretrainingConfig,
    build_pre_pretraining,
    run_pre_pretraining_stage,
)

__all__ = [
    "generate_shuff_dyck",
    "ShuffleDyckIterableDataset",
    "PrePretrainingConfig",
    "build_pre_pretraining",
    "run_pre_pretraining_stage",
]
