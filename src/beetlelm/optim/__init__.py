"""BeetleLM optimizer registry.

Centralises optimizer construction so the trainer / config / tests can
all reach the same set of supported optimizers via a single helper.
"""
from beetlelm.optim.schedule_free import AdamWScheduleFree

__all__ = ["AdamWScheduleFree"]
