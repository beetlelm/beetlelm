"""BGPT package.

Re-exports :class:`GPT2Decoder` under the public name ``BGPT`` so that the
trainer dispatch in ``src/training/beetle_trainer.py`` (which historically
imported ``from beetlelm.bgpt.bgpt import BGPT``) resolves correctly. The actual
implementation lives in :mod:`beetlelm.bgpt.bgpt`.
"""

from beetlelm.bgpt.bgpt import GPT2Decoder, GPT2HF, GPT2HFConfig

# Public alias used by the architecture registry. Intentionally a plain
# binding (not a subclass) so existing checkpoints / state_dicts remain
# byte-identical.
BGPT = GPT2Decoder

__all__ = ["BGPT", "GPT2Decoder", "GPT2HF", "GPT2HFConfig"]
