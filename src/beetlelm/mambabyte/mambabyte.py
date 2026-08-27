"""
src/mambabyte/mambabyte.py
──────────────────────────
MambaByte: byte-level Mamba SSM baseline.

A direct subclass of ``BeetleSSMDecoder`` that defends ``vocab_size=256``.
Used as a tokenizer-free baseline against H-Net in the R-condition sweeps.

Forward signature inherited from ``BeetleSSMDecoder``:
    forward(input_ids, past_key_values=None, use_cache=False) -> (logits, cache)
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from beetlelm.beetle_ssm.beetle_ssm import BeetleSSMDecoder

logger = logging.getLogger(__name__)


class MambaByteDecoder(BeetleSSMDecoder):
    """Byte-level Mamba SSM decoder.

    Differs from ``BeetleSSMDecoder`` only in that ``vocab_size`` is forced
    to 256. Any YAML that sets ``vocab_size`` to something other than 256
    will be silently snapped.
    """

    def __init__(self, model_config) -> None:
        if not isinstance(model_config, SimpleNamespace):
            try:
                cfg = SimpleNamespace(**dict(vars(model_config)))
            except TypeError:
                cfg = model_config
        else:
            cfg = model_config

        existing_vocab = getattr(cfg, "vocab_size", None)
        if existing_vocab is not None and int(existing_vocab) != 256:
            logger.warning(
                "[mambabyte] vocab_size=%s in YAML; forcing 256 for byte-level "
                "training. Set tokenizer.type: byte in your config.",
                existing_vocab,
            )
        cfg.vocab_size = 256
        super().__init__(cfg)
