"""
beetlelm.toy — a minimal, download-free CPU training path for newcomers.

This is a thin, self-contained loop that trains the real BeetleLM PicoDecoder
(``beetlelm.pico_decoder``) with the real byte tokenizer (``beetlelm.bytes``)
on a small in-repo corpus. It shares the model, tokenizer, and checkpoint
format with the full trainer but skips the cluster / sweep / continual-learning
machinery so a run finishes on a laptop CPU in minutes with no downloads and
only the light dependencies (torch, transformers, numpy, pyyaml).

The full research trainer lives in ``beetlelm.training`` and is driven by the
``beetlelm-train`` CLI.
"""

from beetlelm.toy.config import Config, load_config
from beetlelm.toy.trainer import train

__all__ = ["train", "Config", "load_config"]
