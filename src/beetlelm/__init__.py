"""
beetlelm — a framework for transparently training bilingual language models and
storing comprehensive checkpoints for learning-dynamics research.

Importing the package is intentionally light: the heavy training stack
(Lightning, datasets, wandb, HF Hub, the architecture zoo) is pulled in only
when you touch the relevant submodule or run the ``beetlelm-train`` CLI.

Two defaults are set here for safety:
  * Weights & Biases is forced offline unless the user opts in.
  * The HuggingFace tokenizers fork-parallelism warning is silenced.

Newcomers can start with the download-free CPU quickstart in ``beetlelm.toy``::

    from beetlelm.toy import train
    train("configs/tiny_cpu.yaml")
"""

from __future__ import annotations

import os

# Never phone home by default; the user can override these explicitly.
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"

__all__ = ["__version__"]
