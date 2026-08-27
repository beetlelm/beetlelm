"""Local pytest configuration for the tokenizer-free test suite.

These tests run on macOS development boxes where:
- ``pytorch_lightning`` is installed but imports can hang or be very slow due
  to torchmetrics / torch.distributed probing.
- CUDA-only deps (``mamba_ssm``, ``flash_attn``, ``causal_conv1d``, ``hnet``)
  are intentionally not installed.

To keep the byte tokenizer / conditions / config tests fast and portable, we
install a lightweight ``pytorch_lightning`` stub into ``sys.modules`` *before*
any beetlelm import triggers ``src/bilingual/data/__init__.py`` (which
transitively imports the real ``pytorch_lightning`` via ``BilingualDataModule``).

The stub provides only the surface beetlelm needs at import time
(``pytorch_lightning.LightningDataModule`` as a base class). Tests that
exercise actual Lightning behaviour belong outside this directory.
"""

from __future__ import annotations

import sys
import types

if "pytorch_lightning" not in sys.modules:
    stub = types.ModuleType("pytorch_lightning")

    class _StubLightningDataModule:  # noqa: D401 - stub surface only
        """Stub of ``pl.LightningDataModule`` for import-time satisfaction."""

        def __init__(self, *args, **kwargs):
            super().__init__()

    stub.LightningDataModule = _StubLightningDataModule
    sys.modules["pytorch_lightning"] = stub
