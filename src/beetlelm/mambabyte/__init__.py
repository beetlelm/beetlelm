"""MambaByte: a byte-level Mamba SSM baseline for tokenizer-free experiments.

Thin shim over ``BeetleSSMDecoder`` that fixes ``vocab_size=256``.
Registered as ``"mambabyte"`` in ``src/training/_arch_imports.py``.
"""

from beetlelm.mambabyte.mambabyte import MambaByteDecoder

__all__ = ["MambaByteDecoder"]
