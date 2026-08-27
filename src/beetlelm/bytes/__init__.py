"""
Tokenizer-free representation layer for beetlelm.

Provides byte-level and codepoint-level managers that satisfy the surface the
beetle trainer expects of a tokenizer (callable returning ``input_ids`` /
``attention_mask`` tensor dicts, plus ``vocab_size`` / ``pad_token_id`` /
``bos_token_id`` / ``eos_token_id`` / ``__len__`` / ``save_pretrained``).

Registering a new tokenizer-free representation: append it to
``TOKFREE_TOKENIZERS`` below. The trainer branches on this dict in
``src/training/beetle_trainer.py::_setup_tokenizer``.
"""

from __future__ import annotations

from beetlelm.bytes.byte_tokenizer import (
    ByteTokenizerManager,
    CodepointTokenizerManager,
)
from beetlelm.bytes.streaming_dataset import StreamingTokFreeDataset

TOKFREE_TOKENIZERS = {
    "byte": ByteTokenizerManager,
    "codepoint": CodepointTokenizerManager,
}

__all__ = [
    "ByteTokenizerManager",
    "CodepointTokenizerManager",
    "StreamingTokFreeDataset",
    "TOKFREE_TOKENIZERS",
]
