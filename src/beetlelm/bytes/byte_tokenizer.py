"""
Byte- and codepoint-level tokenizer managers for tokenizer-free training.

Surface mirrors ``MultilingualTokenizerManager`` enough that the beetle trainer
can use either interchangeably: callable returning ``{"input_ids",
"attention_mask"}``, plus ``vocab_size``, ``pad_token_id``, ``bos_token_id``,
``eos_token_id``, ``__len__``, ``encode``, ``decode``, ``save_pretrained``,
``from_pretrained``.

PAD handling
------------
For ``ByteTokenizerManager`` we keep ``vocab_size=256`` and use ``pad_id=0``
(the NUL byte). NUL is rare in natural text and the model is expected to
respect ``attention_mask``. H-Net masks ``boundary_mask`` by ``mask`` in
``hnet/modules/dc.py``, so padded positions don't contribute to the
load-balancing loss.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch


BYTE_VOCAB_SIZE = 256
BYTE_PAD_ID = 0
BYTE_BOS_ID = 254
BYTE_EOS_ID = 255


class ByteTokenizerManager:
    """UTF-8 byte tokenizer with the beetlelm tokenizer-manager surface."""

    def __init__(
        self,
        max_length: int = 2048,
        add_bos: bool = True,
        add_eos: bool = False,
        pad_token_id: int = BYTE_PAD_ID,
        bos_token_id: int = BYTE_BOS_ID,
        eos_token_id: int = BYTE_EOS_ID,
        **_unused,
    ) -> None:
        self.max_length = int(max_length)
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = BYTE_VOCAB_SIZE

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> List[int]:
        b = text.encode("utf-8", errors="replace")
        ids = list(b)
        if self.add_bos:
            ids = [self.bos_token_id] + ids
        if self.add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: Sequence[int], errors: str = "replace") -> str:
        bytes_only = [int(i) for i in ids if 0 <= int(i) < 254]
        return bytes(bytes_only).decode("utf-8", errors=errors)

    def __call__(
        self,
        texts: Union[str, Sequence[str]],
        padding: Union[bool, str] = "max_length",
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        max_len = int(max_length) if max_length is not None else self.max_length

        encoded: List[List[int]] = []
        masks: List[List[int]] = []
        for t in texts:
            ids = self.encode(t)
            if truncation and len(ids) > max_len:
                ids = ids[:max_len]
            mask = [1] * len(ids)
            if padding and len(ids) < max_len:
                pad_n = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_n
                mask = mask + [0] * pad_n
            encoded.append(ids)
            masks.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        if return_tensors == "np":
            return {
                "input_ids": np.asarray(encoded, dtype=np.int64),
                "attention_mask": np.asarray(masks, dtype=np.int64),
            }
        return {"input_ids": encoded, "attention_mask": masks}

    def save_pretrained(self, path: Union[str, os.PathLike]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        cfg = {
            "type": "byte",
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "max_length": self.max_length,
        }
        (p / "tokfree_tokenizer.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def from_pretrained(cls, path: Union[str, os.PathLike]) -> "ByteTokenizerManager":
        cfg_path = Path(path) / "tokfree_tokenizer.json"
        cfg = json.loads(cfg_path.read_text())
        cfg.pop("type", None)
        cfg.pop("vocab_size", None)
        return cls(**cfg)


DEFAULT_CODEPOINT_VOCAB = 0x20000  # covers BMP + supplementary plane 1
CODEPOINT_PAD_ID = 0
CODEPOINT_BOS_ID = 1
CODEPOINT_EOS_ID = 2
CODEPOINT_OFFSET = 3  # shift real codepoints so 0/1/2 are reserved for special tokens


class CodepointTokenizerManager:
    """Unicode codepoint tokenizer with the beetlelm tokenizer-manager surface.

    Real codepoints ``c`` are shifted by ``CODEPOINT_OFFSET`` so the special
    tokens (pad/bos/eos) occupy ids 0/1/2 without colliding with real text.
    Codepoints whose shifted id exceeds ``vocab_size`` are mapped to ``pad_id``.
    """

    def __init__(
        self,
        max_length: int = 2048,
        vocab_size: int = DEFAULT_CODEPOINT_VOCAB,
        add_bos: bool = True,
        add_eos: bool = False,
        pad_token_id: int = CODEPOINT_PAD_ID,
        bos_token_id: int = CODEPOINT_BOS_ID,
        eos_token_id: int = CODEPOINT_EOS_ID,
        offset: int = CODEPOINT_OFFSET,
        **_unused,
    ) -> None:
        self.max_length = int(max_length)
        self.vocab_size = int(vocab_size)
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.offset = int(offset)

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for ch in text:
            shifted = ord(ch) + self.offset
            ids.append(shifted if shifted < self.vocab_size else self.pad_token_id)
        if self.add_bos:
            ids = [self.bos_token_id] + ids
        if self.add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: Sequence[int], errors: str = "replace") -> str:
        out: List[str] = []
        for i in ids:
            v = int(i) - self.offset
            if 0 <= v < 0x110000 and int(i) >= self.offset:
                try:
                    out.append(chr(v))
                except ValueError:
                    pass
        return "".join(out)

    def __call__(
        self,
        texts: Union[str, Sequence[str]],
        padding: Union[bool, str] = "max_length",
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        max_len = int(max_length) if max_length is not None else self.max_length

        encoded: List[List[int]] = []
        masks: List[List[int]] = []
        for t in texts:
            ids = self.encode(t)
            if truncation and len(ids) > max_len:
                ids = ids[:max_len]
            mask = [1] * len(ids)
            if padding and len(ids) < max_len:
                pad_n = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_n
                mask = mask + [0] * pad_n
            encoded.append(ids)
            masks.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        if return_tensors == "np":
            return {
                "input_ids": np.asarray(encoded, dtype=np.int64),
                "attention_mask": np.asarray(masks, dtype=np.int64),
            }
        return {"input_ids": encoded, "attention_mask": masks}

    def save_pretrained(self, path: Union[str, os.PathLike]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        cfg = {
            "type": "codepoint",
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "offset": self.offset,
            "max_length": self.max_length,
        }
        (p / "tokfree_tokenizer.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def from_pretrained(cls, path: Union[str, os.PathLike]) -> "CodepointTokenizerManager":
        cfg_path = Path(path) / "tokfree_tokenizer.json"
        cfg = json.loads(cfg_path.read_text())
        cfg.pop("type", None)
        return cls(**cfg)
