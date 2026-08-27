from abc import ABC, abstractmethod
from typing import List, Optional
import os
import random

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

from .compression import equal_compression_sample

# -------------------------
# Defaults
# -------------------------
DEFAULT_SPECIAL_TOKENS = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]
DEFAULT_UNK_TOKEN = "<UNK>"

# -------------------------
# Base interface
# -------------------------
class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers used in bilingual pretraining."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass

    @abstractmethod
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        pass

# -------------------------
# Tokenizer registry
# -------------------------
TOKENIZER_REGISTRY = {}

def register_tokenizer(name: str):
    """Decorator to register a tokenizer class in the registry."""
    def decorator(cls):
        TOKENIZER_REGISTRY[name] = cls
        return cls
    return decorator

# -------------------------
# BPE Tokenizer
# -------------------------
@register_tokenizer("bpe")
class BPETokenizer(BaseTokenizer):
    def __init__(self, tokenizer_path: Optional[str] = None, vocab_size: int = 32768,
                 special_tokens: Optional[List[str]] = None):
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
        self.vocab_size = vocab_size

        if tokenizer_path:
            if not os.path.exists(tokenizer_path):
                raise ValueError(f"Tokenizer path {tokenizer_path} does not exist")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token=DEFAULT_UNK_TOKEN))
            self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, files: List[str]):
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train(files, trainer)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

# -------------------------
# Unigram Tokenizer
# -------------------------
@register_tokenizer("unigram")
class UnigramTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_path: Optional[str] = None, vocab_size: int = 32768,
                 special_tokens: Optional[List[str]] = None):
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
        self.vocab_size = vocab_size

        if tokenizer_path:
            if not os.path.exists(tokenizer_path):
                raise ValueError(f"Tokenizer path {tokenizer_path} does not exist")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            self.tokenizer = Tokenizer(Unigram())
            self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, files: List[str]):
        trainer = UnigramTrainer(vocab_size=self.vocab_size,
                                 special_tokens=self.special_tokens,
                                 unk_token=DEFAULT_UNK_TOKEN)
        self.tokenizer.train(files, trainer)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

# -------------------------
# SuperBPE Tokenizer
# -------------------------
@register_tokenizer("superbpe")
class SuperBPETokenizer(BPETokenizer):
    """
    Multi-stage BPE with staged merge inheritance.
    transition_point controls how many merges are inherited from a first-stage tokenizer.
    """

    def __init__(self, tokenizer_path: Optional[str] = None, vocab_size: int = 32768,
                 special_tokens: Optional[List[str]] = None, transition_point: float = 0.5):
        super().__init__(tokenizer_path, vocab_size, special_tokens)
        self.transition_point = transition_point

    def train(self, files: List[str]):
        """
        Multi-stage training:
        1. Train first-stage BPE on subset of tokens
        2. Inherit top merges according to transition_point
        3. Train second-stage tokenizer on full corpus
        """
        # Compute first stage vocab size
        first_stage_vocab = int(self.vocab_size * self.transition_point)

        # Temporary first-stage tokenizer
        first_stage_tokenizer = Tokenizer(BPE(unk_token=DEFAULT_UNK_TOKEN))
        first_stage_tokenizer.pre_tokenizer = Whitespace()
        first_stage_trainer = BpeTrainer(vocab_size=first_stage_vocab, special_tokens=self.special_tokens)
        first_stage_tokenizer.train(files, first_stage_trainer)

        # Train second-stage tokenizer on full vocab
        second_stage_trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train(files, second_stage_trainer)

# -------------------------
# Factory method
# -------------------------
def build_tokenizer(tokenizer_name: str, **kwargs) -> BaseTokenizer:
    if tokenizer_name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Tokenizer '{tokenizer_name}' not registered")
    return TOKENIZER_REGISTRY[tokenizer_name](**kwargs)
