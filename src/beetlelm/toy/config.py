"""
Configuration schema for the minimal ``beetlelm.toy`` training loop.

A run is described by a single YAML file with four blocks: ``model``,
``tokenizer``, ``data`` and ``training``. This loads that YAML into plain
dataclasses with sensible defaults, so a config only needs to override what
differs from the tiny-CPU baseline. Unknown keys are ignored, which keeps toy
configs compatible with the fuller research configs under ``configs/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    """PicoDecoder hyper-parameters. ``vocab_size`` is fixed by the byte
    tokenizer (256) at build time, so it can be omitted from the YAML."""

    n_layers: int = 4
    d_model: int = 128
    vocab_size: int = 256
    attention_n_heads: int = 4
    attention_n_kv_heads: int = 4
    max_seq_len: int = 128
    batch_size: int = 16
    position_emb_theta: float = 10000.0
    activation_hidden_dim: int = 256
    norm_eps: float = 1e-5
    dropout: float = 0.0


@dataclass
class TokenizerConfig:
    # "byte" (256-way UTF-8) or "codepoint". Both are download-free.
    type: str = "byte"
    max_length: int = 128


@dataclass
class DataConfig:
    # Path to a plain-text corpus, resolved relative to the config file's dir
    # (or the repo root) when not absolute.
    train_file: str = "data/toy_corpus.txt"
    val_fraction: float = 0.1


@dataclass
class TrainingConfig:
    max_steps: int = 50
    batch_size: int = 16
    seq_len: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.01
    warmup_steps: int = 5
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 5
    eval_every: int = 25
    device: str = "cpu"  # "cpu", "cuda", or "mps"
    output_dir: str = "runs/tiny_cpu"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    run_name: str = "beetlelm-tiny"
    config_dir: str = "."

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("config_dir", None)
        return d


def _merge(dc_type, overrides: Dict[str, Any]):
    known = set(dc_type.__dataclass_fields__)
    filtered = {k: v for k, v in (overrides or {}).items() if k in known}
    return dc_type(**filtered)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config`.

    ``model.params`` is accepted as an alias for ``model`` so the schema stays
    compatible with the fuller research configs.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}

    model_block = raw.get("model", {}) or {}
    model_params = model_block.get("params", model_block)

    return Config(
        model=_merge(ModelConfig, model_params),
        tokenizer=_merge(TokenizerConfig, raw.get("tokenizer", {})),
        data=_merge(DataConfig, raw.get("data", {})),
        training=_merge(TrainingConfig, raw.get("training", {})),
        run_name=raw.get("run_name", "beetlelm-tiny"),
        config_dir=str(path.resolve().parent),
    )
