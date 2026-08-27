"""
src/geo_bilingual/geo_bilingual.py
────────────────────────────────────
Geometry-aware bilingual decoder (``geo_bilingual`` architecture).

Initialises from two monolingual checkpoints and a pre-computed Procrustes
rotation matrix R; joint embedding is built as ``R_align @ E_a`` merged
with ``E_b`` via block_init strategy.

Forward signature identical to PicoDecoder:
    forward(input_ids, past_key_values=None, use_cache=False) -> (logits, cache)

YAML params under model.params:
    mono_a_checkpoint: path/to/mono_a.pt
    mono_b_checkpoint: path/to/mono_b.pt
    rotation_path: path/to/R.npy
    block_init: average    # average | stack
    n_layers: 14
    d_model: 768
    ...  (all standard pico params)
"""
from __future__ import annotations

import copy
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast

from beetlelm.pico_decoder.pico_decoder import (
    PicoDecoder,
    PicoDecoderBlock,
    RMSNorm,
)

logger = logging.getLogger(__name__)


class GeoBilingualDecoder(nn.Module):
    """
    Bilingual decoder with geometry-aware initialisation.

    If checkpoint paths and rotation are given, the model is initialised from
    two monolingual PicoDecoder checkpoints with block averaging or stacking
    and Procrustes-aligned embeddings.  If paths are absent (e.g. during
    unit tests or registry checks), the model falls back to standard random
    PicoDecoder initialisation.
    """

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

        # Build a base PicoDecoder architecture.
        base = PicoDecoder(model_config)

        mono_a_path = getattr(model_config, "mono_a_checkpoint", None)
        mono_b_path = getattr(model_config, "mono_b_checkpoint", None)
        rotation_path = getattr(model_config, "rotation_path", None)
        block_init = getattr(model_config, "block_init", "average")

        if mono_a_path and mono_b_path:
            logger.info(
                "[GeoBilingual] Loading mono_a from %s, mono_b from %s",
                mono_a_path, mono_b_path,
            )
            self._init_from_checkpoints(
                base, mono_a_path, mono_b_path, rotation_path, block_init
            )
        else:
            logger.info(
                "[GeoBilingual] No checkpoint paths — using random PicoDecoder init."
            )

        # Copy the (possibly modified) base architecture.
        self.embedding_proj    = base.embedding_proj
        self.layers            = base.layers
        self.output_norm       = base.output_norm
        self.de_embedding_proj = base.de_embedding_proj
        self._working_memory   = None

    # ─── Initialisation ───────────────────────────────────────────────────

    def _init_from_checkpoints(
        self,
        base: PicoDecoder,
        path_a: str,
        path_b: str,
        rotation_path: Optional[str],
        block_init: str,
    ) -> None:
        sd_a = torch.load(path_a, map_location="cpu")
        sd_b = torch.load(path_b, map_location="cpu")

        # Strip Fabric/DDP prefix if present.
        sd_a = _strip_prefix(sd_a)
        sd_b = _strip_prefix(sd_b)

        model_a = copy.deepcopy(base)
        model_b = copy.deepcopy(base)
        model_a.load_state_dict(sd_a, strict=False)
        model_b.load_state_dict(sd_b, strict=False)

        # ── Embedding alignment ─────────────────────────────────────────
        E_a = model_a.embedding_proj.weight.detach().float()   # (V, D)
        E_b = model_b.embedding_proj.weight.detach().float()

        if rotation_path is not None:
            R = torch.tensor(np.load(rotation_path), dtype=torch.float32)
            # Rotate E_a into E_b's space: E_a_aligned = E_a @ R
            E_a_aligned = E_a @ R
        else:
            E_a_aligned = E_a

        if block_init == "average":
            E_joint = (E_a_aligned + E_b[:E_a_aligned.shape[0]]) / 2.0
        else:  # stack: keep E_a_aligned, defer E_b to fine-tuning
            E_joint = E_a_aligned

        base.embedding_proj.weight.data.copy_(
            E_joint[:base.embedding_proj.weight.shape[0]]
        )

        # ── Block initialisation ────────────────────────────────────────
        if block_init == "average":
            _average_layers(base, model_a, model_b)
        elif block_init == "stack":
            _copy_layers(base, model_a)
        else:
            raise ValueError(f"[GeoBilingual] Unknown block_init={block_init!r}.")

        logger.info(
            "[GeoBilingual] Initialised from checkpoints | block_init=%s | "
            "rotation=%s", block_init, rotation_path is not None
        )

    # ─── Forward ──────────────────────────────────────────────────────────

    def set_working_memory(self, scheduler) -> None:
        self._working_memory = scheduler
        hook = (lambda n, d, t: scheduler.bias(n, d, t)) if scheduler is not None else None
        for layer in self.layers:
            layer.attention._working_memory_hook = hook

    def update_progress(self, step: int, max_steps: int) -> None:
        if self._working_memory is not None:
            self._working_memory.step(step, max_steps)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        seq_len = input_ids.shape[-1]
        h = self.embedding_proj(input_ids)
        start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            if past_key_values is not None:
                mask = torch.hstack([torch.zeros((seq_len, start_pos)), mask])
            mask = mask.to(h.device)

        cached_key_values = () if use_cache else None
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            h, layer_cached = layer(
                h, mask=mask, past_key_values=layer_past, use_cache=use_cache
            )
            if use_cache:
                cached_key_values += (layer_cached,)

        logits = self.de_embedding_proj(self.output_norm(h)).float()
        return logits, cached_key_values

    def convert_to_hf_model(self) -> "GeoBilingualHF":
        """Wrap this decoder as a HuggingFace ``PreTrainedModel`` for save_pretrained / push_to_hub."""
        hf = GeoBilingualHF(GeoBilingualHFConfig.from_dataclass(self.model_config))
        hf.load_state_dict(self.state_dict())
        return hf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix added by DDP/Fabric."""
    new = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new[k] = v
    return new


def _average_layers(target: nn.Module, model_a: nn.Module, model_b: nn.Module) -> None:
    """Average weights of matching parameters from model_a and model_b into target.

    Iterates keys in sorted order so the resulting state_dict is bit-identical
    across resumes regardless of dict insertion order in older Python builds.
    Multi-seed ablations rely on this for cross-run comparability.
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    target_sd = target.state_dict()
    for key in sorted(target_sd.keys()):
        if key in sd_a and key in sd_b:
            if sd_a[key].shape == sd_b[key].shape == target_sd[key].shape:
                target_sd[key] = (sd_a[key].float() + sd_b[key].float()) / 2.0
    target.load_state_dict(target_sd, strict=False)


def _copy_layers(target: nn.Module, source: nn.Module) -> None:
    """Copy layer weights from source to target."""
    target.load_state_dict(source.state_dict(), strict=False)


# ── HuggingFace wrapper ──────────────────────────────────────────────────────
#
# Mirrors PicoDecoderHF. Layers are ``PicoDecoderBlock``s so the state-dict
# matches ``GeoBilingualDecoder`` (which builds the same blocks underneath)
# exactly. Path-bearing init kwargs (mono_a_checkpoint / mono_b_checkpoint /
# rotation_path) are intentionally excluded — they belong to training-time
# init and are meaningless after save_pretrained.

class GeoBilingualHFConfig(PretrainedConfig):
    model_type = "geo_bilingual"

    def __init__(
        self,
        n_layers: int = 14,
        d_model: int = 768,
        vocab_size: int = 50000,
        attention_n_heads: int = 12,
        attention_n_kv_heads: int = 1,
        max_seq_len: int = 512,
        batch_size: int = 64,
        position_emb_theta: float = 10000.0,
        activation_hidden_dim: int = 3072,
        norm_eps: float = 1e-5,
        dropout: float = 0.1,
        block_init: str = "average",
        **kwargs: Any,
    ) -> None:
        if not attention_n_kv_heads:
            attention_n_kv_heads = attention_n_heads
        super().__init__(**kwargs)
        self.n_layers              = n_layers
        self.d_model               = d_model
        self.vocab_size            = vocab_size
        self.attention_n_heads     = attention_n_heads
        self.attention_n_kv_heads  = attention_n_kv_heads
        self.max_seq_len           = max_seq_len
        self.batch_size            = batch_size
        self.position_emb_theta    = position_emb_theta
        self.activation_hidden_dim = activation_hidden_dim
        self.norm_eps              = norm_eps
        self.dropout               = dropout
        self.block_init            = block_init

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        # Strip path-bearing fields that are training-time-only.
        cleaned = {
            k: v for k, v in config_dict.items()
            if k not in ("mono_a_checkpoint", "mono_b_checkpoint", "rotation_path")
        }
        cfg = cls(**cleaned)
        return_unused = kwargs.pop("return_unused_kwargs", False)
        unused = {k: v for k, v in kwargs.items() if not hasattr(cfg, k)}
        if return_unused:
            return cfg, unused
        return cfg

    @classmethod
    def from_dataclass(cls, model_config) -> "GeoBilingualHFConfig":
        if is_dataclass(model_config) and not isinstance(model_config, type):
            d = asdict(model_config)
        elif isinstance(model_config, dict):
            d = dict(model_config)
        elif hasattr(model_config, "__dict__"):
            d = dict(vars(model_config))
        else:
            raise TypeError(
                f"Cannot build GeoBilingualHFConfig from {type(model_config).__name__}"
            )
        return cls.from_dict(d)


class GeoBilingualHF(PreTrainedModel, GenerationMixin):
    """HuggingFace wrapper for ``GeoBilingualDecoder``.

    Usage: ``AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)``.
    """

    config_class       = GeoBilingualHFConfig
    _no_split_modules  = ["PicoDecoderBlock"]
    _tied_weights_keys: List[str] = []

    def __init__(self, config: GeoBilingualHFConfig):
        super().__init__(config)
        self.embedding_proj    = nn.Embedding(config.vocab_size, config.d_model)
        self.layers            = nn.ModuleList(
            [PicoDecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.output_norm       = RMSNorm(config)
        self.de_embedding_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def get_input_embeddings(self):        return self.embedding_proj
    def set_input_embeddings(self, value): self.embedding_proj = value

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        use_cache: bool = False,
        labels=None,
        **kwargs,
    ):
        seq_len   = input_ids.shape[-1]
        h         = self.embedding_proj(input_ids)
        start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            if past_key_values is not None:
                mask = torch.hstack(
                    [torch.zeros((seq_len, start_pos), device=h.device), mask]
                )

        cached_key_values = () if use_cache else None
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            h, layer_cached = layer(
                h, mask=mask, past_key_values=layer_past, use_cache=use_cache,
            )
            if use_cache:
                cached_key_values += (layer_cached,)

        logits = self.de_embedding_proj(self.output_norm(h)).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().clamp(0, self.config.vocab_size - 1).view(-1),
            )
        if use_cache:
            return CausalLMOutputWithPast(
                loss=loss, logits=logits, past_key_values=cached_key_values,
            )
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }


GeoBilingualHFConfig.register_for_auto_class()
GeoBilingualHF.register_for_auto_class("AutoModel")
GeoBilingualHF.register_for_auto_class("AutoModelForCausalLM")
