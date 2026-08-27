"""
src/beetle_ssm/beetle_ssm.py
──────────────────────────────
BeetleSSM: Mamba state-space decoder.

The mamba_ssm package is imported **lazily** inside ``BeetleSSMDecoder.__init__``
so that the beetle trainer can be imported and unit-tested on machines that do
not have ``mamba_ssm`` installed (e.g. macOS/MPS development boxes).

Only ``architecture: beetle_ssm`` YAMLs trigger the import; all existing
pico / bgpt paths are unaffected.

Forward signature identical to PicoDecoder:
    forward(input_ids, past_key_values=None, use_cache=False) -> (logits, cache)

YAML extra params under model.params:
    d_state: 16
    d_conv: 4
    expand: 2
    dt_min: 0.001
    dt_max: 0.1
"""
from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

logger = logging.getLogger(__name__)

_MAMBA_ERROR: Optional[str] = None


def _require_mamba():
    """Raise a clear ImportError if mamba_ssm is not available."""
    try:
        import mamba_ssm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "[beetle_ssm] The 'mamba-ssm' package is required to use "
            "architecture: beetle_ssm. Install it with:\n"
            "  pip install mamba-ssm\n"
            "Existing pico/bgpt paths are unaffected."
        ) from e


class BeetleSSMBlock(nn.Module):
    """Single Mamba block + layer norm residual."""

    def __init__(self, config):
        super().__init__()
        from mamba_ssm.modules.mamba_simple import Mamba
        d_model = config.d_model
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=getattr(config, "d_state", 16),
            d_conv=getattr(config, "d_conv", 4),
            expand=getattr(config, "expand", 2),
            dt_min=getattr(config, "dt_min", 0.001),
            dt_max=getattr(config, "dt_max", 0.1),
        )

    def forward(self, x, **_kwargs):
        return x + self.mamba(self.norm(x)), None


class BeetleSSMDecoder(nn.Module):
    """
    Mamba state-space decoder compatible with the BeetleLM trainer registry.

    ``set_working_memory`` / ``update_progress`` are no-op stubs because the
    SSM has no attention heads to bias.  The trainer raises a config error if
    ``working_memory.enabled: true`` is combined with this architecture.
    """

    def __init__(self, model_config):
        _require_mamba()
        super().__init__()
        self.model_config = model_config

        self.embedding_proj    = nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.layers            = nn.ModuleList(
            [BeetleSSMBlock(model_config) for _ in range(model_config.n_layers)]
        )
        self.output_norm       = nn.LayerNorm(model_config.d_model)
        self.de_embedding_proj = nn.Linear(model_config.d_model, model_config.vocab_size, bias=False)

        self._last_aux_losses: Dict[str, torch.Tensor] = {}

    def set_working_memory(self, scheduler) -> None:
        pass   # No-op: SSM has no attention bias hooks.

    def update_progress(self, step: int, max_steps: int) -> None:
        pass   # No-op.

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        h = self.embedding_proj(input_ids)
        for layer in self.layers:
            h, _ = layer(h)
        logits = self.de_embedding_proj(self.output_norm(h)).float()
        return logits, None

    def convert_to_hf_model(self) -> "BeetleSSMHF":
        """Wrap this decoder as a HuggingFace ``PreTrainedModel`` for save_pretrained / push_to_hub."""
        hf = BeetleSSMHF(BeetleSSMHFConfig.from_dataclass(self.model_config))
        hf.load_state_dict(self.state_dict())
        return hf


# ── HuggingFace wrapper ──────────────────────────────────────────────────────
#
# Mirrors the PicoDecoderHF pattern. Layers are ``BeetleSSMBlock``s so the
# state-dict matches ``BeetleSSMDecoder`` exactly. Requires ``mamba-ssm`` at
# load time (raised lazily by ``BeetleSSMBlock``).

class BeetleSSMHFConfig(PretrainedConfig):
    model_type = "beetle_ssm"

    def __init__(
        self,
        n_layers: int = 14,
        d_model: int = 768,
        vocab_size: int = 50000,
        max_seq_len: int = 512,
        batch_size: int = 64,
        norm_eps: float = 1e-5,
        dropout: float = 0.1,
        # SSM-specific
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        # Carried for parity with pico configs (unused by SSM forward).
        attention_n_heads: int = 12,
        attention_n_kv_heads: int = 1,
        position_emb_theta: float = 10000.0,
        activation_hidden_dim: int = 3072,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_layers              = n_layers
        self.d_model               = d_model
        self.vocab_size            = vocab_size
        self.max_seq_len           = max_seq_len
        self.batch_size            = batch_size
        self.norm_eps              = norm_eps
        self.dropout               = dropout
        self.d_state               = d_state
        self.d_conv                = d_conv
        self.expand                = expand
        self.dt_min                = dt_min
        self.dt_max                = dt_max
        self.attention_n_heads     = attention_n_heads
        self.attention_n_kv_heads  = attention_n_kv_heads
        self.position_emb_theta    = position_emb_theta
        self.activation_hidden_dim = activation_hidden_dim

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        cfg = cls(**config_dict)
        return_unused = kwargs.pop("return_unused_kwargs", False)
        unused = {k: v for k, v in kwargs.items() if not hasattr(cfg, k)}
        if return_unused:
            return cfg, unused
        return cfg

    @classmethod
    def from_dataclass(cls, model_config) -> "BeetleSSMHFConfig":
        if is_dataclass(model_config) and not isinstance(model_config, type):
            d = asdict(model_config)
        elif isinstance(model_config, dict):
            d = dict(model_config)
        elif hasattr(model_config, "__dict__"):
            d = dict(vars(model_config))
        else:
            raise TypeError(
                f"Cannot build BeetleSSMHFConfig from {type(model_config).__name__}"
            )
        return cls.from_dict(d)


class BeetleSSMHF(PreTrainedModel, GenerationMixin):
    """HuggingFace wrapper for ``BeetleSSMDecoder``.

    Usage: ``AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)``.
    Requires ``mamba-ssm`` to be installed at load time.
    """

    config_class       = BeetleSSMHFConfig
    _no_split_modules  = ["BeetleSSMBlock"]
    _tied_weights_keys: List[str] = []

    def __init__(self, config: BeetleSSMHFConfig):
        _require_mamba()
        super().__init__(config)
        self.embedding_proj    = nn.Embedding(config.vocab_size, config.d_model)
        self.layers            = nn.ModuleList(
            [BeetleSSMBlock(config) for _ in range(config.n_layers)]
        )
        self.output_norm       = nn.LayerNorm(config.d_model)
        self.de_embedding_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

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
        h = self.embedding_proj(input_ids)
        for layer in self.layers:
            h, _ = layer(h)
        logits = self.de_embedding_proj(self.output_norm(h)).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().clamp(0, self.config.vocab_size - 1).view(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids, "past_key_values": None, "use_cache": False}


BeetleSSMHFConfig.register_for_auto_class()
BeetleSSMHF.register_for_auto_class("AutoModel")
BeetleSSMHF.register_for_auto_class("AutoModelForCausalLM")
