"""
src/beetle_moe/beetle_moe.py
──────────────────────────────
BeetleMoE: Mixture-of-Experts transformer decoder.

Reuses Attention, RoPE, RMSNorm from PicoDecoder; replaces the SwiGLU FFN
with a top-k routed MoE FFN (ST-MoE style: load-balance loss + z-loss).

Aux losses are stored on ``model._last_aux_losses`` so the trainer can read
them from ``_compute_loss`` without modifying the forward signature.

Forward signature identical to PicoDecoder:
    forward(input_ids, past_key_values=None, use_cache=False) -> (logits, cache)

YAML params (in addition to standard pico params):
    moe_num_experts: 8
    moe_top_k: 2
    moe_capacity_factor: 1.25
    moe_router_jitter: 0.0
    moe_load_balance_weight: 0.01
    moe_router_z_loss_weight: 0.001
    moe_target_modules: [ffn]   # only "ffn" supported in v1
"""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast

# Reuse primitives from pico_decoder
from beetlelm.pico_decoder.pico_decoder import Attention, RMSNorm, RoPE


# ── Router ────────────────────────────────────────────────────────────────────

class MoERouter(nn.Module):
    """Linear router with optional jitter noise (training-time only)."""

    def __init__(self, d_model: int, num_experts: int, jitter: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.jitter = jitter
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (router_logits, router_probs), both (*, num_experts)."""
        logits = self.gate(x)  # (B, T, E)
        if self.training and self.jitter > 0:
            logits = logits + torch.randn_like(logits) * self.jitter
        return logits, F.softmax(logits, dim=-1)


# ── Expert FFN ────────────────────────────────────────────────────────────────

class ExpertFFN(nn.Module):
    """Single SwiGLU expert (identical to PicoDecoder SwiGLU)."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_0 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(F.silu(self.w_0(x)) * self.w_1(x))


# ── MoE FFN layer ─────────────────────────────────────────────────────────────

class SparseMoEFFN(nn.Module):
    """
    Top-k sparse MoE FFN with ST-MoE auxiliary losses.

    Aux losses stored on ``self.load_balance_loss`` and
    ``self.router_z_loss`` after each forward call.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float = 1.25,
        jitter: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = MoERouter(d_model, num_experts, jitter)
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, hidden_dim) for _ in range(num_experts)]
        )
        # Populated after each forward; read by trainer.
        self.load_balance_loss: torch.Tensor = torch.tensor(0.0)
        self.router_z_loss:     torch.Tensor = torch.tensor(0.0)
        # Expert dispatch counts for routing diagnostics (B*T assignments).
        self.expert_counts: torch.Tensor = torch.zeros(num_experts, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)   # (N, D) where N = B*T
        N = x_flat.shape[0]

        router_logits, router_probs = self.router(x_flat)  # (N, E)

        # ── Top-k selection ──────────────────────────────────────────────
        topk_vals, topk_ids = router_probs.topk(self.top_k, dim=-1)  # (N, k)
        # Re-normalise selected weights to sum to 1 per token.
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        # ── Aux losses ───────────────────────────────────────────────────
        # Load-balance loss (ST-MoE, Eq. 4):
        #   mean_expert_load * mean_router_prob  (summed over experts)
        expert_load = torch.zeros(self.num_experts, device=x.device)
        expert_load.scatter_add_(
            0,
            topk_ids.view(-1),
            torch.ones(N * self.top_k, device=x.device),
        )
        expert_load_frac = expert_load / (N * self.top_k)
        mean_router_prob = router_probs.mean(dim=0)
        lb_loss = (self.num_experts * (expert_load_frac * mean_router_prob).sum())
        self.load_balance_loss = lb_loss

        # Router z-loss: penalises large logits (improves training stability).
        z_loss = (router_logits.exp().sum(dim=-1).log() ** 2).mean()
        self.router_z_loss = z_loss

        # Expert counts for diagnostics.
        self.expert_counts = expert_load.detach().long().cpu()

        # ── Dispatch and combine ──────────────────────────────────────────
        out = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            # Mask of tokens assigned to this expert (any of the k slots).
            mask = (topk_ids == expert_id).any(dim=-1)  # (N,)
            if not mask.any():
                continue
            token_inputs = x_flat[mask]
            expert_out = self.experts[expert_id](token_inputs)

            # Weighted contribution of this expert for each selected token.
            # Weight = the router probability of this expert for that token.
            expert_weights = torch.where(
                topk_ids[mask] == expert_id,
                topk_weights[mask],
                torch.zeros_like(topk_weights[mask]),
            ).sum(dim=-1, keepdim=True)   # (n_selected, 1)
            out[mask] += expert_weights * expert_out

        return out.view(B, T, D)


# ── BeetleMoE decoder block ───────────────────────────────────────────────────

class BeetleMoEConfig:
    """Thin wrapper so we can pass extra MoE params through SimpleNamespace."""
    pass


class BeetleMoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention      = Attention(config)
        self.moe_ffn        = SparseMoEFFN(
            d_model=config.d_model,
            hidden_dim=config.activation_hidden_dim,
            num_experts=getattr(config, "moe_num_experts", 8),
            top_k=getattr(config, "moe_top_k", 2),
            capacity_factor=getattr(config, "moe_capacity_factor", 1.25),
            jitter=getattr(config, "moe_router_jitter", 0.0),
        )
        self.attention_norm = RMSNorm(config)
        self.ffn_norm       = RMSNorm(config)

    def forward(self, input, mask=None, past_key_values=None, use_cache=False):
        attn_out, cached = self.attention(
            self.attention_norm(input),
            mask=mask, past_key_values=past_key_values, use_cache=use_cache,
        )
        h = input + attn_out
        return h + self.moe_ffn(self.ffn_norm(h)), cached


# ── BeetleMoEDecoder ─────────────────────────────────────────────────────────

class BeetleMoEDecoder(nn.Module):
    """
    Mixture-of-Experts causal transformer decoder.

    Matches PicoDecoder's external API so it slots into the registry without
    trainer changes beyond reading ``model._last_aux_losses`` in
    ``_compute_loss``.
    """

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

        self.embedding_proj    = nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.layers            = nn.ModuleList(
            [BeetleMoEBlock(model_config) for _ in range(model_config.n_layers)]
        )
        self.output_norm       = RMSNorm(model_config)
        self.de_embedding_proj = nn.Linear(
            model_config.d_model, model_config.vocab_size, bias=False
        )

        # Populated by forward(); read by trainer after each step.
        self._last_aux_losses: Dict[str, torch.Tensor] = {}
        self._working_memory = None

    def convert_to_hf_model(self) -> "BeetleMoEHF":
        """Wrap this decoder as a HuggingFace ``PreTrainedModel`` for save_pretrained / push_to_hub."""
        hf = BeetleMoEHF(BeetleMoEHFConfig.from_dataclass(self.model_config))
        hf.load_state_dict(self.state_dict())
        return hf

    # --- API compatibility stubs ---
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
        total_lb   = torch.tensor(0.0, device=h.device)
        total_z    = torch.tensor(0.0, device=h.device)

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            h, layer_cached = layer(
                h, mask=mask, past_key_values=layer_past, use_cache=use_cache
            )
            if use_cache:
                cached_key_values += (layer_cached,)
            total_lb = total_lb + layer.moe_ffn.load_balance_loss
            total_z  = total_z  + layer.moe_ffn.router_z_loss

        n_layers = len(self.layers)
        lb_weight = getattr(self.model_config, "moe_load_balance_weight", 0.01)
        z_weight  = getattr(self.model_config, "moe_router_z_loss_weight", 0.001)
        self._last_aux_losses = {
            "moe/load_balance_loss": lb_weight  * total_lb / n_layers,
            "moe/router_z_loss":     z_weight   * total_z  / n_layers,
        }

        logits = self.de_embedding_proj(self.output_norm(h)).float()
        return logits, cached_key_values


# ── HuggingFace wrapper ──────────────────────────────────────────────────────
#
# Mirrors the PicoDecoderHF pattern (see src/pico_decoder/pico_decoder.py).
# Top-level state-dict keys are identical to ``BeetleMoEDecoder`` so
# ``hf.load_state_dict(decoder.state_dict())`` round-trips losslessly.

class BeetleMoEHFConfig(PretrainedConfig):
    model_type = "beetle_moe"

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
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        moe_router_jitter: float = 0.0,
        moe_load_balance_weight: float = 0.01,
        moe_router_z_loss_weight: float = 0.001,
        **kwargs: Any,
    ) -> None:
        if not attention_n_kv_heads:
            attention_n_kv_heads = attention_n_heads
        super().__init__(**kwargs)
        self.n_layers                 = n_layers
        self.d_model                  = d_model
        self.vocab_size               = vocab_size
        self.attention_n_heads        = attention_n_heads
        self.attention_n_kv_heads     = attention_n_kv_heads
        self.max_seq_len              = max_seq_len
        self.batch_size               = batch_size
        self.position_emb_theta       = position_emb_theta
        self.activation_hidden_dim    = activation_hidden_dim
        self.norm_eps                 = norm_eps
        self.dropout                  = dropout
        self.moe_num_experts          = moe_num_experts
        self.moe_top_k                = moe_top_k
        self.moe_capacity_factor      = moe_capacity_factor
        self.moe_router_jitter        = moe_router_jitter
        self.moe_load_balance_weight  = moe_load_balance_weight
        self.moe_router_z_loss_weight = moe_router_z_loss_weight

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        cfg = cls(**config_dict)
        return_unused = kwargs.pop("return_unused_kwargs", False)
        unused = {k: v for k, v in kwargs.items() if not hasattr(cfg, k)}
        if return_unused:
            return cfg, unused
        return cfg

    @classmethod
    def from_dataclass(cls, model_config) -> "BeetleMoEHFConfig":
        if is_dataclass(model_config) and not isinstance(model_config, type):
            d = asdict(model_config)
        elif isinstance(model_config, dict):
            d = dict(model_config)
        elif hasattr(model_config, "__dict__"):
            d = dict(vars(model_config))
        else:
            raise TypeError(
                f"Cannot build BeetleMoEHFConfig from {type(model_config).__name__}"
            )
        return cls.from_dict(d)


class BeetleMoEHF(PreTrainedModel, GenerationMixin):
    """HuggingFace wrapper for ``BeetleMoEDecoder``.

    Usage: ``AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)``.
    """

    config_class       = BeetleMoEHFConfig
    _no_split_modules  = ["BeetleMoEBlock"]
    _tied_weights_keys: List[str] = []

    def __init__(self, config: BeetleMoEHFConfig):
        super().__init__(config)
        self.embedding_proj    = nn.Embedding(config.vocab_size, config.d_model)
        self.layers            = nn.ModuleList(
            [BeetleMoEBlock(config) for _ in range(config.n_layers)]
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


BeetleMoEHFConfig.register_for_auto_class()
BeetleMoEHF.register_for_auto_class("AutoModel")
BeetleMoEHF.register_for_auto_class("AutoModelForCausalLM")
