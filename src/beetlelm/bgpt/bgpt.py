import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast

########################################################
# Layer Normalization (Standard GPT-2 uses LayerNorm)
########################################################

class GPT2LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, x):
        return self.norm(x)

########################################################
# MLP (Standard GPT-2 uses GELU + 2 Linear Layers)
########################################################

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.activation_hidden_dim)
        self.c_proj = nn.Linear(config.activation_hidden_dim, config.d_model)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

########################################################
# Attention (GPT-2 Multi-Head Attention)
########################################################

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.attention_n_heads
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.n_heads

        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model)
        self.c_proj = nn.Linear(self.d_model, self.d_model)
        # Optional working-memory hook (see src/working_memory).  Set by
        # GPT2Decoder.set_working_memory().  When present, contributes an ALiBi
        # bias to the attention scores.
        self._working_memory_hook = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape

        # GPT-2 style: compute Q, K, V in one linear layer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present = (k, v) if use_cache else None

        attn_mask = mask
        is_causal = mask is None
        if self._working_memory_hook is not None:
            key_len = k.shape[-2]
            alibi_bias = self._working_memory_hook(key_len, q.device, q.dtype)
            alibi_bias = alibi_bias[:, :, key_len - seq_len:key_len, :key_len]
            # SDPA can't combine `is_causal=True` with an additive mask — build
            # an explicit causal float mask and fold ALiBi into it.
            causal = torch.zeros(
                (seq_len, key_len), device=q.device, dtype=q.dtype
            )
            causal_bool = torch.ones((seq_len, key_len), device=q.device, dtype=torch.bool)
            causal_bool = torch.tril(causal_bool, diagonal=key_len - seq_len)
            causal = causal.masked_fill(~causal_bool, float("-inf"))
            attn_mask = causal.view(1, 1, seq_len, key_len) + alibi_bias
            is_causal = False

        # Scaled Dot-Product Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0 if not self.training else 0.1,
            is_causal=is_causal # If no mask provided, assume causal during training
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        output = self.c_proj(attn_output)

        return output, present

########################################################
# GPT2 Block
########################################################

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x, mask=None, past_key_values=None, use_cache=False):
        # Pre-norm architecture
        attn_out, present = self.attn(self.ln_1(x), mask=mask, past_key_values=past_key_values, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

########################################################
# GPT2 Decoder
########################################################

class GPT2Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model) # Learned Positional Embeddings

        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Optional working-memory scheduler — see src/working_memory.  When
        # absent, the forward pass is identical to the vanilla GPT2Decoder.
        self._working_memory = None

    # ──────────────────────────────────────────────────────────────────────
    # Optional working-memory ALiBi hook (see src/working_memory)
    # ──────────────────────────────────────────────────────────────────────
    def set_working_memory(self, scheduler) -> None:
        """Attach a ``WorkingMemoryScheduler``; pass ``None`` to detach."""
        self._working_memory = scheduler
        hook = (lambda n, d, t: scheduler.bias(n, d, t)) if scheduler is not None else None
        for block in self.h:
            block.attn._working_memory_hook = hook

    def update_progress(self, step: int, max_steps: int) -> None:
        """Tick the working-memory scheduler (no-op if not attached)."""
        if self._working_memory is not None:
            self._working_memory.step(step, max_steps)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        bsz, seq_len = input_ids.shape
        start_pos = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        
        # Positions for WPE
        device = input_ids.device
        position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=device)
        
        # Combine Token and Position Embeddings
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)

        # Causal mask logic for SDPA
        mask = None
        if seq_len > 1:
            # Standard causal mask
            mask = torch.ones((seq_len, seq_len + start_pos), device=device, dtype=torch.bool)
            mask = torch.tril(mask, diagonal=start_pos)
            # Convert to float mask for SDPA if needed, or keep bool
            
        presents = () if use_cache else None
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = block(hidden_states, mask=None, past_key_values=layer_past, use_cache=use_cache)
            if use_cache:
                presents += (present,)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, presents

########################################################
# HuggingFace Wrapper
########################################################

class GPT2HFConfig(PretrainedConfig):
    model_type = "gpt2_pico"
    def __init__(
        self,
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        attention_n_heads=12,
        activation_hidden_dim=3072,
        max_seq_len=1024,
        layer_norm_epsilon=1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.attention_n_heads = attention_n_heads
        self.activation_hidden_dim = activation_hidden_dim
        self.max_seq_len = max_seq_len
        self.layer_norm_epsilon = layer_norm_epsilon

class GPT2HF(PreTrainedModel):
    config_class = GPT2HFConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Decoder(config)

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        logits, past_key_values = self.transformer(input_ids, past_key_values, use_cache)
        if use_cache:
            return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
        return CausalLMOutput(logits=logits)


# Public alias so callers can `from beetlelm.bgpt.bgpt import BGPT`. The trainer
# dispatch and the architecture registry both rely on this name.
BGPT = GPT2Decoder
