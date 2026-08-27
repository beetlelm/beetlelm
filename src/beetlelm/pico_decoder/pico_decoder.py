"""
Pico Decoder: A Lightweight Causal Transformer Language Model
Implementation from https://github.com/pico-lm/pico-train/blob/main/src/model/pico_decoder.py

Key features:
- RMSNorm for layer normalization
- Rotary Positional Embeddings (RoPE)
- Multi-head attention with KV-cache support
- SwiGLU activation function
- Residual connections throughout
- KV-cache for faster autoregressive generation

References:
    - RoPE: https://arxiv.org/abs/2104.09864
    - SwiGLU: https://arxiv.org/abs/2002.05202
    - LLAMA: https://arxiv.org/abs/2302.13971

HuggingFace compatibility notes
---------------------------------
PicoDecoderHF stores weights at the TOP LEVEL (embedding_proj, layers,
output_norm, de_embedding_proj) so that state dict keys match the raw
PicoDecoder checkpoint format exactly. Do NOT add a self.pico_decoder
wrapper — it would prepend a key prefix that does not exist in any saved
checkpoint, causing every weight to be MISSING on load.

vocab_size in config.json must be the BASE BPE vocabulary size
(tokenizer.vocab_size), NOT the padded len(tokenizer) which includes
<unusedN> padding tokens. The embedding table was built with the base size.
"""

from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast
try:
    if TYPE_CHECKING:
        from beetlelm.config import ModelConfig
except ImportError:
    pass


class RMSNorm(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps    = config.norm_eps
        self.weight = nn.Parameter(torch.ones(config.d_model))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class RoPE(nn.Module):
    """
    Rotary Position Embedding.
    freqs_cis is computed lazily on first use and cached per-device,
    avoiding meta-tensor issues when HF loads with low_cpu_mem_usage=True.
    The cache auto-extends if a call needs a longer sequence than the
    current cached length (grows geometrically).
    """
    def __init__(self, config):
        super().__init__()
        self.theta   = config.position_emb_theta
        self.dim     = config.d_model // config.attention_n_heads
        self.max_seq = config.max_seq_len
        # NOT a buffer — plain dict so it never touches the meta device
        self._cache: Dict[torch.device, torch.Tensor] = {}
        self._cache_max_seq: Dict[torch.device, int] = {}

    def _build_cache(self, device: torch.device, length: int) -> torch.Tensor:
        freqs = 1.0 / (
            self.theta ** (
                torch.arange(0, self.dim, 2, device=device).float() / self.dim
            )
        )
        t = torch.arange(length, device=device)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def _get_freqs_cis(self, device: torch.device, min_length: int = 0) -> torch.Tensor:
        cached_len = self._cache_max_seq.get(device, 0)
        needed = max(min_length, self.max_seq)
        if device not in self._cache or cached_len < needed:
            new_len = max(needed, cached_len * 2)
            self._cache[device] = self._build_cache(device, new_len)
            self._cache_max_seq[device] = new_len
        return self._cache[device]

    def get_freqs_cis(self, input_shape, start_pos, end_pos, device):
        _f   = self._get_freqs_cis(device, min_length=end_pos)[start_pos:end_pos]
        ndim = len(input_shape)
        if ndim < 2:
            raise ValueError(
                f"RoPE expects input with ndim >= 2, got shape {tuple(input_shape)}"
            )
        expected = (input_shape[1], input_shape[-1])
        if tuple(_f.shape) != expected:
            raise ValueError(
                f"RoPE freqs_cis shape mismatch: got {tuple(_f.shape)}, "
                f"expected {expected} (input_shape={tuple(input_shape)}, "
                f"start_pos={start_pos}, end_pos={end_pos}, "
                f"cached_max_seq={self._cache_max_seq.get(device)}, dim={self.dim})"
            )
        return _f.view(*[d if i == 1 or i == ndim - 1 else 1
                         for i, d in enumerate(input_shape)])

    def forward(self, queries, keys, start_pos=0):
        device = queries.device
        q_ = torch.view_as_complex(queries.float().reshape(*queries.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(keys.float().reshape(*keys.shape[:-1], -1, 2))
        fc  = self.get_freqs_cis(q_.shape, start_pos, start_pos + q_.shape[1], device)
        return (torch.view_as_real(q_ * fc).flatten(3).type_as(queries),
                torch.view_as_real(k_ * fc).flatten(3).type_as(keys))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads     = config.attention_n_heads
        self.n_kv_heads  = config.attention_n_kv_heads
        self.batch_size  = config.batch_size
        self.max_seq_len = config.max_seq_len
        d = config.d_model
        self.head_dim = d // self.n_heads
        self.n_rep    = self.n_heads // self.n_kv_heads
        self.q_proj = nn.Linear(d, self.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d,    bias=False)
        self.rope   = RoPE(config)
        # Optional working-memory hook.  Set by PicoDecoder.set_working_memory().
        # When present, it must be a callable returning a float tensor of shape
        # (1, n_heads, seq_len, seq_len) to be added to the attention mask.
        self._working_memory_hook = None
    def forward(self, input, mask=None, past_key_values=None, use_cache=False):
        bsz, seq_len, _ = input.shape
        queries = self.q_proj(input).view(bsz, seq_len, self.n_heads,    self.head_dim)
        keys    = self.k_proj(input).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        values  = self.v_proj(input).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        start_pos = past_key_values[0].shape[1] if past_key_values is not None else 0
        queries, keys = self.rope(queries, keys, start_pos)
        if past_key_values is not None:
            keys   = torch.cat([past_key_values[0], keys],   dim=1)
            values = torch.cat([past_key_values[1], values], dim=1)
        cached_keys   = keys   if use_cache else None
        cached_values = values if use_cache else None
        queries = queries.transpose(1, 2)
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)
        apply_gqa = self.n_rep > 1
        if apply_gqa and queries.device.type == "mps":
            keys   = keys.repeat_interleave(self.n_rep, dim=-3)
            values = values.repeat_interleave(self.n_rep, dim=-3)
            apply_gqa = False
        attn_mask = mask.to(queries.dtype) if mask is not None else None
        if self._working_memory_hook is not None:
            key_len = keys.shape[-2]
            alibi_bias = self._working_memory_hook(
                key_len, queries.device, queries.dtype
            )
            # Crop / broadcast to (1, n_heads, q_len, k_len).
            alibi_bias = alibi_bias[:, :, key_len - seq_len:key_len, :key_len]
            if attn_mask is None:
                attn_mask = alibi_bias.expand(bsz, -1, -1, -1)
            else:
                # attn_mask may be (q_len, k_len) — broadcast to 4D first.
                while attn_mask.dim() < 4:
                    attn_mask = attn_mask.unsqueeze(0)
                attn_mask = attn_mask + alibi_bias
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
            attn_output = F.scaled_dot_product_attention(
                queries.contiguous(), keys.contiguous(), values.contiguous(),
                attn_mask=attn_mask, enable_gqa=apply_gqa,
            )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output), (cached_keys, cached_values)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_0 = nn.Linear(config.d_model, config.activation_hidden_dim, bias=False)
        self.w_1 = nn.Linear(config.d_model, config.activation_hidden_dim, bias=False)
        self.w_2 = nn.Linear(config.activation_hidden_dim, config.d_model, bias=False)
    def forward(self, x):
        return self.w_2(F.silu(self.w_0(x)) * self.w_1(x))


class PicoDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention      = Attention(config)
        self.swiglu         = SwiGLU(config)
        self.attention_norm = RMSNorm(config)
        self.swiglu_norm    = RMSNorm(config)
    def forward(self, input, mask=None, past_key_values=None, use_cache=False):
        attention_output, cached_key_values = self.attention(
            self.attention_norm(input), mask=mask,
            past_key_values=past_key_values, use_cache=use_cache)
        h = input + attention_output
        return h + self.swiglu(self.swiglu_norm(h)), cached_key_values


class PicoDecoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config            = model_config
        self.embedding_proj    = nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.layers            = nn.ModuleList(
            [PicoDecoderBlock(model_config) for _ in range(model_config.n_layers)])
        self.output_norm       = RMSNorm(model_config)
        self.de_embedding_proj = nn.Linear(
            model_config.d_model, model_config.vocab_size, bias=False)
        # Optional working-memory scheduler (see src/working_memory).  When
        # absent, the forward pass is identical to the vanilla PicoDecoder.
        self._working_memory = None

    # ──────────────────────────────────────────────────────────────────────
    # Optional working-memory ALiBi hook (see src/working_memory)
    # ──────────────────────────────────────────────────────────────────────
    def set_working_memory(self, scheduler) -> None:
        """Attach a ``WorkingMemoryScheduler``; pass ``None`` to detach."""
        self._working_memory = scheduler
        hook = (lambda n, d, t: scheduler.bias(n, d, t)) if scheduler is not None else None
        for layer in self.layers:
            layer.attention._working_memory_hook = hook

    def update_progress(self, step: int, max_steps: int) -> None:
        """Tick the working-memory scheduler (no-op if not attached)."""
        if self._working_memory is not None:
            self._working_memory.step(step, max_steps)

    def convert_to_hf_model(self):
        # Fabric/DDP wrappers expose the inner module via .module; unwrap so
        # state_dict() never carries a "module." prefix that PicoDecoderHF
        # cannot match. Without this, strict=False below silently dropped
        # every weight, save_pretrained wrote an empty checkpoint without a
        # valid model_type key, and the HF-smoke round-trip raised
        # "Unrecognized model in step_0".
        raw = getattr(self, "module", self)
        hf = PicoDecoderHF(PicoDecoderHFConfig.from_dataclass(raw.config))
        # Strip auxiliary heads added by continual-learning modules (MAML's
        # classifier_head, etc.). They are not part of the published HF
        # architecture; loading them would raise "Unexpected key(s)" in
        # PicoDecoderHF.load_state_dict and crash the per-step save.
        backbone_state = {
            k: v for k, v in raw.state_dict().items()
            if not k.startswith("classifier_head")
        }
        missing, unexpected = hf.load_state_dict(backbone_state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"PicoDecoder.convert_to_hf_model: {len(unexpected)} unexpected "
                f"keys in state_dict: {sorted(unexpected)[:5]}…"
            )
        return hf
    def forward(self, input_ids, past_key_values=None, use_cache=False,
                return_hidden=False):
        """
        return_hidden=True returns (hidden_states, cached_key_values) where
        hidden_states is the final-layer post-norm representation (pre-
        de_embedding_proj), used by MAML's classifier head and any other
        consumer that needs a feature view of the model rather than logits.
        Default behaviour is unchanged.
        """
        seq_len   = input_ids.shape[-1]
        h         = self.embedding_proj(input_ids)
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
                h, mask=mask, past_key_values=layer_past, use_cache=use_cache)
            if use_cache:
                cached_key_values += (layer_cached,)
        hidden = self.output_norm(h)
        if return_hidden:
            return hidden, cached_key_values
        return self.de_embedding_proj(hidden).float(), cached_key_values


class PicoDecoderHFConfig(PretrainedConfig):
    model_type = "pico_decoder"
    def __init__(self,
                 n_layers=14, d_model=768, vocab_size=32768,
                 attention_n_heads=12, attention_n_kv_heads=1,
                 max_seq_len=512, batch_size=64, position_emb_theta=10000.0,
                 activation_hidden_dim=3072, norm_eps=1e-5, dropout=0.1,
                 **kwargs):
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
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PicoDecoderHFConfig":
        pico_config = cls(**config_dict)
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        unused_kwargs = {k: v for k, v in kwargs.items() if not hasattr(pico_config, k)}
        if return_unused_kwargs:
            return pico_config, unused_kwargs
        return pico_config
    @classmethod
    def from_dataclass(cls, model_config):
        if is_dataclass(model_config) and not isinstance(model_config, type):
            d = asdict(model_config)
        elif isinstance(model_config, dict):
            d = dict(model_config)
        elif hasattr(model_config, "__dict__"):
            d = dict(vars(model_config))
        else:
            raise TypeError(
                f"Cannot build PicoDecoderHFConfig from {type(model_config).__name__}"
            )
        return cls.from_dict(d)


class PicoDecoderHF(PreTrainedModel, GenerationMixin):
    """
    HuggingFace wrapper for BeetleLM PicoDecoder.
    Usage: AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
    Works with CPU, CUDA (A100, etc.), and MPS out of the box.
    """
    config_class       = PicoDecoderHFConfig
    _no_split_modules  = ["PicoDecoderBlock"]
    _tied_weights_keys = []

    def __init__(self, config: PicoDecoderHFConfig):
        super().__init__(config)
        self.embedding_proj    = nn.Embedding(config.vocab_size, config.d_model)
        self.layers            = nn.ModuleList(
            [PicoDecoderBlock(config) for _ in range(config.n_layers)])
        self.output_norm       = RMSNorm(config)
        self.de_embedding_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Required: lets HF finalize weight init and meta-device materialization
        self.post_init()

    # Required for low_cpu_mem_usage / Accelerate device-dispatch to work
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

    def forward(self, input_ids=None, past_key_values=None,
                use_cache=False, labels=None, **kwargs):
        seq_len   = input_ids.shape[-1]
        h         = self.embedding_proj(input_ids)
        start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            if past_key_values is not None:
                mask = torch.hstack([torch.zeros((seq_len, start_pos), device=h.device), mask])
        cached_key_values = () if use_cache else None
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            h, layer_cached = layer(
                h, mask=mask, past_key_values=layer_past, use_cache=use_cache)
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
                loss=loss, logits=logits, past_key_values=cached_key_values)
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids,
                "past_key_values": past_key_values,
                "use_cache": True}


PicoDecoderHFConfig.register_for_auto_class()
PicoDecoderHF.register_for_auto_class("AutoModel")
PicoDecoderHF.register_for_auto_class("AutoModelForCausalLM")
