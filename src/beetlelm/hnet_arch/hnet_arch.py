"""
src/hnet_arch/hnet_arch.py
──────────────────────────
H-Net architecture wrapper for beetlelm.

The upstream H-Net (``hnet/models/mixer_seq.HNetForCausalLM``) is byte-level
(``vocab_size=256``), hierarchical, and depends on CUDA-only kernels
(``mamba_ssm``, ``flash_attn``, ``causal_conv1d``). We import it lazily inside
``HNetDecoder.__init__`` so the beetle trainer can be imported and unit-tested
on macOS/MPS development boxes without these packages installed — mirroring
``BeetleSSMDecoder``'s ``_require_mamba`` pattern.

Forward signature matches the beetlelm registry contract:
    forward(input_ids, attention_mask=None, past_key_values=None, use_cache=False)
        -> (logits, cache)

The auxiliary H-Net load-balancing loss (one per non-innermost chunking stage)
is summed, scaled by ``model_config.lb_loss_weight``, and stored in
``self._last_aux_losses["hnet/load_balance"]``. The beetle trainer already
accumulates ``model._last_aux_losses`` (see beetle_trainer.py:1254-1260).
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


def _require_hnet() -> None:
    """Raise a clear ImportError if hnet or its CUDA-only deps are missing."""
    missing = []
    for pkg in ("hnet", "mamba_ssm", "flash_attn", "causal_conv1d"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"[hnet_arch] The following packages are required for "
            f"architecture: hnet — {', '.join(missing)}. Install hnet "
            f"(pip install -e ../hnet) plus the CUDA-only deps "
            f"mamba_ssm / flash_attn / causal_conv1d. Existing pico / bgpt / "
            f"beetle_ssm paths are unaffected."
        )


def _as_dict(maybe_obj) -> Dict[str, Any]:
    """Coerce a SimpleNamespace / dict / dataclass-ish thing to a dict."""
    if maybe_obj is None:
        return {}
    if isinstance(maybe_obj, dict):
        return dict(maybe_obj)
    if is_dataclass(maybe_obj) and not isinstance(maybe_obj, type):
        return asdict(maybe_obj)
    if hasattr(maybe_obj, "__dict__"):
        return dict(vars(maybe_obj))
    return dict(maybe_obj)


def _build_hnet_config(model_config) -> "Any":
    """Translate a beetlelm SimpleNamespace into an upstream HNetConfig."""
    from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig

    attn_cfg = AttnConfig(**_as_dict(getattr(model_config, "attn_cfg", {})))
    ssm_cfg = SSMConfig(**_as_dict(getattr(model_config, "ssm_cfg", {})))

    return HNetConfig(
        arch_layout=list(getattr(model_config, "arch_layout", [])),
        d_model=list(getattr(model_config, "d_model", [])),
        d_intermediate=list(getattr(model_config, "d_intermediate", [])),
        vocab_size=int(getattr(model_config, "vocab_size", 256)),
        ssm_cfg=ssm_cfg,
        attn_cfg=attn_cfg,
        tie_embeddings=bool(getattr(model_config, "tie_embeddings", False)),
    )


def _compute_load_balancing_loss(
    bpred_output: List[Any],
    target_downsample: List[float],
) -> torch.Tensor:
    """Sum the per-stage H-Net load-balancing loss.

    ``bpred_output`` is a list with one ``RoutingModuleOutput`` per
    non-innermost chunking stage. ``target_downsample`` is the desired
    chunk-size ``N`` per stage, in the same order.
    """
    from hnet.utils.train import load_balancing_loss

    if not bpred_output:
        # No hierarchical stages → no load-balancing loss (vocab-only or pure
        # isotropic config). Return a scalar zero in the right device/dtype.
        return torch.zeros((), device=_first_param_device(bpred_output))

    if len(target_downsample) < len(bpred_output):
        # Pad with the last value if the user under-specified.
        target_downsample = list(target_downsample) + [target_downsample[-1]] * (
            len(bpred_output) - len(target_downsample)
        )

    total = None
    for stage_out, N in zip(bpred_output, target_downsample):
        loss_s = load_balancing_loss(stage_out, float(N))
        total = loss_s if total is None else total + loss_s
    return total


def _first_param_device(_):
    return torch.device("cpu")


class HNetDecoder(nn.Module):
    """beetlelm-compatible wrapper around ``HNetForCausalLM``.

    YAML ``model.params`` schema (set inside ``configs/tokfree/**/*.yaml``)::

        arch_layout: ["m4", ["T22"], "m4"]
        d_model: [768, 1024]
        d_intermediate: [0, 3072]
        vocab_size: 256
        tie_embeddings: false
        attn_cfg: { num_heads: [12], rotary_emb_dim: [64], window_size: [-1] }
        ssm_cfg:  { d_conv: 4, expand: 2, d_state: 128, chunk_size: 256 }
        initializer_range: 0.02
        lb_loss_weight: 0.03
        lr_multiplier: [3.0, 1.0]
        target_downsample: [4.5]
    """

    def __init__(self, model_config) -> None:
        _require_hnet()
        super().__init__()
        from hnet.models.mixer_seq import HNetForCausalLM

        self.model_config = model_config

        cfg = _build_hnet_config(model_config)
        device = getattr(model_config, "device", None)
        dtype = getattr(model_config, "dtype", None)

        self.hnet = HNetForCausalLM(cfg, device=device, dtype=dtype)

        init_range = float(getattr(model_config, "initializer_range", 0.02))
        self.hnet.init_weights(init_range)

        lr_multiplier = list(getattr(model_config, "lr_multiplier", []))
        if lr_multiplier:
            self.hnet.apply_lr_multiplier(lr_multiplier)

        self.lb_loss_weight = float(getattr(model_config, "lb_loss_weight", 0.03))

        target_ds = getattr(model_config, "target_downsample", None)
        if target_ds is None:
            n_chunking_stages = max(0, len(cfg.arch_layout) - 1)
            target_ds = [4.5] * max(1, n_chunking_stages)
        self.target_downsample = list(target_ds)

        self._last_aux_losses: Dict[str, torch.Tensor] = {}

    def set_working_memory(self, scheduler) -> None:
        return None  # H-Net has no ALiBi-style attention bias hook.

    def update_progress(self, step: int, max_steps: int) -> None:
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        **_,
    ):
        mask = attention_mask
        if mask is not None:
            mask = mask.to(torch.bool)
        else:
            # H-Net's forward switches to packed mode when mask is None, which
            # asserts ``inference_params is None``. The trainer always passes
            # dense batches, so synthesize an all-true mask to stay in the
            # masked-batch code path.
            mask = torch.ones_like(input_ids, dtype=torch.bool)

        out = self.hnet(
            input_ids,
            mask=mask,
            inference_params=past_key_values,
        )

        if out.bpred_output:
            lb = _compute_load_balancing_loss(out.bpred_output, self.target_downsample)
            self._last_aux_losses = {
                "hnet/load_balance": self.lb_loss_weight * lb,
            }
        else:
            self._last_aux_losses = {}

        cache = out.inference_params if use_cache else None
        return out.logits.float(), cache

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kw):
        return self.hnet.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kw)

    def step(self, input_ids: torch.Tensor, inference_params):
        return self.hnet.step(input_ids, inference_params)

    def convert_to_hf_model(self) -> "HNetHF":
        hf_cfg = HNetHFConfig.from_dataclass(self.model_config)
        hf = HNetHF(hf_cfg)
        hf.load_state_dict(self.state_dict())
        return hf


# ── HuggingFace wrapper ──────────────────────────────────────────────────────


class HNetHFConfig(PretrainedConfig):
    model_type = "hnet"

    def __init__(
        self,
        arch_layout: Optional[List] = None,
        d_model: Optional[List[int]] = None,
        d_intermediate: Optional[List[int]] = None,
        vocab_size: int = 256,
        tie_embeddings: bool = False,
        attn_cfg: Optional[Dict[str, Any]] = None,
        ssm_cfg: Optional[Dict[str, Any]] = None,
        initializer_range: float = 0.02,
        lb_loss_weight: float = 0.03,
        lr_multiplier: Optional[List[float]] = None,
        target_downsample: Optional[List[float]] = None,
        max_seq_len: int = 2048,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.arch_layout = arch_layout or ["m4", ["T22"], "m4"]
        self.d_model = d_model or [768, 1024]
        self.d_intermediate = d_intermediate or [0, 3072]
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings
        self.attn_cfg = attn_cfg or {}
        self.ssm_cfg = ssm_cfg or {}
        self.initializer_range = initializer_range
        self.lb_loss_weight = lb_loss_weight
        self.lr_multiplier = lr_multiplier or []
        self.target_downsample = target_downsample or []
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        cfg = cls(**config_dict)
        return_unused = kwargs.pop("return_unused_kwargs", False)
        unused = {k: v for k, v in kwargs.items() if not hasattr(cfg, k)}
        if return_unused:
            return cfg, unused
        return cfg

    @classmethod
    def from_dataclass(cls, model_config) -> "HNetHFConfig":
        if is_dataclass(model_config) and not isinstance(model_config, type):
            d = asdict(model_config)
        elif isinstance(model_config, dict):
            d = dict(model_config)
        elif hasattr(model_config, "__dict__"):
            d = dict(vars(model_config))
        else:
            raise TypeError(
                f"Cannot build HNetHFConfig from {type(model_config).__name__}"
            )
        return cls.from_dict(d)


class HNetHF(PreTrainedModel, GenerationMixin):
    """HuggingFace wrapper for ``HNetDecoder``.

    Usage::

        AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

    Requires the same CUDA-only deps at load time.
    """

    config_class = HNetHFConfig
    _no_split_modules = ["HNetForCausalLM"]
    _tied_weights_keys: List[str] = []

    def __init__(self, config: HNetHFConfig) -> None:
        _require_hnet()
        super().__init__(config)
        from hnet.models.mixer_seq import HNetForCausalLM
        from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig

        hnet_cfg = HNetConfig(
            arch_layout=list(config.arch_layout),
            d_model=list(config.d_model),
            d_intermediate=list(config.d_intermediate),
            vocab_size=config.vocab_size,
            ssm_cfg=SSMConfig(**dict(config.ssm_cfg)),
            attn_cfg=AttnConfig(**dict(config.attn_cfg)),
            tie_embeddings=bool(config.tie_embeddings),
        )
        self.hnet = HNetForCausalLM(hnet_cfg)
        self.lb_loss_weight = float(config.lb_loss_weight)
        self.target_downsample = list(config.target_downsample) or [4.5]
        self.post_init()

    def _init_weights(self, module):
        # H-Net handles its own initialization via init_weights().
        pass

    def get_input_embeddings(self):
        return self.hnet.embeddings

    def set_input_embeddings(self, value):
        self.hnet.embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache: bool = False,
        labels=None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(torch.bool)

        out = self.hnet(input_ids, mask=attention_mask, inference_params=past_key_values)
        logits = out.logits.float()

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().clamp(0, self.config.vocab_size - 1).view(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids, "past_key_values": None, "use_cache": False}


HNetHFConfig.register_for_auto_class()
HNetHF.register_for_auto_class("AutoModel")
HNetHF.register_for_auto_class("AutoModelForCausalLM")
