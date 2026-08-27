"""
src/continual/lora_adapter.py
───────────────────────────────
LoRA (Low-Rank Adaptation) at phase transitions.

v1 policy:
- Auto-detect ``peft`` (``use_peft: auto`` default). If importable, delegates
  to ``peft.LoraConfig`` + ``peft.get_peft_model``.
- Falls back to hand-rolled ``LoRALinear`` if ``peft`` is not installed.
  ``peft`` is NOT a hard runtime dependency.
- ``freeze_backbone: true`` freezes all non-LoRA parameters after apply.

Target modules for pico:  ``q_proj``, ``v_proj``
Target modules for bgpt:  ``c_attn`` (fused QKV)

YAML:

    lora:
      enabled: false
      rank: 8
      alpha: 16
      dropout: 0.05
      target_modules: [q_proj, v_proj]
      freeze_backbone: true
      trigger: phase_change
      use_peft: auto
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Hand-rolled LoRALinear ────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Frozen base weight + low-rank update: y = x @ W.T + x @ (A @ B).T * (alpha/r).

    The base weight is NOT trainable; only A and B are.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features  = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scale = alpha / rank

        # Freeze the original weight.
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.bias   = (
            nn.Parameter(base.bias.data.clone(), requires_grad=False)
            if base.bias is not None else None
        )

        # LoRA matrices: A initialised as Normal(0, 1/sqrt(r)), B as zeros.
        self.lora_A = nn.Parameter(
            torch.randn(rank, base.in_features) / math.sqrt(rank)
        )
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scale
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scale={self.scale:.3f}")


# ── LoRA module manager ───────────────────────────────────────────────────────

class LoRAModule:
    """
    Manages LoRA injection into a model.

    The ``apply`` classmethod walks the model's named modules, replaces every
    ``nn.Linear`` whose name ends with any of the ``target_modules`` strings
    with a ``LoRALinear`` (or peft equivalent), and optionally freezes all
    non-LoRA parameters.
    """

    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        freeze_backbone: bool = True,
        use_peft: str = "auto",
    ):
        self.rank = rank
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.freeze_backbone = freeze_backbone
        self.use_peft = use_peft
        self._is_applied = False
        self._n_replaced = 0

    @property
    def is_applied(self) -> bool:
        return self._is_applied

    def apply(self, model: nn.Module) -> None:
        """Inject LoRA into ``model`` in-place."""
        if self._is_applied:
            logger.warning("[LoRA] apply() called twice — skipping.")
            return

        raw = getattr(model, "module", model)

        if self.use_peft != "never":
            try:
                import peft
                self._apply_peft(raw, peft)
                self._is_applied = True
                return
            except ImportError:
                if self.use_peft == "require":
                    raise
                logger.info("[LoRA] peft not installed — using hand-rolled LoRALinear.")

        self._apply_handrolled(raw)
        self._is_applied = True

    def _apply_peft(self, raw: nn.Module, peft) -> None:
        """Use peft.get_peft_model."""
        cfg = peft.LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias="none",
        )
        peft.get_peft_model(raw, cfg)
        if self.freeze_backbone:
            for name, p in raw.named_parameters():
                if "lora_" not in name:
                    p.requires_grad_(False)
        n_trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        self._n_replaced = len(self.target_modules)
        logger.info(
            "[LoRA] peft applied | target=%s | trainable params: %d",
            self.target_modules, n_trainable,
        )

    def _apply_handrolled(self, raw: nn.Module) -> None:
        """Replace matching nn.Linear modules with LoRALinear."""
        n_replaced = 0
        for parent_name, parent_module in list(raw.named_modules()):
            for attr_name, child in list(parent_module.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                # Match if the attr_name ends with any target module suffix.
                if any(attr_name == t or attr_name.endswith("." + t)
                       for t in self.target_modules):
                    lora_layer = LoRALinear(child, self.rank, self.alpha, self.dropout)
                    setattr(parent_module, attr_name, lora_layer)
                    n_replaced += 1

        self._n_replaced = n_replaced
        if self.freeze_backbone:
            for name, p in raw.named_parameters():
                if not ("lora_A" in name or "lora_B" in name):
                    p.requires_grad_(False)
        n_trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        logger.info(
            "[LoRA] hand-rolled applied | %d layers replaced | "
            "target=%s | trainable params: %d",
            n_replaced, self.target_modules, n_trainable,
        )

    # ─── Persistence ──────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "freeze_backbone": self.freeze_backbone,
            "use_peft": self.use_peft,
            "is_applied": self._is_applied,
            "n_replaced": self._n_replaced,
        }

    def load_state_dict(self, state: dict) -> None:
        self.rank            = int(state.get("rank", self.rank))
        self.alpha           = float(state.get("alpha", self.alpha))
        self.dropout         = float(state.get("dropout", self.dropout))
        self.target_modules  = state.get("target_modules", self.target_modules)
        self.freeze_backbone = bool(state.get("freeze_backbone", self.freeze_backbone))
        self.use_peft        = state.get("use_peft", self.use_peft)
        self._is_applied     = bool(state.get("is_applied", False))
        self._n_replaced     = int(state.get("n_replaced", 0))

    # ─── Diagnostics ──────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        return {
            "lora/is_applied": 1.0 if self._is_applied else 0.0,
            "lora/n_replaced": float(self._n_replaced),
            "lora/rank": float(self.rank),
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def build_lora(cfg: dict) -> Optional["LoRAModule"]:
    """
    Build a LoRAModule from the ``lora:`` YAML block.

    Returns None if absent or ``enabled: false``.
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    return LoRAModule(
        rank=int(cfg.get("rank", 8)),
        alpha=float(cfg.get("alpha", 16.0)),
        dropout=float(cfg.get("dropout", 0.05)),
        target_modules=list(cfg.get("target_modules", ["q_proj", "v_proj"])),
        freeze_backbone=bool(cfg.get("freeze_backbone", True)),
        use_peft=str(cfg.get("use_peft", "auto")),
    )
