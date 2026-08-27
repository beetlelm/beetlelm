"""
src/continual/kl_distillation.py
────────────────────────────────
KL distillation from a frozen L1 teacher.

A middle ground between EWC (which constrains parameter drift) and
generative replay (which constrains output distribution on synthetic
data). Here we constrain the *student's* output distribution toward the
*teacher's* on the **current L2 batch**: at the L1→L2 phase boundary we
deep-copy the model, freeze it, and during L2 training add

    L_KL = T² · KL( softmax(s/T) || softmax(t/T) )

to the language-modelling loss (Hinton et al. 2015 distillation form).
``T² ·`` keeps the gradient magnitude scale-invariant w.r.t. temperature.

Usage:

    from beetlelm.continual.kl_distillation import build_kl_distillation

    kl = build_kl_distillation(cfg)            # None if disabled
    kl.snapshot(model)                         # at L1→L2 boundary
    # inside training loop:
    if kl and kl.is_ready:
        loss = lm_loss + kl.kl_term(student_model, input_ids)
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16":    torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16":    torch.bfloat16,
    "float32": torch.float32,
    "fp32":    torch.float32,
}


class KLDistillationModule:
    """
    KL-distillation regulariser with a frozen teacher snapshotted at
    phase change.

    Parameters
    ----------
    weight : float
        Multiplier λ on the KL term. Final term added to loss:
        ``λ · T² · KL(softmax(s/T) || softmax(t/T))``.
    temperature : float
        Distillation softmax temperature T. T=1 reduces to a standard
        soft-target KL on the model's natural distribution.
    teacher_dtype : str | None
        If given, the teacher is cast to this dtype after deep-copy
        (``float16``/``bfloat16``/``float32``). Saves memory; the student
        keeps its training dtype.
    apply_to_all_phases : bool
        If True, the KL term is computed on every batch once the teacher
        is ready. If False (default), the trainer is expected to gate the
        call by phase (``current_phase != prev_phase`` heuristic stays
        fully driven by the trainer; the module itself just exposes
        ``is_ready``).
    """

    def __init__(
        self,
        weight: float = 1.0,
        temperature: float = 2.0,
        teacher_dtype: Optional[str] = "float16",
        apply_to_all_phases: bool = False,
    ):
        if temperature <= 0:
            raise ValueError(f"[KL-Distill] temperature must be > 0, got {temperature}")
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.teacher_dtype = teacher_dtype
        self.apply_to_all_phases = bool(apply_to_all_phases)

        self._teacher: Optional[torch.nn.Module] = None
        self._is_ready: bool = False
        self._last_term: float = 0.0   # for diagnostics

    # ─── Public API ──────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self._teacher is not None

    def snapshot(self, model: torch.nn.Module) -> None:
        """
        Deep-copy the current model into a frozen teacher.

        Call this at the L1→L2 phase boundary, **before** any L2 data
        reaches the model.
        """
        raw = _unwrap(model)
        teacher = copy.deepcopy(raw)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        if self.teacher_dtype is not None:
            dtype = _DTYPE_MAP.get(self.teacher_dtype.lower())
            if dtype is None:
                raise ValueError(
                    f"[KL-Distill] Unknown teacher_dtype {self.teacher_dtype!r}. "
                    f"Allowed: {sorted(_DTYPE_MAP)}"
                )
            teacher.to(dtype=dtype)

        self._teacher = teacher
        self._is_ready = True
        logger.info(
            "[KL-Distill] Teacher snapshot taken | dtype=%s | weight=%.3f | T=%.2f",
            self.teacher_dtype, self.weight, self.temperature,
        )

    def kl_term(
        self,
        student_model: torch.nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ``λ · T² · KL(softmax(s/T) || softmax(t/T))`` on the given
        batch.

        Returns a zero scalar on the student's device when not ready
        (caller can always do ``loss = loss + kl.kl_term(...)``).
        """
        student_raw = _unwrap(student_model)
        dev = next(student_raw.parameters()).device

        if not self.is_ready or self.weight == 0.0:
            return torch.tensor(0.0, device=dev)

        # Vocab-clamp inputs (mirrors trainer's clamp; teacher and student
        # must see the same token ids).
        V = int(getattr(student_raw.model_config, "vocab_size", 32000))
        x = input_ids[:, :-1].clamp(0, V - 1).to(dev)

        # Student forward — track grads.
        s_out = student_raw(x)
        s_logits = s_out[0] if isinstance(s_out, (tuple, list)) else s_out

        # Teacher forward — no grads, no autocast surprises (cast input
        # to teacher's parameter dtype if it differs, e.g. fp16 teacher).
        teacher = self._teacher
        teacher.to(dev)
        with torch.no_grad():
            t_out = teacher(x)
            t_logits = t_out[0] if isinstance(t_out, (tuple, list)) else t_out
        t_logits = t_logits.to(s_logits.dtype)

        T = self.temperature
        # KL( P_s || P_t ) using F.kl_div which expects log-prob first arg
        # and prob second arg with ``reduction='batchmean'`` summing over
        # the last dim then averaging by batch (B*Tseq).
        log_p_s = F.log_softmax(s_logits / T, dim=-1)
        p_t     = F.softmax(t_logits / T, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean")

        term = self.weight * (T * T) * kl
        # Stash for diagnostics (cheap detach).
        try:
            self._last_term = float(term.detach().item())
        except Exception:
            self._last_term = 0.0
        return term

    # ─── Persistence ─────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """
        Serialisable state. The teacher's parameters are saved on CPU.
        """
        teacher_state = None
        if self._teacher is not None:
            teacher_state = {
                k: v.detach().cpu()
                for k, v in self._teacher.state_dict().items()
            }
        return {
            "weight": self.weight,
            "temperature": self.temperature,
            "teacher_dtype": self.teacher_dtype,
            "apply_to_all_phases": self.apply_to_all_phases,
            "is_ready": self._is_ready,
            "teacher_state": teacher_state,
        }

    def load_state_dict(self, state: dict, *, model: Optional[torch.nn.Module] = None) -> None:
        """
        Restore from a previously saved state_dict.

        ``model`` is required if ``is_ready`` was True at save time — it
        provides the architecture skeleton onto which we load the saved
        teacher parameters. We deep-copy the live model to avoid
        depending on internal constructor signatures, then overwrite its
        parameters with the saved tensors.
        """
        self.weight = float(state.get("weight", self.weight))
        self.temperature = float(state.get("temperature", self.temperature))
        self.teacher_dtype = state.get("teacher_dtype", self.teacher_dtype)
        self.apply_to_all_phases = bool(state.get("apply_to_all_phases", self.apply_to_all_phases))

        teacher_state = state.get("teacher_state")
        was_ready = bool(state.get("is_ready", False))

        if was_ready and teacher_state is not None:
            if model is None:
                logger.warning(
                    "[KL-Distill] state_dict says is_ready=True but no live model "
                    "passed to load_state_dict — teacher will be re-snapshotted "
                    "at next phase change instead."
                )
                self._teacher = None
                self._is_ready = False
                return
            raw = _unwrap(model)
            teacher = copy.deepcopy(raw)
            teacher.load_state_dict(teacher_state)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            if self.teacher_dtype is not None:
                dtype = _DTYPE_MAP.get(self.teacher_dtype.lower(), torch.float32)
                teacher.to(dtype=dtype)
            self._teacher = teacher
            self._is_ready = True
            logger.info("[KL-Distill] Teacher restored from checkpoint.")
        else:
            self._teacher = None
            self._is_ready = False

    # ─── Diagnostics ─────────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, float]:
        return {
            "kl_distill/is_ready": 1.0 if self.is_ready else 0.0,
            "kl_distill/term": float(self._last_term),
            "kl_distill/weight": float(self.weight),
            "kl_distill/temperature": float(self.temperature),
        }


# ── Internal helper ──────────────────────────────────────────────────────

def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip Lightning/DDP wrappers to reach the raw nn.Module."""
    return getattr(model, "module", model)


# ── Factory ──────────────────────────────────────────────────────────────

def build_kl_distillation(cfg: dict) -> Optional["KLDistillationModule"]:
    """
    Build a KLDistillationModule from the ``kl_distillation:`` block of a
    training YAML.

    Returns None if the block is absent or  ``enabled: false``.

    Example YAML:

        kl_distillation:
          enabled: true
          weight: 1.0            # multiplier on the KL term
          temperature: 2.0       # distillation softmax temperature
          teacher_dtype: float16 # float16 | bfloat16 | float32
          apply_to_all_phases: false
          trigger: phase_change  # snapshot trigger (handled by trainer)
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    return KLDistillationModule(
        weight=float(cfg.get("weight", 1.0)),
        temperature=float(cfg.get("temperature", 2.0)),
        teacher_dtype=cfg.get("teacher_dtype", "float16"),
        apply_to_all_phases=bool(cfg.get("apply_to_all_phases", False)),
    )
