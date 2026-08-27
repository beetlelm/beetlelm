"""
src/working_memory/alibi.py
───────────────────────────
Time-varying ALiBi (Attention with Linear Biases) slope scheduler.

Ported from CPLM:
  - transformers/src/transformers/models/gpt2/modeling_gpt2_alibi_linear.py
  - transformers/src/transformers/models/gpt2/modeling_gpt2_alibi_exponential.py

The original implementation is epoch-indexed (``update_epoch(e)`` called once per
epoch).  BeetleLM is step-indexed, so this scheduler exposes a ``step(step,
max_steps)`` hook instead:

  * ``linear``:        m_t(head i) = 1 / (s_t ** (i+1)),
                       s_t = initial_slope * (1 - p) + final_slope * p,
                       p = step / max(1, max_steps - 1).
                       (Matches CPLM's linear ALiBi formula, with per-head slope
                       inversely proportional to head index.)

  * ``exponential``:   m_t (shared across heads) = initial_slope * decay_rate ** phase,
                       where phase = floor(step / max_steps * num_phases).
                       (Matches CPLM's exponential ALiBi — all heads share slope.)

Each attention layer registers a forward-time hook that queries
``scheduler.bias(seq_len, device, dtype)`` and adds the returned ALiBi bias to
its attention mask.  The scheduler is ticked once per optimizer step from the
training loop.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class WorkingMemoryScheduler:
    """
    ALiBi slope scheduler with a bias-tensor builder.

    Parameters
    ----------
    schedule : {"linear", "exponential"}
        Which slope schedule to use.  See module docstring for formulas.
    num_heads : int
        Number of attention heads (required for the linear schedule; the
        exponential schedule shares a single slope across heads).
    initial_slope : float
        Slope at step 0.

        For the **linear** schedule, ``s_t`` interpolates from ``initial_slope``
        (at step 0) to ``final_slope`` (at the final step), and the per-head
        bias coefficient is ``m_h = 1 / s_t ** (h+1)``.  Therefore a
        DEVELOPMENTAL curriculum (locality bias DECAYING over training so the
        working-memory window GROWS) requires ``initial_slope < final_slope``.

        Default ``2.0`` matches the CPLM linear schedule (Salhan et al. 2024).
    final_slope : float
        Slope at the end of training.  Default ``10.0`` for ``linear`` (CPLM
        developmental curriculum); use ``0.1`` for ``exponential`` per CPLM.
    num_phases : int
        (Exponential only.)  Number of discrete slope-decay phases.  Default 10,
        matching CPLM's ``num_epochs=10``.
    decay_rate : float
        (Exponential only.)  Per-phase multiplicative decay.  Default 0.6,
        hardcoded in CPLM.

    Notes
    -----
    The constructor refuses inverted ``linear`` schedules (``initial_slope >
    final_slope``) because that would make ``m_h = 1 / s_t ** (h+1)`` blow up
    toward the end of training, collapsing attention to the current token —
    the OPPOSITE of the intended cognitive curriculum.  Pass
    ``allow_inverted_schedule=True`` if you intentionally want this for an
    ablation; in that case a single warning is logged.
    """

    def __init__(
        self,
        schedule: str = "linear",
        num_heads: int = 12,
        initial_slope: float = 2.0,
        final_slope: float = 10.0,
        num_phases: int = 10,
        decay_rate: float = 0.6,
        allow_inverted_schedule: bool = False,
    ):
        if schedule not in ("linear", "exponential"):
            raise ValueError(
                f"[WorkingMemory] schedule must be 'linear' or 'exponential', got {schedule!r}"
            )
        self.schedule = schedule
        self.num_heads = int(num_heads)
        self.initial_slope = float(initial_slope)
        self.final_slope = float(final_slope)
        self.num_phases = int(num_phases)
        self.decay_rate = float(decay_rate)

        # ── Schedule-direction validation ────────────────────────────────────
        # For the linear schedule, m_h = 1 / s_t ** (h+1) has a 1/0 singularity
        # as s_t -> 0; if initial_slope > final_slope the bias coefficient
        # explodes toward the end of training and the working-memory window
        # shrinks rather than grows. Fail loudly so configs that rely on the
        # old (1.0, 0.0) defaults don't silently train a backwards curriculum.
        if schedule == "linear" and self.initial_slope > self.final_slope:
            msg = (
                f"[WorkingMemory] Inverted linear schedule detected: "
                f"initial_slope={self.initial_slope} > final_slope={self.final_slope}. "
                f"This makes per-head slope m_h = 1/s_t**(h+1) BLOW UP toward "
                f"the end of training (working-memory window SHRINKS — opposite "
                f"of CPLM developmental curriculum). Use initial_slope < final_slope "
                f"(CPLM linear: 2.0 → 10.0). "
                f"Pass allow_inverted_schedule=True to override for ablations."
            )
            if allow_inverted_schedule:
                logger.warning(msg)
            else:
                raise ValueError(msg)

        # Refuse exact zeros that would put the linear formula at a singularity
        # right out of the gate; the floor at 1e-8 inside _compute_m is a
        # numerical safety net, not a license for divide-by-zero configs.
        if schedule == "linear" and (
            self.initial_slope <= 0.0 or self.final_slope <= 0.0
        ):
            raise ValueError(
                f"[WorkingMemory] linear schedule requires both slopes > 0; "
                f"got initial_slope={self.initial_slope}, final_slope={self.final_slope}."
            )

        # Current per-head slopes, shape (num_heads,).  Recomputed by step().
        # Initialise at progress=0 so a pre-train forward pass works before the
        # first optimizer step.
        self._m: torch.Tensor = self._compute_m(progress=0.0)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def step(self, current_step: int, max_steps: int) -> None:
        """Refresh the slope for the current training step."""
        if max_steps <= 1:
            progress = 0.0
        else:
            progress = min(1.0, max(0.0, current_step / (max_steps - 1)))
        self._m = self._compute_m(progress=progress)

    def bias(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return an ALiBi bias tensor of shape ``(1, num_heads, seq_len, seq_len)``
        suitable for addition to a (broadcasted) attention mask.

        Entry ``[0, h, i, j] = m_h * (j - i)``.  For a causal model this is
        non-positive in the attended region (``j <= i``) and the magnitude is
        proportional to the distance ``i - j``.

        Uses ``j - i`` (not ``i - j``) to match CPLM's sign convention, where
        the bias is added BEFORE the causal mask — attending to earlier tokens
        gets progressively penalised as (i - j) grows.
        """
        # relative_positions[i, j] = j - i  (so earlier keys get negative bias)
        rows = torch.arange(seq_len, device=device)
        rel = rows[None, :] - rows[:, None]          # (seq_len, seq_len)
        m = self._m.to(device=device, dtype=dtype)   # (num_heads,)
        bias = m.view(1, -1, 1, 1) * rel.to(dtype).view(1, 1, seq_len, seq_len)
        return bias

    @property
    def current_slopes(self) -> torch.Tensor:
        """Current per-head slope tensor (read-only, CPU)."""
        return self._m.detach().cpu()

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _compute_m(self, progress: float) -> torch.Tensor:
        if self.schedule == "linear":
            s_t = self.initial_slope * (1.0 - progress) + self.final_slope * progress
            # Avoid div-by-zero if final_slope=0 at progress=1.  Floor at a tiny
            # positive number so 1 / (s_t ** (i+1)) stays finite; at s_t=eps the
            # ALiBi bias saturates the mask and the model effectively sees only
            # the causal part — matching the "no working-memory constraint"
            # limiting behaviour.
            s_t = max(s_t, 1e-8)
            head_slopes = [1.0 / (s_t ** (i + 1)) for i in range(self.num_heads)]
            # In CPLM the bias sign is applied as `m * (i - j)`, and m is
            # positive — so bias on future tokens is positive (ignored by causal
            # mask) and on past tokens is negative-of-positive-distance times
            # m = negative.  Our `bias()` flips the sign via `j - i`, so store
            # m as-is and let the negative relative-position do the work.
            return torch.tensor(head_slopes, dtype=torch.float32)
        else:
            phase = int(progress * self.num_phases)
            m_t = self.initial_slope * (self.decay_rate ** phase)
            # Exponential variant shares the slope across all heads.
            return torch.full((self.num_heads,), m_t, dtype=torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "schedule": self.schedule,
            "num_heads": self.num_heads,
            "initial_slope": self.initial_slope,
            "final_slope": self.final_slope,
            "num_phases": self.num_phases,
            "decay_rate": self.decay_rate,
            "m": self._m.cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.schedule = state.get("schedule", self.schedule)
        self.num_heads = int(state.get("num_heads", self.num_heads))
        self.initial_slope = float(state.get("initial_slope", self.initial_slope))
        self.final_slope = float(state.get("final_slope", self.final_slope))
        self.num_phases = int(state.get("num_phases", self.num_phases))
        self.decay_rate = float(state.get("decay_rate", self.decay_rate))
        if "m" in state:
            self._m = state["m"].clone().float()


# ──────────────────────────────────────────────────────────────────────────
# Factory — builds a WorkingMemoryScheduler from a YAML config block
# ──────────────────────────────────────────────────────────────────────────

def build_working_memory(
    cfg: dict,
    *,
    num_heads: Optional[int] = None,
) -> Optional[WorkingMemoryScheduler]:
    """
    Build a ``WorkingMemoryScheduler`` from the ``working_memory:`` YAML block.

    Returns ``None`` if the block is absent or ``enabled: false``.

    Example YAML (CPLM linear developmental curriculum):

        working_memory:
          enabled: true
          schedule: "linear"          # or "exponential"
          initial_slope: 2.0          # CPLM default; locality bias is highest
          final_slope: 10.0           # CPLM default; locality bias decays away
          num_phases: 10              # exponential only
          decay_rate: 0.6             # exponential only

    For the exponential schedule, recommended (CPLM):

        working_memory:
          enabled: true
          schedule: "exponential"
          initial_slope: 1.0
          final_slope: 0.1
          num_phases: 10
          decay_rate: 0.6
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    schedule_name = str(cfg.get("schedule", "linear"))
    # Sensible per-schedule defaults that match CPLM.
    if schedule_name == "linear":
        default_init, default_final = 2.0, 10.0
    else:
        default_init, default_final = 1.0, 0.1
    scheduler = WorkingMemoryScheduler(
        schedule=schedule_name,
        num_heads=int(num_heads if num_heads is not None else cfg.get("num_heads", 12)),
        initial_slope=float(cfg.get("initial_slope", default_init)),
        final_slope=float(cfg.get("final_slope", default_final)),
        num_phases=int(cfg.get("num_phases", 10)),
        decay_rate=float(cfg.get("decay_rate", 0.6)),
        allow_inverted_schedule=bool(cfg.get("allow_inverted_schedule", False)),
    )
    logger.info(
        "[WorkingMemory] Enabled | schedule=%s | initial_slope=%s final_slope=%s "
        "| num_heads=%d",
        scheduler.schedule, scheduler.initial_slope, scheduler.final_slope,
        scheduler.num_heads,
    )
    return scheduler
