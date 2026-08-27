"""
src/continual/ewc.py
────────────────────
Elastic Weight Consolidation (Kirkpatrick et al., 2017) for bilingual LM training.

Designed for BeetleLM's catastrophic-forgetting experiments.  The core idea:
after the model has finished its L1 phase, we (a) store the L1 optimal parameters
θ* and (b) estimate the diagonal of the Fisher information matrix F to weight how
important each parameter was for L1.  During subsequent L2 training the EWC term

    L_EWC = (λ/2) · Σ_i  F_i · (θ_i − θ*_i)²

is added to the language-modelling loss, penalising drift away from the L1
solution in proportion to each parameter's importance.

Usage (controlled entirely via YAML — see example at bottom of file):

    from beetlelm.continual.ewc import EWCRegularizer

    ewc = EWCRegularizer(lambda_=20.0, n_fisher_samples=10)
    ewc.snapshot(model)                          # call at L1→L2 phase boundary
    ewc.compute_fisher(model, l1_batches, device)

    # inside training loop:
    loss = lm_loss + ewc.penalty(model)
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EWCRegularizer:
    """
    Diagonal-Fisher EWC regularizer.

    Parameters
    ----------
    lambda_ : float
        Regularization strength λ.  Prior work found λ ≈ 20 for GPT-2-scale models
        and λ ≈ 150 for RoBERTa-scale models (Salhan et al., 2024).  Set to 0 to
        disable the penalty while still keeping the object alive.
    n_fisher_samples : int
        Number of Monte-Carlo samples (sequences) used to estimate the diagonal
        Fisher.  10 is the value used in Salhan et al. (2024).
    mu : float
        Coefficient for the plain L2 regularizer term (Eq. 10, second term).
        Set to 0 (default) to match the published experiments.
    device : torch.device | None
        If None, the device of the model's first parameter is used at snapshot time.
    """

    def __init__(
        self,
        lambda_: float = 20.0,
        n_fisher_samples: int = 10,
        mu: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.lambda_ = lambda_
        self.n_fisher_samples = n_fisher_samples
        self.mu = mu
        self._device = device

        # populated by snapshot() + compute_fisher()
        self._theta_star: dict[str, torch.Tensor] = {}
        self._fisher_diag: dict[str, torch.Tensor] = {}
        self._is_ready = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once both snapshot and Fisher have been computed."""
        return self._is_ready

    def snapshot(self, model: torch.nn.Module) -> None:
        """
        Store a copy of the current model parameters as θ* (the L1 optimum).
        Call this **at the L1→L2 curriculum phase boundary**, before any L2 data
        reaches the model.
        """
        raw = _unwrap(model)
        self._theta_star = {
            name: param.detach().clone()
            for name, param in raw.named_parameters()
            if param.requires_grad
        }
        logger.info(
            "[EWC] θ* snapshot taken — %d parameter tensors stored.",
            len(self._theta_star),
        )

    def compute_fisher(
        self,
        model: torch.nn.Module,
        batches: Iterable[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Estimate the diagonal of the Fisher information matrix using the
        Monte-Carlo approximation from Eq. (9b) in Salhan et al. (2024):

            F̂_i = (1/K) Σ_{k=1}^{K}  (∂ log p(x_k | θ) / ∂θ_i)²

        where each x_k is drawn from the model's own distribution (or, here,
        from real L1 data — a common and cheaper approximation).

        Parameters
        ----------
        model   : the raw or Fabric-wrapped model.
        batches : iterable of int64 tensors shape (B, T+1).  The first T tokens
                  are inputs; the last T tokens are targets.  We consume at most
                  `self.n_fisher_samples` batches.
        device  : target device; falls back to model's first-parameter device.
        """
        if not self._theta_star:
            raise RuntimeError(
                "[EWC] Call snapshot() before compute_fisher()."
            )

        raw = _unwrap(model)
        dev = device or self._device or next(raw.parameters()).device

        # Accumulate squared gradients
        fisher_acc: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in raw.named_parameters()
            if param.requires_grad
        }

        raw.eval()
        n_processed = 0

        for batch in batches:
            if n_processed >= self.n_fisher_samples:
                break

            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
            else:
                input_ids = batch

            input_ids = input_ids.to(dev)
            x = input_ids[:, :-1]
            y = input_ids[:, 1:]

            raw.zero_grad()
            # Forward — accept both (logits,) and (logits, extras) returns
            out = raw(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), y.contiguous().view(-1))
            loss.backward()

            for name, param in raw.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_acc[name] += param.grad.detach().pow(2)

            n_processed += 1

        if n_processed == 0:
            logger.warning("[EWC] compute_fisher() received 0 batches — Fisher is zero.")
        else:
            for name in fisher_acc:
                fisher_acc[name] /= n_processed

        self._fisher_diag = fisher_acc
        self._is_ready = True
        raw.train()

        total_params = sum(f.numel() for f in fisher_acc.values())
        mean_f = sum(f.mean().item() for f in fisher_acc.values()) / max(1, len(fisher_acc))
        logger.info(
            "[EWC] Fisher diagonal computed over %d batches | "
            "%d params | mean(F)=%.4e",
            n_processed, total_params, mean_f,
        )

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Return the EWC regularization loss:

            (λ/2) · Σ_i  F_i · (θ_i − θ*_i)²   +   (μ/2) · Σ_i (θ_i − θ*_i)²

        If not yet ready (Fisher not computed) returns a zero scalar on the
        model's current device so the caller can always do  loss += ewc.penalty(model).
        """
        raw = _unwrap(model)
        dev = next(raw.parameters()).device

        if not self._is_ready or self.lambda_ == 0.0:
            return torch.tensor(0.0, device=dev)

        penalty = torch.tensor(0.0, device=dev)
        for name, param in raw.named_parameters():
            if name not in self._fisher_diag:
                continue
            theta_star = self._theta_star[name].to(dev)
            fisher = self._fisher_diag[name].to(dev)
            diff = param - theta_star
            penalty = penalty + (fisher * diff.pow(2)).sum()

        penalty = (self.lambda_ / 2.0) * penalty

        # optional plain L2 term (μ > 0 only for ablations)
        if self.mu > 0.0:
            l2 = torch.tensor(0.0, device=dev)
            for name, param in raw.named_parameters():
                if name in self._theta_star:
                    diff = param - self._theta_star[name].to(dev)
                    l2 = l2 + diff.pow(2).sum()
            penalty = penalty + (self.mu / 2.0) * l2

        return penalty

    def state_dict(self) -> dict:
        """Serialisable state for checkpointing."""
        return {
            "lambda_": self.lambda_,
            "n_fisher_samples": self.n_fisher_samples,
            "mu": self.mu,
            "is_ready": self._is_ready,
            "theta_star": {k: v.cpu() for k, v in self._theta_star.items()},
            "fisher_diag": {k: v.cpu() for k, v in self._fisher_diag.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a previously saved state_dict."""
        self.lambda_ = state["lambda_"]
        self.n_fisher_samples = state["n_fisher_samples"]
        self.mu = state.get("mu", 0.0)
        self._is_ready = state["is_ready"]
        self._theta_star = {k: v for k, v in state["theta_star"].items()}
        self._fisher_diag = {k: v for k, v in state["fisher_diag"].items()}
        logger.info("[EWC] State restored from checkpoint (is_ready=%s).", self._is_ready)

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, float]:
        """
        Return a small dict of scalar statistics useful for W&B logging.
        Safe to call even before Fisher is computed.
        """
        if not self._is_ready:
            return {"ewc/is_ready": 0.0}
        all_f = torch.cat([f.view(-1) for f in self._fisher_diag.values()])
        return {
            "ewc/is_ready": 1.0,
            "ewc/fisher_mean": all_f.mean().item(),
            "ewc/fisher_max": all_f.max().item(),
            "ewc/fisher_nonzero_frac": (all_f > 0).float().mean().item(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip Lightning/DDP wrappers to reach the raw nn.Module."""
    return getattr(model, "module", model)


# ──────────────────────────────────────────────────────────────────────────────
# Factory — builds an EWCRegularizer from a YAML config block
# ──────────────────────────────────────────────────────────────────────────────

def build_ewc(cfg: dict) -> Optional["EWCRegularizer"]:
    """
    Build an EWCRegularizer from the `ewc:` block of a training YAML.

    Returns None if the block is absent or  enabled: false.

    Example YAML:

        ewc:
          enabled: true
          lambda_: 20.0          # regularization strength
          n_fisher_samples: 10   # batches used to estimate Fisher
          mu: 0.0                # plain-L2 coefficient (usually 0)
          # trigger: when to compute Fisher + snapshot.
          # 'phase_change' = at the L1→L2 curriculum boundary (recommended)
          # 'manual'       = caller calls ewc.snapshot() / ewc.compute_fisher() directly
          trigger: phase_change
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    return EWCRegularizer(
        lambda_=float(cfg.get("lambda_", 20.0)),
        n_fisher_samples=int(cfg.get("n_fisher_samples", 10)),
        mu=float(cfg.get("mu", 0.0)),
    )
