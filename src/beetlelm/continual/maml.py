"""
src/continual/maml.py
──────────────────────
Simplified MAML (Finn et al., 2017) / SMLMT (Bangor et al., 2020) inner-loop
abstraction for BeetleLM.

The class `SMLMTModule` encapsulates the full meta-learning episode:
  1. Sample disjoint support / query positions from a batch of token sequences.
  2. Run K inner-loop SGD steps on the support set using a lightweight
     classifier head (fine-tuned head, frozen backbone).
  3. Restore the head to its pre-episode state.
  4. Compute the query-set outer loss that flows gradients back through the
     backbone — driving fast adaptability.

It is decoupled from the trainer so it can be unit-tested and swapped out
independently.  The trainer calls:

    loss, meta_stats = maml.episode(_input_ids, model, fabric)

and then does the normal fabric.backward(loss / accum) + optimizer.step().

The `hybrid_ratio` parameter controls how often a meta episode is run versus
a plain autoregressive step — set to 0.0 to disable MAML entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Episode statistics (returned per step for logging)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MetaEpisodeStats:
    did_meta: bool = False
    loss: float = 0.0
    inner_acc_sup: float = 0.0   # avg support accuracy across inner steps
    outer_acc_q: float = 0.0     # query accuracy at outer step
    inner_grad_norm_mean: float = 0.0
    inner_grad_norm_std: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Classifier head builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_classifier_head(
    in_dim: int,
    vocab_size: int,
    num_layers: int,
    hidden_dim: int,
    init_method: str = "xavier",
) -> nn.Sequential:
    """Build and optionally Xavier-initialise the task-head MLP."""
    layers: list[nn.Module] = []
    cur_dim = in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(cur_dim, hidden_dim), nn.ReLU()]
        cur_dim = hidden_dim
    layers.append(nn.Linear(cur_dim, vocab_size))
    head = nn.Sequential(*layers)
    if init_method == "xavier":
        for p in head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return head


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class SMLMTModule:
    """
    SMLMT meta-learning module for BeetleLM.

    The module owns:
      • the classifier head (attached to model.classifier_head on construction)
      • the inner-loop SGD optimizer
      • per-interval accuracy histories for logging

    Parameters
    ----------
    model        : the raw (un-wrapped) PicoDecoder / BGPT model.
    vocab_size   : number of token classes.
    d_model      : hidden dimension of the backbone.
    hybrid_ratio : probability in [0, 1] of choosing a meta episode over AR.
    n_way        : N in N-way classification.
    k_shot       : K in K-shot support set.
    inner_steps  : number of inner-loop SGD steps per episode.
    inner_lr     : learning rate for the inner-loop SGD.
    mask_token_id: token id used to mask support/query positions.
    head_num_layers : depth of the classifier head MLP.
    head_hidden_dim : width of hidden layers in the head.
    head_init_method: weight initialisation for the head ('xavier' | 'default').
    """

    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        d_model: int,
        hybrid_ratio: float = 0.5,
        n_way: int = 5,
        k_shot: int = 1,
        inner_steps: int = 3,
        inner_lr: float = 0.1,
        mask_token_id: int = 0,
        head_num_layers: int = 2,
        head_hidden_dim: int = 256,
        head_init_method: str = "xavier",
    ):
        self.hybrid_ratio = hybrid_ratio
        self.n_way = n_way
        self.k_shot = k_shot
        self.inner_steps = inner_steps
        self.mask_token_id = mask_token_id

        # Build and attach head to the raw model
        head = _build_classifier_head(
            in_dim=d_model,
            vocab_size=vocab_size,
            num_layers=head_num_layers,
            hidden_dim=head_hidden_dim,
            init_method=head_init_method,
        )
        model.classifier_head = head

        self._vocab_size = vocab_size

        # Parameter groups
        self.head_params = list(model.classifier_head.parameters())
        self.backbone_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("classifier_head")
        ]
        self.all_params = self.backbone_params + self.head_params

        # Inner-loop optimiser (only touches the head)
        self.inner_optimizer = torch.optim.SGD(self.head_params, lr=inner_lr)

        # Running histories (reset each logging interval by trainer)
        self.inner_acc_history: list[float] = []
        self.query_acc_history: list[float] = []
        self.step_count_meta: int = 0
        self.step_count_ar: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Core episode method — called once per sub-batch step
    # ──────────────────────────────────────────────────────────────────────────

    def episode(
        self,
        input_ids: torch.Tensor,   # (B, L)  int64, already on device
        model: nn.Module,          # Fabric-wrapped model
        fabric,                    # Lightning Fabric instance (for backward + broadcast)
        force_meta: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, MetaEpisodeStats]:
        """
        Run one training episode — either MAML or plain AR, chosen stochastically.

        Returns (loss, stats).  The loss is **not** divided by grad-accum steps;
        the trainer handles that.
        """
        # ── Choose branch synchronously across all ranks ──────────────────────
        if force_meta is not None:
            do_meta = force_meta
        else:
            rand_val = torch.rand(1, device=input_ids.device)
            rand_val = fabric.broadcast(rand_val, src=0)
            do_meta = rand_val.item() < self.hybrid_ratio

        if do_meta:
            loss, stats = self._meta_episode(input_ids, model, fabric)
            self.step_count_meta += 1
        else:
            loss, stats = self._ar_episode(input_ids, model)
            self.step_count_ar += 1

        return loss, stats

    # ──────────────────────────────────────────────────────────────────────────
    # MAML branch
    # ──────────────────────────────────────────────────────────────────────────

    def _meta_episode(
        self,
        input_ids: torch.Tensor,
        model: nn.Module,
        fabric,
    ) -> Tuple[torch.Tensor, MetaEpisodeStats]:
        raw = _unwrap(model)
        device = input_ids.device
        B, L = input_ids.size()
        N, K = self.n_way, self.k_shot
        V = self._vocab_size
        head_dtype = next(raw.classifier_head.parameters()).dtype

        stats = MetaEpisodeStats(did_meta=True)

        # 1) Snapshot head weights so we can restore after inner loop
        orig_head_state = {
            name: param.clone()
            for name, param in raw.classifier_head.state_dict().items()
        }

        # 2) Sample disjoint support / query positions from [1, L-1)
        perm = torch.randperm(L - 1, device=device) + 1
        pos_sup = perm[:K].unsqueeze(0).expand(B, K)   # (B, K)
        pos_q   = perm[K].repeat(B, 1)                 # (B, 1)

        support_ids = input_ids.clone()
        query_ids   = input_ids.clone()

        # 3) Extract labels and mask those positions
        sup_labels = support_ids[torch.arange(B).unsqueeze(1), pos_sup]   # (B, K)
        q_labels   = query_ids[torch.arange(B), pos_q.squeeze(1)]         # (B,)

        support_ids[torch.arange(B).unsqueeze(1), pos_sup] = self.mask_token_id
        query_ids[torch.arange(B), pos_q.squeeze(1)]       = self.mask_token_id

        # 4) Build vocab→class mapper for N-way classification
        classes, mapper = _build_class_mapper(sup_labels, N, V, device)

        # 5) Remap labels (clamp unmapped → 0 as a safe default)
        sup_targets = _remap(mapper, sup_labels.view(-1)).view(B, K)
        q_targets   = _remap(mapper, q_labels)

        # 6) Inner-loop SGD on support set (backbone frozen)
        inner_accs: list[float] = []
        inner_grad_norms: list[float] = []

        for _ in range(self.inner_steps):
            # Freeze backbone
            for p in self.backbone_params:
                p.requires_grad_(False)

            self.inner_optimizer.zero_grad()

            hidden_sup, _ = raw(support_ids, return_hidden=True)
            hidden_sup = hidden_sup.to(head_dtype)

            logits_sup = raw.classifier_head(hidden_sup)
            # Extract positions → (B, K, N)
            logits_sup = logits_sup[torch.arange(B).unsqueeze(1), pos_sup, :][:, :, classes]
            logits_sup = logits_sup.contiguous().view(B * K, N)

            loss_sup = F.cross_entropy(logits_sup, sup_targets.view(-1))

            with torch.no_grad():
                acc_sup = (logits_sup.argmax(-1) == sup_targets.view(-1)).float().mean().item()
            inner_accs.append(acc_sup)
            self.inner_acc_history.append(acc_sup)

            # Use fabric.backward so DDP hooks stay alive
            fabric.backward(loss_sup, model=model)
            self.inner_optimizer.step()

            # Unfreeze backbone
            for p in self.backbone_params:
                p.requires_grad_(True)

            grads = [p.grad for p in raw.classifier_head.parameters() if p.grad is not None]
            if grads:
                inner_grad_norms.append(
                    torch.stack([g.norm() for g in grads]).mean().item()
                )

        # 7) Query forward for outer loss
        hidden_q, _ = raw(query_ids, return_hidden=True)
        hidden_q = hidden_q.to(head_dtype)

        logits_q = raw.classifier_head(hidden_q)
        logits_q = logits_q[torch.arange(B), pos_q.squeeze(1), :][:, classes]  # (B, N)

        loss = F.cross_entropy(logits_q, q_targets)

        with torch.no_grad():
            acc_q = (logits_q.argmax(-1) == q_targets).float().mean().item()
        self.query_acc_history.append(acc_q)

        # 8) Restore head to pre-episode state (outer grad flows through backbone)
        with torch.no_grad():
            raw.classifier_head.load_state_dict(orig_head_state)

        # Fill stats
        stats.loss = loss.item()
        stats.inner_acc_sup = sum(inner_accs) / max(1, len(inner_accs))
        stats.outer_acc_q   = acc_q
        if inner_grad_norms:
            mn = sum(inner_grad_norms) / len(inner_grad_norms)
            stats.inner_grad_norm_mean = mn
            stats.inner_grad_norm_std  = (
                sum((x - mn) ** 2 for x in inner_grad_norms) / len(inner_grad_norms)
            ) ** 0.5

        return loss, stats

    # ──────────────────────────────────────────────────────────────────────────
    # Standard AR branch
    # ──────────────────────────────────────────────────────────────────────────

    def _ar_episode(
        self,
        input_ids: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, MetaEpisodeStats]:
        raw = _unwrap(model)
        x      = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        logits, _ = raw(x)
        B, T, V  = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), labels.contiguous().view(-1))
        stats = MetaEpisodeStats(did_meta=False, loss=loss.item())
        return loss, stats

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities for trainer
    # ──────────────────────────────────────────────────────────────────────────

    def reset_histories(self) -> None:
        """Call after each logging interval to reset running accuracies."""
        self.inner_acc_history.clear()
        self.query_acc_history.clear()

    def avg_inner_acc(self) -> float:
        return (
            sum(self.inner_acc_history) / len(self.inner_acc_history)
            if self.inner_acc_history else 0.0
        )

    def avg_query_acc(self) -> float:
        return (
            sum(self.query_acc_history) / len(self.query_acc_history)
            if self.query_acc_history else 0.0
        )

    def state_dict(self) -> dict:
        return {
            "inner_optimizer": self.inner_optimizer.state_dict(),
            "step_count_meta": self.step_count_meta,
            "step_count_ar":   self.step_count_ar,
        }

    def load_state_dict(self, state: dict) -> None:
        self.inner_optimizer.load_state_dict(state["inner_optimizer"])
        self.step_count_meta = state.get("step_count_meta", 0)
        self.step_count_ar   = state.get("step_count_ar", 0)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def _build_class_mapper(
    sup_labels: torch.Tensor,
    n_way: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a (classes, mapper) pair for N-way classification.

    classes : (N,) int64 tensor of selected vocab ids
    mapper  : (V,) int64 tensor mapping vocab_id → class_index (or -1)
    """
    mapper  = torch.full((vocab_size,), -1, device=device, dtype=torch.long)
    classes = torch.unique(sup_labels.view(-1))

    if classes.numel() < n_way:
        pad     = torch.randint(0, vocab_size, (n_way - classes.numel(),), device=device)
        classes = torch.cat([classes, pad])
    elif classes.numel() > n_way:
        perm    = torch.randperm(classes.numel(), device=device)
        classes = classes[perm[:n_way]]

    mapper[classes] = torch.arange(n_way, device=device)
    return classes, mapper


def _remap(mapper: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Apply mapper to ids, clamping unmapped (-1) values to 0."""
    raw = mapper[ids.view(-1)]
    return torch.where(raw < 0, torch.zeros_like(raw), raw).view(ids.shape)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_maml(
    cfg: dict,
    model: nn.Module,
    vocab_size: int,
    d_model: int,
    mask_token_id: int,
) -> Optional["SMLMTModule"]:
    """
    Build an SMLMTModule from the `maml:` config block.

    Returns None if the block is absent or  enabled: false.

    Example YAML:

        maml:
          enabled: true
          hybrid_ratio: 0.5        # fraction of steps that are meta episodes
          n_way: 5
          k_shot: 1
          inner_steps: 3
          inner_lr: 0.1
          head:
            num_layers: 2
            hidden_dim: 256
            init_method: xavier
    """
    if not cfg or not cfg.get("enabled", False):
        return None

    head_cfg = cfg.get("head", {}) or {}

    return SMLMTModule(
        model=model,
        vocab_size=vocab_size,
        d_model=d_model,
        hybrid_ratio=float(cfg.get("hybrid_ratio", 0.5)),
        n_way=int(cfg.get("n_way", 5)),
        k_shot=int(cfg.get("k_shot", 1)),
        inner_steps=int(cfg.get("inner_steps", 3)),
        inner_lr=float(cfg.get("inner_lr", 0.1)),
        mask_token_id=mask_token_id,
        head_num_layers=int(head_cfg.get("num_layers", 2)),
        head_hidden_dim=int(head_cfg.get("hidden_dim", 256)),
        head_init_method=str(head_cfg.get("init_method", "xavier")),
    )
