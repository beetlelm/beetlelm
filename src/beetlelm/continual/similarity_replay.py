"""
src/continual/similarity_replay.py
──────────────────────────────────
Task-aware / similarity-based replay (InsCL-style) for continual LM training.

Core idea: instead of replaying all past data uniformly, weight replay by
the embedding-space similarity between the current task/language and each
previous one.  Languages with similar representations (e.g. EN↔ES) get more
replay than distant ones (e.g. EN↔ZH), yielding better cross-lingual transfer
and less interference.

    L = L_new + λ · Σ_{t'} S(t, t') · L_replay(x_{t'})

where S(t, t') = cos(centroid_t, centroid_{t'})  (or Wasserstein distance).

Usage (controlled via YAML):

    similarity_replay:
      enabled: true
      lambda_replay: 0.3
      similarity_metric: "cosine"
      centroid_layer: "last_hidden"
      n_centroid_samples: 100
      replay_buffer_size: 2000
      min_similarity_threshold: 0.0
      trigger: phase_change
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap(model: nn.Module) -> nn.Module:
    """Strip Lightning/DDP wrappers to reach the raw nn.Module."""
    return getattr(model, "module", model)


def _cosine_similarity_matrix(
    centroids: Dict[str, torch.Tensor],
) -> Dict[Tuple[str, str], float]:
    """Pairwise cosine similarity between task centroids."""
    keys = sorted(centroids.keys())
    sim: Dict[Tuple[str, str], float] = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i:]:
            c = F.cosine_similarity(
                centroids[k1].unsqueeze(0),
                centroids[k2].unsqueeze(0),
            ).item()
            sim[(k1, k2)] = c
            sim[(k2, k1)] = c
    return sim


def _wasserstein_distance_matrix(
    centroids: Dict[str, torch.Tensor],
) -> Dict[Tuple[str, str], float]:
    """Pairwise Wasserstein-1 distance (L1) approximated on sorted
    centroid dimensions, converted to a similarity score via exp(-d)."""
    keys = sorted(centroids.keys())
    sim: Dict[Tuple[str, str], float] = {}
    for i, k1 in enumerate(keys):
        s1, _ = centroids[k1].sort()
        for k2 in keys[i:]:
            s2, _ = centroids[k2].sort()
            d = (s1 - s2).abs().mean().item()
            s = float(torch.exp(torch.tensor(-d)).item())
            sim[(k1, k2)] = s
            sim[(k2, k1)] = s
    return sim


# ──────────────────────────────────────────────────────────────────────────────
# Main module
# ──────────────────────────────────────────────────────────────────────────────

class SimilarityReplayModule:
    """
    InsCL-style task-aware replay weighted by inter-task similarity.

    Parameters
    ----------
    lambda_replay : float
        Global weight for the similarity-weighted replay loss.
    similarity_metric : str
        ``"cosine"`` or ``"wasserstein"``.
    centroid_layer : str
        Not used as a layer selector (centroids use the model's final
        hidden state by default).  Reserved for future extension.
    n_centroid_samples : int
        Number of samples per task used to estimate centroids.
    replay_buffer_size : int
        Maximum stored real samples per task.
    min_similarity_threshold : float
        Tasks with similarity below this threshold are skipped during
        replay (saves compute on very distant languages).
    device : torch.device | None
        Target device.  Falls back to the model's device.
    """

    def __init__(
        self,
        lambda_replay: float = 0.3,
        similarity_metric: str = "cosine",
        centroid_layer: str = "last_hidden",
        n_centroid_samples: int = 100,
        replay_buffer_size: int = 2000,
        min_similarity_threshold: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.lambda_replay = lambda_replay
        self.similarity_metric = similarity_metric
        self.centroid_layer = centroid_layer
        self.n_centroid_samples = n_centroid_samples
        self.replay_buffer_size = replay_buffer_size
        self.min_similarity_threshold = min_similarity_threshold
        self._device = device

        # Per-task embedding centroids: {task_key: (d_model,) tensor}
        self._centroids: Dict[str, torch.Tensor] = {}
        # Pairwise similarity: {(task_a, task_b): float}
        self._similarity: Dict[Tuple[str, str], float] = {}
        # Per-task replay buffer: {task_key: [Tensor, ...]}
        self._buffer: Dict[str, List[torch.Tensor]] = defaultdict(list)

        self._is_ready = False
        self._replay_loss_ema = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once centroids and at least one buffer have been computed."""
        return self._is_ready and len(self._buffer) > 0

    @torch.no_grad()
    def compute_centroids(
        self,
        model: nn.Module,
        task_batches: Dict[str, Iterable[torch.Tensor]],
        device: Optional[torch.device] = None,
    ) -> None:
        """Compute mean embedding centroids for each task/language.

        Parameters
        ----------
        model : the raw or Fabric-wrapped model.
        task_batches : {task_key: iterable of (B, T) int64 tensors}.
                       We consume at most ``n_centroid_samples`` total
                       tokens per task.
        device : target device.
        """
        raw = _unwrap(model)
        dev = device or self._device or next(raw.parameters()).device
        raw.eval()

        V = getattr(raw.model_config, "vocab_size", 32000)

        for task_key, batches in task_batches.items():
            embeddings: List[torch.Tensor] = []
            n_seen = 0

            for batch in batches:
                if n_seen >= self.n_centroid_samples:
                    break

                if isinstance(batch, dict):
                    ids = batch["input_ids"]
                else:
                    ids = batch

                ids = ids.to(dev).clamp(0, V - 1)
                x = ids[:, :-1]
                out = raw(x)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                # Use mean of last hidden state as embedding
                # logits shape: (B, T, V) — but we want hidden states
                # For models returning (logits, hidden), use hidden
                if isinstance(out, (tuple, list)) and len(out) > 1:
                    hidden = out[1]
                else:
                    # Approximate: use logits mean as a proxy
                    hidden = logits

                # Mean-pool over sequence dimension -> (B, dim)
                mean_emb = hidden.mean(dim=1).detach().cpu()
                embeddings.append(mean_emb)
                n_seen += mean_emb.size(0)

            if embeddings:
                all_emb = torch.cat(embeddings, dim=0)[
                    : self.n_centroid_samples
                ]
                self._centroids[task_key] = all_emb.mean(dim=0)
                logger.info(
                    "[SimReplay] Centroid for '%s': %d samples, dim=%d.",
                    task_key, all_emb.size(0), all_emb.size(1),
                )

        raw.train()

    def compute_similarity_matrix(self) -> None:
        """(Re-)compute pairwise similarity between all known centroids."""
        if len(self._centroids) < 2:
            logger.info(
                "[SimReplay] Only %d centroid(s) — "
                "similarity matrix trivial.",
                len(self._centroids),
            )
            self._similarity = {}
            if self._centroids:
                k = list(self._centroids.keys())[0]
                self._similarity[(k, k)] = 1.0
            self._is_ready = len(self._buffer) > 0
            return

        if self.similarity_metric == "cosine":
            self._similarity = _cosine_similarity_matrix(self._centroids)
        elif self.similarity_metric == "wasserstein":
            self._similarity = _wasserstein_distance_matrix(self._centroids)
        else:
            raise ValueError(
                f"Unknown similarity_metric: {self.similarity_metric!r}"
            )

        self._is_ready = len(self._buffer) > 0
        logger.info(
            "[SimReplay] Similarity matrix computed (%d tasks): %s",
            len(self._centroids),
            {k: f"{v:.3f}" for k, v in self._similarity.items()},
        )

    def store_replay_samples(
        self,
        task_key: str,
        samples: List[torch.Tensor],
    ) -> None:
        """Store real data from a completed task into the replay buffer.

        Parameters
        ----------
        task_key : language/task identifier (e.g. ``"l1"``, ``"srp"``).
        samples : list of (T,) or (B, T) int64 tensors.
        """
        buf = self._buffer[task_key]
        for s in samples:
            if s.dim() == 2:
                for row in s:
                    buf.append(row.cpu())
            else:
                buf.append(s.cpu())
            # Enforce per-task cap
            while len(buf) > self.replay_buffer_size:
                buf.pop(0)

        logger.info(
            "[SimReplay] Stored %d sample(s) for task '%s' "
            "(buffer now %d).",
            len(samples), task_key, len(buf),
        )

    def weighted_replay_loss(
        self,
        model: nn.Module,
        current_task: str,
        step: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute similarity-weighted replay loss over all buffered tasks.

        Parameters
        ----------
        model : the Fabric-wrapped or raw model.
        current_task : identifier of the task currently being trained.
        step : training step (for diagnostics).

        Returns
        -------
        (weighted_loss, stats_dict)
            weighted_loss is already scaled by ``lambda_replay``.
        """
        raw = _unwrap(model)
        dev = next(raw.parameters()).device
        V = getattr(raw.model_config, "vocab_size", 32000)

        total_loss = torch.tensor(0.0, device=dev)
        n_tasks = 0

        for task_key, buf in self._buffer.items():
            if task_key == current_task or not buf:
                continue

            sim = self._similarity.get(
                (current_task, task_key), 0.0
            )
            if sim < self.min_similarity_threshold:
                continue

            # Sample a mini-batch from this task's buffer
            k = min(8, len(buf))
            indices = torch.randint(0, len(buf), (k,))
            replay_ids = torch.stack([buf[i] for i in indices]).to(dev)
            replay_ids = replay_ids.clamp(0, V - 1)

            x = replay_ids[:, :-1]
            y = replay_ids[:, 1:]
            out = raw(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            B, T, _ = logits.shape
            task_loss = F.cross_entropy(
                logits.reshape(B * T, V), y.contiguous().reshape(-1)
            )

            total_loss = total_loss + sim * task_loss
            n_tasks += 1

        scaled = self.lambda_replay * total_loss

        self._replay_loss_ema = (
            0.95 * self._replay_loss_ema + 0.05 * total_loss.item()
        )

        stats = {
            "sim_replay/loss": total_loss.item(),
            "sim_replay/scaled_loss": scaled.item(),
            "sim_replay/n_replay_tasks": float(n_tasks),
            "sim_replay/buffer_total": float(
                sum(len(b) for b in self._buffer.values())
            ),
        }
        return scaled, stats

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialisable state for checkpointing."""
        return {
            "lambda_replay": self.lambda_replay,
            "similarity_metric": self.similarity_metric,
            "centroid_layer": self.centroid_layer,
            "n_centroid_samples": self.n_centroid_samples,
            "replay_buffer_size": self.replay_buffer_size,
            "min_similarity_threshold": self.min_similarity_threshold,
            "is_ready": self._is_ready,
            "centroids": {
                k: v.cpu() for k, v in self._centroids.items()
            },
            "similarity": dict(self._similarity),
            "buffer": {
                k: [t.cpu() for t in v]
                for k, v in self._buffer.items()
            },
            "replay_loss_ema": self._replay_loss_ema,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a previously saved state_dict."""
        self.lambda_replay = state["lambda_replay"]
        self.similarity_metric = state["similarity_metric"]
        self.centroid_layer = state.get("centroid_layer", "last_hidden")
        self.n_centroid_samples = state["n_centroid_samples"]
        self.replay_buffer_size = state["replay_buffer_size"]
        self.min_similarity_threshold = state.get(
            "min_similarity_threshold", 0.0
        )
        self._is_ready = state["is_ready"]
        self._centroids = state.get("centroids", {})
        # Restore similarity with tuple keys
        raw_sim = state.get("similarity", {})
        self._similarity = {}
        for k, v in raw_sim.items():
            if isinstance(k, (list, tuple)):
                self._similarity[tuple(k)] = v
            else:
                self._similarity[k] = v
        self._buffer = defaultdict(list)
        for k, v in state.get("buffer", {}).items():
            self._buffer[k] = v
        self._replay_loss_ema = state.get("replay_loss_ema", 0.0)
        logger.info(
            "[SimReplay] State restored (is_ready=%s, "
            "%d centroids, %d buffered tasks).",
            self._is_ready, len(self._centroids), len(self._buffer),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def diagnostics(self) -> Dict[str, float]:
        """Scalar statistics for W&B logging."""
        if not self._is_ready:
            return {"sim_replay/is_ready": 0.0}

        avg_sim = 0.0
        n = 0
        for (k1, k2), v in self._similarity.items():
            if k1 != k2:
                avg_sim += v
                n += 1
        if n > 0:
            avg_sim /= n

        return {
            "sim_replay/is_ready": 1.0,
            "sim_replay/n_tasks": float(len(self._centroids)),
            "sim_replay/avg_similarity": avg_sim,
            "sim_replay/buffer_total": float(
                sum(len(b) for b in self._buffer.values())
            ),
            "sim_replay/replay_loss_ema": self._replay_loss_ema,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_similarity_replay(
    cfg: Optional[dict],
) -> Optional[SimilarityReplayModule]:
    """
    Build a SimilarityReplayModule from the ``similarity_replay:`` YAML block.

    Returns None if the block is absent or ``enabled: false``.

    Example YAML::

        similarity_replay:
          enabled: true
          lambda_replay: 0.3
          similarity_metric: "cosine"
          centroid_layer: "last_hidden"
          n_centroid_samples: 100
          replay_buffer_size: 2000
          min_similarity_threshold: 0.0
          trigger: phase_change
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    return SimilarityReplayModule(
        lambda_replay=float(cfg.get("lambda_replay", 0.3)),
        similarity_metric=str(cfg.get("similarity_metric", "cosine")),
        centroid_layer=str(cfg.get("centroid_layer", "last_hidden")),
        n_centroid_samples=int(cfg.get("n_centroid_samples", 100)),
        replay_buffer_size=int(cfg.get("replay_buffer_size", 2000)),
        min_similarity_threshold=float(
            cfg.get("min_similarity_threshold", 0.0)
        ),
    )
