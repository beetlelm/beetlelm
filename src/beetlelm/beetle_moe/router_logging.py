"""
src/beetle_moe/router_logging.py
─────────────────────────────────
Diagnostic utilities for MoE routing statistics.
"""
from __future__ import annotations
import math
from typing import Dict
import torch


def routing_diagnostics(
    expert_counts: torch.Tensor,   # (num_experts,) integer counts
    num_experts: int,
    prefix: str = "moe",
) -> Dict[str, float]:
    """
    Compute routing-quality metrics from per-expert token counts.

    Returns
    -------
    dict
        ``moe/routing_entropy``, ``moe/max_expert_load``,
        ``moe/min_expert_load``, ``moe/load_balance_cv``.
    """
    counts = expert_counts.float()
    total  = counts.sum().item()
    if total == 0:
        return {f"{prefix}/routing_entropy": 0.0}

    probs = counts / total
    # Normalised routing entropy (0 = collapsed to 1 expert; 1 = uniform)
    entropy = -(probs * (probs + 1e-9).log()).sum().item()
    max_entropy = math.log(num_experts)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    cv = (counts.std() / (counts.mean() + 1e-9)).item()

    return {
        f"{prefix}/routing_entropy":   float(norm_entropy),
        f"{prefix}/max_expert_load":   float(counts.max().item() / total),
        f"{prefix}/min_expert_load":   float(counts.min().item() / total),
        f"{prefix}/load_balance_cv":   float(cv),
    }
