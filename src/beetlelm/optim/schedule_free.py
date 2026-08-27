"""
src/optim/schedule_free.py
──────────────────────────
Schedule-Free AdamW (Defazio et al., NeurIPS 2024 — best paper).

Paper:           "The Road Less Scheduled" — https://arxiv.org/abs/2405.15682
Reference impl:  https://github.com/facebookresearch/schedule_free
                 (Apache 2.0; Copyright (c) Meta Platforms, Inc.)

This file is a verbatim port of ``schedulefree.adamw_schedulefree``
adapted to BeetleLM's optimizer registry. The mathematical update is
unchanged from the reference; only the surrounding wiring (imports,
attribution, lifecycle helpers) is BeetleLM-specific.

Key behavioural contract — DO NOT REGRESS:

  * ``optimizer.train()`` MUST be called before each training step.
  * ``optimizer.eval()`` MUST be called before evaluation / checkpointing
    so that the parameters held in ``p`` are the running average ``x``
    used for inference (rather than the auxiliary ``y`` iterate used
    during training). The trainer's ``_run_training_loop`` is responsible
    for this lifecycle; the regression test
    ``tests/test_features_differential.py::TestScheduleFreeAdamW``
    enforces it.
  * Calling ``step()`` while ``train_mode == False`` raises — this is the
    intentional loud-failure path that replaces the LR-scheduler / warmup
    plumbing.

We deliberately do NOT introduce a separate LR scheduler around
Schedule-Free: warmup is handled internally via ``warmup_steps``. The
trainer's ``_apply_lr_scheduler`` step is skipped when this optimizer
is in use.
"""
from __future__ import annotations

# Verbatim port of schedulefree.adamw_schedulefree (Apache 2.0, Meta).
# Original source: https://github.com/facebookresearch/schedule_free/
#                  blob/main/schedulefree/adamw_schedulefree.py

from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
import math

import torch
import torch.optim


try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class AdamWScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free AdamW.

    Defazio, Yaron, Guo, Borovykh, Mishchenko, Mehta, Ahmed, Cutkosky.
    "The Road Less Scheduled." NeurIPS 2024 (Best Paper).

    No external LR scheduler is needed; warmup is internal via
    ``warmup_steps``.

    Lifecycle:
        * Call ``optimizer.train()`` before each training ``step()``.
        * Call ``optimizer.eval()`` before evaluation or checkpointing.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize, or dicts defining parameter
        groups.
    lr : float, default 0.0025
        Base learning rate.
    betas : (float, float), default (0.9, 0.999)
        Coefficients used for the running averages of gradient and its
        square.
    eps : float, default 1e-8
        Term added to the denominator outside the root for numerical
        stability.
    weight_decay : float, default 0.0
        Decoupled weight-decay (a.k.a. AdamW-style L2).
    warmup_steps : int, default 0
        Linear warmup length in optimizer steps.
    r : float, default 0.0
        Polynomial weighting exponent in the running average.
    weight_lr_power : float, default 2.0
        Exponent applied to ``lr_max`` in the running-average weight.
    foreach : bool or None, default None
        If True, use ``torch._foreach_*`` ops for speed at the cost of
        higher peak memory. Defaults to True iff supported.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 0.0025,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: Optional[bool] = None,
    ):
        if foreach is None:
            foreach = hasattr(torch, "_foreach_mul_")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            r=r,
            k=0,
            warmup_steps=warmup_steps,
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,
            weight_lr_power=weight_lr_power,
            weight_decay=weight_decay,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle: train() / eval() switch ``p`` between the y iterate
    # (training) and the x iterate (inference / checkpointing).
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to x = (1 - 1/beta1)·z + (1/beta1)·y inverse.
                        p.lerp_(end=state["z"].to(p.device), weight=1 - 1 / beta1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Restore p back to y (training iterate).
                        p.lerp_(end=state["z"].to(p.device), weight=1 - beta1)
                group["train_mode"] = True

    # ──────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "[AdamWScheduleFree] step() called while train_mode=False. "
                "You must call optimizer.train() before each training step "
                "and optimizer.eval() before evaluation / checkpointing."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            k = group["k"]
            r = group["r"]
            warmup_steps = group["warmup_steps"]
            weight_lr_power = group["weight_lr_power"]

            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            bias_correction2 = 1 - beta2 ** (k + 1)
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr

            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0.0

            active_p = [p for p in group["params"] if p.grad is not None]

            for p in active_p:
                if "z" not in self.state[p]:
                    self.state[p]["z"] = torch.clone(
                        p, memory_format=torch.preserve_format
                    )
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            if group["foreach"] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(
                    *[
                        (p, p.grad, self.state[p]["exp_avg_sq"], self.state[p]["z"])
                        for p in active_p
                    ]
                )

                # Second-moment EMA + denominator
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1 - beta2)
                denom = torch._foreach_div(exp_avg_sq, bias_correction2)
                torch._foreach_sqrt_(denom)
                torch._foreach_add_(denom, eps)

                # Normalise grad in-place (memory)
                torch._foreach_div_(grad, denom)

                # Decoupled weight decay at y
                if decay != 0:
                    torch._foreach_add_(grad, y, alpha=decay)

                # y-step (in place; x is implicit)
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=lr * (beta1 * (1 - ckp1) - 1))

                # z-step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                for p in active_p:
                    y = p
                    grad = p.grad
                    state = self.state[p]
                    z = state["z"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                    grad_normalized = grad.div_(denom)

                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))

                    z.sub_(grad_normalized, alpha=lr)

            group["k"] = k + 1
        return loss
