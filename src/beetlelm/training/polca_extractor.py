"""
src/training/polca_extractor.py
───────────────────────────────
POLCA (Phase-wise Loss Contribution Analysis) artifact extractor.

POLCA — arXiv 2510.05064 — decomposes per-step loss changes along an arbitrary
basis of the low-rank training subspace.  Computing those decompositions
post-hoc requires two artifacts at each checkpoint:

  1. A rank-r SVD basis ``U`` of the per-sequence gradient matrix
     ``G ∈ ℝ^{B × P}`` on a fixed held-out batch, where ``B`` is the
     holdout batch size and ``P`` is the parameter count.  ``U`` spans the
     dominant gradient directions and is what POLCA projects loss changes
     onto.

  2. The top-k eigenvectors ``V`` of the (parameter-space) Hessian on the
     same held-out batch, computed via Lanczos with Hessian-vector products
     obtained through double backward.  These give POLCA's "loss-curvature
     basis" — the directions along which loss changes most quickly under a
     given parameter perturbation.

The artifacts are persisted to
``{runs_dir}/{run_name}/{artifact_dir}/step_{N}.pt`` and (optionally)
pushed to the model's HuggingFace branch as a side-artifact, mirroring the
``learning_dynamics`` pipeline already in ``src/checkpointing/learning_dynamics.py``.

Design constraints
------------------
* Only rank 0 runs the heavy compute.  Other ranks skip and continue.
  The held-out batch is generated from a fixed seed so it is reproducible
  across runs but does not require a cross-rank broadcast.
* Casts to fp32 for the holdout pass even when the main step runs in
  bf16-mixed.  Hessian-vector products through autograd are unstable in
  bf16.
* Memory bound: peak GPU memory for the extraction is ~``2·P·sizeof(fp32)``
  (one param-shaped buffer for HVP, one for accumulated Lanczos vector) plus
  the activations of one fp32 forward/backward on a tiny holdout batch
  (8 × 512 by default).  For the 0.2B Beetle pico, that is ~1.6 GB on top
  of training, leaving ample headroom on 80 GB A100.

Public API
----------
* ``build_polca_extractor(cfg, runs_dir, run_name)`` — factory; returns
  ``None`` when the YAML block is absent or ``enabled: false``.
* ``POLCAExtractor.should_run(step)`` — frequency gate.
* ``POLCAExtractor.set_holdout_batch(batch)`` — installs the fixed
  held-out batch (called once at trainer init).
* ``POLCAExtractor.extract(model, step, fabric, hf_pusher=None)`` — runs
  the extraction and writes the artifact.

This module deliberately does no HF upload itself — the trainer is the one
that owns the async HF pusher, so we expose a callback pattern.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class _POLCASpec:
    enabled: bool
    every_n_steps: int
    k_eigenvectors: int
    basis_rank: int
    holdout_batch_size: int
    holdout_seq_len: int
    artifact_dir: str
    push_to_hf: bool
    precision: str

    @classmethod
    def from_dict(cls, cfg: dict) -> "_POLCASpec":
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            every_n_steps=int(cfg.get("every_n_steps", 500)),
            k_eigenvectors=int(cfg.get("k_eigenvectors", 32)),
            basis_rank=int(cfg.get("basis_rank", 32)),
            holdout_batch_size=int(cfg.get("holdout_batch_size", 8)),
            holdout_seq_len=int(cfg.get("holdout_seq_len", 512)),
            artifact_dir=str(cfg.get("artifact_dir", "polca")),
            push_to_hf=bool(cfg.get("push_to_hf", True)),
            precision=str(cfg.get("precision", "fp32")),
        )


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class POLCAExtractor:
    """Compute and persist POLCA artifacts at checkpoint steps.

    See module docstring for the artifact definitions.  The class is
    stateful in two regards:

    * ``self._holdout`` — the fixed held-out batch.  Setting it lazily via
      :meth:`set_holdout_batch` lets the trainer sample once from its own
      data pipeline (so the batch matches the model's tokenizer and seq
      length without any duplication of the data loading code).

    * ``self._param_signature`` — a hash of (param-name, shape) tuples
      captured at first extraction.  We refuse to run if the signature
      changes mid-training (e.g. layer growth, LoRA insertion) since the
      gradient-SVD basis would no longer be comparable across checkpoints.
    """

    def __init__(self, spec: _POLCASpec, runs_dir: str, run_name: str):
        self.spec = spec
        self.runs_dir = runs_dir
        self.run_name = run_name
        self._holdout: Optional[dict] = None
        self._param_signature: Optional[tuple] = None

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self.spec.enabled

    def should_run(self, step: int) -> bool:
        """True iff extraction should fire at ``step``.

        We fire at step 0 (untrained baseline) and at every multiple of
        ``every_n_steps`` thereafter.  Callers that want extraction tied to
        the trainer's checkpoint set should additionally AND with their own
        gate, mirroring the ``learning_dynamics`` pattern.
        """
        if not self.spec.enabled:
            return False
        if step == 0:
            return True
        return step % self.spec.every_n_steps == 0

    # ── Holdout batch management ─────────────────────────────────────────

    def set_holdout_batch(self, batch: dict) -> None:
        """Install the fixed held-out batch.  ``batch`` must contain the
        ``input_ids`` (and optionally ``labels``) tensors that the trainer
        passes to the model — same convention as a training batch.

        We clone-and-detach onto CPU to decouple the extractor from
        whatever device buffers the data pipeline keeps.
        """
        if "input_ids" not in batch:
            raise ValueError("POLCA holdout batch requires 'input_ids'")
        keep = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                keep[k] = v.detach().to("cpu", copy=True)
            else:
                keep[k] = v
        # Truncate to spec sizes so the holdout never blows past config.
        ids = keep["input_ids"]
        b = min(self.spec.holdout_batch_size, ids.shape[0])
        s = min(self.spec.holdout_seq_len, ids.shape[1])
        for k in list(keep):
            if isinstance(keep[k], torch.Tensor) and keep[k].dim() >= 2:
                keep[k] = keep[k][:b, :s].contiguous()
        self._holdout = keep
        logger.info(
            "[POLCA] holdout batch set: shape=%s",
            tuple(self._holdout["input_ids"].shape),
        )

    # ── Main extraction entry point ──────────────────────────────────────

    @torch.enable_grad()
    def extract(
        self,
        model: nn.Module,
        step: int,
        device: torch.device,
        *,
        rank: int = 0,
        is_global_zero: bool = True,
        hf_pusher: Optional[Callable[[Path, int], None]] = None,
    ) -> Optional[Path]:
        """Run extraction and persist the artifact.

        Returns the artifact path on rank 0 (or None on other ranks / if
        the extractor is disabled / if no holdout batch has been set).

        ``hf_pusher`` is an optional callback (artifact_path, step) → None
        that the trainer can pass to schedule an async HF upload.  We do
        not invoke any HF hub APIs from inside this module.
        """
        if not self.spec.enabled:
            return None
        if not is_global_zero:
            return None
        if self._holdout is None:
            logger.warning(
                "[POLCA] step %d: no holdout batch set; skipping extraction",
                step,
            )
            return None

        unwrapped = _unwrap_model(model)
        sig = _param_signature(unwrapped)
        if self._param_signature is None:
            self._param_signature = sig
        elif sig != self._param_signature:
            logger.warning(
                "[POLCA] step %d: parameter signature changed since first "
                "extraction; skipping (POLCA bases would not be comparable "
                "across checkpoints with a different parameter shape).",
                step,
            )
            return None

        try:
            artifacts = self._compute(unwrapped, device)
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("[POLCA] step %d: extraction failed: %s", step, exc)
            return None

        out_path = self._artifact_path(step)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifacts, out_path)
        logger.info("[POLCA] step %d: saved artifact -> %s", step, out_path)

        if hf_pusher is not None and self.spec.push_to_hf:
            try:
                hf_pusher(out_path, step)
            except Exception:  # pragma: no cover — defensive
                logger.exception(
                    "[POLCA] step %d: HF push callback failed", step,
                )

        return out_path

    # ── Internals ────────────────────────────────────────────────────────

    def _artifact_path(self, step: int) -> Path:
        return (
            Path(self.runs_dir)
            / self.run_name
            / self.spec.artifact_dir
            / f"step_{step}.pt"
        )

    def _compute(self, model: nn.Module, device: torch.device) -> dict:
        """Compute the gradient-SVD basis and Hessian top-k eigenvectors.

        Returns a dict of CPU tensors (for safe ``torch.save``).
        """
        # Enumerate trainable parameters in a stable order.
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("[POLCA] model has no trainable parameters")
        param_tensors = [p for _, p in params]
        param_names = [n for n, _ in params]

        # We work in the requested precision, defaulting to fp32.
        polca_dtype = _resolve_dtype(self.spec.precision)
        prev_dtypes = [p.dtype for p in param_tensors]
        if polca_dtype is not None and any(d != polca_dtype for d in prev_dtypes):
            for p in param_tensors:
                p.data = p.data.to(polca_dtype)

        try:
            ids = self._holdout["input_ids"].to(device)
            labels = self._holdout.get("labels", ids).to(device)
            b = ids.shape[0]

            # ── (1) Per-sequence gradients → SVD basis ──
            grads_per_seq = []
            for i in range(b):
                model.zero_grad(set_to_none=True)
                logits = _forward_logits(model, ids[i : i + 1])
                loss = _causal_lm_loss(logits, labels[i : i + 1])
                gflat = _flat_grad(loss, param_tensors, retain_graph=False)
                grads_per_seq.append(gflat.detach().cpu())

            G = torch.stack(grads_per_seq, dim=0)  # (B, P)
            r = min(self.spec.basis_rank, G.shape[0])
            # Economy SVD on the (B, P) matrix is equivalent to the SVD of
            # G^T but cheaper.  Use ``torch.linalg.svd`` on CPU fp32 to
            # avoid any GPU memory pressure from a (P, P) decomposition.
            U_b, S, Vh = torch.linalg.svd(G.to(torch.float32), full_matrices=False)
            grad_basis = Vh[:r].clone()        # (r, P) — top-r right singular vectors
            grad_singvals = S[:r].clone()      # (r,)

            # ── (2) Lanczos on the Hessian of the *batch* loss ──
            model.zero_grad(set_to_none=True)
            logits = _forward_logits(model, ids)
            batch_loss = _causal_lm_loss(logits, labels)
            grads = torch.autograd.grad(
                batch_loss, param_tensors, create_graph=True
            )
            grads_flat = _cat([g.reshape(-1) for g in grads])
            P = grads_flat.numel()

            def hvp(vec_flat: torch.Tensor) -> torch.Tensor:
                vec_split = _unflatten_like(vec_flat, param_tensors)
                # Sum over (g_i · v_i) — Pearlmutter trick.
                gv = sum((g * v).sum() for g, v in zip(grads, vec_split))
                hv = torch.autograd.grad(
                    gv, param_tensors, retain_graph=True, allow_unused=False,
                )
                return _cat([h.reshape(-1) for h in hv]).detach()

            k = min(self.spec.k_eigenvectors, max(1, P - 1))
            eigvals, eigvecs = _lanczos_topk(
                hvp, dim=P, k=k, device=device, dtype=polca_dtype or torch.float32,
            )
            # Sanity: eigvecs is (k, P)
            hess_basis = eigvecs.detach().cpu()
            hess_eigvals = eigvals.detach().cpu()

            artifacts = {
                "schema_version": 1,
                "param_names": param_names,
                "param_shapes": [tuple(p.shape) for p in param_tensors],
                "param_count": int(P),
                "grad_basis": grad_basis,            # (r, P)
                "grad_singvals": grad_singvals,      # (r,)
                "hess_eigvecs": hess_basis,          # (k, P)
                "hess_eigvals": hess_eigvals,        # (k,)
                "spec": vars(self.spec),
            }
            return artifacts
        finally:
            # Restore original dtypes if we cast.
            if polca_dtype is not None:
                for p, d in zip(param_tensors, prev_dtypes):
                    if p.dtype != d:
                        p.data = p.data.to(d)
            model.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying nn.Module from a Fabric/DDP wrapper."""
    inner = model
    for _ in range(4):  # bounded peel
        if hasattr(inner, "module"):
            inner = inner.module
        else:
            break
    return inner


def _param_signature(model: nn.Module) -> tuple:
    return tuple(
        (n, tuple(p.shape))
        for n, p in model.named_parameters()
        if p.requires_grad
    )


def _resolve_dtype(s: str) -> Optional[torch.dtype]:
    s = (s or "").lower()
    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "": None,
    }.get(s, None)


def _forward_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and return logits, accommodating both HF-style
    ``output.logits`` and pico-style direct-tensor returns."""
    out = model(input_ids)
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(
        f"[POLCA] unsupported model output type: {type(out).__name__}"
    )


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard shifted next-token cross-entropy."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    )


def _flat_grad(
    loss: torch.Tensor, params: list[torch.Tensor], *, retain_graph: bool
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph)
    return _cat([g.reshape(-1) for g in grads])


def _cat(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tensors, dim=0)


def _unflatten_like(
    flat: torch.Tensor, like: list[torch.Tensor]
) -> list[torch.Tensor]:
    out = []
    offset = 0
    for p in like:
        n = p.numel()
        out.append(flat[offset : offset + n].view_as(p))
        offset += n
    return out


def _lanczos_topk(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    *,
    dim: int,
    k: int,
    device: torch.device,
    dtype: torch.dtype,
    chunk_rows: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain Lanczos with full reorthogonalisation; returns the top-k
    Ritz pairs (eigenvalues by absolute value descending, eigenvectors
    stacked row-wise).

    We pick m = min(2k+8, dim) Krylov vectors which is conservative.
    Reorthogonalisation against all previous vectors keeps us numerically
    safe up to the modest k (≤32) we run with.

    The Krylov basis ``Q`` is held on CPU and streamed to ``device`` in
    row-chunks of ``chunk_rows`` rows for the reorthogonalisation and
    Ritz-vector projection.  At ``k=32`` and ``dim≈10⁸`` (≈110M-param
    model in fp32), an on-device Q would be ~32 GB and squeeze the
    create_graph=True training graph that the HVP needs — leaving GPU 0
    OOM at step 0.  Streaming caps Q's GPU footprint at
    ``chunk_rows * dim * itemsize`` (~7 GB for the same scale at
    chunk_rows=16) regardless of ``k``.  The math is identical to the
    all-on-GPU classical-Gram-Schmidt variant: coefficients are computed
    against the *original* residual before any subtraction, requiring two
    passes over Q per outer iteration.
    """
    m = min(2 * k + 8, dim)
    cpu = torch.device("cpu")
    # pin_memory=True only works on CUDA; gracefully degrade on CPU/MPS.
    pin = device.type == "cuda"
    try:
        Q_cpu = torch.zeros(m, dim, dtype=dtype, device=cpu, pin_memory=pin)
    except RuntimeError:
        # Some allocators reject pinning of very large buffers; fall back.
        Q_cpu = torch.zeros(m, dim, dtype=dtype, device=cpu)
    alphas = torch.zeros(m, dtype=dtype, device=device)
    betas = torch.zeros(m - 1, dtype=dtype, device=device)

    def _reorth(r: torch.Tensor, j: int) -> torch.Tensor:
        """In-place classical Gram-Schmidt: r ← r − Σ_{i<j} (q_i · r) q_i.

        Two passes over Q[:j] streamed from CPU.  Pass 1 computes the
        coefficient vector against the original ``r``; pass 2 subtracts
        the projection.  This matches the on-device variant bit-for-bit
        modulo finite-precision matmul ordering.
        """
        if j == 0:
            return r
        coeffs = torch.zeros(j, dtype=r.dtype, device=r.device)
        for s in range(0, j, chunk_rows):
            e = min(s + chunk_rows, j)
            Q_chunk = Q_cpu[s:e].to(device, non_blocking=True)
            coeffs[s:e] = Q_chunk @ r
            del Q_chunk
        for s in range(0, j, chunk_rows):
            e = min(s + chunk_rows, j)
            Q_chunk = Q_cpu[s:e].to(device, non_blocking=True)
            r = r - coeffs[s:e] @ Q_chunk
            del Q_chunk
        return r

    q = torch.randn(dim, dtype=dtype, device=device)
    q = q / q.norm()
    Q_cpu[0].copy_(q.detach(), non_blocking=pin)
    r = hvp(q)
    a = torch.dot(q, r)
    alphas[0] = a
    r = r - a * q

    for j in range(1, m):
        r = _reorth(r, j)
        b = r.norm()
        if b < 1e-10:
            # Restart with a random direction orthogonal to span(Q[:j]).
            r = torch.randn(dim, dtype=dtype, device=device)
            r = _reorth(r, j)
            b = r.norm()
            if b < 1e-10:
                # Effectively zero subspace — truncate.
                m = j
                Q_cpu = Q_cpu[:m]
                alphas = alphas[:m]
                betas = betas[: m - 1]
                break
        betas[j - 1] = b
        q_next = r / b
        Q_cpu[j].copy_(q_next.detach(), non_blocking=pin)
        r = hvp(q_next) - b * q
        a = torch.dot(q_next, r)
        alphas[j] = a
        r = r - a * q_next
        del q
        q = q_next

    # Build the symmetric tridiagonal T.
    T = torch.diag(alphas[:m]) + torch.diag(betas[: m - 1], diagonal=1) + torch.diag(
        betas[: m - 1], diagonal=-1
    )
    eigvals, eigvecs = torch.linalg.eigh(T)
    # Sort by |λ| desc and take top-k.
    order = torch.argsort(eigvals.abs(), descending=True)[:k]
    top_vals = eigvals[order]
    top_vecs = eigvecs[:, order]                      # (m, k)

    # Ritz vectors: ritz[i] = Σ_j top_vecs[j, i] * Q[j].  ``Q_cpu`` is
    # already on CPU and the (k, dim) accumulator is large enough to
    # blow GPU 0 even by itself (e.g. k=32, dim≈180M params, fp32 ⇒
    # 23 GB) — ``ritz = ritz + …`` would then have to allocate a second
    # 23 GB tensor for the temporary, OOMing out of the autograd graph
    # the HVP is keeping alive.  Do the accumulation on CPU with an
    # in-place add: zero allocator pressure on the device, and the
    # caller down-streams ``hess_basis = eigvecs.detach().cpu()``
    # anyway (see ``_compute``).
    ritz_cpu = torch.zeros(k, dim, dtype=dtype, device=cpu)
    top_vecs_cpu = top_vecs.cpu()                      # (m, k)
    for s in range(0, m, chunk_rows):
        e = min(s + chunk_rows, m)
        # ``Q_cpu`` already holds the Krylov rows; no transfer needed.
        ritz_cpu.add_(top_vecs_cpu[s:e].T @ Q_cpu[s:e])  # (k, dim)
    # Re-normalise rows for numerical hygiene.
    ritz_cpu.div_(ritz_cpu.norm(dim=1, keepdim=True).clamp_min(1e-12))
    return top_vals.detach().cpu(), ritz_cpu


# ---------------------------------------------------------------------------
# Factory — mirrors build_ewc / build_maml convention in src/continual/
# ---------------------------------------------------------------------------

def build_polca_extractor(
    cfg: dict,
    *,
    runs_dir: str,
    run_name: str,
) -> Optional[POLCAExtractor]:
    """Construct the extractor from a YAML ``polca:`` block.

    Returns ``None`` when the block is absent or has ``enabled: false``.
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    spec = _POLCASpec.from_dict(cfg)
    extractor = POLCAExtractor(spec, runs_dir=runs_dir, run_name=run_name)
    logger.info(
        "[POLCA] enabled — every_n_steps=%d  k_eigenvectors=%d  basis_rank=%d  "
        "holdout=(%d × %d)  artifact_dir=%s",
        spec.every_n_steps, spec.k_eigenvectors, spec.basis_rank,
        spec.holdout_batch_size, spec.holdout_seq_len, spec.artifact_dir,
    )
    return extractor
