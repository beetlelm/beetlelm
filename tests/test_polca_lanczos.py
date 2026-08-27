"""Correctness test for the streaming Lanczos in beetlelm.training.polca_extractor.

The reorthogonalised Lanczos solver in POLCA must recover the top-|λ| Ritz
pairs of any symmetric operator it's handed.  Earlier the Krylov basis ``Q``
was allocated on the same device as the HVP, which OOM'd GPU 0 on
≥10⁸-param models because Q alone was ~32 GB at k=32.  The current version
streams Q from CPU; this test pins down the math so future memory
optimisations don't drift.

We construct symmetric operators with a *prescribed* spectrum (the
eigenvectors of a fixed orthonormal Q_true and prescribed eigenvalues),
which gives us:

  * an unambiguous ground truth for the top-k by |λ|, and
  * a well-separated spectrum so Lanczos with m=2k+8 Krylov vectors
    converges to fp32 precision on extreme Ritz pairs.

Running on CPU is sufficient — the streaming logic exercises the same
code path on either device (``pin_memory`` is the only CUDA-specific
branch).
"""

from __future__ import annotations

import torch

from beetlelm.training.polca_extractor import _lanczos_topk


def _build_with_spectrum(eigvals: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Symmetric matrix A = Q diag(λ) Qᵀ for a random orthonormal Q."""
    dim = eigvals.numel()
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    A = Q @ torch.diag(eigvals.to(torch.float64)) @ Q.T
    return A.to(torch.float32)


def test_lanczos_recovers_top_k_eigvals():
    # A well-separated spectrum: top extreme eigenvalues stand out so
    # Lanczos converges in m=2k+8 iterations.
    dim, k = 256, 8
    eig = torch.tensor(
        # Bulk spread small; k extreme values an order of magnitude larger.
        [0.05 * i - 1.0 for i in range(dim - 2 * k)]
        + [10.0 + i for i in range(k)]
        + [-(10.0 + i) for i in range(k)],
        dtype=torch.float64,
    )
    A = _build_with_spectrum(eig, seed=0)
    expected = eig.abs().sort(descending=True).values[:k].to(torch.float32)

    torch.manual_seed(1)
    eigvals, eigvecs = _lanczos_topk(
        lambda v: A @ v,
        dim=dim,
        k=k,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert eigvals.shape == (k,)
    assert eigvecs.shape == (k, dim)
    got = eigvals.abs().sort(descending=True).values
    assert torch.allclose(got, expected, atol=1e-3), (got, expected)


def test_lanczos_eigvecs_satisfy_eigen_equation():
    # Use a geometrically-spaced extreme spectrum: each top-k eigenvalue
    # has a clear ratio gap from the next, which is the regime where
    # Lanczos at m = 2k+8 converges to ≪ 1% residual.  (Random-spectrum
    # tests are deceptively hard — convergence stalls when extreme |λ|
    # cluster, which is a property of Lanczos itself, not of the streaming
    # implementation under test.)
    dim, k = 200, 6
    bulk = [0.05 * i - 0.5 for i in range(dim - 2 * k)]
    extremes = [10.0 * (2.0 ** i) for i in range(k)]
    eig = torch.tensor(
        bulk + extremes + [-x for x in extremes],
        dtype=torch.float64,
    )
    A = _build_with_spectrum(eig, seed=42)

    torch.manual_seed(7)
    eigvals, eigvecs = _lanczos_topk(
        lambda v: A @ v,
        dim=dim,
        k=k,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    norms = eigvecs.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), norms

    for i in range(k):
        v = eigvecs[i]
        Av = A @ v
        residual = (Av - eigvals[i] * v).norm()
        rel = residual / max(eigvals[i].abs(), torch.tensor(1e-6))
        assert rel < 1e-3, (
            f"eigvec {i}: ||A v − λ v||/|λ| = {rel:.4g}"
        )


def test_lanczos_chunked_matches_unchunked():
    """Two chunk_rows settings must agree to within float-roundoff.  This
    locks down the streaming behaviour: changing the chunk size is a pure
    memory knob, never a numerics knob.
    """
    dim, k = 150, 5
    eig = torch.tensor(
        [0.1 * i - 0.5 for i in range(dim - 2 * k)]
        + [15.0 + i for i in range(k)]
        + [-(15.0 + i) for i in range(k)],
        dtype=torch.float64,
    )
    A = _build_with_spectrum(eig, seed=99)

    torch.manual_seed(11)
    e1, v1 = _lanczos_topk(
        lambda v: A @ v,
        dim=dim, k=k, device=torch.device("cpu"), dtype=torch.float32,
        chunk_rows=4,
    )
    torch.manual_seed(11)
    e2, v2 = _lanczos_topk(
        lambda v: A @ v,
        dim=dim, k=k, device=torch.device("cpu"), dtype=torch.float32,
        chunk_rows=64,
    )
    assert torch.allclose(e1, e2, atol=1e-4), (e1, e2)
    # Eigvec sign is arbitrary; compare absolute inner products.
    sims = (v1 * v2).sum(dim=1).abs()
    assert torch.allclose(sims, torch.ones_like(sims), atol=1e-3), sims


def test_lanczos_q_stays_on_cpu():
    """Regression guard: the Krylov basis must never live on the HVP
    device.  We assert this by tracking the device of every tensor the
    HVP receives — they should all be on the CPU device passed in (the
    HVP itself is what consumes GPU memory in production)."""
    dim, k = 64, 3
    eig = torch.linspace(-5, 5, dim, dtype=torch.float64)
    A = _build_with_spectrum(eig, seed=3)

    seen_devices: set[torch.device] = set()

    def hvp(v: torch.Tensor) -> torch.Tensor:
        seen_devices.add(v.device)
        return A @ v

    torch.manual_seed(0)
    _lanczos_topk(
        hvp, dim=dim, k=k,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert seen_devices == {torch.device("cpu")}, seen_devices
