"""
src/bilingual/geometry/alignment.py
─────────────────────────────────────
Pure-function geometry utilities for cross-lingual embedding alignment.

All functions operate on (N, D) float tensors and are differentiable
(except vec2vec_align which uses a closed-form ridge solution).

Functions:
  procrustes(X, Y)     — orthogonal Procrustes rotation (SVD-based)
  linear_cka(X, Y)     — linear CKA similarity score in [0, 1]
  vec2vec_align(X, Y)  — ridge-regularised linear map X → Y
"""
from __future__ import annotations

import torch
import torch.linalg as LA


def procrustes(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Solve the orthogonal Procrustes problem: find rotation R such that
    ``X @ R ≈ Y`` (minimises Frobenius norm of residual).

    Both X and Y are (N, D) matrices of corresponding embedding vectors.
    R is (D, D) and orthogonal (R.T @ R = I).

    Implementation: SVD of X.T @ Y, then R = U @ V.T.
    """
    if X.shape != Y.shape:
        raise ValueError(
            f"procrustes: X and Y must have the same shape, "
            f"got {X.shape} vs {Y.shape}."
        )
    M = X.t() @ Y          # (D, D)
    U, _S, Vh = LA.svd(M)  # U (D, D), Vh (D, D)
    R = U @ Vh              # orthogonal rotation
    return R


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute linear Centred Kernel Alignment between two (N, D) matrices.

    Uses the feature-space formula (Kornblith et al. 2019):
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    This is invariant to orthogonal rotations and isotropic scaling.
    Returns a scalar in [0, 1]; 1 means identical representations (up to
    rotation/scaling).

    Reference: Kornblith et al. 2019 (https://arxiv.org/abs/1905.00414).
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic_xy = (Y.t() @ X).norm("fro").pow(2)   # ||Y^T X||_F^2
    hsic_xx = (X.t() @ X).norm("fro")           # ||X^T X||_F
    hsic_yy = (Y.t() @ Y).norm("fro")           # ||Y^T Y||_F
    return hsic_xy / (hsic_xx * hsic_yy + 1e-10)


def vec2vec_align(
    X: torch.Tensor,
    Y: torch.Tensor,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """
    Ridge-regularised linear map W such that X @ W ≈ Y.

    W = (X.T @ X + ridge * I)^{-1} @ X.T @ Y   (D_in, D_out)

    Parameters
    ----------
    X : (N, D_in)  source embeddings.
    Y : (N, D_out) target embeddings.
    ridge : float  L2 regularisation coefficient.

    Returns
    -------
    W : (D_in, D_out) linear map.
    """
    D_in = X.shape[1]
    A = X.t() @ X + ridge * torch.eye(D_in, dtype=X.dtype, device=X.device)
    B = X.t() @ Y
    W = LA.solve(A, B)      # (D_in, D_out)
    return W
