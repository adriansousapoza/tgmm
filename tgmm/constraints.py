r"""
Exact block updates for constrained M-steps of :class:`~tgmm.GaussianMixture`
(``weight_bounds``, ``mean_bounds``, ``covariance_bounds``).

Each function maximises the expected complete-data log-likelihood :math:`Q` over **one** parameter block, given the others, so
the combined M-step is an ECM step and the log-likelihood stays monotone. See
``docs/superpowers/specs/2026-10-21-gmm-constraints-design.md`` for the derivations.
"""
from typing import Union

import torch

Bound = Union[float, torch.Tensor]


def water_fill_weights(counts: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    r"""
    Maximiser of :math:`\sum_k n_k\log\pi_k` subject to :math:`\sum_k\pi_k=1` and :math:`lo_k\le\pi_k\le hi_k`.

    The Lagrangian gives :math:`\pi_k=\mathrm{clip}(n_k/\nu,\,lo_k,\,hi_k)` where the multiplier :math:`\nu>0` solves
    :math:`\sum_k\mathrm{clip}(n_k/\nu,lo_k,hi_k)=1`. The left-hand side is decreasing in :math:`\nu`, so the root is
    bracketed by doubling and found by bisection. When the unconstrained solution :math:`n_k/\sum_j n_j` is feasible it is
    returned as is.

    Parameters
    ----------
    counts : torch.Tensor, shape (K,)
        Non-negative (effective) component counts :math:`n_k`.
    lo, hi : torch.Tensor, shape (K,)
        Bounds; ``sum(lo) <= 1 <= sum(hi)`` is required.

    Returns
    -------
    torch.Tensor, shape (K,)
        Weights summing to 1 (to floating-point precision) inside the bounds.
    """
    n = counts.clamp(min=0)
    lo = lo.to(n.dtype)
    hi = hi.to(n.dtype)
    if float(lo.sum()) > 1.0 + 1e-9 or float(hi.sum()) < 1.0 - 1e-9:
        raise ValueError(
            f"infeasible weight bounds: need sum(lo) <= 1 <= sum(hi), got sum(lo)={float(lo.sum()):.6g}, "
            f"sum(hi)={float(hi.sum()):.6g}."
        )
    total = n.sum()
    if not bool(total > 0):                       # no information: stay as close to uniform as the bounds allow
        n = torch.ones_like(n)
        total = n.sum()
    p = n / total
    if bool(((p >= lo) & (p <= hi)).all()):
        return p

    def mass(nu):
        return torch.minimum(torch.maximum(n / nu, lo), hi).sum()

    nu_lo = nu_hi = total.clone()
    if float(mass(total)) > 1.0:                  # need a larger multiplier to shrink the weights
        for _ in range(200):
            nu_hi = nu_hi * 2.0
            if float(mass(nu_hi)) <= 1.0:
                break
        nu_lo = nu_hi / 2.0
    else:                                          # need a smaller multiplier
        for _ in range(200):
            nu_lo = nu_lo / 2.0
            if float(mass(nu_lo)) >= 1.0:
                break
        nu_hi = nu_lo * 2.0
    for _ in range(200):
        mid = 0.5 * (nu_lo + nu_hi)
        if float(mass(mid)) > 1.0:
            nu_lo = mid
        else:
            nu_hi = mid
    return torch.minimum(torch.maximum(n / (0.5 * (nu_lo + nu_hi)), lo), hi)


def box_qp_coordinate_ascent(
    ybar: torch.Tensor, precision: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, x0: torch.Tensor,
    max_sweeps: int = 50, tol: float = 1e-12,
) -> torch.Tensor:
    r"""
    Maximiser of :math:`-\tfrac12 (x-\bar y)^\top P (x-\bar y)` over the box :math:`lo\le x\le hi`, per row, by exact coordinate
    ascent started at the feasible point ``x0``.

    The conditional maximiser in coordinate :math:`j` is
    :math:`x_j=\mathrm{clip}\big(\bar y_j-\sum_{l\ne j}P_{jl}(x_l-\bar y_l)/P_{jj},\,lo_j,\,hi_j\big)`, so every sweep
    never decreases the objective (ECM) and repeated sweeps converge to the box-QP optimum.

    Parameters
    ----------
    ybar, lo, hi, x0 : torch.Tensor, shape (K, d)
    precision : torch.Tensor, shape (K, d, d)
        Positive-definite :math:`P=\Sigma^{-1}`.
    max_sweeps : int
        Upper bound on the number of full sweeps over the coordinates.
    tol : float
        Stop when no coordinate moved by more than ``tol`` (relative to the scale of ``x``).
    """
    x = x0.clone()
    d = x.shape[1]
    diag = torch.diagonal(precision, dim1=-2, dim2=-1)
    for _ in range(max_sweeps):
        before = x.clone()
        for j in range(d):
            resid = x - ybar
            others = torch.einsum('kl,kl->k', precision[:, j, :], resid) - diag[:, j] * resid[:, j]
            xj = ybar[:, j] - others / diag[:, j]
            x[:, j] = torch.minimum(torch.maximum(xj, lo[:, j]), hi[:, j])
        if float((x - before).abs().max()) <= tol * (1.0 + float(x.abs().max())):
            break
    return x


def clip_eigenvalues(S: torch.Tensor, lo: Bound, hi: Bound) -> torch.Tensor:
    r"""
    Closest feasible covariance in the sense of the Gaussian likelihood: the minimiser of
    :math:`n\log|\Sigma|+\mathrm{tr}(\Sigma^{-1}T)` over :math:`lo\,I\preceq\Sigma\preceq hi\,I` with :math:`S=T/n` has the
    eigenvectors of :math:`S` and its eigenvalues clipped to :math:`[lo,hi]` (von Neumann's trace inequality reduces the problem
    to one scalar problem per eigenvalue).

    Matrices that already satisfy the bounds are returned unchanged.

    Parameters
    ----------
    S : torch.Tensor, shape (..., d, d)
        Symmetric matrices.
    lo, hi : float or torch.Tensor
        Scalars, or tensors broadcastable to the leading shape ``(...)`` (one bound per matrix).
    """
    evals, evecs = torch.linalg.eigh(S)
    lo_t = torch.as_tensor(lo, dtype=S.dtype, device=S.device)
    hi_t = torch.as_tensor(hi, dtype=S.dtype, device=S.device)
    if lo_t.dim() > 0:
        lo_t = lo_t.unsqueeze(-1)
    if hi_t.dim() > 0:
        hi_t = hi_t.unsqueeze(-1)
    violates = ((evals < lo_t) | (evals > hi_t)).any(dim=-1)
    clipped = torch.minimum(torch.maximum(evals, lo_t), hi_t)
    rebuilt = (evecs * clipped.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    rebuilt = 0.5 * (rebuilt + rebuilt.transpose(-1, -2))
    return torch.where(violates.unsqueeze(-1).unsqueeze(-1), rebuilt, S)
