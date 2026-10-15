r"""
Missing-value (``missing='em'``) numerics shared by the Gaussian-type families
(`GaussianMixture`, `StudentTMixture`).

``NaN`` entries mark missing-at-random values (Ghahramani & Jordan, 1994; Hunt &
Jorgensen, 2003). Rows are grouped by *missingness pattern*; within a pattern the
computation is vectorised across rows and components, and only a Python loop over
the (few) patterns remains. For a pattern with observed index set ``o`` (size
``p``) and missing set ``m`` (size ``q``) and component ``k``:

.. math::
    \delta_o = (x_o-\mu_o)^\top\Sigma_{oo}^{-1}(x_o-\mu_o),\qquad
    \hat x_m = \mu_m + \Sigma_{mo}\Sigma_{oo}^{-1}(x_o-\mu_o),\qquad
    C = \Sigma_{mm} - \Sigma_{mo}\Sigma_{oo}^{-1}\Sigma_{om}.

``delta_o`` and ``log|Sigma_oo|`` give the marginal (observed-data) density used by
the E-step; ``x_hat`` / ``C`` are the conditional mean and covariance of the missing
block used by the M-step statistics. See
docs/superpowers/specs/2026-10-15-v05-missing-values-design.md.

Covariances come in two representations, selected by `kind`:

* ``'dense'``: ``(K, d, d)`` matrices (``full``, ``tied_full`` expanded, the GPCM codes).
* ``'diag'``: ``(K, d)`` per-feature variances (``diag``/``spherical``/``tied_*``
  diagonal types, expanded). Here ``Sigma_mo = 0`` so ``x_hat_m = mu_m`` and
  ``C = diag(Sigma_mm)``; no solves are needed.
"""
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class PatternBlock:
    r"""One missingness pattern's precomputed pieces (all tensors on the data's device)."""

    rows: torch.Tensor  # (n_p,) row indices into the chunk
    obs: torch.Tensor  # (p,) observed feature indices
    mis: torch.Tensor  # (q,) missing feature indices
    delta: torch.Tensor  # (n_p, K) squared Mahalanobis distance on the observed block
    logdet: torch.Tensor  # (K,) log|Sigma_oo|
    xhat: Optional[torch.Tensor] = None  # (n_p, K, q) conditional mean of the missing block
    cond_cov: Optional[torch.Tensor] = None  # (K, q, q) dense or (K, q) diag conditional covariance


def has_missing(X: torch.Tensor) -> bool:
    return bool(torch.isnan(X).any().item())


def missing_patterns(X: torch.Tensor):
    r"""
    Group the rows of `X` by which entries are ``NaN``.

    Returns a list of ``(rows, obs, mis)`` index-tensor triples, one per distinct
    pattern (``P = len(result)``).
    """
    mask = torch.isnan(X)
    uniq, inverse = torch.unique(mask, dim=0, return_inverse=True)
    order = torch.argsort(inverse)
    counts = torch.bincount(inverse, minlength=uniq.shape[0]).tolist()
    groups = torch.split(order, counts)
    out = []
    for j, rows in enumerate(groups):
        m = uniq[j]
        out.append((rows, torch.nonzero(~m, as_tuple=True)[0], torch.nonzero(m, as_tuple=True)[0]))
    return out


def count_patterns(X: torch.Tensor) -> int:
    return int(torch.unique(torch.isnan(X), dim=0).shape[0])


def check_incomplete_rows_and_columns(X: torch.Tensor):
    r"""Raise `ValueError` for rows with no observed entry or columns with none."""
    nan = torch.isnan(X)
    if bool(nan.all(dim=1).any().item()):
        raise ValueError(
            "X contains rows whose entries are all missing (NaN); such rows carry no information about "
            "cluster membership -- drop them before fitting."
        )
    if bool(nan.all(dim=0).any().item()):
        cols = torch.nonzero(nan.all(dim=0), as_tuple=True)[0].tolist()
        raise ValueError(f"X has column(s) {cols} with no observed values at all (all NaN); drop them before fitting.")


def fill_columns_with_means(X: torch.Tensor) -> torch.Tensor:
    r"""`X` with every ``NaN`` replaced by its column's mean over the observed entries."""
    col_mean = torch.nanmean(X, dim=0)
    return torch.where(torch.isnan(X), col_mean.unsqueeze(0).expand_as(X), X)


def gaussian_blocks(
    X: torch.Tensor,
    means: torch.Tensor,
    cov: torch.Tensor,
    kind: str,
    need_conditional: bool = True,
) -> List[PatternBlock]:
    r"""
    Per-pattern observed-block Mahalanobis distances / log-determinants and, if
    `need_conditional`, the conditional mean and covariance of the missing block.

    Parameters
    ----------
    X : (n, d) tensor with NaNs.
    means : (K, d).
    cov : ``(K, d, d)`` if `kind == 'dense'`, else ``(K, d)`` variances.
    kind : ``'dense'`` or ``'diag'``.
    need_conditional : bool
        Skip the `Sigma_mo Sigma_oo^{-1}` solves when only the marginal density is needed.
    """
    K = means.shape[0]
    blocks = []
    for rows, obs, mis in missing_patterns(X):
        Xp = X[rows]
        q = mis.numel()
        diff = Xp[:, obs].unsqueeze(1) - means[:, obs].unsqueeze(0)  # (n_p, K, p)
        block = PatternBlock(rows=rows, obs=obs, mis=mis, delta=None, logdet=None)
        if kind == 'dense':
            cov_o = cov[:, obs]  # (K, p, d)
            cov_oo = cov_o[:, :, obs]  # (K, p, p)
            chol = torch.linalg.cholesky(cov_oo)
            block.logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
            diff_t = diff.permute(1, 2, 0).contiguous()  # (K, p, n_p)
            solved = torch.cholesky_solve(diff_t, chol)
            block.delta = (diff_t * solved).sum(dim=1).t()  # (n_p, K)
            if need_conditional and q > 0:
                cov_om = cov_o[:, :, mis]  # (K, p, q)
                sol_om = torch.cholesky_solve(cov_om, chol)  # Sigma_oo^{-1} Sigma_om, (K, p, q)
                block.xhat = means[:, mis].unsqueeze(0) + torch.einsum('kpq,kpn->nkq', sol_om, diff_t)
                block.cond_cov = cov[:, mis][:, :, mis] - sol_om.transpose(1, 2) @ cov_om  # (K, q, q)
        else:
            var_o = cov[:, obs]  # (K, p)
            block.logdet = torch.log(var_o).sum(-1)
            block.delta = (diff.pow(2) / var_o.unsqueeze(0)).sum(-1)
            if need_conditional and q > 0:
                block.xhat = means[:, mis].unsqueeze(0).expand(Xp.shape[0], K, q)
                block.cond_cov = cov[:, mis]  # (K, q)
        blocks.append(block)
    return blocks


def complete_rows(X: torch.Tensor, blocks: List[PatternBlock], n_components: int) -> torch.Tensor:
    r"""
    Per-component completed data `x_tilde_ik`, shape ``(n, K, d)``: observed entries are
    `x_i`, missing entries are the conditional means `x_hat_ik`.
    """
    n, d = X.shape
    Xhat = X.unsqueeze(1).expand(n, n_components, d).clone()
    for b in blocks:
        if b.xhat is not None:
            sub = Xhat[b.rows]
            sub[:, :, b.mis] = b.xhat
            Xhat[b.rows] = sub
    return Xhat


def conditional_cov_sum(
    blocks: List[PatternBlock], weights: torch.Tensor, kind: str, n_features: int
) -> torch.Tensor:
    r"""
    :math:`\sum_i w_{ik} C_{k,\text{pattern}(i)}` embedded in the missing-block entries:
    ``(K, d, d)`` for `kind == 'dense'`, ``(K, d)`` for `'diag'`. `weights` is ``(n, K)``.
    """
    K = weights.shape[1]
    shape = (K, n_features, n_features) if kind == 'dense' else (K, n_features)
    out = torch.zeros(shape, dtype=weights.dtype, device=weights.device)
    for b in blocks:
        if b.cond_cov is None:
            continue
        w = weights[b.rows].sum(dim=0)  # (K,)
        if kind == 'dense':
            out[:, b.mis.unsqueeze(1), b.mis.unsqueeze(0)] += w.view(-1, 1, 1) * b.cond_cov
        else:
            out[:, b.mis] += w.unsqueeze(1) * b.cond_cov
    return out
