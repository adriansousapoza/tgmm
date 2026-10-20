r"""
`ConditionalGaussianMixture`: the conditional distribution of a fitted
Gaussian mixture, plus the dimension-validation helper shared with
``GaussianMixture.marginal`` / ``condition``.

Given a mixture :math:`p(x)=\sum_k\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)` and an
observed block :math:`x_o` (the remaining coordinates :math:`m` are
unobserved), conditioning acts component by component:

.. math::
    A_k &= \Sigma_{k,mo}\,\Sigma_{k,oo}^{-1}, \\
    m_{ik} &= \mu_{k,m} + A_k(x_{o,i}-\mu_{k,o}), \\
    C_k &= \Sigma_{k,mm} - A_k\Sigma_{k,om}, \\
    w_{ik} &\propto \pi_k\,\mathcal N(x_{o,i}\mid\mu_{k,o},\Sigma_{k,oo}),

so that :math:`p(y\mid x_{o,i}) = \sum_k w_{ik}\,\mathcal N(y\mid m_{ik}, C_k)`.
The conditional covariances :math:`C_k` do not depend on the sample, only the
weights and the means do. The mean :math:`\sum_k w_{ik}m_{ik}` is Gaussian
mixture regression (Sung, 2004).
"""
import math
from typing import List, Sequence

import torch


def _validate_dims(dims: Sequence[int], n_features: int, name: str = 'dims') -> List[int]:
    r"""
    Validate and normalise a coordinate selection: a non-empty sequence of
    distinct integers in ``[0, n_features)``. Order is kept.
    """
    if isinstance(dims, torch.Tensor):
        dims = dims.tolist()
    try:
        dims = [int(j) for j in dims]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a sequence of integers, got {dims!r}") from None
    if len(dims) == 0:
        raise ValueError(f"{name} must not be empty")
    if len(set(dims)) != len(dims):
        raise ValueError(f"{name} must not contain duplicates, got {dims}")
    bad = [j for j in dims if not 0 <= j < n_features]
    if bad:
        raise ValueError(f"{name} entries must be in [0, {n_features}), got out-of-range {bad}")
    return dims


class ConditionalGaussianMixture:
    r"""
    The distribution of the unobserved coordinates of a fitted
    `GaussianMixture` given observed ones, one Gaussian mixture per row of
    ``x_obs``. Created by `GaussianMixture.condition`; not meant to be
    constructed directly.

    Attributes
    ----------
    weights_ : torch.Tensor, shape (n, K)
        Conditional component weights :math:`w_{ik}` (rows sum to 1).
    means_ : torch.Tensor, shape (n, K, d_m)
        Conditional means :math:`m_{ik}`.
    covariances_ : torch.Tensor, shape (K, d_m, d_m)
        Conditional covariances :math:`C_k`, shared by all samples.
    obs_dims : list of int
        Observed coordinates, in the order of the columns of ``x_obs``.
    missing_dims : list of int
        The sorted complement of ``obs_dims``; the coordinates the
        distribution is over, in increasing order.
    n_components : int
    n_features : int
        ``d_m``, the number of unobserved coordinates.
    device, dtype
    """

    def __init__(self, weights, means, covariances, obs_dims, missing_dims):
        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.obs_dims = list(obs_dims)
        self.missing_dims = list(missing_dims)
        self.n_components = weights.shape[1]
        self.n_features = means.shape[2]
        self.device = weights.device
        self.dtype = weights.dtype
        self._chol = None

    @property
    def n_samples(self) -> int:
        r"""Number of conditioning rows."""
        return self.weights_.shape[0]

    def _cholesky(self) -> torch.Tensor:
        if self._chol is None:
            self._chol = torch.linalg.cholesky(self.covariances_)
        return self._chol

    def mean(self) -> torch.Tensor:
        r"""
        Gaussian mixture regression: :math:`E[y\mid x_{o,i}]=\sum_kw_{ik}m_{ik}`.

        Returns
        -------
        torch.Tensor, shape (n, d_m)
        """
        return (self.weights_.unsqueeze(-1) * self.means_).sum(dim=1)

    def log_prob(self, Y: torch.Tensor) -> torch.Tensor:
        r"""
        :math:`\log p(y_i\mid x_{o,i})` for one query point per conditioning row.

        Parameters
        ----------
        Y : torch.Tensor, shape (n, d_m)
            Row ``i`` is evaluated under the mixture of row ``i``. A conditional
            with a single row (``n == 1``) is broadcast against any number of
            query points ``(m, d_m)``, which is how a single conditional
            density is evaluated on a grid.

        Returns
        -------
        torch.Tensor, shape (n,) (or (m,) in the single-row case)
        """
        Y = torch.as_tensor(Y, dtype=self.dtype, device=self.device)
        if Y.dim() != 2 or Y.shape[1] != self.n_features:
            raise ValueError(f"Y must have shape (n, {self.n_features}), got {tuple(Y.shape)}")
        if Y.shape[0] != self.n_samples and self.n_samples != 1:
            raise ValueError(f"Y has {Y.shape[0]} rows but the conditional has {self.n_samples}")
        L = self._cholesky()                                                    # (K, dm, dm)
        eye = torch.eye(self.n_features, dtype=self.dtype, device=self.device)
        L_inv = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)  # (K, dm, dm)
        diff = Y.unsqueeze(1) - self.means_                                      # (n|m, K, dm)
        z = torch.einsum('kde,nke->nkd', L_inv, diff)
        maha = z.pow(2).sum(-1)                                                  # (n|m, K)
        log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)   # (K,)
        log_n = -0.5 * (self.n_features * math.log(2.0 * math.pi) + log_det.unsqueeze(0) + maha)
        return torch.logsumexp(torch.log(self.weights_.clamp_min(1e-300)) + log_n, dim=1)

    def sample(self, n_draws: int = 1) -> torch.Tensor:
        r"""
        Draw from each row's conditional mixture.

        Parameters
        ----------
        n_draws : int, default=1
            Number of independent draws per conditioning row.

        Returns
        -------
        torch.Tensor, shape (n, n_draws, d_m)
        """
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}")
        n = self.n_samples
        comp = torch.multinomial(self.weights_, n_draws, replacement=True)       # (n, n_draws)
        rows = torch.arange(n, device=self.device).unsqueeze(1).expand(n, n_draws)
        out = self.means_[rows, comp].clone()                                    # (n, n_draws, dm)
        eps = torch.randn(n, n_draws, self.n_features, dtype=self.dtype, device=self.device)
        L = self._cholesky()
        for k in range(self.n_components):                                       # loop over components, not samples
            sel = comp == k
            if sel.any():
                out[sel] += eps[sel] @ L[k].T
        return out

    def as_gmm(self, i: int):
        r"""
        The conditional distribution of row ``i`` as a fitted
        `GaussianMixture` over the missing coordinates (full covariance).
        """
        from ..gmm import GaussianMixture

        if not -self.n_samples <= i < self.n_samples:
            raise IndexError(f"row {i} out of range for {self.n_samples} conditioning rows")
        gmm = GaussianMixture(
            n_components=self.n_components, n_features=self.n_features, covariance_type='full',
            device=self.device,
        )
        gmm.dtype = self.dtype
        gmm.weights_ = self.weights_[i].clone()
        gmm.means_ = self.means_[i].clone()
        gmm.covariances_ = self.covariances_.clone()
        gmm.fitted_ = True
        gmm.converged_ = True
        gmm.n_components_ = self.n_components
        gmm.n_iter_ = 0
        gmm.lower_bound_ = float('nan')
        return gmm

    def __repr__(self) -> str:
        return (f"ConditionalGaussianMixture(n_samples={self.n_samples}, n_components={self.n_components}, "
                f"obs_dims={self.obs_dims}, missing_dims={self.missing_dims})")
