r"""
`StudentTMixture`: a robust alternative to `GaussianMixture` (Peel &
McLachlan, 2000). Each component is a multivariate Student-t with mean
`mu_k`, scale matrix `Sigma_k`, and degrees of freedom `nu_k` -- outliers
get small `u_ik` weights in the M-step instead of dragging the scale
estimate the way they would a Gaussian covariance.

See docs/superpowers/specs/2026-10-14-phase4-base-mixture-studentt-vmf-design.md
section 4.2 for the design rationale, in particular why the Mahalanobis/
log-determinant helpers below are implemented independently rather than by
importing anything from `tgmm/gmm.py` (keeping the already bit-identical-
verified `GaussianMixture` untouched by this file's existence).
"""
import math
import warnings
from typing import Optional, Tuple, Union

import torch

from ..gmm_init import GMMInitializer
from ..stats import SufficientStats
from .base import BaseMixture
from . import missing as _missing
from .conditional import _validate_dims

_VALID_COVARIANCE_TYPES = ('full', 'diag', 'spherical', 'tied_full')
_VALID_SCALE_UPDATES = ('ecm', 'ecme')


def _mahalanobis_and_logdet(
    X: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor, covariance_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Squared Mahalanobis distance :math:`\delta_{ik} = (x_i - \mu_k)^T
    \Sigma_k^{-1} (x_i - \mu_k)`, shape `(n, K)`, and `log|Sigma_k|`, shape
    `(K,)` (broadcast to `(K,)` for `tied_full` too, for a uniform return
    shape), for one of `_VALID_COVARIANCE_TYPES`.

    For `'full'`/`'tied_full'`, computed via `cholesky_solve` with samples
    reshaped as right-hand-side *columns* rather than relying on
    `MultivariateNormal`-style broadcasting -- the same fix Phase 0 applied
    to `GaussianMixture._estimate_log_gaussian_full`, needed here for the
    same reason (naive broadcasting of `(n, K, d)` against `(K, d, d)`
    would materialize an `(n, K, d, d)` tensor).
    """
    n, d = X.shape
    K = means.shape[0]

    if covariance_type == 'full':
        chol = torch.linalg.cholesky(covariances)  # (K, d, d)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)  # (K,)
        diff = X.unsqueeze(1) - means.unsqueeze(0)  # (n, K, d)
        diff_t = diff.permute(1, 2, 0).contiguous()  # (K, d, n)
        solved = torch.cholesky_solve(diff_t, chol)  # (K, d, n)
        delta = (diff_t * solved).sum(dim=1).t()  # (n, K)
        return delta, logdet

    elif covariance_type == 'diag':
        logdet = torch.log(covariances).sum(-1)  # (K,)
        diff2 = (X.unsqueeze(1) - means.unsqueeze(0)).pow(2)  # (n, K, d)
        delta = (diff2 / covariances.unsqueeze(0)).sum(-1)  # (n, K)
        return delta, logdet

    elif covariance_type == 'spherical':
        logdet = d * torch.log(covariances)  # (K,)
        diff2 = (X.unsqueeze(1) - means.unsqueeze(0)).pow(2).sum(-1)  # (n, K)
        delta = diff2 / covariances.unsqueeze(0)
        return delta, logdet

    elif covariance_type == 'tied_full':
        chol = torch.linalg.cholesky(covariances)  # (d, d)
        logdet_scalar = 2.0 * torch.log(torch.diagonal(chol)).sum()
        diff = X.unsqueeze(1) - means.unsqueeze(0)  # (n, K, d)
        diff_flat = diff.reshape(-1, d).t()  # (d, n*K)
        solved = torch.cholesky_solve(diff_flat, chol)  # (d, n*K)
        delta = (diff_flat * solved).sum(dim=0).reshape(n, K)
        logdet = logdet_scalar.expand(K)
        return delta, logdet

    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type!r}")


class StudentTMixture(BaseMixture):
    r"""
    A multivariate Student-t Mixture Model, fit by ECM (Expectation
    Conditional Maximization; Peel & McLachlan, 2000).

    Each component has a mean `mu_k`, scale matrix `Sigma_k`
    (`covariances_`), and degrees of freedom `nu_k` (`dof_`). The extra
    per-sample weight :math:`u_{ik} = (\nu_k + d) / (\nu_k + \delta_{ik})`
    downweights points far from a component's center (small `u`) relative
    to a Gaussian fit, which is what gives the t-mixture its robustness to
    outliers.

    Shares its entire EM/CEM/SEM/SAEM/online-EM loop, `n_init` best-run
    selection, `predict`/`predict_proba`/`score`/`bic`/`aic`/`icl`, and
    generic `sample` with every other `BaseMixture` subclass -- see that
    class. Gibbs-sampling mode (`n_components=None`) and supervised fitting
    (`fit(X, labels=...)`) are intentionally Gaussian-only; neither is
    available here.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    n_features : int, optional
        Dimensionality of the data. Inferred from `X` at fit time if `None`.
    covariance_type : str, default='full'
        One of `'full'`, `'diag'`, `'spherical'`, `'tied_full'` -- the
        GPCM parameterizations (Phase 3) and Gibbs-only axis-aligned-tied
        types are not supported here.
    dof : {'per_component', 'shared'} or float, default='per_component'
        `'per_component'` fits one `nu_k` per component (closed-form
        bisection+Newton root-find each M-step); `'shared'` fits one `nu`
        for every component (the bracketed digamma statistic is summed
        over `k` before solving); a float fixes `nu` at that value for
        every component for the whole fit (no root-find at all -- e.g.
        `dof=1e4` recovers an (approximately) Gaussian fit, used by the
        parity test against `GaussianMixture`).
    dof_init : float, default=5.0
        Initial `nu_k` for every component, before the first M-step.
    dof_bounds : tuple of float, default=(0.1, 1000.0)
        `(lo, hi)` clamp applied to every fitted `nu_k`.
    scale_update : {'ecm', 'ecme'}, default='ecm'
        `'ecm'`: :math:`\Sigma_k = \sum_i \tau_{ik} u_{ik} (x_i-\mu_k)(x_i-\mu_k)^T / \sum_i \tau_{ik}`
        (the standard ECM update). `'ecme'`: divide by
        :math:`\sum_i \tau_{ik} u_{ik}` instead, which converges faster
        (Liu & Rubin, 1995) but is a different fixed point in general.
    init_means, init_weights : see `BaseMixture`.
    init_covariances : str or torch.Tensor, default='empirical'
        Initial `Sigma_k`: `'empirical'` (per-component empirical
        covariance from a nearest-mean hard assignment), `'global'` (the
        whole dataset's own covariance, shared as the starting point for
        every component), `'eye'`, or an explicit tensor.
    max_iter, tol, reg_covar, n_init, random_state, warm_start, algorithm,
    batch_size, temperature, convergence, step_size, sem_burn_in,
    sem_estimate, verbose, verbose_interval, device : see `BaseMixture`.

    noise : bool, optional
        Add a uniform **noise component** (Banfield & Raftery, 1993): a parameter-free
        extra mixture component with density `1 / V`. Responsibilities then have shape
        `(n_samples, n_components + 1)` (last column = noise), `predict` returns `-1` for
        points the noise component explains best, and `weights_` holds the component
        weights, which sum to `1 - noise_weight_`. Not supported with Gibbs mode,
        `labels=`, online EM / `partial_fit`, `weight_concentration_prior` or constrained
        `sample`. (default: False)
    noise_volume : 'bbox' or float, optional
        Volume `V` of the noise region: the bounding box of the fit data, or a user-supplied
        hypervolume. Ignored by `VonMisesFisherMixture` (uniform on the sphere). (default: 'bbox')
    noise_prior : float, optional
        Initial noise weight `pi_0`, in `(0, 1)`. (default: 0.05)
    noise_init : array-like, optional
        Per-sample initial noise indicator: a bool mask (`True` = noise) or cluster labels
        with `-1` = noise (e.g. `HDBSCAN(...).labels_`). Overrides `noise_prior`; the initial
        parameters are computed from the unflagged rows. (default: None)
    missing : {'error', 'em'}, optional
        `'error'` (default): `NaN` in `X` raises `ValueError`. `'em'`: `NaN` marks missing-at-random
        entries; the E-step uses the marginal t density of the observed block (`d -> p`, the number of
        observed entries) and the ECM statistics use the conditional mean / covariance of the missing
        block (see docs/superpowers/specs/2026-10-15-v05-missing-values-design.md). `predict`, `score`,
        `sample_weights`, `bic`/`aic`/`icl` accept incomplete rows; `impute(X)` fills them in.
        Requires `scale_update='ecm'`; not supported with `noise=True`, `labels=`, online EM or
        `partial_fit`. Identical to `'error'` on data without `NaN`. (default: 'error')

    Attributes
    ----------
    weights_ : torch.Tensor, shape (n_components,)
    means_ : torch.Tensor, shape (n_components, n_features)
    covariances_ : torch.Tensor
        Shape depends on `covariance_type`, same convention as
        `GaussianMixture`.
    dof_ : torch.Tensor, shape (n_components,)
        Fitted `nu_k` -- always one value per component, even when
        `dof='shared'` or a fixed float (all entries equal in that case).
    """

    def __init__(
        self,
        n_components: int,
        n_features: Optional[int] = None,
        covariance_type: str = 'full',
        dof: Union[str, float] = 'per_component',
        dof_init: float = 5.0,
        dof_bounds: Tuple[float, float] = (0.1, 1000.0),
        scale_update: str = 'ecm',
        max_iter: int = 1000,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        init_means='kmeans',
        init_weights='uniform',
        init_covariances='empirical',
        random_state: Optional[int] = None,
        warm_start: bool = False,
        algorithm: str = 'em',
        batch_size: Optional[int] = None,
        temperature: Optional[Tuple[float, int]] = None,
        convergence: str = 'relative',
        step_size=None,
        sem_burn_in: Optional[int] = None,
        sem_estimate: str = 'best',
        components_per_class=1,
        unlabeled_weight: float = 1.0,
        label_confidence: float = 1.0,
        noise: bool = False,
        noise_volume='bbox',
        noise_prior: float = 0.05,
        noise_init=None,
        missing: str = 'error',
        verbose: bool = False,
        verbose_interval: int = 10,
        device: Optional[str] = None,
    ):
        super().__init__(
            n_components=n_components, n_features=n_features, max_iter=max_iter, tol=tol,
            reg_covar=reg_covar, n_init=n_init, init_means=init_means, init_weights=init_weights,
            random_state=random_state, warm_start=warm_start, algorithm=algorithm,
            batch_size=batch_size, temperature=temperature, convergence=convergence,
            step_size=step_size, sem_burn_in=sem_burn_in, sem_estimate=sem_estimate,
            components_per_class=components_per_class, unlabeled_weight=unlabeled_weight,
            label_confidence=label_confidence,
            noise=noise, noise_volume=noise_volume, noise_prior=noise_prior, noise_init=noise_init,
            missing=missing,
            verbose=verbose, verbose_interval=verbose_interval, device=device,
        )

        if covariance_type not in _VALID_COVARIANCE_TYPES:
            raise ValueError(f"covariance_type must be one of {_VALID_COVARIANCE_TYPES}, got {covariance_type!r}")
        self.covariance_type = covariance_type
        self.init_covariances = init_covariances

        if not (isinstance(dof, (int, float)) or dof in ('per_component', 'shared')):
            raise ValueError(f"dof must be 'per_component', 'shared', or a positive float, got {dof!r}")
        if isinstance(dof, (int, float)) and dof <= 0:
            raise ValueError(f"dof must be > 0, got {dof}.")
        self.dof = dof
        self.dof_init = dof_init
        if not (0.0 < dof_bounds[0] < dof_bounds[1]):
            raise ValueError(f"dof_bounds must be (lo, hi) with 0 < lo < hi, got {dof_bounds!r}")
        self.dof_bounds = dof_bounds

        if scale_update not in _VALID_SCALE_UPDATES:
            raise ValueError(f"scale_update must be one of {_VALID_SCALE_UPDATES}, got {scale_update!r}")
        if self.missing == 'em' and scale_update == 'ecme':
            raise NotImplementedError(
                "missing='em' requires scale_update='ecm': the 'ecme' normalisation relies on a fixed-point "
                "identity that does not hold when rows have different numbers of observed entries."
            )
        self.scale_update = scale_update

        self.dof_ = None

    # ===================================================================
    # Parameter allocation
    # ===================================================================
    def _allocate_parameters(self, X: Optional[torch.Tensor] = None, set_random_state: bool = True):
        if set_random_state and self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        # ---- means ----
        if isinstance(self.init_means, torch.Tensor):
            if self.init_means.shape != (self.n_components, self.n_features):
                raise ValueError(
                    f"init_means tensor must be shape ({self.n_components}, {self.n_features}), "
                    f"got {self.init_means.shape}."
                )
            self.means_ = self.init_means.to(device=self.device, dtype=self.dtype)
        elif isinstance(self.init_means, str):
            if X is None:
                self.means_ = torch.randn(self.n_components, self.n_features, device=self.device, dtype=self.dtype)
            else:
                X_cpu = X.cpu()
                method = self.init_means.lower()
                if method == 'kmeans':
                    self.means_ = GMMInitializer.kmeans(X_cpu, self.n_components).to(self.device)
                elif method == 'kpp':
                    self.means_ = GMMInitializer.kpp(X_cpu, self.n_components).to(self.device)
                elif method == 'points':
                    self.means_ = GMMInitializer.points(X_cpu, self.n_components).to(self.device)
                elif method == 'maxdist':
                    self.means_ = GMMInitializer.maxdist(X_cpu, self.n_components).to(self.device)
                elif method == 'random':
                    self.means_ = GMMInitializer.random(X_cpu, self.n_components).to(self.device)
                else:
                    raise ValueError(
                        f"Unsupported init_means method: {method!r}. "
                        "Supported: 'kmeans', 'kpp', 'points', 'maxdist', 'random'."
                    )
        else:
            raise TypeError(f"init_means must be a string or torch.Tensor, got {type(self.init_means)}")
        self.means_ = self.means_.to(dtype=self.dtype)

        # ---- weights ----
        if isinstance(self.init_weights, torch.Tensor):
            if self.init_weights.shape != (self.n_components,):
                raise ValueError(f"init_weights tensor must be shape ({self.n_components},), got {self.init_weights.shape}.")
            weights = self.init_weights.to(device=self.device, dtype=self.dtype)
            if torch.sum(weights) < 1e-20:
                raise ValueError("Initial weights must sum to > 0.")
            self.weights_ = weights / torch.sum(weights)
        elif isinstance(self.init_weights, str):
            method = self.init_weights.lower()
            if method in ('uniform', 'equal'):
                self.weights_ = GMMInitializer.init_weights_uniform(self.n_components, self.device, dtype=self.dtype)
            elif method == 'random':
                self.weights_ = GMMInitializer.init_weights_random(self.n_components, self.device, dtype=self.dtype)
            elif method == 'kmeans':
                if X is None:
                    warnings.warn("'kmeans' weight initialization requires data. Falling back to uniform weights.", UserWarning)
                    self.weights_ = GMMInitializer.init_weights_uniform(self.n_components, self.device, dtype=self.dtype)
                else:
                    self.weights_ = GMMInitializer.init_weights_from_clusters(X.to(self.means_.device), self.means_).to(self.device)
            else:
                raise ValueError(f"Unsupported init_weights method: {method!r}. Supported: 'uniform', 'random', 'kmeans'.")
        else:
            raise TypeError(f"init_weights must be a string or torch.Tensor, got {type(self.init_weights)}")

        # ---- scale (covariances_) ----
        if isinstance(self.init_covariances, torch.Tensor):
            self.covariances_ = self.init_covariances.to(device=self.device, dtype=self.dtype)
        elif isinstance(self.init_covariances, str):
            method = self.init_covariances.lower()
            if method == 'empirical':
                if X is None:
                    raise ValueError("init_covariances='empirical' requires data (X).")
                self.covariances_ = GMMInitializer.init_covariances_empirical(
                    X, self.means_, self.covariance_type, self.reg_covar
                ).to(device=self.device, dtype=self.dtype)
            elif method == 'global':
                if X is None:
                    raise ValueError("init_covariances='global' requires data (X).")
                self.covariances_ = GMMInitializer.init_covariances_global(
                    X, self.n_components, self.covariance_type, self.reg_covar
                ).to(device=self.device, dtype=self.dtype)
            elif method == 'eye':
                self.covariances_ = GMMInitializer.init_covariances_eye(
                    self.n_components, self.n_features, self.covariance_type, self.reg_covar,
                    self.device, dtype=self.dtype,
                )
            else:
                raise ValueError(f"Unsupported init_covariances method: {method!r}. Supported: 'empirical', 'global', 'eye'.")
        else:
            raise TypeError(f"init_covariances must be a string or torch.Tensor, got {type(self.init_covariances)}")

        # ---- degrees of freedom ----
        if isinstance(self.dof, (int, float)):
            dof_value = float(self.dof)
        else:
            dof_value = float(self.dof_init)
        self.dof_ = torch.full((self.n_components,), dof_value, device=self.device, dtype=self.dtype)

        # A fresh allocation starts a fresh fit's bookkeeping -- without
        # this, n_init > 1 could leak a prior run's converged_=True onto a
        # later run that never actually converged (see GaussianMixture's
        # own _allocate_parameters, which does the same reset).
        self.fitted_ = False
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")

    # ===================================================================
    # Snapshotting: dof_ is the one extra fitted tensor beyond
    # weights_/means_/covariances_ -- see BaseMixture._get_param_snapshot.
    # ===================================================================
    def _get_param_snapshot(self) -> dict:
        snapshot = super()._get_param_snapshot()
        snapshot['dof_'] = self.dof_.clone()
        return snapshot

    def _is_tied_covariance(self) -> bool:
        return self.covariance_type == 'tied_full'

    # ===================================================================
    # E-step hook
    # ===================================================================
    def _component_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        .. math::
            \log t(x|\mu_k,\Sigma_k,\nu_k) = \ln\Gamma\!\left(\tfrac{\nu_k+d}{2}\right)
                - \ln\Gamma\!\left(\tfrac{\nu_k}{2}\right) - \tfrac{d}{2}\ln(\nu_k\pi)
                - \tfrac12 \ln|\Sigma_k| - \tfrac{\nu_k+d}{2}\ln\!\left(1 + \tfrac{\delta_{ik}}{\nu_k}\right)
        """
        if self._missing_active(X):
            return self._component_log_prob_missing(X)
        delta, logdet = _mahalanobis_and_logdet(X, self.means_, self.covariances_, self.covariance_type)
        d = self.n_features
        nu = self.dof_  # (K,)

        const = (
            torch.lgamma((nu + d) / 2) - torch.lgamma(nu / 2)
            - (d / 2.0) * torch.log(nu * math.pi) - 0.5 * logdet
        )  # (K,)
        tail = -((nu + d) / 2).unsqueeze(0) * torch.log1p(delta / nu.unsqueeze(0))  # (n, K)
        return const.unsqueeze(0) + tail

    # ===================================================================
    # Missing values (v0.5 item 1.2): see tgmm/mixtures/missing.py and
    # docs/superpowers/specs/2026-10-15-v05-missing-values-design.md.
    # ===================================================================
    _supports_missing = True

    def _missing_cov_repr(self):
        r"""`(kind, cov)` for `missing.gaussian_blocks`: dense `(K, d, d)` or per-feature variances `(K, d)`."""
        K, d = self.n_components, self.n_features
        if self.covariance_type == 'full':
            return 'dense', self.covariances_
        if self.covariance_type == 'tied_full':
            return 'dense', self.covariances_.unsqueeze(0).expand(K, d, d)
        if self.covariance_type == 'diag':
            return 'diag', self.covariances_
        return 'diag', self.covariances_.unsqueeze(1).expand(K, d)  # spherical

    def _missing_u(self, X: torch.Tensor):
        r"""
        Observed-block quantities for incomplete data: the blocks, per-row number of observed
        entries `p` `(n, 1)`, and the robustness weights
        :math:`u_{ik} = (\nu_k + p_i)/(\nu_k + \delta_{o,ik})`, shape `(n, K)`.
        """
        kind, cov = self._missing_cov_repr()
        blocks = _missing.gaussian_blocks(X, self.means_, cov, kind, need_conditional=True)
        n, K = X.shape[0], self.means_.shape[0]
        delta = torch.empty(n, K, dtype=X.dtype, device=X.device)
        p = torch.empty(n, 1, dtype=X.dtype, device=X.device)
        for b in blocks:
            delta[b.rows] = b.delta
            p[b.rows] = float(b.obs.numel())
        nu = self.dof_.unsqueeze(0)
        return blocks, p, (nu + p) / (nu + delta)

    def _component_log_prob_missing(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Marginal t log-density of the observed block (:math:`d \to p`, :math:`\delta \to \delta_o`):

        .. math::
            \ln\Gamma\!\left(\tfrac{\nu+p}{2}\right) - \ln\Gamma\!\left(\tfrac{\nu}{2}\right)
            - \tfrac p2\ln(\nu\pi) - \tfrac12\ln|\Sigma_{oo}| - \tfrac{\nu+p}{2}\ln\!\left(1+\tfrac{\delta_o}{\nu}\right)
        """
        kind, cov = self._missing_cov_repr()
        nu = self.dof_
        out = torch.empty(X.shape[0], self.means_.shape[0], dtype=X.dtype, device=X.device)
        for b in _missing.gaussian_blocks(X, self.means_, cov, kind, need_conditional=False):
            p = float(b.obs.numel())
            const = (
                torch.lgamma((nu + p) / 2) - torch.lgamma(nu / 2) - (p / 2.0) * torch.log(nu * math.pi) - 0.5 * b.logdet
            )
            out[b.rows] = const.unsqueeze(0) - ((nu + p) / 2).unsqueeze(0) * torch.log1p(b.delta / nu.unsqueeze(0))
        return out

    def _compute_stats_missing(self, X: torch.Tensor, resp: torch.Tensor) -> SufficientStats:
        r"""
        ECM statistics given the observed entries (Lin, Lee & Ho 2006), at the current parameters:
        :math:`u_{ik}=(\nu_k+p_i)/(\nu_k+\delta_{o,ik})`, :math:`s1_k=\sum_i\tau_{ik}u_{ik}\tilde x_{ik}`,

        .. math::
            s2_k = \sum_i \tau_{ik}u_{ik}(\tilde x_{ik}-\bar y_k)(\tilde x_{ik}-\bar y_k)^\top
                   + \sum_i \tau_{ik} C_{k,\mathrm{pattern}(i)}

        (the conditional covariance `C` carries **no** `u` factor because :math:`E[w\cdot 1/w]=1`), and
        `aux['log_u_minus_u']` holds :math:`\sum_i\tau_{ik}[\psi(\tfrac{\nu_k+p_i}2)-\ln\tfrac{\nu_k+p_i}2
        +\ln u_{ik}-u_{ik}]`, i.e. :math:`\sum\tau(E[\ln w]-E[w])`, for the exact `nu` update
        (marked by `aux['dof_exact']`). `batch_size` does not chunk this path (as in the complete-data one).
        """
        n_samples, d = X.shape
        n_components = resp.shape[1]
        kind, _ = self._missing_cov_repr()
        blocks, p, u = self._missing_u(X)
        nu = self.dof_.unsqueeze(0)

        tau = resp
        tau_u = tau * u
        nk = tau.sum(dim=0)
        m_k = tau_u.sum(dim=0)
        Xhat = _missing.complete_rows(X, blocks, n_components)  # (n, K, d)
        s1 = torch.einsum('nk,nkd->kd', tau_u, Xhat)
        ybar = s1 / (m_k + 1e-20).unsqueeze(1)

        base_type = self.covariance_type.replace('tied_', '', 1)
        diff = Xhat - ybar.unsqueeze(0)
        if base_type == 'full':
            s2 = torch.einsum('nk,nkd,nke->kde', tau_u, diff, diff)
        elif base_type == 'diag':
            s2 = torch.einsum('nk,nkd->kd', tau_u, diff.pow(2))
        else:  # spherical
            s2 = torch.einsum('nk,nk->k', tau_u, diff.pow(2).sum(dim=2))
        extra = _missing.conditional_cov_sum(blocks, tau, kind, d)
        if base_type == 'spherical':
            extra = extra.sum(dim=-1)
        s2 = s2 + extra

        half = (nu + p) / 2
        e = torch.special.digamma(half) - torch.log(half) + torch.log(u) - u
        aux = {
            'tau_u_sum': m_k,
            'log_u_minus_u': (tau * e).sum(dim=0),
            'dof_exact': torch.ones((), dtype=X.dtype, device=X.device),
        }
        return SufficientStats(nk=nk, s1=s1, s2=s2, n_samples=n_samples, aux=aux)

    # ===================================================================
    # M-step hooks
    # ===================================================================
    def _compute_stats(self, X: torch.Tensor, resp: torch.Tensor, batch_size: Optional[int] = None) -> SufficientStats:
        r"""
        Accumulates the ECM statistics: :math:`s1 = \sum_i \tau_{ik} u_{ik} x_i`,
        :math:`s2 = \sum_i \tau_{ik} u_{ik} (x_i - \bar y_k)(x_i - \bar y_k)^T`
        (centered on :math:`\bar y_k = s1_k / \sum_i \tau_{ik} u_{ik}`, this
        E-step's own u-weighted mean estimate), plus `aux['tau_u_sum']`
        (:math:`\sum_i \tau_{ik} u_{ik}`) and `aux['log_u_minus_u']`
        (:math:`\sum_i \tau_{ik}(\log u_{ik} - u_{ik})`), the statistic the
        `nu_k` root-find needs. `nk` keeps its usual meaning,
        :math:`\sum_i \tau_{ik}`.

        `batch_size` is accepted for interface parity with
        `BaseMixture._m_step`/online-EM but not separately chunked here
        (unlike `GaussianMixture._compute_stats`): Student-t's per-sample
        `u_ik` already requires `delta`, itself an `(n, K)` tensor, in
        memory at once, so chunking rows would save only the final
        reduction, not the dominant cost.
        """
        if self._missing_active(X):
            return self._compute_stats_missing(X, resp)
        n_samples, d = X.shape
        delta, _ = _mahalanobis_and_logdet(X, self.means_, self.covariances_, self.covariance_type)
        nu = self.dof_  # (K,)
        u = (nu.unsqueeze(0) + d) / (nu.unsqueeze(0) + delta)  # (n, K)

        tau = resp
        tau_u = tau * u

        nk = tau.sum(dim=0)
        m_k = tau_u.sum(dim=0)  # sum_i tau_ik u_ik
        s1 = tau_u.t() @ X  # (K, d)
        ybar = s1 / (m_k + 1e-20).unsqueeze(1)

        log_u_minus_u = (tau * (torch.log(u) - u)).sum(dim=0)  # (K,)

        base_type = self.covariance_type.replace('tied_', '', 1)
        diff = X.unsqueeze(1) - ybar.unsqueeze(0)  # (n, K, d)
        if base_type == 'full':
            s2 = torch.einsum('nk,nkd,nke->kde', tau_u, diff, diff)
        elif base_type == 'diag':
            s2 = torch.einsum('nk,nkd->kd', tau_u, diff.pow(2))
        elif base_type == 'spherical':
            s2 = torch.einsum('nk,nk->k', tau_u, diff.pow(2).sum(dim=2))
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

        aux = {'tau_u_sum': m_k, 'log_u_minus_u': log_u_minus_u}
        return SufficientStats(nk=nk, s1=s1, s2=s2, n_samples=n_samples, aux=aux)

    def _m_step_from_stats(self, stats: SufficientStats):
        n_samples = stats.n_samples
        nk = stats.nk + 1e-20  # sum_i tau_ik
        m_k = stats.aux['tau_u_sum'] + 1e-20  # sum_i tau_ik u_ik

        self.weights_ = stats.nk / n_samples
        self.weights_.clamp_(min=1e-20)

        self.means_ = stats.s1 / m_k.unsqueeze(1)

        denom = nk if self.scale_update == 'ecm' else m_k
        base_type = self.covariance_type.replace('tied_', '', 1)

        if self.covariance_type.startswith('tied_'):
            pooled_scatter = stats.s2.sum(dim=0)
            pooled_denom = denom.sum()
            cov = pooled_scatter / pooled_denom
            if base_type == 'full':
                cov = cov + self.reg_covar * torch.eye(self.n_features, device=self.device, dtype=self.dtype)
            else:
                cov = cov + self.reg_covar
            self.covariances_ = cov
        else:
            if base_type == 'full':
                cov = stats.s2 / denom.view(-1, 1, 1)
                eye = torch.eye(self.n_features, device=self.device, dtype=self.dtype).unsqueeze(0)
                cov = cov + self.reg_covar * eye
            elif base_type == 'diag':
                cov = stats.s2 / denom.unsqueeze(1) + self.reg_covar
            else:  # spherical
                cov = stats.s2 / denom + self.reg_covar
            self.covariances_ = cov

        self._update_dof(stats, nk)

    def _update_dof(self, stats: SufficientStats, nk: torch.Tensor):
        d = self.n_features
        if isinstance(self.dof, (int, float)):
            return  # fixed dof: never updated

        log_u_minus_u = stats.aux['log_u_minus_u']
        exact = 'dof_exact' in stats.aux  # incomplete data: textbook form, see `_solve_dof`
        if self.dof == 'per_component':
            c = log_u_minus_u / nk
            self.dof_ = self._solve_dof(c, d, exact=exact)
        else:  # 'shared'
            c_shared = log_u_minus_u.sum() / nk.sum()
            nu_shared = self._solve_dof(c_shared.unsqueeze(0), d, exact=exact)
            self.dof_ = nu_shared.expand(self.n_components).clone()

    def _solve_dof(self, c: torch.Tensor, d: int, exact: bool = False) -> torch.Tensor:
        r"""
        Solve, elementwise over `c`'s shape,

        .. math::
            1 - \psi(\nu/2) + \ln(\nu/2) + c + \psi\!\left(\tfrac{\nu+d}{2}\right)
                - \ln\!\left(\tfrac{\nu+d}{2}\right) = 0

        for :math:`\nu`, by bisection within `self.dof_bounds` followed by
        a few Newton steps on :math:`\ln\nu` (using `torch.special.digamma`/
        `polygamma` for the derivative). Bisection alone would already be
        accurate enough in practice; the Newton polish tightens it per the
        plan's "bisection then Newton" spec.

        With `exact=True` (incomplete data, where `c` already contains the
        :math:`\psi((\nu_{old}+p_i)/2)-\ln((\nu_{old}+p_i)/2)` terms of each row's own
        number of observed entries `p_i`) the last two terms are dropped:
        :math:`1-\psi(\nu/2)+\ln(\nu/2)+c=0`.
        """
        def f(nu):
            val = 1.0 - torch.special.digamma(nu / 2) + torch.log(nu / 2) + c
            if exact:
                return val
            return val + torch.special.digamma((nu + d) / 2) - torch.log((nu + d) / 2)

        lo = torch.full_like(c, self.dof_bounds[0])
        hi = torch.full_like(c, self.dof_bounds[1])
        f_lo = f(lo)

        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            same_sign = torch.sign(f_mid) == torch.sign(f_lo)
            lo = torch.where(same_sign, mid, lo)
            hi = torch.where(same_sign, hi, mid)
            f_lo = torch.where(same_sign, f_mid, f_lo)

        nu = 0.5 * (lo + hi)
        log_nu = torch.log(nu)
        log_lo, log_hi = math.log(self.dof_bounds[0]), math.log(self.dof_bounds[1])
        for _ in range(4):
            nu_cur = torch.exp(log_nu)
            f_val = f(nu_cur)
            fprime = -0.5 * torch.special.polygamma(1, nu_cur / 2) + 1.0 / nu_cur
            if not exact:
                fprime = fprime + 0.5 * torch.special.polygamma(1, (nu_cur + d) / 2) - 1.0 / (nu_cur + d)
            d_f_d_lognu = nu_cur * fprime
            step = f_val / (d_f_d_lognu - torch.sign(d_f_d_lognu) * 1e-20 - 1e-20)
            log_nu = torch.clamp(log_nu - step, log_lo, log_hi)

        return torch.exp(log_nu).clamp(self.dof_bounds[0], self.dof_bounds[1])

    # ===================================================================
    # Model complexity / sampling
    # ===================================================================
    def _n_parameters(self) -> int:
        K, d = self.n_components, self.n_features
        if self.covariance_type == 'full':
            cov_params = K * d * (d + 1) / 2.0
        elif self.covariance_type == 'diag':
            cov_params = K * d
        elif self.covariance_type == 'spherical':
            cov_params = K
        else:  # tied_full
            cov_params = d * (d + 1) / 2.0

        if self.dof == 'per_component':
            dof_params = K
        elif self.dof == 'shared':
            dof_params = 1
        else:
            dof_params = 0

        mean_params = K * d
        weight_params = K - 1
        noise_params = 1 if self.noise else 0
        return int(cov_params + mean_params + weight_params + dof_params + noise_params)

    def _sample_component(self, k: int, n: int) -> torch.Tensor:
        d = self.n_features
        mean = self.means_[k]
        if self.covariance_type == 'full':
            cov = self.covariances_[k]
        elif self.covariance_type == 'diag':
            cov = torch.diag_embed(self.covariances_[k])
        elif self.covariance_type == 'spherical':
            cov = torch.eye(d, device=self.device, dtype=self.dtype) * self.covariances_[k]
        else:  # tied_full
            cov = self.covariances_

        L = torch.linalg.cholesky(cov)
        z = torch.randn(n, d, device=self.device, dtype=self.dtype) @ L.t()

        nu = self.dof_[k]
        gamma = torch.distributions.Gamma(nu / 2, nu / 2)
        g = gamma.sample((n,)).to(device=self.device, dtype=self.dtype)
        return mean.unsqueeze(0) + z / torch.sqrt(g).unsqueeze(-1)

    # ===================================================================
    # Marginalising and conditioning (v0.5 item 1.3)
    # ===================================================================
    def marginal(self, dims) -> "StudentTMixture":
        r"""
        The marginal mixture over a subset of the coordinates. Exact: a
        multivariate t marginal has the same degrees of freedom
        :math:`\nu_k` and the restricted scale matrix
        :math:`\Sigma_k[\text{dims},\text{dims}]`; weights, means and `dof_` are
        carried over.

        Parameters
        ----------
        dims : sequence of int
            Distinct coordinates in ``[0, n_features)``; output coordinate
            ``j`` is input coordinate ``dims[j]``.

        Returns
        -------
        StudentTMixture
            A new fitted model with the same `covariance_type`.
        """
        if self.noise:
            raise NotImplementedError(
                "marginal is not supported for noise=True models: the uniform noise component has no "
                "counterpart after marginalising."
            )
        if not self.fitted_ or self.means_ is None:
            raise RuntimeError("marginal requires a fitted model: call fit() first.")
        K, d = self.means_.shape
        dims = _validate_dims(dims, d, 'dims')
        idx = torch.tensor(dims, dtype=torch.long, device=self.means_.device)
        cov, t = self.covariances_, self.covariance_type
        if t == 'full':
            new_cov = cov[:, idx][:, :, idx]
        elif t == 'diag':
            new_cov = cov[:, idx]
        elif t == 'spherical':
            new_cov = cov.clone()
        else:  # 'tied_full'
            new_cov = cov[idx][:, idx]
        model = StudentTMixture(
            n_components=K, n_features=len(dims), covariance_type=t, dof=self.dof, dof_init=self.dof_init,
            dof_bounds=self.dof_bounds, scale_update=self.scale_update, reg_covar=self.reg_covar,
            missing=self.missing, device=self.device,
        )
        model.dtype = self.dtype
        model.weights_ = self.weights_.clone()
        model.means_ = self.means_[:, idx].clone()
        model.covariances_ = new_cov.clone()
        model.dof_ = self.dof_.clone()
        model.fitted_ = True
        model.converged_ = True
        model.n_components_ = K
        model.n_iter_ = 0
        model.lower_bound_ = float('nan')
        return model

    def condition(self, x_obs, obs_dims):
        r"""
        Not implemented: the conditional of a multivariate t given observed
        coordinates is a t with :math:`\nu_k + d_o` degrees of freedom and a
        scale inflated by :math:`(\nu_k + \delta_o)/(\nu_k + d_o)` that depends
        on the sample, so it is not a member of `ConditionalGaussianMixture`.
        """
        raise NotImplementedError(
            "condition is not implemented for StudentTMixture: the conditional t needs adjusted degrees of freedom "
            "(nu + d_obs) and a per-sample scale ((nu + delta_obs)/(nu + d_obs)). Use marginal(), or "
            "a GaussianMixture for Gaussian mixture regression."
        )

    def sample_weights(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        The robustness weight :math:`u_{i\hat z_i}` at each sample's
        predicted component :math:`\hat z_i = \arg\max_k \tau_{ik}`. Small
        values flag likely outliers -- points far enough from their
        assigned component's center that the t-mixture downweighted them
        relative to what a Gaussian fit would have done.

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)

        Returns
        -------
        torch.Tensor, shape (n_samples,)
        """
        if self._missing_active(X.to(self.device)):
            X = X.to(self.device)
            resp, _ = self._e_step(X)
            z_hat = torch.argmax(resp, dim=1)
            _, _, u = self._missing_u(X)
            return u[torch.arange(X.shape[0], device=self.device), z_hat]
        X = X.to(self.device)
        resp, _ = self._e_step(X)
        z_hat = torch.argmax(resp, dim=1)
        delta, _ = _mahalanobis_and_logdet(X, self.means_, self.covariances_, self.covariance_type)
        u = (self.dof_.unsqueeze(0) + self.n_features) / (self.dof_.unsqueeze(0) + delta)
        return u[torch.arange(X.shape[0], device=self.device), z_hat]

    # ===================================================================
    # Persistence -- independent of GaussianMixture's, see the design doc.
    # ===================================================================
    def to_dict(self) -> dict:
        return {
            'n_components': self.n_components,
            'n_features': self.n_features,
            'covariance_type': self.covariance_type,
            'dof': self.dof,
            'dof_init': self.dof_init,
            'dof_bounds': self.dof_bounds,
            'scale_update': self.scale_update,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'reg_covar': self.reg_covar,
            'n_init': self.n_init,
            'init_means': self.init_means,
            'init_weights': self.init_weights,
            'init_covariances': self.init_covariances,
            'random_state': self.random_state,
            'warm_start': self.warm_start,
            'algorithm': self.algorithm,
            'batch_size': self.batch_size,
            'temperature': self.temperature,
            'convergence': self.convergence,
            'step_size': self.step_size,
            'sem_burn_in': self.sem_burn_in,
            'sem_estimate': self.sem_estimate,
            'verbose': self.verbose,
            'verbose_interval': self.verbose_interval,
            'noise': self.noise,
            'noise_volume': self.noise_volume,
            'noise_prior': self.noise_prior,
            'noise_weight_': self.noise_weight_,
            'noise_bounds_': self.noise_bounds_,
            'missing': self.missing,
            'weights_': self.weights_,
            'means_': self.means_,
            'covariances_': self.covariances_,
            'dof_': self.dof_,
            'fitted_': self.fitted_,
            'converged_': self.converged_,
            'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_,
            'n_components_': self.n_components_,
        }

    def save(self, filepath: str):
        r"""Save this model to `filepath` (see `load`)."""
        torch.save(self.to_dict(), filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "StudentTMixture":
        r"""Load a model previously saved with `save`."""
        if device is None:
            state = torch.load(filepath, weights_only=False)
        else:
            state = torch.load(filepath, map_location=device, weights_only=False)

        model = cls(
            n_components=state['n_components'], n_features=state['n_features'],
            covariance_type=state['covariance_type'], dof=state['dof'], dof_init=state['dof_init'],
            dof_bounds=state['dof_bounds'], scale_update=state['scale_update'],
            max_iter=state['max_iter'], tol=state['tol'], reg_covar=state['reg_covar'],
            n_init=state['n_init'], init_means=state['init_means'], init_weights=state['init_weights'],
            init_covariances=state['init_covariances'], random_state=state['random_state'],
            warm_start=state['warm_start'], algorithm=state['algorithm'], batch_size=state['batch_size'],
            temperature=state['temperature'], convergence=state['convergence'], step_size=state['step_size'],
            sem_burn_in=state['sem_burn_in'], sem_estimate=state['sem_estimate'],
            verbose=state['verbose'], verbose_interval=state['verbose_interval'], device=device,
            noise=state.get('noise', False), noise_volume=state.get('noise_volume', 'bbox'),
            noise_prior=state.get('noise_prior', 0.05), missing=state.get('missing', 'error'),
        )
        model.noise_weight_ = state.get('noise_weight_')
        model.noise_bounds_ = state.get('noise_bounds_')
        model.weights_ = state['weights_']
        model.means_ = state['means_']
        model.covariances_ = state['covariances_']
        model.dof_ = state['dof_']
        model.fitted_ = state['fitted_']
        model.converged_ = state['converged_']
        model.n_iter_ = state['n_iter_']
        model.lower_bound_ = state['lower_bound_']
        model.n_components_ = state['n_components_']
        return model

    def save_state_dict(self) -> dict:
        r"""Alias for `to_dict` -- parity with `GaussianMixture`'s naming."""
        return self.to_dict()

    def load_state_dict(self, state_dict: dict):
        r"""In-place counterpart to `load`: restores this instance's fitted state from `state_dict`."""
        self.noise = state_dict.get('noise', False)
        self.noise_volume = state_dict.get('noise_volume', 'bbox')
        self.noise_prior = state_dict.get('noise_prior', 0.05)
        self.missing = state_dict.get('missing', 'error')
        self.noise_weight_ = state_dict.get('noise_weight_')
        self.noise_bounds_ = state_dict.get('noise_bounds_')
        self.weights_ = state_dict['weights_']
        self.means_ = state_dict['means_']
        self.covariances_ = state_dict['covariances_']
        self.dof_ = state_dict['dof_']
        self.fitted_ = state_dict['fitted_']
        self.converged_ = state_dict['converged_']
        self.n_iter_ = state_dict['n_iter_']
        self.lower_bound_ = state_dict['lower_bound_']
        self.n_components_ = state_dict['n_components_']
