r"""
`VonMisesFisherMixture`: a mixture of von Mises-Fisher distributions on the
unit hypersphere (Banerjee, Dhillon, Ghosh & Sra, 2005), for L2-normalized
data such as text/image embeddings. Each component has a mean direction
`mu_k` (unit norm) and concentration `kappa_k`; higher `kappa_k` means a
tighter cluster around `mu_k`.

See docs/superpowers/specs/2026-10-14-phase4-base-mixture-studentt-vmf-design.md
section 4.3.
"""
import math
import warnings
from typing import Optional, Tuple

import torch
from scipy.special import ive

from ..stats import SufficientStats
from .base import BaseMixture

_VALID_CONCENTRATIONS = ('per_component', 'shared')
_KAPPA_BOUNDS = (1e-3, 1e5)


def _log_bessel_iv(nu: float, kappa: float) -> float:
    r"""
    :math:`\log I_\nu(\kappa)`, via `scipy.special.ive` (`I_\nu(\kappa)
    e^{-\kappa}`) when it hasn't underflowed to 0, falling back to the
    uniform asymptotic expansion for large order (Abramowitz & Stegun
    9.7.7) when it has -- which happens for large `nu` (high-dimensional
    data) and modest `kappa`. `nu`/`kappa` are plain Python floats: this is
    always called once per mixture component (a handful of scalars), never
    batched over samples, so there's nothing to vectorize.
    """
    ive_val = ive(nu, kappa)
    if ive_val > 0.0:
        return math.log(ive_val) + kappa

    z = kappa / nu
    sqrt_term = math.sqrt(1.0 + z * z)
    eta = sqrt_term + math.log(z / (1.0 + sqrt_term))
    return nu * eta - 0.5 * math.log(2.0 * math.pi * nu) - 0.25 * math.log(1.0 + z * z)


def _bessel_ratio_A(kappa: float, d: int) -> float:
    r""":math:`A_d(\kappa) = I_{d/2}(\kappa) / I_{d/2-1}(\kappa)` (Sra, 2012)."""
    log_num = _log_bessel_iv(d / 2.0, kappa)
    log_den = _log_bessel_iv(d / 2.0 - 1.0, kappa)
    return math.exp(log_num - log_den)


def _log_vmf_normalizer(kappa: torch.Tensor, d: int) -> torch.Tensor:
    r"""
    :math:`\log C_d(\kappa) = (d/2-1)\log\kappa - (d/2)\log(2\pi) - \log I_{d/2-1}(\kappa)`,
    elementwise over `kappa`'s shape (looping in Python over components --
    see `_log_bessel_iv`).
    """
    nu = d / 2.0 - 1.0
    kappa_cpu = kappa.detach().cpu().double()
    out = torch.empty_like(kappa_cpu)
    for i in range(kappa_cpu.numel()):
        kap = kappa_cpu.view(-1)[i].item()
        log_iv = _log_bessel_iv(nu, kap)
        out.view(-1)[i] = nu * math.log(kap) - (d / 2.0) * math.log(2.0 * math.pi) - log_iv
    return out.to(device=kappa.device, dtype=kappa.dtype).reshape(kappa.shape)


def _solve_kappa(r_bar: torch.Tensor, d: int) -> torch.Tensor:
    r"""
    Concentration MLE for mean resultant length `r_bar` (elementwise):
    the Banerjee et al. (2005) closed-form approximation
    :math:`\kappa \approx \bar R(d - \bar R^2)/(1-\bar R^2)`, refined by two
    Newton steps on :math:`A_d(\kappa) - \bar R = 0` using
    :math:`A_d'(\kappa) = 1 - A_d(\kappa)^2 - \tfrac{d-1}{\kappa}A_d(\kappa)`
    (Sra, 2012). Clamped to `_KAPPA_BOUNDS`.
    """
    r_bar_cpu = r_bar.detach().cpu().double()
    out = torch.empty_like(r_bar_cpu)
    for i in range(r_bar_cpu.numel()):
        rb = min(max(r_bar_cpu.view(-1)[i].item(), 0.0), 1.0 - 1e-10)
        if rb < 1e-10:
            out.view(-1)[i] = _KAPPA_BOUNDS[0]
            continue
        kappa = rb * (d - rb * rb) / (1.0 - rb * rb)
        kappa = min(max(kappa, _KAPPA_BOUNDS[0]), _KAPPA_BOUNDS[1])
        for _ in range(2):
            A = _bessel_ratio_A(kappa, d)
            a_prime = 1.0 - A * A - ((d - 1.0) / kappa) * A
            if abs(a_prime) < 1e-12:
                break
            kappa = kappa - (A - rb) / a_prime
            kappa = min(max(kappa, _KAPPA_BOUNDS[0]), _KAPPA_BOUNDS[1])
        out.view(-1)[i] = kappa
    return out.to(device=r_bar.device, dtype=r_bar.dtype).reshape(r_bar.shape)


def _spherical_kmeans_pp_init(X: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    r"""
    Cosine-distance k-means++ seeding: pick the first center uniformly at
    random, then each subsequent center with probability proportional to
    `1 - max cosine similarity to any already-chosen center` -- favoring
    points far (in cosine) from what's already picked. Plain uniform-random
    seeding (picking `k` random rows outright) can, for well-separated
    clusters, draw two initial centers from the same true cluster and
    permanently merge the other two; this fixes that failure mode the same
    way Euclidean k-means++ does for ordinary k-means.
    """
    n = X.shape[0]
    centers = torch.empty(k, X.shape[1], device=X.device, dtype=X.dtype)
    idx0 = torch.randint(0, n, (1,), generator=generator, device=X.device).item()
    centers[0] = X[idx0]
    closest_sim = X @ centers[0]

    for i in range(1, k):
        dist = (1.0 - closest_sim).clamp(min=0.0)
        total = dist.sum()
        if total < 1e-12:
            idx = torch.randint(0, n, (1,), generator=generator, device=X.device).item()
        else:
            probs = dist / total
            idx = torch.multinomial(probs, 1, generator=generator).item()
        centers[i] = X[idx]
        closest_sim = torch.maximum(closest_sim, X @ centers[i])

    return centers


def _spherical_kmeans(X: torch.Tensor, k: int, max_iter: int = 100, random_state: Optional[int] = None) -> torch.Tensor:
    r"""
    Spherical k-means (Dhillon & Modha, 2001): k-means with cosine
    similarity instead of Euclidean distance, and centroids renormalized to
    unit length after each update. `X` must already be unit-norm rows.
    """
    generator = torch.Generator(device=X.device)
    if random_state is not None:
        generator.manual_seed(random_state)
    centers = _spherical_kmeans_pp_init(X, k, generator)

    for _ in range(max_iter):
        sims = X @ centers.t()  # (n, k)
        labels = torch.argmax(sims, dim=1)
        new_centers = centers.clone()
        for j in range(k):
            mask = labels == j
            if torch.any(mask):
                c = X[mask].sum(dim=0)
                norm_c = c.norm()
                if norm_c > 1e-12:
                    new_centers[j] = c / norm_c
        if torch.allclose(new_centers, centers, atol=1e-7):
            centers = new_centers
            break
        centers = new_centers
    return centers


class VonMisesFisherMixture(BaseMixture):
    r"""
    A mixture of von Mises-Fisher distributions for L2-normalized data
    (Banerjee, Dhillon, Ghosh & Sra, 2005), fit by EM.

    Shares its entire EM/CEM/SEM/SAEM/online-EM loop, `n_init` best-run
    selection, and `predict`/`predict_proba`/`score`/`bic`/`aic`/`icl` with
    every other `BaseMixture` subclass (wrapped here only to normalize `X`
    first -- see `_check_unit_norm`). Gibbs-sampling mode and supervised
    fitting (`fit(X, labels=...)`) are Gaussian-only, same as
    `StudentTMixture`.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    n_features : int, optional
        Dimensionality of the data (the ambient dimension of the sphere).
        Inferred from `X` at fit time if `None`.
    concentration : {'per_component', 'shared'}, default='per_component'
        `'per_component'` fits one `kappa_k` per component; `'shared'`
        fits one `kappa` for every component, from the pooled mean
        resultant length over all components.
    kappa_init : float, default=10.0
        Initial `kappa_k` for every component, before the first M-step.
    init_means : {'kmeans', 'random', 'points'} or torch.Tensor, default='kmeans'
        `'kmeans'` runs spherical k-means (cosine similarity, see
        `_spherical_kmeans`) -- the vMF analogue of Euclidean k-means used
        for `GaussianMixture`/`StudentTMixture`. `'random'` draws uniform
        random points on the sphere; `'points'` picks random rows of `X`
        (already unit-norm). A tensor is normalized and used directly.
    init_weights : see `BaseMixture`.
    max_iter, tol, reg_covar, n_init, random_state, warm_start, algorithm,
    batch_size, temperature, convergence, step_size, sem_burn_in,
    sem_estimate, verbose, verbose_interval, device : see `BaseMixture`.
        `reg_covar` is accepted for interface parity but unused -- there is no
        covariance here to regularize.

    Attributes
    ----------
    weights_ : torch.Tensor, shape (n_components,)
    means_ : torch.Tensor, shape (n_components, n_features)
        Unit-norm mean directions `mu_k`.
    kappa_ : torch.Tensor, shape (n_components,)
        Fitted concentrations. All entries equal when `concentration='shared'`.
    """

    def __init__(
        self,
        n_components: int,
        n_features: Optional[int] = None,
        concentration: str = 'per_component',
        kappa_init: float = 10.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        init_means='kmeans',
        init_weights='uniform',
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
            verbose=verbose, verbose_interval=verbose_interval, device=device,
        )

        if concentration not in _VALID_CONCENTRATIONS:
            raise ValueError(f"concentration must be one of {_VALID_CONCENTRATIONS}, got {concentration!r}")
        self.concentration = concentration
        self.kappa_init = kappa_init
        self.kappa_ = None

    # ===================================================================
    # Unit-norm input handling: every public entry point that takes raw X
    # routes through here first, then delegates to BaseMixture.
    # ===================================================================
    def _check_unit_norm(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(device=self.device)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        norms = X.norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
            warnings.warn(
                "VonMisesFisherMixture requires L2-normalized rows (unit norm); "
                "normalizing the input automatically.",
                UserWarning,
            )
            X = X / norms.clamp(min=1e-12).unsqueeze(1)
        return X

    def fit(self, X: torch.Tensor, labels=None, max_iter=None, tol=None, random_state=None, warm_start=None):
        return super().fit(
            self._check_unit_norm(X), labels=labels, max_iter=max_iter, tol=tol,
            random_state=random_state, warm_start=warm_start,
        )

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return super().predict(self._check_unit_norm(X))

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        return super().predict_proba(self._check_unit_norm(X))

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        return super().score_samples(self._check_unit_norm(X))

    def score(self, X: torch.Tensor) -> float:
        return super().score(self._check_unit_norm(X))

    def partial_fit(self, X_batch: torch.Tensor):
        return super().partial_fit(self._check_unit_norm(X_batch))

    # ===================================================================
    # Parameter allocation
    # ===================================================================
    def _allocate_parameters(self, X: Optional[torch.Tensor] = None, set_random_state: bool = True):
        if set_random_state and self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        if isinstance(self.init_means, torch.Tensor):
            if self.init_means.shape != (self.n_components, self.n_features):
                raise ValueError(
                    f"init_means tensor must be shape ({self.n_components}, {self.n_features}), "
                    f"got {self.init_means.shape}."
                )
            means = self.init_means.to(device=self.device, dtype=self.dtype)
            self.means_ = means / means.norm(dim=1, keepdim=True).clamp(min=1e-12)
        elif isinstance(self.init_means, str):
            method = self.init_means.lower()
            if X is None:
                v = torch.randn(self.n_components, self.n_features, device=self.device, dtype=self.dtype)
                self.means_ = v / v.norm(dim=1, keepdim=True)
            elif method == 'kmeans':
                self.means_ = _spherical_kmeans(X, self.n_components, random_state=self.random_state)
            elif method == 'random':
                v = torch.randn(self.n_components, self.n_features, device=self.device, dtype=self.dtype)
                self.means_ = v / v.norm(dim=1, keepdim=True)
            elif method == 'points':
                idx = torch.randperm(X.shape[0], device=X.device)[:self.n_components]
                self.means_ = X[idx].clone()
            else:
                raise ValueError(f"Unsupported init_means method: {method!r}. Supported: 'kmeans', 'random', 'points'.")
        else:
            raise TypeError(f"init_means must be a string or torch.Tensor, got {type(self.init_means)}")

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
                self.weights_ = torch.full((self.n_components,), 1.0 / self.n_components, device=self.device, dtype=self.dtype)
            elif method == 'random':
                alpha = torch.ones(self.n_components, device=self.device, dtype=self.dtype)
                self.weights_ = torch.distributions.Dirichlet(alpha).sample()
            elif method == 'kmeans':
                if X is None:
                    warnings.warn("'kmeans' weight initialization requires data. Falling back to uniform weights.", UserWarning)
                    self.weights_ = torch.full((self.n_components,), 1.0 / self.n_components, device=self.device, dtype=self.dtype)
                else:
                    sims = X @ self.means_.t()
                    labels = torch.argmax(sims, dim=1)
                    counts = torch.bincount(labels, minlength=self.n_components).to(self.dtype).clamp(min=1e-10)
                    self.weights_ = counts / counts.sum()
            else:
                raise ValueError(f"Unsupported init_weights method: {method!r}. Supported: 'uniform', 'random', 'kmeans'.")
        else:
            raise TypeError(f"init_weights must be a string or torch.Tensor, got {type(self.init_weights)}")

        self.kappa_ = torch.full((self.n_components,), float(self.kappa_init), device=self.device, dtype=self.dtype)

        # A fresh allocation starts a fresh fit's bookkeeping -- without
        # this, n_init > 1 could leak a prior run's converged_=True onto a
        # later run that never actually converged (see GaussianMixture's
        # own _allocate_parameters, which does the same reset).
        self.fitted_ = False
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")

    # ===================================================================
    # Snapshotting: kappa_ is the extra fitted tensor beyond weights_/means_
    # (there is no covariances_ for this family).
    # ===================================================================
    def _get_param_snapshot(self) -> dict:
        return {
            'weights_': self.weights_.clone(),
            'means_': self.means_.clone(),
            'kappa_': self.kappa_.clone(),
        }

    # ===================================================================
    # E-step hook
    # ===================================================================
    def _component_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        r"""`_component_log_prob = log C_d(kappa_k) + kappa_k * (X . mu_k)`."""
        log_c = _log_vmf_normalizer(self.kappa_, self.n_features)  # (K,)
        cos_sim = X @ self.means_.t()  # (n, K)
        return log_c.unsqueeze(0) + self.kappa_.unsqueeze(0) * cos_sim

    # ===================================================================
    # M-step hooks
    # ===================================================================
    def _compute_stats(self, X: torch.Tensor, resp: torch.Tensor, batch_size: Optional[int] = None) -> SufficientStats:
        r"""
        `nk = sum_i tau_ik`; `s1 = r_k = sum_i tau_ik x_i` (the resultant
        vector each component's mean direction and concentration are
        derived from). `s2` is unused by this family -- set to a harmless
        placeholder so `SufficientStats`'s shared `__add__`/`scale` keep
        working unmodified for online-EM/SAEM blending.
        """
        nk = resp.sum(dim=0)
        s1 = resp.t() @ X
        return SufficientStats(nk=nk, s1=s1, s2=torch.zeros_like(nk), n_samples=X.shape[0])

    def _m_step_from_stats(self, stats: SufficientStats):
        n_samples = stats.n_samples
        nk = stats.nk + 1e-20
        d = self.n_features

        self.weights_ = stats.nk / n_samples
        self.weights_.clamp_(min=1e-20)

        r_norm = stats.s1.norm(dim=1)  # (K,)
        self.means_ = stats.s1 / r_norm.clamp(min=1e-20).unsqueeze(1)

        if self.concentration == 'per_component':
            r_bar = r_norm / nk
            self.kappa_ = _solve_kappa(r_bar, d)
        else:  # 'shared'
            r_bar_shared = r_norm.sum() / nk.sum()
            kappa_shared = _solve_kappa(r_bar_shared.unsqueeze(0), d)
            self.kappa_ = kappa_shared.expand(self.n_components).clone()

    # ===================================================================
    # Model complexity / sampling
    # ===================================================================
    def _n_parameters(self) -> int:
        K, d = self.n_components, self.n_features
        weight_params = K - 1
        mean_params = K * (d - 1)  # each mu_k lives on a (d-1)-sphere
        kappa_params = K if self.concentration == 'per_component' else 1
        return int(weight_params + mean_params + kappa_params)

    def _rotate_to(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        r"""Householder reflection mapping unit vector `a` to unit vector `b`, applied to each row of `x`."""
        v = a - b
        norm_v = v.norm()
        if norm_v < 1e-8:
            return x
        v = v / norm_v
        return x - 2.0 * (x @ v).unsqueeze(1) * v.unsqueeze(0)

    def _sample_component(self, k: int, n: int) -> torch.Tensor:
        r"""Wood's (1994) rejection sampler for vMF(mu_k, kappa_k)."""
        d = self.n_features
        kappa = float(self.kappa_[k].item())
        mu = self.means_[k]

        if kappa < 1e-6:
            v = torch.randn(n, d, device=self.device, dtype=self.dtype)
            return v / v.norm(dim=1, keepdim=True)

        b = (-2.0 * kappa + math.sqrt(4.0 * kappa * kappa + (d - 1.0) ** 2)) / (d - 1.0)
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa * x0 + (d - 1.0) * math.log(1.0 - x0 * x0)

        w = torch.empty(n, device=self.device, dtype=self.dtype)
        filled = torch.zeros(n, dtype=torch.bool, device=self.device)
        beta_dist = torch.distributions.Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0)

        while not bool(filled.all()):
            m = int((~filled).sum().item())
            z = beta_dist.sample((m,)).to(device=self.device, dtype=self.dtype)
            u = torch.rand(m, device=self.device, dtype=self.dtype)
            w_candidate = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            accept = (
                kappa * w_candidate + (d - 1.0) * torch.log((1.0 - x0 * w_candidate).clamp(min=1e-12)) - c
                >= torch.log(u.clamp(min=1e-12))
            )
            idx = (~filled).nonzero(as_tuple=True)[0]
            accepted_idx = idx[accept]
            w[accepted_idx] = w_candidate[accept]
            filled[accepted_idx] = True

        v = torch.randn(n, d - 1, device=self.device, dtype=self.dtype)
        v = v / v.norm(dim=1, keepdim=True)
        x_local = torch.cat([torch.sqrt((1.0 - w * w).clamp(min=0.0)).unsqueeze(1) * v, w.unsqueeze(1)], dim=1)

        e_d = torch.zeros(d, device=self.device, dtype=self.dtype)
        e_d[-1] = 1.0
        return self._rotate_to(x_local, e_d, mu)

    # ===================================================================
    # Persistence -- independent of GaussianMixture's, see the design doc.
    # ===================================================================
    def to_dict(self) -> dict:
        return {
            'n_components': self.n_components,
            'n_features': self.n_features,
            'concentration': self.concentration,
            'kappa_init': self.kappa_init,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'reg_covar': self.reg_covar,
            'n_init': self.n_init,
            'init_means': self.init_means,
            'init_weights': self.init_weights,
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
            'weights_': self.weights_,
            'means_': self.means_,
            'kappa_': self.kappa_,
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
    def load(cls, filepath: str, device: Optional[str] = None) -> "VonMisesFisherMixture":
        r"""Load a model previously saved with `save`."""
        if device is None:
            state = torch.load(filepath, weights_only=False)
        else:
            state = torch.load(filepath, map_location=device, weights_only=False)

        model = cls(
            n_components=state['n_components'], n_features=state['n_features'],
            concentration=state['concentration'], kappa_init=state['kappa_init'],
            max_iter=state['max_iter'], tol=state['tol'], reg_covar=state['reg_covar'],
            n_init=state['n_init'], init_means=state['init_means'], init_weights=state['init_weights'],
            random_state=state['random_state'], warm_start=state['warm_start'], algorithm=state['algorithm'],
            batch_size=state['batch_size'], temperature=state['temperature'], convergence=state['convergence'],
            step_size=state['step_size'], sem_burn_in=state['sem_burn_in'], sem_estimate=state['sem_estimate'],
            verbose=state['verbose'], verbose_interval=state['verbose_interval'], device=device,
        )
        model.weights_ = state['weights_']
        model.means_ = state['means_']
        model.kappa_ = state['kappa_']
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
        self.weights_ = state_dict['weights_']
        self.means_ = state_dict['means_']
        self.kappa_ = state_dict['kappa_']
        self.fitted_ = state_dict['fitted_']
        self.converged_ = state_dict['converged_']
        self.n_iter_ = state_dict['n_iter_']
        self.lower_bound_ = state_dict['lower_bound_']
        self.n_components_ = state_dict['n_components_']
