r"""
BaseMixture: shared EM/CEM/SEM/SAEM/online-EM machinery for every tgmm
mixture family (`GaussianMixture`, `StudentTMixture`,
`VonMisesFisherMixture`).

See docs/superpowers/specs/2026-10-14-phase4-base-mixture-studentt-vmf-design.md
for which methods are genuinely shared here vs. where `GaussianMixture`
keeps its own, more specialized implementation (`fit`'s Gibbs/supervised
dispatch, `sample`'s rich feature set, and all of `save`/`load`/
`save_state_dict`/`load_state_dict`, none of which generalize to every
family without either forcing Gaussian-specific features onto families
that don't need them or redesigning `GaussianMixture`'s serialization
format -- both judged not worth the risk for this phase).

`BaseMixture.__init__` is convenience infrastructure for *new* subclasses
(`StudentTMixture`, `VonMisesFisherMixture`): it validates and stores the
arguments the shared methods below actually need, with defaults such that
calling it with no arguments at all is a safe no-op beyond
`nn.Module.__init__`. `GaussianMixture` does not call it with any
arguments -- its own `__init__`, unchanged by this phase, remains the
sole source of truth for its ~90-attribute configuration.
"""
import math
import numbers
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..stats import SufficientStats
from . import missing as _missing

_VALID_ALGORITHMS = ('em', 'cem', 'sem', 'saem')
_VALID_CONVERGENCE = ('relative', 'aitken')
_VALID_SEM_ESTIMATES = ('best', 'mean')


class BaseMixture(nn.Module, ABC):
    r"""
    Shared scaffolding for EM-family mixture models.

    Subclasses must implement the abstract methods below; everything else
    here is usable as-is. See the design doc for the full rationale.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components.
    n_features : int, optional
        Dimensionality of the data. Inferred from `X` at fit time if `None`.
    max_iter, tol, reg_covar, n_init, init_means, init_weights,
    random_state, warm_start, algorithm, batch_size, temperature,
    convergence, step_size, sem_burn_in, sem_estimate, verbose,
    verbose_interval, device : see `GaussianMixture`'s docstring for each
        -- the semantics are identical here; every one defaults to the
        same value `GaussianMixture` uses.
    components_per_class : int or dict, default=1
        For `fit(X, labels=...)` (semi-)supervised fitting (Phase 5):
        how many components each class owns, MclustDA-style -- an `int`
        gives every class that many components; a `dict` (keyed by the
        original label value) sets it per class. The total across all
        classes present in `labels` must equal `n_components`. Ignored
        when `fit` is called without `labels`.
    unlabeled_weight : float, default=1.0
        Semi-supervised fitting only (some but not all rows labeled):
        multiplies unlabeled rows' responsibilities before they're
        accumulated into the M-step's sufficient statistics (Nigam et al.,
        2000). `0` recovers a fit using only the labeled rows.
    label_confidence : float, default=1.0
        Semi-supervised fitting only: with `c < 1`, a labeled row keeps
        mass `c` on its class's component block and distributes `1 - c`
        over the other components according to its own unclamped
        posterior -- a knob for labels that might themselves be noisy.
        `1.0` (default) fully clamps labeled rows to their class block.
    frozen_components : sequence of int, default=()
        Indices of components whose per-component parameters (`means_`, `covariances_`,
        `dof_`, `kappa_`) are held at their initial values for the whole fit; their mixing
        weights still update. See `docs/superpowers/specs/2026-10-15-v05-frozen-sample-weight-design.md`.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        n_features: Optional[int] = None,
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
        step_size: Optional[object] = None,
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
        frozen_components: Sequence[int] = (),
        verbose: bool = False,
        verbose_interval: int = 10,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.n_components = n_components
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.init_means = init_means
        self.init_weights = init_weights
        self.random_state = random_state
        self.warm_start = warm_start

        if algorithm not in _VALID_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {_VALID_ALGORITHMS}, got {algorithm!r}")
        self.algorithm = algorithm

        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size} (must be > 0).")
        self.batch_size = batch_size

        if temperature is not None:
            beta0, n_anneal = temperature
            if not (0.0 < beta0 < 1.0):
                raise ValueError(f"temperature[0] (beta_0) must be in (0, 1), got {beta0}.")
            if n_anneal <= 0:
                raise ValueError(f"temperature[1] (n_anneal) must be > 0, got {n_anneal}.")
        self.temperature = temperature

        if convergence not in _VALID_CONVERGENCE:
            raise ValueError(f"convergence must be one of {_VALID_CONVERGENCE}, got {convergence!r}")
        self.convergence = convergence

        if step_size is not None:
            if isinstance(step_size, str) and step_size != 'auto':
                raise ValueError(f"step_size must be 'auto', a positive number, or callable, got {step_size!r}")
            if isinstance(step_size, (int, float)) and step_size <= 0:
                raise ValueError(f"step_size must be > 0, got {step_size}.")
        self.step_size = step_size

        self.sem_burn_in = sem_burn_in
        if sem_estimate not in _VALID_SEM_ESTIMATES:
            raise ValueError(f"sem_estimate must be one of {_VALID_SEM_ESTIMATES}, got {sem_estimate!r}")
        self.sem_estimate = sem_estimate

        if not (isinstance(components_per_class, int) or isinstance(components_per_class, dict)):
            raise TypeError(f"components_per_class must be an int or dict, got {type(components_per_class)}")
        if isinstance(components_per_class, int) and components_per_class <= 0:
            raise ValueError(f"components_per_class must be > 0, got {components_per_class}.")
        self.components_per_class = components_per_class

        if unlabeled_weight < 0:
            raise ValueError(f"unlabeled_weight must be >= 0, got {unlabeled_weight}.")
        self.unlabeled_weight = unlabeled_weight

        if not (0.0 <= label_confidence <= 1.0):
            raise ValueError(f"label_confidence must be in [0, 1], got {label_confidence}.")
        self.label_confidence = label_confidence

        self.classes_ = None
        self.class_of_component_ = None

        self._init_noise(noise, noise_volume, noise_prior, noise_init)
        self._init_missing(missing)
        self._init_frozen(frozen_components)

        # Optional callable(X, resp_or_None), called once right after
        # _allocate_parameters (with resp=None) and once after every M-step
        # (with the resp actually used) during semi-supervised fitting
        # only. Not user-facing API: `PositiveUnlabeledMixture` (Phase 5.2)
        # sets this internally on its underlying model to implement its
        # 'fixed' (freeze the positive block's parameters) and 'prior'
        # (NIW-blend the positive block toward its labeled-only fit)
        # anchor strategies, without BaseMixture needing to know either
        # exists. `None` (default) is a no-op everywhere else.
        self._post_mstep_hook = None

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.get_default_dtype()

        self.verbose = verbose
        self.verbose_interval = verbose_interval

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.fitted_ = False
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")
        self.best_random_state_ = None
        self.n_components_ = None
        self.log_likelihood_history_ = None
        self.annealing_schedule_ = None
        self.n_empty_components_ = 0
        self._online_running_stats_ = None
        self._online_step_count_ = 0

    # ===================================================================
    # Frozen components and sample weights (v0.5 item 1.4). See
    # docs/superpowers/specs/2026-10-15-v05-frozen-sample-weight-design.md.
    # ===================================================================
    _fit_sw = None        # row weights (n,) during a weighted fit, else None
    _fit_fw = None        # per-feature weights (n, d) during a per-feature-weighted fit, else None
    _fit_n_eff = None     # sum of the row weights during a weighted fit
    _fw_cur = None        # the chunk of `_fit_fw` the current E-step chunk uses

    @property
    def frozen_components(self) -> Tuple[int, ...]:
        r"""Sorted indices of the components held fixed (see the constructor argument)."""
        return self.__dict__.get('_frozen_components', ())

    @frozen_components.setter
    def frozen_components(self, value):
        self._init_frozen(value)

    def _init_frozen(self, frozen_components):
        r"""Validate (indices distinct integers in ``[0, K)``) and store `frozen_components`; then let the
        family reject the configurations it cannot freeze (`_check_frozen_supported`)."""
        if frozen_components is None:
            frozen_components = ()
        try:
            items = list(frozen_components)
        except TypeError:
            raise TypeError(f"frozen_components must be a sequence of component indices, got {frozen_components!r}.")
        for item in items:
            is_int = isinstance(item, numbers.Integral) and not isinstance(item, bool)
            is_int_tensor = torch.is_tensor(item) and item.dim() == 0 and not torch.is_floating_point(item)
            if not (is_int or is_int_tensor):
                raise ValueError(f"frozen_components must contain integers, got {item!r}.")
        idx = tuple(sorted(int(i) for i in items))
        if len(set(idx)) != len(idx):
            raise ValueError(f"frozen_components must not contain duplicates, got {frozen_components!r}.")
        if idx:
            if self.n_components is None:
                raise ValueError(
                    "frozen_components requires a fixed n_components (it is not available in Gibbs mode, "
                    "n_components=None)."
                )
            if idx[0] < 0 or idx[-1] >= self.n_components:
                raise ValueError(
                    f"frozen_components indices must lie in [0, {self.n_components}), got {frozen_components!r}."
                )
        self.__dict__['_frozen_components'] = idx
        if idx and self.__dict__.get('_frozen_ready', False):
            self._check_frozen_supported()

    def _check_frozen_supported(self):
        r"""Family hook: raise `ValueError` if the current configuration cannot hold a component fixed
        (a parameter shared across components). Called when `frozen_components` is set and at fit time."""

    def _frozen_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.n_components, dtype=torch.bool, device=self.device)
        if self.frozen_components:
            mask[list(self.frozen_components)] = True
        return mask

    def _frozen_param_names(self):
        K = self.n_components
        names = []
        for name in ('means_', 'covariances_', 'dof_', 'kappa_'):
            value = getattr(self, name, None)
            if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == K:
                names.append(name)
        return names

    def _m_step_frozen(self, stats: SufficientStats):
        r"""`_m_step_from_stats`, then restore the frozen components' per-component parameters (the mixing
        weights, and the noise weight, are left as updated)."""
        frozen = self.frozen_components
        if not frozen:
            self._m_step_from_stats(stats)
            return
        idx = torch.tensor(frozen, dtype=torch.long, device=self.device)
        saved = {name: getattr(self, name)[idx].clone() for name in self._frozen_param_names()}
        self._m_step_from_stats(stats)
        for name, value in saved.items():
            getattr(self, name)[idx] = value

    # ---- sample weights -------------------------------------------------
    def _feature_weights_supported(self) -> bool:
        r"""Family hook: whether `sample_weight` of shape `(n, d)` is supported (per-feature factorisation)."""
        return False

    def _has_priors(self) -> bool:
        return False

    def _prepare_sample_weight(self, sample_weight, X: torch.Tensor, labels):
        r"""Validate `sample_weight` and set the per-fit weight context. Returns the row-keep mask
        (rows with weight > 0, for initialisation) or `None`."""
        n, d = X.shape
        w = torch.as_tensor(sample_weight)
        if not torch.is_floating_point(w):
            w = w.to(torch.get_default_dtype())
        w = w.to(device=self.device, dtype=X.dtype)
        if w.dim() not in (1, 2) or w.shape[0] != n or (w.dim() == 2 and w.shape[1] != d):
            raise ValueError(
                f"sample_weight must have shape (n_samples,) = ({n},) or (n_samples, n_features) = ({n}, {d}), "
                f"got {tuple(w.shape)}."
            )
        if not bool(torch.isfinite(w).all()):
            raise ValueError("sample_weight must be finite.")
        if bool((w < 0).any()):
            raise ValueError("sample_weight must be non-negative.")
        if not bool(w.sum() > 0):
            raise ValueError("sample_weight must have a positive sum.")
        if labels is not None:
            raise NotImplementedError("sample_weight is not supported together with labels=... (semi-supervised fitting).")
        if self.algorithm in ('em', 'cem') and self.step_size is not None:
            raise NotImplementedError("sample_weight is not supported with online EM (step_size set).")
        if w.dim() == 2:
            if not self._feature_weights_supported():
                raise ValueError(
                    f"per-feature sample_weight (shape (n_samples, n_features)) is not supported by this "
                    f"{type(self).__name__} configuration: the statistics must factorise per feature "
                    "(GaussianMixture / StudentTMixture with a diagonal covariance, GaussianMixture 'spherical'). "
                    "Use sample_weight of shape (n_samples,) instead."
                )
            if self.noise:
                raise NotImplementedError("per-feature sample_weight cannot be combined with noise=True.")
            if self.missing == 'em':
                raise NotImplementedError("per-feature sample_weight cannot be combined with missing='em'.")
            if self._has_priors():
                raise NotImplementedError(
                    "per-feature sample_weight supports maximum likelihood only (no weight / mean / covariance priors)."
                )
        if bool((w == 1).all()):
            return None  # all-ones is exactly the unweighted fit
        if w.dim() == 1:
            self._fit_sw = w
            self._fit_n_eff = float(w.sum().item())
            return w > 0
        self._fit_fw = w
        return None

    def _clear_sample_weight(self):
        self._fit_sw = None
        self._fit_fw = None
        self._fit_n_eff = None
        self._fw_cur = None

    def _mean_ll(self, log_prob_norm: torch.Tensor) -> float:
        r"""Mean log-likelihood of a fit's E-step: weighted by the row weights when present."""
        sw = self._fit_sw
        if sw is None or sw.shape[0] != log_prob_norm.shape[0]:
            return log_prob_norm.mean().item()
        return ((sw * log_prob_norm).sum() / sw.sum()).item()

    def _fit_stats(self, X: torch.Tensor, resp_k: torch.Tensor) -> SufficientStats:
        r"""Sufficient statistics of the current fit: row weights scale the responsibilities and the effective
        sample count; per-feature weights are passed to the family's `_compute_stats`."""
        sw = self._fit_sw
        if sw is not None:
            resp_k = resp_k * sw.unsqueeze(1)
        if self._fit_fw is None:
            stats = self._compute_stats(X, resp_k, batch_size=self.batch_size)
        else:
            stats = self._compute_stats(X, resp_k, batch_size=self.batch_size, feature_weights=self._fit_fw)
        if sw is not None:
            stats.n_samples = self._fit_n_eff
        return stats

    # ===================================================================
    # Missing values (v0.5 item 1.2). See
    # docs/superpowers/specs/2026-10-15-v05-missing-values-design.md.
    # With missing='error' (default) NaN input raises; with missing='em' (families
    # that set `_supports_missing`) NaN marks missing-at-random entries. On data
    # without NaN both settings take exactly the same code path.
    # ===================================================================
    _supports_missing = False

    def _init_missing(self, missing):
        r"""Validate and store `missing`; reject combinations the EM-with-missing-values path cannot serve."""
        if missing not in ('error', 'em'):
            raise ValueError(f"missing must be 'error' or 'em', got {missing!r}.")
        self.missing = missing
        if missing != 'em':
            return
        if not self._supports_missing:
            raise ValueError(
                f"missing='em' is not supported by {type(self).__name__} (it is available for GaussianMixture "
                "and StudentTMixture)."
            )
        if self.noise:
            raise NotImplementedError(
                "missing='em' cannot be combined with noise=True: the noise density needs the volume of each "
                "row's observed subspace, which is not supported."
            )
        if self.algorithm in ('em', 'cem') and self.step_size is not None:
            raise NotImplementedError(
                "missing='em' does not support online EM (step_size set); use batch_size alone or algorithm='saem'."
            )

    def _missing_active(self, X: torch.Tensor) -> bool:
        r"""True if `missing='em'` and `X` actually contains NaN (otherwise the ordinary path runs)."""
        return self.missing == 'em' and _missing.has_missing(X)

    def _check_nan_input(self, X: torch.Tensor):
        r"""`missing='error'`: raise on NaN input (called at the top of every E-step and of `fit`)."""
        if self.missing == 'em' or not torch.is_floating_point(X):
            return
        if bool(torch.isnan(X).any().item()):
            hint = (
                "pass missing='em' to treat NaN as missing-at-random values"
                if self._supports_missing else
                "this model does not support missing values (missing='em' is available for "
                "GaussianMixture and StudentTMixture); impute or drop the NaN rows first"
            )
            raise ValueError(f"X contains NaN; {hint}.")

    def _prepare_missing_fit(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Fit-time checks for incomplete data; returns the column-mean-imputed copy used for
        initialisation (k-means etc. never see NaN) and for SEM re-seeding.
        """
        _missing.check_incomplete_rows_and_columns(X)
        n_patterns = _missing.count_patterns(X)
        if n_patterns > 1000:
            warnings.warn(
                f"X has {n_patterns} distinct missingness patterns; the E-step loops over patterns in Python, "
                "so fitting will be slow.",
                UserWarning,
            )
        return _missing.fill_columns_with_means(X)

    def _missing_cov_repr(self):
        r"""Family hook: `(kind, cov)` with `kind in {'dense', 'diag'}` (see `missing.gaussian_blocks`)."""
        raise NotImplementedError

    def _completed_rows(self, X: torch.Tensor) -> torch.Tensor:
        r"""Per-component completed data `x_tilde`, shape `(n, K, d)`, with the *current* parameters."""
        kind, cov = self._missing_cov_repr()
        blocks = _missing.gaussian_blocks(X, self.means_, cov, kind, need_conditional=True)
        return _missing.complete_rows(X, blocks, self.means_.shape[0])

    def impute(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Replace every NaN of `X` by its posterior-weighted conditional mean
        :math:`\sum_k \tau_{ik}\hat x_{ik}`, where :math:`\hat x_{ik}` is the conditional mean of the missing
        block given the observed entries under component `k` (observed entries are returned unchanged).

        Requires `missing='em'`.
        """
        if self.missing != 'em':
            raise RuntimeError("impute requires a model constructed with missing='em'.")
        X = X.to(self.device)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if not _missing.has_missing(X):
            return X.clone()
        resp, _ = self._e_step(X)
        completed = (resp.unsqueeze(-1) * self._completed_rows(X)).sum(dim=1)
        return torch.where(torch.isnan(X), completed, X)

    # ===================================================================
    # Uniform noise component (v0.5 item 1.1). See
    # docs/superpowers/specs/2026-10-15-v05-noise-component-design.md.
    # Inert unless `noise=True`: every hook below is a no-op / pass-through
    # when `self.noise` is False, so default fits are bit-identical.
    # ===================================================================
    # Families whose noise density is a uniform over the data's bounding box
    # (rather than e.g. over the sphere) need `noise_bounds_` at fit time.
    _noise_uses_bbox = True

    def _init_noise(self, noise, noise_volume, noise_prior, noise_init):
        r"""Validate and store the noise arguments (see the module docs / spec)."""
        if not (0.0 < float(noise_prior) < 1.0):
            raise ValueError(f"noise_prior must be in (0, 1), got {noise_prior}.")
        if isinstance(noise_volume, str):
            if noise_volume != 'bbox':
                raise ValueError(f"noise_volume must be 'bbox' or a positive float, got {noise_volume!r}.")
        elif isinstance(noise_volume, bool) or not float(noise_volume) > 0.0:
            raise ValueError(f"noise_volume must be 'bbox' or a positive float, got {noise_volume!r}.")
        self.noise = bool(noise)
        self.noise_volume = noise_volume
        self.noise_prior = float(noise_prior)
        self.noise_init = noise_init
        self.noise_weight_ = None
        self.noise_bounds_ = None

    def _noise_log_volume(self) -> float:
        r"""`log V` of the uniform noise region (bounding box or the user-supplied volume)."""
        if not isinstance(self.noise_volume, str):
            return math.log(float(self.noise_volume))
        if self.noise_bounds_ is None:
            raise RuntimeError("noise_bounds_ is not set; fit the model first.")
        extent = (self.noise_bounds_[1] - self.noise_bounds_[0]).clamp(min=1e-12)
        return torch.log(extent).sum().item()

    def _noise_log_density(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Log-density of the noise component at each row of `X`, shape `(n_samples,)`.

        Default (`GaussianMixture`, every GPCM code, `StudentTMixture`): the
        constant `-log V`, with `V` the volume of the bounding box of the fit
        data (or `noise_volume`). Families override it: `VonMisesFisherMixture`
        uses the uniform density on the sphere; discrete/count families must
        raise ``ValueError('noise component requires bounded continuous support')``.
        """
        return torch.full((X.shape[0],), -self._noise_log_volume(), dtype=X.dtype, device=X.device)

    def _sample_noise(self, n: int) -> torch.Tensor:
        r"""Draw `n` points from the noise component: uniform in the stored bounding box."""
        if self.noise_bounds_ is None:
            raise RuntimeError("noise_bounds_ is not set; fit the model first.")
        lo, hi = self.noise_bounds_[0], self.noise_bounds_[1]
        u = torch.rand(n, self.n_features, dtype=lo.dtype, device=lo.device)
        return lo + (hi - lo) * u

    def _prepare_noise(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        r"""
        Fit-time setup for `noise=True`: store the bounding box and resolve
        `noise_init` into a boolean mask over the rows of `X` (or `None`).
        """
        if self._noise_uses_bbox:
            self.noise_bounds_ = torch.stack([X.min(dim=0).values, X.max(dim=0).values]).detach()
        init = self.noise_init
        if init is None:
            return None
        if not torch.is_tensor(init):
            init = torch.as_tensor(init)
        init = init.to(self.device)
        if init.dim() != 1 or init.shape[0] != X.shape[0]:
            raise ValueError(
                f"noise_init must be a 1D array of length n_samples={X.shape[0]}, got shape {tuple(init.shape)}."
            )
        mask = init if init.dtype == torch.bool else (init == -1)
        if bool(mask.all()):
            raise ValueError("noise_init flags every row as noise; at least some rows must be clustered.")
        return mask

    def _init_noise_weights(self, noise_mask: Optional[torch.Tensor]):
        r"""After `_allocate_parameters`: set `pi_0` and rescale `weights_` to sum to `1 - pi_0`."""
        if noise_mask is not None:
            pi0 = min(max(noise_mask.double().mean().item(), 1e-3), 0.9)
        else:
            pi0 = self.noise_prior
        self.noise_weight_ = torch.tensor(pi0, dtype=self.weights_.dtype, device=self.weights_.device)
        self.weights_ = self.weights_ * (1.0 - self.noise_weight_)

    def _split_noise(self, resp: torch.Tensor):
        r"""`(resp[:, :K], resp[:, K])` when noise is on, else `(resp, None)`."""
        if not self.noise:
            return resp, None
        return resp[:, :-1], resp[:, -1]

    def _check_noise_supported(self, labels=None):
        if not self.noise:
            return
        if labels is not None:
            raise NotImplementedError(
                "noise=True does not support fit(X, labels=...): a point cannot be labelled as known "
                "noise in v0.5 (-1 already means 'unlabelled')."
            )
        if self.algorithm in ('em', 'cem') and self.step_size is not None:
            raise NotImplementedError(
                "noise=True does not support online EM (step_size set); use algorithm='saem' or batch_size alone."
            )

    # ===================================================================
    # Abstract methods -- every subclass implements these
    # ===================================================================
    @abstractmethod
    def _component_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Per-component log-density, shape `(n_samples, n_components)`,
        *before* adding `log(weights_)` or tempering by `beta` -- those
        are applied uniformly by `_e_step_chunk`.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_stats(self, X: torch.Tensor, resp: torch.Tensor, batch_size: Optional[int] = None) -> SufficientStats:
        raise NotImplementedError

    @abstractmethod
    def _m_step_from_stats(self, stats: SufficientStats):
        raise NotImplementedError

    @abstractmethod
    def _allocate_parameters(self, X: Optional[torch.Tensor] = None, set_random_state: bool = True):
        r"""Initialize `weights_`/`means_`/`covariances_` (and any other
        family-specific parameters) before a fit. Named to match the
        pre-Phase-4 `GaussianMixture` method rather than introducing a
        second, aliased name (the plan's "`_init_params`")."""
        raise NotImplementedError

    @abstractmethod
    def _n_parameters(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _sample_component(self, k: int, n: int) -> torch.Tensor:
        r"""Draw `n` samples from component `k` alone, shape `(n, n_features)`."""
        raise NotImplementedError

    # ===================================================================
    # Snapshotting fitted parameters, for the n_init loop and SEM's
    # post-burn-in best/mean selection. Override to add family-specific
    # parameters beyond weights_/means_/covariances_ (e.g. StudentTMixture's
    # dof_) -- both _get_fit_snapshot and SEM automatically pick up any
    # extra keys an override adds, no further changes needed anywhere else.
    # ===================================================================
    def _get_param_snapshot(self) -> dict:
        snapshot = {
            'weights_': self.weights_.clone(),
            'means_': self.means_.clone(),
            'covariances_': self.covariances_.clone(),
        }
        if self.noise:
            snapshot['noise_weight_'] = self.noise_weight_.clone()
        return snapshot

    def _apply_snapshot(self, snapshot: dict):
        for key, value in snapshot.items():
            setattr(self, key, value)

    def _get_fit_snapshot(self) -> dict:
        snapshot = self._get_param_snapshot()
        snapshot.update({
            'converged_': self.converged_,
            'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_,
            'log_likelihood_history_': self.log_likelihood_history_,
            'annealing_schedule_': self.annealing_schedule_,
            'n_empty_components_': self.n_empty_components_,
            '_online_running_stats_': self._online_running_stats_,
            '_online_step_count_': self._online_step_count_,
        })
        return snapshot

    # ===================================================================
    # M-step
    # ===================================================================
    def _m_step(self, X: torch.Tensor, resp: torch.Tensor):
        r"""Thin wrapper over `_compute_stats`/`_m_step_from_stats`.

        With `noise=True`, `resp` has an extra last (noise) column: the K
        components are updated from `resp[:, :K]` and `pi_0` is the mean
        noise responsibility."""
        resp_k, resp_noise = self._split_noise(resp)
        self._m_step_frozen(self._fit_stats(X, resp_k))
        if resp_noise is not None:
            if self._fit_sw is not None:
                noise_mass = (resp_noise.reshape(-1) * self._fit_sw).sum()
                self.noise_weight_ = (noise_mass / self._fit_n_eff).to(self.weights_.dtype).clamp(1e-12, 1.0 - 1e-12)
            else:
                self.noise_weight_ = (resp_noise.sum() / resp.shape[0]).to(self.weights_.dtype).clamp(1e-12, 1.0 - 1e-12)

    # ===================================================================
    # E-step, C-step, S-step
    # ===================================================================
    def _e_step(self, X: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        E-step: compute responsibilities, chunked over rows when
        `self.batch_size` is set. Chunking is exact (per-row softmax).

        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).
        beta : float, optional
            Deterministic-annealing temperature (Ueda & Nakano, 1998).

        Returns
        -------
        resp : torch.Tensor, shape (n_samples, n_components)
        log_prob_norm : torch.Tensor, shape (n_samples,)
        """
        self._check_nan_input(X)
        fw = self._fit_fw if (self._fit_fw is not None and self._fit_fw.shape[0] == X.shape[0]) else None
        if self.batch_size is None or self.batch_size >= X.shape[0]:
            if fw is None:
                return self._e_step_chunk(X, beta=beta)
            self._fw_cur = fw
            try:
                return self._e_step_chunk(X, beta=beta)
            finally:
                self._fw_cur = None

        resp_chunks = []
        log_prob_norm_chunks = []
        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, X.shape[0])
            if fw is not None:
                self._fw_cur = fw[start:end]
            try:
                resp_chunk, log_prob_norm_chunk = self._e_step_chunk(X[start:end], beta=beta)
            finally:
                self._fw_cur = None
            resp_chunks.append(resp_chunk)
            log_prob_norm_chunks.append(log_prob_norm_chunk)
        return torch.cat(resp_chunks, dim=0), torch.cat(log_prob_norm_chunks, dim=0)

    def _e_step_chunk(self, X: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        log_prob = self._component_log_prob(X)
        log_weights = torch.log(self.weights_ + 1e-20)
        log_prob = log_prob + log_weights.unsqueeze(0)
        if self.noise:
            noise_col = torch.log(self.noise_weight_ + 1e-20) + self._noise_log_density(X)
            log_prob = torch.cat([log_prob, noise_col.unsqueeze(1).to(log_prob.dtype)], dim=1)
        if beta != 1.0:
            log_prob = beta * log_prob

        log_prob_norm = torch.logsumexp(log_prob, dim=1)
        log_resp = log_prob - log_prob_norm.unsqueeze(1)
        resp = torch.exp(log_resp)
        return resp, log_prob_norm

    def _c_step(self, resp: torch.Tensor) -> torch.Tensor:
        r"""C-step: hard-assign each row to its highest-responsibility component."""
        max_resp_indices = torch.argmax(resp, dim=1)
        n_samples = resp.size(0)
        hard_resp = torch.zeros_like(resp)
        hard_resp[torch.arange(n_samples, device=resp.device), max_resp_indices] = 1.0
        return hard_resp

    def _s_step(self, resp: torch.Tensor) -> torch.Tensor:
        r"""S-step: sample one hard assignment per row from its responsibility distribution."""
        z = torch.multinomial(resp, 1).squeeze(1)
        return F.one_hot(z, resp.shape[1]).to(resp.dtype)

    def _beta_for_iteration(self, n_iter: int) -> float:
        r"""Deterministic-annealing temperature for E-step iteration `n_iter`."""
        if self.temperature is None:
            return 1.0
        beta0, n_anneal = self.temperature
        if n_iter >= n_anneal:
            return 1.0
        return beta0 ** (1.0 - n_iter / n_anneal)

    def _check_convergence(self, current: float, previous: float, previous2: Optional[float], tol: float) -> bool:
        r"""Whether the EM/CEM/SAEM loop should stop, per `self.convergence`."""
        if self.convergence == 'aitken' and previous2 is not None and previous2 != -float('inf'):
            denom = previous - previous2
            if abs(denom) > 1e-300:
                a = (current - previous) / denom
                if a < 1.0:
                    l_inf = previous + (current - previous) / (1.0 - a)
                    return (l_inf - current) < tol
        return abs(current - previous) / (abs(previous) + 1e-20) < tol

    # ===================================================================
    # Online EM / SAEM statistics blending
    # ===================================================================
    @staticmethod
    def _blend_stats(
        running: SufficientStats, new: SufficientStats, gamma: float, n_samples: int,
        new_scale: float = 1.0,
    ) -> SufficientStats:
        r"""`(1 - gamma) * running + gamma * new_scale * new`, with `n_samples`
        pinned to the given nominal size afterward."""
        blended = running.scale(1.0 - gamma) + new.scale(gamma * new_scale)
        blended.n_samples = n_samples
        return blended

    def _resolve_step_size_fn(self) -> Callable[[int], float]:
        r"""`self.step_size` as a callable `t -> gamma_t`. `None` resolves to `'auto'`."""
        step_size = self.step_size if self.step_size is not None else 'auto'
        if callable(step_size):
            return step_size
        if step_size == 'auto':
            return lambda t: (t + 2) ** -0.7
        value = float(step_size)
        return lambda t: value

    # ===================================================================
    # Fitting loops
    # ===================================================================
    def _fit_single_run(self, X: torch.Tensor, max_iter: int, tol: float, run_idx: int = 0):
        r"""One complete optimization run, dispatching on `self.algorithm`/`self.step_size`."""
        X = X.to(self.device)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features}.")

        if self.algorithm == 'sem':
            self._fit_single_run_sem(X, max_iter, run_idx)
            return
        if self.algorithm == 'saem':
            self._fit_single_run_saem(X, max_iter, tol, run_idx)
            return
        if self.step_size is not None:
            self._fit_single_run_online(X, max_iter, run_idx)
            return

        prev_lower_bound = -float("inf")
        prev_prev_lower_bound = None

        beta = self._beta_for_iteration(0)
        betas_used = [beta]
        resp, log_prob_norm = self._e_step(X, beta=beta)
        self.lower_bound_ = self._mean_ll(log_prob_norm)
        ll_history = [self.lower_bound_]

        for n_iter in range(max_iter):
            if self.algorithm == 'cem':
                resp = self._c_step(resp)

            self._m_step(X, resp)

            if beta >= 1.0 and self._check_convergence(self.lower_bound_, prev_lower_bound, prev_prev_lower_bound, tol):
                self.converged_ = True
                if self.verbose:
                    print(f"[Run {run_idx+1}] Iteration {n_iter}: "
                          f"log-likelihood={self.lower_bound_:.5f}, Converged!")
                break

            prev_prev_lower_bound = prev_lower_bound
            prev_lower_bound = self.lower_bound_
            beta = self._beta_for_iteration(n_iter + 1)
            betas_used.append(beta)
            resp, log_prob_norm = self._e_step(X, beta=beta)
            self.lower_bound_ = self._mean_ll(log_prob_norm)
            ll_history.append(self.lower_bound_)

            if self.verbose and (n_iter % self.verbose_interval == 0):
                print(f"[Run {run_idx+1}] Iteration {n_iter}: "
                      f"log-likelihood={self.lower_bound_:.5f}")

        if self.converged_:
            resp, log_prob_norm = self._e_step(X, beta=beta)
            if self.algorithm == 'cem':
                resp = self._c_step(resp)
            self.lower_bound_ = self._mean_ll(log_prob_norm)
        else:
            import warnings
            warnings.warn(
                f"Run {run_idx+1}: EM did not converge after {max_iter} iterations.",
                UserWarning
            )

        self.n_iter_ = n_iter
        self.log_likelihood_history_ = torch.tensor(ll_history)
        if self.temperature is not None:
            self.annealing_schedule_ = betas_used

    def _fit_single_run_sem(self, X: torch.Tensor, max_iter: int, run_idx: int = 0):
        r"""Stochastic EM (Celeux & Diebolt, 1985): E-step -> S-step -> M-step,
        for exactly `max_iter` sweeps (no pointwise convergence)."""
        import warnings
        sem_burn_in = self.sem_burn_in if self.sem_burn_in is not None else max(1, max_iter // 5)

        ll_history = []
        betas_used = []
        post_burn_in = []  # list of (snapshot_dict, ll)
        n_empty_total = 0

        beta = self._beta_for_iteration(0)
        betas_used.append(beta)
        resp, log_prob_norm = self._e_step(X, beta=beta)
        ll = self._mean_ll(log_prob_norm)
        ll_history.append(ll)

        for n_iter in range(max_iter):
            z_onehot = self._s_step(resp)

            nk = z_onehot.sum(dim=0)
            if self.noise:
                nk = nk[:-1]  # the noise column is never "empty" in the re-seeding sense
            empty_mask = nk < 1
            if self.frozen_components:
                empty_mask = empty_mask & ~self._frozen_mask()
            if torch.any(empty_mask):
                X_reseed = _missing.fill_columns_with_means(X) if self._missing_active(X) else X
                n_empty_this_sweep = self._reseed_empty_components(X_reseed, log_prob_norm, empty_mask)
                n_empty_total += n_empty_this_sweep
                warnings.warn(
                    f"[Run {run_idx + 1}] Sweep {n_iter}: {n_empty_this_sweep} component(s) "
                    "emptied during SEM and were re-seeded; n_components may be too large "
                    "for this data.",
                    UserWarning
                )

            self._m_step(X, z_onehot)

            beta = self._beta_for_iteration(n_iter + 1)
            betas_used.append(beta)
            resp, log_prob_norm = self._e_step(X, beta=beta)
            ll = self._mean_ll(log_prob_norm)
            ll_history.append(ll)

            if n_iter >= sem_burn_in:
                post_burn_in.append((self._get_param_snapshot(), ll))

            if self.verbose and (n_iter % self.verbose_interval == 0):
                print(f"[Run {run_idx + 1}] Sweep {n_iter}: log-likelihood={ll:.5f}")

        if not post_burn_in:
            post_burn_in = [(self._get_param_snapshot(), ll)]

        if self.sem_estimate == 'best':
            best_snapshot, _ = max(post_burn_in, key=lambda p: p[1])
            self._apply_snapshot(best_snapshot)
        else:  # 'mean'
            keys = post_burn_in[0][0].keys()
            mean_snapshot = {
                key: torch.stack([snap[key] for snap, _ in post_burn_in]).mean(dim=0)
                for key in keys
            }
            self._apply_snapshot(mean_snapshot)

        _, log_prob_norm = self._e_step(X)
        self.lower_bound_ = self._mean_ll(log_prob_norm)
        self.converged_ = True
        self.n_iter_ = max_iter
        self.log_likelihood_history_ = torch.tensor(ll_history)
        self.n_empty_components_ = n_empty_total
        if self.temperature is not None:
            self.annealing_schedule_ = betas_used

    def _reseed_empty_components(self, X: torch.Tensor, log_prob_norm: torch.Tensor, empty_mask: torch.Tensor) -> int:
        r"""
        SEM empty-component handling, generic default: re-seed every
        flagged component's mean to the data point with the lowest
        log-likelihood among points not already claimed by another
        re-seed this sweep. Subclasses with more than one parameter tensor
        (e.g. a covariance to also reset) should override this.
        """
        empty_indices = torch.nonzero(empty_mask, as_tuple=True)[0]
        n_empty = empty_indices.shape[0]
        worst_order = torch.argsort(log_prob_norm)
        worst_points = worst_order[:n_empty]
        self.means_[empty_indices] = X[worst_points]
        return n_empty

    def _fit_single_run_online(self, X: torch.Tensor, max_iter: int, run_idx: int = 0):
        r"""Mini-batch / online EM (Cappe & Moulines, 2009): `algorithm='em'` with `step_size` set."""
        if self.batch_size is None:
            raise ValueError(
                "step_size requires batch_size to also be set -- there is no "
                "mini-batch size to use for online EM otherwise."
            )
        n_samples = X.shape[0]
        gamma_fn = self._resolve_step_size_fn()

        running_stats = None
        t = 0
        ll_history = []

        for epoch in range(max_iter):
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                X_batch = X[batch_idx]
                resp_batch, _ = self._e_step(X_batch)
                batch_stats = self._compute_stats(X_batch, resp_batch)
                gamma = gamma_fn(t)

                if running_stats is None:
                    running_stats = batch_stats.scale(n_samples / batch_stats.n_samples)
                    running_stats.n_samples = n_samples
                else:
                    running_stats = self._blend_stats(
                        running_stats, batch_stats, gamma, n_samples,
                        new_scale=n_samples / batch_stats.n_samples,
                    )

                self._m_step_frozen(running_stats)
                t += 1

            _, log_prob_norm = self._e_step(X)
            ll = self._mean_ll(log_prob_norm)
            ll_history.append(ll)
            self.lower_bound_ = ll

            if self.verbose and (epoch % self.verbose_interval == 0):
                print(f"[Run {run_idx + 1}] Epoch {epoch}: log-likelihood={ll:.5f}")

        self.converged_ = True
        self.n_iter_ = max_iter
        self.log_likelihood_history_ = torch.tensor(ll_history)
        self._online_running_stats_ = running_stats
        self._online_step_count_ = t

    def _fit_single_run_saem(self, X: torch.Tensor, max_iter: int, tol: float, run_idx: int = 0):
        r"""SAEM (Delyon, Lavielle & Moulines, 1999): SEM blended into a running accumulator."""
        import warnings
        n_samples = X.shape[0] if self._fit_n_eff is None else self._fit_n_eff
        gamma_fn = self._resolve_step_size_fn()

        prev_lower_bound = -float("inf")
        prev_prev_lower_bound = None
        running_stats = None
        running_noise = None

        resp, log_prob_norm = self._e_step(X)
        self.lower_bound_ = self._mean_ll(log_prob_norm)
        ll_history = [self.lower_bound_]

        for n_iter in range(max_iter):
            z_onehot = self._s_step(resp)
            z_k, z_noise = self._split_noise(z_onehot)
            batch_stats = self._fit_stats(X, z_k)
            gamma = gamma_fn(n_iter)
            noise_frac = None
            if z_noise is not None:
                noise_mass = z_noise.sum() if self._fit_sw is None else (z_noise.reshape(-1) * self._fit_sw).sum()
                noise_frac = noise_mass / n_samples

            if running_stats is None:
                running_stats = batch_stats
                running_noise = noise_frac
            else:
                running_stats = self._blend_stats(running_stats, batch_stats, gamma, n_samples, new_scale=1.0)
                if z_noise is not None:
                    running_noise = (1.0 - gamma) * running_noise + gamma * noise_frac

            self._m_step_frozen(running_stats)
            if z_noise is not None:
                self.noise_weight_ = running_noise.to(self.weights_.dtype).clamp(1e-12, 1.0 - 1e-12)

            if self._check_convergence(self.lower_bound_, prev_lower_bound, prev_prev_lower_bound, tol):
                self.converged_ = True
                if self.verbose:
                    print(f"[Run {run_idx + 1}] Iteration {n_iter}: "
                          f"log-likelihood={self.lower_bound_:.5f}, Converged!")
                break

            prev_prev_lower_bound = prev_lower_bound
            prev_lower_bound = self.lower_bound_
            resp, log_prob_norm = self._e_step(X)
            self.lower_bound_ = self._mean_ll(log_prob_norm)
            ll_history.append(self.lower_bound_)

            if self.verbose and (n_iter % self.verbose_interval == 0):
                print(f"[Run {run_idx + 1}] Iteration {n_iter}: log-likelihood={self.lower_bound_:.5f}")

        self.log_likelihood_history_ = torch.tensor(ll_history)

        if self.converged_:
            resp, log_prob_norm = self._e_step(X)
            self.lower_bound_ = self._mean_ll(log_prob_norm)
        else:
            warnings.warn(
                f"Run {run_idx + 1}: SAEM did not converge after {max_iter} iterations.",
                UserWarning
            )

        self.n_iter_ = n_iter

    def partial_fit(self, X_batch: torch.Tensor):
        r"""One streaming online-EM update (Cappe & Moulines, 2009)."""
        if not self.fitted_ or self.weights_ is None:
            raise RuntimeError(
                "partial_fit requires the model to already be initialized -- call "
                "fit() on a first batch, or set weights_/means_/covariances_ "
                "explicitly, before the first partial_fit call."
            )
        if self.noise:
            raise NotImplementedError(
                "partial_fit is not supported with noise=True (online EM does not track the noise weight)."
            )
        if self.missing == 'em':
            raise NotImplementedError("partial_fit is not supported with missing='em'.")
        X_batch = X_batch.to(self.device)
        if X_batch.dim() == 1:
            X_batch = X_batch.unsqueeze(1)

        gamma_fn = self._resolve_step_size_fn()
        resp, _ = self._e_step(X_batch)
        batch_stats = self._compute_stats(X_batch, resp)
        gamma = gamma_fn(self._online_step_count_)

        if self._online_running_stats_ is None:
            self._online_running_stats_ = batch_stats
        else:
            self._online_running_stats_ = self._blend_stats(
                self._online_running_stats_, batch_stats, gamma,
                batch_stats.n_samples, new_scale=1.0,
            )

        self._m_step_frozen(self._online_running_stats_)
        self._online_step_count_ += 1

        _, log_prob_norm = self._e_step(X_batch)
        self.lower_bound_ = self._mean_ll(log_prob_norm)
        return self

    # ===================================================================
    # fit(): generic n_init loop. GaussianMixture overrides this to add
    # Gibbs-mode and supervised-fit dispatch, calling super().fit(...) for
    # the plain EM/CEM/SEM/SAEM/online path -- see the design doc.
    # ===================================================================
    def fit(
        self,
        X: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        random_state: Optional[int] = None,
        warm_start: Optional[bool] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        r"""
        Fit the mixture by EM / CEM / SEM / SAEM (see `GaussianMixture.fit` for the shared arguments).

        Parameters
        ----------
        sample_weight : torch.Tensor, optional
            Row weights of shape `(n_samples,)` (row `i` counts `w_i` times: integer weights equal
            duplicating rows, weight 0 equals dropping the row, an all-ones vector is bit-identical to
            `None`), or per-feature weights of shape `(n_samples, n_features)` for models whose statistics
            factorise per feature (`GaussianMixture` with `'diag'`/`'spherical'` covariance and
            `StudentTMixture` with `'diag'`; maximum likelihood only): weight 0 on entry `(i, j)` hides
            feature `j` of row `i`. Not supported with `labels=...`, Gibbs mode or online EM. See
            `docs/superpowers/specs/2026-10-15-v05-frozen-sample-weight-design.md`. (default: None)
        """
        self._clear_sample_weight()
        if sample_weight is None:
            return self._fit_main(X, labels, max_iter, tol, random_state, warm_start, None)
        Xw = X.to(self.device)
        if Xw.dim() == 1:
            Xw = Xw.unsqueeze(1)
        keep = self._prepare_sample_weight(sample_weight, Xw, labels)
        try:
            return self._fit_main(X, labels, max_iter, tol, random_state, warm_start, keep)
        finally:
            self._clear_sample_weight()

    def _fit_main(self, X, labels, max_iter, tol, random_state, warm_start, keep):
        import warnings

        if self.frozen_components:
            self._check_frozen_supported()
        if X.size(0) < self.n_components:
            raise ValueError(f"n_components={self.n_components} must be <= n_samples={X.size(0)}.")
        if self.n_components <= 0:
            raise ValueError(f"Invalid n_components: {self.n_components} (must be > 0).")
        if tol is not None and tol <= 0:
            raise ValueError(f"Invalid tol: {tol} (must be > 0).")
        if max_iter is not None and max_iter <= 0:
            raise ValueError(f"Invalid max_iter: {max_iter} (must be > 0).")

        warm_start = self.warm_start if warm_start is None else warm_start
        max_iter = self.max_iter if max_iter is None else max_iter
        tol = self.tol if tol is None else tol
        if random_state is not None:
            self.random_state = random_state

        X = X.to(self.device)
        self.dtype = X.dtype
        if self.n_features is None:
            self.n_features = X.shape[1]
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but expected {self.n_features}.")

        self._check_nan_input(X)
        if self.missing == 'em' and labels is not None:
            raise NotImplementedError(
                "missing='em' does not support fit(X, labels=...) (supervised / semi-supervised fitting)."
            )
        X_fill = self._prepare_missing_fit(X) if self._missing_active(X) else None

        self._check_noise_supported(labels)
        noise_mask = self._prepare_noise(X) if self.noise else None

        # ===============================================================
        # (Semi-)supervised fitting (Phase 5): bypass the unsupervised
        # n_init loop below entirely. See the Phase 5 design doc.
        # ===============================================================
        if labels is not None:
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels)
            labels = labels.to(device=self.device)
            if labels.dim() != 1 or labels.size(0) != X.size(0):
                raise ValueError(
                    f"labels must be a 1D array of length n_samples={X.size(0)}, "
                    f"got shape {tuple(labels.shape)}."
                )
            if torch.all(labels != -1):
                return self._fit_supervised(X, labels)
            return self._fit_semisupervised(X, labels, max_iter, tol)

        self.classes_ = None
        self.class_of_component_ = None

        best_lower_bound = -float("inf")
        best_snapshot = None
        best_random_state = None

        if self.random_state is not None and self.n_init > 1:
            warnings.warn(
                f"With n_init={self.n_init} and random_state={self.random_state}, "
                f"initializations will use random states [{self.random_state}, "
                f"{self.random_state + 1}, ..., {self.random_state + self.n_init - 1}]. "
                f"The best initialization's random state will be stored in best_random_state_.",
                UserWarning
            )

        for init_idx in range(self.n_init):
            if warm_start and self.n_init > 1:
                warnings.warn(
                    "warm_start=True with n_init > 1 will not re-initialize parameters "
                    "for each run, which may lead to identical results.",
                    UserWarning
                )

            if self.random_state is not None:
                current_random_state = self.random_state + init_idx
                torch.manual_seed(current_random_state)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_random_state)

            if not warm_start or not self.fitted_ or init_idx > 0:
                X_init = X if noise_mask is None else X[~noise_mask]
                if X_fill is not None:
                    X_init = X_fill  # k-means etc. must not see NaN (column-mean imputed copy)
                if keep is not None:  # weight-0 rows are dropped from the initialisation
                    if X_fill is not None or noise_mask is None:
                        X_init = X_init[keep]
                    else:
                        X_init = X[~noise_mask & keep]
                self._allocate_parameters(X_init, set_random_state=False)
                if self.noise:
                    self._init_noise_weights(noise_mask)

            self._fit_single_run(X, max_iter, tol, run_idx=init_idx)

            if torch.any(self.weights_ < 1e-8):
                warnings.warn(
                    "Some cluster(s) have near-zero weight. This may indicate degenerate solutions.",
                    UserWarning
                )

            if self.lower_bound_ > best_lower_bound:
                best_lower_bound = self.lower_bound_
                best_random_state = (self.random_state + init_idx
                                     if self.random_state is not None else None)
                best_snapshot = self._get_fit_snapshot()

            self.fitted_ = True

        if best_snapshot is not None:
            self._apply_snapshot(best_snapshot)
            self.best_random_state_ = best_random_state

        self.n_components_ = self.n_components
        self.active_ = torch.ones(self.n_components, dtype=torch.bool, device=self.device)

        if best_random_state is not None and self.n_init > 1:
            warnings.warn(
                f"Best result from random_state={best_random_state}. "
                f"To reproduce this specific result: use random_state={best_random_state} with n_init=1.",
                UserWarning
            )

        if not self.converged_:
            warnings.warn(
                "EM did not converge. Consider increasing max_iter or adjusting tol.",
                UserWarning
            )

        return self

    # ===================================================================
    # (Semi-)supervised fitting (Phase 5). See the design doc at
    # docs/superpowers/specs/2026-10-14-phase5-semi-supervised-pu-design.md
    # for the full rationale, in particular the two documented scope
    # reductions: tied covariance types aren't supported when any class's
    # `components_per_class` block is > 1, and `'sem'`/`'saem'`/online EM
    # aren't supported for semi-supervised (partially-labeled) fitting.
    # ===================================================================
    def _is_tied_covariance(self) -> bool:
        r"""
        Whether this instance's current covariance-like parameterization is
        shared ("tied") across *every* component, making it incompatible
        with fitting each class's component block as an independent
        sub-mixture (`_fit_supervised`'s `components_per_class > 1` path).
        `False` by default (e.g. `VonMisesFisherMixture`, which has no
        covariance at all); `GaussianMixture`/`StudentTMixture` override
        this with their own `covariance_type` check.
        """
        return False

    def _post_supervised_fit_warnings(self, component_counts: torch.Tensor):
        r"""
        Hook called once at the end of `_fit_supervised`, given the number
        of (labeled) rows that went into each fitted component, in
        component order. No-op by default; `GaussianMixture` overrides
        this with its pre-existing sparse-component warning.
        """
        pass

    def _resolve_classes_and_blocks(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        `classes_` and `class_of_component_` (shape `(n_components,)`,
        which class -- an index into `classes_` -- each component belongs
        to), resolved from `labels` and `self.components_per_class`.
        Components are laid out in contiguous per-class blocks, in
        `classes_`'s sorted order (MclustDA-style).

        When `components_per_class` is a plain `int`, `classes_` is the
        sorted distinct non-`-1` values actually present in `labels`. When
        it's a `dict`, its *keys* define the complete, authoritative set of
        classes instead -- every labeled value must be one of those keys,
        but a key needn't appear in `labels` at all. This is what lets
        `PositiveUnlabeledMixture` (Phase 5.2) reserve a block of
        `n_components_negative` components for an implicit "negative"
        class that is never directly labeled (every non-positive row is
        simply unlabeled, `-1`, not labeled `0`).

        Raises
        ------
        ValueError
            If `labels` has no non-`-1` values; if `components_per_class`
            is a dict and a label value isn't one of its keys, or any of
            its values is a non-positive int; or if the resulting total
            component count doesn't equal `self.n_components`.
        """
        known = labels[labels != -1]
        if known.numel() == 0:
            raise ValueError("labels must contain at least one non- -1 (known) value.")

        if isinstance(self.components_per_class, dict):
            labeled_classes = torch.unique(known, sorted=True)
            unknown_labels = [c for c in labeled_classes.tolist() if c not in self.components_per_class]
            if unknown_labels:
                raise ValueError(
                    f"components_per_class is missing an entry for label value(s) "
                    f"{unknown_labels}, which are present in labels."
                )
            class_values = sorted(self.components_per_class.keys())
            block_sizes = []
            for class_label in class_values:
                m_c = self.components_per_class[class_label]
                if not isinstance(m_c, int) or m_c <= 0:
                    raise ValueError(
                        f"components_per_class[{class_label!r}] must be a positive int, got {m_c!r}."
                    )
                block_sizes.append(m_c)
            classes = torch.tensor(class_values, dtype=labels.dtype, device=labels.device)
        else:
            classes = torch.unique(known, sorted=True)
            block_sizes = [self.components_per_class] * classes.numel()

        total = sum(block_sizes)
        if total != self.n_components:
            raise ValueError(
                f"components_per_class implies {total} total component(s) "
                f"({dict(zip(classes.tolist(), block_sizes))}), but "
                f"n_components={self.n_components}."
            )

        class_of_component = torch.repeat_interleave(
            torch.arange(classes.numel(), device=labels.device),
            torch.tensor(block_sizes, device=labels.device),
        )
        return classes, class_of_component

    def _fit_component_block(self, X_block: torch.Tensor, n_block: int, random_state: Optional[int]) -> dict:
        r"""
        Fit a block of `n_block` components via ordinary unsupervised EM on
        `X_block` alone, reusing `BaseMixture.fit`'s own n_init loop
        directly (not `self.fit`, to avoid re-entering a subclass's own
        `fit` override or this class's (semi-)supervised dispatch).
        `self.n_components`/`self.random_state` are temporarily
        reconfigured for the call and restored afterward -- the full
        fitted bookkeeping (`fitted_`/`converged_`/etc.) is deliberately
        *not* restored, since `_fit_supervised`, the only caller, always
        overwrites it itself once every class's block has been fit.

        Returns
        -------
        dict
            `self._get_param_snapshot()` for the fitted block.
        """
        orig_n_components = self.n_components
        orig_random_state = self.random_state
        try:
            self.n_components = n_block
            if random_state is not None:
                self.random_state = random_state
            BaseMixture.fit(self, X_block)
            return self._get_param_snapshot()
        finally:
            self.n_components = orig_n_components
            self.random_state = orig_random_state

    def _fit_supervised(self, X: torch.Tensor, labels: torch.Tensor):
        r"""
        Fully-labeled (no `-1`) fit: one component block per class. See the
        design doc for the two sub-cases (every block size 1: a single
        one-hot M-step over the whole dataset; any block > 1: each class's
        block fit independently, then assembled -- tied covariance types
        unsupported in that sub-case, see `_is_tied_covariance`).
        """
        import warnings

        classes, class_of_component = self._resolve_classes_and_blocks(labels)
        n_classes = classes.numel()
        block_sizes = [int((class_of_component == c).sum().item()) for c in range(n_classes)]
        n_samples = X.shape[0]

        if all(m == 1 for m in block_sizes):
            component_idx = torch.searchsorted(classes, labels)
            self._allocate_parameters(X)
            resp = torch.zeros(n_samples, self.n_components, device=self.device, dtype=self.dtype)
            resp[torch.arange(n_samples, device=self.device), component_idx] = 1.0
            self._m_step(X, resp)
            component_counts = torch.bincount(component_idx, minlength=self.n_components)
        else:
            if self._is_tied_covariance():
                raise ValueError(
                    "components_per_class > 1 is not supported together with a tied "
                    "covariance type (tied_full/tied_diag/tied_spherical): a single "
                    "shared covariance can't be assembled from independently-fit "
                    "per-class component blocks. Use a per-component covariance "
                    "type (e.g. 'full'/'diag'/'spherical') instead."
                )
            per_class_snapshots = []
            component_counts_parts = []
            for c_idx, class_label in enumerate(classes.tolist()):
                mask = labels == class_label
                X_c = X[mask]
                n_c = int(mask.sum().item())
                m_c = block_sizes[c_idx]
                snapshot = self._fit_component_block(X_c, m_c, random_state=self.random_state)
                snapshot['weights_'] = snapshot['weights_'] * (n_c / n_samples)
                per_class_snapshots.append(snapshot)
                component_counts_parts.append(torch.full((m_c,), n_c, dtype=torch.long, device=self.device))
            merged = {
                key: torch.cat([snap[key] for snap in per_class_snapshots], dim=0)
                for key in per_class_snapshots[0]
            }
            self._apply_snapshot(merged)
            component_counts = torch.cat(component_counts_parts)

        self.classes_ = classes
        self.class_of_component_ = class_of_component
        self.converged_ = True
        self.n_iter_ = 1
        self.best_random_state_ = None
        self.fitted_ = True
        self.n_components_ = self.n_components
        self.active_ = torch.ones(self.n_components, dtype=torch.bool, device=self.device)
        self.lower_bound_ = self.score(X)

        self._post_supervised_fit_warnings(component_counts)
        return self

    def _clamp_responsibilities(
        self, resp: torch.Tensor, is_labeled: torch.Tensor, block_mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        For labeled rows, zero responsibility outside the row's class block
        and renormalize inside it (one-hot when the block has 1 component);
        `self.label_confidence < 1` blends `label_confidence` mass on that
        renormalized in-block posterior with `1 - label_confidence` on the
        *original* (unclamped) out-of-block posterior, rescaled to sum to
        that remainder. Unlabeled rows pass through untouched.
        """
        if not bool(is_labeled.any()):
            return resp

        resp = resp.clone()
        labeled_resp = resp[is_labeled]
        block = block_mask[is_labeled]
        within_mass = (labeled_resp * block).sum(dim=1, keepdim=True)
        clamped = torch.where(block, labeled_resp / within_mass.clamp(min=1e-20), torch.zeros_like(labeled_resp))

        if self.label_confidence >= 1.0:
            resp[is_labeled] = clamped
            return resp

        outside_mass = (1.0 - within_mass).clamp(min=1e-20)
        outside_part = torch.where(block, torch.zeros_like(labeled_resp), labeled_resp) / outside_mass
        blended = self.label_confidence * clamped + (1.0 - self.label_confidence) * outside_part
        resp[is_labeled] = blended
        return resp

    def _semisupervised_log_likelihood(
        self, X: torch.Tensor, is_labeled: torch.Tensor, block_mask: torch.Tensor,
    ) -> float:
        r"""
        The plan's own semi-supervised objective:
        :math:`\sum_{\text{labeled}} \log \sum_{k \in \text{block}(c_i)} \pi_k f_k(x_i)
        + \sum_{\text{unlabeled}} \log \sum_k \pi_k f_k(x_i)`, averaged over
        samples (matching every other family's `lower_bound_` convention).
        Computed directly from `_component_log_prob`, not from the clamped
        responsibilities -- clamping is a statistics-accumulation device
        for the M-step, not part of the likelihood itself.
        """
        log_joint = self._component_log_prob(X) + torch.log(self.weights_ + 1e-20).unsqueeze(0)
        ll = torch.empty(X.shape[0], device=X.device, dtype=X.dtype)
        if bool(is_labeled.any()):
            masked = log_joint[is_labeled].masked_fill(~block_mask[is_labeled], float('-inf'))
            ll[is_labeled] = torch.logsumexp(masked, dim=1)
        if bool((~is_labeled).any()):
            ll[~is_labeled] = torch.logsumexp(log_joint[~is_labeled], dim=1)
        return ll.mean().item()

    def _fit_single_run_semisupervised(
        self, X: torch.Tensor, is_labeled: torch.Tensor, label_class_idx: torch.Tensor,
        max_iter: int, tol: float, run_idx: int = 0,
    ):
        import warnings

        block_mask = torch.zeros(X.shape[0], self.n_components, dtype=torch.bool, device=X.device)
        if bool(is_labeled.any()):
            block_mask[is_labeled] = (
                self.class_of_component_.unsqueeze(0) == label_class_idx[is_labeled].unsqueeze(1)
            )

        prev_lower_bound = -float("inf")
        prev_prev_lower_bound = None

        resp, _ = self._e_step(X)
        resp = self._clamp_responsibilities(resp, is_labeled, block_mask)
        self.lower_bound_ = self._semisupervised_log_likelihood(X, is_labeled, block_mask)
        ll_history = [self.lower_bound_]

        for n_iter in range(max_iter):
            if self.algorithm == 'cem':
                hard = self._c_step(resp)
                resp = torch.where(is_labeled.unsqueeze(1), resp, hard)

            if self.unlabeled_weight == 1.0:
                resp_for_mstep = resp
            else:
                resp_for_mstep = resp.clone()
                resp_for_mstep[~is_labeled] = resp_for_mstep[~is_labeled] * self.unlabeled_weight
            self._m_step(X, resp_for_mstep)
            if self._post_mstep_hook is not None:
                self._post_mstep_hook(X, resp_for_mstep)

            if self._check_convergence(self.lower_bound_, prev_lower_bound, prev_prev_lower_bound, tol):
                self.converged_ = True
                if self.verbose:
                    print(f"[Run {run_idx+1}] Iteration {n_iter}: "
                          f"log-likelihood={self.lower_bound_:.5f}, Converged!")
                break

            prev_prev_lower_bound = prev_lower_bound
            prev_lower_bound = self.lower_bound_
            resp, _ = self._e_step(X)
            resp = self._clamp_responsibilities(resp, is_labeled, block_mask)
            self.lower_bound_ = self._semisupervised_log_likelihood(X, is_labeled, block_mask)
            ll_history.append(self.lower_bound_)

            if self.verbose and (n_iter % self.verbose_interval == 0):
                print(f"[Run {run_idx+1}] Iteration {n_iter}: log-likelihood={self.lower_bound_:.5f}")

        if self.converged_:
            resp, _ = self._e_step(X)
            resp = self._clamp_responsibilities(resp, is_labeled, block_mask)
            if self.algorithm == 'cem':
                hard = self._c_step(resp)
                resp = torch.where(is_labeled.unsqueeze(1), resp, hard)
            self.lower_bound_ = self._semisupervised_log_likelihood(X, is_labeled, block_mask)
        else:
            warnings.warn(
                f"Run {run_idx+1}: semi-supervised EM did not converge after {max_iter} iterations.",
                UserWarning
            )

        self.n_iter_ = n_iter
        self.log_likelihood_history_ = torch.tensor(ll_history)

    def _fit_semisupervised(self, X: torch.Tensor, labels: torch.Tensor, max_iter: int, tol: float):
        r"""
        Partially-labeled (some `-1`) fit: semi-supervised EM, with labeled
        rows' responsibilities clamped to their class's component block
        each E-step (`_clamp_responsibilities`) and unlabeled rows'
        responsibilities scaled by `self.unlabeled_weight` before the
        M-step. See the design doc for the `algorithm`/`step_size` scope
        restriction below.
        """
        import warnings

        if self.algorithm not in ('em', 'cem'):
            raise NotImplementedError(
                f"Semi-supervised fitting (labels containing -1) with algorithm="
                f"{self.algorithm!r} is not implemented in this release -- only "
                f"'em' and 'cem' are supported for partially-labeled fits. See "
                f"docs/superpowers/specs/2026-10-14-phase5-semi-supervised-pu-design.md."
            )
        if self.step_size is not None:
            raise NotImplementedError(
                "Semi-supervised fitting (labels containing -1) with online/"
                "mini-batch EM (step_size set) is not implemented in this release. "
                "See docs/superpowers/specs/2026-10-14-phase5-semi-supervised-pu-design.md."
            )

        classes, class_of_component = self._resolve_classes_and_blocks(labels)
        self.classes_ = classes
        self.class_of_component_ = class_of_component

        is_labeled = labels != -1
        safe_labels = torch.where(is_labeled, labels, classes[0])
        label_class_idx = torch.searchsorted(classes, safe_labels)
        label_class_idx = torch.where(is_labeled, label_class_idx, torch.full_like(label_class_idx, -1))

        best_lower_bound = -float("inf")
        best_snapshot = None
        best_random_state = None

        if self.random_state is not None and self.n_init > 1:
            warnings.warn(
                f"With n_init={self.n_init} and random_state={self.random_state}, "
                f"initializations will use random states [{self.random_state}, "
                f"{self.random_state + 1}, ..., {self.random_state + self.n_init - 1}]. "
                f"The best initialization's random state will be stored in best_random_state_.",
                UserWarning
            )

        for init_idx in range(self.n_init):
            if self.random_state is not None:
                current_random_state = self.random_state + init_idx
                torch.manual_seed(current_random_state)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_random_state)

            self._allocate_parameters(X, set_random_state=False)
            if self._post_mstep_hook is not None:
                self._post_mstep_hook(X, None)
            self._fit_single_run_semisupervised(X, is_labeled, label_class_idx, max_iter, tol, run_idx=init_idx)

            if self.lower_bound_ > best_lower_bound:
                best_lower_bound = self.lower_bound_
                best_random_state = (self.random_state + init_idx
                                     if self.random_state is not None else None)
                best_snapshot = self._get_fit_snapshot()

            self.fitted_ = True

        if best_snapshot is not None:
            self._apply_snapshot(best_snapshot)
            self.best_random_state_ = best_random_state

        self.n_components_ = self.n_components
        self.active_ = torch.ones(self.n_components, dtype=torch.bool, device=self.device)

        if best_random_state is not None and self.n_init > 1:
            warnings.warn(
                f"Best result from random_state={best_random_state}. "
                f"To reproduce this specific result: use random_state={best_random_state} with n_init=1.",
                UserWarning
            )

        if not self.converged_:
            warnings.warn(
                "Semi-supervised EM did not converge. Consider increasing max_iter or adjusting tol.",
                UserWarning
            )

        return self

    # ===================================================================
    # Prediction and scoring
    # ===================================================================
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        r"""Hard cluster assignment (argmax responsibility)."""
        import warnings
        if not self.fitted_:
            warnings.warn("Model has not been fitted. Results may be unreliable.", UserWarning)
        elif not self.converged_:
            warnings.warn("Model did not converge. Results may be unreliable.", UserWarning)
        resp, _ = self._e_step(X.to(self.device))
        labels = torch.argmax(resp, dim=1)
        if self.noise:
            labels = torch.where(labels == resp.shape[1] - 1, torch.full_like(labels, -1), labels)
        return labels

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        r"""Soft cluster assignment (responsibilities); shape `(n, K + 1)` with `noise=True`
        (the last column is the noise responsibility, see `noise_proba`)."""
        import warnings
        if not self.fitted_:
            warnings.warn("Model has not been fitted. Results may be unreliable.", UserWarning)
        elif not self.converged_:
            warnings.warn("Model did not converge. Results may be unreliable.", UserWarning)
        resp, _ = self._e_step(X.to(self.device))
        return resp

    def noise_proba(self, X: torch.Tensor) -> torch.Tensor:
        r"""Posterior probability that each row of `X` is noise, shape `(n,)` (needs `noise=True`)."""
        if not self.noise:
            raise RuntimeError("noise_proba requires a model constructed with noise=True.")
        return self.predict_proba(X)[:, -1]

    def predict_class(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Predicted *class* label for each sample, after a (semi-)supervised
        `fit(X, labels=...)` (Phase 5): sums responsibility over each
        class's component block and returns the class with the highest
        total, mapped back through `classes_` to the original label
        values. `predict` still returns raw component indices.

        Raises
        ------
        RuntimeError
            If this model wasn't fit with `labels=...` (`classes_` is `None`).
        """
        if self.classes_ is None:
            raise RuntimeError(
                "predict_class requires a model fit with labels=... (classes_ is None); "
                "use predict for the raw component assignment instead."
            )
        resp = self.predict_proba(X)
        n_classes = self.classes_.numel()
        class_resp = torch.zeros(resp.shape[0], n_classes, device=resp.device, dtype=resp.dtype)
        class_resp.index_add_(1, self.class_of_component_, resp)
        class_idx = torch.argmax(class_resp, dim=1)
        return self.classes_[class_idx]

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        r"""Per-sample log-likelihood `log p(x_i | theta)`."""
        import warnings
        if not self.fitted_:
            warnings.warn("Model has not been fitted. Results may be unreliable.", UserWarning)
        elif not self.converged_:
            warnings.warn("Model did not converge. Results may be unreliable.", UserWarning)
        _, log_prob_norm = self._e_step(X.to(self.device))
        return log_prob_norm

    def score(self, X: torch.Tensor) -> float:
        r"""Mean log-likelihood of `X`."""
        return self.score_samples(X).mean().item()

    # ===================================================================
    # Information criteria. _score_for_information_criterion and
    # _resp_for_information_criterion are overridable hooks:
    # GaussianMixture overrides both with its Gibbs-mode-aware versions;
    # the defaults here (plain score/E-step) are correct for every family
    # without pruned components.
    # ===================================================================
    def _score_for_information_criterion(self, X: torch.Tensor) -> float:
        return self.score(X)

    def _resp_for_information_criterion(self, X: torch.Tensor) -> torch.Tensor:
        resp, _ = self._e_step(X.to(self.device))
        return resp

    def bic(self, X: torch.Tensor) -> float:
        r"""Bayesian Information Criterion (lower is better)."""
        n_samples = X.shape[0]
        ll = self._score_for_information_criterion(X)
        return -2.0 * ll * n_samples + self._n_parameters() * math.log(n_samples)

    def aic(self, X: torch.Tensor) -> float:
        r"""Akaike Information Criterion (lower is better)."""
        ll = self._score_for_information_criterion(X)
        return -2.0 * ll * X.shape[0] + 2.0 * self._n_parameters()

    def icl(self, X: torch.Tensor, mode: str = 'hard') -> float:
        r"""Integrated Completed Likelihood (Biernacki, Celeux & Govaert, 2000). Lower is better."""
        valid_modes = ('hard', 'soft')
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

        bic_value = self.bic(X)
        resp = self._resp_for_information_criterion(X)

        if mode == 'hard':
            z_hat = torch.argmax(resp, dim=1)
            tau_hat = resp[torch.arange(resp.shape[0], device=resp.device), z_hat]
            penalty = -2.0 * torch.log(tau_hat).sum().item()
        else:
            penalty = -2.0 * torch.special.xlogy(resp, resp).sum().item()

        return bic_value + penalty

    # ===================================================================
    # Sampling: a plain shell. GaussianMixture overrides `sample` entirely
    # with its much richer implementation (confidence regions, rejection
    # sampling, etc.); this is what StudentTMixture/VonMisesFisherMixture
    # use.
    # ===================================================================
    def sample(self, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Draw `n_samples` from the fitted mixture: choose a component for
        each sample according to `weights_`, then draw from that
        component via `_sample_component`.

        Returns
        -------
        samples : torch.Tensor, shape (n_samples, n_features)
        component_indices : torch.Tensor, shape (n_samples,)
        """
        if self.noise:
            probs = torch.cat([self.weights_, self.noise_weight_.reshape(1)])
            component_indices = torch.multinomial(probs, n_samples, replacement=True)
        else:
            component_indices = torch.multinomial(self.weights_, n_samples, replacement=True)
        samples = torch.empty(n_samples, self.n_features, dtype=self.dtype, device=self.device)
        K = self.n_components_ or self.n_components
        for k in range(K):
            mask = component_indices == k
            count = int(mask.sum().item())
            if count > 0:
                samples[mask] = self._sample_component(k, count)
        if self.noise:
            noise_mask = component_indices == K
            count = int(noise_mask.sum().item())
            if count > 0:
                samples[noise_mask] = self._sample_noise(count).to(samples.dtype)
            component_indices = torch.where(noise_mask, torch.full_like(component_indices, -1), component_indices)
        return samples, component_indices

    # ===================================================================
    # Generic GEM step: an autograd-based update for a parameter that has
    # no closed form, accepted only if it improves the objective. Lives
    # here (rather than duplicated per-family) so any family -- or a
    # future GenericMixture -- can use it without re-deriving the
    # LBFGS-and-accept-on-improvement pattern Phase 3's GPCM orientation
    # optimization already established.
    # ===================================================================
    @staticmethod
    def _gem_step(
        objective: Callable[[], torch.Tensor], params, n_steps: int = 20,
    ) -> bool:
        r"""
        Minimize `objective()` (a closure reading the current values of
        `params`, a list of `torch.nn.Parameter`) via up to `n_steps`
        `torch.optim.LBFGS` steps. Accepts the update only if `objective()`
        strictly decreased; otherwise the parameters are restored to their
        pre-call values.

        Returns
        -------
        bool
            Whether the step was accepted.
        """
        with torch.no_grad():
            initial = objective().item()
            initial_values = [p.detach().clone() for p in params]

        optimizer = torch.optim.LBFGS(params, max_iter=n_steps, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            final = objective().item()
            if final < initial:
                return True
            for p, v in zip(params, initial_values):
                p.copy_(v)
            return False
