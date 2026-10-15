import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from typing import Callable, Optional, Sequence, Tuple, Union
import warnings
import math
from .gmm_init import GMMInitializer
from .stats import SufficientStats
from .covariance.gpcm import mstep_gpcm, GPCMResult
from .mixtures.base import BaseMixture
from .mixtures import missing as _missing
from .mixtures.conditional import ConditionalGaussianMixture, _validate_dims
from scipy.stats import chi2


# ===========================================================================
# GPCM (Celeux & Govaert 1995) covariance-type naming: the 14 mclust codes
# and their tgmm aliases for the 6 pre-existing covariance types. See
# docs/superpowers/specs/2026-10-14-phase3-gpcm-covariance-models-design.md.
# ===========================================================================
_MCLUST_TO_TGMM_NAME = {
    'EII': 'tied_spherical',
    'VII': 'spherical',
    'EEI': 'tied_diag',
    'VVI': 'diag',
    'EEE': 'tied_full',
    'VVV': 'full',
}
_TGMM_NAME_TO_MCLUST = {v: k for k, v in _MCLUST_TO_TGMM_NAME.items()}
_NEW_GPCM_CODES = ('VEI', 'EVI', 'VEE', 'EVE', 'EEV', 'VVE', 'VEV', 'EVV')
_ALL_MCLUST_CODES = tuple(_MCLUST_TO_TGMM_NAME.keys()) + _NEW_GPCM_CODES


def _resolve_covariance_type(covariance_type: str):
    r"""
    Resolve a user-supplied `covariance_type` (a tgmm name, an mclust code,
    or the legacy `'tied'`/`'isotropic'` aliases, all case-insensitive for
    the mclust codes) into `(covariance_type, covariance_type_)`:
    `covariance_type` is the tgmm name for the 6 pre-existing types (for
    backward compatibility -- existing code reading `self.covariance_type`
    keeps working) or the mclust code for the 8 new types (no tgmm name
    exists for those); `covariance_type_` is always the canonical mclust
    code.
    """
    if covariance_type == "tied":
        covariance_type = "tied_full"
    elif covariance_type == "isotropic":
        covariance_type = "spherical"

    if covariance_type in _TGMM_NAME_TO_MCLUST:
        return covariance_type, _TGMM_NAME_TO_MCLUST[covariance_type]

    code = covariance_type.upper() if isinstance(covariance_type, str) else covariance_type
    if code in _ALL_MCLUST_CODES:
        if code in _MCLUST_TO_TGMM_NAME:
            return _MCLUST_TO_TGMM_NAME[code], code
        return code, code

    valid = ("full", "diag", "spherical", "tied_full", "tied_diag", "tied_spherical") + _ALL_MCLUST_CODES
    raise ValueError(f"covariance_type must be one of {valid}, got {covariance_type!r}")



class GaussianMixture(BaseMixture):
    r"""
    A Gaussian Mixture Model (GMM) based on Expectation-Maximisation (EM) implemented in PyTorch.

    This GMM supports:
    - The Expectation-Maximization (EM) algorithm
    - The Classification EM (CEM) algorithm as an alternative to standard EM
    - Multiple random initializations (n_init)
    - Configurable covariance types (full, diag, spherical, tied_full, tied_diag, tied_spherical)
    - Maximum Likelihood Estimation (MLE) and Maximum a Posteriori (MAP) estimation with priors
    - Bayesian/Akaike information criteria (`bic`, `aic`) for model selection

    Parameters
    ----------
    n_components : int or None, optional
        Number of mixture components. If None, selects Gibbs-sampling mode:
        the number of components is inferred during `fit` (up to
        `max_components`) instead of fixed in advance, and mean_prior,
        mean_precision_prior, covariance_prior, and degrees_of_freedom_prior
        become required. See `max_components`, `alpha`, and `suggest_priors`.
        (default: None)
    n_features : int, optional
        Dimensionality of the input data (number of features). If None, inferred from data. (default: None)
    covariance_type : str, optional
        Type of covariance parameters to use. Either a tgmm name --
        'full', 'diag', 'spherical', 'tied_full', 'tied_diag',
        'tied_spherical' -- or any of the 14 Celeux & Govaert (1995) GPCM
        mclust codes (case-insensitive): 'EII', 'VII', 'EEI', 'VEI', 'EVI',
        'VVI', 'EEE', 'VEE', 'EVE', 'EEV', 'VVE', 'VEV', 'EVV', 'VVV' (the
        six tgmm names are aliases for 'EII'/'VII'/'EEI'/'VVI'/'EEE'/'VVV'
        respectively -- see `covariance_type_`, and the "GPCM Covariance
        Models" user-guide page for the 8 codes without a tgmm name).
        Aliases: 'tied' → 'tied_full', 'isotropic' → 'spherical'.
        (default: 'full')
    max_iter : int, optional
        EM mode: maximum number of EM iterations to perform; EM stops
        early once the relative log-likelihood improvement drops below
        `tol` (see `tol`). Gibbs mode (`n_components=None`): the *exact*
        number of Gibbs sweeps run -- there is no early-stopping /
        convergence check in Gibbs mode (see `converged_`), so `fit()`
        always runs precisely `max_iter` sweeps, never fewer. Each Gibbs
        sweep is an O(n_samples) Python-level loop over every point
        (measured ~0.114 ms/point/sweep), so the shared default of 1000
        can mean tens of minutes at moderate `n_samples` (e.g. roughly
        19 minutes at n_samples=10,000). The default is left unchanged
        here so existing EM-tuned code isn't surprised by a different
        value -- when using Gibbs mode, pass an explicit, usually much
        smaller, `max_iter` instead of relying on this default.
        (default: 1000)
    tol : float, optional
        Convergence threshold based on relative improvement in log-likelihood. (default: 1e-4)
    reg_covar : float, optional
        Non-negative regularization added to the diagonal of covariance matrices
        to prevent singularity. (default: 1e-6)
    n_init : int, optional
        Number of random initializations to try. The best run (highest log-likelihood)
        is kept. When n_init > 1, each initialization uses random_state + i as its seed
        to ensure both reproducibility and diversity. (default: 1)
    init_means : str or torch.Tensor, optional
        Method for initializing means, or a tensor of initial means.
        - If str: 'kmeans', 'kpp', 'random', 'points', 'maxdist'
        - If tensor: shape (n_components, n_features) exactly (no broadcasting)
        (default: 'kmeans')
    init_weights : str or torch.Tensor, optional
        Method for initializing weights, or a tensor of initial weights.
        - If str: 'uniform', 'random', 'kmeans'
        - If tensor: shape (n_components,)
        (default: 'uniform')
    init_covariances : str or torch.Tensor, optional
        Method for initializing covariances, or a tensor of initial covariances.
        - If str: 'empirical', 'eye', 'random', 'global'
        - If tensor: shape depends on covariance_type (supports broadcasting)
          * full: (n_features, n_features) or (n_components, n_features, n_features)
          * diag: (n_features,) or (n_components, n_features)
          * spherical: scalar or (n_components,)
        (default: 'empirical')
    random_state : int or None, optional
        Random seed for reproducibility. If None, uses PyTorch's internal seed. 
        When n_init > 1, serves as the base seed (initialization i uses random_state + i).
        (default: None)
    warm_start : bool, optional
        If True, reuse the solution of the previous call to `fit` as initialization.
        (default: False)
    algorithm : str, optional
        Which EM variant to run: 'em' (standard); 'cem' (Classification EM,
        hard assignment between the E- and M-steps); 'sem' (Stochastic EM,
        Celeux & Diebolt 1985 -- samples a hard assignment from the
        responsibilities each iteration instead of taking the argmax; use
        it to escape poor local optima or to detect an overspecified `K`
        via `n_empty_components_`); or 'saem' (Stochastic Approximation EM,
        Delyon, Lavielle & Moulines 1999 -- SEM with a `step_size`-weighted
        running-statistics blend across iterations, which *does* converge
        pointwise, unlike plain SEM). Gibbs mode (`n_components=None`) only
        supports `'em'`. (default: 'em')
    cem : bool, optional
        Deprecated; use `algorithm='cem'` instead. If True, raises a
        `DeprecationWarning` and is treated as `algorithm='cem'`. Passing
        `cem=True` together with an explicit `algorithm` other than 'em'
        or 'cem' raises `ValueError`. (default: False)
    batch_size : int or None, optional
        If set, the E-step and sufficient-statistics accumulation process
        `X` in row chunks of this size instead of all at once, bounding
        peak memory to roughly `O(batch_size * n_components * n_features)`
        regardless of `n_samples`. With `step_size=None` (the default) this
        is purely a memory/speed trade-off: results are the same (up to
        ordinary floating-point summation-order differences) as
        `batch_size=None`. With `step_size` set, `batch_size` instead
        becomes the mini-batch size for online EM -- see `step_size`.
        (default: None)
    convergence : str, optional
        Stopping rule used by the EM/CEM/SAEM loop (SEM never checks
        convergence; see `algorithm`): 'relative' (stop when the
        relative change in mean log-likelihood drops below `tol`, as
        before) or 'aitken' (stop once the Aitken-accelerated estimate of
        the asymptotic log-likelihood is within `tol` of the current
        value, which can detect convergence earlier on slowly-crawling
        fits; falls back to the relative rule for any iteration where the
        Aitken acceleration factor is undefined or >= 1). (default:
        'relative')
    temperature : tuple of (float, int) or None, optional
        Deterministic annealing EM (Ueda & Nakano, 1998), for
        `algorithm in ('em', 'cem', 'sem')`. `(beta_0, n_anneal)`: the
        E-step's joint log-density is scaled by a temperature `beta` that
        increases geometrically from `beta_0` (in `(0, 1)`) at iteration 0
        to `1.0` at iteration `n_anneal` and is held at `1.0` after.
        Convergence checks are suspended while `beta < 1.0`. Use this when
        initialization is unreliable (e.g. a saddle point equidistant from
        several true clusters) -- annealing smooths the likelihood surface
        early on, when standard EM would otherwise get stuck. The schedule
        actually used is recorded in `annealing_schedule_`. `None` disables
        annealing (`beta` fixed at `1.0`, today's behaviour). (default:
        None)
    step_size : float, str, or callable, optional
        Enables online/mini-batch EM for `algorithm='em'` (requires
        `batch_size` to also be set) and controls `algorithm='saem'`'s
        stochastic-approximation step size (Cappé & Moulines, 2009;
        Delyon, Lavielle & Moulines, 1999). A float is a constant step
        size; `'auto'` resolves to `gamma_t = (t + 2) ** -0.7` (the
        documented default, `t` the running update count -- mini-batches
        for `'em'`, iterations for `'saem'`); a callable `t -> gamma_t`
        gives a custom schedule. For `algorithm='saem'`, `None` (the
        default) is treated as `'auto'` rather than raising, since SAEM's
        convergence depends on a decaying step size by construction. For
        `algorithm='em'`, `None` (the default) keeps `batch_size` as a
        pure memory optimization (exact EM) rather than switching to
        online EM. (default: None)
    sem_burn_in : int or None, optional
        `algorithm='sem'` only: sweeps excluded from the `sem_estimate`
        selection below. If `None`, resolved to `max_iter // 5`. (default:
        None)
    sem_estimate : str, optional
        `algorithm='sem'` only: how to summarise the post-burn-in sweeps
        into a final estimate. `'best'` keeps the sweep with the highest
        observed log-likelihood; `'mean'` averages `weights_`/`means_`/
        `covariances_` across every post-burn-in sweep (the covariance
        average is a plain arithmetic mean, which stays positive
        semi-definite). Averaging assumes no label switching across
        sweeps -- when `K` may be overspecified, prefer `'best'`.
        (default: 'best')
    weight_concentration_prior : torch.Tensor or None, optional
        Dirichlet concentration prior for the mixture weights (MAP estimation).
        Shape: (n_components,) or scalar to broadcast. (default: None)
    mean_prior : torch.Tensor or None, optional
        Prior for the component means (MAP estimation). Must be used with mean_precision_prior.
        Shape: (n_features,) or (n_components, n_features). (default: None)
    mean_precision_prior : float or None, optional
        Precision (inverse variance) of the mean prior (MAP estimation). (default: None)
    covariance_prior : torch.Tensor or None, optional
        Prior for the covariances (MAP estimation). Must be used with degrees_of_freedom_prior.
        Shape depends on covariance_type (supports broadcasting like init_covariances).
        (default: None)
    degrees_of_freedom_prior : float or None, optional
        Degrees of freedom for the Wishart/Inverse-Wishart prior on covariances (MAP).
        Must be > n_features - 1. (default: None)
    max_components : int or None, optional
        Gibbs mode only (n_components=None): upper bound on the number of
        components tracked during sampling. Ignored in EM mode. (default: 20)
    alpha : float, optional
        Gibbs mode only: Dirichlet process concentration parameter, controlling
        how readily new components are proposed. Ignored in EM mode. (default: 1.0)
    burn_in : int or None, optional
        Gibbs mode, unbounded variant only (`max_components=None`):
        number of initial sweeps excluded when recording
        `n_components_history_` (the per-sweep active-component-count
        trace). Does **not** affect the fitted `weights_`/`means_`/
        `covariances_` themselves -- those are always taken from the
        final sweep regardless of `burn_in`, in both truncated and
        unbounded Gibbs mode. Ignored entirely in truncated Gibbs mode
        (`max_components` is an int) and in EM mode. If None, resolved
        internally to `max(1, max_iter // 5)`. (default: None)
    weight_threshold : float or None, optional
        Gibbs mode only: a component is treated as inactive (excluded
        from `active_`/`n_components_`) when its *expected point count*
        -- `weight_i * n_samples`, not the raw weight/probability itself
        -- is not greater than this threshold. E.g. the resolved default
        of 1.0 means "expected to own at most one point." Ignored in EM
        mode. (default: None, resolved internally to 1.0)
    init_k : int or None, optional
        Gibbs mode only: number of components to seed the sampler with.
        Ignored in EM mode. (default: None)
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
    frozen_components : sequence of int, optional
        Indices of components whose `means_` / `covariances_` are held at their *initial* values for the
        whole fit (set them with tensor `init_means` / `init_covariances`, or `warm_start`); their mixing
        weights still update, and priors never shrink them. Not available for the tied covariance types or
        the 8 GPCM codes without a classic name (a factor is shared across components) and not in Gibbs
        mode. (default: ())
    missing : {'error', 'em'}, optional
        How `NaN` entries of `X` are handled. `'error'` (default): `NaN` raises `ValueError`.
        `'em'`: `NaN` marks missing-at-random values and the likelihood of the *observed* entries
        is maximised by EM (rows are grouped by missingness pattern; the E-step uses the marginal
        `N(x_o; mu_o, Sigma_oo)` and the M-step statistics use the conditional mean/covariance of
        the missing block). Works for every covariance type, `algorithm` in
        `{'em', 'cem', 'sem', 'saem'}`, `temperature` and `batch_size`; `predict`, `predict_proba`,
        `score_samples`, `score`, `bic`/`aic`/`icl` accept incomplete rows and `impute(X)` fills
        them in. Not supported with `noise=True`, Gibbs mode, `labels=`, online EM / `partial_fit`.
        On data without `NaN` the result is identical to `'error'`. (default: 'error')
    verbose : bool, optional
        If True, print progress during EM iterations. (default: False)
    verbose_interval : int, optional
        Frequency (in iterations) at which to print progress when verbose=True. (default: 10)
    device : str or None, optional
        Device on which to run computations ('cpu' or 'cuda'). If None, uses GPU if
        available, otherwise CPU. (default: None)

    Attributes
    ----------
    weights_ : torch.Tensor
        Mixture component weights of shape (n_components,).
    means_ : torch.Tensor
        Mixture component means of shape (n_components, n_features).
    covariances_ : torch.Tensor
        Mixture component covariances. Shape depends on `covariance_type`.
    initial_weights_ : torch.Tensor
        Initial mixture component weights before EM optimization, shape (n_components,).
    initial_means_ : torch.Tensor
        Initial mixture component means before EM optimization, shape (n_components, n_features).
    initial_covariances_ : torch.Tensor
        Initial mixture component covariances before EM optimization.
    dtype : torch.dtype
        Floating point dtype used for all model parameters. Set to the dtype
        of the data passed to `fit` (e.g. torch.float64 is fully supported),
        or torch.get_default_dtype() before the first call to `fit`.
    fitted_ : bool
        Whether the model has been fitted.
    converged_ : bool
        Whether the EM algorithm converged in the best run. In Gibbs mode
        (`n_components=None`), this is unconditionally `True` after
        `fit()` regardless of how many sweeps ran or any actual
        convergence assessment -- Gibbs sampling has no equivalent of
        EM's log-likelihood-plateau stopping criterion, so this flag
        does not indicate sampler convergence there (and `sample()`'s
        "did not converge" warning, gated on this flag, can never fire
        for a Gibbs-mode fit). If precision matters for a Gibbs-mode fit,
        inspect `n_components_history_` (unbounded mode only) for
        component-count stability across sweeps, or increase `max_iter`.
    n_iter_ : int
        Number of EM iterations performed in the best run.
    lower_bound_ : float
        Log-likelihood lower bound on the fitted data for the best run.
    best_random_state_ : int or None
        The random state that produced the best result when n_init > 1.
        Useful for reproducing the specific best initialization.
    n_components_ : int or None
        Number of components after fitting. Equal to `n_components` in EM
        mode; in Gibbs mode, the inferred count (set once fit() supports
        Gibbs mode). None before the first fit.
    active_ : torch.Tensor or None
        Gibbs mode only: boolean/index mask of which of the tracked
        `max_components` slots are active in the fitted result. None in EM
        mode and before the first fit.
    classes_ : torch.Tensor or None
        Sorted distinct non-`-1` label values seen by the most recent
        `fit(X, labels=...)` call (supervised or semi-supervised);
        `class_of_component_[k]` indexes into `classes_` for component
        `k`'s class. `None` after an unsupervised fit (no `labels`) or
        before the first fit. Not persisted by `save`/`load` or
        `save_state_dict`/`load_state_dict`.
    class_of_component_ : torch.Tensor or None
        Shape `(n_components,)`: which class (an index into `classes_`)
        each component belongs to, per `components_per_class`. `None`
        after an unsupervised fit or before the first fit.
    log_likelihood_history_ : torch.Tensor or None
        Per-iteration/sweep/epoch mean log-likelihood trace from the most
        recent EM-mode fit (every `algorithm`, including online EM). `None`
        before the first fit, and in Gibbs mode (which has no equivalent
        notion of a monotone trace to record).
    annealing_schedule_ : list of float or None
        The `beta` value actually used at each E-step of the most recent
        fit, when `temperature` is set. `None` otherwise.
    n_empty_components_ : int
        `algorithm='sem'` only: total number of component re-seeds across
        every sweep of the most recent fit (a component emptying out is
        evidence `n_components` may be too large for the data). `0` for
        every other algorithm.
    covariance_type_ : str
        The canonical mclust GPCM code for `covariance_type` (e.g. `'VVV'`
        for `covariance_type='full'`), always set regardless of which
        name/code was passed to the constructor.
    volume_ : torch.Tensor or None
        GPCM decomposition `Sigma_k = volume_k * orientation_k @
        diag(shape_k) @ orientation_k^T`: scalar (E-volume models) or
        shape `(n_components,)` (V-volume models). Set after the first fit
        for the 8 new GPCM covariance types (`'VEI'`, `'EVI'`, `'VEE'`,
        `'EVE'`, `'EEV'`, `'VVE'`, `'VEV'`, `'EVV'`); `None` for the 6
        pre-existing types.
    shape_ : torch.Tensor or None
        See `volume_`: shape `(n_features,)` (E-shape models) or
        `(n_components, n_features)` (V-shape models), determinant 1.
    orientation_ : torch.Tensor or None
        See `volume_`: shape `(n_features, n_features)` (E- or
        I-orientation models) or `(n_components, n_features, n_features)`
        (V-orientation models).
    """

    def __init__(
        self,
        # Core model parameters
        n_components: Optional[int] = None,
        n_features: int = None,
        covariance_type: str = 'full',

        # Convergence and training parameters
        max_iter: int = 1000,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        gpcm_tol: float = 1e-6,
        gpcm_max_iter: int = 20,

        # Initialization parameters (accepts str method or torch.Tensor values)
        init_means='kmeans',
        init_weights='uniform',
        init_covariances='empirical',

        # Random state and restart options
        random_state: int = None,
        warm_start: bool = False,

        # Algorithm options
        algorithm: str = 'em',
        cem: bool = False,
        batch_size: Optional[int] = None,
        convergence: str = 'relative',
        temperature: Optional[Tuple[float, int]] = None,
        step_size: Optional[Union[float, str, Callable[[int], float]]] = None,
        sem_burn_in: Optional[int] = None,
        sem_estimate: str = 'best',

        # Semi-supervised / positive-unlabeled fitting (Phase 5; see fit()'s
        # labels= parameter)
        components_per_class: Union[int, dict] = 1,
        unlabeled_weight: float = 1.0,
        label_confidence: float = 1.0,

        # Uniform noise component (v0.5 item 1.1; see BaseMixture._init_noise)
        noise: bool = False,
        noise_volume: Union[str, float] = 'bbox',
        noise_prior: float = 0.05,
        noise_init=None,

        # Missing values (v0.5 item 1.2; see BaseMixture._init_missing)
        missing: str = 'error',

        # Frozen components (v0.5 item 1.4; see BaseMixture._init_frozen)
        frozen_components: Sequence[int] = (),

        # Prior parameters for MAP estimation (EM mode) / required NIW prior (Gibbs mode)
        weight_concentration_prior: torch.Tensor = None,
        mean_prior: torch.Tensor = None,
        mean_precision_prior: float = None,
        covariance_prior: torch.Tensor = None,
        degrees_of_freedom_prior: float = None,

        # Gibbs-sampling mode (n_components=None) parameters -- ignored in EM mode
        max_components: Optional[int] = 20,
        alpha: float = 1.0,
        burn_in: Optional[int] = None,
        weight_threshold: Optional[float] = None,
        init_k: Optional[int] = None,

        # Output and device options
        verbose: bool = False,
        verbose_interval: int = 10,
        device: str = None,

        **kwargs  # Catch deprecated parameters
    ):
        super().__init__()
        
        # ===================================================================
        # 1. Validate deprecated parameters
        # ===================================================================
        deprecated_params = {
            'init_params': ('init_means', 'init_means=\'kmeans\''),
            'cov_init_method': ('init_covariances', 'init_covariances=\'empirical\''),
            'weights_init': ('init_weights', 'init_weights=torch.tensor([...])'),
            'means_init': ('init_means', 'init_means=torch.tensor([...])'),
            'covariances_init': ('init_covariances', 'init_covariances=torch.tensor([...])')
        }
        
        for old_param, (new_param, example) in deprecated_params.items():
            if old_param in kwargs:
                raise TypeError(
                    f"Parameter '{old_param}' has been removed. "
                    f"Use '{new_param}' instead.\n"
                    f"The '{new_param}' parameter accepts both strings (method names) and tensors (explicit values).\n"
                    f"Example: {example}"
                )
        
        if kwargs:
            unexpected = ', '.join(f"'{k}'" for k in kwargs.keys())
            raise TypeError(f"GaussianMixture.__init__() got unexpected keyword argument(s): {unexpected}")

        # ===================================================================
        # 2. Store core model parameters
        # ===================================================================
        self.n_components = n_components
        self.n_features = n_features

        # Resolve covariance type: tgmm names, the legacy 'tied'/'isotropic'
        # aliases, and all 14 mclust GPCM codes (case-insensitive) are
        # accepted. self.covariance_type is the tgmm name for the 6
        # pre-existing types (backward compatible) or the mclust code for
        # the 8 new ones; self.covariance_type_ is always the canonical
        # mclust code. See _resolve_covariance_type.
        covariance_type, covariance_type_ = _resolve_covariance_type(covariance_type)
        self.covariance_type = covariance_type
        self.covariance_type_ = covariance_type_

        # Gibbs mode (n_components=None): tied_* and the 8 new GPCM
        # covariance types are rejected immediately -- a shared/
        # cross-component-coupled covariance couples every component's
        # residuals together, so Neal's Algorithm 3 (fully-collapsed Gibbs)
        # does not factor per component. See
        # docs/superpowers/specs/2026-07-28-dpgmm-design.md.
        if n_components is None and (
            covariance_type in ("tied_full", "tied_diag", "tied_spherical")
            or covariance_type_ in _NEW_GPCM_CODES
        ):
            raise NotImplementedError(
                f"Gibbs sampling does not support covariance_type={covariance_type!r}. "
                "A shared/tied covariance couples every component's residuals together, so "
                "Neal's Algorithm 3 (fully-collapsed Gibbs) does not factor per component -- "
                "this holds for both truncated and unbounded Gibbs. A correct treatment needs "
                "a partially-collapsed sampler (explicit Inverse-Wishart resampling of the "
                "shared covariance each sweep), which is out of scope for this class. See "
                "docs/superpowers/specs/2026-07-28-dpgmm-design.md."
            )

        # ===================================================================
        # 3. Store convergence and training parameters
        # ===================================================================
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        if gpcm_tol <= 0:
            raise ValueError(f"Invalid gpcm_tol: {gpcm_tol} (must be > 0).")
        if gpcm_max_iter <= 0:
            raise ValueError(f"Invalid gpcm_max_iter: {gpcm_max_iter} (must be > 0).")
        self.gpcm_tol = gpcm_tol
        self.gpcm_max_iter = gpcm_max_iter

        # ===================================================================
        # 4. Store initialization parameters
        # ===================================================================
        self.init_means = init_means
        self.init_weights = init_weights
        self.init_covariances = init_covariances

        # ===================================================================
        # 5. Store random state and restart options
        # ===================================================================
        self.random_state = random_state
        self.warm_start = warm_start

        # ===================================================================
        # 6. Store algorithm options
        # ===================================================================
        valid_algorithms = ('em', 'cem', 'sem', 'saem')
        if cem:
            warnings.warn(
                "cem=True is deprecated; use algorithm='cem' instead.",
                DeprecationWarning
            )
            if algorithm not in ('em', 'cem'):
                raise ValueError(
                    f"cem=True conflicts with algorithm={algorithm!r}; pass only one of them."
                )
            algorithm = 'cem'

        if algorithm not in valid_algorithms:
            raise ValueError(
                f"algorithm must be one of {valid_algorithms}, got {algorithm!r}"
            )

        if n_components is None and algorithm != 'em':
            raise ValueError(
                f"Gibbs sampling (n_components=None) only supports algorithm='em' -- "
                f"got algorithm={algorithm!r}. Gibbs sampling has no S-step, "
                "stochastic-approximation, or annealing analogue in this codebase."
            )

        self.algorithm = algorithm
        self.cem = (algorithm == 'cem')

        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size} (must be > 0).")
        self.batch_size = batch_size

        valid_convergence = ('relative', 'aitken')
        if convergence not in valid_convergence:
            raise ValueError(
                f"convergence must be one of {valid_convergence}, got {convergence!r}"
            )
        self.convergence = convergence

        if temperature is not None:
            beta0, n_anneal = temperature
            if not (0.0 < beta0 < 1.0):
                raise ValueError(f"temperature[0] (beta_0) must be in (0, 1), got {beta0}.")
            if n_anneal <= 0:
                raise ValueError(f"temperature[1] (n_anneal) must be > 0, got {n_anneal}.")
        self.temperature = temperature

        if step_size is not None:
            if isinstance(step_size, str) and step_size != 'auto':
                raise ValueError(f"step_size must be 'auto', a positive number, or callable, got {step_size!r}")
            if isinstance(step_size, (int, float)) and step_size <= 0:
                raise ValueError(f"step_size must be > 0, got {step_size}.")
        self.step_size = step_size

        self.sem_burn_in = sem_burn_in
        valid_sem_estimates = ('best', 'mean')
        if sem_estimate not in valid_sem_estimates:
            raise ValueError(f"sem_estimate must be one of {valid_sem_estimates}, got {sem_estimate!r}")
        self.sem_estimate = sem_estimate

        if not isinstance(components_per_class, (int, dict)):
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

        self._init_noise(noise, noise_volume, noise_prior, noise_init)
        if missing == 'em' and n_components is None:
            raise NotImplementedError(
                "missing='em' is not supported in Gibbs mode (n_components=None); use a fixed n_components."
            )
        self._init_missing(missing)
        self._init_frozen(frozen_components)
        self.__dict__['_frozen_ready'] = True
        if self.frozen_components:
            self._check_frozen_supported()
        if self.noise and n_components is None:
            raise NotImplementedError(
                "noise=True is not supported in Gibbs mode (n_components=None); use a fixed n_components."
            )
        if self.noise and weight_concentration_prior is not None:
            raise ValueError(
                "noise=True cannot be combined with weight_concentration_prior: a Dirichlet prior over the "
                "K component weights has no counterpart once the noise weight is part of the mixture."
            )

        self.classes_ = None
        self.class_of_component_ = None

        # ===================================================================
        # 6b. Store Gibbs-mode parameters (ignored in EM mode)
        # ===================================================================
        self.max_components = max_components
        self.alpha = float(alpha)
        self.burn_in = burn_in
        self.weight_threshold = weight_threshold if weight_threshold is not None else 1.0
        self.init_k = init_k

        # ===================================================================
        # 7. Configure device
        # ===================================================================
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Working floating dtype. Overridden by the dtype of X at fit() time,
        # so the model follows the data (e.g. torch.float64) instead of
        # silently downcasting to float32.
        self.dtype = torch.get_default_dtype()

        # ===================================================================
        # 8. Store output options
        # ===================================================================
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        # ===================================================================
        # 9. Initialize and validate priors
        # ===================================================================
        self.use_weight_prior = weight_concentration_prior is not None
        self.use_mean_prior = (mean_prior is not None) and (mean_precision_prior is not None)
        self.use_covariance_prior = (covariance_prior is not None) and (degrees_of_freedom_prior is not None)

        if n_components is None:
            # Gibbs mode: Neal's Algorithm 3 requires a full NIW prior to
            # integrate out means/covariances -- there is no MLE fallback
            # the way EM has. Rather than silently resolving a default (an
            # earlier version of this project's DPGMM class did this, and
            # the auto-resolved default measurably changed fitted
            # covariances -- see the covariance-prior-fix revision-history
            # entry in docs/superpowers/specs/2026-07-28-dpgmm-design.md),
            # require the caller to supply all four explicitly.
            missing = [
                name for name, value in [
                    ("mean_prior", mean_prior),
                    ("mean_precision_prior", mean_precision_prior),
                    ("covariance_prior", covariance_prior),
                    ("degrees_of_freedom_prior", degrees_of_freedom_prior),
                ] if value is None
            ]
            if missing:
                raise ValueError(
                    f"Gibbs sampling (n_components=None) requires an explicit NIW prior -- "
                    f"missing: {', '.join(missing)}. There is no default: an earlier "
                    "auto-resolved default materially changed fitted covariances in this "
                    "project's development (see docs/superpowers/specs/2026-07-28-dpgmm-design.md). "
                    "Use GaussianMixture.suggest_priors(X, n_components=<your max_components guess>) "
                    "for a principled starting point, inspect/adjust it, and pass it in explicitly."
                )
            if mean_precision_prior <= 0:
                raise ValueError("mean_precision_prior must be > 0.")

            # Stored directly (not broadcast per-component the way EM's
            # _init_priors does): Gibbs mode's components are created and
            # destroyed dynamically, so there is no fixed K to broadcast
            # against -- a single shared mean vector / covariance matrix,
            # not one per component slot. Reshaped to this class's
            # NIW-posterior-formula shape at fit() time by
            # _resolve_priors_gibbs (Task 2), once n_features is known.
            self.mean_prior = mean_prior
            self.mean_precision_prior = float(mean_precision_prior)
            self.covariance_prior = covariance_prior
            self.degrees_of_freedom_prior = float(degrees_of_freedom_prior)
            # Gibbs mode has no weight-concentration-prior concept (no
            # Dirichlet/stick-breaking weight prior parameter the way EM's
            # variational path does -- DPGMM never had one either); set to
            # None so the attribute always exists post-__init__, matching
            # every other mode/attribute in this class (e.g. save()/load()
            # round-tripping via __dict__ requires the attribute to exist).
            # Whatever the caller passed for weight_concentration_prior is
            # silently ignored here -- same treatment as other EM-only args
            # (n_init, warm_start) that Gibbs mode simply doesn't use.
            # use_weight_prior must agree with this: it's the flag other
            # code trusts to mean "weight_concentration_prior is a real,
            # usable tensor," so it's reset to False here rather than left
            # at whatever the raw constructor argument implied above.
            self.weight_concentration_prior = None
            self.use_weight_prior = False
        else:
            if covariance_type_ in _NEW_GPCM_CODES and (
                mean_prior is not None or covariance_prior is not None
            ):
                raise NotImplementedError(
                    f"MAP estimation is not implemented for GPCM model {covariance_type_}; "
                    "use 'full', 'diag', 'spherical', or one of the 'tied_*' types for MAP/NIW "
                    "fitting."
                )
            self._init_priors(
                weight_concentration_prior,
                mean_prior,
                mean_precision_prior,
                covariance_prior,
                degrees_of_freedom_prior
            )

        # ===================================================================
        # 10. Initialize model state variables
        # ===================================================================
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.initial_weights_ = None
        self.initial_means_ = None
        self.initial_covariances_ = None
        self.fitted_ = False
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")
        self.best_random_state_ = None
        self.n_components_ = None
        self.active_ = None
        self.log_likelihood_history_ = None
        self.annealing_schedule_ = None
        self.n_empty_components_ = 0
        self._online_running_stats_ = None
        self._online_step_count_ = 0
        self.volume_ = None
        self.shape_ = None
        self.orientation_ = None
        self._gpcm_state_ = None

    @staticmethod
    def suggest_priors(X: torch.Tensor, n_components: int, covariance_type: str = "full", random_state: int = None):
        r"""
        A principled starting point for the NIW prior Gibbs-mode fitting
        (`n_components=None`) requires, estimated from a quick k-means
        partition of X into `n_components` groups.

        Returns `(mean_prior, mean_precision_prior, covariance_prior,
        degrees_of_freedom_prior)` -- pass these back into the constructor
        explicitly; nothing calls this automatically (Gibbs mode has no
        default prior -- see `__init__`).

        `mean_prior` is X's overall mean. `covariance_prior` is the pooled
        within-cluster scatter of the k-means partition (mirrors the
        pooled-variance estimator from ANOVA/LDA: sum of each cluster's own
        scatter around its own mean, divided by total within-cluster
        degrees of freedom) -- deliberately not X's marginal covariance,
        which for well-separated multi-modal data conflates *between*-
        cluster spread with *within*-cluster spread and inflates every
        component's fitted covariance (a 10-d, 5-cluster, well-separated
        synthetic case measured a 74x mismatch between the two). See
        docs/superpowers/specs/2026-07-28-dpgmm-design.md. `mean_precision_prior`
        defaults to 0.1 and `degrees_of_freedom_prior` to `n_features + 2`,
        chosen (per that same design doc) to keep the prior weak relative
        to real data while staying numerically safe for near-empty
        components.

        Parameters
        ----------
        X : torch.Tensor
            Data to base the suggestion on, shape (n_samples, n_features).
        n_components : int
            Number of k-means groups to pool within-cluster scatter over.
            Doesn't need to match `max_components` exactly -- this is a
            prior guess at typical component scale, refined every sweep by
            Gibbs sampling itself, not the final answer.
        covariance_type : str, default='full'
            One of 'full', 'diag', 'spherical', 'tied_full', 'tied_diag',
            'tied_spherical' -- determines `covariance_prior`'s shape.
        random_state : int, optional
            Random seed for reproducibility of k-means initialization.
            If None, randomness is not controlled.

        Returns
        -------
        mean_prior : torch.Tensor, shape (n_features,)
        mean_precision_prior : float
        covariance_prior : torch.Tensor, shape depends on covariance_type
        degrees_of_freedom_prior : float

        See Also
        --------
        __init__ : `max_iter`'s docstring there explains that Gibbs mode
            treats it as an exact sweep count with no early stopping, and
            that the shared default of 1000 can be far slower than
            necessary for Gibbs mode -- worth passing an explicit,
            smaller value alongside the priors returned here.
        """
        n_features = X.shape[1]
        device, dtype = X.device, X.dtype

        X_cpu = X.cpu()
        centers = GMMInitializer.kmeans(X_cpu, n_components, random_state=random_state).to(device=device, dtype=dtype)
        init_labels = torch.cdist(X, centers).argmin(dim=1)

        mean_prior = X.mean(dim=0)
        mean_precision_prior = 0.1
        degrees_of_freedom_prior = float(n_features + 2)

        if covariance_type in ("full", "tied_full"):
            pooled_S = torch.zeros(n_features, n_features, device=device, dtype=dtype)
        elif covariance_type in ("diag", "tied_diag"):
            pooled_S = torch.zeros(n_features, device=device, dtype=dtype)
        else:
            pooled_S = torch.zeros((), device=device, dtype=dtype)

        dof = 0.0
        for c in range(n_components):
            mask = init_labels == c
            n_c = int(mask.sum().item())
            if n_c < 2:
                continue
            diff = X[mask] - X[mask].mean(dim=0)
            if covariance_type in ("full", "tied_full"):
                pooled_S = pooled_S + diff.t() @ diff
            elif covariance_type in ("diag", "tied_diag"):
                pooled_S = pooled_S + diff.pow(2).sum(dim=0)
            else:
                pooled_S = pooled_S + diff.pow(2).sum()
            dof += n_c - 1

        if dof > 0:
            covariance_prior = pooled_S / dof
        else:
            # Degenerate fallback (e.g. every initial cluster a singleton):
            # pooling is undefined, so fall back to the marginal covariance.
            empirical_cov = torch.cov(X.T) if X.shape[0] > 1 else torch.eye(
                n_features, device=device, dtype=dtype)
            if covariance_type in ("full", "tied_full"):
                covariance_prior = empirical_cov.clone()
            elif covariance_type in ("diag", "tied_diag"):
                covariance_prior = torch.diagonal(empirical_cov).clone()
            else:
                covariance_prior = torch.diagonal(empirical_cov).mean().clone()

        return mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior

    def _init_priors(
        self,
        weight_concentration_prior: Optional[torch.Tensor],
        mean_prior: Optional[torch.Tensor],
        mean_precision_prior: Optional[float],
        covariance_prior: Optional[torch.Tensor],
        degrees_of_freedom_prior: Optional[float]
    ):
        r"""
        Validate and store prior parameters for MAP estimation.

        This method validates the shapes and values of prior parameters and stores them
        for use during the M-step. Supports broadcasting for convenience.

        Parameters
        ----------
        weight_concentration_prior : torch.Tensor or None
            Dirichlet concentration prior for mixture weights.
        mean_prior : torch.Tensor or None
            Prior means for Gaussian components.
        mean_precision_prior : float or None
            Precision (inverse variance) for the mean prior.
        covariance_prior : torch.Tensor or None
            Prior for covariances (shape depends on covariance_type).
        degrees_of_freedom_prior : float or None
            Degrees of freedom for the Wishart/Inverse-Wishart prior.
        """
        # ---------------------------------------------------------------
        # Weight prior (Dirichlet)
        # ---------------------------------------------------------------
        if self.use_weight_prior:
            if not isinstance(weight_concentration_prior, torch.Tensor):
                weight_concentration_prior = torch.tensor(weight_concentration_prior, device=self.device, dtype=self.dtype)
            
            # Broadcast scalar or single value to all components
            if weight_concentration_prior.dim() == 0 or (weight_concentration_prior.dim() == 1 and weight_concentration_prior.numel() == 1):
                weight_concentration_prior = weight_concentration_prior.expand(self.n_components)
            elif weight_concentration_prior.dim() == 1 and weight_concentration_prior.numel() != self.n_components:
                raise ValueError(
                    f"weight_concentration_prior must be of shape ({self.n_components},) or a scalar, "
                    f"got {weight_concentration_prior.shape}."
                )
            self.weight_concentration_prior = weight_concentration_prior.to(device=self.device, dtype=self.dtype)
        else:
            self.weight_concentration_prior = None

        # ---------------------------------------------------------------
        # Mean prior (Gaussian)
        # ---------------------------------------------------------------
        if self.use_mean_prior:
            # Broadcast (n_features,) to (n_components, n_features)
            if mean_prior.shape == (self.n_features,):
                mean_prior = mean_prior.unsqueeze(0).expand(self.n_components, -1)
            elif mean_prior.shape != (self.n_components, self.n_features):
                raise ValueError(
                    f"mean_prior must be of shape ({self.n_components}, {self.n_features}) "
                    f"or ({self.n_features},). Got {mean_prior.shape}."
                )
            if mean_precision_prior <= 0:
                raise ValueError("mean_precision_prior must be > 0.")
            
            self.mean_prior = mean_prior.to(device=self.device, dtype=self.dtype)
            self.mean_precision_prior = float(mean_precision_prior)
        else:
            self.mean_prior = None
            self.mean_precision_prior = None

        # ---------------------------------------------------------------
        # Covariance prior (Wishart/Inverse-Wishart)
        # ---------------------------------------------------------------
        if self.use_covariance_prior:
            self.degrees_of_freedom_prior = float(degrees_of_freedom_prior)
            
            # Validate degrees of freedom
            if self.degrees_of_freedom_prior <= self.n_features - 1:
                raise ValueError(
                    f"degrees_of_freedom_prior must be > {self.n_features - 1}, "
                    f"got {self.degrees_of_freedom_prior}."
                )
            
            expected_shape = self._expected_covar_shape()
            
            # Handle broadcasting for non-tied covariance types
            if self.covariance_type == 'full':
                if covariance_prior.shape == (self.n_features, self.n_features):
                    # Broadcast single matrix to all components
                    covariance_prior = covariance_prior.unsqueeze(0).expand(
                        self.n_components, self.n_features, self.n_features
                    ).clone()
                elif covariance_prior.shape != expected_shape:
                    raise ValueError(
                        f"covariance_prior must be of shape {expected_shape} or "
                        f"({self.n_features}, {self.n_features}) for 'full' covariance. "
                        f"Got {covariance_prior.shape}."
                    )
            
            elif self.covariance_type == 'diag':
                if covariance_prior.shape == (self.n_features,):
                    # Broadcast single vector to all components
                    covariance_prior = covariance_prior.unsqueeze(0).expand(
                        self.n_components, self.n_features
                    ).clone()
                elif covariance_prior.shape != expected_shape:
                    raise ValueError(
                        f"covariance_prior must be of shape {expected_shape} or "
                        f"({self.n_features},) for 'diag' covariance. "
                        f"Got {covariance_prior.shape}."
                    )
            
            elif self.covariance_type == 'spherical':
                if covariance_prior.ndim == 0:
                    # Broadcast scalar to all components
                    covariance_prior = covariance_prior.expand(self.n_components).clone()
                elif covariance_prior.shape != expected_shape:
                    raise ValueError(
                        f"covariance_prior must be of shape {expected_shape} or a scalar "
                        f"for 'spherical' covariance. Got {covariance_prior.shape}."
                    )
            
            else:
                # Tied covariance types: no broadcasting (single shared covariance)
                if covariance_prior.shape != expected_shape:
                    raise ValueError(
                        f"covariance_prior must be of shape {expected_shape} for "
                        f"'{self.covariance_type}' covariance. Got {covariance_prior.shape}."
                    )
            
            self.covariance_prior = covariance_prior.to(device=self.device, dtype=self.dtype)
        else:
            self.degrees_of_freedom_prior = None
            self.covariance_prior = None

        # ---------------------------------------------------------------
        # Validate and report NIW prior usage
        # ---------------------------------------------------------------
        if self.use_mean_prior and self.use_covariance_prior:
            # Normal-Inverse-Wishart conjugate priors for joint estimation
            if self.verbose:
                print("INFO: Using Normal-Inverse-Wishart (NIW) conjugate priors for joint mean-covariance estimation.")
        elif self.use_mean_prior:
            if self.verbose:
                print("INFO: Using Gaussian prior for means only.")
        elif self.use_covariance_prior:
            if self.verbose:
                print("INFO: Using Inverse-Wishart prior for covariances only.")

    def _expected_covar_shape(self) -> Tuple:
        r"""
        Return the expected shape of covariances_ given self.covariance_type.

        Returns
        -------
        shape : Tuple
            The shape that self.covariances_ should have for the specified
            covariance type.
        """
        if self.covariance_type == 'full':
            return (self.n_components, self.n_features, self.n_features)
        elif self.covariance_type == 'diag':
            return (self.n_components, self.n_features)
        elif self.covariance_type == 'spherical':
            return (self.n_components,)
        elif self.covariance_type == 'tied_full':
            return (self.n_features, self.n_features)
        elif self.covariance_type == 'tied_diag':
            return (self.n_features,)
        elif self.covariance_type == 'tied_spherical':
            return ()  # Single scalar for entire dataset
        elif self.covariance_type_ in _NEW_GPCM_CODES:
            # Always stored assembled as full (K, d, d) matrices, regardless
            # of whether the code is axis-aligned -- see
            # docs/superpowers/specs/2026-10-14-phase3-gpcm-covariance-models-design.md.
            return (self.n_components, self.n_features, self.n_features)
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def _allocate_parameters(self, X: Optional[torch.Tensor] = None, set_random_state: bool = True):
        r"""
        Allocate and initialize model parameters (means, weights, covariances).

        Parameters are initialized in a specific order because some initialization
        methods depend on others (e.g., kmeans weight initialization needs means first).

        Parameters
        ----------
        X : torch.Tensor, optional
            Input data for data-based initialization methods. If None, uses random initialization.
        set_random_state : bool, optional
            Whether to set the random state. Set to False when random state is already
            set externally (e.g., for multiple initializations in n_init > 1). (default: True)

        Notes
        -----
        Initialization order:
        1. Means (required by some weight and covariance methods)
        2. Weights (may depend on means for kmeans method)
        3. Covariances (may depend on means for empirical method)
        """
        # ===============================================================
        # Set random seed if requested
        # ===============================================================
        if set_random_state and self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        # ===============================================================
        # 1. Initialize means
        # ===============================================================
        if isinstance(self.init_means, torch.Tensor):
            # User-provided tensor
            if self.init_means.shape != (self.n_components, self.n_features):
                raise ValueError(
                    f"init_means tensor must be shape ({self.n_components}, {self.n_features}), "
                    f"got {self.init_means.shape}."
                )
            self.means_ = self.init_means.to(device=self.device, dtype=self.dtype)

        elif isinstance(self.init_means, str):
            # Initialization method
            if X is None:
                # No data: fallback to random
                self.means_ = torch.randn(
                    self.n_components,
                    self.n_features,
                    device=self.device,
                    dtype=self.dtype
                )
            else:
                # Data-based initialization
                X_cpu = X.cpu()
                init_method = self.init_means.lower()
                
                if init_method == 'kmeans':
                    self.means_ = GMMInitializer.kmeans(X_cpu, self.n_components).to(self.device)
                elif init_method == 'kpp':
                    self.means_ = GMMInitializer.kpp(X_cpu, self.n_components).to(self.device)
                elif init_method == 'points':
                    self.means_ = GMMInitializer.points(X_cpu, self.n_components).to(self.device)
                elif init_method == 'maxdist':
                    self.means_ = GMMInitializer.maxdist(X_cpu, self.n_components).to(self.device)
                elif init_method == 'random':
                    self.means_ = GMMInitializer.random(X_cpu, self.n_components).to(self.device)
                else:
                    raise ValueError(
                        f"Unsupported init_means method: '{init_method}'. "
                        f"Supported: 'kmeans', 'kpp', 'points', 'maxdist', 'random'."
                    )
        else:
            raise TypeError(
                f"init_means must be a string (method name) or torch.Tensor, got {type(self.init_means)}"
            )

        self.initial_means_ = self.means_.clone().detach()

        # ===============================================================
        # 2. Initialize weights
        # ===============================================================
        if isinstance(self.init_weights, torch.Tensor):
            # User-provided tensor
            if self.init_weights.shape != (self.n_components,):
                raise ValueError(
                    f"init_weights tensor must be shape ({self.n_components},), "
                    f"got {self.init_weights.shape}."
                )
            weights = self.init_weights.to(device=self.device, dtype=self.dtype)
            if torch.sum(weights) < 1e-20:
                raise ValueError("Initial weights must sum to > 0.")
            self.weights_ = weights / torch.sum(weights)
        elif isinstance(self.init_weights, str):
            # Use initialization method from GMMInitializer
            init_method = self.init_weights.lower()

            if init_method in ('uniform', 'equal'):
                self.weights_ = GMMInitializer.init_weights_uniform(
                    self.n_components, self.device, dtype=self.dtype
                )
            elif init_method == 'random':
                self.weights_ = GMMInitializer.init_weights_random(
                    self.n_components, self.device, dtype=self.dtype
                )
            elif init_method == 'kmeans':
                if X is None:
                    warnings.warn(
                        "'kmeans' weight initialization requires data. "
                        "Falling back to uniform weights.",
                        UserWarning
                    )
                    self.weights_ = GMMInitializer.init_weights_uniform(
                        self.n_components, self.device, dtype=self.dtype
                    )
                else:
                    # Ensure data is on the same device as means for init_weights_from_clusters
                    X_for_weights = X.to(self.means_.device)
                    weights = GMMInitializer.init_weights_from_clusters(
                        X_for_weights, self.means_
                    )
                    self.weights_ = weights.to(self.device)
            else:
                raise ValueError(
                    f"Unsupported init_weights method: '{init_method}'. "
                    f"Supported methods: 'uniform', 'random', 'kmeans'."
                )
        else:
            raise TypeError(
                f"init_weights must be a string (method name) or torch.Tensor, got {type(self.init_weights)}"
            )

        # Store the initial weights for later access
        self.initial_weights_ = self.weights_.clone().detach()

        # ----------------------
        # 3) Allocate covariances
        # ----------------------
        if isinstance(self.init_covariances, torch.Tensor):
            # User provided explicit tensor
            expected_shape = self._expected_covar_shape()
            init_cov = self.init_covariances
            
            # Allow broadcasting: if user provides a single covariance matrix for non-tied types,
            # expand it to all components
            if self.covariance_type == 'full':
                # Can be (n_features, n_features) or (n_components, n_features, n_features)
                if init_cov.shape == (self.n_features, self.n_features):
                    # Broadcast to all components
                    init_cov = init_cov.unsqueeze(0).expand(
                        self.n_components, self.n_features, self.n_features
                    ).clone()
                elif init_cov.shape != expected_shape:
                    raise ValueError(
                        f"init_covariances tensor must be of shape {expected_shape} or "
                        f"({self.n_features}, {self.n_features}) for 'full' covariance type. "
                        f"Got {init_cov.shape}."
                    )
            elif self.covariance_type == 'diag':
                # Can be (n_features,) or (n_components, n_features)
                if init_cov.shape == (self.n_features,):
                    # Broadcast to all components
                    init_cov = init_cov.unsqueeze(0).expand(
                        self.n_components, self.n_features
                    ).clone()
                elif init_cov.shape != expected_shape:
                    raise ValueError(
                        f"init_covariances tensor must be of shape {expected_shape} or "
                        f"({self.n_features},) for 'diag' covariance type. "
                        f"Got {init_cov.shape}."
                    )
            elif self.covariance_type == 'spherical':
                # Can be a scalar or (n_components,)
                if init_cov.ndim == 0:
                    # Broadcast scalar to all components
                    init_cov = init_cov.expand(self.n_components).clone()
                elif init_cov.shape != expected_shape:
                    raise ValueError(
                        f"init_covariances tensor must be of shape {expected_shape} or a scalar "
                        f"for 'spherical' covariance type. Got {init_cov.shape}."
                    )
            elif self.covariance_type_ in _NEW_GPCM_CODES:
                # Same broadcasting convention as 'full': a single
                # (n_features, n_features) matrix broadcasts to all
                # components.
                if init_cov.shape == (self.n_features, self.n_features):
                    init_cov = init_cov.unsqueeze(0).expand(
                        self.n_components, self.n_features, self.n_features
                    ).clone()
                elif init_cov.shape != expected_shape:
                    raise ValueError(
                        f"init_covariances tensor must be of shape {expected_shape} or "
                        f"({self.n_features}, {self.n_features}) for "
                        f"'{self.covariance_type_}' covariance type. Got {init_cov.shape}."
                    )
            else:
                # For tied types, no broadcasting needed
                if init_cov.shape != expected_shape:
                    raise ValueError(
                        f"init_covariances tensor must be of shape {expected_shape} for "
                        f"'{self.covariance_type}' covariance type. Got {init_cov.shape}."
                    )

            self.covariances_ = init_cov.to(device=self.device, dtype=self.dtype)
        elif isinstance(self.init_covariances, str):
            # Use initialization method from GMMInitializer
            init_method = self.init_covariances.lower()

            if self.covariance_type_ in _NEW_GPCM_CODES:
                # 'eye'/'random' don't need a GPCM projection (there's
                # nothing fit yet to project) -- just build plain full
                # matrices directly, same as the 'full' type; volume_/
                # shape_/orientation_ populate from the first real M-step.
                # 'empirical'/'global' do need the projection (the plan's
                # own instruction): compute per-component scatter from a
                # hard nearest-mean assignment ('empirical') or from the
                # whole dataset around its own mean ('global'), then one
                # mstep_gpcm call.
                if init_method == 'eye':
                    self.covariances_ = GMMInitializer.init_covariances_eye(
                        self.n_components, self.n_features,
                        'full', self.reg_covar, self.device, dtype=self.dtype
                    )
                elif init_method == 'random':
                    self.covariances_ = GMMInitializer.init_covariances_random(
                        self.n_components, self.n_features,
                        'full', self.reg_covar, self.device, dtype=self.dtype
                    )
                elif init_method == 'global':
                    if X is None:
                        warnings.warn(
                            "Global covariance initialization requires data. "
                            "Falling back to identity initialization.",
                            UserWarning
                        )
                        self.covariances_ = GMMInitializer.init_covariances_eye(
                            self.n_components, self.n_features,
                            'full', self.reg_covar, self.device, dtype=self.dtype
                        )
                    else:
                        self.covariances_ = self._init_covariances_gpcm_global(X.to(self.device))
                elif init_method == 'empirical':
                    if X is None:
                        warnings.warn(
                            "Empirical covariance initialization requires data. "
                            "Falling back to identity initialization.",
                            UserWarning
                        )
                        self.covariances_ = GMMInitializer.init_covariances_eye(
                            self.n_components, self.n_features,
                            'full', self.reg_covar, self.device, dtype=self.dtype
                        )
                    else:
                        self.covariances_ = self._init_covariances_gpcm_empirical(X.to(self.means_.device))
                else:
                    raise ValueError(
                        f"Unsupported init_covariances method: '{init_method}'. "
                        f"Supported methods: 'eye', 'random', 'global', 'empirical'."
                    )
            elif init_method == 'eye':
                self.covariances_ = GMMInitializer.init_covariances_eye(
                    self.n_components, self.n_features,
                    self.covariance_type, self.reg_covar, self.device, dtype=self.dtype
                )
            elif init_method == 'random':
                self.covariances_ = GMMInitializer.init_covariances_random(
                    self.n_components, self.n_features,
                    self.covariance_type, self.reg_covar, self.device, dtype=self.dtype
                )
            elif init_method == 'global':
                if X is None:
                    warnings.warn(
                        "Global covariance initialization requires data. "
                        "Falling back to identity initialization.",
                        UserWarning
                    )
                    self.covariances_ = GMMInitializer.init_covariances_eye(
                        self.n_components, self.n_features,
                        self.covariance_type, self.reg_covar, self.device
                    )
                else:
                    # Ensure data is on the correct device
                    X_for_cov = X.to(self.device)
                    covs = GMMInitializer.init_covariances_global(
                        X_for_cov, self.n_components,
                        self.covariance_type, self.reg_covar
                    )
                    self.covariances_ = covs.to(self.device)
            elif init_method == 'empirical':
                if X is None:
                    warnings.warn(
                        "Empirical covariance initialization requires data. "
                        "Falling back to identity initialization.",
                        UserWarning
                    )
                    self.covariances_ = GMMInitializer.init_covariances_eye(
                        self.n_components, self.n_features,
                        self.covariance_type, self.reg_covar, self.device
                    )
                else:
                    # Ensure data and means are on the same device
                    X_for_cov = X.to(self.means_.device)
                    covs = GMMInitializer.init_covariances_empirical(
                        X_for_cov, self.means_,
                        self.covariance_type, self.reg_covar
                    )
                    self.covariances_ = covs.to(self.device)
            else:
                raise ValueError(
                    f"Unsupported init_covariances method: '{init_method}'. "
                    f"Supported methods: 'eye', 'random', 'global', 'empirical'."
                )
        else:
            raise TypeError(
                f"init_covariances must be a string (method name) or torch.Tensor, "
                f"got {type(self.init_covariances)}"
            )

        # Store the initial covariances for later access
        self.initial_covariances_ = self.covariances_.clone().detach()

        # A fresh allocation starts a fresh GPCM warm-start state (stale
        # state from a previous fit() call would warm-start the iterative
        # codes' inner loop from an unrelated decomposition).
        self._gpcm_state_ = None

        # Mark that we've allocated
        self.fitted_ = False
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")

    def _init_covariances_gpcm_empirical(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        `init_covariances='empirical'` for the 8 new GPCM codes: assign
        each point to its nearest current mean (hard, k-means-style),
        build the per-component scatter from that assignment, and project
        it onto the requested GPCM parameterization with one `mstep_gpcm`
        call -- the plan's own instruction (projecting the *constrained*
        MLE of that starting point, not the raw unconstrained empirical
        covariance, which wouldn't satisfy the code's constraints).
        """
        distances = torch.cdist(X, self.means_)
        labels = torch.argmin(distances, dim=1)
        nk = torch.zeros(self.n_components, dtype=self.dtype, device=self.device)
        W = torch.zeros(self.n_components, self.n_features, self.n_features, dtype=self.dtype, device=self.device)
        for k in range(self.n_components):
            mask = labels == k
            count = int(mask.sum().item())
            if count > 0:
                diff = X[mask] - self.means_[k]
                W[k] = diff.t() @ diff
                nk[k] = count
            else:
                W[k] = torch.eye(self.n_features, dtype=self.dtype, device=self.device)
                nk[k] = 1.0

        result = mstep_gpcm(
            self.covariance_type_, W, nk, self.n_features,
            reg_covar=self.reg_covar, gpcm_tol=self.gpcm_tol, gpcm_max_iter=self.gpcm_max_iter,
        )
        self.volume_, self.shape_, self.orientation_ = result.volume, result.shape, result.orientation
        self._gpcm_state_ = result
        return result.cov_full

    def _init_covariances_gpcm_global(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        `init_covariances='global'` for the 8 new GPCM codes: the whole
        dataset's own scatter (around its overall mean, not per-component),
        projected onto the requested GPCM parameterization once, then
        broadcast to every component as a starting point.
        """
        overall_mean = X.mean(dim=0)
        diff = X - overall_mean
        W_global = diff.t() @ diff
        n = X.shape[0]
        nk = torch.full((self.n_components,), float(n), dtype=self.dtype, device=self.device)
        W = W_global.unsqueeze(0).expand(self.n_components, self.n_features, self.n_features).clone()

        result = mstep_gpcm(
            self.covariance_type_, W, nk, self.n_features,
            reg_covar=self.reg_covar, gpcm_tol=self.gpcm_tol, gpcm_max_iter=self.gpcm_max_iter,
        )
        self.volume_, self.shape_, self.orientation_ = result.volume, result.shape, result.orientation
        self._gpcm_state_ = result
        return result.cov_full


    def fit(
        self,
        X: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        random_state: Optional[int] = None,
        warm_start: Optional[bool] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> "GaussianMixture":
        r"""
        Fit the GMM to the data using the Expectation-Maximization algorithm.

        Supports multiple random initializations (n_init > 1) to find the best solution.
        The model with the highest log-likelihood is selected.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features).
        labels : torch.Tensor or array-like, optional
            Per-sample ground-truth class labels, shape (n_samples,). If
            given, `fit` performs **supervised fitting**: one Gaussian
            component is fit per distinct label directly from that label's
            data (the classification-likelihood limit of CEM where the
            classification step uses the true label instead of
            ``argmax(resp)`` — see the "Supervised Fitting" section of the
            user guide). Requires ``n_components`` to equal the number of
            distinct values in `labels`. Label values don't need to be
            contiguous or zero-indexed (see `classes_`). Bypasses the
            EM/CEM loop and `n_init` entirely: the fit is a single M-step
            and is deterministic, since the assignment never depends on the
            current parameter estimates. `warm_start` is ignored as well,
            for the same reason. `converged_` is always ``True`` and
            `n_iter_` is always ``1`` after a supervised fit. (default: None)
        max_iter : int, optional
            Maximum number of EM iterations. Overrides `self.max_iter` if provided. (default: None)
        tol : float, optional
            Convergence tolerance. Overrides `self.tol` if provided. (default: None)
        random_state : int, optional
            Random seed. Overrides `self.random_state` if provided. (default: None)
        warm_start : bool, optional
            Whether to warm-start from previously fitted parameters. Overrides `self.warm_start`
            if provided. (default: None)
        sample_weight : torch.Tensor, optional
            Row weights `(n_samples,)` (integer weights equal duplicating rows, weight 0 drops a row) or,
            for `covariance_type='diag'`/`'spherical'` without priors, per-feature weights
            `(n_samples, n_features)` (weight 0 on an entry hides that feature for that row). Not supported
            with `labels`, Gibbs mode or online EM. See `BaseMixture.fit`. (default: None)

        Returns
        -------
        self : GaussianMixture
            The fitted model instance (allows method chaining).

        Raises
        ------
        ValueError
            If n_components > n_samples, or if parameters are invalid.
            If `labels` is given and its length doesn't match `X`, or its
            number of distinct values doesn't equal `n_components`.
        """
        # ===============================================================
        # 0. Gibbs-sampling mode (n_components=None): dispatch and return
        # early, before any of the EM-specific validation below runs (it
        # assumes self.n_components is an int).
        # ===============================================================
        if self.n_components is None:
            if sample_weight is not None:
                raise NotImplementedError("sample_weight is not supported in Gibbs mode (n_components=None).")
            if labels is not None:
                raise ValueError(
                    "Supervised fitting (labels=...) requires a fixed n_components; "
                    "got n_components=None (Gibbs sampling infers the component count "
                    "itself, so there's no fixed-K target for supervised labels)."
                )
            if max_iter is not None and max_iter <= 0:
                raise ValueError(f"Invalid max_iter: {max_iter} (must be > 0).")
            X = X.to(device=self.device)
            self.dtype = X.dtype
            self.n_features = X.shape[1]
            if random_state is not None:
                self.random_state = random_state
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            original_max_iter = self.max_iter
            if max_iter is not None:
                self.max_iter = max_iter
            try:
                self._fit_gibbs(X)
            finally:
                self.max_iter = original_max_iter
            return self

        # ===============================================================
        # 1. Validate input parameters
        # ===============================================================
        if X.size(0) < self.n_components:
            raise ValueError(
                f"n_components={self.n_components} must be <= n_samples={X.size(0)}."
            )
        if self.n_components <= 0:
            raise ValueError(f"Invalid n_components: {self.n_components} (must be > 0).")
        if tol is not None and tol <= 0:
            raise ValueError(f"Invalid tol: {tol} (must be > 0).")
        if max_iter is not None and max_iter <= 0:
            raise ValueError(f"Invalid max_iter: {max_iter} (must be > 0).")

        # ===============================================================
        # 2. Set up parameters (use overrides or defaults)
        # ===============================================================
        warm_start = self.warm_start if warm_start is None else warm_start
        max_iter = self.max_iter if max_iter is None else max_iter
        tol = self.tol if tol is None else tol
        
        if random_state is not None:
            self.random_state = random_state

        # ===============================================================
        # 3. Prepare data
        # ===============================================================
        X = X.to(self.device)

        # Follow the data's floating dtype (e.g. float64) instead of the
        # float32 default. Any priors already validated in __init__ are
        # recast to match, since they were processed before X was known.
        self.dtype = X.dtype
        if self.weight_concentration_prior is not None:
            self.weight_concentration_prior = self.weight_concentration_prior.to(dtype=self.dtype)
        if self.mean_prior is not None:
            self.mean_prior = self.mean_prior.to(dtype=self.dtype)
        if self.covariance_prior is not None:
            self.covariance_prior = self.covariance_prior.to(dtype=self.dtype)

        # Infer n_features from data if not set
        if self.n_features is None:
            self.n_features = X.shape[1]
        
        # Handle 1D data
        if X.dim() == 1:
            X = X.unsqueeze(1)
        
        # Validate feature dimension
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features}."
            )

        # ===============================================================
        # 3b-5. (Semi-)supervised fitting (labels given) and the ordinary
        # unsupervised n_init loop are both handled generically by
        # BaseMixture now (Phase 5) -- see the design doc. X/max_iter/tol/
        # random_state/warm_start are already fully resolved above, so
        # this re-does X/parameter validation idempotently and nothing
        # about the result changes.
        # ===============================================================
        return super().fit(X, labels=labels, max_iter=max_iter, tol=tol, random_state=random_state,
                           warm_start=warm_start, sample_weight=sample_weight)

    def _is_tied_covariance(self) -> bool:
        return self.covariance_type in ('tied_full', 'tied_diag', 'tied_spherical')

    def _check_frozen_supported(self):
        if self.covariance_type.startswith('tied_'):
            raise ValueError(
                f"frozen_components is not supported with covariance_type={self.covariance_type!r}: the "
                "covariance is shared across components, so a single component cannot be held fixed."
            )
        if self.covariance_type_ in _NEW_GPCM_CODES:
            raise ValueError(
                f"frozen_components is not supported with the GPCM code {self.covariance_type_!r}: it shares "
                "volume, shape or orientation across components, so a single component cannot be held fixed. "
                "Use 'full', 'diag' or 'spherical'."
            )

    def _feature_weights_supported(self) -> bool:
        return self.covariance_type in ('diag', 'spherical')

    def _has_priors(self) -> bool:
        return bool(self.use_weight_prior or self.use_mean_prior or self.use_covariance_prior)

    def _post_supervised_fit_warnings(self, component_counts: torch.Tensor):
        r"""
        Warn if any class has too few samples to reliably estimate a
        per-component `'full'`/`'diag'`/`'spherical'` covariance from that
        class's own data alone (rank-deficient for `'full'`, a zero real
        variance propped up only by `reg_covar` for `'diag'`/`'spherical'`).
        Tied covariance types pool residuals across *all* classes, so a
        single small class doesn't by itself make the shared estimate
        degenerate -- no check for those. A weight-based degeneracy check
        isn't needed either: supervised/semi-supervised weights are
        `n_c / n` with `n_c >= 1` for every class actually present in
        `labels`, so that would need `n > 1e8` to ever fire.
        """
        if self.covariance_type in ('full', 'diag', 'spherical'):
            min_required = self.n_features + 1 if self.covariance_type == 'full' else 2
            sparse_mask = component_counts < min_required
            if torch.any(sparse_mask):
                sparse_classes = self.classes_[self.class_of_component_[sparse_mask]].tolist()
                warnings.warn(
                    f"Component(s) belonging to class(es) {sparse_classes} have too "
                    f"few samples (need > {min_required - 1}) to reliably estimate a "
                    f"'{self.covariance_type}' covariance from that class's own data "
                    "alone. The resulting covariance may be rank-deficient or "
                    "dominated entirely by reg_covar.",
                    UserWarning
                )

    # ===================================================================
    # SEM (algorithm='sem')
    # ===================================================================
    def _reseed_empty_components(
        self, X: torch.Tensor, log_prob_norm: torch.Tensor, empty_mask: torch.Tensor
    ) -> int:
        r"""
        SEM empty-component handling: every component flagged by
        `empty_mask` (`nk < 1` after the S-step) gets its mean reset to
        the data point with the lowest log-likelihood among points not
        already claimed by another re-seed in this same sweep (using the
        single worst point for every emptied component would collide if
        more than one empties at once), and -- for `full`/`diag`/
        `spherical` covariance only, since `tied_*` has nothing
        per-component to reset -- its covariance reset to the whole
        dataset's own covariance.

        Returns
        -------
        int
            Number of components re-seeded.
        """
        empty_indices = torch.nonzero(empty_mask, as_tuple=True)[0]
        n_empty = empty_indices.shape[0]
        worst_order = torch.argsort(log_prob_norm)  # ascending: worst (lowest log-lik) first
        worst_points = worst_order[:n_empty]
        self.means_[empty_indices] = X[worst_points]

        if self.covariance_type in ('full', 'diag', 'spherical'):
            global_cov = self._global_covariance(X)
            for k in empty_indices.tolist():
                self.covariances_[k] = global_cov

        return n_empty

    def _global_covariance(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        The whole dataset's own covariance, shaped to match
        `self.covariance_type` (`'full'`: `(d, d)`; `'diag'`: `(d,)`;
        `'spherical'`: scalar). Used by SEM's empty-component re-seeding;
        not defined for `tied_*` types (nothing per-component to reset).
        """
        if self.covariance_type == 'full':
            cov = torch.cov(X.T)
            return cov + self.reg_covar * torch.eye(self.n_features, device=self.device, dtype=self.dtype)
        elif self.covariance_type == 'diag':
            return X.var(dim=0, unbiased=True) + self.reg_covar
        elif self.covariance_type == 'spherical':
            return X.var(dim=0, unbiased=True).mean() + self.reg_covar
        else:
            raise ValueError(f"_global_covariance is not defined for covariance_type={self.covariance_type!r}")

    # ===================================================================
    # Component log-probability: BaseMixture._e_step_chunk's abstract hook
    # ===================================================================
    def _component_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Per-component Gaussian log-density log p(x|z_k, theta_k), shape
        (n_samples, n_components), dispatching on `self.covariance_type`.
        GPCM covariance types new in Phase 3 always use the full-matrix
        density, since they're stored assembled as full (K, d, d) matrices
        -- see
        docs/superpowers/specs/2026-10-14-phase3-gpcm-covariance-models-design.md.

        This is `BaseMixture._e_step_chunk`'s abstract hook: the mixing-
        weight term and deterministic-annealing tempering are applied
        uniformly there, not here. See `BaseMixture._e_step`/`_e_step_chunk`.
        """
        if self._missing_active(X):
            return self._component_log_prob_missing(X)
        if self.covariance_type == 'full':
            return self._estimate_log_gaussian_full(X)
        elif self.covariance_type == 'diag':
            return self._estimate_log_gaussian_diag(X)
        elif self.covariance_type == 'spherical':
            return self._estimate_log_gaussian_spherical(X)
        elif self.covariance_type == 'tied_full':
            return self._estimate_log_gaussian_tied_full(X)
        elif self.covariance_type == 'tied_diag':
            return self._estimate_log_gaussian_tied_diag(X)
        elif self.covariance_type == 'tied_spherical':
            return self._estimate_log_gaussian_tied_spherical(X)
        elif self.covariance_type_ in _NEW_GPCM_CODES:
            return self._estimate_log_gaussian_full(X)
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    # ===================================================================
    # Log-Gaussian Probability Estimation
    # ===================================================================
    # These methods compute log p(x|z_k, θ_k) for each sample-component pair
    # using the multivariate Gaussian density:
    #
    #   log p(x|μ,Σ) = -½[d·log(2π) + log|Σ| + (x-μ)ᵀΣ⁻¹(x-μ)]
    #
    # All computations use log-space for numerical stability.
    # ===================================================================
    
    def _estimate_log_gaussian_full(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for full covariance matrices.

        Each component has its own full covariance matrix Σₖ ∈ ℝ^(d×d).
        Uses Cholesky decomposition for numerical stability and efficiency.
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
            
        Notes
        -----
        Cholesky decomposition: Σ = LLᵀ where L is lower triangular.
        log|Σ| = 2·sum(log(diag(L)))
        Σ⁻¹(x-μ) solved via triangular solves.
        """
        # ===============================================================
        # Compute deviations from component means
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        
        # ===============================================================
        # Cholesky decomposition: Σ = LLᵀ
        # ===============================================================
        try:
            chol = torch.linalg.cholesky(self.covariances_)  # (n_components, n_features, n_features)
        except RuntimeError as e:
            raise ValueError(f"Cholesky decomposition failed. Covariances may not be positive definite: {e}")

        # ===============================================================
        # Compute log determinant: log|Σ| = 2·sum(log(diag(L)))
        # ===============================================================
        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1)  # (n_components,)
        
        # ===============================================================
        # Compute Mahalanobis distance: (x-μ)ᵀΣ⁻¹(x-μ)
        #
        # diff is reshaped to (n_components, n_features, n_samples) --
        # samples as the right-hand-side *columns* of each component's
        # triangular solve, rather than diff.unsqueeze(-1)'s (n_samples,
        # n_components, n_features, 1), which puts n_samples on its own
        # broadcast batch axis. cholesky_solve/solve_triangular need equal
        # batch shapes between the two operands; broadcasting chol's
        # (n_components, d, d) up against a (n_samples, n_components) batch
        # forces an actual (n_samples, n_components, d, d) materialisation
        # internally (observed directly: ~3.3 GB for n=20000, K=10, d=64 in
        # float32, the same blow-up Phase 0.2 fixed in the M-step). With
        # samples folded into the solve's right-hand side instead, both
        # operands share the same (n_components,) batch shape and no
        # broadcast-driven copy happens -- peak memory back to
        # O(n_samples * n_components * n_features).
        # ===============================================================
        diff_t = diff.permute(1, 2, 0).contiguous()  # (n_components, n_features, n_samples)
        solve = torch.cholesky_solve(diff_t, chol)  # Σ⁻¹(x-μ), same shape
        mahal = (diff_t * solve).sum(dim=1).t()  # (n_samples, n_components)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = math.log(2.0 * math.pi)
        return -0.5 * (self.n_features * log_2pi + log_det.unsqueeze(0) + mahal)

    def _estimate_log_gaussian_diag(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for diagonal covariance matrices.

        Each component has a diagonal covariance: Σₖ = diag(σ²ₖ₁, ..., σ²ₖd).
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
        """
        # ===============================================================
        # Compute deviations and precisions (inverse variances)
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        precisions = 1.0 / (self.covariances_ + 1e-20)  # (n_components, n_features)

        if self._fw_cur is not None:
            # per-feature sample weights (v0.5 item 1.4): feature j of row i contributes with weight w_ij
            fw = self._fw_cur  # (n, d)
            log_var = torch.log(self.covariances_ + 1e-20)  # (K, d)
            mahal_w = torch.einsum('nj,nkj,kj->nk', fw, diff.pow(2), precisions)
            return -0.5 * (fw.sum(dim=1, keepdim=True) * math.log(2.0 * math.pi) + fw @ log_var.t() + mahal_w)

        # ===============================================================
        # Log determinant and Mahalanobis distance
        # ===============================================================
        log_det = torch.sum(torch.log(self.covariances_ + 1e-20), dim=1)  # (n_components,)
        mahal = torch.sum(diff.pow(2) * precisions.unsqueeze(0), dim=2)  # (n_samples, n_components)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = math.log(2.0 * math.pi)
        return -0.5 * (self.n_features * log_2pi + log_det.unsqueeze(0) + mahal)

    def _estimate_log_gaussian_spherical(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for spherical covariance matrices.

        Each component has spherical covariance: Σₖ = σ²ₖ·I.
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
        """
        # ===============================================================
        # Compute deviations and squared distances
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        if self._fw_cur is not None:
            # per-feature sample weights (v0.5 item 1.4)
            fw = self._fw_cur  # (n, d)
            weighted_sq = torch.einsum('nj,nkj->nk', fw, diff.pow(2))
            var = self.covariances_ + 1e-20  # (K,)
            return -0.5 * (fw.sum(dim=1, keepdim=True) * (math.log(2.0 * math.pi) + torch.log(var).unsqueeze(0))
                           + weighted_sq / var.unsqueeze(0))
        sq_dist = torch.sum(diff.pow(2), dim=2)  # (n_samples, n_components)
        
        # ===============================================================
        # Scale by precision (1/σ²)
        # ===============================================================
        precisions = 1.0 / (self.covariances_ + 1e-20)  # (n_components,)
        mahal = sq_dist * precisions.unsqueeze(0)
        
        # ===============================================================
        # Log determinant: d·log(σ²)
        # ===============================================================
        log_det = self.n_features * torch.log(self.covariances_ + 1e-20)  # (n_components,)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = math.log(2.0 * math.pi)
        return -0.5 * (self.n_features * log_2pi + log_det.unsqueeze(0) + mahal)

    def _estimate_log_gaussian_tied_full(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for tied full covariance.

        All components share the same full covariance matrix Σ ∈ ℝ^(d×d).
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
        """
        # ===============================================================
        # Compute deviations from component means
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        
        # ===============================================================
        # Cholesky decomposition of shared covariance
        # ===============================================================
        try:
            chol = torch.linalg.cholesky(self.covariances_)  # (n_features, n_features)
        except RuntimeError as e:
            raise ValueError(f"Tied full: Cholesky decomposition failed: {e}")

        # ===============================================================
        # Log determinant (same for all components)
        # ===============================================================
        log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()  # scalar
        
        # ===============================================================
        # Mahalanobis distance for all samples and components. `chol` has
        # no batch dimension here (one shared covariance), so every
        # (sample, component) deviation can be solved as a single system
        # with (n_samples * n_components) right-hand-side columns -- no
        # broadcasting (and hence no implicit (n_samples, n_components, d,
        # d) materialisation, see `_estimate_log_gaussian_full`) needed at
        # all.
        # ===============================================================
        diff_flat = diff.reshape(-1, self.n_features).t()  # (n_features, n_samples * n_components)
        solve = torch.cholesky_solve(diff_flat, chol)
        mahal = (diff_flat * solve).sum(dim=0).view(diff.shape[0], diff.shape[1])  # (n_samples, n_components)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = math.log(2.0 * math.pi)
        return -0.5 * (self.n_features * log_2pi + log_det + mahal)

    def _estimate_log_gaussian_tied_diag(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for tied diagonal covariance.

        All components share the same diagonal covariance: Σ = diag(σ²₁, ..., σ²d).
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
        """
        # ===============================================================
        # Compute deviations
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        
        # ===============================================================
        # Shared diagonal covariance
        # ===============================================================
        cov_vector = self.covariances_ + 1e-20  # (n_features,)
        log_det = torch.sum(torch.log(cov_vector))  # scalar
        precisions = 1.0 / cov_vector  # (n_features,)
        
        # ===============================================================
        # Mahalanobis distance
        # ===============================================================
        mahal = torch.sum(diff.pow(2) * precisions, dim=2)  # (n_samples, n_components)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = math.log(2.0 * math.pi)
        return -0.5 * (self.n_features * log_2pi + log_det + mahal)

    def _estimate_log_gaussian_tied_spherical(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for tied spherical covariance.

        All components share the same spherical covariance: Σ = σ²·I.
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
        """
        # ===============================================================
        # Compute deviations and squared distances
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        sq_dist = torch.sum(diff.pow(2), dim=2)  # (n_samples, n_components)
        
        # ===============================================================
        # Shared spherical variance
        # ===============================================================
        var = self.covariances_ + 1e-20  # scalar
        prec = 1.0 / var
        mahal = sq_dist * prec
        log_det = self.n_features * torch.log(var)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = math.log(2.0 * math.pi)
        return -0.5 * (self.n_features * log_2pi + log_det + mahal)

    # ---------------------------
    # M-step
    # ---------------------------
    # ===================================================================
    # M-step: Maximization
    # ===================================================================
    def _second_moment(self, resp: torch.Tensor, diff: torch.Tensor, base_type: str) -> torch.Tensor:
        r"""
        Covariance-type-specific second moment
        :math:`\sum_i r_{ik} \mathrm{diff}_{ik} \mathrm{diff}_{ik}^T`
        (or its diag/spherical contraction), computed without ever
        materialising an ``(n, K, d, d)`` tensor -- see
        docs/superpowers/specs/2026-10-14-phase0-sufficient-stats-design.md
        for why the pre-refactor elementwise-broadcast version of this did.

        Parameters
        ----------
        resp : torch.Tensor, shape (n, K)
        diff : torch.Tensor, shape (n, K, d)
            Deviations from whatever center point the caller used.
        base_type : str
            One of 'full', 'diag', 'spherical' (tied_* types pass their
            non-tied base type; pooling across K happens in the caller).
        """
        if base_type == 'full':
            return torch.einsum('nk,nkd,nke->kde', resp, diff, diff)
        elif base_type == 'diag':
            return torch.einsum('nk,nkd->kd', resp, diff.pow(2))
        elif base_type == 'spherical':
            return torch.einsum('nk,nk->k', resp, diff.pow(2).sum(dim=2))
        else:
            raise ValueError(f"Unsupported covariance type: {base_type}")

    def _compute_stats_feature_weighted(
        self, X: torch.Tensor, resp: torch.Tensor, batch_size: Optional[int], W: torch.Tensor,
    ) -> SufficientStats:
        r"""
        Per-feature weighted statistics for `'diag'`/`'spherical'` (v0.5 item 1.4): with entry weights
        :math:`w_{ij}`, ``s1`` is :math:`\sum_i r_{ik} w_{ij} x_{ij}`, ``s2`` the correspondingly weighted scatter
        about :math:`\bar y_{kj} = s1_{kj}/N_{kj}` (a per-component scalar sum for `'spherical'`), and
        ``aux['nk_feat']`` holds :math:`N_{kj} = \sum_i r_{ik} w_{ij}`.
        """
        n_samples, n_features = X.shape
        n_components = resp.shape[1]
        spherical = self.covariance_type == 'spherical'
        chunks = [(0, n_samples)] if (batch_size is None or batch_size >= n_samples) else [
            (a, min(a + batch_size, n_samples)) for a in range(0, n_samples, batch_size)]
        nk = torch.zeros(n_components, device=X.device, dtype=X.dtype)
        nkf = torch.zeros(n_components, n_features, device=X.device, dtype=X.dtype)
        s1 = torch.zeros(n_components, n_features, device=X.device, dtype=X.dtype)
        for a, b in chunks:
            nk += resp[a:b].sum(dim=0)
            nkf += resp[a:b].t() @ W[a:b]
            s1 += resp[a:b].t() @ (W[a:b] * X[a:b])
        ybar = s1 / (nkf + 1e-20)
        s2 = None
        for a, b in chunks:
            diff2 = (X[a:b].unsqueeze(1) - ybar.unsqueeze(0)).pow(2)
            part = torch.einsum('nk,nj,nkj->k' if spherical else 'nk,nj,nkj->kj', resp[a:b], W[a:b], diff2)
            s2 = part if s2 is None else s2 + part
        return SufficientStats(nk=nk, s1=s1, s2=s2, n_samples=n_samples, aux={'nk_feat': nkf})

    def _compute_stats(self, X: torch.Tensor, resp: torch.Tensor, batch_size: Optional[int] = None,
                       feature_weights: Optional[torch.Tensor] = None) -> SufficientStats:
        r"""
        Build a :class:`~tgmm.stats.SufficientStats` from ``(X, resp)``.

        ``s2`` is centered on each component's own empirical mean for this
        E-step, :math:`\bar y_k = s1_k / n_k` -- not on ``self.means_`` --
        which is what makes chunked accumulation exact (see the design doc).
        When ``batch_size`` is set, this runs as two passes over row chunks
        (first ``nk``/``s1``, then, once :math:`\bar y_k` is known, ``s2``)
        so peak memory stays ``O(batch_size * K * d)`` regardless of
        ``n_samples``.

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
        resp : torch.Tensor, shape (n_samples, n_components)
        batch_size : int, optional
            Row-chunk size. ``None`` (default) processes all of ``X`` in
            one shot.

        Returns
        -------
        SufficientStats
        """
        if feature_weights is not None:
            return self._compute_stats_feature_weighted(X, resp, batch_size, feature_weights)
        if self._missing_active(X):
            return self._compute_stats_missing(X, resp, batch_size)
        n_samples, n_features = X.shape
        # GPCM covariance types (Phase 3) always store/need a full (K, d, d)
        # second moment, regardless of whether the parameterization itself
        # is axis-aligned -- see
        # docs/superpowers/specs/2026-10-14-phase3-gpcm-covariance-models-design.md.
        base_type = 'full' if self.covariance_type_ in _NEW_GPCM_CODES \
            else self.covariance_type.replace('tied_', '', 1)

        if batch_size is None or batch_size >= n_samples:
            nk = resp.sum(dim=0)
            s1 = resp.t() @ X
            ybar = s1 / (nk + 1e-20).unsqueeze(1)
            diff = X.unsqueeze(1) - ybar.unsqueeze(0)
            s2 = self._second_moment(resp, diff, base_type)
            return SufficientStats(nk=nk, s1=s1, s2=s2, n_samples=n_samples)

        n_components = resp.shape[1]
        nk = torch.zeros(n_components, device=X.device, dtype=X.dtype)
        s1 = torch.zeros(n_components, n_features, device=X.device, dtype=X.dtype)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            nk += resp[start:end].sum(dim=0)
            s1 += resp[start:end].t() @ X[start:end]

        ybar = s1 / (nk + 1e-20).unsqueeze(1)
        s2 = None
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            diff_chunk = X[start:end].unsqueeze(1) - ybar.unsqueeze(0)
            chunk_s2 = self._second_moment(resp[start:end], diff_chunk, base_type)
            s2 = chunk_s2 if s2 is None else s2 + chunk_s2

        return SufficientStats(nk=nk, s1=s1, s2=s2, n_samples=n_samples)

    # ===================================================================
    # Missing values (v0.5 item 1.2): see tgmm/mixtures/missing.py and
    # docs/superpowers/specs/2026-10-15-v05-missing-values-design.md.
    # ===================================================================
    _supports_missing = True

    def _missing_cov_repr(self):
        r"""`(kind, cov)` for `missing.gaussian_blocks`: dense `(K, d, d)` or per-feature variances `(K, d)`."""
        K, d = self.n_components, self.n_features
        ctype = self.covariance_type
        if ctype == 'full' or self.covariance_type_ in _NEW_GPCM_CODES:
            return 'dense', self.covariances_
        if ctype == 'tied_full':
            return 'dense', self.covariances_.unsqueeze(0).expand(K, d, d)
        if ctype == 'diag':
            return 'diag', self.covariances_
        if ctype == 'spherical':
            return 'diag', self.covariances_.unsqueeze(1).expand(K, d)
        if ctype == 'tied_diag':
            return 'diag', self.covariances_.unsqueeze(0).expand(K, d)
        if ctype == 'tied_spherical':
            return 'diag', self.covariances_.expand(K, d)
        raise ValueError(f"Unsupported covariance type: {ctype}")

    def _component_log_prob_missing(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Marginal log-density of the *observed* entries of each row under each component,
        :math:`\log N(x_o;\mu_o,\Sigma_{oo}) = -\tfrac12[p\log 2\pi + \log|\Sigma_{oo}| + \delta_o]`,
        shape `(n_samples, n_components)`.
        """
        kind, cov = self._missing_cov_repr()
        out = torch.empty(X.shape[0], self.means_.shape[0], dtype=X.dtype, device=X.device)
        log_2pi = math.log(2.0 * math.pi)
        for b in _missing.gaussian_blocks(X, self.means_, cov, kind, need_conditional=False):
            p = b.obs.numel()
            out[b.rows] = -0.5 * (p * log_2pi + b.logdet.unsqueeze(0) + b.delta)
        return out

    def _compute_stats_missing(self, X: torch.Tensor, resp: torch.Tensor, batch_size: Optional[int] = None) -> SufficientStats:
        r"""
        Expected complete-data statistics given the observed entries, at the *current* parameters
        (those the E-step just used): :math:`n_k=\sum_i r_{ik}`, :math:`s1_k=\sum_i r_{ik}\tilde x_{ik}` and

        .. math::
            s2_k = \sum_i r_{ik}(\tilde x_{ik}-\bar y_k)(\tilde x_{ik}-\bar y_k)^\top
                   + \sum_i r_{ik} C_{k,\mathrm{pattern}(i)}

        with :math:`\tilde x_{ik}` the completed row (conditional mean in the missing block) and `C` the
        conditional covariance placed in the missing block, reduced to the diag / spherical form the
        covariance type's M-step expects. Chunked over rows like the complete-data version.
        """
        n_samples, n_features = X.shape
        n_components = resp.shape[1]
        kind, cov = self._missing_cov_repr()
        base_type = 'full' if self.covariance_type_ in _NEW_GPCM_CODES \
            else self.covariance_type.replace('tied_', '', 1)
        step = n_samples if (batch_size is None or batch_size >= n_samples) else batch_size

        def chunk(start):
            end = min(start + step, n_samples)
            Xc, rc = X[start:end], resp[start:end]
            blocks = _missing.gaussian_blocks(Xc, self.means_, cov, kind, need_conditional=True)
            return Xc, rc, blocks, _missing.complete_rows(Xc, blocks, n_components)

        nk = torch.zeros(n_components, device=X.device, dtype=X.dtype)
        s1 = torch.zeros(n_components, n_features, device=X.device, dtype=X.dtype)
        for start in range(0, n_samples, step):
            _, rc, _, Xhat = chunk(start)
            nk += rc.sum(dim=0)
            s1 += torch.einsum('nk,nkd->kd', rc, Xhat)
        ybar = s1 / (nk + 1e-20).unsqueeze(1)

        s2 = None
        for start in range(0, n_samples, step):
            Xc, rc, blocks, Xhat = chunk(start)
            part = self._second_moment(rc, Xhat - ybar.unsqueeze(0), base_type)
            extra = _missing.conditional_cov_sum(blocks, rc, kind, n_features)
            if base_type == 'spherical':
                extra = extra.sum(dim=-1)
            part = part + extra
            s2 = part if s2 is None else s2 + part
        return SufficientStats(nk=nk, s1=s1, s2=s2, n_samples=n_samples)

    def _scatter_about_means(self, stats: SufficientStats, nk: torch.Tensor, ybar: torch.Tensor) -> torch.Tensor:
        r"""
        :math:`\sum_i r_{ik} (x_i - \mu_k)(x_i - \mu_k)^T` (or its
        diag/spherical form) with :math:`\mu_k` = ``self.means_``,
        reconstructed from ``stats.s2`` (centered on
        :math:`\bar y_k` = ``ybar``, not necessarily ``self.means_``) via
        the parallel-axis identity

        .. math::
            \sum_i r_{ik} (x_i - \mu_k)(x_i - \mu_k)^T
                = s2_k + n_k (\bar y_k - \mu_k)(\bar y_k - \mu_k)^T

        (the cross term vanishes because
        :math:`\sum_i r_{ik}(x_i - \bar y_k) = 0` by definition of
        :math:`\bar y_k`). Used by the MLE and independent-MAP covariance
        updates, both of which need the scatter centered on whatever
        ``self.means_`` currently holds (which equals ``ybar`` only when
        there is no mean prior). The NIW branch needs no such correction --
        its scatter term is defined directly in terms of ``ybar``, so it
        uses ``stats.s2`` as-is.

        Returns a per-component result (``full``/``diag``/``spherical``
        shape matching ``stats.s2``); callers for ``tied_*`` types pool it
        across components themselves.
        """
        base_type = self.covariance_type.replace('tied_', '', 1)
        delta = ybar - self.means_  # (K, d)
        if base_type == 'full':
            mismatch = nk.view(-1, 1, 1) * torch.einsum('kd,ke->kde', delta, delta)
        elif base_type == 'diag':
            mismatch = nk.unsqueeze(-1) * delta.pow(2)
        elif base_type == 'spherical':
            mismatch = nk * delta.pow(2).sum(dim=1)
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")
        return stats.s2 + mismatch

    def _m_step_feature_weighted(self, stats: SufficientStats):
        r"""
        Maximum-likelihood M-step with per-feature sample weights (v0.5 item 1.4, `'diag'`/`'spherical'`):
        :math:`\pi_k = \sum_i r_{ik}/n`, :math:`\mu_{kj} = s1_{kj}/N_{kj}`, and
        :math:`\sigma^2_{kj} = s2_{kj}/N_{kj}` (`'diag'`) or :math:`\sigma^2_k = s2_k/\sum_j N_{kj}` (`'spherical'`),
        plus `reg_covar`.
        """
        nk = stats.nk + 1e-20
        nkf = stats.aux['nk_feat'] + 1e-20
        self.weights_ = (nk / stats.n_samples).clamp(min=1e-20)
        self.means_ = stats.s1 / nkf
        if self.covariance_type == 'diag':
            self.covariances_ = stats.s2 / nkf + self.reg_covar
        else:
            self.covariances_ = stats.s2 / nkf.sum(dim=1) + self.reg_covar

    def _m_step_from_stats(self, stats: SufficientStats):
        r"""
        M-step: update model parameters from accumulated sufficient
        statistics.

        Updates weights, means, and covariances using either:
        - MLE (Maximum Likelihood Estimation) if no priors
        - MAP (Maximum a Posteriori) if priors are specified
        - NIW (Normal-Inverse-Wishart) conjugate updates if both mean and
          covariance priors are given

        Parameters
        ----------
        stats : SufficientStats
            Accumulated statistics for the current E-step's
            responsibilities (see :meth:`_compute_stats`).

        Notes
        -----
        Behaviourally identical to the pre-Phase-0 ``_m_step(X, resp)``
        (see docs/superpowers/specs/2026-10-14-phase0-sufficient-stats-design.md
        for the derivation showing why reading from ``stats`` instead of
        re-touching ``X``/``resp`` changes nothing about the result).
        """
        n_samples = stats.n_samples

        if stats.aux is not None and 'nk_feat' in stats.aux:
            self._m_step_feature_weighted(stats)
            return

        # ===============================================================
        # 1. Compute effective sample counts per component
        # ===============================================================
        nk = stats.nk + 1e-20  # (n_components,)

        # ===============================================================
        # 2. Update mixing weights
        # ===============================================================
        if self.use_weight_prior:
            # MAP update with Dirichlet prior
            alpha = self.weight_concentration_prior  # (n_components,)
            total_alpha = alpha.sum()
            self.weights_ = (nk + alpha - 1.0) / (n_samples + total_alpha - self.n_components)
        else:
            # MLE update
            self.weights_ = nk / n_samples

        self.weights_.clamp_(min=1e-20)  # Ensure numerical stability

        # ===============================================================
        # 3. Update means and covariances
        # ===============================================================
        use_niw = self.use_mean_prior and self.use_covariance_prior

        if use_niw:
            # NIW conjugate updates: joint update for means and covariances
            self._update_niw_conjugate(stats, nk)
        else:
            ybar = stats.s1 / nk.unsqueeze(1)

            # Update means
            if self.use_mean_prior:
                # MAP update with Gaussian prior
                kappa0 = self.mean_precision_prior
                self.means_ = (stats.s1 + kappa0 * self.mean_prior) / (nk + kappa0).unsqueeze(1)
            else:
                # MLE update
                self.means_ = ybar

            # Update covariances
            if self.covariance_type_ in _NEW_GPCM_CODES:
                # GPCM covariance types have no MAP path (validated at
                # __init__; use_covariance_prior is guaranteed False here).
                self._update_covariances_gpcm(stats, nk)
            elif self.use_covariance_prior:
                # MAP update with Wishart/Inverse-Wishart prior
                self._update_covariances_map(stats, nk, ybar)
            else:
                # MLE update
                self._update_covariances_mle(stats, nk, ybar)

    # ===================================================================
    # NIW (Normal-Inverse-Wishart) Conjugate Prior Updates
    # ===================================================================
    def _update_niw_conjugate(self, stats: SufficientStats, nk: torch.Tensor):
        r"""
        Update means and covariances jointly using Normal-Inverse-Wishart
        (NIW) conjugate priors.

        .. math::
            \mu_n = \frac{\lambda \mu_0 + n \bar y}{\lambda + n}

            \lambda_n = \lambda + n

            \nu_n = \nu + n

            \Psi_n = \Psi + S + \frac{\lambda n}{\lambda + n} (\bar y - \mu_0)(\bar y - \mu_0)^T

        where :math:`\bar y` is the empirical mean, :math:`S` is the
        scatter matrix, and :math:`(\mu_0, \lambda, \Psi, \nu)` are the NIW
        hyperparameters.

        Parameters
        ----------
        stats : SufficientStats
        nk : torch.Tensor
            Effective sample count per component (already epsilon-guarded),
            shape (n_components,).

        Notes
        -----
        This method routes to the appropriate covariance-type-specific NIW
        update. The covariance point estimate used throughout this module
        (both here and in `_update_covariances_map`) is Σ = Ψₙ / νₙ, *not*
        the textbook Inverse-Wishart posterior mean
        E[Σ] = Ψₙ / (νₙ - n_features - 1). Ψₙ / νₙ is used deliberately:
        unlike the textbook mean, it stays finite and positive for every
        νₙ > 0 (the mean formula is undefined once νₙ <= n_features + 1,
        which a near-empty component can reach), and the two agree closely
        whenever νₙ >> n_features anyway.
        """
        # ===============================================================
        # Extract NIW prior parameters
        # ===============================================================
        mu0 = self.mean_prior  # (n_components, n_features)
        lambda0 = self.mean_precision_prior  # scalar
        psi0 = self.covariance_prior  # shape depends on covariance_type
        nu0 = self.degrees_of_freedom_prior  # scalar

        # ===============================================================
        # Empirical mean ȳ_k for each component -- the same quantity
        # stats.s2 is already centered on, so the covariance branches
        # below need no recentring correction.
        # ===============================================================
        ybar = stats.s1 / nk.unsqueeze(1)  # (n_components, n_features)

        # ===============================================================
        # Compute NIW posterior parameters
        # ===============================================================
        lambda_n = lambda0 + nk  # (n_components,)
        nu_n = nu0 + nk  # (n_components,)

        # Update means using NIW posterior mean
        mu_n = (lambda0 * mu0 + nk.unsqueeze(1) * ybar) / lambda_n.unsqueeze(1)
        self.means_ = mu_n

        # ===============================================================
        # Update covariances (route to covariance-type-specific method)
        # ===============================================================
        if self.covariance_type == 'full':
            self._update_niw_full(stats, nk, ybar, lambda0, lambda_n, nu_n, psi0)
        elif self.covariance_type == 'diag':
            self._update_niw_diag(stats, nk, ybar, lambda0, lambda_n, nu_n, psi0)
        elif self.covariance_type == 'spherical':
            self._update_niw_spherical(stats, nk, ybar, lambda0, lambda_n, nu_n, psi0)
        elif self.covariance_type == 'tied_full':
            self._update_niw_tied_full(stats, nk, ybar, lambda0, lambda_n, nu_n, psi0)
        elif self.covariance_type == 'tied_diag':
            self._update_niw_tied_diag(stats, nk, ybar, lambda0, lambda_n, nu_n, psi0)
        elif self.covariance_type == 'tied_spherical':
            self._update_niw_tied_spherical(stats, nk, ybar, lambda0, lambda_n, nu_n, psi0)
        else:
            raise ValueError(f"Unsupported covariance_type: {self.covariance_type}")

    def _update_niw_full(self, stats, nk, ybar, lambda0, lambda_n, nu_n, psi0):
        r"""
        NIW posterior update for full covariance matrices.

        Computes: Ψₙ = Ψ₀ + S + (λ₀n/(λ₀+n))(ȳ - μ₀)(ȳ - μ₀)ᵀ
        Then: Σₖ = Ψₙ / νₙ
        """
        S = stats.s2  # already centered on ȳ_k

        mean_diff = ybar - self.mean_prior  # (n_components, n_features)
        cross_term_coeff = (lambda0 * nk) / lambda_n  # (n_components,)
        cross_term = cross_term_coeff.view(-1, 1, 1) * torch.einsum('kd,ke->kde', mean_diff, mean_diff)

        psi_n = psi0 + S + cross_term
        psi_n = psi_n + self.reg_covar * torch.eye(self.n_features, device=self.device, dtype=self.dtype).unsqueeze(0)

        self.covariances_ = psi_n / nu_n.view(-1, 1, 1)

    def _update_niw_diag(self, stats, nk, ybar, lambda0, lambda_n, nu_n, psi0):
        r"""NIW posterior update for diagonal covariance matrices."""
        S = stats.s2  # (n_components, n_features)

        mean_diff_sq = (ybar - self.mean_prior).pow(2)
        cross_term_coeff = (lambda0 * nk) / lambda_n
        cross_term = cross_term_coeff.unsqueeze(-1) * mean_diff_sq

        psi_n = psi0 + S + cross_term
        psi_n = psi_n + self.reg_covar

        self.covariances_ = psi_n / nu_n.unsqueeze(-1)

    def _update_niw_spherical(self, stats, nk, ybar, lambda0, lambda_n, nu_n, psi0):
        r"""NIW posterior update for spherical covariances."""
        S = stats.s2  # (n_components,)

        mean_diff_norm_sq = (ybar - self.mean_prior).pow(2).sum(dim=1)
        cross_term_coeff = (lambda0 * nk) / lambda_n
        cross_term = cross_term_coeff * mean_diff_norm_sq

        psi_n = psi0 + S + cross_term
        psi_n = psi_n + self.reg_covar * self.n_features

        self.covariances_ = psi_n / (nu_n * self.n_features)

    def _update_niw_tied_full(self, stats, nk, ybar, lambda0, lambda_n, nu_n, psi0):
        r"""NIW posterior update for tied full covariance."""
        S = stats.s2.sum(dim=0)  # (n_features, n_features)

        mean_diff = ybar - self.mean_prior
        cross_term_coeff = (lambda0 * nk) / lambda_n
        cross_term = torch.einsum('k,kd,ke->de', cross_term_coeff, mean_diff, mean_diff)

        psi_n = psi0 + S + cross_term
        psi_n = psi_n + self.reg_covar * torch.eye(self.n_features, device=self.device, dtype=self.dtype)

        total_nu_n = self.degrees_of_freedom_prior + nk.sum()
        self.covariances_ = psi_n / total_nu_n

    def _update_niw_tied_diag(self, stats, nk, ybar, lambda0, lambda_n, nu_n, psi0):
        r"""NIW posterior update for tied diagonal covariance."""
        S = stats.s2.sum(dim=0)  # (n_features,)

        mean_diff_sq = (ybar - self.mean_prior).pow(2)
        cross_term_coeff = (lambda0 * nk) / lambda_n
        cross_term = torch.einsum('k,kd->d', cross_term_coeff, mean_diff_sq)

        psi_n = psi0 + S + cross_term
        psi_n = psi_n + self.reg_covar

        total_nu_n = self.degrees_of_freedom_prior + nk.sum()
        self.covariances_ = psi_n / total_nu_n

    def _update_niw_tied_spherical(self, stats, nk, ybar, lambda0, lambda_n, nu_n, psi0):
        r"""NIW posterior update for tied spherical covariance."""
        S = stats.s2.sum()  # scalar

        mean_diff_norm_sq = (ybar - self.mean_prior).pow(2).sum(dim=1)
        cross_term_coeff = (lambda0 * nk) / lambda_n
        cross_term = (cross_term_coeff * mean_diff_norm_sq).sum()

        psi_n = psi0 + S + cross_term
        psi_n = psi_n + self.reg_covar * nk.sum() * self.n_features

        total_nu_n = self.degrees_of_freedom_prior + nk.sum()
        self.covariances_ = psi_n / (total_nu_n * self.n_features)

    # ===================================================================
    # MAP (Maximum A Posteriori) Covariance Updates
    # ===================================================================
    def _update_covariances_map(self, stats: SufficientStats, nk: torch.Tensor, ybar: torch.Tensor):
        r"""Route to the appropriate covariance-type-specific MAP update."""
        if self.covariance_type == 'full':
            self._update_map_full(stats, nk, ybar)
        elif self.covariance_type == 'diag':
            self._update_map_diag(stats, nk, ybar)
        elif self.covariance_type == 'spherical':
            self._update_map_spherical(stats, nk, ybar)
        elif self.covariance_type == 'tied_full':
            self._update_map_tied_full(stats, nk, ybar)
        elif self.covariance_type == 'tied_diag':
            self._update_map_tied_diag(stats, nk, ybar)
        elif self.covariance_type == 'tied_spherical':
            self._update_map_tied_spherical(stats, nk, ybar)
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def _update_map_full(self, stats, nk, ybar):
        r"""
        MAP update for full covariance matrices.

        Uses Inverse-Wishart prior: Σₖ ~ IW(Ψ, ν).
        Posterior: Σₖ = (Ψ + S_k + prior_mean_term) / (ν + n_k + d)
        """
        sum_diff = self._scatter_about_means(stats, nk, ybar)  # (K, d, d)

        if self.use_mean_prior:
            mean_diff = (self.means_ - self.mean_prior).unsqueeze(-1)  # (K, d, 1)
            prior_term = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1).unsqueeze(-1) \
                         * mean_diff @ mean_diff.transpose(-1, -2)
        else:
            prior_term = torch.zeros_like(sum_diff)

        nu_n = self.degrees_of_freedom_prior + nk.unsqueeze(-1).unsqueeze(-1)

        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar * torch.eye(self.n_features, device=self.device, dtype=self.dtype).unsqueeze(0)
        ) / nu_n

    def _update_map_diag(self, stats, nk, ybar):
        r"""MAP update for diagonal covariance matrices."""
        sum_diff = self._scatter_about_means(stats, nk, ybar)  # (K, d)

        if self.use_mean_prior:
            mean_diff2 = (self.means_ - self.mean_prior).pow(2)
            prior_term = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1) * mean_diff2
        else:
            prior_term = torch.zeros_like(sum_diff)

        nu_n = self.degrees_of_freedom_prior + nk.unsqueeze(-1)

        self.covariances_ = (
            self.covariance_prior + sum_diff + prior_term + self.reg_covar
        ) / nu_n

    def _update_map_spherical(self, stats, nk, ybar):
        r"""MAP update for spherical covariance (single variance per component)."""
        sum_diff = self._scatter_about_means(stats, nk, ybar)  # (K,)

        if self.use_mean_prior:
            mean_diff2 = (self.means_ - self.mean_prior).pow(2).sum(dim=1)
            prior_term = (nk / (nk + self.mean_precision_prior)) * mean_diff2
        else:
            prior_term = torch.zeros_like(sum_diff)

        nu_n = self.degrees_of_freedom_prior + nk

        self.covariances_ = (
            self.covariance_prior + sum_diff + prior_term + self.reg_covar
        ) / (nu_n * self.n_features)

    def _update_map_tied_full(self, stats, nk, ybar):
        r"""MAP update for tied full covariance (shared across all components)."""
        sum_diff = self._scatter_about_means(stats, nk, ybar).sum(dim=0)  # (d, d)

        if self.use_mean_prior:
            mean_diff = (self.means_ - self.mean_prior).unsqueeze(-1)
            prior_term = (
                (nk / (nk + self.mean_precision_prior)).unsqueeze(-1).unsqueeze(-1)
                * torch.matmul(mean_diff, mean_diff.transpose(-1, -2))
            )
            prior_term = prior_term.sum(dim=0)
        else:
            prior_term = torch.zeros_like(sum_diff)

        nu_n = self.degrees_of_freedom_prior + nk.sum()

        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar * torch.eye(self.n_features, device=self.device, dtype=self.dtype)
        ) / nu_n

    def _update_map_tied_diag(self, stats, nk, ybar):
        r"""MAP update for tied diagonal covariance (shared across components)."""
        sum_diff = self._scatter_about_means(stats, nk, ybar).sum(dim=0)  # (d,)

        if self.use_mean_prior:
            mean_diff2 = (self.means_ - self.mean_prior).pow(2)
            prior_term = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1) * mean_diff2
            prior_term = prior_term.sum(dim=0)
        else:
            prior_term = torch.zeros_like(sum_diff)

        nu_n = self.degrees_of_freedom_prior + nk.sum()

        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar
        ) / nu_n

    def _update_map_tied_spherical(self, stats, nk, ybar):
        r"""MAP update for tied spherical covariance (single variance for all)."""
        sum_diff = self._scatter_about_means(stats, nk, ybar).sum()  # scalar

        if self.use_mean_prior:
            mean_diff2 = (self.means_ - self.mean_prior).pow(2).sum(dim=1)
            prior_term = (nk / (nk + self.mean_precision_prior)) * mean_diff2
            prior_term_total = prior_term.sum()
        else:
            prior_term_total = 0.0

        nu_n = self.degrees_of_freedom_prior + nk.sum()

        self.covariances_ = (
            self.covariance_prior + sum_diff + prior_term_total + self.reg_covar
        ) / (nu_n * self.n_features)

    # ===================================================================
    # GPCM (Celeux & Govaert 1995) Covariance Updates -- the 8 new types
    # ===================================================================
    def _update_covariances_gpcm(self, stats: SufficientStats, nk: torch.Tensor):
        r"""
        M-step for the 8 new GPCM covariance types, via
        `tgmm.covariance.gpcm.mstep_gpcm`. `stats.s2` is already `W_k =
        sum_i r_ik (x_i - ybar_k)(x_i - ybar_k)^T` (Phase 0's convention
        for `full`-shaped second moments), exactly `mstep_gpcm`'s `W`
        argument -- no further computation from `X` needed. Warm-starts
        the 5 iterative codes from `self._gpcm_state_` (the previous
        M-step's result; `None` on the first M-step of a run).
        """
        result = mstep_gpcm(
            self.covariance_type_, stats.s2, nk, self.n_features,
            state=self._gpcm_state_, reg_covar=self.reg_covar,
            gpcm_tol=self.gpcm_tol, gpcm_max_iter=self.gpcm_max_iter,
        )
        self.covariances_ = result.cov_full
        self.volume_ = result.volume
        self.shape_ = result.shape
        self.orientation_ = result.orientation
        self._gpcm_state_ = result

    # ===================================================================
    # MLE (Maximum Likelihood Estimation) Covariance Updates
    # ===================================================================
    def _update_covariances_mle(self, stats: SufficientStats, nk: torch.Tensor, ybar: torch.Tensor):
        r"""Route to the appropriate covariance-type-specific MLE update."""
        if self.covariance_type == 'full':
            self._update_mle_full(stats, nk, ybar)
        elif self.covariance_type == 'diag':
            self._update_mle_diag(stats, nk, ybar)
        elif self.covariance_type == 'spherical':
            self._update_mle_spherical(stats, nk, ybar)
        elif self.covariance_type == 'tied_full':
            self._update_mle_tied_full(stats, nk, ybar)
        elif self.covariance_type == 'tied_diag':
            self._update_mle_tied_diag(stats, nk, ybar)
        elif self.covariance_type == 'tied_spherical':
            self._update_mle_tied_spherical(stats, nk, ybar)
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def _update_mle_full(self, stats, nk, ybar):
        r"""
        MLE update for full covariance matrices.

        Σₖ = (1/n_k) Σᵢ γᵢₖ(xᵢ - μₖ)(xᵢ - μₖ)ᵀ + reg·I

        (``self.means_`` equals ``ybar`` unless a mean prior shrank it
        elsewhere in this same M-step -- see `_scatter_about_means`.)
        """
        sum_diff = self._scatter_about_means(stats, nk, ybar)
        covs = sum_diff / nk.unsqueeze(-1).unsqueeze(-1)
        covs = covs + self.reg_covar * torch.eye(self.n_features, device=self.device, dtype=self.dtype).unsqueeze(0)
        self.covariances_ = covs

    def _update_mle_diag(self, stats, nk, ybar):
        r"""MLE update for diagonal covariance matrices."""
        sum_diff = self._scatter_about_means(stats, nk, ybar)
        cov_diag = sum_diff / nk.unsqueeze(-1)
        cov_diag = cov_diag + self.reg_covar
        self.covariances_ = cov_diag

    def _update_mle_spherical(self, stats, nk, ybar):
        r"""MLE update for spherical covariance (single variance per component)."""
        sum_diff = self._scatter_about_means(stats, nk, ybar)
        cov_spherical = sum_diff / (nk * self.n_features)
        cov_spherical = cov_spherical + self.reg_covar
        self.covariances_ = cov_spherical

    def _update_mle_tied_full(self, stats, nk, ybar):
        r"""MLE update for tied full covariance (shared across components)."""
        sum_diff = self._scatter_about_means(stats, nk, ybar).sum(dim=0)
        cov_tied = sum_diff / nk.sum()
        cov_tied = cov_tied + self.reg_covar * torch.eye(self.n_features, device=self.device, dtype=self.dtype)
        self.covariances_ = cov_tied

    def _update_mle_tied_diag(self, stats, nk, ybar):
        r"""MLE update for tied diagonal covariance (shared across components)."""
        sum_diff = self._scatter_about_means(stats, nk, ybar).sum(dim=0)
        cov_tied_diag = sum_diff / nk.sum()
        cov_tied_diag = cov_tied_diag + self.reg_covar
        self.covariances_ = cov_tied_diag

    def _update_mle_tied_spherical(self, stats, nk, ybar):
        r"""MLE update for tied spherical covariance (single variance for all)."""
        sum_diff = self._scatter_about_means(stats, nk, ybar).sum()
        cov_tied_spherical = sum_diff / (nk.sum() * self.n_features)
        cov_tied_spherical = cov_tied_spherical + self.reg_covar
        self.covariances_ = cov_tied_spherical

    # ===================================================================
    # Gibbs sampling (collapsed, Neal's Algorithm 3) -- n_components=None
    #
    # Ported from tgmm.dpgmm.DPGMM (see
    # docs/superpowers/specs/2026-07-28-dpgmm-design.md for the full
    # derivation). Operates on self.mean_prior_/self.covariance_prior_
    # (set by _resolve_priors_gibbs below) -- distinct from
    # self.mean_prior/self.covariance_prior, the raw constructor args
    # validated in __init__, since Gibbs mode's prior is a single shared
    # vector/matrix (components are created/destroyed dynamically, so
    # there's no fixed K to broadcast a per-component prior against) while
    # EM's _init_priors broadcasts one prior per component slot.
    # ===================================================================

    def _resolve_max_components(self, n_samples: int) -> int:
        if self.max_components is not None:
            return self.max_components
        if self.init_k is not None:
            return self.init_k
        # max_components=None and init_k=None: unbounded with no caller-
        # supplied seed. Single-site Gibbs moves split a cluster that got
        # merged at init only very slowly, so an overly conservative seed
        # can permanently starve the sampler of components -- keep this
        # generous.
        return max(2, min(20, n_samples // 2))

    def _finalize_active_components(self, n_samples: int):
        expected_counts = self.weights_ * n_samples
        self.active_ = expected_counts > self.weight_threshold
        self.n_components_ = int(self.active_.sum().item())

    def _resolve_priors_gibbs(self, X: torch.Tensor, init_labels: torch.Tensor):
        r"""
        Normalize the (already-required, non-None -- see __init__) NIW
        prior into the shapes this section's per-sweep math expects: a
        single shared mean vector / covariance matrix. `init_labels` is
        accepted for signature parity with the pre-merge DPGMM version
        but is unused now that there's no data-dependent default left to
        resolve from it (see suggest_priors for the equivalent explicit
        helper).
        """
        n_features = self.n_features

        mp = self.mean_prior if isinstance(self.mean_prior, torch.Tensor) \
            else torch.tensor(self.mean_prior, device=self.device, dtype=self.dtype)
        self.mean_prior_ = mp.reshape(-1)[:n_features].to(device=self.device, dtype=self.dtype)

        if self.degrees_of_freedom_prior <= n_features - 1:
            raise ValueError(
                f"degrees_of_freedom_prior must be > {n_features - 1}, "
                f"got {self.degrees_of_freedom_prior}."
            )

        cp = self.covariance_prior if isinstance(self.covariance_prior, torch.Tensor) \
            else torch.tensor(self.covariance_prior, device=self.device, dtype=self.dtype)
        self.covariance_prior_ = cp.to(device=self.device, dtype=self.dtype)

    def _niw_posterior_full(self, n_k: torch.Tensor, sum_x_k: torch.Tensor, sum_xxT_k: torch.Tensor):
        r"""
        Batched NIW posterior parameters for 'full' covariance, given cached
        per-component sufficient statistics. Vectorized over the leading
        component dimension so all K slots can be scored for one point in a
        single call.

        Parameters
        ----------
        n_k : (K,)
        sum_x_k : (K, d)
        sum_xxT_k : (K, d, d)

        Returns
        -------
        mu_n : (K, d)
        lambda_n : (K,)
        nu_n : (K,)
        psi_n : (K, d, d)
        """
        d = self.n_features
        lambda0 = self.mean_precision_prior
        nu0 = self.degrees_of_freedom_prior
        mu0 = self.mean_prior_
        psi0 = self.covariance_prior_

        n_k_safe = n_k.clamp(min=1e-12)
        empirical_mean = sum_x_k / n_k_safe.unsqueeze(-1)
        S = sum_xxT_k - n_k.unsqueeze(-1).unsqueeze(-1) * (
            empirical_mean.unsqueeze(-1) @ empirical_mean.unsqueeze(-2)
        )

        lambda_n = lambda0 + n_k
        nu_n = nu0 + n_k
        mu_n = (lambda0 * mu0.unsqueeze(0) + sum_x_k) / lambda_n.unsqueeze(-1)

        mean_diff = empirical_mean - mu0.unsqueeze(0)
        cross_coeff = (lambda0 * n_k) / lambda_n
        cross_term = cross_coeff.unsqueeze(-1).unsqueeze(-1) * (
            mean_diff.unsqueeze(-1) @ mean_diff.unsqueeze(-2)
        )

        psi_n = psi0.unsqueeze(0) + S + cross_term
        psi_n = psi_n + self.reg_covar * torch.eye(d, device=self.device, dtype=self.dtype).unsqueeze(0)
        return mu_n, lambda_n, nu_n, psi_n

    def _mvt_log_prob_full(self, x: torch.Tensor, mu_n: torch.Tensor, lambda_n: torch.Tensor,
                            nu_n: torch.Tensor, psi_n: torch.Tensor) -> torch.Tensor:
        r"""
        Log-density of the multivariate Student-t posterior predictive
        (NIW marginal likelihood), batched over the leading component
        dimension:

            t_{nu_n-d+1}(x; mu_n, Psi_n(lambda_n+1) / (lambda_n(nu_n-d+1)))
        """
        d = self.n_features
        df = nu_n - d + 1.0
        scale = psi_n * ((lambda_n + 1.0) / (lambda_n * df)).unsqueeze(-1).unsqueeze(-1)
        chol = torch.linalg.cholesky(scale)

        diff = x.unsqueeze(0) - mu_n
        diff_ = diff.unsqueeze(-1)
        solve = torch.cholesky_solve(diff_, chol)
        mahal = (diff_ * solve).sum(dim=(1, 2))
        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            torch.lgamma((df + d) / 2.0) - torch.lgamma(df / 2.0)
            - 0.5 * d * torch.log(df * math.pi)
            - 0.5 * log_det
            - 0.5 * (df + d) * torch.log1p(mahal / df)
        )

    def _niw_posterior_diag(self, n_k: torch.Tensor, sum_x_k: torch.Tensor, sum_x2_k: torch.Tensor):
        r"""
        Batched NIW posterior for 'diag' covariance. Each dimension is an
        independent Normal-Inverse-Gamma model with its own scale
        `psi0_j`, but sharing one degrees-of-freedom parameter `nu0` across
        dimensions -- identical in form to `_niw_posterior_full` /
        `_update_niw_diag`, just without off-diagonal terms. `sum_x2_k` is
        the per-dimension sum of squares (K, d), not a full outer-product
        matrix.
        """
        lambda0 = self.mean_precision_prior
        nu0 = self.degrees_of_freedom_prior
        mu0 = self.mean_prior_
        psi0 = self.covariance_prior_

        n_k_safe = n_k.clamp(min=1e-12)
        empirical_mean = sum_x_k / n_k_safe.unsqueeze(-1)
        S = sum_x2_k - n_k.unsqueeze(-1) * empirical_mean.pow(2)

        lambda_n = lambda0 + n_k
        nu_n = nu0 + n_k
        mu_n = (lambda0 * mu0.unsqueeze(0) + sum_x_k) / lambda_n.unsqueeze(-1)

        mean_diff_sq = (empirical_mean - mu0.unsqueeze(0)).pow(2)
        cross_coeff = (lambda0 * n_k) / lambda_n
        cross_term = cross_coeff.unsqueeze(-1) * mean_diff_sq

        psi_n = psi0.unsqueeze(0) + S + cross_term + self.reg_covar
        return mu_n, lambda_n, nu_n, psi_n

    def _mvt_log_prob_diag(self, x: torch.Tensor, n_k: torch.Tensor, mu_n: torch.Tensor,
                            lambda_n: torch.Tensor, nu_n: torch.Tensor, psi_n: torch.Tensor) -> torch.Tensor:
        r"""
        Log-density of the per-dimension Student-t posterior predictive for
        'diag' covariance, summed (i.e. product in probability space) over
        dimensions. Unlike `_mvt_log_prob_full`, the degrees of freedom are
        `nu_n` directly -- there is no "-d+1" correction here, since each
        dimension marginalizes an independent scalar variance rather than a
        joint d x d Wishart-distributed matrix. `n_k` is accepted for a
        uniform call signature across covariance types but unused.
        """
        df = nu_n.unsqueeze(-1)  # (K, 1), broadcasts against (K, d)
        scale = psi_n * (lambda_n + 1.0).unsqueeze(-1) / (nu_n.unsqueeze(-1) * lambda_n.unsqueeze(-1))
        diff = x.unsqueeze(0) - mu_n

        log_prob_per_dim = (
            torch.lgamma((df + 1.0) / 2.0) - torch.lgamma(df / 2.0)
            - 0.5 * torch.log(df * math.pi * scale)
            - 0.5 * (df + 1.0) * torch.log1p(diff.pow(2) / (df * scale))
        )
        return log_prob_per_dim.sum(dim=-1)

    def _niw_posterior_spherical(self, n_k: torch.Tensor, sum_x_k: torch.Tensor, sum_sq_k: torch.Tensor):
        r"""
        Batched NIW posterior for 'spherical' covariance: a single shared
        scalar variance pools all `d` dimensions together (an isotropic
        Normal-Inverse-Gamma model), identical in form to
        `_update_niw_spherical`. `sum_sq_k` is the total sum of squares
        across samples AND dimensions (K,), not a per-dimension or
        per-sample statistic.
        """
        mu0 = self.mean_prior_
        lambda0 = self.mean_precision_prior
        nu0 = self.degrees_of_freedom_prior
        psi0 = self.covariance_prior_

        n_k_safe = n_k.clamp(min=1e-12)
        empirical_mean = sum_x_k / n_k_safe.unsqueeze(-1)
        S = sum_sq_k - n_k * empirical_mean.pow(2).sum(dim=-1)

        lambda_n = lambda0 + n_k
        nu_n = nu0 + n_k  # matches the covariance point-estimate convention elsewhere
        mu_n = (lambda0 * mu0.unsqueeze(0) + sum_x_k) / lambda_n.unsqueeze(-1)

        mean_diff_norm_sq = (empirical_mean - mu0.unsqueeze(0)).pow(2).sum(dim=-1)
        cross_coeff = (lambda0 * n_k) / lambda_n
        cross_term = cross_coeff * mean_diff_norm_sq

        psi_n = psi0 + S + cross_term + self.reg_covar * self.n_features
        return mu_n, lambda_n, nu_n, psi_n

    def _mvt_log_prob_spherical(self, x: torch.Tensor, n_k: torch.Tensor, mu_n: torch.Tensor,
                                 lambda_n: torch.Tensor, nu_n: torch.Tensor, psi_n: torch.Tensor) -> torch.Tensor:
        r"""
        Log-density of the isotropic multivariate Student-t posterior
        predictive for 'spherical' covariance. Degrees of freedom here are
        `nu0 + n_k * d` (an isotropic-NIG marginal pooling all `d`
        dimensions' worth of "observations" into one shared variance) --
        deliberately *not* the `nu_n = nu0 + n_k` used for the point-estimate
        covariance elsewhere; these are two different, each internally
        valid, uses of the same prior hyperparameters. See
        docs/superpowers/specs/2026-07-28-dpgmm-design.md.
        """
        d = self.n_features
        nu0 = self.degrees_of_freedom_prior
        df = nu0 + n_k * d
        scale = psi_n * (lambda_n + 1.0) / (lambda_n * df)
        diff_sq = (x.unsqueeze(0) - mu_n).pow(2).sum(dim=-1)

        return (
            torch.lgamma((df + d) / 2.0) - torch.lgamma(df / 2.0)
            - 0.5 * d * torch.log(df * math.pi * scale)
            - 0.5 * (df + d) * torch.log1p(diff_sq / (df * scale))
        )

    def _gibbs_log_likelihood(self, x: torch.Tensor, n_k: torch.Tensor, sum_x_k: torch.Tensor,
                               stat2_k: torch.Tensor) -> torch.Tensor:
        r"""Batched (over active components) NIW marginal log-likelihood of
        a single point `x`, dispatching on `covariance_type`."""
        if self.covariance_type == "full":
            mu_n, lambda_n, nu_n, psi_n = self._niw_posterior_full(n_k, sum_x_k, stat2_k)
            return self._mvt_log_prob_full(x, mu_n, lambda_n, nu_n, psi_n)
        elif self.covariance_type == "diag":
            mu_n, lambda_n, nu_n, psi_n = self._niw_posterior_diag(n_k, sum_x_k, stat2_k)
            return self._mvt_log_prob_diag(x, n_k, mu_n, lambda_n, nu_n, psi_n)
        elif self.covariance_type == "spherical":
            mu_n, lambda_n, nu_n, psi_n = self._niw_posterior_spherical(n_k, sum_x_k, stat2_k)
            return self._mvt_log_prob_spherical(x, n_k, mu_n, lambda_n, nu_n, psi_n)
        else:
            raise NotImplementedError(
                f"Gibbs sampling for covariance_type={self.covariance_type!r} is not implemented."
            )

    def _gibbs_final_point_estimate(self, n_k: torch.Tensor, sum_x_k: torch.Tensor, stat2_k: torch.Tensor):
        r"""Point-estimate (means_, covariances_) from final sufficient stats, dispatching on covariance_type."""
        if self.covariance_type == "full":
            mu_n, lambda_n, nu_n, psi_n = self._niw_posterior_full(n_k, sum_x_k, stat2_k)
            return mu_n, psi_n / nu_n.unsqueeze(-1).unsqueeze(-1)
        elif self.covariance_type == "diag":
            mu_n, lambda_n, nu_n, psi_n = self._niw_posterior_diag(n_k, sum_x_k, stat2_k)
            return mu_n, psi_n / nu_n.unsqueeze(-1)
        elif self.covariance_type == "spherical":
            mu_n, lambda_n, nu_n, psi_n = self._niw_posterior_spherical(n_k, sum_x_k, stat2_k)
            return mu_n, psi_n / (nu_n * self.n_features)
        else:
            raise NotImplementedError(
                f"Gibbs sampling for covariance_type={self.covariance_type!r} is not implemented."
            )

    def _gibbs_point_stat2(self, xi: torch.Tensor) -> torch.Tensor:
        r"""Per-point contribution to the type-specific 'second statistic'
        cache: outer product (full), per-dim square (diag), or total
        sum-of-squares scalar (spherical)."""
        if self.covariance_type == "full":
            return torch.outer(xi, xi)
        elif self.covariance_type == "diag":
            return xi.pow(2)
        elif self.covariance_type == "spherical":
            return xi.pow(2).sum()
        else:
            raise NotImplementedError(
                f"Gibbs sampling for covariance_type={self.covariance_type!r} is not implemented."
            )

    def _gibbs_batch_stat2(self, Xk: torch.Tensor) -> torch.Tensor:
        r"""Batch version of `_gibbs_point_stat2`, used once at initialization."""
        if self.covariance_type == "full":
            return Xk.t() @ Xk
        elif self.covariance_type == "diag":
            return (Xk ** 2).sum(dim=0)
        elif self.covariance_type == "spherical":
            return (Xk ** 2).sum()
        else:
            raise NotImplementedError(
                f"Gibbs sampling for covariance_type={self.covariance_type!r} is not implemented."
            )

    def _gibbs_zero_stat2(self, K: int) -> torch.Tensor:
        r"""Zero-initialized type-specific 'second statistic' cache for `K` components."""
        d = self.n_features
        if self.covariance_type == "full":
            return torch.zeros(K, d, d, device=self.device, dtype=self.dtype)
        elif self.covariance_type == "diag":
            return torch.zeros(K, d, device=self.device, dtype=self.dtype)
        elif self.covariance_type == "spherical":
            return torch.zeros(K, device=self.device, dtype=self.dtype)
        else:
            raise NotImplementedError(
                f"Gibbs sampling for covariance_type={self.covariance_type!r} is not implemented."
            )

    def _fit_gibbs(self, X: torch.Tensor):
        if self.covariance_type not in ("full", "diag", "spherical"):
            raise NotImplementedError(
                f"Gibbs sampling for covariance_type={self.covariance_type!r} "
                "is not yet implemented."
            )
        if self.max_components is not None:
            self._fit_gibbs_truncated(X, self.max_components)
        else:
            self._fit_gibbs_unbounded(X)

    def _fit_gibbs_truncated(self, X: torch.Tensor, K: int):
        r"""
        Truncated collapsed Gibbs sampling: K fixed slots (some may end up
        empty), each point's assignment resampled from

            p(z_i = k | z_-i) propto (n_k^{-i} + alpha/K) * marginal_lik(x_i | table k)

        the finite-dimensional Dirichlet-multinomial (Polya urn)
        approximation to the CRP -- exact for a symmetric
        Dirichlet(alpha/K, ..., alpha/K) prior over K weights, and
        converging to the true CRP rule (`propto n_k^{-i}`, plus a
        separate `propto alpha` new-table term) as K grows relative to the
        number of clusters actually in use. This is deliberately the same
        occupancy-driven rule `_fit_gibbs_unbounded` uses (`torch.log(n_k)`
        there, `torch.log(n_k + alpha/K)` here) -- see
        docs/superpowers/specs/2026-07-28-dpgmm-design.md.
        """
        n_samples, d = X.shape

        X_cpu = X.cpu()
        init_means = GMMInitializer.kmeans(X_cpu, K).to(device=self.device, dtype=self.dtype)
        z = torch.cdist(X, init_means).argmin(dim=1)

        self._resolve_priors_gibbs(X, z)

        n_k = torch.zeros(K, device=self.device, dtype=self.dtype)
        sum_x_k = torch.zeros(K, d, device=self.device, dtype=self.dtype)
        stat2_k = self._gibbs_zero_stat2(K)
        for k in range(K):
            mask = z == k
            n_k[k] = mask.sum()
            if n_k[k] > 0:
                Xk = X[mask]
                sum_x_k[k] = Xk.sum(dim=0)
                stat2_k[k] = self._gibbs_batch_stat2(Xk)

        alpha_per_slot = self.alpha / K

        for sweep in range(self.max_iter):
            for i in range(n_samples):
                xi = X[i]
                k_old = int(z[i].item())
                n_k[k_old] -= 1
                sum_x_k[k_old] -= xi
                stat2_k[k_old] -= self._gibbs_point_stat2(xi)

                log_lik = self._gibbs_log_likelihood(xi, n_k, sum_x_k, stat2_k)
                log_pi = torch.log(n_k + alpha_per_slot)
                log_probs = log_pi + log_lik
                probs = torch.softmax(log_probs, dim=0)
                k_new = int(torch.multinomial(probs, 1).item())

                z[i] = k_new
                n_k[k_new] += 1
                sum_x_k[k_new] += xi
                stat2_k[k_new] += self._gibbs_point_stat2(xi)

            order = torch.argsort(n_k, descending=True)
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(K, device=self.device)
            n_k = n_k[order]
            sum_x_k = sum_x_k[order]
            stat2_k = stat2_k[order]
            z = inv_order[z]

        self.means_, self.covariances_ = self._gibbs_final_point_estimate(n_k, sum_x_k, stat2_k)
        self.weights_ = n_k / n_k.sum()

        self.n_iter_ = self.max_iter
        self.fitted_ = True
        self.converged_ = True
        self._finalize_active_components(n_samples)

    def _fit_gibbs_unbounded(self, X: torch.Tensor):
        r"""
        Neal's Algorithm 3: collapsed Gibbs sampling for a Dirichlet Process
        mixture with no fixed component cap. Each point is assigned to an
        existing component with probability proportional to that
        component's current occupancy (the Chinese Restaurant Process
        "sit at an occupied table" term), or to a brand-new component with
        probability proportional to `alpha` (the "start a new table" term),
        scored by the NIW marginal likelihood under the prior alone.

        Components are keyed by an ever-increasing integer id rather than a
        fixed-size array index, since components are created and destroyed
        every sweep. The final sweep's partition is reported directly (a
        single posterior draw) rather than an average -- see the design
        doc for why cross-sweep averaging isn't meaningful here.
        """
        n_samples, d = X.shape

        K_init = self._resolve_max_components(n_samples)
        X_cpu = X.cpu()
        init_means = GMMInitializer.kmeans(X_cpu, K_init).to(device=self.device, dtype=self.dtype)
        z_init = torch.cdist(X, init_means).argmin(dim=1)

        self._resolve_priors_gibbs(X, z_init)

        next_id = 0
        stats = {}
        z = torch.empty(n_samples, dtype=torch.long, device=self.device)
        for k in range(K_init):
            mask = z_init == k
            if not mask.any():
                continue
            cid = next_id
            next_id += 1
            Xk = X[mask]
            stats[cid] = [
                torch.tensor(float(Xk.shape[0]), device=self.device, dtype=self.dtype),
                Xk.sum(dim=0),
                self._gibbs_batch_stat2(Xk),
            ]
            z[mask] = cid

        burn_in = self.burn_in if self.burn_in is not None else max(1, self.max_iter // 5)
        k_history = []
        log_alpha = math.log(self.alpha)

        zero_n = torch.zeros(1, device=self.device, dtype=self.dtype)
        zero_sum_x = torch.zeros(1, d, device=self.device, dtype=self.dtype)
        zero_stat2 = self._gibbs_zero_stat2(1)

        for sweep in range(self.max_iter):
            for i in range(n_samples):
                xi = X[i]
                cid_old = int(z[i].item())
                stats[cid_old][0] -= 1.0
                stats[cid_old][1] -= xi
                stats[cid_old][2] -= self._gibbs_point_stat2(xi)
                if stats[cid_old][0].item() <= 0:
                    del stats[cid_old]

                active_ids = list(stats.keys())
                if active_ids:
                    n_k = torch.stack([stats[c][0] for c in active_ids])
                    sum_x_k = torch.stack([stats[c][1] for c in active_ids])
                    stat2_k = torch.stack([stats[c][2] for c in active_ids])
                    log_probs_existing = torch.log(n_k) + self._gibbs_log_likelihood(
                        xi, n_k, sum_x_k, stat2_k)
                else:
                    log_probs_existing = torch.empty(0, device=self.device, dtype=self.dtype)

                log_prob_new = log_alpha + self._gibbs_log_likelihood(
                    xi, zero_n, zero_sum_x, zero_stat2)[0]

                all_log_probs = torch.cat([log_probs_existing, log_prob_new.unsqueeze(0)])
                probs = torch.softmax(all_log_probs, dim=0)
                choice = int(torch.multinomial(probs, 1).item())

                if choice == len(active_ids):
                    cid_new = next_id
                    next_id += 1
                    stats[cid_new] = [
                        torch.tensor(1.0, device=self.device, dtype=self.dtype),
                        xi.clone(),
                        self._gibbs_point_stat2(xi).clone(),
                    ]
                else:
                    cid_new = active_ids[choice]
                    stats[cid_new][0] += 1.0
                    stats[cid_new][1] += xi
                    stats[cid_new][2] += self._gibbs_point_stat2(xi)
                z[i] = cid_new

            if sweep >= burn_in:
                k_history.append(len(stats))

        active_ids = list(stats.keys())
        n_k = torch.stack([stats[c][0] for c in active_ids])
        sum_x_k = torch.stack([stats[c][1] for c in active_ids])
        stat2_k = torch.stack([stats[c][2] for c in active_ids])
        self.means_, self.covariances_ = self._gibbs_final_point_estimate(n_k, sum_x_k, stat2_k)
        self.weights_ = n_k / n_k.sum()
        self.n_iter_ = self.max_iter
        self.fitted_ = True
        self.converged_ = True
        self._finalize_active_components(n_samples)
        self.n_components_history_ = k_history

    def _n_parameters(self) -> int:
        r"""
        Number of free parameters in the fitted model.

        Counts mixture weights ($n_{components\_} - 1$, since they sum to 1),
        means ($n_{components\_} \times n_{features}$), and covariance
        parameters, whose count depends on `covariance_type` (see
        `_expected_covar_shape` for the corresponding parameter *shapes*).
        Used by `bic`/`aic`. Uses `self.n_components_` (set after every
        fit, in both EM and Gibbs mode -- see `fit`), not `self.n_components`
        directly: in Gibbs mode `self.n_components` is `None` and
        `n_components_` is the inferred *active* count (pruned slots
        contribute negligibly to `score`'s likelihood, so they shouldn't
        count as parameters "actually in use"); in EM mode the two are
        equal, since EM has no pruning concept. If `n_components_` hasn't
        been populated at all (e.g. `weights_`/`means_`/`covariances_` were
        set directly rather than via `fit()`, to score externally-fitted
        parameters), falls back to `self.weights_.shape[0]` -- the actual
        fitted slot count, same quantity `sample()`'s validation uses.
        """
        n_features = self.n_features
        n_components = self.n_components_
        if n_components is None:
            n_components = self.weights_.shape[0]
        if self.covariance_type == 'full':
            cov_params = n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == 'diag':
            cov_params = n_components * n_features
        elif self.covariance_type == 'spherical':
            cov_params = n_components
        elif self.covariance_type == 'tied_full':
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == 'tied_diag':
            cov_params = n_features
        elif self.covariance_type == 'tied_spherical':
            cov_params = 1
        elif self.covariance_type_ in _NEW_GPCM_CODES:
            K, d = n_components, n_features
            cov_params = {
                'VEI': d + K - 1,
                'EVI': 1 + K * (d - 1),
                'VEE': K - 1 + d * (d + 1) / 2.0,
                'EVE': 1 + K * (d - 1) + d * (d - 1) / 2.0,
                'EEV': d + K * d * (d - 1) / 2.0,
                'VVE': K * d + d * (d - 1) / 2.0,
                'VEV': K + d - 1 + K * d * (d - 1) / 2.0,
                'EVV': 1 + K * (d - 1) + K * d * (d - 1) / 2.0,
            }[self.covariance_type_]
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

        mean_params = n_features * n_components
        weight_params = n_components - 1
        noise_params = 1 if self.noise else 0  # the noise weight pi_0
        return int(cov_params + mean_params + weight_params + noise_params)

    def _active_score(self, X: torch.Tensor) -> float:
        r"""
        Mean log-likelihood of `X` under only the *active* components
        (`self.active_`), with their weights renormalized to sum to 1
        over just that subset.

        Used by `bic`/`aic` to stay internally consistent with
        `_n_parameters()` in Gibbs mode: `_n_parameters()` charges for
        `self.n_components_` (the active count), while plain `score()`
        evaluates the likelihood of the *entire* fitted mixture
        (`self.weights_`, unmasked) -- correct as a general-purpose
        score, but not what a fair BIC/AIC comparison wants when some
        components have been pruned from the parameter count.

        Reuses `_e_step`'s per-covariance-type math (rather than
        duplicating it) by temporarily swapping `weights_`/`means_`/
        `covariances_` down to the active-only, renormalized subset for
        the duration of the call -- mirroring the restrict-to-`active_`
        pattern used for prediction elsewhere (e.g. the `restricted_predict`
        helper in notebooks/11_dpgmm.ipynb), just applied to the likelihood
        instead of a hard assignment. Not part of the public API.

        Parameters
        ----------
        X : torch.Tensor
            Data to score, shape (n_samples, n_features).

        Returns
        -------
        float
            Mean log-likelihood under the active-only, renormalized
            mixture.
        """
        resp, log_prob_norm = self._active_e_step(X)
        return log_prob_norm.mean().item()

    def _active_e_step(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        `_e_step(X)` restricted to the active components (`self.active_`),
        with their weights renormalized to sum to 1 over just that subset.
        Shared by `_active_score` (needs only the log-likelihood) and
        `icl` (also needs the responsibilities) so the active-components
        swap-and-restore logic lives in exactly one place. See
        `_active_score`'s docstring for why this restriction exists.

        Parameters
        ----------
        X : torch.Tensor
            Data to score, shape (n_samples, n_features).

        Returns
        -------
        resp : torch.Tensor
            Responsibilities under the active-only, renormalized mixture,
            shape (n_samples, n_active_components).
        log_prob_norm : torch.Tensor
            Per-sample log-likelihood under that same mixture, shape
            (n_samples,).
        """
        active_weights = self.weights_[self.active_]
        active_weights = active_weights / active_weights.sum()
        orig_weights, orig_means, orig_covariances = self.weights_, self.means_, self.covariances_
        try:
            self.weights_ = active_weights
            self.means_ = self.means_[self.active_]
            self.covariances_ = self.covariances_[self.active_]
            return self._e_step(X.to(self.device))
        finally:
            self.weights_, self.means_, self.covariances_ = orig_weights, orig_means, orig_covariances

    def _is_pruned_gibbs_fit(self) -> bool:
        r"""
        Whether this is a Gibbs-mode fit with components pruned below
        `self.weights_.shape[0]` -- the condition under which `bic`/`aic`/
        `icl` must use the active-only mixture instead of the full fitted
        one, to stay consistent with what `_n_parameters()` charges for.
        """
        return (self.n_components is None and self.active_ is not None
                and self.n_components_ is not None
                and self.n_components_ != self.weights_.shape[0])

    def _score_for_information_criterion(self, X: torch.Tensor) -> float:
        r"""
        The log-likelihood `bic`/`aic` should use: plain `score(X)` in
        every case except a Gibbs-mode fit where some components have
        been pruned, in which case `_active_score(X)` is used instead so
        the likelihood term is charged against the same component set
        `_n_parameters()` counts. See `_active_score`, `_is_pruned_gibbs_fit`,
        and `bic`'s docstring.
        """
        if self._is_pruned_gibbs_fit():
            return self._active_score(X)
        return self.score(X)

    def _resp_for_information_criterion(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        The responsibilities `icl` should use: mirrors
        `_score_for_information_criterion`, but returns `resp` instead of
        the scalar log-likelihood, for the same active-components-only
        reason in Gibbs mode.
        """
        if self._is_pruned_gibbs_fit():
            resp, _ = self._active_e_step(X)
            return resp
        resp, _ = self._e_step(X.to(self.device))
        return resp

    # ===================================================================
    # Sampling Method
    # ===================================================================
    
    # ===================================================================
    # Marginalising and conditioning (v0.5 item 1.3)
    # ===================================================================
    def _require_derivable(self, what: str):
        r"""Common guard for `marginal` / `condition`: not Gibbs, not noise, fitted."""
        if self.n_components is None:
            raise NotImplementedError(
                f"{what} is not supported for Gibbs-mode fits (n_components=None): the active-component "
                "bookkeeping has no meaning for a derived model. Fit with a fixed n_components instead."
            )
        if self.noise:
            raise NotImplementedError(
                f"{what} is not supported for noise=True models: the uniform noise component has no "
                "Gaussian-mixture representation after marginalising or conditioning."
            )
        if not self.fitted_ or self.means_ is None:
            raise RuntimeError(f"{what} requires a fitted model: call fit() first.")

    def _dense_covariances(self) -> torch.Tensor:
        r"""Assembled `(K, d, d)` covariance matrices for every covariance type."""
        K, d = self.means_.shape
        cov = self.covariances_
        eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
        t = self.covariance_type
        if t == 'diag':
            return torch.diag_embed(cov)
        if t == 'spherical':
            return cov.view(K, 1, 1) * eye
        if t == 'tied_full':
            return cov.unsqueeze(0).expand(K, d, d)
        if t == 'tied_diag':
            return torch.diag_embed(cov).unsqueeze(0).expand(K, d, d)
        if t == 'tied_spherical':
            return (cov * eye).unsqueeze(0).expand(K, d, d)
        return cov  # 'full' and the 8 GPCM codes without a classic name are stored assembled

    def marginal(self, dims) -> "GaussianMixture":
        r"""
        The marginal mixture over a subset of the coordinates.

        Component :math:`k` becomes
        :math:`\mathcal N(\mu_k[\text{dims}],\ \Sigma_k[\text{dims},\text{dims}])` with
        the same weights.

        Parameters
        ----------
        dims : sequence of int
            Distinct coordinates in ``[0, n_features)``; output coordinate
            ``j`` is input coordinate ``dims[j]`` (order is kept).

        Returns
        -------
        GaussianMixture
            A new fitted model (no priors are copied). `diag`, `spherical`
            and the `tied_*` types keep their type; `full` and the 8 GPCM
            codes without a classic name give `covariance_type='full'` (the
            volume/shape/orientation decomposition does not survive
            restriction).

        Raises
        ------
        NotImplementedError
            For Gibbs-mode (`n_components=None`) and `noise=True` models.
        """
        self._require_derivable('marginal')
        K, d = self.means_.shape
        dims = _validate_dims(dims, d, 'dims')
        idx = torch.tensor(dims, dtype=torch.long, device=self.means_.device)
        cov, t = self.covariances_, self.covariance_type
        new_type = t
        if t == 'diag' or t == 'tied_diag':
            new_cov = cov[..., idx]
        elif t == 'spherical' or t == 'tied_spherical':
            new_cov = cov.clone()
        elif t == 'tied_full':
            new_cov = cov[idx][:, idx]
        else:  # 'full' and the 8 GPCM codes without a classic name
            new_cov = cov[:, idx][:, :, idx]
            new_type = 'full'
        model = GaussianMixture(
            n_components=K, n_features=len(dims), covariance_type=new_type,
            reg_covar=self.reg_covar, missing=self.missing, device=self.device,
        )
        model.dtype = self.dtype
        model.weights_ = self.weights_.clone()
        model.means_ = self.means_[:, idx].clone()
        model.covariances_ = new_cov.clone()
        model.fitted_ = True
        model.converged_ = True
        model.n_components_ = K
        model.n_iter_ = 0
        model.lower_bound_ = float('nan')
        return model

    def condition(self, x_obs, obs_dims) -> ConditionalGaussianMixture:
        r"""
        Condition the mixture on observed coordinates (Gaussian mixture
        regression when `mean()` is taken).

        For each component, with observed block :math:`o` and unobserved
        block :math:`m`,

        .. math::
            A_k = \Sigma_{k,mo}\Sigma_{k,oo}^{-1},\quad
            m_{ik} = \mu_{k,m} + A_k(x_{o,i} - \mu_{k,o}),\quad
            C_k = \Sigma_{k,mm} - A_k\Sigma_{k,om},

        and the weights are :math:`w_{ik}\propto\pi_k\mathcal N(x_{o,i}\mid\mu_{k,o},\Sigma_{k,oo})`.

        Parameters
        ----------
        x_obs : torch.Tensor, shape (n, len(obs_dims)) or (len(obs_dims),)
            Observed values; column ``j`` is coordinate ``obs_dims[j]``. A
            1-D vector is one row.
        obs_dims : sequence of int
            Distinct observed coordinates; at least one coordinate must stay
            unobserved. The conditional is over the sorted complement.

        Returns
        -------
        ConditionalGaussianMixture
            With ``weights_ (n, K)``, ``means_ (n, K, d_m)``,
            ``covariances_ (K, d_m, d_m)``, `mean`, `log_prob`, `sample`,
            `as_gmm`.

        Raises
        ------
        NotImplementedError
            For Gibbs-mode (`n_components=None`) and `noise=True` models.
        """
        self._require_derivable('condition')
        K, d = self.means_.shape
        obs = _validate_dims(obs_dims, d, 'obs_dims')
        if len(obs) == d:
            raise ValueError("obs_dims must leave at least one coordinate unobserved")
        miss = [j for j in range(d) if j not in set(obs)]
        dev, dt = self.means_.device, self.means_.dtype
        x = torch.as_tensor(x_obs, dtype=dt, device=dev)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2 or x.shape[1] != len(obs):
            raise ValueError(f"x_obs must have shape (n, {len(obs)}) for obs_dims={obs}, got {tuple(x.shape)}")
        o = torch.tensor(obs, dtype=torch.long, device=dev)
        m = torch.tensor(miss, dtype=torch.long, device=dev)

        S = self._dense_covariances()
        S_oo = S[:, o][:, :, o]
        S_mo = S[:, m][:, :, o]
        S_mm = S[:, m][:, :, m]

        L = torch.linalg.cholesky(S_oo)                                               # (K, do, do)
        eye = torch.eye(len(obs), dtype=dt, device=dev)
        L_inv = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)
        diff = x.unsqueeze(1) - self.means_[:, o].unsqueeze(0)                         # (n, K, do)
        z = torch.einsum('kde,nke->nkd', L_inv, diff)
        maha = z.pow(2).sum(-1)                                                       # (n, K)
        log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)        # (K,)
        log_w = torch.log(self.weights_.clamp_min(1e-300)).unsqueeze(0) - 0.5 * (
            len(obs) * math.log(2.0 * math.pi) + log_det.unsqueeze(0) + maha)
        weights = torch.softmax(log_w, dim=1)

        A = S_mo @ (L_inv.transpose(-1, -2) @ L_inv)                                  # (K, dm, do)
        means = self.means_[:, m].unsqueeze(0) + torch.einsum('kmo,nko->nkm', A, diff)  # (n, K, dm)
        C = S_mm - A @ S_mo.transpose(-1, -2)
        C = 0.5 * (C + C.transpose(-1, -2))
        return ConditionalGaussianMixture(weights, means, C, obs, miss)

    def sample(self, n_samples: int = 1, component: int = None, std_radius: float = None, 
               std_range: Tuple[float, float] = None, confidence: float = None,
               confidence_range: Tuple[float, float] = None,
               center_point: torch.Tensor = None, center_radius: float = None,
               max_attempts_per_sample: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Generate new samples from the fitted GMM.

        Supports both standard sampling and constrained sampling with rejection.
        Can sample from all components (according to weights) or a specific component,
        with optional constraints on distance from mean or a center point.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
            
        component : int, optional
            If specified, samples only from this component. If None, samples 
            from all components according to their weights.
            
        std_radius : float, optional
            Only return samples within this many standard deviations from the
            component mean (Mahalanobis distance ≤ std_radius). Uses rejection
            sampling. Cannot be combined with other distance constraints.
            
        std_range : tuple of (float, float), optional
            Only return samples whose Mahalanobis distance falls within
            (min_std, max_std). Examples:
            - (1.0, 2.0): between 1 and 2 standard deviations
            - (3.0, inf): further than 3 standard deviations (outliers)
            Cannot be combined with other distance constraints.
            
        confidence : float, optional
            Only return samples within the confidence ellipse (e.g., 0.95 for 95%).
            Converted to std_radius using χ² distribution. For 2D data, 0.95 ≈ 2.45σ.
            Cannot be combined with other distance constraints.
            
        confidence_range : tuple of (float, float), optional
            Only return samples between two confidence ellipses. Example:
            (0.68, 0.95) returns samples between 68% and 95% confidence regions.
            Cannot be combined with other distance constraints.
            
        center_point : torch.Tensor, optional
            Only return samples within center_radius Euclidean distance from this
            point. Shape (n_features,). Must be used with center_radius.
            Cannot be combined with other distance constraints.
            
        center_radius : float, optional
            Euclidean distance threshold from center_point.
            Must be used with center_point.
            
        max_attempts_per_sample : int, default=1000
            Maximum rejection sampling attempts per sample before raising error.
            Increase for very restrictive constraints.

        Returns
        -------
        samples : torch.Tensor
            Generated samples, shape (n_samples, n_features).
        indices : torch.Tensor
            Component index for each sample, shape (n_samples,).
            
        Raises
        ------
        ValueError
            If parameter combinations are invalid or constraints conflict.
        RuntimeError
            If rejection sampling fails to find valid samples.
            
        Warnings
        --------
        Issues warning if model hasn't been fitted or didn't converge.
        """
        # ===============================================================
        # 1. Check model state
        # ===============================================================
        if not self.fitted_:
            warnings.warn("GMM has not been fitted. Results may be unreliable.", UserWarning)
        elif not self.converged_:
            warnings.warn("GMM did not converge. Results may be unreliable.", UserWarning)

        # With noise=True only plain sampling is supported: noise draws (index -1) come from
        # BaseMixture.sample; constrained/rejection sampling has no meaning for the noise column.
        if self.noise:
            if any(v is not None for v in (component, std_radius, std_range, confidence,
                                           confidence_range, center_point, center_radius)):
                raise NotImplementedError(
                    "Constrained sampling (component=, std_radius=, std_range=, confidence=, "
                    "confidence_range=, center_point=/center_radius=) is not supported with noise=True."
                )
            return BaseMixture.sample(self, n_samples)

        # ===============================================================
        # 2. Validate constraint parameter combinations
        # ===============================================================
        constraint_params = [std_radius, std_range, confidence, confidence_range, 
                            (center_point, center_radius)]
        non_none_params = [p for p in constraint_params 
                          if p is not None and (not isinstance(p, tuple) or 
                                               all(x is not None for x in p))]
        
        if len(non_none_params) > 1:
            raise ValueError(
                "Cannot specify more than one of: std_radius, std_range, confidence, "
                "confidence_range, or center constraints. Use only one."
            )
        
        # ===============================================================
        # 3. Validate and convert std_range
        # ===============================================================
        if std_range is not None:
            if len(std_range) != 2:
                raise ValueError("std_range must be a tuple of exactly 2 values (min_std, max_std)")
            min_std, max_std = std_range
            if min_std < 0:
                raise ValueError("std_range minimum must be non-negative")
            if max_std <= min_std:
                raise ValueError("std_range maximum must be greater than minimum")
        
        # ===============================================================
        # 4. Convert confidence to std_radius
        # ===============================================================
        if confidence is not None:
            if not (0 < confidence < 1):
                raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
            std_radius = self._confidence_to_std_radius(confidence)
        
        # ===============================================================
        # 5. Convert confidence_range to std_range
        # ===============================================================
        if confidence_range is not None:
            if len(confidence_range) != 2:
                raise ValueError(
                    "confidence_range must be a tuple of exactly 2 values "
                    "(min_confidence, max_confidence)"
                )
            min_conf, max_conf = confidence_range
            if not (0 <= min_conf < 1) or not (0 < max_conf <= 1):
                raise ValueError(
                    f"confidence_range values must be in [0, 1), got ({min_conf}, {max_conf})"
                )
            if max_conf <= min_conf:
                raise ValueError("confidence_range maximum must be greater than minimum")
            
            # Convert to std_range (handle min_conf=0 special case)
            min_std = 0.0 if min_conf == 0.0 else self._confidence_to_std_radius(min_conf)
            max_std = self._confidence_to_std_radius(max_conf)
            std_range = (min_std, max_std)

        # ===============================================================
        # 6. Validate center point constraints
        # ===============================================================
        if (center_point is None) != (center_radius is None):
            raise ValueError("center_point and center_radius must be specified together")
        
        if center_point is not None:
            center_point = center_point.to(self.device)
            if center_point.dim() != 1 or center_point.size(0) != self.n_features:
                raise ValueError(
                    f"center_point must be a 1D tensor with {self.n_features} features, "
                    f"got shape {center_point.shape}"
                )
            if center_radius <= 0:
                raise ValueError(f"center_radius must be positive, got {center_radius}")

        # ===============================================================
        # 7. Select component indices
        # ===============================================================
        if component is not None:
            # Validate component index. Uses self.weights_.shape[0] (the
            # actual number of fitted slots) rather than self.n_components
            # directly: the latter is None in Gibbs mode, and this must
            # work whether the model was EM- or Gibbs-fit.
            n_slots = self.weights_.shape[0]
            if not (0 <= component < n_slots):
                raise ValueError(
                    f"component must be between 0 and {n_slots - 1}, got {component}"
                )
            # Sample only from specified component
            indices = torch.full((n_samples,), component, dtype=torch.long, device=self.device)
        else:
            # Choose components according to mixture weights
            indices = torch.multinomial(self.weights_, n_samples, replacement=True)
        
        # ===============================================================
        # 8a. Standard sampling (no distance constraints)
        # ===============================================================
        if std_radius is None and std_range is None and center_point is None:
            means = self.means_[indices]
            samples = self._sample_affine(means, indices)
            return samples, indices
        
        # ===============================================================
        # 8b. Rejection sampling (with distance constraints)
        # ===============================================================
        else:
            # Define distance check function and description
            if std_radius is not None:
                if std_radius <= 0:
                    raise ValueError("std_radius must be positive")
                distance_check = lambda sample, comp_idx: (
                    self._compute_mahalanobis_distance(
                        sample.unsqueeze(0), 
                        torch.tensor([comp_idx], device=self.device)
                    ).item() <= std_radius
                )
                constraint_desc = f"within {std_radius} standard deviations"
                
            elif std_range is not None:
                min_std, max_std = std_range
                distance_check = lambda sample, comp_idx: (
                    min_std <= self._compute_mahalanobis_distance(
                        sample.unsqueeze(0), 
                        torch.tensor([comp_idx], device=self.device)
                    ).item() <= max_std
                )
                if max_std == float('inf'):
                    constraint_desc = f"further than {min_std} standard deviations"
                else:
                    constraint_desc = f"between {min_std} and {max_std} standard deviations"
                    
            else:  # center_point is not None
                distance_check = lambda sample, comp_idx: (
                    torch.norm(sample - center_point).item() <= center_radius
                )
                constraint_desc = f"within {center_radius} units from center point"
            
            # Perform rejection sampling
            valid_samples = []
            valid_indices = []
            
            for i in range(n_samples):
                comp_idx = indices[i].item()
                attempts = 0
                
                while attempts < max_attempts_per_sample:
                    # Generate candidate sample
                    mean = self.means_[comp_idx:comp_idx+1]  # Keep batch dimension
                    cov = self._build_covariances_for_sampling(
                        torch.tensor([comp_idx], device=self.device), 1
                    )
                    sample = MultivariateNormal(mean, covariance_matrix=cov).sample()
                    
                    # Accept if constraint satisfied
                    if distance_check(sample.squeeze(0), comp_idx):
                        valid_samples.append(sample.squeeze(0))
                        valid_indices.append(comp_idx)
                        break
                    
                    attempts += 1
                
                # Raise error if max attempts exceeded
                if attempts >= max_attempts_per_sample:
                    raise RuntimeError(
                        f"Could not generate sample {i+1} {constraint_desc} "
                        f"after {max_attempts_per_sample} attempts. "
                        f"Constraint is too restrictive. Try increasing "
                        f"max_attempts_per_sample or relaxing the constraint."
                    )
            
            # Stack results
            samples = torch.stack(valid_samples)
            indices = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
            return samples, indices

    # ===================================================================
    # Helper Methods for Sampling
    # ===================================================================
    
    def _build_covariances_for_sampling(self, indices, n_samples):
        r"""
        Construct batch of full covariance matrices for sampling.

        Converts the stored covariance format (which may be tied, diagonal, or spherical)
        into full (n_features × n_features) covariance matrices suitable for sampling.

        Parameters
        ----------
        indices : torch.Tensor
            Component indices for each sample, shape (n_samples,).
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        covs : torch.Tensor
            Full covariance matrices, shape (n_samples, n_features, n_features).
        """
        # ===============================================================
        # Component-specific covariances
        # ===============================================================
        if self.covariance_type == 'full':
            # Already in full format
            return self.covariances_[indices]

        elif self.covariance_type == 'diag':
            # Convert diagonal to full matrix (zeros off-diagonal)
            return torch.diag_embed(self.covariances_[indices])

        elif self.covariance_type == 'spherical':
            # σ²·I for each component
            eye = torch.eye(self.n_features, device=self.device, dtype=self.dtype).unsqueeze(0)
            return eye * self.covariances_[indices].view(-1, 1, 1)

        # ===============================================================
        # Tied (shared) covariances
        # ===============================================================
        elif self.covariance_type == 'tied_full':
            # Same full matrix for all samples
            return self.covariances_.unsqueeze(0).expand(n_samples, -1, -1)

        elif self.covariance_type == 'tied_diag':
            # Same diagonal for all samples
            diag_mat = torch.diag_embed(self.covariances_)
            return diag_mat.unsqueeze(0).expand(n_samples, -1, -1)

        elif self.covariance_type == 'tied_spherical':
            # Same σ²·I for all samples
            eye = torch.eye(self.n_features, device=self.device, dtype=self.dtype).unsqueeze(0)
            return eye * self.covariances_

        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def _sample_affine(self, means: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        r"""
        Draw samples for non-'full' covariance types via a direct affine
        transform of standard normal noise, instead of constructing a
        MultivariateNormal with a covariance matrix that must be broadcast
        across samples.

        MultivariateNormal(loc, covariance_matrix=cov) densifies `cov` to a
        full (n_samples, n_features, n_features) tensor whenever it has to
        broadcast a covariance across the batch -- for large n_features and
        n_samples this exhausts memory (GitHub issue #1). Even 'full' only
        has n_components distinct covariance matrices, so it is handled here
        too by grouping samples per component instead of gathering into an
        (n_samples, n_features, n_features) tensor.

        Parameters
        ----------
        means : torch.Tensor
            Per-sample component means, shape (n_samples, n_features).
        indices : torch.Tensor
            Component index for each sample, shape (n_samples,).

        Returns
        -------
        samples : torch.Tensor
            Generated samples, shape (n_samples, n_features).
        """
        z = torch.randn(means.shape, device=self.device, dtype=self.dtype)

        if self.covariance_type == 'full':
            samples = torch.empty_like(means)
            for comp in torch.unique(indices).tolist():
                mask = indices == comp
                L = torch.linalg.cholesky(self.covariances_[comp])
                samples[mask] = means[mask] + z[mask] @ L.T
            return samples

        elif self.covariance_type == 'diag':
            std = self.covariances_[indices].sqrt()
            return means + z * std

        elif self.covariance_type == 'spherical':
            std = self.covariances_[indices].sqrt().unsqueeze(-1)
            return means + z * std

        elif self.covariance_type == 'tied_full':
            L = torch.linalg.cholesky(self.covariances_)
            return means + z @ L.T

        elif self.covariance_type == 'tied_diag':
            std = self.covariances_.sqrt()
            return means + z * std

        elif self.covariance_type == 'tied_spherical':
            std = self.covariances_.sqrt()
            return means + z * std

        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def _sample_component(self, k: int, n: int) -> torch.Tensor:
        r"""
        Draw `n` samples from component `k` alone -- the abstract-method
        contract `BaseMixture.sample()` needs. `GaussianMixture.sample()`
        does not call this (its own implementation is far richer --
        confidence regions, rejection sampling, `std_range`, etc. -- see
        `sample`'s own docstring); this exists only so the abstract
        contract is satisfiable for a `GaussianMixture` instance used
        generically.
        """
        mean = self.means_[k]
        if self.covariance_type == 'full' or self.covariance_type_ in _NEW_GPCM_CODES:
            cov = self.covariances_[k]
        elif self.covariance_type == 'diag':
            cov = torch.diag_embed(self.covariances_[k])
        elif self.covariance_type == 'spherical':
            cov = torch.eye(self.n_features, device=self.device, dtype=self.dtype) * self.covariances_[k]
        elif self.covariance_type == 'tied_full':
            cov = self.covariances_
        elif self.covariance_type == 'tied_diag':
            cov = torch.diag_embed(self.covariances_)
        elif self.covariance_type == 'tied_spherical':
            cov = torch.eye(self.n_features, device=self.device, dtype=self.dtype) * self.covariances_
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

        L = torch.linalg.cholesky(cov)
        z = torch.randn(n, self.n_features, device=self.device, dtype=self.dtype)
        return mean.unsqueeze(0) + z @ L.T

    def _confidence_to_std_radius(self, confidence: float) -> float:
        r"""
        Convert confidence level to standard deviation radius.
        
        For multivariate Gaussian, the squared Mahalanobis distance follows χ²
        distribution with d degrees of freedom. This finds the radius containing
        the specified probability mass.
        
        For example:
        - confidence=0.95 in 2D → radius ≈ 2.45σ (95% of points within ellipse)
        - confidence=0.68 in 2D → radius ≈ 1.51σ (68% of points within ellipse)
        
        Parameters
        ----------
        confidence : float
            Confidence level in (0, 1), e.g., 0.95 for 95%.
            
        Returns
        -------
        std_radius : float
            Standard deviation radius: sqrt(χ²_d(confidence))
        """
        if not (0 < confidence < 1):
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

        # χ² quantile for given confidence and dimensionality
        chi2_quantile = chi2.ppf(confidence, df=self.n_features)
        return math.sqrt(chi2_quantile)

    def _compute_mahalanobis_distance(self, samples: torch.Tensor, component_indices: torch.Tensor) -> torch.Tensor:
        r"""
        Compute Mahalanobis distance from samples to component means.
        
        Mahalanobis distance accounts for covariance structure:
        d²(x, μ) = (x - μ)ᵀ Σ⁻¹ (x - μ)
        
        Parameters
        ----------
        samples : torch.Tensor
            Sample points, shape (n_samples, n_features).
        component_indices : torch.Tensor
            Component for each sample, shape (n_samples,).
            
        Returns
        -------
        distances : torch.Tensor
            Mahalanobis distances, shape (n_samples,).
            Distance of 1.0 = one standard deviation from mean.
        """
        # ===============================================================
        # Compute deviations from component means
        # ===============================================================
        means = self.means_[component_indices]  # (n_samples, n_features)
        diff = samples - means  # (n_samples, n_features)
        
        # ===============================================================
        # Compute d² = diff^T Σ⁻¹ diff (depends on covariance type)
        # ===============================================================
        if self.covariance_type == 'full':
            # Component-specific full covariances
            covs = self.covariances_[component_indices]  # (n_samples, n_features, n_features)
            diff_expanded = diff.unsqueeze(-1)  # (n_samples, n_features, 1)
            try:
                inv_covs = torch.inverse(covs)
            except RuntimeError:
                # Handle singular matrices
                inv_covs = torch.pinverse(covs)
            mahal_sq = torch.bmm(torch.bmm(diff.unsqueeze(1), inv_covs), diff_expanded).squeeze()
            
        elif self.covariance_type == 'diag':
            # Component-specific diagonal: d² = Σᵢ (xᵢ - μᵢ)² / σᵢ²
            vars = self.covariances_[component_indices]  # (n_samples, n_features)
            mahal_sq = torch.sum(diff.pow(2) / vars, dim=1)
            
        elif self.covariance_type == 'spherical':
            # Component-specific spherical: d² = ||x - μ||² / σ²
            vars = self.covariances_[component_indices]  # (n_samples,)
            mahal_sq = torch.sum(diff.pow(2), dim=1) / vars
            
        elif self.covariance_type == 'tied_full':
            # Shared full covariance
            try:
                inv_cov = torch.inverse(self.covariances_)
            except RuntimeError:
                inv_cov = torch.pinverse(self.covariances_)
            mahal_sq = torch.sum(diff * torch.matmul(diff, inv_cov), dim=1)
            
        elif self.covariance_type == 'tied_diag':
            # Shared diagonal covariance
            mahal_sq = torch.sum(diff.pow(2) / self.covariances_, dim=1)
            
        elif self.covariance_type == 'tied_spherical':
            # Shared spherical covariance
            mahal_sq = torch.sum(diff.pow(2), dim=1) / self.covariances_
            
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")
            
        return torch.sqrt(mahal_sq)

    # ===================================================================
    # Model Persistence (Save/Load)
    # ===================================================================
    
    def save(self, filepath: str):
        r"""
        Save GMM model to disk.

        Saves all model parameters, configuration, training state, and priors
        to a PyTorch file that can be reloaded later.

        Parameters
        ----------
        filepath : str
            Path where to save the model. Typically ends with '.pth' or '.pt'.
            
        See Also
        --------
        load : Class method to load a saved model.
        save_state_dict : Get state dictionary without saving to file.
        """
        state_dict = {
            # ===============================================================
            # Model parameters (fitted)
            # ===============================================================
            'weights_': self.weights_,
            'means_': self.means_,
            'covariances_': self.covariances_,
            'volume_': self.volume_,
            'shape_': self.shape_,
            'orientation_': self.orientation_,
            
            # ===============================================================
            # Initial parameters (for reproducibility)
            # ===============================================================
            'initial_weights_': self.initial_weights_,
            'initial_means_': self.initial_means_,
            'initial_covariances_': self.initial_covariances_,
            
            # ===============================================================
            # Model configuration
            # ===============================================================
            'n_components': self.n_components,
            'n_features': self.n_features,
            'covariance_type': self.covariance_type,
            'covariance_type_': self.covariance_type_,
            'tol': self.tol,
            'reg_covar': self.reg_covar,
            'gpcm_tol': self.gpcm_tol,
            'gpcm_max_iter': self.gpcm_max_iter,
            'max_iter': self.max_iter,
            'init_means': self.init_means,
            'init_weights': self.init_weights,
            'init_covariances': self.init_covariances,
            'n_init': self.n_init,
            'random_state': self.random_state,
            'warm_start': self.warm_start,
            'verbose': self.verbose,
            'verbose_interval': self.verbose_interval,
            'cem': self.cem,
            'algorithm': self.algorithm,
            'batch_size': self.batch_size,
            'convergence': self.convergence,
            'temperature': self.temperature,
            'step_size': self.step_size,
            'sem_burn_in': self.sem_burn_in,
            'sem_estimate': self.sem_estimate,
            'max_components': self.max_components,
            'alpha': self.alpha,
            'burn_in': self.burn_in,
            'weight_threshold': self.weight_threshold,
            'init_k': self.init_k,
            'noise': self.noise,
            'noise_volume': self.noise_volume,
            'noise_prior': self.noise_prior,
            'noise_weight_': self.noise_weight_,
            'noise_bounds_': self.noise_bounds_,
            'missing': self.missing,
            'frozen_components': list(self.frozen_components),

            # ===============================================================
            # Training state
            # ===============================================================
            'fitted_': self.fitted_,
            'converged_': self.converged_,
            'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_,
            'n_components_': self.n_components_,
            'active_': self.active_,

            # ===============================================================
            # Prior settings
            # ===============================================================
            'use_weight_prior': self.use_weight_prior,
            'use_mean_prior': self.use_mean_prior,
            'use_covariance_prior': self.use_covariance_prior,
            'weight_concentration_prior': self.weight_concentration_prior,
            'mean_prior': self.mean_prior,
            'mean_precision_prior': self.mean_precision_prior,
            'covariance_prior': self.covariance_prior,
            'degrees_of_freedom_prior': self.degrees_of_freedom_prior,
        }

        torch.save(state_dict, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = None) -> "GaussianMixture":
        r"""
        Load GMM model from disk.

        Creates a new GaussianMixture instance with all parameters and state
        restored from a saved file.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.
        device : str, optional
            Device to load model tensors on ('cpu' or 'cuda'). 
            If None, uses device from saved model or defaults to GPU if available.

        Returns
        -------
        model : GaussianMixture
            The loaded GMM model, ready for prediction or continued training.
            
        Notes
        -----
        Handles backward compatibility with older saved models that used
        deprecated parameter names like 'init_params' and 'cov_init_method'.
        
        See Also
        --------
        save : Save a model to disk.
        load_state_dict : Load from a state dictionary.
        """
        # ===============================================================
        # Load state dictionary from file
        # ===============================================================
        if device is None:
            state_dict = torch.load(filepath, weights_only=False)
        else:
            state_dict = torch.load(filepath, map_location=device, weights_only=False)
        
        # ===============================================================
        # Handle backward compatibility
        # ===============================================================
        if 'init_params' in state_dict and 'init_means' not in state_dict:
            state_dict['init_means'] = state_dict['init_params']
        if 'cov_init_method' in state_dict and 'init_covariances' not in state_dict:
            state_dict['init_covariances'] = state_dict['cov_init_method']
        if 'init_weights' not in state_dict:
            state_dict['init_weights'] = 'uniform'
        
        # ===============================================================
        # Create new instance with saved configuration
        # ===============================================================
        model = cls(
            n_components=state_dict['n_components'],
            n_features=state_dict['n_features'],
            covariance_type=state_dict['covariance_type'],
            tol=state_dict['tol'],
            reg_covar=state_dict['reg_covar'],
            gpcm_tol=state_dict.get('gpcm_tol', 1e-6),
            gpcm_max_iter=state_dict.get('gpcm_max_iter', 20),
            max_iter=state_dict['max_iter'],
            init_means=state_dict['init_means'],
            init_weights=state_dict['init_weights'],
            init_covariances=state_dict['init_covariances'],
            n_init=state_dict['n_init'],
            random_state=state_dict['random_state'],
            warm_start=state_dict['warm_start'],
            verbose=state_dict['verbose'],
            verbose_interval=state_dict['verbose_interval'],
            device=device,
            weight_concentration_prior=state_dict['weight_concentration_prior'],
            mean_prior=state_dict['mean_prior'],
            mean_precision_prior=state_dict['mean_precision_prior'],
            covariance_prior=state_dict['covariance_prior'],
            degrees_of_freedom_prior=state_dict['degrees_of_freedom_prior'],
            # Prefer the new 'algorithm' key when present; fall back to
            # deriving it from the legacy 'cem' boolean for files saved
            # before Phase 0. Passing `algorithm=` directly (rather than
            # `cem=`) avoids re-raising the cem=True DeprecationWarning on
            # every load of an already-cem model.
            algorithm=state_dict.get(
                'algorithm', 'cem' if state_dict.get('cem', False) else 'em'
            ),
            batch_size=state_dict.get('batch_size'),
            convergence=state_dict.get('convergence', 'relative'),
            temperature=state_dict.get('temperature'),
            step_size=state_dict.get('step_size'),
            sem_burn_in=state_dict.get('sem_burn_in'),
            sem_estimate=state_dict.get('sem_estimate', 'best'),
            max_components=state_dict.get('max_components', 20),
            alpha=state_dict.get('alpha', 1.0),
            burn_in=state_dict.get('burn_in'),
            weight_threshold=state_dict.get('weight_threshold'),
            init_k=state_dict.get('init_k'),
            noise=state_dict.get('noise', False),
            noise_volume=state_dict.get('noise_volume', 'bbox'),
            noise_prior=state_dict.get('noise_prior', 0.05),
            missing=state_dict.get('missing', 'error'),
            frozen_components=state_dict.get('frozen_components', ()),
        )
        model.noise_weight_ = state_dict.get('noise_weight_')
        model.noise_bounds_ = state_dict.get('noise_bounds_')

        # ===============================================================
        # Load trained parameters and state
        # ===============================================================
        model.weights_ = state_dict['weights_']
        model.means_ = state_dict['means_']
        model.covariances_ = state_dict['covariances_']
        model.volume_ = state_dict.get('volume_')
        model.shape_ = state_dict.get('shape_')
        model.orientation_ = state_dict.get('orientation_')
        model.dtype = model.means_.dtype
        model.initial_weights_ = state_dict['initial_weights_']
        model.initial_means_ = state_dict['initial_means_']
        model.initial_covariances_ = state_dict['initial_covariances_']
        model.fitted_ = state_dict['fitted_']
        model.converged_ = state_dict['converged_']
        model.n_iter_ = state_dict['n_iter_']
        model.lower_bound_ = state_dict['lower_bound_']
        model.n_components_ = state_dict.get('n_components_', state_dict['n_components'])
        model.active_ = state_dict.get('active_')
        if model.active_ is None and model.n_components_ is not None:
            model.active_ = torch.ones(model.n_components_, dtype=torch.bool, device=model.device)

        # ===============================================================
        # Load prior flags
        # ===============================================================
        model.use_weight_prior = state_dict['use_weight_prior']
        model.use_mean_prior = state_dict['use_mean_prior']
        model.use_covariance_prior = state_dict['use_covariance_prior']
        
        return model

    def save_state_dict(self) -> dict:
        r"""
        Get model state as a dictionary (PyTorch-style).
        
        Useful for custom saving/loading workflows or integration with
        other PyTorch models.

        Returns
        -------
        state_dict : dict
            Dictionary containing all model parameters, configuration,
            and training state.
            
        See Also
        --------
        load_state_dict : Load from a state dictionary.
        to_dict : Get a simplified dictionary (basic parameters only).
        save : Save directly to a file.
        """
        return {
            # ===============================================================
            # Model parameters
            # ===============================================================
            'weights_': self.weights_,
            'means_': self.means_,
            'covariances_': self.covariances_,
            'volume_': self.volume_,
            'shape_': self.shape_,
            'orientation_': self.orientation_,
            
            # ===============================================================
            # Initial parameters
            # ===============================================================
            'initial_weights_': self.initial_weights_,
            'initial_means_': self.initial_means_,
            'initial_covariances_': self.initial_covariances_,
            
            # ===============================================================
            # Model configuration
            # ===============================================================
            'n_components': self.n_components,
            'n_features': self.n_features,
            'covariance_type': self.covariance_type,
            'covariance_type_': self.covariance_type_,
            'tol': self.tol,
            'reg_covar': self.reg_covar,
            'gpcm_tol': self.gpcm_tol,
            'gpcm_max_iter': self.gpcm_max_iter,
            'max_iter': self.max_iter,
            'init_means': self.init_means,
            'init_weights': self.init_weights,
            'init_covariances': self.init_covariances,
            'n_init': self.n_init,
            'random_state': self.random_state,
            'warm_start': self.warm_start,
            'verbose': self.verbose,
            'verbose_interval': self.verbose_interval,
            'cem': self.cem,
            'algorithm': self.algorithm,
            'batch_size': self.batch_size,
            'convergence': self.convergence,
            'temperature': self.temperature,
            'step_size': self.step_size,
            'sem_burn_in': self.sem_burn_in,
            'sem_estimate': self.sem_estimate,
            'max_components': self.max_components,
            'alpha': self.alpha,
            'burn_in': self.burn_in,
            'weight_threshold': self.weight_threshold,
            'init_k': self.init_k,
            'noise': self.noise,
            'noise_volume': self.noise_volume,
            'noise_prior': self.noise_prior,
            'noise_weight_': self.noise_weight_,
            'noise_bounds_': self.noise_bounds_,
            'missing': self.missing,
            'frozen_components': list(self.frozen_components),

            # ===============================================================
            # Training state
            # ===============================================================
            'fitted_': self.fitted_,
            'converged_': self.converged_,
            'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_,
            'n_components_': self.n_components_,
            'active_': self.active_,

            # ===============================================================
            # Prior settings
            # ===============================================================
            'use_weight_prior': self.use_weight_prior,
            'use_mean_prior': self.use_mean_prior,
            'use_covariance_prior': self.use_covariance_prior,
            'weight_concentration_prior': self.weight_concentration_prior,
            'mean_prior': self.mean_prior,
            'mean_precision_prior': self.mean_precision_prior,
            'covariance_prior': self.covariance_prior,
            'degrees_of_freedom_prior': self.degrees_of_freedom_prior,
        }

    def load_state_dict(self, state_dict: dict):
        r"""
        Load model state from a dictionary (PyTorch-style).

        Updates the current model instance with parameters and configuration
        from the state dictionary.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing model parameters and metadata.
            Typically obtained from save_state_dict().
            
        Warnings
        --------
        Issues warning if loading a state dict with different n_components
        than the current model.
        
        Notes
        -----
        Handles backward compatibility with older state dictionaries.
        
        See Also
        --------
        save_state_dict : Get state dictionary from current model.
        load : Load model from file (creates new instance).
        """
        # ===============================================================
        # Validate compatibility
        # ===============================================================
        if hasattr(self, 'n_components') and self.n_components != state_dict['n_components']:
            warnings.warn(
                f"n_components mismatch: current={self.n_components}, "
                f"loaded={state_dict['n_components']}"
            )
        
        # ===============================================================
        # Handle backward compatibility
        # ===============================================================
        if 'init_params' in state_dict and 'init_means' not in state_dict:
            state_dict['init_means'] = state_dict['init_params']
        if 'cov_init_method' in state_dict and 'init_covariances' not in state_dict:
            state_dict['init_covariances'] = state_dict['cov_init_method']
        if 'init_weights' not in state_dict:
            state_dict['init_weights'] = 'uniform'
        
        # ===============================================================
        # Update configuration
        # ===============================================================
        self.n_components = state_dict['n_components']
        self.n_features = state_dict['n_features']
        self.covariance_type = state_dict['covariance_type']
        self.covariance_type_ = state_dict.get(
            'covariance_type_', _resolve_covariance_type(state_dict['covariance_type'])[1]
        )
        self.tol = state_dict['tol']
        self.reg_covar = state_dict['reg_covar']
        self.gpcm_tol = state_dict.get('gpcm_tol', 1e-6)
        self.gpcm_max_iter = state_dict.get('gpcm_max_iter', 20)
        self.max_iter = state_dict['max_iter']
        self.init_means = state_dict['init_means']
        self.init_weights = state_dict['init_weights']
        self.init_covariances = state_dict['init_covariances']
        self.n_init = state_dict['n_init']
        self.random_state = state_dict['random_state']
        self.warm_start = state_dict['warm_start']
        self.verbose = state_dict['verbose']
        self.verbose_interval = state_dict['verbose_interval']
        self.cem = state_dict['cem']
        self.algorithm = state_dict.get(
            'algorithm', 'cem' if state_dict.get('cem', False) else 'em'
        )
        self.batch_size = state_dict.get('batch_size')
        self.convergence = state_dict.get('convergence', 'relative')
        self.temperature = state_dict.get('temperature')
        self.step_size = state_dict.get('step_size')
        self.sem_burn_in = state_dict.get('sem_burn_in')
        self.sem_estimate = state_dict.get('sem_estimate', 'best')
        self.max_components = state_dict.get('max_components', 20)
        self.alpha = state_dict.get('alpha', 1.0)
        self.burn_in = state_dict.get('burn_in')
        self.weight_threshold = state_dict.get('weight_threshold')
        self.init_k = state_dict.get('init_k')
        self.noise = state_dict.get('noise', False)
        self.noise_volume = state_dict.get('noise_volume', 'bbox')
        self.noise_prior = state_dict.get('noise_prior', 0.05)
        self.missing = state_dict.get('missing', 'error')
        self.frozen_components = state_dict.get('frozen_components', ())
        self.noise_weight_ = state_dict.get('noise_weight_')
        self.noise_bounds_ = state_dict.get('noise_bounds_')

        # ===============================================================
        # Load parameters
        # ===============================================================
        self.weights_ = state_dict['weights_']
        self.means_ = state_dict['means_']
        self.covariances_ = state_dict['covariances_']
        self.volume_ = state_dict.get('volume_')
        self.shape_ = state_dict.get('shape_')
        self.orientation_ = state_dict.get('orientation_')
        self._gpcm_state_ = None  # not persisted; re-derived from scratch on the next M-step if continued
        self.dtype = self.means_.dtype
        self.initial_weights_ = state_dict['initial_weights_']
        self.initial_means_ = state_dict['initial_means_']
        self.initial_covariances_ = state_dict['initial_covariances_']

        # ===============================================================
        # Load training state
        # ===============================================================
        self.fitted_ = state_dict['fitted_']
        self.converged_ = state_dict['converged_']
        self.n_iter_ = state_dict['n_iter_']
        self.lower_bound_ = state_dict['lower_bound_']
        self.n_components_ = state_dict.get('n_components_', state_dict['n_components'])
        self.active_ = state_dict.get('active_')
        if self.active_ is None and self.n_components_ is not None:
            self.active_ = torch.ones(self.n_components_, dtype=torch.bool, device=self.device)

        # ===============================================================
        # Load prior settings
        # ===============================================================
        self.use_weight_prior = state_dict['use_weight_prior']
        self.use_mean_prior = state_dict['use_mean_prior']
        self.use_covariance_prior = state_dict['use_covariance_prior']
        self.weight_concentration_prior = state_dict['weight_concentration_prior']
        self.mean_prior = state_dict['mean_prior']
        self.mean_precision_prior = state_dict['mean_precision_prior']
        self.covariance_prior = state_dict['covariance_prior']
        self.degrees_of_freedom_prior = state_dict['degrees_of_freedom_prior']

    def to_dict(self) -> dict:
        r"""
        Alias for save_state_dict() for convenience.
        
        Returns the state dictionary of the model.

        Returns
        -------
        state_dict : dict
            Dictionary containing model parameters and metadata.
        """
        return self.save_state_dict()