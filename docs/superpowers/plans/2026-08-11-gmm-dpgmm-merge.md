# Merge DPGMM into GaussianMixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `tgmm.DPGMM`'s collapsed-Gibbs-sampling fitting into `tgmm.GaussianMixture`, selected by `n_components=None`, and remove `DPGMM` as a public class.

**Architecture:** `GaussianMixture.__init__` gains Gibbs-mode constructor params (`max_components`, `alpha`, `burn_in`, `weight_threshold`, `init_k`) and validates eagerly (reject `tied_*` covariance in Gibbs mode; require all four NIW prior args in Gibbs mode, raising `ValueError` immediately rather than auto-resolving a default). `fit()` branches once at the top on `self.n_components is None` to either the existing EM/CEM path (untouched) or ported Gibbs internals. Everything downstream of fitting (`predict`, `predict_proba`, `score`, `score_samples`, `sample`, `bic`, `aic`) needs no mode-awareness — it already only reads `weights_`/`means_`/`covariances_`/`covariance_type`, which is unchanged after this merge. `n_components_`/`active_` become set for EM-mode fits too (previously Gibbs-only), so `_n_parameters()` collapses to one implementation.

**Tech Stack:** Python, PyTorch, pytest, Jupyter/nbconvert.

**Reference:** `docs/superpowers/specs/2026-08-11-gmm-dpgmm-merge-design.md` (approved design this plan implements). Original Gibbs math: `docs/superpowers/specs/2026-07-28-dpgmm-design.md`.

## Global Constraints

- EM/CEM mode (`n_components` an int) must have **zero behavior change** — the full existing `test_gmm.py` (and other non-DPGMM test files) suite must stay green after every task.
- Gibbs mode (`n_components=None`) **requires all four NIW prior args explicit** — no auto-resolved default. `__init__` raises `ValueError` immediately if any of `mean_prior`/`mean_precision_prior`/`covariance_prior`/`degrees_of_freedom_prior` is `None` while `n_components is None`.
- `tied_full`/`tied_diag`/`tied_spherical` + `n_components=None` raises `NotImplementedError` at `__init__` (not `fit()`).
- Ported Gibbs sampling math (NIW posteriors, Student-t marginal likelihoods, CRP weight rules) must not change numerically — this plan ports already-tested code, it does not re-derive it. Where a step says "copy verbatim," the copied code must be byte-identical modulo the specific renames called out.
- Every code-writing step runs its test before being considered done. `source .venv/bin/activate` before any `pytest`/`python` invocation (this repo's venv).

---

## File Structure

- **Modify:** `tgmm/gmm.py` — constructor, `fit()` dispatch, ported Gibbs internals, `suggest_priors`, attribute unification, `_n_parameters`, `save`/`load`/`save_state_dict`/`load_state_dict`. Grows from ~3,307 to ~3,900 lines.
- **Delete:** `tgmm/dpgmm.py`.
- **Modify:** `tgmm/__init__.py` — drop the `DPGMM` export.
- **Modify:** `tests/test_dpgmm.py`, `tests/test_dpgmm_gibbs.py`, `tests/test_dpgmm_gibbs_unbounded.py`, `tests/test_dpgmm_covariance_calibration.py` — rewritten for the merged API, same filenames (git history continuity).
- **Modify:** `notebooks/dpgmm.ipynb`, `notebooks/dpgmm_gibbs_sampling.ipynb` — every `DPGMM(...)` call site reworked to `GaussianMixture(n_components=None, ...)` with explicit priors via `suggest_priors`, then fully re-executed.
- **Modify:** `docs/superpowers/specs/2026-07-28-dpgmm-design.md` — superseded-by pointer added.
- **Modify:** `README.md`, `docs/user-guide/clustering-metrics.md` — minor text fixes referencing `DPGMM` as a separate class.

---

### Task 1: Constructor changes — Gibbs-mode params, eager validation, `suggest_priors`

**Files:**
- Modify: `tgmm/gmm.py:132-283` (`__init__`)
- Test: `tests/test_gmm.py` (append new tests)

**Interfaces:**
- Produces: `GaussianMixture.__init__` accepts `n_components: Optional[int] = None` (was `int = 1`), plus new `max_components: Optional[int] = 20`, `alpha: float = 1.0`, `burn_in: Optional[int] = None`, `weight_threshold: Optional[float] = None`, `init_k: Optional[int] = None`. Raises `ValueError`/`NotImplementedError` at construction per the rules below. Produces `GaussianMixture.suggest_priors(X, n_components, covariance_type="full")` (`@staticmethod`) returning `(mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior)`.

- [ ] **Step 1: Write the failing constructor-validation tests**

Append to `tests/test_gmm.py` (find the file's existing `# ============...` section markers and add a new section near the other construction/validation tests, e.g. after `test_invalid_covariance_type_raises` around line 502):

```python
# ============================================================================
# Gibbs mode (n_components=None) construction
# ============================================================================

def test_gibbs_mode_requires_all_four_priors():
    with pytest.raises(ValueError, match="mean_prior"):
        GaussianMixture(n_components=None, mean_precision_prior=0.1,
                         covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)


def test_gibbs_mode_requires_mean_precision_prior():
    with pytest.raises(ValueError, match="mean_precision_prior"):
        GaussianMixture(n_components=None, mean_prior=torch.zeros(2),
                         covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)


def test_gibbs_mode_requires_covariance_prior():
    with pytest.raises(ValueError, match="covariance_prior"):
        GaussianMixture(n_components=None, mean_prior=torch.zeros(2),
                         mean_precision_prior=0.1, degrees_of_freedom_prior=4.0)


def test_gibbs_mode_requires_degrees_of_freedom_prior():
    with pytest.raises(ValueError, match="degrees_of_freedom_prior"):
        GaussianMixture(n_components=None, mean_prior=torch.zeros(2),
                         mean_precision_prior=0.1, covariance_prior=torch.eye(2))


def test_gibbs_mode_error_message_lists_all_missing_priors():
    with pytest.raises(ValueError, match="mean_prior.*mean_precision_prior.*covariance_prior.*degrees_of_freedom_prior"):
        GaussianMixture(n_components=None)


def test_gibbs_mode_with_all_priors_constructs_successfully():
    model = GaussianMixture(n_components=None, mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                             covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
    assert model.n_components is None
    assert model.max_components == 20  # default


@pytest.mark.parametrize("covariance_type", ["tied_full", "tied_diag", "tied_spherical"])
def test_gibbs_mode_rejects_tied_covariance_at_construction(covariance_type):
    # Must raise at __init__, before any prior is even checked -- so no
    # prior args are needed to trigger this.
    with pytest.raises(NotImplementedError, match="tied"):
        GaussianMixture(n_components=None, covariance_type=covariance_type)


def test_em_mode_unaffected_by_gibbs_validation():
    # n_components=K (EM mode) must still work with zero priors -- plain MLE,
    # exactly as today. This is the regression guard for "no behavior change."
    model = GaussianMixture(n_components=3)
    assert model.n_components == 3
    assert model.use_mean_prior is False
    assert model.use_covariance_prior is False


def test_em_mode_silently_accepts_gibbs_only_params():
    # Gibbs-only params (max_components, alpha, burn_in, weight_threshold,
    # init_k) must not raise when passed alongside a fixed n_components --
    # they're simply unused by the EM path, not an error condition.
    model = GaussianMixture(n_components=3, max_components=99, alpha=2.0, burn_in=5,
                             weight_threshold=0.5, init_k=10)
    model.fit(X_3D)
    assert model.fitted_
    assert model.n_components_ == 3


def test_gibbs_mode_silently_accepts_em_only_params():
    # EM-only params (n_init, warm_start) must not raise when passed
    # alongside n_components=None -- simply unused by the Gibbs path.
    model = GaussianMixture(n_components=None, max_components=4, n_init=5, warm_start=True,
                             mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                             covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
    assert model.n_components is None


def test_gibbs_mode_new_params_stored():
    model = GaussianMixture(n_components=None, max_components=15, alpha=2.0, burn_in=5,
                             weight_threshold=0.5, init_k=10,
                             mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                             covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
    assert model.max_components == 15
    assert model.alpha == 2.0
    assert model.burn_in == 5
    assert model.weight_threshold == 0.5
    assert model.init_k == 10


# ============================================================================
# suggest_priors
# ============================================================================

def test_suggest_priors_returns_four_element_tuple():
    X = torch.randn(100, 3, dtype=torch.float64)
    result = GaussianMixture.suggest_priors(X, n_components=4)
    assert len(result) == 4
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = result
    assert mean_prior.shape == (3,)
    assert isinstance(mean_precision_prior, float)
    assert covariance_prior.shape == (3, 3)  # covariance_type='full' default
    assert degrees_of_freedom_prior == pytest.approx(5.0)  # n_features + 2


def test_suggest_priors_covariance_type_diag_shape():
    X = torch.randn(100, 3, dtype=torch.float64)
    _, _, covariance_prior, _ = GaussianMixture.suggest_priors(X, n_components=4, covariance_type="diag")
    assert covariance_prior.shape == (3,)


def test_suggest_priors_covariance_type_spherical_shape():
    X = torch.randn(100, 3, dtype=torch.float64)
    _, _, covariance_prior, _ = GaussianMixture.suggest_priors(X, n_components=4, covariance_type="spherical")
    assert covariance_prior.shape == ()


def test_suggest_priors_result_is_directly_usable():
    # The whole point: pass suggest_priors' output straight into the
    # constructor without any reshaping.
    torch.manual_seed(0)
    X = torch.cat([torch.randn(50, 2, dtype=torch.float64) - 5.0,
                    torch.randn(50, 2, dtype=torch.float64) + 5.0])
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=2)
    model = GaussianMixture(n_components=None, max_components=5, mean_prior=mean_prior,
                             mean_precision_prior=mean_precision_prior, covariance_prior=covariance_prior,
                             degrees_of_freedom_prior=degrees_of_freedom_prior, random_state=0)
    assert model.fitted_ is False  # constructed but not yet fit -- this test only checks construction
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_gmm.py -k "gibbs_mode or suggest_priors or em_mode_unaffected or silently_accepts" -v`
Expected: FAIL — `GaussianMixture.__init__() got unexpected keyword argument(s): 'mean_precision_prior'` style errors (params don't exist yet), or `AttributeError: 'GaussianMixture' object has no attribute 'suggest_priors'`.

- [ ] **Step 3: Modify `__init__`'s signature**

In `tgmm/gmm.py`, change the signature at line 132-170. Replace:

```python
    def __init__(
        self,
        # Core model parameters
        n_components: int = 1,
        n_features: int = None,
        covariance_type: str = 'full',
        
        # Convergence and training parameters
        max_iter: int = 1000,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        
        # Initialization parameters (accepts str method or torch.Tensor values)
        init_means='kmeans',
        init_weights='uniform',
        init_covariances='empirical',
        
        # Random state and restart options
        random_state: int = None,
        warm_start: bool = False,
        
        # Algorithm options
        cem: bool = False,
        
        # Prior parameters for MAP estimation
        weight_concentration_prior: torch.Tensor = None,
        mean_prior: torch.Tensor = None,
        mean_precision_prior: float = None,
        covariance_prior: torch.Tensor = None,
        degrees_of_freedom_prior: float = None,
        
        # Output and device options
        verbose: bool = False,
        verbose_interval: int = 10,
        device: str = None,
        
        **kwargs  # Catch deprecated parameters
    ):
```

with:

```python
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
        
        # Initialization parameters (accepts str method or torch.Tensor values)
        init_means='kmeans',
        init_weights='uniform',
        init_covariances='empirical',
        
        # Random state and restart options
        random_state: int = None,
        warm_start: bool = False,
        
        # Algorithm options
        cem: bool = False,
        
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
```

- [ ] **Step 4: Store the new params and add eager validation**

In `tgmm/gmm.py`, the block currently reading (lines 197-268):

```python
        # ===================================================================
        # 2. Store core model parameters
        # ===================================================================
        self.n_components = n_components
        self.n_features = n_features
        
        # Handle covariance type aliases
        if covariance_type == "tied":
            covariance_type = "tied_full"
        elif covariance_type == "isotropic":
            covariance_type = "spherical"
        self.covariance_type = covariance_type

        # ===================================================================
        # 3. Store convergence and training parameters
        # ===================================================================
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        
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
        self.cem = cem
        
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

        self._init_priors(
            weight_concentration_prior,
            mean_prior,
            mean_precision_prior,
            covariance_prior,
            degrees_of_freedom_prior
        )
```

Replace with (adds Gibbs-mode param storage, the `tied_*` rejection, the required-priors check, and skips `_init_priors`'s EM-specific per-component broadcasting for Gibbs mode -- Gibbs mode stores the raw prior values directly and resolves/reshapes them at `fit()` time via `_resolve_priors_gibbs`, added in Task 2):

```python
        # ===================================================================
        # 2. Store core model parameters
        # ===================================================================
        self.n_components = n_components
        self.n_features = n_features
        
        # Handle covariance type aliases
        if covariance_type == "tied":
            covariance_type = "tied_full"
        elif covariance_type == "isotropic":
            covariance_type = "spherical"
        self.covariance_type = covariance_type

        # Gibbs mode (n_components=None): tied_* covariance is rejected
        # immediately -- a shared covariance couples every component's
        # residuals together, so Neal's Algorithm 3 (fully-collapsed Gibbs)
        # does not factor per component. See
        # docs/superpowers/specs/2026-07-28-dpgmm-design.md.
        if n_components is None and covariance_type in ("tied_full", "tied_diag", "tied_spherical"):
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
        self.cem = cem

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
        else:
            self._init_priors(
                weight_concentration_prior,
                mean_prior,
                mean_precision_prior,
                covariance_prior,
                degrees_of_freedom_prior
            )
```

- [ ] **Step 5: Add `n_components_`/`active_` to the state-variables block**

In `tgmm/gmm.py`, the block currently reading (lines 270-283):

```python
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
```

Replace with (adds the two new attributes, initialized `None` like the other `_`-suffixed state, matching this file's existing convention):

```python
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
```

- [ ] **Step 6: Add `suggest_priors`**

Insert as a new `@staticmethod` directly after `__init__` ends (i.e. immediately before the `def _init_priors(` line, currently line 285):

```python
    @staticmethod
    def suggest_priors(X: torch.Tensor, n_components: int, covariance_type: str = "full"):
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

        Returns
        -------
        mean_prior : torch.Tensor, shape (n_features,)
        mean_precision_prior : float
        covariance_prior : torch.Tensor, shape depends on covariance_type
        degrees_of_freedom_prior : float
        """
        n_features = X.shape[1]
        device, dtype = X.device, X.dtype

        X_cpu = X.cpu()
        centers = GMMInitializer.kmeans(X_cpu, n_components).to(device=device, dtype=dtype)
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

```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_gmm.py -k "gibbs_mode or suggest_priors or em_mode_unaffected or silently_accepts" -v`
Expected: PASS, all tests from Step 1.

- [ ] **Step 8: Run the full existing test suite to check for EM-mode regressions**

Run: `source .venv/bin/activate && python -m pytest tests/ -q --ignore=tests/test_dpgmm.py --ignore=tests/test_dpgmm_gibbs.py --ignore=tests/test_dpgmm_gibbs_unbounded.py --ignore=tests/test_dpgmm_covariance_calibration.py`
Expected: PASS (the four DPGMM test files are ignored here since they still import the not-yet-deleted `tgmm.dpgmm.DPGMM` and are rewritten in Tasks 5-8; everything else, especially `test_gmm.py`, must be green).

- [ ] **Step 9: Commit**

```bash
git add tgmm/gmm.py tests/test_gmm.py
git commit -m "Add Gibbs-mode constructor params, eager validation, and suggest_priors to GaussianMixture"
```

---

### Task 2: Port Gibbs fitting internals, wire `fit()` dispatch

**Files:**
- Modify: `tgmm/gmm.py:719-996` (`fit`), insert new section after line 2264 (end of `_update_niw_tied_spherical`, before the "Prediction and Scoring Methods" comment at line 2266)
- Reference (read-only, for verbatim copy): `tgmm/dpgmm.py` (not yet deleted — deleted in Task 4)
- Test: `tests/test_gmm.py` (append)

**Interfaces:**
- Consumes: `self.mean_prior_`/`self.covariance_prior_` are new instance attributes set by `_resolve_priors_gibbs` (this task), distinct from `self.mean_prior`/`self.covariance_prior` (the raw, un-reshaped constructor args, from Task 1) — same naming convention `DPGMM` used.
- Produces: `GaussianMixture.fit(X)` with `n_components=None` fits via Gibbs sampling. Private methods `_resolve_max_components`, `_finalize_active_components`, `_resolve_priors_gibbs`, `_niw_posterior_full/diag/spherical`, `_mvt_log_prob_full/diag/spherical`, `_gibbs_log_likelihood`, `_gibbs_final_point_estimate`, `_gibbs_point_stat2`, `_gibbs_batch_stat2`, `_gibbs_zero_stat2`, `_fit_gibbs`, `_fit_gibbs_truncated`, `_fit_gibbs_unbounded`.

- [ ] **Step 1: Write the failing end-to-end Gibbs-fit test**

Append to `tests/test_gmm.py`:

```python
def test_gibbs_mode_truncated_fit_produces_valid_partition():
    torch.manual_seed(0)
    centers = [[0.0, 0.0], [15.0, 0.0], [7.5, 13.0]]
    covs = [1.5 * torch.eye(2) for _ in range(3)]
    from tgmm.synthetic_data import generate_gmm_data
    X, true_labels = generate_gmm_data(centers, [c.numpy() for c in covs], [30, 30, 30], random_state=0)
    X = X.double()

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6)
    model = GaussianMixture(n_components=None, max_components=6, covariance_type="full",
                             alpha=1.0, max_iter=60, burn_in=20, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    assert model.fitted_
    assert model.weights_.shape == (6,)
    assert model.weights_.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert 2 <= model.n_components_ <= 6

    from sklearn.metrics import adjusted_rand_score
    predicted = model.predict(X)
    assert adjusted_rand_score(true_labels.numpy(), predicted.numpy()) > 0.5


def test_gibbs_mode_unbounded_fit_produces_valid_partition():
    torch.manual_seed(0)
    centers = [[0.0, 0.0], [15.0, 0.0], [7.5, 13.0]]
    covs = [1.5 * torch.eye(2) for _ in range(3)]
    from tgmm.synthetic_data import generate_gmm_data
    X, true_labels = generate_gmm_data(centers, [c.numpy() for c in covs], [25, 25, 25], random_state=0)
    X = X.double()

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6)
    model = GaussianMixture(n_components=None, max_components=None, covariance_type="full",
                             alpha=1.0, max_iter=50, burn_in=15, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    assert model.fitted_
    assert model.max_components is None  # unbounded config untouched by fitting
    assert model.n_components_ == int(model.active_.sum().item())
    assert model.n_components_ >= 1

    from sklearn.metrics import adjusted_rand_score
    predicted = model.predict(X)
    assert adjusted_rand_score(true_labels.numpy(), predicted.numpy()) > 0.3


def test_gibbs_mode_supervised_labels_raises():
    with pytest.raises(ValueError, match="labels"):
        model = GaussianMixture(n_components=None, mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                                 covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
        model.fit(torch.randn(20, 2, dtype=torch.float64), labels=torch.zeros(20, dtype=torch.long))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_gmm.py -k "gibbs_mode_truncated_fit or gibbs_mode_unbounded_fit or gibbs_mode_supervised" -v`
Expected: FAIL — `fit()` currently does `X.size(0) < self.n_components` at line 778, which raises `TypeError: '<' not supported between instances of 'int' and 'NoneType'` since `self.n_components` is `None`.

- [ ] **Step 3: Insert the ported Gibbs internals**

In `tgmm/gmm.py`, insert the following new section immediately after line 2264 (`self.covariances_ = psi_n / (total_nu_n * self.n_features)`, the last line of `_update_niw_tied_spherical`) and before line 2266 (`# ===... Prediction and Scoring Methods`):

```python

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
```

Note the two renames applied relative to `tgmm/dpgmm.py` (everything else in this insertion is byte-identical to the source file): `_resolve_truncation_level` → `_resolve_max_components` (method name, and its call site inside `_fit_gibbs_unbounded`), and `self.truncation_level` → `self.max_components` (inside `_resolve_max_components` and `_fit_gibbs`). No other method in this block references either name.

- [ ] **Step 4: Wire `fit()`'s Gibbs-mode dispatch**

In `tgmm/gmm.py`, the `fit()` method currently begins (lines 774-788):

```python
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
```

Insert a new block immediately before this (i.e. right after the docstring's closing `"""` and before the `# === 1. Validate input parameters ===` comment):

```python
        # ===============================================================
        # 0. Gibbs-sampling mode (n_components=None): dispatch and return
        # early, before any of the EM-specific validation below runs (it
        # assumes self.n_components is an int).
        # ===============================================================
        if self.n_components is None:
            if labels is not None:
                raise ValueError(
                    "Supervised fitting (labels=...) requires a fixed n_components; "
                    "got n_components=None (Gibbs sampling infers the component count "
                    "itself, so there's no fixed-K target for supervised labels)."
                )
            X = X.to(device=self.device)
            self.dtype = X.dtype
            self.n_features = X.shape[1]
            if random_state is not None:
                self.random_state = random_state
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            self._fit_gibbs(X)
            return self

```

(The rest of `fit()` — EM validation, supervised-fit branch, `n_init` loop — is unchanged.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_gmm.py -k "gibbs_mode_truncated_fit or gibbs_mode_unbounded_fit or gibbs_mode_supervised" -v`
Expected: PASS.

- [ ] **Step 6: Run the full non-DPGMM test suite**

Run: `source .venv/bin/activate && python -m pytest tests/ -q --ignore=tests/test_dpgmm.py --ignore=tests/test_dpgmm_gibbs.py --ignore=tests/test_dpgmm_gibbs_unbounded.py --ignore=tests/test_dpgmm_covariance_calibration.py`
Expected: PASS (EM-mode regression guard).

- [ ] **Step 7: Commit**

```bash
git add tgmm/gmm.py tests/test_gmm.py
git commit -m "Port DPGMM's Gibbs sampling internals into GaussianMixture, wire fit() dispatch"
```

---

### Task 3: Attribute unification, `_n_parameters`, `save`/`load`/`save_state_dict`/`load_state_dict`

**Files:**
- Modify: `tgmm/gmm.py` (`fit()`'s EM path and supervised branch, `_n_parameters`, `save`, `load`, `save_state_dict`, `load_state_dict`)
- Test: `tests/test_gmm.py` (append)

**Interfaces:**
- Produces: `n_components_`/`active_` set after every EM-mode fit too (previously Gibbs-only). `_n_parameters()` uses `self.n_components_` uniformly. `save`/`load` round-trip the new Gibbs constructor params and `n_components_`/`active_` for both modes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gmm.py`:

```python
def test_em_mode_sets_n_components_and_active_after_fit():
    model = GaussianMixture(n_components=3, random_state=0)
    model.fit(X_3D)
    assert model.n_components_ == 3
    assert torch.equal(model.active_, torch.ones(3, dtype=torch.bool))


def test_em_mode_supervised_fit_sets_n_components_and_active():
    labels = torch.tensor([0, 0, 1, 1, 2, 2] * 5, dtype=torch.long)
    X = torch.cat([X_3D[:len(labels) // 3], X_3D[:len(labels) // 3], X_3D[:len(labels) // 3]])[:len(labels)]
    model = GaussianMixture(n_components=3, n_features=3)
    model.fit(X, labels=labels)
    assert model.n_components_ == 3
    assert torch.equal(model.active_, torch.ones(3, dtype=torch.bool))


def test_bic_aic_use_n_components_uniformly():
    # Regression guard: _n_parameters must read self.n_components_, not
    # self.n_components, for both modes to agree on the same formula.
    model = GaussianMixture(n_components=3, random_state=0)
    model.fit(X_3D)
    assert model._n_parameters() > 0
    assert model.bic(X_3D) == pytest.approx(
        -2.0 * model.score(X_3D) * X_3D.shape[0] + model._n_parameters() * math.log(X_3D.shape[0])
    )


def test_save_load_round_trip_gibbs_mode():
    torch.manual_seed(0)
    from tgmm.synthetic_data import generate_gmm_data
    centers = [[0.0, 0.0], [15.0, 0.0], [7.5, 13.0]]
    covs = [1.5 * torch.eye(2).numpy() for _ in range(3)]
    X, _ = generate_gmm_data(centers, covs, [20, 20, 20], random_state=0)
    X = X.double()

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=5)
    model = GaussianMixture(n_components=None, max_components=5, alpha=1.0, burn_in=5,
                             weight_threshold=1.0, init_k=None, max_iter=20, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)
    expected = model.predict(X)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "gibbs_gmm.pth")
        model.save(path)
        loaded = GaussianMixture.load(path)

    assert loaded.n_components is None
    assert loaded.max_components == 5
    assert loaded.n_components_ == model.n_components_
    assert torch.equal(loaded.active_, model.active_)
    assert torch.equal(loaded.predict(X), expected)


def test_sample_with_explicit_component_works_in_gibbs_mode():
    # Regression guard: sample(n_samples, component=k)'s index validation
    # used to read self.n_components directly, which is None in Gibbs
    # mode and would raise TypeError comparing int < None.
    torch.manual_seed(0)
    from tgmm.synthetic_data import generate_gmm_data
    centers = [[0.0, 0.0], [15.0, 0.0]]
    covs = [1.5 * torch.eye(2).numpy() for _ in range(2)]
    X, _ = generate_gmm_data(centers, covs, [20, 20], random_state=0)
    X = X.double()

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=4)
    model = GaussianMixture(n_components=None, max_components=4, max_iter=15, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    samples, indices = model.sample(5, component=0)
    assert samples.shape == (5, 2)
    assert (indices == 0).all()

    with pytest.raises(ValueError, match="component"):
        model.sample(5, component=model.weights_.shape[0])  # one past the last valid slot


def test_save_state_dict_load_state_dict_round_trip_gibbs_mode():
    torch.manual_seed(0)
    from tgmm.synthetic_data import generate_gmm_data
    centers = [[0.0, 0.0], [15.0, 0.0]]
    covs = [1.5 * torch.eye(2).numpy() for _ in range(2)]
    X, _ = generate_gmm_data(centers, covs, [20, 20], random_state=0)
    X = X.double()

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=4)
    model = GaussianMixture(n_components=None, max_components=4, max_iter=15, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)
    state = model.save_state_dict()

    reloaded = GaussianMixture(n_components=None, max_components=4, max_iter=15, random_state=0,
                                mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                                covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    reloaded.load_state_dict(state)
    assert reloaded.n_components_ == model.n_components_
    assert torch.equal(reloaded.active_, model.active_)
    assert torch.equal(reloaded.predict(X), model.predict(X))
```

`X_3D` is already defined at module level in `tests/test_gmm.py` (line 25: `X_3D = generate_test_data(n_samples=200, n_features=3, n_clusters=4)`), so the tests above can reference it directly. `math` is **not** currently imported in this file (only `os`, `tempfile`, `pytest`, `torch`, `from tgmm.gmm import GaussianMixture`) — add `import math` alongside the existing `import os` / `import tempfile` lines near the top of `tests/test_gmm.py` before running these tests.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_gmm.py -k "n_components_and_active or bic_aic_use_n_components or round_trip_gibbs_mode or sample_with_explicit_component" -v`
Expected: FAIL — `test_em_mode_sets_n_components_and_active_after_fit` fails because EM mode doesn't set `n_components_` yet; the round-trip tests fail with `TypeError: __init__() got an unexpected keyword argument 'max_components'` or `KeyError` on load.

- [ ] **Step 3: Set `n_components_`/`active_` after EM fits**

In `tgmm/gmm.py`'s `fit()`, the supervised-fit branch currently ends with (around line 896-898, right before `return self`):

```python
            return self
```

(the one immediately following the sparse-covariance warning block, inside the `if labels is not None:` branch). Replace with:

```python
            self.n_components_ = self.n_components
            self.active_ = torch.ones(self.n_components, dtype=torch.bool, device=self.device)
            return self
```

Then, the unsupervised EM path currently ends with (around lines 976-996):

```python
        # ===============================================================
        # 5. Save best result
        # ===============================================================
        if best_params is not None:
            (self.weights_, self.means_, self.covariances_, 
             self.converged_, self.n_iter_, self.lower_bound_) = best_params
            self.best_random_state_ = best_random_state
        
        # Report which random state produced the best result
        if best_random_state is not None and self.n_init > 1:
            warnings.warn(
                f"Best result from random_state={best_random_state}. "
                f"To reproduce this specific result: use random_state={best_random_state} with n_init=1.",
                UserWarning
            )
        
        # Warn if convergence failed
        if not self.converged_:
            warnings.warn(
                "EM did not converge. Consider increasing max_iter or adjusting tol.",
                UserWarning
            )

        return self
```

Replace with (adds the two new attributes right after `best_params` is applied, so they reflect the winning run):

```python
        # ===============================================================
        # 5. Save best result
        # ===============================================================
        if best_params is not None:
            (self.weights_, self.means_, self.covariances_, 
             self.converged_, self.n_iter_, self.lower_bound_) = best_params
            self.best_random_state_ = best_random_state

        self.n_components_ = self.n_components
        self.active_ = torch.ones(self.n_components, dtype=torch.bool, device=self.device)
        
        # Report which random state produced the best result
        if best_random_state is not None and self.n_init > 1:
            warnings.warn(
                f"Best result from random_state={best_random_state}. "
                f"To reproduce this specific result: use random_state={best_random_state} with n_init=1.",
                UserWarning
            )
        
        # Warn if convergence failed
        if not self.converged_:
            warnings.warn(
                "EM did not converge. Consider increasing max_iter or adjusting tol.",
                UserWarning
            )

        return self
```

- [ ] **Step 4: Collapse `_n_parameters` to use `n_components_` uniformly**

In `tgmm/gmm.py`, `_n_parameters` currently reads (lines 2393-2421):

```python
    def _n_parameters(self) -> int:
        r"""
        Number of free parameters in the fitted model.

        Counts mixture weights ($n_{components} - 1$, since they sum to 1),
        means ($n_{components} \times n_{features}$), and covariance
        parameters, whose count depends on `covariance_type` (see
        `_expected_covar_shape` for the corresponding parameter *shapes*).
        Used by `bic`/`aic`.
        """
        n_features = self.n_features
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        elif self.covariance_type == 'tied_full':
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == 'tied_diag':
            cov_params = n_features
        elif self.covariance_type == 'tied_spherical':
            cov_params = 1
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

        mean_params = n_features * self.n_components
        weight_params = self.n_components - 1
        return int(cov_params + mean_params + weight_params)
```

Replace with (every `self.n_components` reference becomes `self.n_components_` -- for EM mode this is now the same fixed value per Step 3; for Gibbs mode it's the active count, matching what `DPGMM._n_parameters` already did):

```python
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
        equal, since EM has no pruning concept.
        """
        n_features = self.n_features
        n_components = self.n_components_
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
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

        mean_params = n_features * n_components
        weight_params = n_components - 1
        return int(cov_params + mean_params + weight_params)
```

- [ ] **Step 4b: Fix `sample()`'s `component=` validation for Gibbs mode**

In `tgmm/gmm.py`'s `sample()` method, the block currently reading (around line 2634-2642):

```python
        # ===============================================================
        # 7. Select component indices
        # ===============================================================
        if component is not None:
            # Validate component index
            if not (0 <= component < self.n_components):
                raise ValueError(
                    f"component must be between 0 and {self.n_components - 1}, got {component}"
                )
```

Replace with (uses `self.weights_.shape[0]`, the actual number of fitted slots, instead of `self.n_components` directly -- correct for both modes: equals `self.n_components` in EM mode, and the total slot count, truncated or unbounded, in Gibbs mode, since a caller may reasonably want to sample from an inactive-but-present slot, not just an active one):

```python
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
```

- [ ] **Step 5: Update `save`/`load`**

In `tgmm/gmm.py`'s `save()` method, the state dict currently includes (around line 3002-3010):

```python
            'cem': self.cem,
            
            # ===============================================================
            # Training state
            # ===============================================================
            'fitted_': self.fitted_,
            'converged_': self.converged_,
            'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_,
```

Replace with:

```python
            'cem': self.cem,
            'max_components': self.max_components,
            'alpha': self.alpha,
            'burn_in': self.burn_in,
            'weight_threshold': self.weight_threshold,
            'init_k': self.init_k,
            
            # ===============================================================
            # Training state
            # ===============================================================
            'fitted_': self.fitted_,
            'converged_': self.converged_,
            'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_,
            'n_components_': self.n_components_,
            'active_': self.active_,
```

In `load()`, the constructor call currently reads (around line 3079-3101):

```python
        model = cls(
            n_components=state_dict['n_components'],
            n_features=state_dict['n_features'],
            covariance_type=state_dict['covariance_type'],
            tol=state_dict['tol'],
            reg_covar=state_dict['reg_covar'],
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
            cem=state_dict['cem'],
        )
```

Replace with (adds the five new Gibbs params, defaulted via `.get()` for backward compatibility with files saved before this merge):

```python
        model = cls(
            n_components=state_dict['n_components'],
            n_features=state_dict['n_features'],
            covariance_type=state_dict['covariance_type'],
            tol=state_dict['tol'],
            reg_covar=state_dict['reg_covar'],
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
            cem=state_dict['cem'],
            max_components=state_dict.get('max_components', 20),
            alpha=state_dict.get('alpha', 1.0),
            burn_in=state_dict.get('burn_in'),
            weight_threshold=state_dict.get('weight_threshold'),
            init_k=state_dict.get('init_k'),
        )
```

Then, right after `model.lower_bound_ = state_dict['lower_bound_']` (around line 3116), add:

```python
        model.n_components_ = state_dict.get('n_components_', state_dict['n_components'])
        model.active_ = state_dict.get('active_')
        if model.active_ is None and model.n_components_ is not None:
            model.active_ = torch.ones(model.n_components_, dtype=torch.bool, device=model.device)
```

- [ ] **Step 6: Update `save_state_dict`/`load_state_dict`**

Apply the identical additions to `save_state_dict()` (mirrors `save()`'s dict literal, around line 3178-3186) and `load_state_dict()` (mirrors `load()`'s reconstruction, around line 3264-3283): add `'max_components'`, `'alpha'`, `'burn_in'`, `'weight_threshold'`, `'init_k'` to the dict after `'cem': self.cem,`; add `'n_components_'`, `'active_'` after `'lower_bound_': self.lower_bound_,`; in `load_state_dict`, add the corresponding five `self.X = state_dict.get('X', ...)` assignments after `self.cem = state_dict['cem']`, and the `n_components_`/`active_` restoration (same three lines as Step 5) after `self.lower_bound_ = state_dict['lower_bound_']`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_gmm.py -k "n_components_and_active or bic_aic_use_n_components or round_trip_gibbs_mode or sample_with_explicit_component" -v`
Expected: PASS.

- [ ] **Step 8: Run the full non-DPGMM test suite**

Run: `source .venv/bin/activate && python -m pytest tests/ -q --ignore=tests/test_dpgmm.py --ignore=tests/test_dpgmm_gibbs.py --ignore=tests/test_dpgmm_gibbs_unbounded.py --ignore=tests/test_dpgmm_covariance_calibration.py`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tgmm/gmm.py tests/test_gmm.py
git commit -m "Unify n_components_/active_ across EM and Gibbs modes, update save/load for new params"
```

---

### Task 4: Delete `tgmm/dpgmm.py`, update `tgmm/__init__.py`

**Files:**
- Delete: `tgmm/dpgmm.py`
- Modify: `tgmm/__init__.py`

**Interfaces:**
- Consumes: nothing new (Tasks 1-3 already made `GaussianMixture` fully cover `DPGMM`'s functionality).
- Produces: `tgmm.DPGMM` no longer exists; `from tgmm import DPGMM` raises `ImportError`.

- [ ] **Step 1: Confirm nothing outside tests/notebooks still imports DPGMM**

Run: `grep -rn "dpgmm\|DPGMM" /home/asp/Downloads/HeaDS/tgmm/tgmm/ --include="*.py"`
Expected: only `tgmm/dpgmm.py` itself and `tgmm/__init__.py`'s import line. If anything else in `tgmm/` references it, stop and investigate before proceeding (this plan doesn't cover that case).

- [ ] **Step 2: Update `tgmm/__init__.py`**

Current full content (8 lines):

```python
__version__ = "0.2.1"

from .gmm import GaussianMixture
from .dpgmm import DPGMM
from .gmm_init import GMMInitializer
from .metrics import ClusteringMetrics
from .plotting import plot_gmm, dynamic_figsize, match_predicted_to_true_labels
from .synthetic_data import generate_gmm_data
```

Remove line 4 (`from .dpgmm import DPGMM`) only. Resulting file:

```python
__version__ = "0.2.1"

from .gmm import GaussianMixture
from .gmm_init import GMMInitializer
from .metrics import ClusteringMetrics
from .plotting import plot_gmm, dynamic_figsize, match_predicted_to_true_labels
from .synthetic_data import generate_gmm_data
```

- [ ] **Step 3: Delete `tgmm/dpgmm.py`**

```bash
git rm tgmm/dpgmm.py
```

- [ ] **Step 4: Verify the package still imports cleanly**

Run: `source .venv/bin/activate && python -c "import tgmm; from tgmm import GaussianMixture; print('OK')"`
Expected: prints `OK`, no `ImportError`.

Run: `source .venv/bin/activate && python -c "from tgmm import DPGMM"`
Expected: `ImportError: cannot import name 'DPGMM' from 'tgmm'` (confirms the removal took effect).

- [ ] **Step 5: Commit**

```bash
git add tgmm/__init__.py
git commit -m "Remove tgmm.DPGMM -- fully absorbed into GaussianMixture(n_components=None)"
```

(Note: `git rm` in Step 3 already stages the deletion; this commit picks it up along with `__init__.py`.)

---

### Task 5: Rewrite `tests/test_dpgmm.py`

**Files:**
- Modify: `tests/test_dpgmm.py` (full rewrite)

**Interfaces:**
- Consumes: `GaussianMixture(n_components=None, max_components=..., mean_prior=..., mean_precision_prior=..., covariance_prior=..., degrees_of_freedom_prior=..., ...)` from Tasks 1-3.

**Context:** The original file's `make_fitted` helper constructs a model and manually injects fitted state, bypassing `fit()` entirely, to test `predict`/`sample`/`save`/`load` scaffolding in isolation. Under the merged class, construction alone (even without calling `fit()`) now requires the four NIW priors whenever `n_components=None` (Task 1's eager validation) — so `make_fitted` needs dummy prior values threaded through, even though they're never actually used (the test overrides fitted state directly). Also: the original file's `test_predict_proba_works_for_every_covariance_type` parametrized over all 6 covariance types including `tied_*`, but `n_components=None` + `tied_*` now raises at construction (Task 1) — that coverage already exists for EM-fitted models in `tests/test_gmm.py`, so this test narrows to the 3 Gibbs-native types.

- [ ] **Step 1: Replace the full file content**

```python
"""Pytest suite for GaussianMixture's Gibbs-mode (n_components=None) scaffolding, construction, and prediction."""
import os
import tempfile

import pytest
import torch

from tgmm import GaussianMixture


def _dummy_priors(n_features, covariance_type="full"):
    """Trivial NIW prior values satisfying __init__'s required-priors check.

    Used only by tests that bypass fit() and manually inject fitted state
    (make_fitted below) -- the actual values are never consulted, since
    _resolve_priors_gibbs (which reads them) is never called.
    """
    if covariance_type in ("full", "tied_full"):
        covariance_prior = torch.eye(n_features, dtype=torch.float64)
    elif covariance_type in ("diag", "tied_diag"):
        covariance_prior = torch.ones(n_features, dtype=torch.float64)
    else:
        covariance_prior = torch.tensor(1.0, dtype=torch.float64)
    return dict(
        mean_prior=torch.zeros(n_features, dtype=torch.float64),
        mean_precision_prior=1.0,
        covariance_prior=covariance_prior,
        degrees_of_freedom_prior=float(n_features + 2),
    )


def make_fitted(covariance_type="full", n_components=3, n_features=2):
    """Build a Gibbs-mode GaussianMixture with a manually-injected fitted
    state, bypassing fit().

    Mirrors the isolation technique used in test_sampling_memory.py: lets us
    test predict/score/sample scaffolding independently of the (much
    heavier) Gibbs fitting machinery.
    """
    model = GaussianMixture(n_components=None, max_components=n_components, n_features=n_features,
                             covariance_type=covariance_type, **_dummy_priors(n_features, covariance_type))
    model.dtype = torch.float64
    model.means_ = torch.stack([
        torch.full((n_features,), 10.0 * k, dtype=torch.float64) for k in range(n_components)
    ])
    model.weights_ = torch.full((n_components,), 1.0 / n_components, dtype=torch.float64)

    if covariance_type == "full":
        model.covariances_ = torch.eye(n_features, dtype=torch.float64).unsqueeze(0).repeat(n_components, 1, 1)
    elif covariance_type == "diag":
        model.covariances_ = torch.ones(n_components, n_features, dtype=torch.float64)
    elif covariance_type == "spherical":
        model.covariances_ = torch.ones(n_components, dtype=torch.float64)

    model.fitted_ = True
    model.converged_ = True
    model.n_components_ = n_components
    model.active_ = torch.ones(n_components, dtype=torch.bool)
    return model


# ============================================================================
# Construction / validation
# ============================================================================

def test_invalid_covariance_type_raises():
    with pytest.raises(ValueError, match="covariance_type"):
        GaussianMixture(n_components=None, covariance_type="not_a_real_type",
                         **_dummy_priors(2))


def test_defaults():
    model = GaussianMixture(n_components=None, **_dummy_priors(2))
    assert model.max_components == 20
    assert model.alpha == 1.0
    assert model.covariance_type == "full"
    assert model.fitted_ is False


def test_max_components_none_allowed_for_unbounded_gibbs():
    # Per design: None is accepted at construction time and triggers the
    # unbounded (dynamic component count) Gibbs sampler at fit() time.
    GaussianMixture(n_components=None, max_components=None, **_dummy_priors(2))


def test_resolved_seed_k_for_unbounded_gibbs_is_generous():
    # The unbounded Gibbs sampler's initial kmeans seed K must be generous:
    # single-site Gibbs moves split a merged-at-init cluster into two only
    # very slowly, so a too-small seed K can get permanently stuck below
    # the true number of clusters.
    model = GaussianMixture(n_components=None, max_components=None, **_dummy_priors(2))
    assert model._resolve_max_components(1000) == 20


# ============================================================================
# Prediction / scoring on a manually-fitted state (full covariance)
# ============================================================================

def test_predict_assigns_nearest_component():
    model = make_fitted("full", n_components=3, n_features=2)
    X = torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=torch.float64)
    labels = model.predict(X)
    assert labels.tolist() == [0, 1, 2]  # X[k] sits exactly at component k's mean


def test_predict_proba_rows_sum_to_one():
    model = make_fitted("full")
    X = torch.randn(10, 2, dtype=torch.float64)
    proba = model.predict_proba(X)
    assert proba.shape == (10, 3)
    assert torch.allclose(proba.sum(dim=1), torch.ones(10, dtype=torch.float64), atol=1e-6)


def test_score_samples_matches_score_mean():
    model = make_fitted("full")
    X = torch.randn(10, 2, dtype=torch.float64)
    per_sample = model.score_samples(X)
    assert per_sample.shape == (10,)
    assert model.score(X) == pytest.approx(per_sample.mean().item())


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_predict_proba_works_for_every_gibbs_native_covariance_type(covariance_type):
    # Narrowed from the pre-merge version's all-6-types parametrization:
    # n_components=None + tied_* now raises at construction (see
    # test_gibbs_mode_rejects_tied_covariance_at_construction in
    # test_gmm.py), so tied_* is unreachable here. predict_proba's
    # tied_*-covariance coverage lives in test_gmm.py via EM-fitted models
    # instead -- the code path is identical either way (predict_proba is
    # mode-agnostic).
    model = make_fitted(covariance_type)
    X = torch.randn(10, 2, dtype=torch.float64)
    proba = model.predict_proba(X)
    assert proba.shape == (10, 3)
    assert torch.allclose(proba.sum(dim=1), torch.ones(10, dtype=torch.float64), atol=1e-6)


# ============================================================================
# Sampling
# ============================================================================

def test_sample_shape_and_component_membership():
    model = make_fitted("full")
    X, indices = model.sample(100)
    assert X.shape == (100, 2)
    assert indices.shape == (100,)
    assert set(indices.tolist()) <= {0, 1, 2}


# ============================================================================
# Save / load round trip
# ============================================================================

def test_save_load_round_trip_preserves_predictions():
    model = make_fitted("full")
    X = torch.randn(10, 2, dtype=torch.float64)
    expected = model.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "gmm_gibbs.pt")
        model.save(path)
        loaded = GaussianMixture.load(path)

    assert torch.equal(loaded.predict(X), expected)
    assert loaded.covariance_type == model.covariance_type
    assert loaded.n_components_ == model.n_components_
    assert loaded.dtype == model.dtype

    sampled, indices = loaded.sample(10)  # exercises dtype consistency (means_ vs. sampled noise)
    assert sampled.shape == (10, 2)
```

- [ ] **Step 2: Run the tests**

Run: `source .venv/bin/activate && python -m pytest tests/test_dpgmm.py -v`
Expected: PASS, all tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dpgmm.py
git commit -m "Port test_dpgmm.py to GaussianMixture(n_components=None) API"
```

---

### Task 6: Rewrite `tests/test_dpgmm_gibbs.py`

**Files:**
- Modify: `tests/test_dpgmm_gibbs.py` (full rewrite)

**Context:** Same required-priors wrinkle as Task 5 for the low-level `_niw_posterior_*`/`_mvt_log_prob_*` unit tests (they construct without calling `fit()`, so need dummy priors). `test_truncated_gibbs_rejects_tied_covariance` moves its assertion from `fit()`-time to construction-time (Task 1's eager `__init__` check) and no longer needs any prior args, since the `tied_*` check fires before the priors check.

- [ ] **Step 1: Replace the full file content**

```python
"""Pytest suite for GaussianMixture's collapsed Gibbs inference (full covariance)."""
import math

import numpy as np
import torch
import pytest
from sklearn.metrics import adjusted_rand_score

from tgmm import GaussianMixture
from tgmm.synthetic_data import generate_gmm_data


def three_blob_data(n_per_cluster=30, seed=0):
    centers = [[0.0, 0.0], [15.0, 0.0], [7.5, 13.0]]
    covs = [1.5 * torch.eye(2).numpy() for _ in range(3)]
    X, labels = generate_gmm_data(centers, covs, [n_per_cluster] * 3, random_state=seed)
    return X.double(), labels


def _dummy_priors(n_features):
    return dict(
        mean_prior=torch.zeros(n_features, dtype=torch.float64),
        mean_precision_prior=1.0,
        covariance_prior=torch.eye(n_features, dtype=torch.float64),
        degrees_of_freedom_prior=float(n_features + 2),
    )


# ============================================================================
# NIW posterior / multivariate-t marginal likelihood (full covariance)
# ============================================================================

def test_niw_posterior_full_matches_hand_computation_for_two_points():
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="full", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64
    model.mean_prior_ = torch.zeros(d, dtype=torch.float64)
    model.covariance_prior_ = torch.eye(d, dtype=torch.float64)
    model.mean_precision_prior = 1.0
    model.degrees_of_freedom_prior = float(d + 2)
    model.reg_covar = 0.0

    x1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    x2 = torch.tensor([3.0, 0.0], dtype=torch.float64)
    n_k = torch.tensor([2.0], dtype=torch.float64)
    sum_x_k = (x1 + x2).unsqueeze(0)
    sum_xxT_k = (torch.outer(x1, x1) + torch.outer(x2, x2)).unsqueeze(0)

    mu_n, lambda_n, nu_n, psi_n = model._niw_posterior_full(n_k, sum_x_k, sum_xxT_k)

    lambda0, nu0 = 1.0, float(d + 2)
    empirical_mean = (x1 + x2) / 2
    expected_lambda_n = lambda0 + 2
    expected_mu_n = (lambda0 * torch.zeros(d, dtype=torch.float64) + (x1 + x2)) / expected_lambda_n
    S = torch.outer(x1 - empirical_mean, x1 - empirical_mean) + torch.outer(x2 - empirical_mean, x2 - empirical_mean)
    cross = (lambda0 * 2 / expected_lambda_n) * torch.outer(empirical_mean, empirical_mean)
    expected_psi_n = torch.eye(d, dtype=torch.float64) + S + cross

    assert lambda_n.item() == pytest.approx(expected_lambda_n)
    assert nu_n.item() == pytest.approx(nu0 + 2)
    assert torch.allclose(mu_n.squeeze(0), expected_mu_n)
    assert torch.allclose(psi_n.squeeze(0), expected_psi_n)


def test_niw_posterior_full_empty_component_falls_back_to_prior():
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="full", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64
    model.mean_prior_ = torch.tensor([1.0, -1.0], dtype=torch.float64)
    model.covariance_prior_ = 2.0 * torch.eye(d, dtype=torch.float64)
    model.mean_precision_prior = 0.5
    model.degrees_of_freedom_prior = float(d + 2)
    model.reg_covar = 0.0

    n_k = torch.tensor([0.0], dtype=torch.float64)
    sum_x_k = torch.zeros(1, d, dtype=torch.float64)
    sum_xxT_k = torch.zeros(1, d, d, dtype=torch.float64)

    mu_n, lambda_n, nu_n, psi_n = model._niw_posterior_full(n_k, sum_x_k, sum_xxT_k)

    assert torch.allclose(mu_n.squeeze(0), model.mean_prior_)
    assert lambda_n.item() == pytest.approx(0.5)
    assert nu_n.item() == pytest.approx(d + 2)
    assert torch.allclose(psi_n.squeeze(0), model.covariance_prior_)


def test_mvt_log_prob_full_matches_scipy_multivariate_t():
    scipy_stats = pytest.importorskip("scipy.stats")
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="full", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64

    mu_n = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    lambda_n = torch.tensor([2.0], dtype=torch.float64)
    nu_n = torch.tensor([6.0], dtype=torch.float64)  # df = nu_n - d + 1 = 5
    psi_n = torch.eye(d, dtype=torch.float64).unsqueeze(0) * 3.0

    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    log_prob = model._mvt_log_prob_full(x, mu_n, lambda_n, nu_n, psi_n)

    df = 5.0
    scale = (psi_n[0] * (lambda_n[0] + 1) / (lambda_n[0] * df)).numpy()
    expected = scipy_stats.multivariate_t.logpdf(x.numpy(), loc=mu_n[0].numpy(), shape=scale, df=df)

    assert log_prob.shape == (1,)
    assert log_prob.item() == pytest.approx(expected, abs=1e-6)


# ============================================================================
# NIW posterior / marginal likelihood: diag covariance
# ============================================================================

def test_mvt_log_prob_diag_matches_scipy_product_of_t():
    scipy_stats = pytest.importorskip("scipy.stats")
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="diag", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64

    n_k = torch.tensor([3.0], dtype=torch.float64)
    mu_n = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    lambda_n = torch.tensor([4.0], dtype=torch.float64)
    nu_n = torch.tensor([5.0], dtype=torch.float64)  # df = nu_n (no -d+1 for diag)
    psi_n = torch.tensor([[2.0, 1.0]], dtype=torch.float64)

    x = torch.tensor([1.0, -1.0], dtype=torch.float64)
    log_prob = model._mvt_log_prob_diag(x, n_k, mu_n, lambda_n, nu_n, psi_n)

    df = 5.0
    expected = 0.0
    for j in range(d):
        scale_j = (psi_n[0, j] * (lambda_n[0] + 1) / (nu_n[0] * lambda_n[0])).item()
        expected += scipy_stats.t.logpdf(x[j].item(), df=df, loc=mu_n[0, j].item(), scale=math.sqrt(scale_j))

    assert log_prob.item() == pytest.approx(expected, abs=1e-6)


def test_niw_posterior_diag_empty_component_falls_back_to_prior():
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="diag", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64
    model.mean_prior_ = torch.tensor([1.0, -1.0], dtype=torch.float64)
    model.covariance_prior_ = torch.tensor([2.0, 3.0], dtype=torch.float64)
    model.mean_precision_prior = 0.5
    model.degrees_of_freedom_prior = float(d + 2)
    model.reg_covar = 0.0

    n_k = torch.tensor([0.0], dtype=torch.float64)
    sum_x_k = torch.zeros(1, d, dtype=torch.float64)
    sum_x2_k = torch.zeros(1, d, dtype=torch.float64)

    mu_n, lambda_n, nu_n, psi_n = model._niw_posterior_diag(n_k, sum_x_k, sum_x2_k)
    assert torch.allclose(mu_n.squeeze(0), model.mean_prior_)
    assert torch.allclose(psi_n.squeeze(0), model.covariance_prior_)
    assert nu_n.item() == pytest.approx(d + 2)


# ============================================================================
# NIW posterior / marginal likelihood: spherical covariance
# ============================================================================

def test_mvt_log_prob_spherical_matches_scipy_isotropic_multivariate_t():
    scipy_stats = pytest.importorskip("scipy.stats")
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="spherical", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64
    model.degrees_of_freedom_prior = 4.0  # nu0

    n_k = torch.tensor([3.0], dtype=torch.float64)
    mu_n = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    lambda_n = torch.tensor([4.0], dtype=torch.float64)
    nu_n = torch.tensor([7.0], dtype=torch.float64)  # nu0 + n_k, unused by the predictive itself
    psi_n = torch.tensor([6.0], dtype=torch.float64)

    x = torch.tensor([1.0, -1.0], dtype=torch.float64)
    log_prob = model._mvt_log_prob_spherical(x, n_k, mu_n, lambda_n, nu_n, psi_n)

    df = 4.0 + 3.0 * d  # nu0 + n_k * d
    scale = (psi_n[0] * (lambda_n[0] + 1) / (lambda_n[0] * df)).item()
    expected = scipy_stats.multivariate_t.logpdf(
        x.numpy(), loc=mu_n[0].numpy(), shape=scale * torch.eye(d).numpy(), df=df)

    assert log_prob.item() == pytest.approx(expected, abs=1e-6)


def test_niw_posterior_spherical_empty_component_falls_back_to_prior():
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="spherical", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64
    model.mean_prior_ = torch.tensor([1.0, -1.0], dtype=torch.float64)
    model.covariance_prior_ = torch.tensor(2.0, dtype=torch.float64)
    model.mean_precision_prior = 0.5
    model.degrees_of_freedom_prior = float(d + 2)
    model.reg_covar = 0.0

    n_k = torch.tensor([0.0], dtype=torch.float64)
    sum_x_k = torch.zeros(1, d, dtype=torch.float64)
    sum_sq_k = torch.zeros(1, dtype=torch.float64)

    mu_n, lambda_n, nu_n, psi_n = model._niw_posterior_spherical(n_k, sum_x_k, sum_sq_k)
    assert torch.allclose(mu_n.squeeze(0), model.mean_prior_)
    assert psi_n.item() == pytest.approx(model.covariance_prior_.item())


# ============================================================================
# End-to-end truncated Gibbs fit
# ============================================================================

@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_truncated_gibbs_fit_produces_valid_weights(covariance_type):
    X, true_labels = three_blob_data()
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6, covariance_type=covariance_type)
    model = GaussianMixture(n_components=None, max_components=6, covariance_type=covariance_type,
                             alpha=1.0, max_iter=60, burn_in=20, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    assert model.fitted_
    assert model.weights_.shape == (6,)
    assert model.weights_.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert 2 <= model.n_components_ <= 6

    predicted = model.predict(X)
    assert adjusted_rand_score(true_labels.numpy(), predicted.numpy()) > 0.5


def test_truncated_gibbs_weights_are_self_consistent_with_final_partition():
    # weights_ must reflect the same partition that means_/covariances_ and
    # predict() do -- every sweep re-sorts slots by descending occupancy,
    # so slot index k is not the same semantic component across sweeps (see
    # docs/superpowers/specs/2026-07-28-dpgmm-design.md for the failure
    # mode this guards against: a cross-sweep occupancy average can
    # silently split one real cluster's mass across two reported slots).
    rng = np.random.RandomState(0)
    centers = [rng.randn(10) * 8.0 for _ in range(10)]
    covs = [np.eye(10) for _ in range(10)]
    X, true_labels = generate_gmm_data(centers, covs, [100] * 10, random_state=0)
    X = X.double()

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=20)
    model = GaussianMixture(n_components=None, max_components=20, covariance_type="full",
                             alpha=1.0, max_iter=50, burn_in=15, random_state=0, weight_threshold=10.0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    predicted = model.predict(X)
    predicted_counts = torch.bincount(predicted, minlength=model.weights_.shape[0]).double()
    assert torch.allclose(predicted_counts / X.shape[0], model.weights_, atol=1e-6)
    assert model.n_components_ == 10


def test_truncated_gibbs_rejects_tied_covariance():
    # Moved from fit()-time (pre-merge DPGMM) to construction-time (Task 1's
    # eager __init__ validation) -- no priors needed, the tied_* check
    # fires before the priors-required check.
    with pytest.raises(NotImplementedError, match="tied"):
        GaussianMixture(n_components=None, max_components=5, covariance_type="tied_full")
```

- [ ] **Step 2: Run the tests**

Run: `source .venv/bin/activate && python -m pytest tests/test_dpgmm_gibbs.py -v`
Expected: PASS, all tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dpgmm_gibbs.py
git commit -m "Port test_dpgmm_gibbs.py to GaussianMixture(n_components=None) API"
```

---

### Task 7: Rewrite `tests/test_dpgmm_gibbs_unbounded.py`

**Files:**
- Modify: `tests/test_dpgmm_gibbs_unbounded.py` (full rewrite)

**Context:** `model.truncation_level` attribute access renamed to `model.max_components`. `test_init_k_overrides_auto_heuristic`/`test_init_k_unset_falls_back_to_auto_heuristic` construct without calling `fit()`, so need dummy priors (same pattern as Tasks 5-6). `test_unbounded_gibbs_rejects_tied_covariance` moves to construction-time like Task 6's equivalent.

- [ ] **Step 1: Replace the full file content**

```python
"""Pytest suite for GaussianMixture's unbounded collapsed Gibbs sampler (Neal's Algorithm 3, full covariance)."""
import torch
import pytest
from sklearn.metrics import adjusted_rand_score

from tgmm import GaussianMixture
from tgmm.synthetic_data import generate_gmm_data


def three_blob_data(n_per_cluster=25, seed=0):
    centers = [[0.0, 0.0], [15.0, 0.0], [7.5, 13.0]]
    covs = [1.5 * torch.eye(2).numpy() for _ in range(3)]
    X, labels = generate_gmm_data(centers, covs, [n_per_cluster] * 3, random_state=seed)
    return X.double(), labels


def _dummy_priors(n_features=2):
    return dict(
        mean_prior=torch.zeros(n_features, dtype=torch.float64),
        mean_precision_prior=1.0,
        covariance_prior=torch.eye(n_features, dtype=torch.float64),
        degrees_of_freedom_prior=float(n_features + 2),
    )


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_unbounded_gibbs_fit_produces_valid_partition(covariance_type):
    X, true_labels = three_blob_data()
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6, covariance_type=covariance_type)
    model = GaussianMixture(n_components=None, max_components=None, covariance_type=covariance_type,
                             alpha=1.0, max_iter=50, burn_in=15, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    assert model.fitted_
    assert model.max_components is None  # unbounded config is untouched by fitting
    assert model.n_components_ == int(model.active_.sum().item())
    assert model.n_components_ >= 1
    assert model.weights_.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (model.weights_ > 0).all()  # every surviving component is non-empty by construction

    predicted = model.predict(X)
    assert adjusted_rand_score(true_labels.numpy(), predicted.numpy()) > 0.3


def test_unbounded_gibbs_tracks_component_count_history():
    X, _ = three_blob_data(n_per_cluster=15)
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6)
    model = GaussianMixture(n_components=None, max_components=None, covariance_type="full",
                             alpha=1.0, max_iter=30, burn_in=10, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    assert hasattr(model, "n_components_history_")
    assert len(model.n_components_history_) == 30 - 10
    assert all(k >= 1 for k in model.n_components_history_)


def test_unbounded_gibbs_n_components_respects_weight_threshold():
    # A far-flung outlier likely ends up in its own near-singleton component.
    # n_components_ must apply the same weight_threshold rule as the other
    # inference path (_finalize_active_components), not just count every
    # surviving component regardless of how little data supports it.
    X, _ = three_blob_data()
    outlier = torch.tensor([[200.0, 200.0]], dtype=torch.float64)
    X = torch.cat([X, outlier], dim=0)

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6)
    model = GaussianMixture(n_components=None, max_components=None, covariance_type="full",
                             alpha=1.0, max_iter=50, burn_in=15, random_state=0, weight_threshold=1.0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    expected_active = model.weights_ * X.shape[0] > model.weight_threshold
    assert torch.equal(model.active_, expected_active)
    assert model.n_components_ == int(expected_active.sum().item())


@pytest.mark.parametrize("covariance_type", ["tied_full", "tied_diag", "tied_spherical"])
def test_unbounded_gibbs_rejects_tied_covariance(covariance_type):
    with pytest.raises(NotImplementedError, match="tied"):
        GaussianMixture(n_components=None, max_components=None, covariance_type=covariance_type)


def test_init_k_overrides_auto_heuristic():
    # _resolve_max_components's automatic seed (max(2, min(20, n // 2)))
    # must be overridden whenever init_k is set, regardless of n_samples --
    # that's the whole point of init_k (giving unbounded a caller-chosen
    # starting K instead of one implicitly coupled to n_samples).
    model = GaussianMixture(n_components=None, max_components=None, init_k=7, **_dummy_priors())
    assert model._resolve_max_components(n_samples=10_000) == 7
    assert model._resolve_max_components(n_samples=3) == 7


def test_init_k_unset_falls_back_to_auto_heuristic():
    model = GaussianMixture(n_components=None, max_components=None, **_dummy_priors())
    assert model._resolve_max_components(n_samples=1000) == 20  # min(20, 1000 // 2)
    assert model._resolve_max_components(n_samples=6) == 3      # min(20, 6 // 2)


def test_unbounded_matches_truncated_given_same_init_k():
    # On well-separated data, truncated (max_components=K) and unbounded
    # (max_components=None, init_k=K) should start from the identical
    # k-means partition and settle on the identical final one for the same
    # random_state -- the scenario notebooks/dpgmm_gibbs_sampling.ipynb's
    # "Truncated and unbounded Gibbs agree" section demonstrates. This is
    # not a universal guarantee on harder (overlapping/non-Gaussian) data,
    # where the two samplers' slightly different per-sweep rules can still
    # drift apart from the same start -- see the design doc.
    X, true_labels = three_blob_data(n_per_cluster=40)
    K_SEED = 10
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=K_SEED)

    truncated = GaussianMixture(n_components=None, max_components=K_SEED, covariance_type="full",
                                 alpha=1.0, max_iter=50, random_state=0, weight_threshold=1.0,
                                 mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                                 covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    truncated.fit(X)

    unbounded = GaussianMixture(n_components=None, max_components=None, init_k=K_SEED, covariance_type="full",
                                 alpha=1.0, max_iter=50, burn_in=15, random_state=0, weight_threshold=1.0,
                                 mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                                 covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    unbounded.fit(X)

    assert truncated.n_components_ == unbounded.n_components_ == 3
    agreement = adjusted_rand_score(truncated.predict(X).numpy(), unbounded.predict(X).numpy())
    assert agreement == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests**

Run: `source .venv/bin/activate && python -m pytest tests/test_dpgmm_gibbs_unbounded.py -v`
Expected: PASS, all tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dpgmm_gibbs_unbounded.py
git commit -m "Port test_dpgmm_gibbs_unbounded.py to GaussianMixture(n_components=None) API"
```

---

### Task 8: Rewrite `tests/test_dpgmm_covariance_calibration.py`

**Files:**
- Modify: `tests/test_dpgmm_covariance_calibration.py` (full rewrite)

**Context:** The original file's premise was "check the auto-resolved default prior doesn't inflate covariance." That default no longer exists (Task 1 requires explicit priors). This rewrite preserves the regression coverage by explicitly calling `suggest_priors` first — now testing that the *helper* produces well-calibrated priors, which is the more accurate framing post-merge.

- [ ] **Step 1: Replace the full file content**

```python
"""Regression tests for GaussianMixture.suggest_priors' covariance calibration.

Gibbs-mode fitting (n_components=None) always requires an NIW prior --
Neal's Algorithm 3 requires one to integrate out means/covariances
analytically, unlike EM mode which can run as plain MLE with no prior at
all. suggest_priors is the library's principled-starting-point helper for
that required prior; an earlier version of this project's (now-merged)
DPGMM class auto-resolved a default using this same logic, and an even
earlier version of that default used the *marginal* covariance of the
whole dataset, which conflated *between*-cluster spread with *within*-
cluster spread: for well-separated multi-modal data, the global covariance
is dominated by how far apart the cluster centers are, not by how tight
each cluster actually is, and the mean-prior cross term's coefficient
`(lambda0 * n_k) / (lambda0 + n_k)` converges to `lambda0` (not 0) as
`n_k -> infinity`, so it never shrinks away when a cluster's own mean sits
far from the global mean. Net effect: every component's fitted covariance
was systematically inflated, and *more* inflated the more cleanly
separated the clusters were -- backwards from what you'd want.

Reproduced on 10-d, 5-cluster synthetic data with true (identity) trace 10:
classical GaussianMixture (EM/MLE) recovered trace ~9.70; Gibbs mode with
the marginal-covariance prior recovered ~16.11 (well-separated, spread=8.0)
but only ~9.88 (overlapping, spread=2.5) -- confirming the inflation
tracked how much the global covariance overstated the true within-cluster
scale, not some fixed sampler bias. suggest_priors' pooled-within-cluster-
scatter estimator (this file's subject) fixes it.
"""
import numpy as np
import pytest
import torch

from tgmm import GaussianMixture
from tgmm.synthetic_data import generate_gmm_data


def make_isotropic_blobs(k, n_per_cluster, spread, d=10, seed=0):
    rng = np.random.RandomState(seed)
    centers = [rng.randn(d) * spread for _ in range(k)]
    covs = [np.eye(d) for _ in range(k)]
    X, labels = generate_gmm_data(centers, covs, [n_per_cluster] * k, random_state=seed)
    return X.double(), labels


@pytest.mark.parametrize("max_components", [None, 20])
def test_gibbs_covariance_not_inflated_on_well_separated_clusters(max_components):
    d = 10
    X, _ = make_isotropic_blobs(k=5, n_per_cluster=200, spread=8.0, d=d)
    extra = dict(max_iter=50, burn_in=15) if max_components is None else dict(max_iter=50)

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=5)
    model = GaussianMixture(n_components=None, covariance_type="full", alpha=1.0, random_state=0,
                             weight_threshold=10, device="cpu", max_components=max_components,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior,
                             **extra)
    model.fit(X)

    true_trace = float(d)  # true per-cluster covariance is identity
    mean_trace = model.covariances_[model.active_].diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()

    # Classical GaussianMixture (EM/MLE) on the same data recovers ~9.7;
    # the marginal-covariance-prior version of Gibbs mode reported ~16.1
    # here (a ~60% inflation) -- suggest_priors' pooled estimator fixes it.
    assert mean_trace == pytest.approx(true_trace, rel=0.25)


def test_gibbs_covariance_stays_accurate_on_overlapping_clusters():
    # Non-regression guard: the overlapping regime was already close to
    # unbiased even under the old marginal-covariance prior (~9.88 vs true
    # 10) -- suggest_priors must not make it worse.
    d = 10
    X, _ = make_isotropic_blobs(k=5, n_per_cluster=200, spread=2.5, d=d)

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=5)
    model = GaussianMixture(n_components=None, covariance_type="full", alpha=1.0, random_state=0,
                             weight_threshold=10, device="cpu", max_components=None, max_iter=50, burn_in=15,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    true_trace = float(d)
    mean_trace = model.covariances_[model.active_].diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()
    assert mean_trace == pytest.approx(true_trace, rel=0.25)
```

- [ ] **Step 2: Run the tests**

Run: `source .venv/bin/activate && python -m pytest tests/test_dpgmm_covariance_calibration.py -v`
Expected: PASS, all 3 tests (2 from the `max_components` parametrization + 1).

- [ ] **Step 3: Commit**

```bash
git add tests/test_dpgmm_covariance_calibration.py
git commit -m "Port test_dpgmm_covariance_calibration.py to suggest_priors + GaussianMixture(n_components=None)"
```

---

### Task 9: Full test suite checkpoint

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `source .venv/bin/activate && python -m pytest tests/ -q`
Expected: all tests pass (this repo was at 343 passing before this plan started: 340 pre-existing + 3 `init_k`-equivalence tests added earlier this session — after Tasks 1-8 the count will be higher, from the new `test_gmm.py` additions in Tasks 1-3). Zero failures, zero errors.

- [ ] **Step 2: If anything fails, stop and fix before proceeding**

Do not start Task 10 (notebook rework) with a red test suite — the notebooks import the same `tgmm` package under active development, and debugging notebook execution failures on top of an already-broken library layer is much harder to diagnose.

---

### Task 10: Rework `notebooks/dpgmm.ipynb`

**Files:**
- Modify: `notebooks/dpgmm.ipynb`

**Context:** Every `DPGMM(...)` call site in this notebook currently fits with zero explicit priors (relying on the removed auto-default). This notebook has two call sites: `fit_dpgmm_and_summarize` (in the cell defining `WEIGHT_THRESHOLD`/`K_SWEEP_MAX`/`GIBBS_INIT_K`/`method_configs`/`fit_dpgmm_and_summarize`/`fit_gmm_and_summarize`/`fit_gmm_via_bic_sweep`/`run_round`) is the only one that actually constructs a Gibbs model — it's called once per `(round, true_k)` combination from inside `run_round`. Each call needs a `GaussianMixture.suggest_priors(X, k)` computed and threaded in.

- [ ] **Step 1: Write and run a Python script that patches the notebook JSON**

Create `/tmp/dpgmm_notebook_patch.py` (or use the session's scratchpad directory) with this content:

```python
import json

NB_PATH = "notebooks/dpgmm.ipynb"
nb = json.load(open(NB_PATH))

# Cell 1 imports: DPGMM -> GaussianMixture
cell1_src = "".join(nb["cells"][1]["source"])
assert "from tgmm import DPGMM, GaussianMixture, plot_gmm, dynamic_figsize" in cell1_src
cell1_src = cell1_src.replace(
    "from tgmm import DPGMM, GaussianMixture, plot_gmm, dynamic_figsize",
    "from tgmm import GaussianMixture, plot_gmm, dynamic_figsize",
)
nb["cells"][1]["source"] = cell1_src.splitlines(keepends=True)

# Cell 10 (comparison driver): fit_dpgmm_and_summarize constructs DPGMM ->
# GaussianMixture(n_components=None, ...), with an explicit suggest_priors
# call threaded into method_configs' caller. Locate and replace exactly.
cell10_src = "".join(nb["cells"][10]["source"])

old_fit_dpgmm = '''def fit_dpgmm_and_summarize(X, labels, pca, label, truncation_level, weight_threshold, random_state, init_k=None):
    extra = dict(max_iter=50) if truncation_level is not None else dict(max_iter=50, burn_in=15)
    # device='cpu' explicitly: DPGMM/GaussianMixture default to CUDA when
    # available, and this notebook's numpy()/adjusted_rand_score calls
    # assume CPU tensors. The data here (1,000 x 10) gains nothing from a
    # GPU anyway.
    model = DPGMM(covariance_type='full', alpha=1.0, random_state=random_state,
                  weight_threshold=weight_threshold, device='cpu',
                  truncation_level=truncation_level, init_k=init_k, **extra)'''

new_fit_dpgmm = '''def fit_dpgmm_and_summarize(X, labels, pca, label, truncation_level, weight_threshold, random_state, init_k=None):
    extra = dict(max_iter=50) if truncation_level is not None else dict(max_iter=50, burn_in=15)
    # device='cpu' explicitly: GaussianMixture defaults to CUDA when
    # available, and this notebook's numpy()/adjusted_rand_score calls
    # assume CPU tensors. The data here (1,000 x 10) gains nothing from a
    # GPU anyway.
    #
    # Gibbs mode (n_components=None) requires an explicit NIW prior --
    # suggest_priors gives a principled starting point from a quick
    # k-means partition into GIBBS_INIT_K groups (the same K both
    # truncated and unbounded start from -- see method_configs).
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \\
        GaussianMixture.suggest_priors(X, n_components=GIBBS_INIT_K)
    model = GaussianMixture(n_components=None, covariance_type='full', alpha=1.0, random_state=random_state,
                             weight_threshold=weight_threshold, device='cpu',
                             max_components=truncation_level, init_k=init_k,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior,
                             **extra)'''

assert old_fit_dpgmm in cell10_src, "fit_dpgmm_and_summarize block not found verbatim -- check for drift since this plan was written"
cell10_src = cell10_src.replace(old_fit_dpgmm, new_fit_dpgmm)
nb["cells"][10]["source"] = cell10_src.splitlines(keepends=True)
nb["cells"][10]["outputs"] = []
nb["cells"][10]["execution_count"] = None

json.dump(nb, open(NB_PATH, "w"), indent=1)
print("patched")
```

Run: `source .venv/bin/activate && python /tmp/dpgmm_notebook_patch.py`
Expected: prints `patched`. If the `AssertionError` fires, open `notebooks/dpgmm.ipynb` cell 10 and diff by hand against the block above — something changed since this plan was written; adapt the replacement to match.

- [ ] **Step 2: Dry-run the patched `fit_dpgmm_and_summarize` standalone before a full notebook re-execution**

Run this as a standalone script (not inside the notebook) to catch errors cheaply:

```bash
source .venv/bin/activate && python3 - <<'EOF'
import torch, numpy as np
from tgmm import GaussianMixture
from tgmm.synthetic_data import generate_gmm_data

torch.set_default_dtype(torch.float64)
GIBBS_INIT_K = 30

def make_high_dim_blobs(k, n_per_cluster, spread, d=10, seed=0):
    rng = np.random.RandomState(seed)
    centers = [rng.randn(d) * spread for _ in range(k)]
    covs = [np.eye(d) for _ in range(k)]
    X, labels = generate_gmm_data(centers, covs, [n_per_cluster] * k, random_state=seed)
    return X.double(), labels

X, labels = make_high_dim_blobs(10, 100, spread=4.0, d=10, seed=0)
mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
    GaussianMixture.suggest_priors(X, n_components=GIBBS_INIT_K)
model = GaussianMixture(n_components=None, covariance_type='full', alpha=1.0, random_state=0,
                         weight_threshold=10, device='cpu', max_components=GIBBS_INIT_K, init_k=None,
                         mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                         covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior,
                         max_iter=50)
model.fit(X)
print("n_components_:", model.n_components_, "fitted_:", model.fitted_)
EOF
```

Expected: prints `n_components_: 10 fitted_: True` (or close to 10 -- exact count depends on the sampler, but it must run without exception and report a plausible count).

- [ ] **Step 3: Re-execute the full notebook**

Run: `source .venv/bin/activate && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=580 --ExecutePreprocessor.kernel_name=python3 notebooks/dpgmm.ipynb`

This will take several minutes (re-runs all 4 rounds' BIC sweeps and Gibbs fits). If it exceeds a single command's timeout, run it with `run_in_background: true` and wait for completion.

- [ ] **Step 4: Verify clean execution**

Run:
```bash
source .venv/bin/activate && python3 -c "
import json
nb = json.load(open('notebooks/dpgmm.ipynb'))
errors = [(i, o.get('ename'), o.get('evalue')) for i, c in enumerate(nb['cells']) for o in c.get('outputs', []) if o.get('output_type') == 'error']
print('errors:', errors if errors else 'none')
"
```
Expected: `errors: none`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/dpgmm.ipynb
git commit -m "Rework dpgmm.ipynb for merged GaussianMixture(n_components=None) API"
```

---

### Task 11: Rework `notebooks/dpgmm_gibbs_sampling.ipynb`

**Files:**
- Modify: `notebooks/dpgmm_gibbs_sampling.ipynb`

**Context:** This notebook has more `DPGMM(...)` call sites than `dpgmm.ipynb`: cell 3 (stick-breaking, no DPGMM construction — skip), cell 7 (data gen, no DPGMM — skip), cell 9 (sweep-by-sweep snapshots, constructs `DPGMM` once per snapshot), cell 12 (component-count trajectory), cell 14 (truncated-vs-unbounded agreement, two constructions). All operate on the same `X`/`true_k=10` dataset set up in cell 7, so one `suggest_priors` call can be computed once (in cell 7, right after `X` is created) and reused by cells 9/12/14.

- [ ] **Step 1: Patch cell 1 (imports)**

```python
import json

NB_PATH = "notebooks/dpgmm_gibbs_sampling.ipynb"
nb = json.load(open(NB_PATH))

cell1_src = "".join(nb["cells"][1]["source"])
assert "from tgmm import DPGMM, plot_gmm, dynamic_figsize" in cell1_src
cell1_src = cell1_src.replace(
    "from tgmm import DPGMM, plot_gmm, dynamic_figsize",
    "from tgmm import GaussianMixture, plot_gmm, dynamic_figsize",
)
nb["cells"][1]["source"] = cell1_src.splitlines(keepends=True)
json.dump(nb, open(NB_PATH, "w"), indent=1)
print("cell 1 patched")
```

Run: `source .venv/bin/activate && python3 <above script>`

- [ ] **Step 2: Patch cell 7 to compute and store a shared prior**

Read the current cell 7 first (it's the data-generation cell, ends with `plt.show()` after the ground-truth scatter plot). Append the `suggest_priors` call at the end of that cell, after the existing content:

```python
import json

NB_PATH = "notebooks/dpgmm_gibbs_sampling.ipynb"
nb = json.load(open(NB_PATH))

cell7_src = "".join(nb["cells"][7]["source"])
addition = '''

# Gibbs mode (n_components=None) requires an explicit NIW prior -- computed
# once here from a k-means partition into 20 groups (matching this
# notebook's own K_SEED used later, and _resolve_max_components's default
# heuristic for this many samples) and reused by every DPGMM fit below.
GIBBS_PRIOR = GaussianMixture.suggest_priors(X, n_components=20)
MEAN_PRIOR, MEAN_PRECISION_PRIOR, COVARIANCE_PRIOR, DEGREES_OF_FREEDOM_PRIOR = GIBBS_PRIOR'''
cell7_src = cell7_src + addition
nb["cells"][7]["source"] = cell7_src.splitlines(keepends=True)
nb["cells"][7]["outputs"] = []
nb["cells"][7]["execution_count"] = None
json.dump(nb, open(NB_PATH, "w"), indent=1)
print("cell 7 patched")
```

Run: `source .venv/bin/activate && python3 <above script>`

- [ ] **Step 3: Patch cell 9 (sweep-by-sweep snapshots)**

```python
import json

NB_PATH = "notebooks/dpgmm_gibbs_sampling.ipynb"
nb = json.load(open(NB_PATH))

cell9_src = "".join(nb["cells"][9]["source"])
old = '''    model = DPGMM(truncation_level=None, max_iter=n_iter, burn_in=0,
                 random_state=RANDOM_STATE, device='cpu', weight_threshold=1.0)'''
new = '''    model = GaussianMixture(n_components=None, max_components=None, max_iter=n_iter, burn_in=0,
                            random_state=RANDOM_STATE, device='cpu', weight_threshold=1.0,
                            mean_prior=MEAN_PRIOR, mean_precision_prior=MEAN_PRECISION_PRIOR,
                            covariance_prior=COVARIANCE_PRIOR, degrees_of_freedom_prior=DEGREES_OF_FREEDOM_PRIOR)'''
assert old in cell9_src, "cell 9's DPGMM construction not found verbatim -- check for drift"
cell9_src = cell9_src.replace(old, new)
nb["cells"][9]["source"] = cell9_src.splitlines(keepends=True)
nb["cells"][9]["outputs"] = []
nb["cells"][9]["execution_count"] = None
json.dump(nb, open(NB_PATH, "w"), indent=1)
print("cell 9 patched")
```

Run: `source .venv/bin/activate && python3 <above script>`

- [ ] **Step 4: Patch cell 12 (component-count trajectory)**

```python
import json

NB_PATH = "notebooks/dpgmm_gibbs_sampling.ipynb"
nb = json.load(open(NB_PATH))

cell12_src = "".join(nb["cells"][12]["source"])
old = '''model = DPGMM(truncation_level=None, max_iter=60, burn_in=0,
             random_state=RANDOM_STATE, device='cpu', weight_threshold=1.0)'''
new = '''model = GaussianMixture(n_components=None, max_components=None, max_iter=60, burn_in=0,
                        random_state=RANDOM_STATE, device='cpu', weight_threshold=1.0,
                        mean_prior=MEAN_PRIOR, mean_precision_prior=MEAN_PRECISION_PRIOR,
                        covariance_prior=COVARIANCE_PRIOR, degrees_of_freedom_prior=DEGREES_OF_FREEDOM_PRIOR)'''
assert old in cell12_src, "cell 12's DPGMM construction not found verbatim -- check for drift"
cell12_src = cell12_src.replace(old, new)
nb["cells"][12]["source"] = cell12_src.splitlines(keepends=True)
nb["cells"][12]["outputs"] = []
nb["cells"][12]["execution_count"] = None
json.dump(nb, open(NB_PATH, "w"), indent=1)
print("cell 12 patched")
```

Run: `source .venv/bin/activate && python3 <above script>`

- [ ] **Step 5: Patch cell 14 (truncated-vs-unbounded agreement)**

```python
import json

NB_PATH = "notebooks/dpgmm_gibbs_sampling.ipynb"
nb = json.load(open(NB_PATH))

cell14_src = "".join(nb["cells"][14]["source"])
old = '''model_trunc = DPGMM(truncation_level=K_SEED, max_iter=50,
                    random_state=RANDOM_STATE, device='cpu', weight_threshold=1.0)
model_trunc.fit(X)

model_unbounded = DPGMM(truncation_level=None, init_k=K_SEED, max_iter=50, burn_in=10,
                        random_state=RANDOM_STATE, device='cpu', weight_threshold=1.0)
model_unbounded.fit(X)'''
new = '''model_trunc = GaussianMixture(n_components=None, max_components=K_SEED, max_iter=50,
                              random_state=RANDOM_STATE, device='cpu', weight_threshold=1.0,
                              mean_prior=MEAN_PRIOR, mean_precision_prior=MEAN_PRECISION_PRIOR,
                              covariance_prior=COVARIANCE_PRIOR, degrees_of_freedom_prior=DEGREES_OF_FREEDOM_PRIOR)
model_trunc.fit(X)

model_unbounded = GaussianMixture(n_components=None, max_components=None, init_k=K_SEED, max_iter=50, burn_in=10,
                                  random_state=RANDOM_STATE, device='cpu', weight_threshold=1.0,
                                  mean_prior=MEAN_PRIOR, mean_precision_prior=MEAN_PRECISION_PRIOR,
                                  covariance_prior=COVARIANCE_PRIOR, degrees_of_freedom_prior=DEGREES_OF_FREEDOM_PRIOR)
model_unbounded.fit(X)'''
assert old in cell14_src, "cell 14's DPGMM constructions not found verbatim -- check for drift"
cell14_src = cell14_src.replace(old, new)
nb["cells"][14]["source"] = cell14_src.splitlines(keepends=True)
nb["cells"][14]["outputs"] = []
nb["cells"][14]["execution_count"] = None
json.dump(nb, open(NB_PATH, "w"), indent=1)
print("cell 14 patched")
```

Run: `source .venv/bin/activate && python3 <above script>`

- [ ] **Step 6: Re-execute the full notebook**

Run: `source .venv/bin/activate && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=580 --ExecutePreprocessor.kernel_name=python3 notebooks/dpgmm_gibbs_sampling.ipynb`

- [ ] **Step 7: Verify clean execution**

Run:
```bash
source .venv/bin/activate && python3 -c "
import json
nb = json.load(open('notebooks/dpgmm_gibbs_sampling.ipynb'))
errors = [(i, o.get('ename'), o.get('evalue')) for i, c in enumerate(nb['cells']) for o in c.get('outputs', []) if o.get('output_type') == 'error']
print('errors:', errors if errors else 'none')
"
```
Expected: `errors: none`.

- [ ] **Step 8: Commit**

```bash
git add notebooks/dpgmm_gibbs_sampling.ipynb
git commit -m "Rework dpgmm_gibbs_sampling.ipynb for merged GaussianMixture(n_components=None) API"
```

---

### Task 12: Documentation updates and final verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-28-dpgmm-design.md`
- Modify: `README.md`
- Modify: `docs/user-guide/clustering-metrics.md`

- [ ] **Step 1: Add a superseded-by pointer to the old DPGMM design doc**

At the very top of `docs/superpowers/specs/2026-07-28-dpgmm-design.md`, immediately after the `# DPGMM: Dirichlet Process Gaussian Mixture Model — Design` title line, insert:

```markdown

> **Superseded.** `DPGMM` was merged into `GaussianMixture` (selected by
> `n_components=None`) and no longer exists as a separate public class --
> see `docs/superpowers/specs/2026-08-11-gmm-dpgmm-merge-design.md` for the
> merge design and this doc's own "Revision history" section for the
> Gibbs-sampling math itself, which the merge did not change. This doc is
> kept for that math and the design rationale behind it.
```

- [ ] **Step 2: Fix README.md**

Read the current `README.md`. Find the line(s) referencing `GaussianMixture`/`DPGMM` as two classes (from the BIC/AIC section added earlier this session: `"- Model Selection: bic/aic methods for comparing fits at different n_components"` and the clustering-metrics bullet mentioning both class names). Update any sentence naming `DPGMM` as a separate class to instead describe `GaussianMixture(n_components=None, ...)`. Since the exact current wording may have shifted since this plan was written, search for the literal string `DPGMM` in the file and adapt each occurrence in place rather than a blind block-replace.

Run: `grep -n "DPGMM" README.md` to find the exact lines before editing.

- [ ] **Step 3: Fix `docs/user-guide/clustering-metrics.md`**

Same approach: `grep -n "DPGMM" docs/user-guide/clustering-metrics.md`, update each occurrence to describe `GaussianMixture(n_components=None, ...)` instead of a separate `DPGMM` class.

- [ ] **Step 4: Repo-wide sanity sweep for stale references**

Run: `grep -rn "DPGMM" --include="*.py" --include="*.md" /home/asp/Downloads/HeaDS/tgmm/ | grep -v "/site/" | grep -v ".git/"`

Expected remaining matches: only the two design spec docs (`2026-07-28-dpgmm-design.md`, which intentionally keeps `DPGMM` in its historical text, and `2026-08-11-gmm-dpgmm-merge-design.md`, which discusses the merge). Anything else (docs, README, code) still saying `DPGMM` needs fixing.

- [ ] **Step 5: Final full verification**

Run: `source .venv/bin/activate && python -m pytest tests/ -q`
Expected: all green, zero failures.

Run: `source .venv/bin/activate && python -c "import tgmm; print([n for n in dir(tgmm) if not n.startswith('_')])"`
Expected: no `DPGMM` in the printed list.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-07-28-dpgmm-design.md README.md docs/user-guide/clustering-metrics.md
git commit -m "Update docs for DPGMM -> GaussianMixture(n_components=None) merge"
```
