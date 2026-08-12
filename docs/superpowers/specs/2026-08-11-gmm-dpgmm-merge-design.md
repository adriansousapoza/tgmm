# Merge DPGMM into GaussianMixture — Design

## Background

`tgmm.DPGMM` (`tgmm/dpgmm.py`, see
`docs/superpowers/specs/2026-07-28-dpgmm-design.md`) was deliberately built as
a standalone class, copied from `GaussianMixture`'s scaffolding rather than
inheriting from it, specifically so the two could evolve independently while
DPGMM's Gibbs-sampling math was still being worked out. That math is now
correct and tested (collapsed Gibbs sampling, per-covariance-type NIW
posteriors, the pooled-within-cluster-scatter covariance-prior fix, the
`init_k` shared-starting-point mechanism) and `notebooks/dpgmm.ipynb` /
`notebooks/dpgmm_gibbs_sampling.ipynb` have exercised it thoroughly.

The two-class split is no longer paying for itself: a user who wants "fit a
Gaussian mixture, and let the number of components be inferred instead of
fixed" has to know a second class exists, with a mostly-parallel but
not-quite-identical API (`truncation_level` instead of `n_components`,
different covariance-type support, separate `bic`/`aic`, etc.). This design
merges `DPGMM`'s functionality into `GaussianMixture` itself, selected by
`n_components=None`, and removes `DPGMM` as a public class.

## Goals

- `GaussianMixture(n_components=None, ...)` fits via collapsed Gibbs sampling
  (Neal's Algorithm 3) instead of EM/CEM — truncated (`max_components=K`) or
  fully unbounded (`max_components=None`).
- `GaussianMixture(n_components=K, ...)` (the existing, unchanged API) keeps
  fitting via EM/CEM exactly as it does today. **No behavior change for
  existing callers.**
- The post-fit API (`predict`, `predict_proba`, `score`, `score_samples`,
  `sample`, `bic`, `aic`, `save`, `load`) works identically regardless of
  which mode produced the fitted `weights_`/`means_`/`covariances_`.
- Gibbs sampling requires an NIW prior to run at all (it's how Neal's
  Algorithm 3 integrates out means/covariances) and prior choice has been
  shown, in this project, to change results substantially depending on
  data scale (see the covariance-prior fix in the DPGMM design doc's
  revision history). Rather than silently resolving a default (as `DPGMM`
  did), the merged class **requires the caller to supply the full NIW prior
  explicitly** when using Gibbs mode, and raises immediately if they don't.
- `tgmm.DPGMM` is removed. `tgmm/dpgmm.py` is deleted; its logic moves into
  `tgmm/gmm.py`.

## Non-goals

- No change to EM/CEM's own default behavior: `GaussianMixture(n_components=K)`
  with no prior arguments still runs plain MLE, exactly as today.
- No attempt to make Gibbs mode support `tied_full`/`tied_diag`/`tied_spherical`
  covariance — still `NotImplementedError`, same underlying reason as the
  original DPGMM design (a shared covariance doesn't factor per-component
  under Algorithm 3). Checked at `__init__` now instead of `fit()`.
- No change to the Gibbs sampling algorithm itself (weight rules, NIW
  posterior formulas, `init_k` mechanism) — this is a relocation and API
  change, not a math change.

## Constructor changes

- **`n_components: Optional[int] = None`** *(was a required int, no
  default)*. An int selects EM/CEM — identical to today. `None` selects
  Gibbs sampling.
- **`max_components: Optional[int] = 20`** *(new; replaces DPGMM's
  `truncation_level`, renamed to read naturally next to `n_components`)*.
  Only consulted when `n_components=None`. An int caps Gibbs at that many
  slots (truncated sampler). `None` runs the unbounded sampler (dynamic
  component count). Ignored in EM mode.
- **`alpha: float = 1.0`, `burn_in: Optional[int] = None`,
  `weight_threshold: Optional[float] = None`, `init_k: Optional[int] = None`**
  — ported from `DPGMM` unchanged (same semantics, same defaults). Gibbs-only;
  ignored in EM mode.
- **`covariance_type`**: validated at `__init__` now, not `fit()`. If
  `n_components=None` and `covariance_type` is one of the three `tied_*`
  variants, raises `NotImplementedError` immediately (data-independent
  check, so no reason to defer it to `fit()` the way DPGMM did).
- **NIW prior params** (`mean_prior`, `mean_precision_prior`,
  `covariance_prior`, `degrees_of_freedom_prior`): unchanged in EM mode
  (all optional; MLE if omitted). **In Gibbs mode, all four are required.**
  `__init__` raises `ValueError` immediately if any is `None`, naming which
  argument(s) are missing, and pointing at the new
  `GaussianMixture.suggest_priors(X, n_components)` static method as
  the way to get a starting point — plus a one-line reason: Gibbs sampling
  has no principled default, and an auto-resolved one measurably changed
  fitted covariances in practice during this project's development (a
  74x mismatch between marginal and within-cluster covariance scale on
  well-separated data).
- **`GaussianMixture.suggest_priors(X, n_components)`** *(new)* — the
  pooled-within-cluster-scatter estimator built for `DPGMM`
  (`_pooled_within_cluster_covariance_prior`), exposed as an explicit,
  opt-in helper rather than an automatic default. Runs a quick k-means
  partition into `n_components` groups and returns
  `(mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior)`
  — a starting point the caller inspects/adjusts and passes back in
  explicitly. Nothing calls this automatically.

### Mode-mismatched params

Neither direction raises: `n_init`/`warm_start` (EM-only) are silently
ignored in Gibbs mode; `max_components`/`alpha`/`burn_in`/`weight_threshold`/
`init_k` (Gibbs-only) are silently ignored in EM mode. Unlike the missing-prior
case, there's no correctness risk from a stray mode-mismatched kwarg — just
no effect — so no need to force the caller to scrub these out.

## Fitting behavior

`fit(X)` branches once, at the top, on `self.n_components is None`:

- **EM mode**: unchanged internals (`_fit_single_run`, `n_init` restarts,
  `_m_step`/`_e_step`, etc.) — zero changes to this path.
- **Gibbs mode**: calls `_fit_gibbs_truncated` or `_fit_gibbs_unbounded`
  (ported from `DPGMM`, `max_components` replacing `truncation_level`),
  which populate `weights_`/`means_`/`covariances_`/`n_components_`/`active_`
  exactly as `DPGMM` does today.

Downstream of fitting, almost nothing needs mode-awareness: `predict`,
`predict_proba`, `score`, `score_samples`, and `sample` only ever read
`weights_`/`means_`/`covariances_`/`covariance_type` — they don't know or
care which algorithm produced them, so none of them need branching.
`save`/`load` need the new constructor args added to the round-trip dict
(mechanical).

## Attribute unification

After `fit()`, regardless of mode:

- **`n_components_`**: Gibbs mode — the inferred active count (unchanged
  from `DPGMM`). EM mode — **new**, set to `self.n_components` (the fixed
  value), so code reading `.n_components_` works uniformly.
- **`active_`**: Gibbs mode — unchanged (the `weight_threshold` mask). EM
  mode — **new**, `torch.ones(n_components, dtype=torch.bool)` (every fixed-K
  component counts active; EM has no pruning concept).
- **`_n_parameters()`**: collapses to a single implementation that always
  reads `self.n_components_` (no more separate EM-fixed-count vs.
  Gibbs-active-count branching). `bic()`/`aic()` need no mode-awareness
  after this.

## Code organization

Direct port: `DPGMM`'s internals (per-covariance-type NIW posteriors,
Student-t marginal likelihoods, `_gibbs_*` sufficient-statistic helpers,
`_resolve_priors_gibbs` minus its auto-default computation,
`_fit_gibbs_truncated`/`_fit_gibbs_unbounded`,
`_pooled_within_cluster_covariance_prior` renamed/exposed as
`suggest_priors`) move into `gmm.py` largely as-is. `gmm.py` grows from
~2,500 to roughly ~4,500 lines as a single class — the highest-risk
alternative (a separate mixin file) was considered and rejected in favor of
matching this codebase's existing single-large-class convention and
minimizing the chance of introducing new bugs while relocating
already-debugged math.

`tgmm/dpgmm.py` is deleted. `tgmm/__init__.py` drops the `DPGMM` export.

## Consequences / migration scope

- **Four test files rewritten**: `test_dpgmm.py`, `test_dpgmm_gibbs.py`,
  `test_dpgmm_gibbs_unbounded.py`, `test_dpgmm_covariance_calibration.py`
  (~600 lines combined) — every `DPGMM(...)` call becomes
  `GaussianMixture(n_components=None, ...)`, and every one needs explicit
  prior args added (none currently pass priors; all relied on the removed
  auto-default).
- **Both notebooks reworked, not just find-replaced**: every `DPGMM(...)`
  call site in `dpgmm.ipynb` (Rounds 1-4) and `dpgmm_gibbs_sampling.ipynb`
  (all 6 sections) currently fits with zero explicit priors. Each needs a
  `suggest_priors(X, K)` call threaded in first, then both notebooks need
  full re-execution. This is the largest single cost of the
  no-silent-default decision, and is done in the same pass as the rest of
  this plan (not deferred).
- **`docs/superpowers/specs/2026-07-28-dpgmm-design.md`**: gets a short
  "superseded by this doc" pointer added at the top rather than being
  deleted, consistent with how this repo keeps historical spec docs (e.g.
  the supervised-gmm design doc remains after that feature shipped).
- **README.md / `docs/user-guide/clustering-metrics.md`**: minor text
  fixes anywhere they say "`GaussianMixture`/`DPGMM`" as two classes.
  `DPGMM` was never wired into `mkdocs.yml` nav or given a `docs/api/`
  page, so there's no nav/API-doc cleanup needed there.

## Testing strategy

- Existing `DPGMM` test coverage (NIW posterior correctness per covariance
  type, truncated/unbounded end-to-end fits, `init_k` equivalence, the
  covariance-calibration regression tests) is preserved, ported to call
  `GaussianMixture(n_components=None, ...)` with explicit priors instead of
  `DPGMM(...)`.
- New tests for the merge itself: `__init__` raises on missing priors in
  Gibbs mode (per missing argument), raises on `tied_*` + Gibbs mode,
  `n_components_`/`active_` unification for EM-mode fits, `suggest_priors`
  returns a usable prior tuple, mode-mismatched params are silently
  accepted rather than erroring.
- Full existing EM/CEM test suite (`test_gmm.py` and friends) must pass
  unchanged — this is the regression bar for "EM mode has zero behavior
  change."
