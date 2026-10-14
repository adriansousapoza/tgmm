# Phase 4: `BaseMixture`, Student-t, and von Mises-Fisher mixtures — Design

Part of the [tgmm v0.4 roadmap](../../../tgmm%20v0.4%20roadmap%20coding-agent%20plan.md).
Depends on Phase 2 (the `algorithm`/`SufficientStats`/online-EM machinery
being extracted here). Builds on, but is independent of, Phase 3.

## Why this is the highest-risk phase so far

`GaussianMixture` is a 5191-line, 90-method class with 647+ tests
(including, from Phase 0, an exact bit-for-bit fixture-parity check across
every covariance type x {MLE, MAP, NIW}). The plan's own acceptance
criterion for the rebase is explicit: *"the entire existing
`GaussianMixture` test suite passes unchanged, and seeded fits are
bit-identical in float64."* That existing suite (particularly
`test_phase0.py::test_matches_pre_refactor_fixture`) already **is** the
bit-identical check the plan wants -- no new fixture is needed for this
phase; running the full suite after the rebase and requiring zero
failures *is* the acceptance test.

## Scope decisions: what's genuinely shared vs. where `GaussianMixture` keeps its own code

The plan lists `fit`, `sample`, and `save`/`load`/`save_state_dict`/
`load_state_dict` as "concrete, shared" in `BaseMixture`. Read literally,
"shared" could mean either (a) one implementation every subclass uses
as-is, or (b) a base implementation subclasses are free to extend/override
-- ordinary OOP, and no less "shared" in the sense that matters (one real
piece of logic to maintain, not three independent copies). This phase
takes reading (b) for the three methods below, and documents why:

- **`fit`**: `BaseMixture.fit` implements the generic n_init loop
  (validation, per-init seeding, `_fit_single_run`, best-run tracking via a
  new `_get_fit_snapshot`/`_set_fit_snapshot` hook pair -- see below,
  restoring the best run). `GaussianMixture.fit` overrides it to add its
  two pieces of behaviour that don't generalize to other families at all
  -- Gibbs-sampling dispatch (`n_components=None`) and supervised fitting
  (`labels=...`) -- then calls `super().fit(...)` for the ordinary
  EM/CEM/SEM/SAEM/online path. This *is* "one shared loop, not three": the
  n_init/best-run machinery that Student-t and vMF actually need lives in
  exactly one place.
- **`sample`**: `BaseMixture.sample` is a plain shell (draw which
  component each sample comes from via `weights_`, call the abstract
  `_sample_component(k, n)` per component, concatenate) -- enough for
  Student-t and vMF's much simpler sampling needs (see 4.2/4.3 below).
  `GaussianMixture.sample` keeps its existing, much richer implementation
  untouched (confidence regions, rejection sampling by Mahalanobis
  distance or std-dev range, a specific center point, etc. -- none of
  which the plan asks any other family to support) rather than trying to
  force those Gaussian-specific features into a shape every family must
  carry. `GaussianMixture` still implements `_sample_component(k, n)` (a
  new, small, independent method -- a single-component multivariate
  draw) so the abstract contract is genuinely satisfied, even though
  `GaussianMixture`'s own `sample()` doesn't route through it.
- **`save`/`load`/`save_state_dict`/`load_state_dict`/`to_dict`**: **not**
  generalized into `BaseMixture` in this phase. `GaussianMixture` keeps
  its own, completely unchanged (every field name, every
  backward-compatible `.get(...)` fallback for older save files is
  exactly as it was pre-Phase-4) -- this is the one deliberate, flagged
  deviation from the plan's literal list, made because (a) designing a
  truly generic serialization scheme that still round-trips
  `GaussianMixture`'s ~40 persisted fields byte-for-bit identically is a
  substantial, separate design problem in its own right, and (b) getting
  it wrong would silently break every previously-saved `.pt` file, which
  is a much worse failure mode than two families having their own
  (smaller, simpler) save/load pair. `StudentTMixture` and
  `VonMisesFisherMixture` get their own small, independent
  `save`/`load`/`save_state_dict`/`load_state_dict` implementations
  (their parameter sets are far smaller -- no priors, no Gibbs mode, no
  GPCM -- so this isn't the same undertaking for them).

Genuinely shared, used as-is by every subclass, no override needed:
`_m_step` (the one-line `_m_step_from_stats(_compute_stats(...))`
wrapper), `_e_step`/`_e_step_chunk` (via the abstract
`_component_log_prob`), `_c_step`, `_s_step`, `_fit_single_run` and its
SEM/online/SAEM variants, `_beta_for_iteration`, `_check_convergence`,
`_blend_stats`, `_resolve_step_size_fn`, `partial_fit`, `predict`,
`predict_proba`, `score_samples`, `score`, `bic`, `aic`, `icl`, and the new
`_gem_step` utility.

`bic`/`aic`/`icl` reach the Gibbs-mode active-components pruning through
two hooks that already existed pre-Phase-4 for exactly this purpose --
`_score_for_information_criterion(X)` and `_resp_for_information_criterion(X)`
-- which just needed a trivial default added to `BaseMixture`
(`self.score(X)` / plain `self._e_step(X)[0]`, respectively).
`GaussianMixture` overrides both with its existing Gibbs-aware
implementations, unchanged; `_active_score`, `_active_e_step`, and
`_is_pruned_gibbs_fit` (the private helpers those two overrides use) stay
on `GaussianMixture` entirely -- they assume `weights_`/`means_`/
`covariances_` are the complete parameter set to swap for the
active-only subset, which only holds for a Gaussian mixture (a vMF
mixture's parameters have different names and shapes entirely), so they
were never candidates for `BaseMixture` in the first place.

## Abstract methods (`BaseMixture`)

Using existing `GaussianMixture` method names as the contract wherever one
already exists, rather than introducing parallel names and aliasing them
(lower risk: no rename touches `GaussianMixture`'s own 90-method surface):

- `_component_log_prob(X) -> (n, K)`: per-component log-density *before*
  adding `log(weights_)` or tempering by `beta`. `GaussianMixture`
  implements it as the covariance-type dispatch
  `_e_step_chunk` used to do directly (moved here verbatim, the six
  `_estimate_log_gaussian_*` helper methods stay on `GaussianMixture`
  unchanged and `_component_log_prob` just dispatches to them, plus the
  new GPCM branch from Phase 3).
- `_compute_stats(X, resp, batch_size=None) -> SufficientStats`:
  `GaussianMixture`'s existing implementation, unchanged.
- `_m_step_from_stats(stats)`: `GaussianMixture`'s existing implementation,
  unchanged.
- `_allocate_parameters(X=None, set_random_state=True)`: `GaussianMixture`'s
  existing implementation, unchanged (this is the plan's "`_init_params`" --
  kept under its existing name rather than introduced as a second,
  aliased name).
- `_n_parameters() -> int`: `GaussianMixture`'s existing implementation,
  unchanged.
- `_sample_component(k, n) -> torch.Tensor`: new on every family
  (`GaussianMixture`'s is a small new method per above; it does not
  replace `GaussianMixture.sample`'s own rich implementation).

## `_get_fit_snapshot`/`_set_fit_snapshot`: generalizing the n_init "keep the best run" step

The pre-Phase-4 `fit()` hardcodes a tuple of exactly `(weights_, means_,
covariances_, converged_, n_iter_, lower_bound_, log_likelihood_history_,
annealing_schedule_, n_empty_components_, _online_running_stats_,
_online_step_count_)` to clone/restore. Student-t has one more fitted
tensor (`dof_`); vMF has an entirely different parameter set. Rather than
hardcode a family-specific tuple inside the shared loop, `BaseMixture`
defines:

```python
def _get_fit_snapshot(self) -> dict:
    return {
        'weights_': self.weights_.clone(), 'means_': self.means_.clone(),
        'covariances_': self.covariances_.clone(),
        'converged_': self.converged_, 'n_iter_': self.n_iter_,
        'lower_bound_': self.lower_bound_,
        'log_likelihood_history_': self.log_likelihood_history_,
        'annealing_schedule_': self.annealing_schedule_,
        'n_empty_components_': self.n_empty_components_,
        '_online_running_stats_': self._online_running_stats_,
        '_online_step_count_': self._online_step_count_,
    }

def _set_fit_snapshot(self, snapshot: dict):
    for key, value in snapshot.items():
        setattr(self, key, value)
```

`GaussianMixture` needs **no override** -- the default dict's keys are
exactly its own existing tuple, so the rebase changes nothing about what
gets cloned/restored for it. `StudentTMixture` overrides
`_get_fit_snapshot` to also clone `dof_`; `VonMisesFisherMixture` overrides
it with its own `mean_directions_`/`concentration_`.

The same generalization problem shows up a second time inside
`_fit_single_run_sem` (SEM's post-burn-in `'best'`/`'mean'` selection also
hardcoded a `(weights_, means_, covariances_, ll)` tuple). Factored the
fitted-parameter-only subset out of `_get_fit_snapshot` into its own
`_get_param_snapshot()`/`_apply_snapshot(snapshot)` pair (bookkeeping
fields like `converged_`/`log_likelihood_history_` don't belong in a
"snapshot to average across SEM sweeps"); `_get_fit_snapshot` is now just
`_get_param_snapshot()` plus the bookkeeping keys. SEM's `'mean'` option
then averages generically over every key in the returned dict
(`torch.stack([s[key] for s in snapshots]).mean(dim=0)`), which works
unchanged for any number/shape of parameter tensors a subclass's override
adds -- including fixing a latent gap this surfaced: without this,
`algorithm='sem'` on a future `StudentTMixture` would silently never
track/restore `dof_` through SEM's best/mean selection.

## `_gem_step`: generalizing Phase 3's GEM/Stiefel pattern

`tgmm/covariance/gpcm.py`'s `_optimize_orientation` (Phase 3) is kept
exactly as-is (it is already correct, tested, and doesn't need
`BaseMixture` to exist). `BaseMixture._gem_step(objective, params,
n_steps=20)` is a new, more general version of the same idea (objective
closure over arbitrary leaf tensors, `torch.optim.LBFGS`, accept only on
strict improvement) for any family needing a non-closed-form update on
*unconstrained* parameters -- neither `StudentTMixture` nor
`VonMisesFisherMixture` actually needs it (both use closed forms / Newton
root-finding for their one non-trivial parameter, `nu_k`/`kappa_k`), so
it's exercised by its own direct unit test here and left for the
Phase 4.4 `GenericMixture` stretch goal (not implemented this phase, see
Non-goals) or a future family to use.

## 4.2 `StudentTMixture`

Implemented per the plan's own equations (Peel & McLachlan, 2000):
log-density, the `u_ik` weight, the ECM mean/scale updates, and the
digamma-based `nu_k` equation, solved by bisection-then-Newton on `log nu`
within `dof_bounds` using `torch.special.digamma`/`torch.special.polygamma`.

Deviation from the plan text above: rather than importing
`GaussianMixture`'s Mahalanobis/log-det computations (which are folded
into its `_estimate_log_gaussian_*` methods, each also baking in the
*Gaussian* normalizing constant -- not a separable "Mahalanobis + logdet"
piece to begin with), `tgmm/mixtures/student_t.py` implements its own
`_mahalanobis_and_logdet(X, means, covariances, covariance_type)` free
function, using the same samples-as-right-hand-side-columns
`cholesky_solve` technique Phase 0 used in `gmm.py` for the same memory
reason. This keeps `GaussianMixture` (already bit-identical-verified
against its pre-rebase behavior) completely untouched by this file's
existence, at the cost of a small, independently-tested amount of
duplicated linear algebra (about 30 lines) rather than extracting shared
free functions out of already-verified code. Verified against
`scipy.stats.multivariate_t.logpdf` directly (~1e-6 agreement) rather than
transitively through `GaussianMixture`.

`SufficientStats.aux: dict[str, Tensor]` (new field, added in this phase
since it's genuinely only needed starting here -- Phase 0 flagged this as
deliberately deferred) carries the `u`-weighted accumulators
(`sum_ik tau*u`, `sum_ik tau*u*x`, `sum_ik tau*u*(x-mu)(x-mu)^T`, `sum_ik
tau*(log u - u)`) a plain Gaussian `SufficientStats` has no slot for.

## 4.3 `VonMisesFisherMixture`

Implemented per Banerjee, Dhillon, Ghosh & Sra (2005) and Sra (2012) as
specified in the plan. `_log_bessel_iv` uses `scipy.special.ive` with the
asymptotic fallback for underflow, tested against `mpmath.besseli` (a
test-only dependency, as the plan specifies) at `d in {2, 10, 128, 768}`
(max observed deviation 2.2e-4, at d=768, well inside the asymptotic-
fallback regime).

Bug found and fixed during implementation: `init_means='kmeans'`'s
spherical k-means (cosine similarity in place of Euclidean distance) first
seeded its `k` initial centers by picking `k` rows uniformly at random
outright. On well-separated synthetic data (3 true vMF components, d=50)
this reliably drew two initial centers from the same true cluster,
permanently merging the other two once EM converged (ARI ~0.46). Fixed by
seeding with cosine-distance k-means++ instead (first center uniform
random, each subsequent center drawn with probability proportional to
`1 - max cosine similarity to any center already chosen`) -- the same
fix ordinary Euclidean k-means needs for the same failure mode. ARI=1.0
across 5 random seeds on the same synthetic data afterward.

## Non-goals for this phase

- `GenericMixture` (4.4): explicitly optional/stretch in the plan ("if
  time allows"). Not implemented -- `_gem_step` is built and tested on its
  own so a future session can add `GenericMixture` without redoing that
  groundwork, but the class itself, and its Laplace-mixture example, are
  left for later. Tracked the same way the plan tracks other out-of-scope
  items.
- Generic `save`/`load` on `BaseMixture` (see Scope decisions above) --
  `GaussianMixture`'s stays exactly as it was; `StudentTMixture`/
  `VonMisesFisherMixture` get their own.
- Gibbs mode for Student-t/vMF (the plan already scopes this to
  Gaussian-only).
