# DPGMM: Dirichlet Process Gaussian Mixture Model — Design

> **Superseded.** `DPGMM` was merged into `GaussianMixture` (selected by
> `n_components=None`) and no longer exists as a separate public class --
> see `docs/superpowers/specs/2026-08-11-gmm-dpgmm-merge-design.md` for the
> merge design and this doc's own "Revision history" section for the
> Gibbs-sampling math itself, which the merge did not change. This doc is
> kept for that math and the design rationale behind it.

## Background

`tgmm.GaussianMixture` (`tgmm/gmm.py`) already accepts a `weight_concentration_prior` and
computes a MAP weight update under a **finite symmetric Dirichlet prior**
(`gmm.py:1391-1398`, `weights_ = (nk + alpha - 1) / (n_samples + total_alpha - n_components)`).
This has no mechanism to shrink unused components toward zero weight — it just
reweights whatever `n_components` was given. This is the root cause of the
previously-observed failure mode: fitting kept using as many components as
were allocated, regardless of how many clusters were actually present in the
data.

A true Dirichlet Process treatment needs a stick-breaking (or CRP) prior over
mixture weights, where the posterior itself can push a component's expected
weight toward ~0 when the data doesn't support it. This design adds that as a
new, standalone class: **`DPGMM`**.

## Goals

- Automatically infer the effective number of mixture components from data,
  rather than requiring the caller to fix it in advance.
- Fit via collapsed Gibbs sampling (Neal's Algorithm 3), either truncated at
  a fixed `truncation_level` or fully unbounded.
- Keep `GaussianMixture` completely untouched; `DPGMM` is a new, independent
  class that happens to start from a copy of `gmm.py`'s scaffolding.

## Non-goals (v1)

- Auto-inferring the concentration parameter `alpha` (fixed hyperparameter
  only; larger `alpha` favors more components).
- GPU-parallelized Gibbs sweeps. The per-point resampling loop is inherently
  sequential; it may be slow for large `n_samples`. Acceptable for a
  comparison exercise, not claimed as a production-scale path.
- Warm-start / incremental fitting.
- Gibbs sampling (both truncated and unbounded) for `tied_full`,
  `tied_diag`, `tied_spherical` covariance types (see "Known scope caveat"
  below). A shared/tied covariance does not factor per-component under
  Algorithm 3, so these three types raise `NotImplementedError` at fit time
  regardless of `truncation_level`.

## Revision history

- **Variational (mean-field stick-breaking) inference was removed.** An
  earlier version of this design and of `DPGMM` supported
  `inference='variational'` alongside Gibbs, specifically so the two could
  be compared on which recovers the true component count more reliably (the
  motivating question behind this whole class). That comparison was run
  (see `notebooks/dpgmm.ipynb`): on overlapping data, variational
  consistently over-estimated the component count (e.g. 12-25 components
  reported on data with 5-20 true clusters), while both Gibbs variants
  tracked the true count much more closely. Variational inference,
  `_fit_variational`, its stick-breaking-expectation NIW M-step, and the
  `inference` constructor parameter were deleted; `DPGMM` is Gibbs-only
  going forward. If variational is ever wanted again, recover it from git
  history rather than re-deriving it.
- **Truncated Gibbs's weight rule was corrected.** The original truncated
  sampler reused the variational method's expected-log-weight formula
  (digamma of stick-breaking Beta parameters) in place of proper CRP
  occupancy counts. This was inconsistent with unbounded Gibbs (which uses
  `log(n_k)`) and could make the two disagree even when both reported far
  fewer components than `truncation_level` — see "Inference: Gibbs
  sampling" below for the fix and why it matters.
- **`covariance_prior_`'s data-dependent default was corrected.** When not
  supplied, it was resolved from the marginal covariance of the whole
  dataset (`torch.cov(X.T)`) — for well-separated multi-modal data this
  conflates *between*-cluster spread with *within*-cluster spread (a
  10-d, 5-cluster, well-separated synthetic case measured a 74x mismatch:
  marginal trace ~745 vs. true within-cluster trace 10), which leaked
  into every component's posterior via `psi_n = psi0 + S + cross_term`
  and inflated every fitted covariance — *worse* the more separated the
  true clusters were (well-separated trace ~16 vs. classical GMM's ~9.7;
  overlapping trace ~9.9, already close). The default `mean_precision_prior`
  (previously `1.0`) compounded this: the mean-prior cross-term's
  coefficient `(lambda0 * n_k) / (lambda0 + n_k)` converges to `lambda0`
  (not 0) as `n_k -> infinity`, so it never shrank away when a
  well-separated cluster's own mean sat far from the prior's global mean.
  Fix: `covariance_prior_` is now the pooled within-cluster scatter of the
  initial k-means partition (`_pooled_within_cluster_covariance_prior`,
  the pooled-variance/ANOVA estimator), and `mean_precision_prior`
  defaults to `0.1`. This is a property of the shared NIW conjugate math
  (`psi_n = psi0 + S + cross_term`, `Sigma = psi_n / nu_n`), not something
  specific to Gibbs sampling — `GaussianMixture` has the identical defect
  when given equivalent explicit priors, it just never auto-resolves
  defaults, so the path went unexercised there. See
  `tests/test_dpgmm_covariance_calibration.py`.
- **`init_k` was added** to give unbounded Gibbs a caller-controlled
  starting `K`, independent of `truncation_level` (which stays `None` for
  unbounded) and independent of the automatic
  `_resolve_truncation_level` heuristic (`max(2, min(20, n_samples // 2))`,
  which is coupled to `n_samples` and can silently differ from whatever
  `truncation_level` a paired truncated run uses). Setting
  `truncation_level=K` on one run and `truncation_level=None, init_k=K` on
  another gives both the identical k-means starting partition for the
  same `random_state` — see "Why the two rules were made to match" below
  for what this does and doesn't guarantee.

## Module & class

New file `tgmm/dpgmm.py`, new class `DPGMM(nn.Module)`. Copied scaffold from
`GaussianMixture` (parameter allocation, `predict`, `predict_proba`,
`score_samples`, `sample`, `save`/`load`) — same patterns, but no inheritance
from or import-coupling to `gmm.py`. Math that's shared in spirit (Gaussian
log-density formulas, NIW-conjugate updates) is copied into `dpgmm.py`, not
imported, so the two classes can evolve independently.

## Constructor

- `truncation_level: Optional[int] = 20` — replaces `GaussianMixture`'s
  `n_components`. This is `K_max`, the number of component "slots" allocated
  for truncated Gibbs. `None` triggers the unbounded sampler (dynamic
  component count, no fixed cap; internally seeded via
  `_resolve_truncation_level`, see below).
- `init_k: Optional[int] = None` — ignored when `truncation_level` is an
  int (truncated already gets its starting `K` directly from
  `truncation_level`). When `truncation_level=None`, overrides
  `_resolve_truncation_level`'s automatic seed with a caller-chosen `K`
  for the initial k-means partition — the mechanism for giving a
  truncated/unbounded pair the same starting point (see "Why the two
  rules were made to match" below).
- `alpha: float = 1.0` — DP concentration parameter, fixed hyperparameter.
- `mean_prior`, `mean_precision_prior`, `covariance_prior`,
  `degrees_of_freedom_prior` — same NIW-prior arguments as `GaussianMixture`,
  resolved to data-dependent defaults at fit time if left unset (prior mean
  = data mean, weak precision/dof) since Gibbs sampling is conjugate-NIW
  based throughout. These are a fixed prior, not learned; what Gibbs
  sampling infers, per component, every sweep, is the NIW *posterior*
  `(mu_n, lambda_n, nu_n, Psi_n)` given that fixed prior and whichever
  points currently belong to the component.
- `covariance_type` — same 6 options as `GaussianMixture`: `full`, `diag`,
  `spherical`, `tied_full`, `tied_diag`, `tied_spherical`. Only the first
  three are supported by Gibbs sampling (see Non-goals).
- `weight_threshold: float` — used post-fit to decide which slots count as
  "active" (default: expected count under the fitted weight < 1, i.e.
  `weights_ * n_samples < 1`).

### Fitted attributes

- `weights_`, `means_`, `covariances_` — same semantics as `GaussianMixture`,
  but only entries above `weight_threshold` are considered meaningfully
  "used". Truncated Gibbs always returns exactly `truncation_level` slots
  (many possibly near-zero); unbounded Gibbs may return a different number
  of slots than it started with.
- `n_components_: int` — the inferred effective component count.
- `active_: torch.BoolTensor` — mask over slots indicating which are
  "active" per `weight_threshold`.

## Inference: Gibbs sampling (collapsed, Neal's Algorithm 3)

Both truncated and unbounded modes integrate out means/covariances
analytically (NIW conjugacy), using Student-t predictive marginal
likelihoods, differing only in whether components are capped:

**Truncated** (`truncation_level` set to an int, `K`): `K` fixed slots.
Each sweep resamples every point's assignment from its full conditional
over the `K` slots (some may sit empty):

```
p(z_i = k | z_-i) propto (n_k^{-i} + alpha/K) * marginal_lik(x_i | table k)
```

the finite-dimensional Dirichlet-multinomial (Polya urn) approximation to
the CRP — exact for a symmetric `Dirichlet(alpha/K, ..., alpha/K)` prior
over the `K` weights, and converging to the true CRP rule as `K` grows
relative to the number of clusters actually in use. This is recomputed
fresh for every point (not just once per sweep) from the current
leave-one-out counts, then slots are re-sorted by descending occupancy at
the end of each sweep (a bookkeeping convenience only — the rule is
invariant under simultaneous relabeling of slots).
`weights_`/`means_`/`covariances_` are reported from the *final* sweep
only, not averaged across sweeps -- since sorting reassigns slot indices
every sweep, slot `k` is not the same semantic component from one sweep to
the next, so naively averaging `occupancy[k]` across sweeps can silently
split one real cluster's mass across two reported slots (observed
directly: a 10-cluster fit reporting `n_components_=11` with weights
`[100]*9 + [77.5, 22.5]` per 1000 points, while the actual final partition
-- and everything `predict()` returns -- was a clean 10-way split).

**Unbounded** (`truncation_level=None`): dynamic component count. Per
point: remove it from its current component, score each currently-active
component's Student-t marginal likelihood weighted by `log(n_k)` (the CRP
"sit at an occupied table" term), plus a "new component" term
`log(alpha) + marginal_likelihood(x_i | prior alone)`, sample the
assignment, instantiate a new component if chosen, delete any component
that becomes empty.

**Why the two rules were made to match (and why runs can still differ).**
An earlier version of the truncated sampler used the variational method's
expected-log-weight formula (digamma of stick-breaking Beta parameters,
a deterministic function of the *sorted* counts) in place of `n_k^{-i} +
alpha/K`. That formula does not converge to the CRP rule as `K` grows, so
truncated and unbounded Gibbs could disagree even when both reported far
fewer components than `K` — which is not supposed to happen, since a
generous truncation should behave like the untruncated process. Fixing
the weight rule (as above) was verified to resolve this: given the *same*
initial `K` for the kmeans seeding both samplers use, truncated and
unbounded Gibbs often produce bit-for-bit identical partitions for a given
`random_state` (checked on 20-true-cluster, overlapping 10-d data across 3
seeds). Runs can still differ when the two are configured with
*different* initial `K` (e.g. a generous `truncation_level` for truncated
vs. unbounded's own internal seed from `_resolve_truncation_level`) --
Gibbs samplers for mixture models are well known to merge components
easily but split them only slowly (splitting a merged cluster requires
several points to independently, coincidentally choose "start a new
table" before the new table gains enough support to hold), so a different
starting partition can lead to a different local structure within a
finite number of sweeps. This is expected sampler behavior, not a bug --
see `notebooks/dpgmm.ipynb` for a worked comparison.

**`init_k` removes the *different starting point* source of divergence,
but not the only one.** Passing the same value to one run's
`truncation_level` and another's `init_k` guarantees an identical initial
k-means partition (see "Constructor" above) -- but a broader check across
all 9 `(round, true_k)` combinations in `notebooks/dpgmm.ipynb` (3 spread
regimes x `true_k` in {5, 10, 20}, `K_seed=30`) found exact (ARI=1.0)
agreement in 6 of 9, and partial agreement (ARI 0.91-0.99) in the
remaining 3 -- all in the harder regimes (well-separated `true_k=20`,
and the non-Gaussian round). The two rules (`n_k + alpha/K` over `K`
fixed slots vs. `n_k` over occupied tables plus a separate `alpha`-weighted
new-table term) converge to each other as `K` grows relative to the true
count, but aren't identical move-for-move -- so even from an identical
starting partition, enough stochastic sweeps under two slightly different
rules can occasionally settle into different local structure. Treat
`init_k` as removing one confound, not as a guarantee of identical
output.

### Known scope caveat: `tied_*` + Gibbs (revised)

`tied_full`/`tied_diag`/`tied_spherical` covariance is a single global
parameter shared across all components, not something that marginalizes
independently per component. Fully-collapsed Algorithm 3 requires
`p(x_i | X_k^{-i})` to depend only on component `k`'s own points; a shared
covariance couples every component's residuals together, so it does not
factor per component — **this breaks truncated Gibbs, not only unbounded
Gibbs** (the original design underestimated this: fixing `K_max` does not
fix the coupling problem). A correct treatment would need a
*partially-collapsed* sampler (explicitly resample the shared covariance
from its Inverse-Wishart posterior once per sweep, conditional on the
current assignments, then collapse only the per-component means) — a
different sampler from the one built here, not a variant of it.

Scope for v1: fitting a `tied_*` covariance type raises `NotImplementedError`
regardless of whether `truncation_level` is an int or `None`, with a message pointing at
this caveat.

## Comparison harness

`notebooks/dpgmm.ipynb` (not a hard-assertion unit test, since Gibbs
sampling is inherently stochastic): generates synthetic Gaussian blobs at
`true_k in {5, 10, 20}` in `d=10`, well-separated and overlapping, plus a
round of non-Gaussian cluster shapes, fits `DPGMM` under truncated and
unbounded Gibbs, benchmarks both against a classical fixed-`K`
`GaussianMixture` at the DPGMM-inferred `K`, and reports `n_components_`,
`adjusted_rand_score`, and log-likelihood against the ground-truth labels
for each. Purpose: directly compare which configuration recovers the true
component count most reliably, and how that compares to knowing the
answer and fitting a classical GMM at it — the original motivating
question.

## Testing strategy

- Per-covariance-type unit tests for the log-density and Student-t marginal
  likelihood formulas, checked against direct numerical recomputation.
- End-to-end smoke tests per inference mode on well-separated synthetic
  data: fixed random seed, loose tolerance on recovered `n_components_`.
  Gibbs-mode tests likely marked slow given the sequential per-point loop.
- Reuse of copied math (Gaussian log-density, NIW updates) is verified
  independently in `dpgmm.py`'s own test file rather than assumed identical
  to `gmm.py`'s coverage.
