# Phase 3: the 14 GPCM covariance models — Design

Part of the [tgmm v0.4 roadmap](../../../tgmm%20v0.4%20roadmap%20coding-agent%20plan.md).
Depends on Phase 2's `algorithm`/`SufficientStats` plumbing (not Phase 1).
Licensing note (binding on this whole phase): tgmm is MIT, `mclust` and the R
`mixture` package are GPL. Every M-step below is re-derived from first
principles (the Gaussian-mixture complete-data objective) and cross-checked
against the *equations* in Celeux & Govaert (1995) and Browne & McNicholas
(2014) as summarized in session research notes — never against `mclust`'s
Fortran/R source, which was not read.

## The objective every M-step below minimizes

Write `W_k = sum_i r_ik (x_i - mu_k)(x_i - mu_k)^T` (`stats.s2` for `full`,
Phase 0's convention), `n_k = sum_i r_ik`, `n = sum_k n_k`, `Sigma_k =
lambda_k D_k A_k D_k^T` (`D_k` orthogonal, `A_k` diagonal with determinant 1
and decreasing entries, `lambda_k` scalar). The part of the expected
complete-data log-likelihood that depends on the covariance parameters,
negated so it's a *minimization* target, is

```
J = sum_k [ n_k log|Sigma_k| + tr(Sigma_k^{-1} W_k) ]
  = sum_k [ n_k d log(lambda_k) + (1/lambda_k) tr(A_k^{-1} D_k^T W_k D_k) ]
```

(`|Sigma_k| = lambda_k^d`, `|A_k| = 1`, `|D_k| = +-1`.) Every closed form and
alternation below is a partial minimization of this `J`, verified by hand
(see each model's equation box) before being checked numerically against a
plain `torch.optim`-based brute-force minimizer in `tests/test_phase3.py`
for small toy cases, and against mclust fixtures when available (see
[[v0_4_mclust_oracle_gap]]).

Two facts used repeatedly:

- **Diagonal AM-GM**: minimizing `tr(A^{-1} B)` over diagonal, unit-determinant,
  positive `A` (`B` a fixed positive diagonal matrix) gives `A = B / det(B)^{1/d}`,
  achieving `tr(A^{-1}B) = d * det(B)^{1/d}`.
- **Matrix AM-GM**: minimizing `tr(C^{-1} W)` over unit-determinant SPD `C`
  (`W` a fixed SPD matrix) gives `C = W / det(W)^{1/d}`, same achieved value
  `d * det(W)^{1/d}`.
- **Orientation (Celeux & Govaert's Theorem 1)**: for *fixed* diagonal `A`
  with decreasing entries, the orthogonal `D` minimizing `tr(A^{-1} D^T W D)`
  diagonalizes `W` with eigenvalues sorted **decreasing** (pairs the
  smallest `1/A_jj` with the largest eigenvalue) -- a rearrangement-
  inequality argument, independent of any `lambda`/`lambda_k` scaling
  outside the trace.

## Representation

`covariances_` stays `(K, d, d)` (assembled full matrices) for all 14
codes, so `_estimate_log_gaussian_full`/`predict`/`sample`/`plot_gmm`/
`save`/`load` need no changes for the 8 new codes -- confirmed by reading
those call sites: none inspect `covariance_type` beyond dispatch, they just
consume whatever `(K, d, d)` tensor is in `covariances_`.

- `covariance_type_` (new, always set after `__init__`): canonical mclust
  code (`'EII'`, `'VEI'`, ..., `'VVV'`), resolved from either naming
  scheme, case-insensitively. `covariance_type` keeps its **existing**
  string for the 6 original types (`'full'`, `'diag'`, etc. -- backward
  compatible, existing code reading `self.covariance_type` keeps working)
  and becomes the mclust code for the 8 new types (no tgmm name exists for
  those -- the plan's table only gives tgmm names to the original 6).
- `volume_` (`(K,)` or scalar for E-volume models), `shape_` (`(K, d)` or
  `(d,)` for E-shape models), `orientation_` (`(K, d, d)` or `(d, d)` for
  E-orientation models) -- the decomposition, read-only diagnostics.
- `_expected_covar_shape()` returns `(K, d, d)` for all 8 new codes.
- `mean_prior`/`covariance_prior` raise `NotImplementedError` for the 8 new
  codes (MAP needs a conjugate-prior derivation per parameterization, out
  of scope -- matches the plan). `weight_concentration_prior` is
  independent of the covariance parameterization and keeps working
  unchanged. Gibbs mode (`n_components=None`) raises the same
  `NotImplementedError` it already raises for `tied_*` (a shared
  **or regularized-shape/orientation** covariance doesn't factor per
  component under Algorithm 3 either).

## `tgmm/covariance/gpcm.py`

One function, `mstep_gpcm(code, W, nk, d, state=None, reg_covar=0.0,
gpcm_tol=1e-6, gpcm_max_iter=20) -> GPCMResult(volume, shape, orientation,
cov_full)`, where `W` is `(K, d, d)` (Phase 0's `stats.s2`, which for
`full`-shaped `s2` is *already* `sum_i r_ik (x_i - ybar_k)(x_i - ybar_k)^T`
per component -- exactly this function's `W_k`, reused rather than
recomputed). `state` is the previous call's `GPCMResult` (or `None` on the
first M-step of a run), used to warm-start the iterative codes.
`reg_covar * I` is added to the **assembled** `cov_full` only, never to the
decomposition (so `volume_`/`shape_` stay exact, per the plan).

Closed-form codes (`EVI`, `EEV`, `EVV`, plus the 6 pre-existing ones handled
by tgmm's existing per-type code, not this module) need no `state` and
converge their "iteration" in one call. Iterative codes (`VEI`, `VEE`,
`VEV`, `EVE`, `VVE`) alternate until `J` changes by less than `gpcm_tol`
(relative) or `gpcm_max_iter` is reached, which the plan calls a GEM/ECM
step (an increase in the overall EM log-likelihood is only guaranteed
non-decreasing *across whole M-steps*, not necessarily monotone within
the few-iteration inner loop in isolation -- warm-starting from the
previous EM iteration's `state` is what keeps this to 1-2 inner iterations
in practice and keeps the outer EM log-likelihood monotone, verified in
`tests/test_phase3.py`).

### Per-code derivation summary (verified against the plan's equations independently, via the `J` above, before implementing)

| Code | Update | Verified via |
| --- | --- | --- |
| VEI | `lambda_k = tr(A^{-1} diag(W_k)) / (n_k d)`; `B = sum_k diag(W_k)/lambda_k`, `A = B/det(B)^{1/d}` | diagonal AM-GM, alternating |
| EVI | `B_k = diag(W_k)`, `A_k = B_k/det(B_k)^{1/d}` (closed form, no alternation -- doesn't depend on `lambda`); `lambda = sum_k det(B_k)^{1/d} / n` | diagonal AM-GM; `tr(A_k^{-1}B_k) = d*det(B_k)^{1/d}` substituted into the shared-`lambda` optimum |
| EEV | `eigh(W_k)` descending -> `D_k`, `Omega_k`; `Omega = sum_k Omega_k`; `A = Omega/det(Omega)^{1/d}`; `lambda = det(Omega)^{1/d}/n` | orientation theorem (D_k fixed by data, independent of lambda/A) reduces this to EVI's shared-lambda case on the already-diagonalized `Omega_k`'s |
| VEV | `D_k` as in EEV (same independence-of-lambda/A argument); alternate `lambda_k = tr(A^{-1}Omega_k)/(n_k d)` given `A`, `A = Omega'/det(Omega')^{1/d}` with `Omega' = sum_k Omega_k/lambda_k` given `lambda_k` | diagonal AM-GM on the per-k alternation, same structure as VEI but with `Omega_k` (already diagonal from `D_k`) in place of `diag(W_k)` |
| EVV | `C_k = W_k/det(W_k)^{1/d}` (closed form); `lambda = sum_k det(W_k)^{1/d}/n`; `D_k, A_k` from `eigh(C_k)` for the exposed attributes only | matrix AM-GM, same substitution trick as EVI |
| VEE | `C = DAD^T` (shared, `det(C)=1`); alternate `lambda_k = tr(C^{-1}W_k)/(n_k d)` given `C`, `C = B/det(B)^{1/d}` with `B = sum_k W_k/lambda_k` given `lambda_k` | matrix AM-GM, same structure as VEI with full matrices; `D, A` from one `eigh(C)` at the end |
| EVE | given shared `D`: `B_k = diag(D^T W_k D)`, `A_k = B_k/det(B_k)^{1/d}` (only the diagonal of the rotated scatter matters, since `A_k` is diagonal); `lambda = sum_k tr(A_k^{-1}B_k)/(nd)`; given `A_k`: `D* = argmin_{D orthogonal} sum_k tr(A_k^{-1} D^T W_k D) / lambda` -- Stiefel optimization (see below) | diagonal AM-GM for `A_k`/`lambda`; the `D` sub-problem has no closed form when `D` must simultaneously (approximately) diagonalize every `W_k` |
| VVE | same as EVE but `lambda_k = tr(A_k^{-1}B_k)/(n_k d)` (per-component) and the `D` objective is `sum_k tr(A_k^{-1}D^TW_kD)/lambda_k` | same as EVE, per-k volume |

### The `D` sub-problem for EVE/VVE: Stiefel-manifold optimization

No closed form exists once `D` must jointly serve every component's
(different) `A_k`-weighted scatter. Implemented exactly as the plan
specifies: `torch.nn.utils.parametrizations.orthogonal` on a `(d, d)`
parameter, initialized at the previous `state`'s `D` via the
parametrization's `right_inverse` (`module.D = D_init` under the
parametrization re-maps a given orthogonal matrix to the correct
underlying unconstrained representation), optimized with up to 20
`torch.optim.LBFGS` (`line_search_fn='strong_wolfe'`) steps minimizing
`sum_k tr(A_k^{-1} D^T W_k D) / lambda_k[or lambda]`, accepted only if the
objective strictly decreased (a GEM step, not guaranteed descent every
call) -- otherwise `D` from `state` is kept unchanged for this M-step.
Kept as the reference implementation per the plan; the Flury-Gautschi/
Browne-McNicholas MM alternative is explicitly out of scope unless
profiling later shows this is a bottleneck.

### `gpcm_tol`/`gpcm_max_iter` are constructor arguments

Exposed as `GaussianMixture(..., gpcm_tol=1e-6, gpcm_max_iter=20)` (not
hardcoded module constants) since they're genuinely user-tunable knobs on
fit quality/speed for the 5 iterative codes -- ignored by the other 9.

### Warm-starting across EM iterations

`self._gpcm_state_` (private, reset to `None` at the start of each
`_fit_single_run`/`_allocate_parameters`) holds the previous M-step's
`GPCMResult`, passed as `mstep_gpcm`'s `state` argument on the next M-step.
This is what the plan means by "warm-start from the previous EM iteration's
decomposition so one or two inner iterations usually suffice" -- without
it, every M-step would re-solve the inner alternation/Stiefel optimization
from a generic starting point (identity `D`, equal `A`), wasting the whole
point of warm-starting.

## Naming and aliasing

`covariance_type` accepts any of the 14 mclust codes or the 6 existing
tgmm names, case-insensitively, resolved once in `__init__`/`load`/
`load_state_dict` into `self.covariance_type_` (canonical mclust code,
always set) and `self.covariance_type` (tgmm name for the original 6,
mclust code otherwise -- see Representation above). A small
`_CODE_ALIASES` dict in `gmm.py` maps both directions; `covariance_type`
validation extends to accept the 8 new codes alongside the existing 6
names, case-insensitively for all 14.

## `_n_parameters()`

Table-driven from the plan's own table (covariance-parameter-count column),
keyed by `self.covariance_type_` (the canonical code, always available
regardless of which name the user passed): `EII: 1`, `VII: K`, `EEI: d`,
`VEI: d + K - 1`, `EVI: 1 + K*(d-1)`, `VVI: K*d`, `EEE: d*(d+1)/2`,
`VEE: K - 1 + d*(d+1)/2`, `EVE: 1 + K*(d-1) + d*(d-1)/2`,
`EEV: d + K*d*(d-1)/2`, `VVE: K*d + d*(d-1)/2`,
`VEV: K + d - 1 + K*d*(d-1)/2`, `EVV: 1 + K*(d-1) + K*d*(d-1)/2`,
`VVV: K*d*(d+1)/2`. Mean/weight parameter counts unchanged (`K*d`,
`K - 1`).

## Integration

- `init_covariances='empirical'`/`'global'` for the 8 new codes: compute
  the full empirical covariance per component as today (or the global
  covariance for `'global'`), then project it through one `mstep_gpcm`
  call seeded with hard k-means labels (`nk`/`W` built from one-hot
  labels) -- i.e. the GPCM-constrained MLE of that empirical starting
  point, not the raw unconstrained empirical covariance itself (which
  wouldn't satisfy the code's constraints).
- `select_model`'s default `covariance_types` becomes the 14 mclust codes
  (`EII, VII, EEI, VEI, EVI, VVI, EEE, VEE, EVE, EEV, VVE, VEV, EVV, VVV`)
  -- this is the one default-value change the plan calls for after this
  phase; `select_model`'s own docstring already flagged this as pending.

## Test oracle: still no R here

Per [[v0_4_mclust_oracle_gap]], `tests/fixtures/make_gpcm_fixtures.R` is
committed as reproducibility documentation (unrun): three seeded datasets
(`d in {2, 4, 6}`, `K=3`), a fixed k-means `z`, and `mclust::mstep()`/
`estep()`/`nVarParams()` output for all 14 model names, exactly as the plan
specifies. The Python-side fixture-comparison test is
`skipif(not fixture_exists)`. In its place, `tests/test_phase3.py` relies
on:

- **Monotone log-likelihood** for all 14 codes on 3 seeded datasets (the
  plan's own acceptance criterion, and the strongest available check that
  an M-step is a genuine ascent step even without an external oracle).
- **Nested-model consistency** (also the plan's own criterion): starting
  every model from the *same* `z`, the one-shot `mstep_gpcm` fit's
  log-likelihood for a more-constrained model is `<=` every model it's
  nested in (e.g. `EEE <= VEE, EVE, EEV <= ... <= VVV`).
- **A from-scratch brute-force check** for each closed-form/alternating
  equation on small random `W_k`/`nk`: minimize `J` directly with
  `torch.optim.Adam` over an unconstrained-but-penalized parameterization
  (or grid search for 2x2 cases) and compare to `mstep_gpcm`'s answer --
  this is what stands in for "did I derive the equation right" without
  mclust, independent of the monotonicity/nesting checks above (which
  would pass even for a *consistent* but wrong implementation of two
  related models).
- `_n_parameters` cross-checked against the table directly (no oracle
  needed, it's arithmetic) for all 14 codes across a few `(K, d)` pairs.

## Non-goals for this phase

- MAP priors for the 8 new codes (explicit `NotImplementedError`, per the
  plan).
- Gibbs mode for the 8 new codes (same error as `tied_*`).
- Replacing the autograd Stiefel optimizer with Flury-Gautschi/
  Browne-McNicholas MM (explicitly deferred unless profiling demands it).
- The order-constrained EVE/VVE estimator (Punzo, Browne & McNicholas
  2014) -- out of scope per the plan.
