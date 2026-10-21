# GaussianMixture: fixed values and bounds

Date: 2026-10-21. Branch `feature-constraints`, on top of 0.4.2.

## Goal

Let users (a) **fix** parameters they already know and (b) **bound** parameters (means in `[0, 1]`, variances `<= 0.5`, weights in
`[0.1, 0.5]`), while EM stays a valid algorithm (monotone log-likelihood). All new arguments default to "off"; with them off a fit is
bit-identical to 0.4.2.

## API (`GaussianMixture` constructor arguments, last in the signature)

| argument | meaning |
| --- | --- |
| `weight_bounds=(lo, hi)` | each a scalar or a `(K,)` tensor; mixing weights stay in `[lo_k, hi_k]` and sum to 1 |
| `fixed_weights=tensor(K)` | weights held at these values (positive, sum to 1) |
| `mean_bounds=(lo, hi)` | each a scalar, `(d,)` or `(K, d)` (use `-inf` / `inf` for one-sided) |
| `covariance_bounds=(lo, hi)` | bounds on the **variances**: the variances themselves for `diag`, `spherical`, `tied_diag`, `tied_spherical`; the **eigenvalues** of each covariance for `full`, `tied_full`. Scalars or `(K,)` (scalar for tied types); `lo >= 0` |
| `frozen_means=[k, ...]` | freeze only the mean of these components (they keep their initial value) |
| `frozen_covariances=[k, ...]` | freeze only the covariance of these components |

`frozen_components` (0.4.1) keeps its meaning: mean and covariance both frozen.

* Frozen values come from `init_means` / `init_covariances`, which must then be tensors (a method name would freeze an arbitrary
  initialisation); `fixed_weights` carries its own values.
* All initial parameters are **projected** into the feasible set (means clipped, variances/eigenvalues clipped, weights projected);
  a `UserWarning` is issued when a user-given initial tensor changed.
* Diagnostic `active_constraints_`: after a fit, a dict of boolean masks saying which bounds are binding.

## Mathematics

Notation: `n_k = sum_i r_ik`, `ybar_k = s1_k / n_k`, `W_k` the scatter about `ybar_k`, and for a mean `mu_k`
`T_k = W_k + n_k (ybar_k - mu_k)(ybar_k - mu_k)^T` (what `_scatter_about_means` returns).

The expected complete-data log-likelihood splits into three blocks,
`Q = sum_k n_k log pi_k - 1/2 sum_k [ n_k (mu_k - ybar_k)^T Sigma_k^-1 (mu_k - ybar_k) + n_k log|Sigma_k| + tr(Sigma_k^-1 W_k) ] + const`.
The constrained M-step maximises `Q` **block by block**, each block exactly given the others (an ECM step), in the order weights,
means given the previous covariances, covariances given the new means. Every block update does not decrease `Q`, so the
log-likelihood is monotone, exactly as for plain EM.

1. **Weights.** `max sum_k n_k log pi_k` s.t. `sum pi = 1`, `lo_k <= pi_k <= hi_k`. The Lagrangian gives
   `pi_k = clip(n_k / nu, lo_k, hi_k)`, with the multiplier `nu > 0` the root of the decreasing function
   `f(nu) = sum_k clip(n_k / nu, lo_k, hi_k) = 1` (bracketed by doubling, then bisection; exact up to float precision). Feasibility
   `sum lo <= 1 <= sum hi` is checked at construction. With a Dirichlet weight prior `n_k` is replaced by
   `max(n_k + alpha_k - 1, 0)` (the MAP numerator). `fixed_weights` skips all of this.
2. **Means.** `max -1/2 n_k (mu - ybar_k)^T Sigma_k^-1 (mu - ybar_k)` over the box `[lo, hi]`. For diagonal and spherical
   covariances (also tied) the problem separates per coordinate and `mu = clip(ybar_k, lo, hi)` is **exact**. For full covariances
   (`full`, `tied_full`, and the assembled GPCM matrices) it is a box-constrained quadratic program; we run coordinate ascent
   from the previous (feasible) mean: each coordinate update `mu_j = clip(ybar_j - sum_{l != j} P_jl (mu_l - ybar_l) / P_jj, lo_j, hi_j)`
   with `P = Sigma^-1` is the exact conditional maximiser, so a sweep never decreases `Q`; sweeps are repeated up to a small cap or
   until the update is below a tolerance. Repeated sweeps converge to the box-QP optimum (tested against L-BFGS-B); across EM
   iterations the remaining gap closes as well.
3. **Covariances.** With `T_k` the scatter about the (new, possibly clipped or frozen) mean, the unconstrained MLE is
   `T_k / n_k`. For a scalar variance `J(s) = n log s + t / s` is unimodal in `s`, so the constrained optimum is
   `clip(t / n, lo, hi)` (**exact**; entrywise for `diag`, per component for `spherical`, with `T` pooled over components for the
   tied types). For full matrices and spectral bounds `lo I <= Sigma <= hi I`, minimise `n log|Sigma| + tr(Sigma^-1 T)`: by von
   Neumann's trace inequality the minimiser shares the eigenvectors of `T`, and the problem reduces to the scalar one per
   eigenvalue, so the solution clips the eigenvalues of `T / n` to `[lo, hi]` (**exact**; Won, Lim, Kim & Rubin 2013, ill-conditioned
   covariance estimation). `reg_covar` is added first (as in the unconstrained M-step) and the bounds are applied afterwards, so
   `reg_covar` is the exact constrained solution of the regularised scatter and `lo` acts as a hard floor.
4. **Frozen means / covariances.** A frozen mean is restored *before* the covariance update, so the covariance MLE uses the scatter
   about the fixed mean (parallel-axis correction already in the code); a frozen covariance leaves the mean update as usual.

Exactness summary: weights exact; means exact for diagonal/spherical, ECM (coordinate ascent) for full; covariances exact.

## Decisions

* **bic/aic count free parameters.** `fixed_weights` removes the `K - 1` weight parameters, each frozen mean removes `d`, each frozen
  covariance removes its per-component covariance parameters. Bounds do not change the count. `frozen_components` keeps the
  0.4.1 behaviour (not subtracted).
* **Not supported (each raises with a message):** Gibbs mode (`n_components=None`); mean/covariance bounds together with a mean or
  covariance prior (NIW/MAP); `covariance_bounds` for the 8 GPCM-only codes (the covariance is parameterised by volume, shape and
  orientation); `frozen_covariances` for tied covariance types and GPCM-only codes (a shared covariance cannot be frozen for one
  component); weight constraints together with `noise=True` or semi-supervised `labels=` (those rescale the weights afterwards);
  every constraint together with per-feature `sample_weight` of shape `(n, d)`.
* **Works with:** `em`, `cem`, `sem`, `saem`, `batch_size`, online EM / `partial_fit` (the projections run inside the M-step on the
  accumulated statistics), row `sample_weight`, `missing='em'`, `select_model(..., **gmm_kwargs)`, `n_init`, `warm_start`, save/load.
* **Binding bounds mean a constrained maximum likelihood fit** (not a Bayesian one): do not read the result as unconstrained
  estimates; bounds that bind everywhere act like a prior in disguise.

## Tests

`tests/test_constraints.py`: defaults bit-identical; each block against an independent optimiser (water-filling vs bisection/SLSQP,
box QP vs L-BFGS-B, covariance clipping vs brute-force numeric minimisation); monotone likelihood for every constraint type and
covariance type; bounds hold after every iteration; fixed and frozen parameters stay bit-identical; recovery examples;
algorithms, chunking, persistence, `select_model`; every error combination; projection of initial values with a warning.
