# tied_spherical: restore the 0.3.0 summation order

Date: 2026-10-19. Reported by a user training a model in a loop (one shared spherical variance, `tied_spherical`): the
same code gave bit-different mixtures under 0.2.1 / 0.3.0 and 0.4.0, a 1e-16 difference at epoch 25 that grew into a
different model over the next 175 epochs.

## Cause

`tgmm <= 0.3.0` computed the pooled scatter of the tied spherical MLE in one reduction,
`einsum('nk,nkd->', resp, diff**2)`. The v0.4 sufficient-statistics refactor (Phase 0) computes the per-component
second moment `einsum('nk,nk->k', ...)` first and the tied update then sums over components. The two are equal in exact
arithmetic and differ in the last digit.

## Measurement

Fits from identical initial means on seeded data (K = 4, d = 3, n = 2000), `tol = 1e-30`, comparing the 0.3.0 release with
0.4.0 bit for bit after 1, 2, 5, 25 iterations:

* 0.4.0 is **not** bit-identical to 0.3.0 for any of the six classic covariance types, in float32 or float64 (differences
  of about 1e-15 in float64 and 1e-6 in float32 in the covariances, from the first iteration; `full` and `tied_full` also
  differ in the weights after one iteration). The cause is the same refactor: statistics centred on the responsibility
  weighted mean and reduced per component.
* Restoring only the single-reduction order for `tied_spherical` makes the fit **bit-identical to 0.3.0** in float32 and
  float64 for 1, 2, 5, 25 and 100 iterations.

## Change

`GaussianMixture._compute_stats` (unchunked path) also stores `aux['s2_total']`, the single-reduction pooled scatter,
when `covariance_type == 'tied_spherical'`; `_update_mle_tied_spherical` uses it when there is no mean prior (then
`means_ == ybar`, so no parallel-axis correction is needed). Chunked (`batch_size`) and online updates keep the
per-component statistics, which are what make them additive; the MAP/NIW paths are unchanged. Memory is unchanged (the
`(n, K, d)` difference tensor already exists in that branch).

## Scope and non-goals

Bit-reproducibility across releases is not a documented guarantee, and other covariance types still differ from 0.3.0 at
rounding level. Reproducing an old experiment exactly is best done by pinning the release; this change removes the
specific regression for the shared spherical variance. Extending it to the other types would mean restoring the pre-v0.4
arithmetic of each (E-step included for `full`/`tied_full`) and giving up the single sufficient-statistics code path.

## Tests

`tests/test_tied_spherical_order.py`: one M-step equals the single-reduction formula exactly (`torch.equal`, float32 and
float64); agreement with the per-component sum to rounding; chunked equals unchunked; mean-prior fallback; online path.
