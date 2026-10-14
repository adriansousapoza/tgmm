# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] -- v0.4.0 in progress

### Added (Phase 0: sufficient-statistics M-step, algorithm switch, memory fixes)

- `tgmm.stats.SufficientStats`: a small accumulator (`nk`, `s1`, `s2`,
  `n_samples`) for the Gaussian mixture M-step, with `__add__` and
  `scale()` for combining/reweighting accumulators across chunks or
  mini-batches. Exported from `tgmm.__init__`.
- `GaussianMixture(..., algorithm='em')`: explicit algorithm selection,
  `'em'` (standard) or `'cem'` (Classification EM). `cem=True` still works
  but is deprecated in favour of `algorithm='cem'` (emits
  `DeprecationWarning`). `algorithm='sem'` is accepted for forward
  compatibility but not runnable yet.
- `GaussianMixture(..., batch_size=None)`: when set, the E-step and
  sufficient-statistics accumulation process `X` in row chunks, bounding
  peak memory to roughly `O(batch_size * n_components * n_features)`
  regardless of `n_samples`. Chunking is exact (not an approximation).
- `GaussianMixture(..., convergence='relative')`: new `'aitken'` option, an
  Aitken-accelerated stopping rule that can detect convergence earlier on
  slowly-crawling fits.

### Fixed

- `GaussianMixture` with `covariance_type='full'` (and the MAP/NIW variants
  of it) no longer materializes an `(n_samples, n_components, n_features,
  n_features)` tensor during the M-step -- replaced with an
  einsum-based second-moment computation. Peak memory for a
  `n_samples=20000, n_components=10, n_features=64` float32 fit dropped
  from ~3.3 GB to well under 2 GB.
- The E-step for `covariance_type='full'` and `'tied_full'` no longer
  triggers an implicit `(n_samples, n_components, n_features,
  n_features)` broadcast inside `torch.cholesky_solve` (PyTorch expands
  mismatched batch shapes to a contiguous copy before solving); samples
  are now passed as right-hand-side columns within each component's
  triangular solve instead of as an independent broadcast batch
  dimension, eliminating the same class of memory blow-up from the E-step
  as well as the M-step.

### Internal

- `GaussianMixture._m_step(X, resp)` is now a thin wrapper over
  `_compute_stats(X, resp, batch_size=self.batch_size)` and
  `_m_step_from_stats(stats)`. All MLE/MAP/NIW covariance update methods
  now read from a `SufficientStats` instance instead of re-deriving
  deviations from `X`/`resp`; see
  `docs/superpowers/specs/2026-10-14-phase0-sufficient-stats-design.md`
  for the derivation showing this is a zero-behaviour-change refactor
  (verified against a pre-refactor fixture at `atol=1e-10`, float64, for
  all six covariance types x {MLE, MAP, NIW}).
- `_fit_single_run` now dispatches the classification step on
  `self.algorithm` rather than `self.cem`.

## [0.3.0] and earlier

Not retroactively documented here -- see git history.
