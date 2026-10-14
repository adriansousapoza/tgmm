# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] -- v0.4.0 in progress

### Added (Phase 1: clustering metrics and model selection)

- `GaussianMixture.icl(X, mode='hard'|'soft')`: Integrated Completed
  Likelihood (Biernacki, Celeux & Govaert, 2000) -- `bic` plus an entropy
  penalty for ambiguous/overlapping-cluster assignments.
- `tgmm.model_selection.select_model(...)` / `ModelSelectionResult`:
  sweeps `GaussianMixture` over a grid of `(n_components, covariance_type)`
  and compares by BIC/AIC/ICL. A combination whose fit raises gets a
  `None`-scored row instead of aborting the sweep. Both exported from
  `tgmm` top-level.
- Three new `ClusteringMetrics` internal (no-ground-truth) indexes that
  ranked at the top of the Simpson, Campello & Stojanovski (2025)
  clustering-validity benchmark: `wb_index` (Zhao & Fränti, 2014),
  `wemmert_gancarski_index`, and `dbcv` (Moulavi et al., 2014).
- `silhouette_score`, `davies_bouldin_index`, `calinski_harabasz_score`,
  `dunn_index`, `wb_index`, `wemmert_gancarski_index` all gained: optional
  `n_components` (inferred from `labels` when omitted; labels no longer
  need to be contiguous or 0-indexed), `metric` (`'euclidean'`,
  `'manhattan'`, `'cosine'`, `'mahalanobis'` via `VI`, and -- for
  `silhouette_score`/`dunn_index` -- `'precomputed'`), and `noise_label`
  (drops those rows, rescales the result by `(n - n_noise) / n`).
  `silhouette_score`/`dunn_index` also gained `chunk_size`-bounded memory.
- `normalized_information_distance` (Vinh, Epps & Bailey, 2010).
- `v_measure_score(labels_true, labels_pred, beta=1.0)`: weighted
  homogeneity/completeness harmonic mean (`beta=1.0` recovers the previous
  unweighted formula exactly).

### Fixed

- `homogeneity_score`/`completeness_score` (and therefore `v_measure_score`,
  which is built from them) had their entropy denominators swapped --
  `homogeneity_score` divided by `H(K)` instead of `H(C)` and vice versa
  for `completeness_score`. Invisible on roughly-balanced same-cardinality
  labels (where `H(C) ~= H(K)`, which is what the only prior test
  exercised) but seriously wrong otherwise: on an asymmetric-cardinality
  check written while fixing this (3 true classes vs. 10 predicted
  clusters), `homogeneity_score` returned 0.64 against sklearn's 0.34, and
  `completeness_score` returned **-0.53** (outside its own [0, 1] range)
  against sklearn's 0.18. Rewritten using `torch.special.xlogy` (and
  `float64` internally) throughout, now within 1e-6 of sklearn across both
  the original symmetric case and the asymmetric one.
- `tgmm.hdbscan.prim_mst` (factored out of `HDBSCAN._build_linkage_tree`
  below) allocated its internal `min_distances` tensor at the default
  dtype regardless of the input distance matrix's dtype, raising
  `RuntimeError` on any float64 input. Latent in `HDBSCAN` before this
  phase (not previously exercised at float64); surfaced immediately by
  `dbcv`'s float64 usage.

### Internal

- `tgmm.hdbscan.prim_mst`: the Prim's-algorithm MST construction inside
  `HDBSCAN._build_linkage_tree` is now a standalone function (same
  algorithm, zero behaviour change there), so `ClusteringMetrics.dbcv` can
  reuse it for its own per-cluster MSTs instead of a second implementation.
- `GaussianMixture.bic`/`aic`'s active-components-only machinery (Gibbs
  mode with pruned slots) is shared with `icl` via a new
  `_active_e_step`/`_is_pruned_gibbs_fit`/`_resp_for_information_criterion`
  split, rather than duplicating the weights/means/covariances swap.

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
