# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] -- v0.4.0 in progress

### Added (Phase 4: `BaseMixture`, `StudentTMixture`, `VonMisesFisherMixture`)

- New `tgmm.mixtures` package with `BaseMixture`, the shared base class
  every mixture family now builds on: the EM/CEM/SEM/SAEM/online-EM loop,
  `n_init` best-run selection, `predict`/`predict_proba`/`score`/
  `score_samples`/`bic`/`aic`/`icl`, and a generic `sample()` shell, all
  behind a small abstract-method contract
  (`_component_log_prob`/`_compute_stats`/`_m_step_from_stats`/
  `_allocate_parameters`/`_n_parameters`/`_sample_component`).
  `GaussianMixture` is rebased onto `BaseMixture` with **zero behavior
  change**: it keeps its own `fit()` (Gibbs-sampling and supervised-fit
  dispatch), `sample()` (confidence regions, rejection sampling), and
  `save`/`load`/`save_state_dict`/`load_state_dict` exactly as they were,
  overriding `BaseMixture`'s versions where its existing behavior doesn't
  generalize. Verified: the full pre-existing test suite passes unchanged,
  and seeded fits are bit-identical pre/post-rebase across every algorithm
  (EM/CEM/SEM/SAEM/online/DAEM/Aitken/supervised/Gibbs) and covariance
  type tested.
- `tgmm.StudentTMixture`: a robust alternative to `GaussianMixture` (Peel
  & McLachlan, 2000). Each component is a multivariate Student-t with
  mean, scale matrix (`covariances_`), and degrees of freedom (`dof_`,
  `'per_component'`/`'shared'`/a fixed float, found each M-step by
  bisection+Newton on the digamma equation). `covariance_type` supports
  `'full'`/`'diag'`/`'spherical'`/`'tied_full'`; `scale_update='ecm'`
  (default) or `'ecme'`. `sample_weights(X)` returns each point's
  robustness weight at its predicted component, for flagging outliers.
  Verified against `scipy.stats.multivariate_t.logpdf` (~1e-6 agreement),
  parity with `GaussianMixture` at a large fixed `dof`, and real
  robustness gains over `GaussianMixture` under 5% uniform contamination.
- `tgmm.VonMisesFisherMixture`: a mixture of von Mises-Fisher
  distributions (Banerjee, Dhillon, Ghosh & Sra, 2005) for L2-normalized
  (directional) data, e.g. cosine-similarity embeddings. Each component
  has a unit-norm mean direction and a concentration (`kappa_`,
  `'per_component'` or `'shared'`, fit via the Banerjee closed-form
  approximation refined by two Newton steps, Sra 2012). `init_means=
  'kmeans'` runs spherical k-means (cosine similarity) with cosine-
  distance k-means++ seeding. `fit`/`predict`/`predict_proba`/
  `score_samples`/`score`/`partial_fit` all auto-normalize non-unit-norm
  input with a one-time warning. `sample()` uses Wood's (1994) rejection
  sampler. `_log_bessel_iv` (the log Bessel-I term in the normalizing
  constant) verified against `mpmath.besseli` at `d` in
  `{2, 10, 128, 768}`.
- `SufficientStats` gains an optional `aux: dict[str, Tensor]` field for
  family-specific accumulators that don't fit the `nk`/`s1`/`s2` shape
  (used by `StudentTMixture`'s u-weighted sums); fully backward
  compatible -- `GaussianMixture` never sets it, and `__add__`/`scale`
  are no-ops on it when absent.
- New user guide page "Other Mixture Families" and notebook
  `16_student_t_and_vmf.ipynb` (t-mixture vs Gaussian under contamination
  with the outlier-weight map; vMF vs `GaussianMixture(covariance_type=
  'diag')` on simulated 768-d unit-norm embeddings).

### Added (Phase 3: the 14 GPCM covariance models)

- `GaussianMixture(covariance_type=...)` now accepts all 14 Celeux &
  Govaert (1995) GPCM mclust codes (`EII`, `VII`, `EEI`, `VEI`, `EVI`,
  `VVI`, `EEE`, `VEE`, `EVE`, `EEV`, `VVE`, `VEV`, `EVV`, `VVV`,
  case-insensitive), alongside the 6 existing tgmm names (which are
  aliases for `EII`/`VII`/`EEI`/`VVI`/`EEE`/`VVV`). New
  `covariance_type_` attribute always holds the canonical mclust code.
  M-steps for the 8 new codes live in `tgmm.covariance.gpcm.mstep_gpcm`,
  derived from the Celeux & Govaert (1995) equations directly (tgmm is
  MIT; `mclust`/`mixture` are GPL and were not read or ported -- see the
  design doc). 5 of the 8 (`VEI`, `VEE`, `VEV`, `EVE`, `VVE`) are
  iterative, warm-started each EM iteration from the previous iteration's
  decomposition; `EVE`/`VVE` additionally solve a small
  orthogonal-manifold (Stiefel) optimization each inner iteration for the
  shared orientation, since no closed form exists once that orientation
  must jointly serve every component's own differently-shaped scatter.
- `volume_`/`shape_`/`orientation_`: the fitted decomposition
  `Sigma_k = volume_k * orientation_k @ diag(shape_k) @ orientation_k^T`
  for the 8 new codes (`None` for the 6 pre-existing ones).
- `gpcm_tol`/`gpcm_max_iter` constructor args controlling the 5 iterative
  codes' inner-loop stopping rule.
- `select_model`'s default `covariance_types` is now all 14 GPCM codes
  (previously 4 of the 6 pre-existing types) -- the mclust-style
  model-selection workflow the plan describes.
- MAP/NIW priors (`mean_prior`/`covariance_prior`) and Gibbs mode
  (`n_components=None`) both raise `NotImplementedError` for the 8 new
  codes (a conjugate-prior derivation per parameterization, and a
  per-component-factored Gibbs sampler for a covariance that couples
  components, are both out of scope); `weight_concentration_prior` works
  with all 14, since it doesn't depend on the covariance parameterization.

### Added (Phase 2: Stochastic EM, deterministic annealing, SAEM, mini-batch EM)

- `algorithm='sem'` (Stochastic EM, Celeux & Diebolt 1985): samples a hard
  assignment from the responsibilities each iteration instead of taking
  the argmax. Always runs the full `max_iter` sweeps (no pointwise
  convergence; `converged_` is unconditionally `True`, same convention as
  Gibbs mode). New `sem_burn_in`/`sem_estimate` ('best'/'mean', both
  restricted to post-burn-in sweeps) control the final estimate;
  `n_empty_components_` counts component re-seeds (mean -> worst-explained
  point, covariance -> dataset's own covariance for non-tied types).
- `algorithm='saem'` (Stochastic Approximation EM, Delyon, Lavielle &
  Moulines 1999): SEM blended into a running statistics accumulator with a
  decaying step size, so it *does* converge pointwise and `tol` applies.
- `temperature=(beta_0, n_anneal)` (Deterministic Annealing EM, Ueda &
  Nakano 1998): tempers the E-step's joint log-density by a `beta` rising
  geometrically from `beta_0` to `1.0`; convergence checks suspended while
  `beta < 1.0`. Works with `algorithm in ('em', 'cem', 'sem')`. Schedule
  actually used recorded in `annealing_schedule_`.
- `step_size` + existing `batch_size` (Phase 0): mini-batch/online EM for
  `algorithm='em'` (`fit()` shuffles into mini-batches per epoch, blending
  each batch's sufficient statistics into a running accumulator) and
  `partial_fit(X_batch)` for streaming continuation after an initial fit.
- `log_likelihood_history_`: per-iteration/sweep/epoch log-likelihood
  trace, now recorded for every EM-mode algorithm (not just SEM/online EM
  -- extended during this phase so the comparison notebook could plot all
  four algorithms' traces together).
- Gibbs mode (`n_components=None`) now rejects any `algorithm` but `'em'`
  at construction, with a clear error (previously silently ignored, like
  `cem`).

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
