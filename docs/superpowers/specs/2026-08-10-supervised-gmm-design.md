# Supervised GMM fitting via labels — Design

## Background

`GaussianMixture.fit` (`tgmm/gmm.py`) currently only supports unsupervised
fitting: standard EM (soft responsibilities from `_e_step`) and, when
`cem=True`, Classification EM (Celeux & Govaert, 1992), which adds a `_c_step`
that hardens responsibilities to the arg-max component before each `_m_step`
(`gmm.py:983-1020`, `_fit_single_run` at `gmm.py:895-981`).

The ask: make it possible to fit a *supervised* GMM — same CEM idea, but the
hard assignment in the C-step comes from ground-truth class labels instead of
`argmax(resp)`. This is a real, named technique, not a novel one:

- It is the fully-labeled limiting case of classification-likelihood
  maximization. When every `z_i` is known instead of estimated, maximizing
  the classification log-likelihood reduces to independent per-class MLE —
  there is no iteration left to do, since the assignment never depends on the
  current parameter estimates (McLachlan & Peel, *Finite Mixture Models*,
  Wiley 2000, §2.5, "Classification Likelihood").
- Equivalently, it's Gaussian discriminant analysis: one Gaussian fit per
  class using only that class's labeled points. With `covariance_type=
  'tied_full'` this is exactly LDA; with independent per-class covariances
  it's QDA. Fraley & Raftery's **MclustDA** (*Model-Based Clustering,
  Discriminant Analysis, and Density Estimation*, JASA 97(458), 2002) is the
  standard reference for precisely this (generalized there to multiple
  components per class — out of scope here, since the ask is one Gaussian
  per label).
- Hastie & Tibshirani's Mixture Discriminant Analysis (JRSS-B 58(1), 1996)
  is the same idea when >1 component/class is wanted.
- It composes with semi-supervised EM (Nigam, McCallum, Thrun & Mitchell,
  *Text Classification from Labeled and Unlabeled Documents using EM*,
  Machine Learning 39, 2000): labeled points get a fixed one-hot
  responsibility, unlabeled points still go through the ordinary E-step. Only
  the fully-labeled case is being built now, but the mechanism (fixed
  one-hot responsibility fed into the existing `_m_step`) is the same one
  that would extend to partial labels later.

## Goals

- `GaussianMixture.fit(X, labels=None, ...)`: when `labels` is provided, fit
  one Gaussian component per distinct label directly from the labeled data,
  bypassing the EM/CEM loop.
- Reuse the existing `_m_step` untouched (it already supports MLE, MAP, and
  NIW-conjugate updates for every `covariance_type`) — supervised fitting is
  parametrized the same way as unsupervised fitting, including priors.
- `predict`, `predict_proba`, `score`, `score_samples`, `sample`, `bic`,
  `aic`, `save`/`load` all keep working unmodified, since they only read
  `weights_`/`means_`/`covariances_`.
- New `notebooks/supervised_gmm.ipynb` comparing plain EM, CEM, and
  supervised fitting on the same 3-component synthetic dataset (a 3-Gaussian
  subset of `notebooks/gmm.ipynb`'s data), both visually and via
  `ClusteringMetrics`.
- Document the new parameter: docstring (flows into
  `docs/api/gaussian-mixture.md` via mkdocstrings) plus a short section in
  `docs/user-guide/gaussian-mixture.md`.

## Non-goals (v1)

- Semi-supervised / partial labels (some points labeled, some not). The
  mechanism generalizes cleanly (see Background), but the ask is fully
  supervised only — not building the partial-label path now.
- Multiple mixture components per class (MDA/MclustDA-style). One label ==
  one component, matching the request; `n_components` must equal the number
  of distinct labels.
- Any change to unsupervised EM/CEM behavior. `labels=None` (the default)
  must be byte-for-byte the existing code path.

## API design

```python
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X, labels=y)   # y: 1D tensor/array-like, length n_samples
```

- `labels=None` (default): existing unsupervised behavior, untouched.
- `labels` given: supervised path, described below.

### Label validation (`_validate_labels`, new private method)

- Accept any 1D tensor or array-like of length `n_samples`; coerce to a
  tensor on `self.device`.
- Map label values to contiguous component indices via
  `classes_ = torch.unique(labels, sorted=True)`, storing `self.classes_ =
  classes_` and returning `torch.searchsorted(classes_, labels)` as the
  0-indexed component assignment. This lets `classes_[k]` recover the
  original label for component `k`, so labels don't need to already be
  `0..n_components-1` — e.g. `{2, 5, 9}` works.
- Require `classes_.numel() == self.n_components`; otherwise raise
  `ValueError` naming both the found and expected count (one Gaussian
  component per distinct label is the only supported mapping — see
  Non-goals).

### Supervised fit path (in `fit`, branching on `labels is not None`)

1. Run existing input validation/dtype-setup (`fit`'s steps 1-3) unchanged.
2. Validate `labels` as above.
3. Call `self._allocate_parameters(X)` once, purely so `weights_`/`means_`/
   `covariances_` exist with correct shape/device/dtype (their *values* are
   immediately overwritten in the next step, so the configured
   `init_means`/`init_covariances` strategy runs but is discarded — no new
   allocation code needed, and this keeps `initial_covariances_` etc.
   consistent with the unsupervised path for anyone inspecting them).
4. Build the one-hot responsibility matrix from the mapped labels — this
   *is* "the C-step, but from true labels instead of `argmax(resp)`" — and
   call `self._m_step(X, resp)` **once**.
5. Set `self.n_iter_ = 1`, `self.converged_ = True`,
   `self.best_random_state_ = None`, `self.fitted_ = True`.
6. Compute `self.lower_bound_` via one `self._e_step(X)` call on the fitted
   parameters (for parity with the unsupervised path — `bic`/`aic`/`score`
   all remain meaningful).
7. Keep the existing near-zero-weight warning (a class with very few/no
   points still produces a degenerate component — same warning unsupervised
   fitting already gives for empty clusters).
8. Return `self` early, skipping the `n_init` loop entirely: the result is
   deterministic (a fixed point independent of initialization or
   `random_state`), so looping would just recompute the same answer
   `n_init` times. Note this behavior in the docstring.

`_c_step` is not modified or reused directly — supervised fitting computes
the equivalent one-hot matrix from labels rather than from `argmax(resp)`,
then goes straight to `_m_step`, since (per Background) the CEM loop with a
label-fixed C-step is already at its fixed point after one M-step; looping
E-step → label-C-step → M-step would recompute byte-identical parameters
every round.

## Tests (extend `tests/test_gmm.py`)

- Supervised fit on synthetic 3-class well-separated data recovers each
  component's mean/covariance matching the closed-form per-class sample
  mean/covariance (direct `torch` computation), for `covariance_type in
  {'full', 'diag', 'spherical', 'tied_full'}`.
- `classes_` round-trips non-contiguous label values (e.g. `{2, 5, 9}`):
  fit with those labels, confirm `predict` on the training data, mapped
  through `classes_`, matches the input labels for well-separated data.
- `n_components` mismatched against the number of distinct labels raises
  `ValueError`.
- `converged_ is True` and `n_iter_ == 1` after a supervised fit.
- `predict_proba` rows still sum to 1 after a supervised fit.
- A configured MAP/NIW prior (`mean_prior`, `covariance_prior`, etc.) still
  measurably shrinks the supervised estimate vs. the unregularized MLE on a
  small-sample class (regression coverage that `_m_step`'s prior handling is
  actually exercised by this path).

## Notebook: `notebooks/supervised_gmm.ipynb`

- Reuse 3 of the 4 `(center, cov)` pairs from `notebooks/gmm.ipynb`'s
  `generate_gmm_data` call (same synthetic-data generator, same style as
  `cem.ipynb`), producing a 3-class, moderately overlapping dataset.
- Fit three models on the same data: plain EM (`cem=False`), CEM
  (`cem=True`), and supervised (`labels=y_true`).
- Visual comparison: `plot_gmm` side by side (matched-label ellipses, same
  presentation as `cem.ipynb`), plus true-label reference panel.
- Quantitative comparison via `ClusteringMetrics`: `adjusted_rand_score`,
  `purity_score`, `classification_report`, mean log-likelihood/`bic`, plus
  iteration count, for each of the three fits.
- Markdown discussion cells: explain why supervised fitting converges in one
  step (fixed-point argument from Background), and cite Celeux & Govaert
  1992, McLachlan & Peel 2000, and Fraley & Raftery 2002.

## Docs

- Extend the `fit` docstring in `gmm.py` with the `labels` parameter,
  including the one-step/no-`n_init`-looping behavior — flows automatically
  into `docs/api/gaussian-mixture.md` via mkdocstrings.
- Add a "Supervised fitting" section to `docs/user-guide/gaussian-mixture.md`
  with a short example and a pointer to `notebooks/supervised_gmm.ipynb`.
