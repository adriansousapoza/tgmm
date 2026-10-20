# v0.5 item 1.1: uniform noise component

Date: 2026-10-15. Plan: "tgmm v0.5 roadmap: more mixture families", section 1.1 (Banfield & Raftery 1993; the
`noise` option of mclust, described in Scrucca et al. 2016 -- implemented from the paper, no mclust code read).

## What

A mixture of `K` components plus one extra, parameter-free **noise component**: a uniform density over a region of
volume `V`. Points that none of the `K` components explains well are absorbed by the noise column instead of dragging
the component means and covariances (compare `StudentTMixture`, which downweights them continuously; the noise
component gives a hard "is this an outlier" label and works for any family with a bounded support).

## API (on `BaseMixture`, hence `GaussianMixture`, `StudentTMixture`, `VonMisesFisherMixture`)

| argument | default | meaning |
| --- | --- | --- |
| `noise` | `False` | switch the noise component on. Everything below is inert when `False`: fits are bit-identical to before. |
| `noise_volume` | `'bbox'` | `'bbox'`: volume of the axis-aligned bounding box of the fit data, `prod(max - min)` (each extent floored at `1e-12` so a constant feature cannot give `V = 0`). A positive float: a user-supplied hypervolume. Ignored by `VonMisesFisherMixture` (see below). |
| `noise_prior` | `0.05` | initial noise weight `pi_0`, in `(0, 1)`. |
| `noise_init` | `None` | per-sample initial noise indicator of length `n`. A `bool` tensor is a mask (`True` = noise). Any other dtype is read as cluster labels with `-1` = noise, so `HDBSCAN(...).labels_` can be passed as is. It overrides `noise_prior` (`pi_0` = fraction flagged, clamped to `[1e-3, 0.9]`) and the initial means/covariances are computed from the unflagged rows only. |

New fitted attributes: `noise_weight_` (0-dim tensor, `pi_0`), `noise_bounds_` (`(2, d)`, per-feature min/max of the fit
data; `None` for vMF). New method `noise_proba(X)`.

## Model

`p(x) = sum_k pi_k f_k(x) + pi_0 / V`, with `sum_k pi_k + pi_0 = 1`.

* **E-step.** A column `log pi_0 - log V` is appended to the `(n, K)` log-probabilities, so responsibilities have shape
  `(n, K + 1)`; the last column is the noise responsibility `tau_i0`. `log V` is a constant, so the noise log-density is the
  same for every `x` (outside the bounding box too, as in mclust: new points beyond the box are not given zero noise density).
* **M-step.** `pi_0 = sum_i tau_i0 / n`. The `K` components are updated from `resp[:, :K]` by the family's unchanged
  `_compute_stats` / `_m_step_from_stats`; since the statistics are normalised by the total `n`, `weights_ = n_k / n`
  automatically sums to `1 - pi_0`.
* **Family hook `_noise_log_density(X)`.** Default (`GaussianMixture` with all 14 covariance parametrisations, `StudentTMixture`): `-log V`.
  `VonMisesFisherMixture`: `-log A_d`, `A_d = 2 pi^{d/2} / Gamma(d/2)` the surface area of `S^{d-1}` (the uniform density on the sphere),
  independent of `noise_volume` and of the data. Families without bounded continuous support (the future count/discrete
  families) override the hook to raise `ValueError('noise component requires bounded continuous support')`.
* **`_n_parameters`** adds 1 (the `K + 1` mixture weights have `K` free parameters, i.e. `K - 1 + 1`); `bic`, `aic`, `icl` and `select_model`
  work unchanged (ICL's entropy term runs over the `K + 1` columns).

## Public behaviour

* `weights_` keeps shape `(K,)` and holds the **component** weights, which sum to `1 - noise_weight_` (not 1) when `noise=True`. This keeps
  every consumer of `weights_` (`sample`, `plot_gmm`, `save`/`load`, `plot_vmf`) working unchanged; the noise weight is read from `noise_weight_`.
* `predict(X)` -> `(n,)` long; `-1` where the noise column has the highest responsibility (the convention of `HDBSCAN` and
  `ClusteringMetrics(noise_label=-1)`).
* `predict_proba(X)` -> `(n, K + 1)`; `noise_proba(X)` -> `(n,)` (the last column). `score_samples`/`score` include the noise term.
* `sample(n)`: each draw is noise with probability `pi_0` and then uniform in the stored bounding box (uniform on the sphere for vMF);
  its component index is `-1`. `GaussianMixture.sample`'s constrained variants (`component=`, `std_radius=`, `confidence=` ...) raise
  `NotImplementedError` under `noise=True`.
* Algorithms: `'em'`, `'cem'` (the hard step includes the noise column), `'sem'` (the sampled assignment includes it), `'saem'` (the noise weight is blended
  with the same step size as the statistics), `temperature` (DAEM tempers the noise column like any other), `batch_size` (chunked equals unchunked).
* `n_init`, `warm_start` (the previous `noise_weight_` is reused, the bounding box is recomputed), `save`/`load`/`save_state_dict`/`load_state_dict`/`to_dict` round-trip
  `noise`, `noise_volume`, `noise_prior`, `noise_weight_`, `noise_bounds_` (the `noise_init` mask is a fit-time input and is not persisted).

## Unsupported combinations (raise, and are tested)

| combination | error |
| --- | --- |
| `noise=True` with Gibbs mode (`n_components=None`) | `NotImplementedError` |
| `noise=True` with `fit(X, labels=...)` (a point cannot be labelled "known noise" in v0.5; `-1` still means "unlabelled" elsewhere) | `NotImplementedError` |
| `noise=True` with online EM (`step_size=` set) or `partial_fit` | `NotImplementedError` |
| `noise=True` with `weight_concentration_prior` (a Dirichlet prior over `K` weights has no counterpart for `K + 1`) | `ValueError` |
| `noise_prior` outside `(0, 1)`, non-positive float `noise_volume`, a string other than `'bbox'`, wrong-length `noise_init`, all rows flagged | `ValueError` |

## Decisions

* The noise density is a constant over the whole space (see E-step), not an indicator of the box.
* `noise_init` accepts both masks and `HDBSCAN` label vectors; the dtype decides.
* No `noise_weight_` prior and no per-component noise: out of scope.

## Tests (`tests/test_v05_noise.py`)

Three blobs plus 10% uniform background: noise recall > 0.8, ARI on the non-noise points > 0.95, EM log-likelihood monotone; `noise=False` bit-identical
(same fit as a model constructed without the argument and the existing suite); parametrised over `em`/`cem`/`sem`/`saem`; `StudentTMixture`, `VonMisesFisherMixture`
and a GPCM code; `batch_size` equality in float64; save/load round trips; `predict`/`predict_proba`/`noise_proba`/`sample`; `noise_init` from a mask and from `HDBSCAN`; every unsupported combination above.
