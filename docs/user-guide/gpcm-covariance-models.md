# GPCM Covariance Models

`GaussianMixture` supports all 14 Celeux & Govaert (1995) Gaussian
Parsimonious Clustering Models (GPCM) -- the same family `mclust` implements
-- not just the 6 covariance types in the main
[Gaussian Mixture Model](gaussian-mixture.md#covariance-types) guide.

## The idea: a decomposed covariance

Each component's covariance is written as

$$ \Sigma_k = \lambda_k D_k A_k D_k^T $$

- $\lambda_k$ (**volume**): a scalar controlling the component's overall
  size.
- $A_k$ (**shape**): a diagonal matrix with determinant 1, decreasing
  entries -- how stretched the component is along its own axes.
- $D_k$ (**orientation**): an orthogonal matrix -- which way those axes
  point.

Each of the three can be **E**qual across components, **V**ary per
component, or (volume/shape only) fixed to the **I**dentity (axis-aligned,
no rotation). The 14 valid combinations are named by their 3-letter mclust
code (Volume-shape-orientation):

| Code | tgmm name | Meaning |
| --- | --- | --- |
| `EII` | `'tied_spherical'` | one shared variance, spherical |
| `VII` | `'spherical'` | per-component variance, spherical |
| `EEI` | `'tied_diag'` | one shared diagonal covariance |
| `VEI` | *(new)* | per-component volume, shared diagonal shape |
| `EVI` | *(new)* | shared volume, per-component diagonal shape |
| `VVI` | `'diag'` | per-component diagonal covariance |
| `EEE` | `'tied_full'` | one shared full covariance |
| `VEE` | *(new)* | per-component volume, shared shape & orientation |
| `EVE` | *(new)* | shared volume & orientation, per-component shape |
| `EEV` | *(new)* | shared volume & shape, per-component orientation |
| `VVE` | *(new)* | shared orientation, per-component volume & shape |
| `VEV` | *(new)* | shared shape, per-component volume & orientation |
| `EVV` | *(new)* | shared volume, per-component shape & orientation |
| `VVV` | `'full'` | fully unconstrained per-component covariance |

Pass either name to `covariance_type` (mclust codes are case-insensitive):

```python
from tgmm import GaussianMixture

gmm = GaussianMixture(n_components=3, n_features=4, covariance_type='EVE')
gmm.fit(X)

print(gmm.covariance_type_)   # always the canonical mclust code, 'EVE' here
print(gmm.volume_, gmm.shape_, gmm.orientation_)  # the decomposition
```

`covariances_` is always the assembled `(n_components, n_features,
n_features)` matrices regardless of which of the 14 codes you use --
`predict`/`sample`/`score`/`plot_gmm`/`save`/`load` all work unchanged.
`volume_`/`shape_`/`orientation_` expose the decomposition itself (`None`
for the 6 pre-existing types, which don't compute one).

## Why bother with 8 more models?

More constrained models (fewer free parameters) are less prone to
overfitting on small or high-dimensional data, at the cost of flexibility.
The 14-model family lets you search this trade-off systematically instead
of jumping straight from "fully tied" to "fully free" -- which is exactly
what `select_model` does by default:

```python
from tgmm import select_model

result = select_model(X, n_components=range(1, 8), criterion='bic')
print(result.best_.covariance_type_, result.best_.n_components)
```

`select_model`'s default `covariance_types` is all 14 GPCM codes (the
mclust workflow); pass an explicit subset to narrow the search.

## What's and isn't supported for the 8 new codes

- **MLE only.** `mean_prior`/`covariance_prior` (MAP/NIW estimation) raise
  `NotImplementedError` for the 8 new codes -- a conjugate-prior derivation
  specific to each parameterization is out of scope. `full`/`diag`/
  `spherical`/`tied_*` still support MAP/NIW as before.
  `weight_concentration_prior` (the Dirichlet weight prior) is independent
  of the covariance parameterization and works with all 14.
- **No Gibbs mode.** `GaussianMixture(n_components=None, covariance_type=...)`
  raises `NotImplementedError` for the 8 new codes, the same way it already
  does for `tied_*` -- a covariance that couples components (shared shape
  or orientation) doesn't factor per component the way Neal's Algorithm 3
  needs.
- **`init_covariances='empirical'`/`'global'`** project the usual empirical
  starting point onto the requested GPCM parameterization (the constrained
  MLE of that starting point); `'eye'`/`'random'` just start from plain
  identity/random full matrices, since there's nothing to project yet.

## The 5 iterative models

9 of the 14 codes (the 6 pre-existing ones, plus `EVI`, `EEV`, `EVV`) have
a closed-form M-step. `VEI`, `VEE`, `VEV`, `EVE`, and `VVE` alternate
between sub-steps (volume/shape/orientation, whichever vary) until the
model's own complete-data objective stops improving by more than
`gpcm_tol` (default `1e-6`), or `gpcm_max_iter` (default `20`) inner
iterations are reached:

```python
gmm = GaussianMixture(n_components=3, n_features=4, covariance_type='VEE',
                       gpcm_tol=1e-8, gpcm_max_iter=50)
```

Each EM iteration's inner alternation is *warm-started* from the previous
EM iteration's decomposition, so in practice it usually takes only 1-2
inner iterations to re-settle, not the full `gpcm_max_iter`.

`EVE` and `VVE` additionally need to choose a shared orientation `D` that
best serves every component's own (differently-shaped) scatter -- there's
no closed form for this, so it's solved as a small orthogonal-manifold
optimization (gradient-based, on the Stiefel manifold) each inner
iteration, accepted only if it actually improves the fit (a generalized
EM/GEM step, not a guaranteed-exact M-step) -- still monotone overall
across EM iterations, as for every other covariance type.

## Reference

Celeux, G. & Govaert, G. (1995). Gaussian parsimonious clustering models.
*Pattern Recognition*, 28(5), 781-793. `mclust`'s own naming and model
definitions (Scrucca, Fop, Murphy & Raftery, 2016, *The R Journal* 8(1))
are followed here for compatibility, but every M-step in tgmm is derived
independently from the papers' equations -- see
`docs/superpowers/specs/2026-10-14-phase3-gpcm-covariance-models-design.md`
for the full derivation (tgmm is MIT-licensed; `mclust` is GPL, and its
source was not read or ported).
