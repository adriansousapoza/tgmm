# Other Mixture Families

`GaussianMixture` assumes every component is normally distributed. That's
the right assumption for a lot of data, but not all of it: outliers can
drag a Gaussian component's covariance around, and directional data (unit
vectors -- normalized embeddings, in practice) isn't naturally described
by a density defined over all of $\mathbb{R}^d$ in the first place. tgmm
has two more mixture families for exactly those two cases,
`StudentTMixture` and `VonMisesFisherMixture`, built on the same shared
`BaseMixture` machinery as `GaussianMixture` -- the same `fit`/`predict`/
`score`/`bic`/`aic`/`icl` API, the same `algorithm`/`temperature`/
`step_size` options for the EM loop itself.

Neither supports Gibbs-sampling mode (`n_components=None`) or supervised
fitting (`fit(X, labels=...)`) -- both stay `GaussianMixture`-only for now.

## StudentTMixture: robust to outliers

A multivariate Student-t distribution looks like a Gaussian near its
center but has heavier tails, controlled by its degrees of freedom $\nu$:
small $\nu$ means heavy tails (very forgiving of far-away points), and as
$\nu \to \infty$ it converges to a Gaussian exactly. Fitting a mixture of
these (Peel & McLachlan, 2000) gives each component an extra per-sample
weight $u_{ik}$ in the M-step that shrinks toward zero for points far from
that component's center -- those points get downweighted instead of
pulling the covariance estimate toward them the way they would for
`GaussianMixture`.

```python
from tgmm import StudentTMixture

stm = StudentTMixture(n_components=3, covariance_type='full', dof='per_component')
stm.fit(X)

stm.means_        # (n_components, n_features)
stm.covariances_  # the scale matrix, not technically a covariance for finite dof
stm.dof_          # (n_components,) -- fitted nu_k
```

- `covariance_type`: one of `'full'`, `'diag'`, `'spherical'`,
  `'tied_full'` (the GPCM parameterizations and Gibbs-only axis-aligned
  types aren't supported here).
- `dof`: `'per_component'` (default, one $\nu_k$ per component, found by
  root-finding each M-step), `'shared'` (one $\nu$ for every component),
  or a fixed float (e.g. `dof=1e4` recovers an approximately Gaussian
  fit -- a useful sanity check against `GaussianMixture` on clean data).
- `scale_update`: `'ecm'` (default) or `'ecme'` -- two valid M-step
  normalizations for the scale matrix; `'ecme'` tends to converge faster
  but is a different fixed point in general.

### Flagging outliers with `sample_weights`

After fitting, `sample_weights(X)` returns each point's $u_{ik}$ at its
*predicted* component -- small values flag points the model effectively
treated as outliers:

```python
weights = stm.sample_weights(X)
likely_outliers = X[weights < 0.3]
```

## VonMisesFisherMixture: directional data

For L2-normalized data -- points on the unit hypersphere, as you get from
normalizing text or image embeddings before clustering by cosine
similarity -- a Gaussian (or Student-t) mixture is the wrong model: it's
defined over all of $\mathbb{R}^d$ and has no notion of "direction only."
The von Mises-Fisher (vMF) distribution is the natural analogue restricted
to the sphere: component $k$ has a mean *direction* $\mu_k$ (itself
unit-norm) and a concentration $\kappa_k$ controlling how tightly points
cluster around it -- higher $\kappa_k$ is a tighter cluster, and
$\kappa_k \to 0$ approaches a uniform distribution over the whole sphere.

```python
from tgmm import VonMisesFisherMixture

vmf = VonMisesFisherMixture(n_components=5, concentration='per_component')
vmf.fit(X_normalized)  # X_normalized rows must have unit L2 norm

vmf.means_   # (n_components, n_features), each row unit-norm
vmf.kappa_   # (n_components,) -- fitted concentration
```

If `X` isn't already unit-norm, `fit`/`predict`/`score` all normalize it
automatically and warn once per call -- but fitting on already-normalized
data is still the right thing to do, since that's what the ambient
dimensionality (`n_features`) and the density itself assume.

- `concentration`: `'per_component'` (default) or `'shared'` (one
  $\kappa$ for every component).
- `init_means='kmeans'` (the default) runs *spherical* k-means -- k-means
  with cosine similarity instead of Euclidean distance, seeded with a
  cosine-distance k-means++ scheme to avoid collapsing two well-separated
  clusters into one (plain random seeding occasionally does exactly that;
  see the [design
  doc](https://github.com/adriansousapoza/tgmm/blob/main/docs/superpowers/specs/2026-10-14-phase4-base-mixture-studentt-vmf-design.md)
  for the failure mode this fixes).

### Comparing against `GaussianMixture` on normalized data

Nothing stops you from fitting `GaussianMixture(covariance_type='diag')`
on the same L2-normalized data -- it's a reasonable baseline, just not one
that respects the data living on a sphere. Compare the two with `icl` (or
a cosine-based clustering metric) to see whether the vMF assumption
actually buys you anything on a given dataset; see notebook 16 for a
worked example on high-dimensional embeddings.

## Model selection and shared machinery

`bic`/`aic`/`icl` work the same way on both new classes as on
`GaussianMixture` -- lower is better, and they're directly comparable
*within* a family (not across families with a structurally different
density). `sample(n)` draws from the fitted mixture for both: a Gaussian
scaled by $1/\sqrt{\mathrm{Gamma}(\nu_k/2, \nu_k/2)}$ for
`StudentTMixture`, and Wood's (1994) rejection sampler for
`VonMisesFisherMixture`. `algorithm='cem'/'sem'/'saem'`, `temperature`
(deterministic annealing), `step_size` (mini-batch/online EM, plus
`partial_fit`), and `n_init` all work exactly as documented for
`GaussianMixture`, since they're implemented once, in `BaseMixture`, and
shared by every family.
