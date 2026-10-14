# Semi-Supervised and Positive-Unlabeled Fitting

Every tgmm mixture family (`GaussianMixture`, `StudentTMixture`,
`VonMisesFisherMixture`) can be fit with partial label information, from
"every row has a known class" down to "I only know some rows are
positive, and nothing about the rest." This page covers the terms, the
API, and which option to reach for.

## Semi-supervised mixture modelling

**Semi-supervised mixture modelling** means fitting with a *mix* of
labeled and unlabeled rows -- the usual setting for clustering when you
have a handful of confirmed examples per class but labeling the rest
would be too expensive. Pass `-1` for the unknown rows:

```python
import torch
from tgmm import GaussianMixture

labels = torch.full((X.shape[0],), -1)
labels[known_idx] = known_labels  # -1 everywhere else

gmm = GaussianMixture(n_components=3)
gmm.fit(X, labels=labels)
gmm.predict_class(X)  # class labels, not raw component indices
```

With **no** `-1` present, this is the fully-labeled case tgmm has always
supported: one (or more, see below) components fit directly from each
class's own data, no iteration needed. With **any** `-1` present, it runs
semi-supervised EM: each labeled row's responsibility is clamped to its
own class's components every E-step (so it can never "vote" for the wrong
class), while unlabeled rows keep the full, ordinary posterior. A few
labels per class are often enough to resolve ambiguity plain unsupervised
EM can't -- which cluster is which class, and where the boundary falls on
overlapping data.

Two knobs refine this:

- `unlabeled_weight` (default `1.0`): down-weights unlabeled rows'
  contribution to the fit. `0.0` ignores them entirely, recovering a fit
  on the labeled rows alone.
- `label_confidence` (default `1.0`): for labels you don't fully trust,
  `c < 1` keeps `c` of a labeled row's responsibility on its class and
  lets the rest follow the unclamped posterior -- a soft correction for
  occasional mislabeling.

**Scope note:** semi-supervised fitting (partial labels) currently
supports `algorithm='em'`/`'cem'` only; `'sem'`/`'saem'`/online EM
(`step_size` set) raise `NotImplementedError`. Gibbs-sampling mode
(`n_components=None`) and tied covariance types combined with
`components_per_class > 1` (below) aren't supported either.

### More than one component per class

Real classes aren't always unimodal. `components_per_class` (an `int`, or
a `dict` keyed by label value) gives a class more than one component --
the same idea as mclust's MclustDA:

```python
gmm = GaussianMixture(n_components=3, components_per_class={0: 2, 1: 1})
gmm.fit(X, labels=labels)  # class 0 gets 2 components, class 1 gets 1
```

## Positive-unlabeled (PU) learning

**Positive-unlabeled learning** is the more extreme case: you know some
rows are positive (e.g. confirmed cancer cells), and *everything else is
unlabeled* -- it could be positive or negative, you just don't know.
`PositiveUnlabeledMixture` models the unlabeled density as
$\pi f_P(x) + (1-\pi) f_N(x)$ and estimates both densities and $\pi$
jointly:

```python
from tgmm import PositiveUnlabeledMixture

pu = PositiveUnlabeledMixture(n_components_positive=1, n_components_negative=5)
pu.fit(X, positive_mask)  # bool mask, True = confirmed positive

pu.class_prior_                  # estimated fraction of *unlabeled* rows that are positive
pu.predict_proba_positive(X)     # P(positive) for arbitrary X
pu.predict_positive(X, threshold=0.5)
```

This assumes **SCAR** (Selected Completely At Random; Elkan & Noto,
2008): the labeled positives are a uniform random sample of *all* true
positives, not a biased subset (e.g. "the easiest-to-confirm cases"). If
your labeled positives were selected some other way, `class_prior_` and
the fitted densities can be biased -- SCAR is a real assumption, not a
formality.

This is also sometimes called **mixture proportion estimation**: the
whole point is estimating $\pi$, the mixing proportion of the
(fully-unobserved) positive class within the unlabeled pool. A related,
distinct problem is **semi-supervised novelty detection** (Blanchard, Lee
& Scott, 2010): given a labeled "normal" sample and an unlabeled mix of
normal and novel points, decide *which* unlabeled points are novel --
closer to `anchor='fixed'` below (treating the labeled class as a known,
unmoving reference density) than to PU's `'prior'` default.

### Picking `anchor`

The mixture proportion $\pi$ is only identifiable if the negative block
can't simply absorb the positive one. `anchor` controls how tightly the
positive block stays tied to the labeled positives:

| `anchor` | Positive block behavior | Corresponds to |
| --- | --- | --- |
| `'fixed'` | Fit once on labeled positives, then frozen completely; only $\pi$ and the negative block keep moving. | Classical mixture-proportion estimation / semi-supervised novelty detection (the "normal" class is a fixed reference). |
| `'prior'` (default) | Fit once on labeled positives, then used as an NIW prior so unlabeled positives can still refine it without drifting toward the negatives. **Gaussian-only.** | A softer version of the same -- lets genuinely-positive-but-different-looking unlabeled points still pull the estimate, within reason. |
| `'free'` | Initialized from labeled positives, then plain semi-supervised EM, no further anchoring. | Plain PU learning with no identifiability safeguard -- can drift toward the negatives on strongly overlapping data. |

`family='gaussian'` (default), `'student_t'`, or `'vmf'` picks which
mixture family models both blocks; `'student_t'` is a reasonable choice
when either class has outliers, `'vmf'` when `X` is L2-normalized
directional data. `anchor='prior'` is Gaussian-only (NIW conjugate priors
aren't a concept for the other two families) -- use `'fixed'` or `'free'`
with `family='student_t'`/`'vmf'`.

Any keyword not listed above (`covariance_type`, `random_state`,
`max_iter`, ...) is forwarded to the underlying mixture's constructor;
anything not defined on `PositiveUnlabeledMixture` itself (`bic`, `score`,
`sample`, ...) is transparently forwarded to the fitted mixture
(`pu.model_`) too.

See notebook `17_semi_supervised_pu.ipynb` for a worked example of both
features.
