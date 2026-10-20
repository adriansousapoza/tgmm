# Semi-Supervised and Positive-Unlabeled Fitting

Every tgmm mixture family (`GaussianMixture`, `StudentTMixture`,
`VonMisesFisherMixture`) can be fit with partial label information, from
"every row has a known class" down to "I only know some rows are
positive, and nothing about the rest." This page covers the model, the
algorithm (with the equations tgmm actually implements), the API, and which
option to reach for.

**Notation:** $n$ samples $x_i$, $K$ components with weights $\pi_k$ and
densities $f_k$ (Gaussian, Student-t or von Mises-Fisher), responsibilities
$r_{ik}$, effective counts $n_k = \sum_i r_{ik}$, log-likelihood $\ell(\theta)$.

## The model: classes own blocks of components

Let the classes be $c \in \{1, \dots, C\}$ (these are `classes_`, the sorted
distinct label values). The $K$ components are partitioned into contiguous
**blocks** $B_c$, one per class: with `components_per_class=m` every class
owns $m$ components (so $K = mC$), and with a dict, class $c$ owns
`components_per_class[c]` components -- the same idea as mclust's MclustDA.
`class_of_component_[k]` says which class component $k$ belongs to.

The density of an observation *without* a label is the ordinary mixture

$$
p(x) = \sum_{k=1}^{K} \pi_k f_k(x),
$$

and the joint density of an observation $x$ *with* known class $c$ is the
restriction of that sum to class $c$'s block:

$$
p(x, c) = \sum_{k \in B_c} \pi_k f_k(x).
$$

So the weights $\pi_k$ are joint class-and-component probabilities
(summing to one over *all* components), and class proportions in the
labeled data influence them. With $L$ the set of labeled rows (class $c_i$
for $i \in L$) and $U$ the unlabeled rows, the semi-supervised
log-likelihood is

$$
\ell_{\text{SS}}(\theta) = \sum_{i \in L} \log \sum_{k \in B_{c_i}} \pi_k f_k(x_i)
\;+\; \sum_{i \in U} \log \sum_{k=1}^{K} \pi_k f_k(x_i).
$$

This is exactly what `lower_bound_` reports for a semi-supervised fit
(averaged over rows).

## Fitting with labels

```python
import torch
from tgmm import GaussianMixture

labels = torch.full((X.shape[0],), -1)
labels[known_idx] = known_labels  # -1 everywhere else

gmm = GaussianMixture(n_components=3)
gmm.fit(X, labels=labels)
gmm.predict_class(X)  # class labels, not raw component indices
```

`n_components` must equal the total number of components the
`components_per_class` layout implies (for the default `1`, the number of
distinct labels). Label values needn't be contiguous integers.

### Fully labeled (no `-1`): no iteration needed

When every row is labeled, the class of each row never depends on the
current parameters, so one M-step is exact:

- With one component per class, each row's responsibility is the one-hot
  vector of its class, and a single M-step gives the per-class weight
  $\pi_c = n_c/n$, mean and covariance (or $\mu_c$, $\kappa_c$, ...). This
  is the fully-labeled limit of Classification EM. `n_iter_` is `1`,
  `converged_` is `True`.
- With $m_c > 1$ components for a class, that class's block is fit by
  ordinary unsupervised EM on *only* that class's rows, and its weights are
  rescaled by $n_c / n$ so they sum to the class proportion.
  (Tied covariance types can't be assembled from independent blocks and
  raise a `ValueError` in this case.)

Any configured `mean_prior`/`covariance_prior` (Gaussian) still applies,
giving a regularized supervised fit for classes with few samples.

### Partially labeled (some `-1`): semi-supervised EM

**E-step.** Compute the ordinary posterior
$r_{ik} \propto \pi_k f_k(x_i)$ for every row, then *clamp* the labeled
rows to their class block: for $i \in L$,

$$
r_{ik} \;\leftarrow\;
\begin{cases}
\dfrac{\pi_k f_k(x_i)}{\sum_{j \in B_{c_i}} \pi_j f_j(x_i)} & k \in B_{c_i},\\[2mm]
0 & k \notin B_{c_i},
\end{cases}
$$

(one-hot when the block has a single component). A labeled row can then
never "vote" for another class, but with $m_c > 1$ it still softly splits
across its own class's components. Unlabeled rows keep the full, ordinary
posterior. This is the exact EM E-step for $\ell_{\text{SS}}$, because the
posterior over components given that $x_i$ is known to belong to class $c_i$
is the in-block renormalization above.

**M-step.** The family's usual M-step, run on the (clamped) $r_{ik}$ -- for
the Gaussian, $n_k$, $\bar y_k = \sum_i r_{ik}x_i / n_k$ and
$W_k = \sum_i r_{ik}(x_i - \bar y_k)(x_i - \bar y_k)^\top$ feed the same
updates as in the [Gaussian guide](gaussian-mixture.md), and the
Student-t/vMF M-steps are as in [Other Mixture
Families](other-mixture-families.md).

With `unlabeled_weight=1` and `label_confidence=1` this is plain EM on
$\ell_{\text{SS}}$, so the log-likelihood never decreases. A few labels per
class are often enough to resolve what plain unsupervised EM can't --
which cluster is which class, and where the boundary falls on overlapping
data. (The initialization itself is still unsupervised, e.g. k-means; the
clamping then pulls each block onto its class. Use `n_init > 1` if a poor
start matters.)

### The two weighting knobs

**`unlabeled_weight` $=w$ (default `1.0`).** Before the M-step, every
unlabeled row's responsibilities are multiplied by $w$:
$r_{ik} \leftarrow w\, r_{ik}$ for $i \in U$. Unlabeled rows then
contribute $w\sum_{i\in U} r_{ik}$ to $n_k$, $\bar y_k$ and $W_k$, while
labeled rows count fully. $w = 0$ ignores the unlabeled data (a fit on the
labeled rows alone); $w = 1$ treats all rows equally. Small $w$ protects
against a mis-specified component density letting the (typically much more
numerous) unlabeled rows swamp the labels.

**`label_confidence` $=c$ (default `1.0`).** For labels you don't fully
trust, a labeled row's responsibility is the blend

$$
r_{i\cdot} = c\; r^{\text{in-block}}_{i\cdot} + (1 - c)\; r^{\text{out-of-block}}_{i\cdot},
\qquad
r^{\text{out-of-block}}_{ik} = \frac{\pi_k f_k(x_i)\,\mathbb{1}[k \notin B_{c_i}]}{\sum_{j \notin B_{c_i}} \pi_j f_j(x_i)},
$$

i.e. mass $c$ stays on the stated class (renormalized inside the block, as
above) and mass $1-c$ goes to the *other* classes' components in
proportion to their posterior -- a soft correction for occasional
mislabeling. It is a heuristic for noisy labels, not part of the
likelihood: `lower_bound_` still evaluates $\ell_{\text{SS}}$ with the labels
taken at face value.

**CEM.** With `algorithm='cem'`, unlabeled rows are hard-assigned to their
highest-responsibility component each iteration; labeled rows keep their
clamped (soft or one-hot) responsibilities.

**Scope note:** semi-supervised fitting (partial labels) currently
supports `algorithm='em'`/`'cem'` only; `'sem'`/`'saem'`/online EM
(`step_size` set) raise `NotImplementedError`. Gibbs-sampling mode
(`n_components=None`) is not supported with `labels`, and tied covariance
types combined with `components_per_class > 1` aren't supported either.

### More than one component per class

Real classes aren't always unimodal. `components_per_class` (an `int`, or
a `dict` keyed by label value) gives a class more than one component:

```python
gmm = GaussianMixture(n_components=3, components_per_class={0: 2, 1: 1})
gmm.fit(X, labels=labels)  # class 0 gets 2 components, class 1 gets 1
```

`predict(X)` returns raw component indices; `predict_class(X)` sums each
row's responsibility over every class block and returns the class with the
largest total, mapped back through `classes_`:

$$
\hat c(x) = \arg\max_c \sum_{k \in B_c} r_k(x).
$$

When `components_per_class` is a dict, its *keys* define the full set of
classes (a key needn't appear in `labels`); every labeled value must be one
of those keys.

## Positive-unlabeled (PU) learning

**Positive-unlabeled learning** is the more extreme case: you know some
rows are positive (e.g. confirmed cancer cells), and *everything else is
unlabeled* -- it could be positive or negative, you just don't know.
`PositiveUnlabeledMixture` models the unlabeled density as

$$
p_U(x) = \pi\, f_P(x) + (1-\pi)\, f_N(x),
$$

with $f_P$ and $f_N$ themselves mixtures (a positive block of
`n_components_positive` components and a negative block of
`n_components_negative`), and estimates both densities and $\pi$ jointly:

```python
from tgmm import PositiveUnlabeledMixture

pu = PositiveUnlabeledMixture(n_components_positive=1, n_components_negative=5)
pu.fit(X, positive_mask)  # bool mask, True = confirmed positive

pu.class_prior_                  # estimated fraction of *unlabeled* rows that are positive
pu.predict_proba_positive(X)     # P(positive) for arbitrary X
pu.predict_positive(X, threshold=0.5)
```

### How it is fit

`PositiveUnlabeledMixture` is a thin wrapper over the semi-supervised fit
above: it builds the chosen `family`'s mixture with
`components_per_class={1: n_components_positive, 0: n_components_negative}`
(so the negative block comes *first* in component order, the positive block
last), labels `1` on `positive_mask` and `-1` everywhere else, and fits it.
The labeled positives are clamped to the positive block; each unlabeled row
keeps the full posterior over both blocks. The estimates are

$$
P(\text{positive} \mid x) = \sum_{k \in P} r_k(x) \quad(\texttt{predict\_proba\_positive}),
\qquad
\hat\pi = \frac{1}{|U|} \sum_{i \in U} \sum_{k \in P} r_{ik} \quad(\texttt{class\_prior\_}),
$$

i.e. `class_prior_` is the average total positive-block responsibility over
the *unlabeled* rows only. (Do not read it off `weights_`: the mixture
weights are estimated over *all* rows, labeled positives included, so the
positive block's total weight is $\approx (n_L + \hat\pi\,|U|)/n$, larger
than $\hat\pi$.)

This assumes **SCAR** (Selected Completely At Random; Elkan & Noto, 2008):
the labeled positives are a uniform random sample of *all* true positives,
not a biased subset (e.g. "the easiest-to-confirm cases"). Under SCAR the
labeled positives are draws from $f_P$ itself, which is what lets them be
clamped to the positive block. If your labeled positives were selected some
other way, `class_prior_` and the fitted densities can be biased -- SCAR is
a real assumption, not a formality.

This is also sometimes called **mixture proportion estimation**: the whole
point is estimating $\pi$, the mixing proportion of the (fully-unobserved)
positive class within the unlabeled pool. A related, distinct problem is
**semi-supervised novelty detection** (Blanchard, Lee & Scott, 2010): given
a labeled "normal" sample and an unlabeled mix of normal and novel points,
decide *which* unlabeled points are novel -- closer to `anchor='fixed'`
below (treating the labeled class as a known, unmoving reference density)
than to PU's `'prior'` default.

### Picking `anchor`

The mixture proportion $\pi$ is only identifiable if the negative block
can't simply absorb the positive one (a sufficiently flexible negative
block can always explain every unlabeled point, driving $\hat\pi \to 0$).
`anchor` controls how tightly the positive block stays tied to the labeled
positives. All three start from a first fit of the positive block on the
labeled positives alone, $(\mu_P^{0}, \Sigma_P^{0})$:

| `anchor` | Positive block behavior | Corresponds to |
| --- | --- | --- |
| `'fixed'` | Frozen at the labeled-only fit after every M-step; only the negative block (and the weights) keep moving. | Classical mixture-proportion estimation / semi-supervised novelty detection (the "normal" class is a fixed reference). |
| `'prior'` (default) | Used as an NIW prior of strength $n_L$ (below), so unlabeled positives can still refine it without drifting toward the negatives. **Gaussian-only.** | A softer version of the same -- lets genuinely-positive-but-different-looking unlabeled points still pull the estimate, within reason. |
| `'free'` | Initialized from the labeled-only fit, then plain semi-supervised EM, no further anchoring. | Plain PU learning with no identifiability safeguard -- can drift toward the negatives on strongly overlapping data. |

For `'prior'`, after each M-step the positive block's parameters are
replaced by an NIW posterior (the same conjugate update as in the
[Gaussian guide](gaussian-mixture.md)) with prior mean $\mu_P^0$, prior
scale $\Psi_0 = n_L\,\Sigma_P^0$ and prior strength
$\lambda_0 = \nu_0 = n_L$ (the number of labeled positives), using the
positive block's current responsibilities:

$$
\mu_P = \frac{\lambda_0 \mu_P^0 + n_P \bar y_P}{\lambda_0 + n_P}, \qquad
\Sigma_P = \frac{\Psi_0 + W_P + \frac{\lambda_0 n_P}{\lambda_0 + n_P}(\bar y_P - \mu_P^0)(\bar y_P - \mu_P^0)^\top}{\nu_0 + n_P},
$$

where $n_P$, $\bar y_P$, $W_P$ are that block's effective count, weighted
mean and scatter. The labeled positives therefore enter twice -- once
through the prior, once as data with responsibility $1$ -- which anchors
the block strongly; the prior is worth about $n_L$ pseudo-observations.

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
(`pu.model_`) too. `n_components_negative=None` (an unbounded Gibbs
sampler for the negative block) is not implemented and raises
`NotImplementedError`.

### Caveats

- **`class_prior_` quality** depends on the negative block being able to
  model the negative class without wandering into the positive region;
  more negative components make that easier *and* make identifiability
  harder -- use `anchor='fixed'`/`'prior'` and check `class_prior_` is
  stable across `n_components_negative`.
- Labeled positives are clamped to the positive block, so (with the default
  `label_confidence=1`) a mislabeled "positive" is never corrected. To allow
  that, pass `label_confidence=c` with `c < 1` through `**family_kwargs`;
  the unlabeled rows are handled as usual.
- `predict_proba_positive` uses the fitted mixture weights, which include
  the labeled positives (see above); `class_prior_` is the right estimate
  of the positive fraction in a fresh unlabeled sample.

See notebook `18_semi_supervised_pu.ipynb` for a worked example of both
features.

## References

- Elkan, C. & Noto, K. (2008). Learning classifiers from only positive and
  unlabeled data. *KDD*.
- Fraley, C. & Raftery, A. E. (2002). Model-based clustering, discriminant
  analysis, and density estimation. *JASA*, 97(458), 611-631 (MclustDA).
- Blanchard, G., Lee, G. & Scott, C. (2010). Semi-supervised novelty
  detection. *JMLR*, 11, 2973-3009.
