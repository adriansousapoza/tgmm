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

Both families also support partial and full label information through
`fit(X, labels=...)` (see [Semi-Supervised and Positive-Unlabeled
Fitting](semi-supervised-pu.md)). Neither supports Gibbs-sampling mode
(`n_components=None`), the 14 GPCM covariance codes, or MAP/NIW priors --
those stay `GaussianMixture`-only.

**Notation** used on this page and throughout the user guide: $n$ samples
$x_i \in \mathbb{R}^d$, $K$ components, mixing weights $\pi_k$,
responsibilities $r_{ik} = P(z_i = k \mid x_i)$, effective component counts
$n_k = \sum_i r_{ik}$.

## The shared machinery: one EM loop for every family

Every mixture family in tgmm is a finite mixture

$$
p(x) = \sum_{k=1}^{K} \pi_k \, f_k(x \mid \theta_k), \qquad
\ell(\theta) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k \, f_k(x_i \mid \theta_k),
$$

and `BaseMixture` implements the generic loop once. A family only supplies
the component log-density $\log f_k$ (`_component_log_prob`), the
accumulation of its sufficient statistics (`_compute_stats`), and its
M-step (`_m_step_from_stats`).

**E-step.** The responsibilities are computed in log space (a softmax over
components, so nothing under/overflows):

$$
r_{ik} = \frac{\pi_k f_k(x_i)}{\sum_{j} \pi_j f_j(x_i)}
= \exp\!\Big(\log \pi_k + \log f_k(x_i) - \operatorname{logsumexp}_j\big(\log \pi_j + \log f_j(x_i)\big)\Big).
$$

The per-sample normalizer $\log \sum_j \pi_j f_j(x_i)$ is also what
`score_samples` returns, and its mean is `lower_bound_` (the average
log-likelihood $\ell(\theta)/n$ that `tol` is checked against).

**M-step.** Each family maximizes (or, for Student-t, conditionally
maximizes) the expected complete-data log-likelihood
$\sum_{i,k} r_{ik} \big[\log \pi_k + \log f_k(x_i)\big]$. The mixing
weights are always

$$
\pi_k = \frac{n_k}{n}.
$$

**Variants of the loop**, all shared by every family:

| `algorithm` | What replaces the soft $r_{ik}$ fed to the M-step |
| --- | --- |
| `'em'` | $r_{ik}$ itself. |
| `'cem'` | One-hot $\mathbb{1}[k = \arg\max_j r_{ij}]$ (Classification EM). |
| `'sem'` | One-hot of a *sample* $z_i \sim \mathrm{Categorical}(r_{i\cdot})$ (Stochastic EM). |
| `'saem'` | A running average of sampled-assignment statistics with a decaying step size. |

`temperature=(beta_0, n_anneal)` (deterministic annealing) tempers the
E-step: $r_{ik} \propto \big(\pi_k f_k(x_i)\big)^{\beta}$ with $\beta$
rising geometrically from `beta_0` to $1$ over the first `n_anneal`
iterations, $\beta_t = \beta_0^{\,1 - t/n_{\text{anneal}}}$. `step_size`
switches to online/mini-batch EM, blending each batch's sufficient
statistics into a running accumulator. See the [Gaussian Mixture
Model](gaussian-mixture.md#fitting-algorithms) guide for when to use each.

**Model selection.** All families share

$$
\mathrm{BIC} = -2\,\ell(\hat\theta) + p \ln n, \qquad
\mathrm{AIC} = -2\,\ell(\hat\theta) + 2p,
$$

$$
\mathrm{ICL} = \mathrm{BIC} + 2\,\mathrm{EN}(r), \qquad
\mathrm{EN}(r) = -\sum_{i} \ln r_{i\hat z_i} \;\;(\texttt{mode='hard'}),
\quad \text{or} \quad -\sum_{i,k} r_{ik}\ln r_{ik} \;\;(\texttt{mode='soft'}),
$$

(all "lower is better"), where $p$ is the number of free parameters
(`_n_parameters()`, family-specific; given in each section below) and
$\hat z_i = \arg\max_k r_{ik}$. The entropy term is $\ge 0$ and vanishes
when every responsibility is one-hot, so $\mathrm{ICL} \ge \mathrm{BIC}$
with equality only for perfectly separated clusters. Criteria are directly
comparable *within* a family at different $K$ (or covariance structure), and
between families whose densities share a base measure -- Gaussian and Student-t,
both on $\mathbb{R}^d$. They are **not** comparable between a vMF mixture
(density on the sphere) and the other two.

## StudentTMixture: robust to outliers

### Why heavy tails help

A multivariate Student-t distribution looks like a Gaussian near its
center but has polynomially decaying tails, controlled by its degrees of
freedom $\nu$: small $\nu$ means heavy tails (very forgiving of far-away
points), and as $\nu \to \infty$ it converges to a Gaussian exactly. A
Gaussian component's covariance estimate is a plain average of squared
deviations, so a single far-away point can inflate it arbitrarily; the
t-component instead *downweights* such points automatically (derived
below).

### Model

Component $k$ has mean $\mu_k$, scale matrix $\Sigma_k$ and degrees of
freedom $\nu_k$:

$$
t(x \mid \mu_k, \Sigma_k, \nu_k) =
\frac{\Gamma\!\big(\tfrac{\nu_k + d}{2}\big)}{\Gamma\!\big(\tfrac{\nu_k}{2}\big)\,(\nu_k \pi)^{d/2}\,|\Sigma_k|^{1/2}}
\Big(1 + \frac{\delta_{ik}}{\nu_k}\Big)^{-\frac{\nu_k + d}{2}},
\qquad
\delta_{ik} = (x_i - \mu_k)^\top \Sigma_k^{-1} (x_i - \mu_k).
$$

This is exactly what `_component_log_prob` evaluates (and it matches
`scipy.stats.multivariate_t` to machine precision). The key to fitting it is
its **scale-mixture representation**: draw a latent precision multiplier
$w \sim \mathrm{Gamma}(\nu_k/2, \nu_k/2)$ (shape/rate, mean $1$), then
$x \mid w \sim \mathcal{N}(\mu_k, \Sigma_k / w)$. Marginalizing $w$ gives
the t density above. (`sample` uses this representation directly:
$x = \mu_k + z/\sqrt{w}$ with $z \sim \mathcal{N}(0, \Sigma_k)$.)

!!! note "Scale matrix vs. covariance"
    `covariances_` holds the **scale matrix** $\Sigma_k$, not the covariance
    of the distribution. For $\nu_k > 2$ the actual covariance is
    $\frac{\nu_k}{\nu_k - 2}\,\Sigma_k$; for $\nu_k \le 2$ it is infinite.

### E-step: responsibilities *and* robustness weights

Besides the usual $r_{ik}$, the latent multiplier has a closed-form
posterior given $x_i$ and $z_i = k$:
$w_{ik} \mid x_i \sim \mathrm{Gamma}\!\big(\tfrac{\nu_k + d}{2}, \tfrac{\nu_k + \delta_{ik}}{2}\big)$,
whose mean is

$$
u_{ik} = \mathbb{E}[w_{ik} \mid x_i, z_i = k] = \frac{\nu_k + d}{\nu_k + \delta_{ik}}.
$$

Points near the center ($\delta_{ik} \ll \nu_k$) get $u_{ik} \approx
(\nu_k + d)/\nu_k > 1$; points far away ($\delta_{ik} \gg \nu_k$) get
$u_{ik} \to 0$. That is the robustness mechanism: $u_{ik}$ is a per-sample,
per-component weight that shrinks with Mahalanobis distance.
`sample_weights(X)` returns $u_{i\hat z_i}$ at each point's predicted
component, so small values flag likely outliers.

### M-step (ECM)

With $m_k = \sum_i r_{ik} u_{ik}$, the updates are

$$
\pi_k = \frac{n_k}{n}, \qquad
\mu_k = \frac{\sum_i r_{ik} u_{ik}\, x_i}{m_k}, \qquad
\Sigma_k = \frac{\sum_i r_{ik} u_{ik}\,(x_i - \mu_k)(x_i - \mu_k)^\top}{n_k}
\;\;(+\, \texttt{reg\_covar}\cdot I).
$$

Compare with the Gaussian M-step: the only change is the extra weight
$u_{ik}$ inside the sums (and the mean's denominator being $m_k$, not
$n_k$). With $\nu \to \infty$, $u_{ik} \to 1$ and these reduce to the
Gaussian updates (`dof=1e4` is an easy sanity check against
`GaussianMixture`). The statistics $\mu_k$ and $\Sigma_k$ are computed with
$u_{ik}$ evaluated at the *previous* parameters within the same iteration,
which is what makes this an *ECM* (expectation conditional maximization)
algorithm rather than plain EM: each parameter block is maximized
conditionally on the others.

For `covariance_type='diag'` the scatter and the denominator are the
per-feature (diagonal) versions; `'spherical'` uses
$\sigma_k^2 = \sum_i r_{ik} u_{ik}\,\lVert x_i - \mu_k\rVert^2 / (d\, n_k)$;
`'tied_full'` pools the scatter and the denominator across components,
$\Sigma = \sum_k \sum_i r_{ik} u_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top \big/ \sum_k n_k$.

`scale_update='ecme'` divides by $m_k$ instead of $n_k$. At the
maximum-likelihood fixed point the two coincide: the stationarity
condition for $\Sigma_k$ gives $\sum_i r_{ik} u_{ik} \delta_{ik} = d\, n_k$,
and since $u\,\delta = (\nu + d) - \nu\, u$ this forces $m_k = n_k$. Both
variants therefore converge to the same estimate; `'ecme'` just takes a
different path and often needs a few fewer iterations.

### Degrees of freedom

For `dof='per_component'` each $\nu_k$ solves a one-dimensional equation
every M-step. Writing

$$
c_k = \frac{1}{n_k}\sum_i r_{ik}\,\big(\ln u_{ik} - u_{ik}\big),
$$

tgmm solves

$$
1 - \psi\!\Big(\frac{\nu}{2}\Big) + \ln\!\Big(\frac{\nu}{2}\Big) + c_k
+ \psi\!\Big(\frac{\nu + d}{2}\Big) - \ln\!\Big(\frac{\nu + d}{2}\Big) = 0
$$

for $\nu$ ($\psi$ the digamma function) by 60 bisection steps inside
`dof_bounds` followed by 4 Newton steps on $\ln\nu$, clamping to
`dof_bounds` (default `(0.1, 1000)`) -- if the root lies outside the bounds
the fit lands on the bound. This is the first-order condition of the
expected complete-data log-likelihood in $\nu$ (the Gamma$(\nu/2,\nu/2)$
part). For `dof='shared'`, $c$ is the pooled average over all components,
$\sum_k \sum_i r_{ik}(\ln u_{ik} - u_{ik}) / \sum_k n_k$, and one $\nu$ is
solved for. A numeric `dof` fixes every $\nu_k$ for the whole fit.

!!! note "Where this deviates from the textbook"
    The textbook update (Peel & McLachlan, 2000) evaluates the last two
    terms, $\psi\big(\tfrac{\nu^{(t)} + d}{2}\big) - \ln\big(\tfrac{\nu^{(t)} + d}{2}\big)$,
    at the *previous* iterate $\nu^{(t)}$. tgmm evaluates them at the
    candidate $\nu$ itself. The two share the same fixed point (at
    convergence $\nu^{(t)} = \nu$; the residual of the textbook equation at
    a converged tgmm fit is $\sim 10^{-6}$), so the final estimate is the
    same maximum-likelihood solution, but the per-iteration step is not
    literally the textbook EM step.

### Parameters, attributes, and counts

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `weights_` | $(K,)$ | $\pi_k$ |
| `means_` | $(K, d)$ | $\mu_k$ |
| `covariances_` | by `covariance_type` | scale matrix $\Sigma_k$ (not the covariance) |
| `dof_` | $(K,)$ | fitted $\nu_k$ (all entries equal for `'shared'` or a fixed float) |

The number of free parameters used by `bic`/`aic`/`icl` is
$p = (K-1) + Kd + p_\Sigma + p_\nu$, with $p_\Sigma = K\,d(d+1)/2$ (`full`),
$Kd$ (`diag`), $K$ (`spherical`) or $d(d+1)/2$ (`tied_full`), and
$p_\nu = K$ (`per_component`), $1$ (`shared`) or $0$ (fixed `dof`).

### Usage

```python
from tgmm import StudentTMixture

stm = StudentTMixture(n_components=3, covariance_type='full', dof='per_component')
stm.fit(X)

stm.means_        # (n_components, n_features)
stm.covariances_  # the scale matrix, not technically a covariance for finite dof
stm.dof_          # (n_components,) -- fitted nu_k
```

- `covariance_type`: one of `'full'`, `'diag'`, `'spherical'`,
  `'tied_full'` (the GPCM parameterizations and the Gibbs-only axis-aligned
  tied types aren't supported here).
- `dof`: `'per_component'` (default, one $\nu_k$ per component), `'shared'`
  (one $\nu$ for every component), or a fixed float. `dof_init` (default
  `5.0`) is the starting value, `dof_bounds` the clamp.
- `scale_update`: `'ecm'` (default) or `'ecme'`, see above.

### Flagging outliers with `sample_weights`

After fitting, `sample_weights(X)` returns each point's $u_{ik}$ at its
*predicted* component -- small values flag points the model effectively
treated as outliers:

```python
weights = stm.sample_weights(X)
likely_outliers = X[weights < 0.3]
```

### Caveats

- A fitted $\nu_k$ near `dof_bounds[1]` means that component is
  indistinguishable from a Gaussian; near `dof_bounds[0]` means a very
  heavy tail (infinite mean for $\nu \le 1$).
- Each iteration needs the full $(n, K)$ matrix of Mahalanobis distances in
  memory (`batch_size` only chunks the E-step's reduction, not this).
- Student-t and Gaussian densities share the same base measure (Lebesgue on
  $\mathbb{R}^d$), so a `StudentTMixture`'s `score`, `bic`, `aic` and `icl`
  *are* directly comparable with those of a `GaussianMixture` fitted to the
  same data (the t pays for its extra $\nu$ parameters in the count).

## VonMisesFisherMixture: directional data

### Why a different density

For L2-normalized data -- points on the unit hypersphere
$\mathbb{S}^{d-1} = \{x : \lVert x\rVert = 1\}$, as you get from
normalizing text or image embeddings before clustering by cosine
similarity -- a Gaussian (or Student-t) mixture is the wrong model: it's
defined over all of $\mathbb{R}^d$ and has no notion of "direction only."
The von Mises-Fisher (vMF) distribution is the natural analogue restricted
to the sphere.

### Model

Component $k$ has a mean *direction* $\mu_k$ (unit norm) and a
concentration $\kappa_k \ge 0$:

$$
f(x \mid \mu_k, \kappa_k) = C_d(\kappa_k)\,\exp\!\big(\kappa_k\, \mu_k^\top x\big),
\qquad
C_d(\kappa) = \frac{\kappa^{d/2 - 1}}{(2\pi)^{d/2}\, I_{d/2 - 1}(\kappa)},
$$

where $I_\nu$ is the modified Bessel function of the first kind. The
log-density tgmm evaluates is $\log C_d(\kappa_k) + \kappa_k\, x^\top \mu_k$
(verified against `scipy.stats.vonmises_fisher`). The density depends on
$x$ only through the **cosine similarity** $\mu_k^\top x$: higher $\kappa_k$
is a tighter cluster around $\mu_k$, and $\kappa_k \to 0$ approaches the
uniform distribution on the sphere. $\log I_\nu(\kappa)$ is computed from
`scipy.special.ive` (exponentially scaled, so it doesn't overflow), with a
uniform large-order asymptotic expansion as a fallback when `ive`
underflows -- which happens in high dimension at modest $\kappa$.

### E-step and M-step

The E-step is the generic one above with these $f_k$. For the M-step, form
the **resultant vector** of each component,

$$
\mathbf{R}_k = \sum_{i} r_{ik}\, x_i, \qquad
\bar R_k = \frac{\lVert \mathbf{R}_k \rVert}{n_k} \in [0, 1],
$$

(the length of the responsibility-weighted mean vector; $\bar R_k$ near $1$
means the points are nearly parallel). The maximum-likelihood updates are

$$
\pi_k = \frac{n_k}{n}, \qquad
\mu_k = \frac{\mathbf{R}_k}{\lVert \mathbf{R}_k \rVert}, \qquad
A_d(\hat\kappa_k) = \bar R_k, \quad
A_d(\kappa) = \frac{I_{d/2}(\kappa)}{I_{d/2 - 1}(\kappa)}.
$$

$A_d(\kappa) = \bar R$ has no closed-form solution. tgmm follows Banerjee,
Dhillon, Ghosh & Sra (2005): start from the closed-form approximation

$$
\kappa^{(0)} = \frac{\bar R\,(d - \bar R^2)}{1 - \bar R^2},
$$

then refine with **two Newton steps** on $A_d(\kappa) - \bar R = 0$ using
$A_d'(\kappa) = 1 - A_d(\kappa)^2 - \frac{d-1}{\kappa} A_d(\kappa)$ (Sra,
2012), clamping $\kappa$ to $[10^{-3}, 10^{5}]$. (The Banerjee
approximation alone leaves a residual $A_d(\kappa^{(0)}) - \bar R$ of up to
$\sim 10^{-2}$; after the two Newton steps the residual is $\lesssim
10^{-7}$ in the cases we checked.) With `concentration='shared'`, one
$\kappa$ is solved from the pooled resultant length
$\sum_k \lVert\mathbf{R}_k\rVert / \sum_k n_k$.

!!! note "Relation to spherical k-means"
    With `algorithm='cem'`, equal weights and a common $\kappa$, the
    E/C-step assigns each point to the component of largest cosine
    similarity and the M-step sets each mean direction to the normalized sum
    of its members -- i.e. **spherical k-means** (Dhillon & Modha, 2001).
    tgmm's default `init_means='kmeans'` runs exactly that, seeded with a
    cosine-distance k-means++ scheme (the next center is drawn with
    probability proportional to $1 - \max_{\text{chosen}} \cos$) to avoid
    two initial centers landing in one true cluster.

### Parameters, attributes, and counts

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `weights_` | $(K,)$ | $\pi_k$ |
| `means_` | $(K, d)$ | unit-norm mean directions $\mu_k$ |
| `kappa_` | $(K,)$ | fitted concentrations $\kappa_k$ (all equal for `'shared'`) |

Each $\mu_k$ lives on a $(d-1)$-sphere, so $p = (K-1) + K(d-1) + p_\kappa$
with $p_\kappa = K$ (`per_component`) or $1$ (`shared`).

### Usage

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
  $\kappa$ for every component). `kappa_init` (default `10.0`) is the
  starting value.
- `init_means`: `'kmeans'` (default, spherical k-means as above),
  `'random'` (uniform on the sphere), `'points'` (random rows of `X`), or a
  tensor (normalized and used directly). See the [design
  doc](https://github.com/adriansousapoza/tgmm/blob/main/docs/superpowers/specs/2026-10-14-phase4-base-mixture-studentt-vmf-design.md)
  for the seeding failure mode this fixes.
- `sample(n)` uses Wood's (1994) rejection sampler. `reg_covar` is accepted
  for interface parity but unused (there is no covariance).

### Visualizing a vMF fit

`plot_gmm` cannot draw data on a sphere, so `tgmm.plotting` has vMF-specific tools: `plot_vmf` draws a fitted
model on the circle ($d=2$) or sphere ($d=3$) -- points, mean-direction arrows, and spherical caps holding a
chosen probability mass of each component -- and `plot_vmf_cosine` compares, in any dimension, each
component's histogram of angles to its mean direction with the fitted density
$f_{\kappa}(\theta)\propto e^{\kappa\cos\theta}\sin^{d-2}\theta$. See
[von Mises-Fisher mixtures in the plotting guide](plotting.md#von-mises-fisher-mixtures) and the
[tutorial](../notebooks/17_von_mises_fisher.ipynb).

### Caveats

- $\kappa$ is clamped to $[10^{-3}, 10^{5}]$; an extremely tight cluster
  (or duplicate points, $\bar R \to 1$) lands on the upper bound.
- The Bessel normalizer is evaluated per component in Python each E-step,
  which is cheap ($K$ scalars) but not GPU-resident.
- vMF has no notion of cluster *shape* -- every component is isotropic
  around its mean direction. For elongated clusters on the sphere,
  consider PCA-reducing and a `GaussianMixture`/`StudentTMixture` instead.

### Comparing against `GaussianMixture` on normalized data

Nothing stops you from fitting `GaussianMixture(covariance_type='diag')`
on the same L2-normalized data -- it's a reasonable baseline, just not one
that respects the data living on a sphere. Do **not** compare their `bic`/
`aic`/`icl`: a vMF density is taken with respect to the surface measure of
the sphere, a Gaussian density with respect to Lebesgue measure on
$\mathbb{R}^d$, so the log-likelihoods are not on the same scale (information
criteria are comparable only among vMF fits). Compare across families with a
cosine-based clustering metric or agreement with known labels instead; see
notebook 17 for a worked example on high-dimensional embeddings.

## Supervised and semi-supervised fitting

Because `fit(X, labels=...)` lives in `BaseMixture`, both families accept
it: with every row labeled they fit one block of components per class
directly, and with `-1` marking unknown rows they run semi-supervised EM
(`components_per_class`, `unlabeled_weight`, `label_confidence`,
`predict_class`). The math and the options are covered in [Semi-Supervised
and Positive-Unlabeled Fitting](semi-supervised-pu.md).

## Noise, missing values and custom families

`noise=True` works for all three families (the noise density is uniform in the data's bounding box, or on the sphere for vMF);
`missing='em'` for `GaussianMixture` and `StudentTMixture`; `frozen_components` and `sample_weight` for all of them. To fit a family that
tgmm does not ship, implement the three-method `Component` protocol and use `GenericMixture`. See
[Noise, Missing Values and Conditioning](noise-missing-conditioning.md).

## Shared machinery summary

`bic`/`aic`/`icl` work the same way on both classes as on
`GaussianMixture` (see the formulas above). `sample(n)` draws from the
fitted mixture for both: a Gaussian scaled by
$1/\sqrt{\mathrm{Gamma}(\nu_k/2, \nu_k/2)}$ for `StudentTMixture`, and
Wood's (1994) rejection sampler for `VonMisesFisherMixture`.
`algorithm='cem'/'sem'/'saem'`, `temperature` (deterministic annealing),
`step_size` (mini-batch/online EM, plus `partial_fit`), and `n_init` all
work exactly as documented for `GaussianMixture`, since they're
implemented once, in `BaseMixture`, and shared by every family.

## References

- Peel, D. & McLachlan, G. J. (2000). Robust mixture modelling using the t
  distribution. *Statistics and Computing*, 10, 339-348.
- Banerjee, A., Dhillon, I. S., Ghosh, J. & Sra, S. (2005). Clustering on
  the unit hypersphere using von Mises-Fisher distributions. *JMLR*, 6,
  1345-1382.
- Sra, S. (2012). A short note on parameter approximation for von
  Mises-Fisher distributions. *Computational Statistics*, 27, 177-190.
- Wood, A. T. A. (1994). Simulation of the von Mises Fisher distribution.
  *Communications in Statistics -- Simulation and Computation*, 23(1),
  157-164.
- Dhillon, I. S. & Modha, D. S. (2001). Concept decompositions for large
  sparse text data using clustering. *Machine Learning*, 42, 143-175.
- Biernacki, C., Celeux, G. & Govaert, G. (2000). Assessing a mixture model
  for clustering with the integrated completed likelihood. *IEEE TPAMI*,
  22(7), 719-725.
