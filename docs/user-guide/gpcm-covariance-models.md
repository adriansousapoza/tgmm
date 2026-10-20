# GPCM Covariance Models

`GaussianMixture` supports all 14 Celeux & Govaert (1995) Gaussian
Parsimonious Clustering Models (GPCM) -- the same family `mclust` implements
-- not just the six classic covariance types in the main
[Gaussian Mixture Model](gaussian-mixture.md#covariance-types-at-a-glance) guide. This
page defines the models, states the M-step each one uses and why it is
correct, and lists what is and isn't supported. The
[Covariance Structures tutorial](../notebooks/15_covariance_models.ipynb)
walks through the same material with runnable code, figures, and a numerical
check of every formula below.

## Where the covariance enters EM

Given responsibilities $r_{ik}$ from the E-step, write

$$
n_k=\sum_{i} r_{ik},\qquad
\bar y_k=\frac1{n_k}\sum_i r_{ik}x_i,\qquad
W_k=\sum_{i} r_{ik}\,(x_i-\bar y_k)(x_i-\bar y_k)^{\top}.
$$

After maximising the expected complete-data log-likelihood over $\pi_k$ and
$\mu_k$, the only terms left that depend on the covariances are, up to a
factor $-\tfrac12$,

$$
J(\Sigma_1,\dots,\Sigma_K)=\sum_{k=1}^{K}\Big[\,n_k\log|\Sigma_k|+\operatorname{tr}\big(\Sigma_k^{-1}W_k\big)\Big].
$$

**Every covariance type is the same objective $J$ minimised over a different
feasible set**, and the M-step needs the data only through $(n_k, W_k)$ (the
`SufficientStats.nk` and `.s2` fields). For `full` the minimiser is
$\Sigma_k=W_k/n_k$; the other classic types are the restrictions listed in the
[Gaussian Mixture Model guide](gaussian-mixture.md#covariance-types-at-a-glance).
`reg_covar` is added to the diagonal of the assembled matrices afterwards.

## The idea: a decomposed covariance

Each component's covariance is written through its eigendecomposition as

$$
\Sigma_k=\lambda_k\,D_k\,A_k\,D_k^{\top},\qquad
\det A_k=1,\quad D_k^{\top}D_k=I,\quad A_k=\operatorname{diag}(a_{k1}\ge\dots\ge a_{kd}>0).
$$

- $\lambda_k=|\Sigma_k|^{1/d}$ (**volume**): a scalar controlling the
  component's overall size (its ellipsoid's volume scales as $\lambda_k^{d/2}$).
- $A_k$ (**shape**): a diagonal matrix with determinant 1 and decreasing
  entries -- how stretched the component is along its own axes.
- $D_k$ (**orientation**): an orthogonal matrix -- which way those axes point.

The semi-axes of the one-standard-deviation ellipsoid are
$\sqrt{\lambda_k a_{kj}}$ along the $j$-th column of $D_k$.

Each factor can be **E**qual across components, **V**ary per component, or
(for shape and orientation only) be the **I**dentity: $A_k=I$ makes the
component spherical, and $D_k=I$ makes it axis-aligned (no rotation). Volume
has no identity option -- it is always a free scalar, shared or not. The 14
valid combinations are named by their three-letter mclust code
(Volume-Shape-Orientation); the last column is the number of free covariance
parameters, which is what `bic`/`aic` charge for the covariances:

| Code | tgmm name | Meaning | Free covariance parameters |
| --- | --- | --- | --- |
| `EII` | `'tied_spherical'` | one shared variance, spherical | $1$ |
| `VII` | `'spherical'` | per-component variance, spherical | $K$ |
| `EEI` | `'tied_diag'` | one shared diagonal covariance | $d$ |
| `VEI` | *(new)* | per-component volume, shared diagonal shape | $K+(d-1)$ |
| `EVI` | *(new)* | shared volume, per-component diagonal shape | $1+K(d-1)$ |
| `VVI` | `'diag'` | per-component diagonal covariance | $Kd$ |
| `EEE` | `'tied_full'` | one shared full covariance | $d(d+1)/2$ |
| `VEE` | *(new)* | per-component volume, shared shape & orientation | $K+(d-1)+d(d-1)/2$ |
| `EVE` | *(new)* | shared volume & orientation, per-component shape | $1+K(d-1)+d(d-1)/2$ |
| `EEV` | *(new)* | shared volume & shape, per-component orientation | $d+K\,d(d-1)/2$ |
| `VVE` | *(new)* | shared orientation, per-component volume & shape | $Kd+d(d-1)/2$ |
| `VEV` | *(new)* | shared shape, per-component volume & orientation | $K+(d-1)+K\,d(d-1)/2$ |
| `EVV` | *(new)* | shared volume, per-component shape & orientation | $1+K(d-1)+K\,d(d-1)/2$ |
| `VVV` | `'full'` | fully unconstrained per-component covariance | $K\,d(d+1)/2$ |

The counts are sums of three independent contributions:

| factor | `E` | `V` | `I` |
| --- | --- | --- | --- |
| volume $\lambda$ | $1$ | $K$ | -- |
| shape $A$ ($d-1$ free ratios) | $d-1$ | $K(d-1)$ | $0$ |
| orientation $D$ ($d(d-1)/2$ free angles) | $d(d-1)/2$ | $K\,d(d-1)/2$ | $0$ |

The total number of parameters adds $K-1$ free mixture weights and $Kd$ mean
coordinates. The 14 models are *nested*: a model is a special case of another
if it is at least as constrained in all three factors (V before E before I),
so $\min J$ can only grow as constraints are added.

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
for the six classic types, which don't compute one). `reg_covar` is added to
the assembled matrices, not to the decomposition, so the three factors
describe the unregularised fit.

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

## How each model is estimated

With the decomposition, $|\Sigma_k|=\lambda_k^d$ and
$\Sigma_k^{-1}=\lambda_k^{-1}D_kA_k^{-1}D_k^{\top}$, so

$$
J=\sum_{k}\Big[\,n_k\,d\log\lambda_k+\frac1{\lambda_k}\operatorname{tr}\big(A_k^{-1}D_k^{\top}W_kD_k\big)\Big].
$$

Two lemmas do most of the work.

!!! note "Lemma 1 (shape)"
    For symmetric positive-definite $B$ and positive-definite $C$ with
    $\det C=1$: $\operatorname{tr}(C^{-1}B)\ge d(\det B)^{1/d}$, with equality
    iff $C=B/(\det B)^{1/d}$.

    *Proof.* The eigenvalues $\mu_j$ of $C^{-1/2}BC^{-1/2}$ multiply to
    $\det B$. By AM-GM, $\sum_j\mu_j\ge d(\prod_j\mu_j)^{1/d}$, with equality
    iff all $\mu_j$ are equal, i.e. $B\propto C$.

!!! note "Lemma 2 (orientation)"
    For symmetric $W$ with eigenvalues $\omega_1\ge\dots\ge\omega_d$ and
    $a_1\ge\dots\ge a_d>0$, over orthogonal $D$:
    $\operatorname{tr}(A^{-1}D^{\top}WD)\ge\sum_j\omega_j/a_j$, attained when
    the columns of $D$ are the eigenvectors of $W$ in decreasing eigenvalue
    order. (The diagonal of $D^{\top}WD$ is majorised by $\omega$ -- Schur-Horn --
    and increasing weights $1/a_j$ are minimised by pairing the largest
    $\omega$ with the largest $a$.) This is why `tgmm` keeps shapes sorted in
    decreasing order.

### Closed-form models

The six classic codes (`EII`, `VII`, `EEI`, `VVI`, `EEE`, `VVV`) have the
closed forms of the
[Gaussian Mixture Model guide](gaussian-mixture.md#covariance-types-at-a-glance). Three
of the new ones do too:

| Code | Estimate |
| --- | --- |
| `EVI` | $A_k=B_k/\lvert B_k\rvert^{1/d}$ with $B_k=\operatorname{diag}(W_k)$; $\lambda=\frac1n\sum_k\lvert B_k\rvert^{1/d}$ |
| `EEV` | $D_k$ = eigenvectors of $W_k$ (eigenvalues $\Omega_k$); with $\Omega=\sum_k\Omega_k$: $A=\Omega/\lvert\Omega\rvert^{1/d}$, $\lambda=\lvert\Omega\rvert^{1/d}/n$ |
| `EVV` | $C_k=D_kA_kD_k^{\top}=W_k/\lvert W_k\rvert^{1/d}$; $\lambda=\frac1n\sum_k\lvert W_k\rvert^{1/d}$ |

(For `EVI`, only $\operatorname{diag}(W_k)$ enters $\operatorname{tr}(A_k^{-1}W_k)$;
Lemma 1 gives $A_k$ and $J=nd\log\lambda+\frac d\lambda\sum_k|B_k|^{1/d}$, minimised
at the stated $\lambda$. `EEV` and `EVV` follow the same pattern, using
Lemma 2 for `EEV`'s orientations.)

### The iterative models: `VEI`, `VEE`, `VEV`

With per-component volumes and a shared shape, the factors are coupled but each
has a closed form *given the other*, so tgmm alternates:

- *given the shape*: $\lambda_k=\operatorname{tr}(A^{-1}D_k^{\top}W_kD_k)/(n_k d)$;
- *given the volumes*: $A=B/|B|^{1/d}$ (Lemma 1) with
  $B=\sum_kD_k^{\top}W_kD_k/\lambda_k$ -- namely $\operatorname{diag}(\sum_kW_k/\lambda_k)$
  for `VEI`; $\sum_kW_k/\lambda_k$ for `VEE`, where the shared matrix
  $C=DAD^{\top}$ is updated as a whole; and $\sum_k\Omega_k/\lambda_k$ for
  `VEV`, where each $D_k$ is fixed once and for all to the eigenvectors of
  $W_k$ (Lemma 2).

Each sub-step is an exact minimiser, so $J$ is non-increasing
(coordinate descent).

### The two numerically-optimised models: `EVE`, `VVE`

Given a shared orientation $D$ everything else is closed-form: with
$B_k=\operatorname{diag}(D^{\top}W_kD)$,

$$
A_k=\frac{B_k}{|B_k|^{1/d}},\qquad
\lambda=\frac1n\sum_k|B_k|^{1/d}\ \ (\texttt{EVE}),\qquad
\lambda_k=\frac{|B_k|^{1/d}}{n_k}\ \ (\texttt{VVE}).
$$

Given the shapes and volumes, the best $D$ minimises
$\sum_k\lambda_k^{-1}\operatorname{tr}(A_k^{-1}D^{\top}W_kD)$ over orthogonal
matrices -- the *common principal components* problem (Flury, 1984), which has
no closed form for $K>1$. (Profiling out $\lambda_k$ and $A_k$ from `VVE`
gives Flury's criterion $\sum_kn_k\log|\operatorname{diag}(D^{\top}S_kD)|$,
$S_k=W_k/n_k$, which by Hadamard's inequality is bounded below by the `VVV`
optimum, with equality iff one $D$ diagonalises every $S_k$.) tgmm solves it
with a few L-BFGS steps on the orthogonal manifold (PyTorch's `orthogonal`
parametrisation), warm-started at the previous $D$, and **accepts the result
only if the objective decreases** -- a *generalised* EM step (Dempster, Laird
& Rubin, 1977): $Q$ is improved rather than exactly maximised, which keeps the
log-likelihood monotone across EM iterations as for every other covariance
type.

### Stopping and warm starts

`VEI`, `VEE`, `VEV`, `EVE` and `VVE` repeat their inner alternation until the
relative change in $J$ falls below `gpcm_tol` (default `1e-6`) or
`gpcm_max_iter` (default `20`) sweeps are done:

```python
gmm = GaussianMixture(n_components=3, n_features=4, covariance_type='VEE',
                       gpcm_tol=1e-8, gpcm_max_iter=50)
```

Each EM iteration's alternation is *warm-started* from the previous EM
iteration's decomposition, so in practice it usually takes only 1-2 sweeps to
re-settle.

### Numerical verification

The tutorial compares every code's `mstep_gpcm` objective
value with an independent brute-force L-BFGS minimisation of $J$ over each
model's feasible set (log-volumes, mean-centred log-shapes for $\det A=1$,
orthogonal matrices from the matrix exponential of a skew-symmetric one): all
14 agree to optimiser tolerance, including the alternating and
orthogonal-manifold models.

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

## Reference

Celeux, G. & Govaert, G. (1995). Gaussian parsimonious clustering models.
*Pattern Recognition*, 28(5), 781-793. Banfield, J. D. & Raftery, A. E.
(1993). Model-based Gaussian and non-Gaussian clustering. *Biometrics*, 49(3),
803-821. Flury, B. N. (1984). Common principal components in $k$ groups.
*JASA*, 79(388), 892-898. `mclust`'s own naming and model definitions
(Scrucca, Fop, Murphy & Raftery, 2016, *The R Journal* 8(1)) are followed here
for compatibility, but every M-step in tgmm is derived independently from the
papers' equations -- see
`docs/superpowers/specs/2026-10-14-phase3-gpcm-covariance-models-design.md`
for the original derivation (tgmm is MIT-licensed; `mclust` is GPL, and its
source was not read or ported).
