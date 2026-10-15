# Gaussian Mixture Model

The `GaussianMixture` class is the core component of tgmm. It fits a mixture of Gaussians by
Expectation-Maximization (EM) or one of its variants (classification, stochastic, annealed,
online), with optional Bayesian priors (MAP estimation), a choice of 6 classic or 14
eigen-decomposed (GPCM) covariance structures, semi-supervised fitting, and a Dirichlet-process
(Gibbs-sampling) mode that infers the number of components.

This page explains **what is computed**: the model, every update equation the code implements,
and how each option changes them. For runnable examples see the [Tutorials](../tutorials/index.md).

!!! note "Notation (used throughout the documentation)"
    $n$ samples $x_i\in\mathbb{R}^d$, $K$ components. $r_{ik}$ is the responsibility of component
    $k$ for sample $i$, and

    $$
    n_k=\sum_i r_{ik},\qquad
    \bar y_k=\frac{1}{n_k}\sum_i r_{ik}\,x_i,\qquad
    W_k=\sum_i r_{ik}\,(x_i-\bar y_k)(x_i-\bar y_k)^\top .
    $$

    $n_k$ is the "effective sample count", $\bar y_k$ the responsibility-weighted mean and $W_k$ the
    responsibility-weighted scatter matrix. Together with $n$ these are the **sufficient
    statistics** of the M-step (`tgmm.SufficientStats` stores $n_k$, $\sum_i r_{ik}x_i$ and
    $W_k$ -- or its diagonal / trace for `diag` / `spherical` types).

## The model

A Gaussian mixture represents the density of the data as a weighted sum of $K$ Gaussians:

$$
p(x\mid\theta)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x\mid\mu_k,\Sigma_k),\qquad
\pi_k\ge 0,\ \ \sum_k\pi_k=1,
$$

$$
\mathcal{N}(x\mid\mu,\Sigma)=(2\pi)^{-d/2}\,|\Sigma|^{-1/2}
\exp\!\Big(-\tfrac12 (x-\mu)^\top\Sigma^{-1}(x-\mu)\Big),
$$

with parameters $\theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^K$. Equivalently, there is a hidden
(latent) label $z_i\in\{1,\dots,K\}$ per sample with $p(z_i=k)=\pi_k$ and
$x_i\mid z_i=k\sim\mathcal{N}(\mu_k,\Sigma_k)$; clustering means inferring $z_i$.

Maximum likelihood maximizes the observed-data log-likelihood

$$
\ell(\theta)=\sum_{i=1}^n\log\sum_{k=1}^K\pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k),
$$

which has a log of a sum and so no closed-form maximizer. If the labels *were* observed, the
**complete-data** log-likelihood

$$
\ell_c(\theta)=\sum_{i,k}\mathbb{1}[z_i=k]\,\big(\log\pi_k+\log\mathcal{N}(x_i\mid\mu_k,\Sigma_k)\big)
$$

would split into one independent, closed-form problem per component. EM exploits this.

```python
from tgmm import GaussianMixture

gmm = GaussianMixture(
    n_components=3,         # number of Gaussian components K
    n_features=2,           # data dimensionality d (optional: inferred from X)
    covariance_type='full'  # structure of each Sigma_k
)
gmm.fit(X)

labels = gmm.predict(X)              # hard assignments, argmax_k r_ik
probabilities = gmm.predict_proba(X)  # soft assignments r_ik
log_likelihood = gmm.score(X)         # average log-likelihood (1/n) * ell(theta)
```

## How EM works

For any distribution $q_{ik}$ over the labels, Jensen's inequality gives a lower bound on $\ell$:

$$
\ell(\theta)\ \ge\ \mathcal{F}(q,\theta)=\sum_{i,k}q_{ik}\log\frac{\pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{q_{ik}},
$$

with equality exactly when $q_{ik}=p(z_i=k\mid x_i,\theta)$. EM alternates

1. **E-step**: set $q$ to that posterior (making the bound tight at the current $\theta^{(t)}$);
2. **M-step**: maximize $\mathcal{F}(q,\theta)$ over $\theta$.

Then $\ell(\theta^{(t+1)})\ge\mathcal{F}(q,\theta^{(t+1)})\ge\mathcal{F}(q,\theta^{(t)})=\ell(\theta^{(t)})$:
**the log-likelihood never decreases**, so EM converges to a stationary point (a local
maximum in practice, which is why initialization and `n_init` matter). With priors the same
argument applies to the log-posterior $\ell(\theta)+\log p(\theta)$ (MAP-EM).

### E-step: responsibilities

$$
r_{ik}=p(z_i=k\mid x_i,\theta)=
\frac{\pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_{j}\pi_j\,\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}
=\operatorname{softmax}_k\!\big(\log\pi_k+\log\mathcal{N}(x_i\mid\mu_k,\Sigma_k)\big).
$$

tgmm works entirely in log space: the per-component log-density
$\log\mathcal{N}(x\mid\mu_k,\Sigma_k)=-\tfrac12\big[d\log 2\pi+\log|\Sigma_k|+(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)\big]$
is evaluated per covariance type (for `full`, via a Cholesky factor $\Sigma_k=L_kL_k^\top$:
$\log|\Sigma_k|=2\sum\log\operatorname{diag}L_k$, and the Mahalanobis term by a triangular
solve; for `diag` / `spherical` / `tied_*` the corresponding cheap closed forms), the mixing
weight is added, and the normalizer is a `logsumexp`. The same normalizer gives the
per-sample log-likelihood $\log p(x_i\mid\theta)$, whose mean is `lower_bound_`
(and what `score` returns).

### M-step: maximum-likelihood updates

Maximizing $\mathcal{F}$ over $\theta$ (using the responsibilities as soft counts) gives

$$
\pi_k=\frac{n_k}{n},\qquad \mu_k=\bar y_k,
$$

and a covariance update that depends on `covariance_type`. For each component the
$\Sigma_k$-dependent part of $\mathcal{F}$ is

$$
-\tfrac12\Big[n_k\log|\Sigma_k|+\operatorname{tr}\big(\Sigma_k^{-1}W_k\big)\Big]
\qquad(\text{at }\mu_k=\bar y_k),
$$

so constraining $\Sigma_k$ to a family and minimizing $n_k\log|\Sigma_k|+\operatorname{tr}(\Sigma_k^{-1}W_k)$
within that family yields the closed forms below. With $\varepsilon$ = `reg_covar`
(added to keep $\Sigma$ positive definite):

| `covariance_type` | Shape of `covariances_` | Update | Free covariance parameters |
| --- | --- | --- | --- |
| `full` | $(K,d,d)$ | $\Sigma_k=\dfrac{W_k}{n_k}+\varepsilon I$ | $K\,d(d+1)/2$ |
| `diag` | $(K,d)$ | $\sigma^2_{kj}=\dfrac{(W_k)_{jj}}{n_k}+\varepsilon$ | $Kd$ |
| `spherical` | $(K,)$ | $\sigma_k^2=\dfrac{\operatorname{tr}W_k}{n_k\,d}+\varepsilon$ | $K$ |
| `tied_full` | $(d,d)$ | $\Sigma=\dfrac{\sum_kW_k}{\sum_kn_k}+\varepsilon I$ | $d(d+1)/2$ |
| `tied_diag` | $(d,)$ | $\sigma^2_{j}=\dfrac{\sum_k(W_k)_{jj}}{\sum_kn_k}+\varepsilon$ | $d$ |
| `tied_spherical` | scalar | $\sigma^2=\dfrac{\sum_k\operatorname{tr}W_k}{d\sum_kn_k}+\varepsilon$ | $1$ |

Why these forms: for `full` the minimizer of $n_k\log|\Sigma|+\operatorname{tr}(\Sigma^{-1}W_k)$ is
$W_k/n_k$ (the weighted sample covariance); for a diagonal $\Sigma$ the objective separates per
coordinate; for $\Sigma=\sigma^2I$ it is $n_kd\log\sigma^2+\operatorname{tr}(W_k)/\sigma^2$, minimized at
$\operatorname{tr}(W_k)/(n_kd)$; for *tied* types the same objective is summed over $k$ with a single
shared $\Sigma$, so the scatter is pooled over components and divided by the total count.

!!! note "More covariance structures: the 14 GPCM models"
    `covariance_type` also accepts all 14 Celeux & Govaert (1995) mclust codes
    (`'EII'`, `'VEI'`, `'EVE'`, `'VVV'`, ... -- the six names above are aliases for six of them).
    These decompose $\Sigma_k=\lambda_kD_kA_kD_k^\top$ into volume, shape and orientation, each
    shared across or free per component. See
    [GPCM Covariance Models](gpcm-covariance-models.md) for the decomposition, the M-step
    derivations and parameter counts; `covariance_type_` always reports the canonical code.

!!! note "Where `reg_covar` enters"
    In the plain MLE updates above, $\varepsilon$ is added to the covariance itself. In the MAP / NIW
    updates below it is added to the posterior scale matrix $\Psi_n$ *before* dividing by
    $\nu_n$, so its effective size is $\varepsilon/\nu_n$.

### Parameter counts

The total number of free parameters used by `bic` / `aic` / `icl` is
$p=(K-1)+Kd+p_\Sigma$, where $p_\Sigma$ is the last column of the table above (for the 8 extra
GPCM codes see the GPCM page). In Gibbs mode $K$ is the inferred number of *active* components.

## Covariance types at a glance

Per-component covariances (`full`, `diag`, `spherical`) let each cluster have its own size,
shape and orientation to the degree the type allows; `tied_*` types force every cluster to
share one covariance (different centers only), which saves parameters and stabilizes fits
with few points per cluster.

```python
GaussianMixture(n_components=3, n_features=2, covariance_type='full')            # arbitrary ellipsoids
GaussianMixture(n_components=3, n_features=2, covariance_type='diag')            # axis-aligned ellipsoids
GaussianMixture(n_components=3, n_features=2, covariance_type='spherical')       # one variance per cluster
GaussianMixture(n_components=3, n_features=2, covariance_type='tied_full')       # one shared full covariance
GaussianMixture(n_components=3, n_features=2, covariance_type='tied_diag')       # one shared diagonal
GaussianMixture(n_components=3, n_features=2, covariance_type='tied_spherical')  # one shared variance
GaussianMixture(n_components=3, n_features=2, covariance_type='EVE')             # any of the 14 mclust GPCM codes
```

Aliases: `'tied'` is `'tied_full'` and `'isotropic'` is `'spherical'`. The
[covariance tutorial](../notebooks/15_covariance_models.ipynb) compares all of them visually.

## Estimation methods

### Maximum Likelihood Estimation (MLE)

The default: the updates above.

```python
gmm = GaussianMixture(n_components=3, n_features=2)
gmm.fit(X)
```

### Maximum A Posteriori (MAP) with priors

Priors regularize each M-step. All priors are conjugate:

| Parameter | Prior | Constructor arguments |
| --- | --- | --- |
| weights $\pi$ | $\operatorname{Dirichlet}(\alpha_1,\dots,\alpha_K)$ | `weight_concentration_prior` $=\alpha$ |
| mean $\mu_k$ | $\mathcal{N}(\mu_{0,k},\ \Sigma_k/\kappa_0)$ | `mean_prior` $=\mu_0$, `mean_precision_prior` $=\kappa_0$ |
| covariance $\Sigma_k$ | $\operatorname{Inverse\text{-}Wishart}(\Psi_0,\nu_0)$ | `covariance_prior` $=\Psi_0$, `degrees_of_freedom_prior` $=\nu_0$ |

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    weight_concentration_prior=torch.ones(3) * 2.0,                  # Dirichlet on weights
    mean_prior=torch.zeros(3, 2),                                     # Gaussian on means
    mean_precision_prior=0.01,
    covariance_prior=torch.eye(2).unsqueeze(0).repeat(3, 1, 1),      # Inverse-Wishart on covariances
    degrees_of_freedom_prior=3.0,
)
gmm.fit(X)
```

**Weights** (independent of everything else; works with all covariance types, including all
14 GPCM codes): the MAP estimate of a Dirichlet-multinomial,

$$
\pi_k=\frac{n_k+\alpha_k-1}{\,n+\sum_j\alpha_j-K\,} .
$$

**Means only** (`mean_prior` without `covariance_prior`):

$$
\mu_k=\frac{\sum_ir_{ik}x_i+\kappa_0\mu_{0,k}}{n_k+\kappa_0}
=\frac{n_k\bar y_k+\kappa_0\mu_{0,k}}{n_k+\kappa_0},
$$

a precision-weighted average of the data mean and the prior mean (shrinkage toward $\mu_0$ that is
strong when $n_k\ll\kappa_0$). The covariance is then the ordinary MLE form, but with the scatter
taken about the *shrunk* mean $\mu_k$ rather than $\bar y_k$ (by the parallel-axis identity
$\sum_ir_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top=W_k+n_k(\bar y_k-\mu_k)(\bar y_k-\mu_k)^\top$).

**Covariance only** (`covariance_prior` without `mean_prior`), with scatter $S_k$ about $\mu_k=\bar y_k$:

$$
\Sigma_k=\frac{\Psi_0+W_k+\varepsilon I}{\nu_0+n_k}\quad(\texttt{full}),
$$

with the analogous diagonal / trace forms for the other types (`diag`: per coordinate, divide by
$\nu_0+n_k$; `spherical`: $\sigma_k^2=(\psi_0+\operatorname{tr}W_k+\varepsilon)/\big((\nu_0+n_k)d\big)$; `tied_*`: pool
$W_k$ over $k$ and use $\nu_0+\sum_kn_k$).

### Normal-Inverse-Wishart (NIW) conjugate prior

When **both** `mean_prior` and `covariance_prior` are given, tgmm uses the joint NIW prior
$\mu_k,\Sigma_k\sim\operatorname{NIW}(\mu_0,\kappa_0,\Psi_0,\nu_0)$, i.e.
$\Sigma_k\sim\operatorname{IW}(\Psi_0,\nu_0)$ and $\mu_k\mid\Sigma_k\sim\mathcal{N}(\mu_0,\Sigma_k/\kappa_0)$.
The posterior is again NIW with

$$
\kappa_n=\kappa_0+n_k,\quad
\nu_n=\nu_0+n_k,\quad
\mu_n=\frac{\kappa_0\mu_0+n_k\bar y_k}{\kappa_n},
$$

$$
\Psi_n=\Psi_0+W_k+\frac{\kappa_0\,n_k}{\kappa_0+n_k}\,(\bar y_k-\mu_0)(\bar y_k-\mu_0)^\top+\varepsilon I,
$$

and the M-step sets

$$
\mu_k=\mu_n,\qquad \Sigma_k=\frac{\Psi_n}{\nu_n}.
$$

The third term of $\Psi_n$ penalizes a data mean that disagrees with the prior mean: it is the
price of the prior's uncertainty about $\mu_k$ and vanishes as $n_k\to\infty$ or $\kappa_0\to0$.

!!! note "$\Psi_n/\nu_n$ is a deliberate choice, not the textbook posterior mean or mode"
    For an Inverse-Wishart posterior the textbook point estimates are the mean
    $\Psi_n/(\nu_n-d-1)$ and the mode $\Psi_n/(\nu_n+d+1)$. tgmm uses $\Psi_n/\nu_n$ for every
    covariance type, in both MAP and NIW updates and in Gibbs mode. The mean is undefined whenever
    $\nu_n\le d+1$, which a nearly empty component can reach, while $\Psi_n/\nu_n$ stays finite and
    positive for every $\nu_n>0$; and the three agree closely when $\nu_n\gg d$. In particular,
    with $\kappa_0\to0$ and $\Psi_0\to0$ it reduces to the MLE $W_k/n_k$.

The other covariance types use the same $\Psi_n$ in structured form:

| Type | $\Psi_n$ | $\Sigma$ |
| --- | --- | --- |
| `diag` | per coordinate: $\psi_{0,j}+(W_k)_{jj}+\frac{\kappa_0n_k}{\kappa_n}(\bar y_{kj}-\mu_{0,kj})^2+\varepsilon$ | $\Psi_n/\nu_n$ per coordinate |
| `spherical` | $\psi_0+\operatorname{tr}W_k+\frac{\kappa_0n_k}{\kappa_n}\lVert\bar y_k-\mu_{0,k}\rVert^2+d\,\varepsilon$ | $\sigma_k^2=\Psi_n/(\nu_n\,d)$ |
| `tied_full` | $\Psi_0+\sum_kW_k+\sum_k\frac{\kappa_0n_k}{\kappa_n}(\bar y_k-\mu_{0,k})(\bar y_k-\mu_{0,k})^\top+\varepsilon I$ | $\Psi_n/(\nu_0+\sum_kn_k)$ |
| `tied_diag`, `tied_spherical` | the tied analogues (sums over $k$) | $\Psi_n/(\nu_0+\sum_kn_k)$, divided by $d$ for spherical |

(For `tied_spherical` the code adds $\varepsilon\cdot n\cdot d$ rather than $\varepsilon\cdot d$ to the
numerator, so there $\varepsilon$ acts at its face value on the variance.)

!!! warning "MAP/NIW are for the six classic covariance types"
    `mean_prior` / `covariance_prior` raise `NotImplementedError` for the 8 GPCM codes without a
    tgmm name; a conjugate prior would have to be derived per parameterization.
    `weight_concentration_prior` is independent of the covariance parameterization and works with all 14.

See the [Priors tutorial](../notebooks/09_priors.ipynb) and the
[NIW comparison](../notebooks/10_niw_priors_comparison.ipynb) for worked examples.

## Initialization strategies

EM converges to a local optimum, so where it starts matters. `init_means` chooses the initial
means (the E-step then yields the first responsibilities from them), `init_weights` and
`init_covariances` the rest; all three also accept an explicit tensor.

| `init_means` | How the $K$ initial centers are chosen |
| --- | --- |
| `'kmeans'` (default) | Lloyd's algorithm (pure PyTorch), started from k-means++ |
| `'kpp'` | k-means++: pick each new center with probability $\propto D(x)^2$ (squared distance to the nearest chosen center) |
| `'random'` | $\mu=\bar x+Lz$, $z\sim\mathcal{N}(0,I)$, $LL^\top$ = the data covariance |
| `'points'` | $K$ data points chosen uniformly at random |
| `'maxdist'` | greedy farthest-point selection |

`init_weights`: `'uniform'` (default), `'random'` (Dirichlet draw), `'kmeans'` (cluster
proportions). `init_covariances`: `'empirical'` (default; covariance of the points nearest
each mean), `'eye'`, `'random'`, `'global'` (the data covariance for every component). See the
[GMM Initializer guide](gmm-initializer.md).

```python
gmm = GaussianMixture(n_components=3, n_features=2, init_means='kpp')

initial_means = torch.tensor([[0, 0], [5, 5], [0, 5]], dtype=torch.float32)
gmm = GaussianMixture(n_components=3, n_features=2, init_means=initial_means)   # custom
```

### Multiple restarts

`n_init=m` runs the whole fit $m$ times from different initializations and keeps the run with the
highest final log-likelihood. With `random_state=s`, run $i$ is seeded with $s+i$ and the winner's
seed is stored in `best_random_state_`:

```python
gmm = GaussianMixture(n_components=3, n_features=2, n_init=10, random_state=42)
gmm.fit(X)
print(gmm.best_random_state_)
```

## Fitting algorithms

`algorithm` selects the EM variant that drives `fit()`. Every variant feeds some matrix of
(soft or hard) assignments into the same M-step, which computes $n_k,\bar y_k,W_k$ from it exactly as above.

| `algorithm` | What goes into the M-step | When to use it |
| --- | --- | --- |
| `'em'` (default) | soft responsibilities $r_{ik}$ | the right default |
| `'cem'` | hard one-hot $\mathbb{1}[k=\arg\max_jr_{ij}]$ | faster, crisper partition; more local-minimum prone |
| `'sem'` | a hard label *sampled* from $r_{i\cdot}$ | escape poor optima; diagnose an overspecified $K$ |
| `'saem'` | statistics averaged over sampled sweeps with step $\gamma_t$ | SEM's exploration with a convergent stopping point |

`temperature` (deterministic annealing) and `step_size` (online / mini-batch EM) are independent
knobs layered on top.

### Classification EM (CEM)

CEM inserts a *classification* step between E and M: $\hat r_{ik}=\mathbb{1}\big[k=\arg\max_jr_{ij}\big]$.
It maximizes the classification likelihood $\sum_i\log\big(\pi_{z_i}\mathcal{N}(x_i\mid\mu_{z_i},\Sigma_{z_i})\big)$
over parameters *and* labels, rather than the mixture likelihood. It is a hard-assignment
generalization of k-means (k-means is CEM with equal weights and a shared spherical covariance, `tied_spherical`) and converges
in few iterations, but it estimates the parameters from truncated data, so it is biased when
clusters overlap.

```python
gmm = GaussianMixture(n_components=3, n_features=2, algorithm='cem')
gmm.fit(X)
```

!!! note "`cem=True` is deprecated"
    The older `cem=True` keyword still works (treated as `algorithm='cem'`) but emits a
    `DeprecationWarning`; passing it together with `algorithm='sem'`/`'saem'` raises. Use
    `algorithm='cem'`.

### Stochastic EM (SEM)

SEM (Celeux & Diebolt, 1985) replaces the soft assignment by one draw per sample,
$z_i\sim\operatorname{Categorical}(r_{i1},\dots,r_{iK})$, and runs the M-step on the resulting
one-hot matrix. The iterates form a Markov chain that keeps exploring instead of settling, so
SEM can leave poor local optima.

```python
gmm = GaussianMixture(n_components=3, n_features=2, algorithm='sem', max_iter=200)
gmm.fit(X)

print(gmm.log_likelihood_history_)   # per-sweep log-likelihood trace
print(gmm.n_empty_components_)       # > 0 re-seeds suggest n_components is too large
```

SEM has no pointwise convergence, so it always runs exactly `max_iter` sweeps (and sets
`converged_=True`). A component that receives fewer than one sampled point in a sweep has its mean re-seeded at one of the worst-fit data points (and, for
`full`/`diag`/`spherical`, its covariance reset to the data covariance); each re-seed is counted in `n_empty_components_`. The returned parameters
are computed from the sweeps at or after `sem_burn_in` (default `max(1, max_iter // 5)`):
`sem_estimate='best'` (default) keeps the snapshot with the highest log-likelihood;
`'mean'` averages the parameters across all of them, which presumes no label switching between
sweeps -- prefer `'best'` whenever $K$ may be overspecified.

### Stochastic Approximation EM (SAEM)

SAEM (Delyon, Lavielle & Moulines, 1999) averages the *sufficient statistics* of the sampled
sweeps with a decreasing step size $\gamma_t$ (Robbins-Monro):

$$
\tilde S_t=(1-\gamma_t)\,\tilde S_{t-1}+\gamma_t\,S\big(z^{(t)}\big),\qquad \tilde S_0=S\big(z^{(0)}\big),
$$

where $S=(n_k,\ \sum_ir_{ik}x_i,\ W_k)$, and the M-step then maps $\tilde S_t$ to parameters. The
default `step_size='auto'` is $\gamma_t=(t+2)^{-0.7}$ (so $\sum\gamma_t=\infty$ while
$\sum\gamma_t^2<\infty$); a positive float gives a constant step and a callable $t\mapsto\gamma_t$ a custom
schedule. Because the averaged statistics converge, the usual `tol` test applies -- SAEM is "the SEM that converges".

```python
gmm = GaussianMixture(n_components=3, n_features=2, algorithm='saem', max_iter=200)
gmm.fit(X)
print(gmm.converged_)
```

### Deterministic Annealing (DAEM)

DAEM (Ueda & Nakano, 1998) tempers the posterior with an inverse temperature $\beta\in(0,1]$:

$$
r_{ik}(\beta)=\frac{\big(\pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)\big)^{\beta}}
{\sum_j\big(\pi_j\,\mathcal{N}(x_i\mid\mu_j,\Sigma_j)\big)^{\beta}} .
$$

At $\beta\to0$ every sample is assigned almost uniformly to all components (a smooth, nearly unimodal
objective); at $\beta=1$ this is the ordinary E-step. `temperature=(beta_0, n_anneal)` starts at
$\beta_0\in(0,1)$ and raises $\beta$ geometrically to 1 over `n_anneal` iterations:

$$
\beta_t=\begin{cases}\beta_0^{\,1-t/n_{\text{anneal}}}, & t<n_{\text{anneal}}\\[2pt] 1, & t\ge n_{\text{anneal}}\end{cases}
$$

```python
gmm = GaussianMixture(n_components=3, n_features=2, temperature=(0.1, 30))
gmm.fit(X)
print(gmm.annealing_schedule_)  # beta actually used at each E-step
```

Works with `algorithm` in `{'em', 'cem', 'sem'}`. While $\beta<1$ convergence checks are suspended,
and the entries of `log_likelihood_history_` recorded then are the *tempered* normalizers, not true
log-likelihoods -- compare only the entries from $\beta=1$ on. Annealing is a heuristic, not a
guarantee: a perfectly symmetric start (e.g. identical means) stays symmetric under any $\beta$,
so some asymmetry in the initialization is needed for annealing to amplify.

### Mini-batch and online EM

With `batch_size` alone, the E-step and statistics accumulation run in row chunks. This is **exact**
(a per-row softmax and additive statistics) and only bounds peak memory at
$O(\text{batch\_size}\cdot K\cdot d)$. Adding `step_size` switches to **online EM**
(Cappe & Moulines, 2009): for each shuffled mini-batch $B$ of the epoch, compute $S_B$ from the current parameters,
rescale it to the full data size, blend it into a running estimate with step $\gamma_t$ (where $t$
counts mini-batches globally), and run the M-step from it:

$$
\tilde S\leftarrow(1-\gamma_t)\,\tilde S+\gamma_t\,\frac{n}{|B|}\,S_B ,\qquad
\theta\leftarrow\text{M-step}(\tilde S).
$$

`max_iter` counts epochs; there is no `tol` test (`converged_` is set to `True`), and
`log_likelihood_history_` has one entry per epoch. `step_size` requires `batch_size`.

```python
gmm = GaussianMixture(
    n_components=3, n_features=2,
    batch_size=256, step_size='auto',  # or a float, or a callable t -> gamma_t
    max_iter=10,                        # epochs
)
gmm.fit(X)
```

For streaming data, `partial_fit` performs one such update per call on an already-initialized model:

```python
gmm = GaussianMixture(n_components=3, n_features=2, batch_size=256, step_size='auto')
gmm.fit(first_batch)
for batch in stream_of_batches:
    gmm.partial_fit(batch)
```

`partial_fit` assumes a roughly constant batch size across calls: unlike `fit()`'s epoch loop it does
not rescale to a known total size (there is none in a true stream).

!!! note "A caveat on blended scatter"
    $W_k$ is centered on each accumulation's *own* mean $\bar y_k$. SAEM and online EM blend $W_k$ (and
    $n_k$, $\sum_ir_{ik}x_i$) linearly, which is exact for the first two but only approximate for $W_k$ when
    the means shift between accumulations: the true scatter about the blended mean also contains a
    between-accumulation term. The approximation vanishes as the iterates stabilize.

### Convergence control

```python
gmm = GaussianMixture(
    n_components=3, n_features=2,
    max_iter=200,        # maximum EM iterations
    tol=1e-4,            # convergence tolerance
    verbose=True, verbose_interval=10,
)
gmm.fit(X)
print(gmm.converged_, gmm.n_iter_, gmm.lower_bound_)
```

Let $L_t=\frac1n\ell(\theta^{(t)})$ be the average log-likelihood (`lower_bound_`).
`convergence='relative'` (default) stops when

$$
\frac{|L_t-L_{t-1}|}{|L_{t-1}|}<\texttt{tol}.
$$

`convergence='aitken'` uses Aitken's acceleration: EM's log-likelihood sequence is asymptotically
geometric, so with $a_t=\dfrac{L_t-L_{t-1}}{L_{t-1}-L_{t-2}}$ (when $a_t<1$) the limit is estimated as

$$
L_\infty=L_{t-1}+\frac{L_t-L_{t-1}}{1-a_t},
$$

and the fit stops when $L_\infty-L_t<\texttt{tol}$ (an absolute gap to the extrapolated limit). It can stop
earlier on fits that improve slowly but steadily, and falls back to the relative test until three
values exist (or when $a_t\ge1$).

```python
gmm = GaussianMixture(n_components=3, n_features=2, convergence='aitken')
```

`fit()` also accepts per-call overrides of `max_iter`, `tol`, `random_state` and `warm_start`
(continue from the previous fit's parameters) without changing the model's defaults.

## Model selection

Choosing $K$ (and the covariance structure) is a trade-off between fit and complexity. With
$p$ the parameter count above and $\bar L=\frac1n\ell(\hat\theta)$:

$$
\operatorname{AIC}=-2n\bar L+2p,\qquad
\operatorname{BIC}=-2n\bar L+p\ln n,\qquad
\operatorname{ICL}=\operatorname{BIC}+2\,\mathrm{EN}.
$$

All are "lower is better" (`gmm.aic(X)`, `gmm.bic(X)`, `gmm.icl(X)`). $\mathrm{EN}\ge0$ is the
classification-entropy penalty of Biernacki, Celeux & Govaert (2000): with `mode='hard'` (default)
$\mathrm{EN}=-\sum_i\log\hat\tau_{i}$, where $\hat\tau_i=\max_kr_{ik}$ is the posterior of the assigned
cluster; with `mode='soft'` it is the entropy $-\sum_{i,k}r_{ik}\log r_{ik}$. ICL therefore penalizes overlapping
clusters on top of BIC and favors well-separated partitions.

`select_model` sweeps a grid and returns the best fit (by `criterion='bic'`, `'aic'` or `'icl'`);
by default it tries **all 14 GPCM covariance codes**:

```python
from tgmm import select_model

result = select_model(X, n_components=range(1, 8), criterion='bic')
print(result.best_.covariance_type_, result.best_.n_components)
df = result.to_dataframe()   # every (K, covariance_type) with bic, aic, icl, log_likelihood, converged
```

See the [Model Selection tutorial](../notebooks/13_model_selection.ipynb) and
[Clustering Metrics](clustering-metrics.md) for label-free validity indices.

## Supervised and semi-supervised fitting

Passing `labels` to `fit` uses known class information; `-1` marks an unknown label.

```python
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X, labels=y)             # fully labeled
gmm.fit(X, labels=y_partial)     # some entries -1: semi-supervised EM
gmm.predict_class(X)             # class labels (via classes_), not component indices
```

**Fully labeled.** Each point's responsibility is the one-hot vector of its true class, so there is
nothing to iterate: a single M-step recovers the per-class $\pi_k=n_k/n$, $\mu_k$, $\Sigma_k$ -- exactly
what CEM would converge to with the correct partition. `n_components` must equal the number of
distinct labels (times `components_per_class` for a `components_per_class > 1`; then each class's block of
components is fit by ordinary EM on that class's rows, weights scaled by the class fraction, and `n_init`
applies within each block); label values need not be contiguous
(`classes_[k]` recovers the original value). `n_init` and `warm_start` are ignored, `converged_` is
`True` and `n_iter_` is 1. Priors still apply (a regularized fit for classes with few samples).

**Partially labeled.** Semi-supervised EM maximizes

$$
\sum_{i\in\text{labeled}}\log\sum_{k\in\text{block}(c_i)}\pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k)
+\sum_{i\in\text{unlabeled}}\log\sum_{k=1}^K\pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k),
$$

by clamping each labeled row's responsibility to its class's block of components (zero outside,
renormalized inside) and leaving unlabeled rows with the full posterior. `unlabeled_weight` $u$
multiplies the unlabeled rows' responsibilities before the M-step ($u=0$ uses the labeled rows only),
`label_confidence` $c<1$ keeps mass $c$ on the class block and gives $1-c$ to the unclamped posterior
outside it (for noisy labels), and `components_per_class` (an int or a dict) gives a class several components.
Partial labels support `algorithm='em'`/`'cem'` only. See
[Semi-Supervised and Positive-Unlabeled Fitting](semi-supervised-pu.md) for the full treatment,
including `PositiveUnlabeledMixture`, and the
[Supervised Fitting notebook](../notebooks/05_supervised_gmm.ipynb) and
[Semi-supervised notebook](../notebooks/18_semi_supervised_pu.ipynb) for worked examples.

## Gibbs sampling: inferring the number of components

If $K$ is unknown, `GaussianMixture(n_components=None, ...)` replaces EM by a collapsed Gibbs
sampler for a **Dirichlet-process mixture**, which infers the number of components from the data.

### The model

A Dirichlet process $G\sim\operatorname{DP}(\alpha,G_0)$ with a NIW base measure $G_0$ gives
mixture weights by *stick-breaking*:

$$
v_k\sim\operatorname{Beta}(1,\alpha),\qquad \pi_k=v_k\prod_{j<k}(1-v_j),
$$

an infinite sequence of weights summing to 1 -- larger $\alpha$ spreads mass over more components.
Marginalizing the weights gives the equivalent *Chinese Restaurant Process*: sample $i$ joins an existing
cluster $k$ with probability $\propto n_k^{-i}$ (the other points already there) or opens a new one with
probability $\propto\alpha$.

### The sampler

Component parameters are integrated out too (collapsed sampling, Neal's Algorithm 3). With the NIW
prior $(\mu_0,\kappa_0,\Psi_0,\nu_0)$, the posterior given the other points of cluster $k$ has the
same form as the NIW update above ($\kappa_n,\nu_n,\mu_n,\Psi_n$), and the posterior predictive of a
point is a multivariate Student-$t$:

$$
x\mid X_k^{-i}\ \sim\ t_{\nu_n-d+1}\!\Big(\mu_n,\ \frac{\kappa_n+1}{\kappa_n(\nu_n-d+1)}\,\Psi_n\Big).
$$

Each sweep resamples every assignment from

$$
p(z_i=k\mid z_{-i},X)\ \propto\ n_k^{-i}\;t_{\nu_n-d+1}\big(x_i;\ \mu_n,\ \cdots\big)
\quad(\text{existing } k),\qquad
p(z_i=\text{new}\mid z_{-i},X)\ \propto\ \alpha\;t_{\nu_0-d+1}\big(x_i;\ \mu_0,\ \cdots\big).
$$

For `diag` the predictive is a product of per-coordinate Student-$t$ with $\nu_n$ degrees of freedom; for
`spherical` a single isotropic $t$ with $\nu_0+n_kd$ degrees of freedom. `tied_*` types and the 8
extra GPCM codes are not supported (a coupling across components breaks the per-cluster factorization).

```python
from tgmm import GaussianMixture

# Gibbs mode has no default prior; suggest_priors gives a sensible start from a k-means pass.
mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
    GaussianMixture.suggest_priors(X, n_components=10)

gmm = GaussianMixture(
    n_components=None,       # switches to Gibbs sampling
    covariance_type='full',  # 'full', 'diag' or 'spherical'
    max_components=20,       # upper bound on tracked components (None = unbounded)
    alpha=1.0,               # DP concentration
    mean_prior=mean_prior,
    mean_precision_prior=mean_precision_prior,
    covariance_prior=covariance_prior,
    degrees_of_freedom_prior=degrees_of_freedom_prior,
    max_iter=50,             # an exact sweep count in Gibbs mode
)
gmm.fit(X)

print(gmm.n_components_)  # the inferred component count
print(gmm.active_)        # which of the tracked slots are in use
```

- **Truncated** (`max_components=K`, default 20): $K$ fixed slots. The CRP rule is replaced by its finite
  Dirichlet-multinomial approximation, $p(z_i=k)\propto(n_k^{-i}+\alpha/K)\times\text{predictive}_k$, which
  converges to the CRP as $K$ grows past the number of clusters actually used.
- **Unbounded** (`max_components=None`): components are created and destroyed on the fly
  (Neal's Algorithm 3 exactly). `init_k` sets the number of components the sampler is seeded with, and `burn_in`
  (default `max_iter // 5`) the number of sweeps excluded from `n_components_history_`, the per-sweep trace of the
  number of active components.
- The fitted parameters come from the **final sweep**'s partition (one posterior draw, not an average --
  averaging over sweeps is not meaningful because labels permute): $\mu_k=\mu_n$,
  $\Sigma_k=\Psi_n/\nu_n$, $\pi_k=n_k/n$. A component is *active* (counted in `n_components_` / `active_`) when its
  expected point count $\pi_kn$ exceeds `weight_threshold` (default `1.0`).
- `suggest_priors(X, n_components)` sets $\mu_0=\bar x$, $\kappa_0=0.1$, $\nu_0=d+2$ and
  $\Psi_0$ to the *pooled within-cluster* scatter of a $k$-means partition (not the marginal data covariance,
  which for well-separated clusters conflates between- and within-cluster spread and inflates every
  fitted covariance).
- There is no EM-style convergence test: `max_iter` is an *exact* sweep count and `converged_` is always
  `True` there. Each sweep is an $O(n)$ Python-level loop, so the EM default of 1000 can be very slow --
  pass an explicit, usually much smaller, `max_iter`. Supervised fitting (`labels=`) is not available.

See the [Dirichlet Process Mixture notebook](../notebooks/11_dpgmm.ipynb) for the stick-breaking and
Gibbs-sampling math, a step-by-step walkthrough and a comparison against classical EM, and the
[design document](https://github.com/adriansousapoza/tgmm/blob/main/docs/superpowers/specs/2026-07-28-dpgmm-design.md)
for the full derivation.

## Frozen components and sample weights

`frozen_components=(1, 3)` holds those components' parameters at their *initial* values for the whole fit
(give them with tensor `init_means` / `init_covariances`, or with `warm_start`); their mixing weights still update,
and MAP / NIW priors never shrink them. It is useful for anchoring known classes (a "reference" population) while the
rest of the mixture is learned, and it is what `PositiveUnlabeledMixture(anchor='fixed')` uses. It is not available
for the tied covariance types or the 8 GPCM codes without a classic name (one factor is shared across components).

`fit(X, sample_weight=w)` fits a weighted sample. With `w` of shape `(n,)` row $i$ counts $w_i$ times: every
responsibility is multiplied by $w_i$ before the statistics of the M-step, the mixing weights use
$n_\mathrm{eff}=\sum_iw_i$ as the sample count, and the convergence criterion is the weighted mean
log-likelihood. Integer weights are therefore equivalent to duplicating rows and weight $0$ to dropping the row.
For `covariance_type='diag'` or `'spherical'` (maximum likelihood only) `w` may also have shape `(n, d)`: entry
$w_{ij}$ then weights feature $j$ of row $i$ in both steps,

$$
\mu_{kj}=\frac{\sum_i r_{ik}w_{ij}x_{ij}}{\sum_i r_{ik}w_{ij}},\qquad
\sigma^2_{kj}=\frac{\sum_i r_{ik}w_{ij}(x_{ij}-\mu_{kj})^2}{\sum_i r_{ik}w_{ij}},
$$

and $w_{ij}=0$ makes that entry invisible, which is exact marginalisation of an independent feature. `inertia` (as in
pomegranate) is not provided: `step_size` (online EM / SAEM) is the same control on the sufficient statistics.

## Sampling from the model

Generate new samples from the fitted mixture: pick $z\sim\operatorname{Categorical}(\pi)$, then draw
$x=\mu_z+L_zg$ with $g\sim\mathcal{N}(0,I)$ and $L_zL_z^\top=\Sigma_z$.

```python
samples, component_ids = gmm.sample(100)                          # basic sampling
samples, _ = gmm.sample(50, component=0)                          # one component only
samples, _ = gmm.sample(100, confidence=0.95)                     # within the 95% region
samples, _ = gmm.sample(100, std_range=(1.0, 2.0))                # Mahalanobis radius in [1, 2]
outliers, _ = gmm.sample(20, std_range=(3.0, float('inf')))       # beyond 3 standard deviations
```

Constraints are applied by rejection sampling on the Mahalanobis distance
$\sqrt{(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)}$; `confidence` $q$ is converted to a radius through the
$\chi^2_d$ quantile, $\sqrt{\chi^2_{d,q}}$ (so in $d=2$, 95% is a radius of about 2.45, not 2). See the
[Sampling tutorial](../notebooks/07_sampling.ipynb).

## Large datasets: `batch_size`

```python
gmm = GaussianMixture(n_components=10, n_features=64, batch_size=4096)
gmm.fit(X)  # same result as batch_size=None, bounded peak memory
```

As explained under [online EM](#mini-batch-and-online-em), `batch_size` alone changes nothing about the
result (up to floating-point summation order) -- it is purely a memory/speed knob.

## Model persistence

```python
gmm.save('my_model.pth')
loaded_gmm = GaussianMixture.load('my_model.pth', device='cuda')

state_dict = gmm.save_state_dict()          # PyTorch-style
torch.save(state_dict, 'model_state.pth')
new_gmm = GaussianMixture(n_components=3, n_features=2)
new_gmm.load_state_dict(torch.load('model_state.pth'))
```

`classes_` / `class_of_component_` are not persisted.

## GPU acceleration

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gmm = GaussianMixture(n_components=3, n_features=2, device=device)
gmm.fit(X)  # X can be on CPU or GPU; it is moved to `device`
```

## Fitted attributes

After fitting, the model exposes:

| Attribute | Meaning |
| --- | --- |
| `weights_` | mixing weights $\pi$, shape $(K,)$ |
| `means_` | component means, shape $(K,d)$ |
| `covariances_` | covariances; shape depends on `covariance_type` (table above); always the assembled $(K,d,d)$ matrices for the 8 extra GPCM codes |
| `covariance_type_` | canonical mclust code of `covariance_type` (e.g. `'VVV'` for `'full'`) |
| `volume_`, `shape_`, `orientation_` | the GPCM decomposition $\Sigma_k=\lambda_kD_kA_kD_k^\top$ for the 8 extra codes; `None` otherwise |
| `converged_`, `n_iter_` | whether EM converged and in how many iterations (`True` / sweeps count for SEM, online EM, Gibbs) |
| `lower_bound_` | final average log-likelihood of the best run |
| `log_likelihood_history_` | per-iteration / sweep / epoch average log-likelihood (EM modes; `None` in Gibbs mode) |
| `best_random_state_` | seed of the winning run when `n_init > 1` |
| `initial_weights_`, `initial_means_`, `initial_covariances_` | parameters before the first EM iteration |
| `annealing_schedule_` | the $\beta$ used at each E-step when `temperature` is set |
| `n_empty_components_` | `algorithm='sem'`: number of component re-seeds |
| `classes_`, `class_of_component_` | label values seen by a `fit(X, labels=...)` call and each component's class; `None` otherwise |
| `n_components_`, `active_` | inferred component count and active-slot mask (Gibbs mode; `n_components_` equals `n_components` in EM mode) |
| `n_components_history_` | Gibbs mode, unbounded variant: active-component trace after `burn_in` |
| `fitted_` | whether the model has been fitted |

## Related models

`StudentTMixture` (robust to outliers) and `VonMisesFisherMixture` (directional data) share this
page's `fit` / `predict` / `bic` / algorithm / semi-supervised machinery through `BaseMixture`;
see [Other Mixture Families](other-mixture-families.md).

## Complete API reference

For all parameters and methods, see the [API Reference](../api/gaussian-mixture.md).
