# GMM Initializer

The `GMMInitializer` class provides various strategies for initializing Gaussian Mixture Models. Good initialization is crucial for the EM algorithm to converge to a good solution.

`GMMInitializer` is a collection of `@staticmethod`s -- there is no instance to construct. Each
method takes the data (and sometimes the current means) directly and returns a tensor.

## Why initialization matters

The mixture log-likelihood $\ell(\theta) = \sum_i \log \sum_k \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$
is non-convex with many local maxima (and singular maxima, where a
component collapses onto one point and $\ell \to \infty$ -- the reason for
`reg_covar`). EM only climbs monotonically from wherever it starts
($\ell$ never decreases from one iteration to the next), so it ends at a
*local* optimum determined by the starting $(\pi_k, \mu_k, \Sigma_k)$.
The initial means matter most; the initial weights and covariances matter
mainly because the first E-step computes responsibilities
$r_{ik} \propto \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$ from them.
Running `n_init > 1` independent initializations and keeping the best
$\ell$ is the standard guard.

## Overview

Available initialization methods:

**Means** (`init_means`):

- **kmeans**: Lloyd's algorithm, started from a k-means++ initialization (recommended)
- **kpp**: K-means++ initialization only (no Lloyd iterations)
- **random**: Sample from a Gaussian fit to the data's mean/covariance
- **points**: Randomly selected data points
- **maxdist**: Greedily maximize the minimum distance between centers

**Weights** (`init_weights`):

- **uniform** / **equal**: All components start with equal weight (default)
- **random**: Sample from a symmetric Dirichlet distribution
- **kmeans**: Proportional to cluster sizes from a nearest-mean assignment

**Covariances** (`init_covariances`):

- **empirical**: Estimated from data assigned to each component by nearest mean (default)
- **eye**: Identity-like matrices/vectors, scaled by `1 + reg_covar`
- **random**: Random positive semi-definite matrices/vectors
- **global**: The single global covariance of the whole dataset, shared or repeated across components

All three also accept a `torch.Tensor` of explicit values instead of a method name.

## Usage

`GMMInitializer` is used internally by `GaussianMixture` (and by
`StudentTMixture` for the same method names), but its static methods can
also be called directly to build custom initial parameters:

```python
from tgmm import GMMInitializer
import torch

# Initialize means using K-means (or .kpp, .random, .points, .maxdist)
means = GMMInitializer.kmeans(X, k=3)

# Initialize weights proportional to cluster sizes
weights = GMMInitializer.init_weights_from_clusters(X, means)

# Initialize covariances from the data assigned to each component
covariances = GMMInitializer.init_covariances_empirical(
    X, means, covariance_type='full', reg_covar=1e-6
)
```

## Mean Initialization Methods

### K-means Initialization

A pure PyTorch implementation of Lloyd's algorithm, started from a k-means++ initialization
(no scikit-learn dependency). Alternate

$$
c_j = \arg\min_{i} \lVert x_j - \mu_i \rVert^2
\qquad\text{(assignment)},
\qquad
\mu_i = \frac{1}{|C_i|}\sum_{x_j \in C_i} x_j
\qquad\text{(update)}
$$

until the centers stop moving (`torch.allclose` with `atol=1e-4`) or
`max_iter=1000` is reached. Each Lloyd step does not increase the k-means
objective $\sum_j \lVert x_j - \mu_{c_j}\rVert^2$, so it converges to a
local minimum of it.

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_means='kmeans',
    random_state=42
)
```

**Advantages**: Usually provides good starting points
**Disadvantages**: Slower than a single k-means++ pass, no convergence guarantee within `max_iter`

### K-means++ Initialization

Smart initialization that spreads centers apart. The first center is a
uniformly random data point; each next center $x_j$ is drawn with probability
proportional to its squared distance to the nearest center already chosen:

$$
P(x_j) = \frac{D(x_j)}{\sum_{j'=1}^{N} D(x_{j'})}, \qquad
D(x_j) = \min_{l < i} \lVert x_j - \mu_l \rVert^2 .
$$

(This is $D^2$-sampling; Arthur & Vassilvitskii (2007) show the resulting
k-means objective is within $O(\log K)$ of optimal in expectation.)

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_means='kpp',
    random_state=42
)
```

**Advantages**: Better than random, avoids clustering centers too close
**Disadvantages**: Slightly slower than random

### Random Initialization

Draws centers from a Gaussian fit to the data's empirical mean and covariance:

$$
\mu_i = \bar x + L z, \qquad z \sim \mathcal{N}(0, I_d), \qquad \hat\Sigma = L L^\top,
$$

with $L$ the Cholesky factor of the sample covariance $\hat\Sigma$ (so the
data covariance must be positive definite -- constant or duplicated
features fail here). Centers follow the *global* data distribution but are
not data points.

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_means='random',
    random_state=42
)
```

**Advantages**: Fast, simple
**Disadvantages**: May lead to poor convergence

### Random Data Points

`init_means='points'` picks $K$ distinct data points uniformly at random
(sampling without replacement):

```python
gmm = GaussianMixture(n_components=3, n_features=2, init_means='points')
```

### Custom Points

Provide specific initial centers:

```python
initial_means = torch.tensor([
    [0.0, 0.0],
    [5.0, 5.0],
    [0.0, 5.0]
], dtype=torch.float32)

gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_means=initial_means
)
```

### Maximum Distance Initialization

Greedily select centers to maximize the minimum distance to those already
chosen (a deterministic, farthest-first traversal after a random first
pick):

$$
\mu_i = \arg\max_{x \in \mathcal{D}} \min_{l = 1,\dots,i-1} \lVert x - \mu_l \rVert,
\qquad
\mu_1 \leftarrow \arg\max_{x \in \mathcal{D}} \min_{l = 2,\dots,K} \lVert x - \mu_l \rVert,
$$

where the second formula re-selects the first center once the others are
known (so the arbitrary first pick doesn't bias the result). It spreads
centers maximally, but is sensitive to outliers, which are the farthest
points.

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_means='maxdist'
)
```

## Weight Initialization

| `init_weights` | Initial $\pi_k$ |
| --- | --- |
| `'uniform'` / `'equal'` | $1/K$ for every component |
| `'random'` | one draw from $\mathrm{Dirichlet}(1, \dots, 1)$ (uniform over the simplex) |
| `'kmeans'` | $n_k/n$, where $n_k$ counts points whose nearest initial mean is $\mu_k$ (empty components get a tiny floor) |
| tensor | the given values, renormalized to sum to 1 |

```python
# Equal weight for every component (default)
gmm = GaussianMixture(n_components=3, n_features=2, init_weights='uniform')

# Random weights sampled from a symmetric Dirichlet distribution
gmm = GaussianMixture(n_components=3, n_features=2, init_weights='random')

# Proportional to cluster sizes (nearest-mean assignment)
gmm = GaussianMixture(n_components=3, n_features=2, init_weights='kmeans')

# Explicit weights (normalized to sum to 1)
gmm = GaussianMixture(n_components=3, n_features=2, init_weights=torch.tensor([0.5, 0.3, 0.2]))
```

## Covariance Initialization

Written for the `full` type (the diagonal, spherical and tied types use the
corresponding diagonal / mean-of-diagonal / pooled forms):

| `init_covariances` | Initial $\Sigma_k$ |
| --- | --- |
| `'empirical'` (default) | assign every point to its nearest initial mean to get clusters $C_k$; $\Sigma_k = \widehat{\mathrm{Cov}}(C_k) + \texttt{reg\_covar}\,I$, with the *unbiased* ($n_k - 1$) sample covariance; an empty cluster gets $(1 + \texttt{reg\_covar})\,I$ |
| `'global'` | the whole dataset's sample covariance $+\ \texttt{reg\_covar}\,I$, the same for every component |
| `'eye'` | $(1 + \texttt{reg\_covar})\,I$ |
| `'random'` | $A A^\top + \texttt{reg\_covar}\,I$ with $A_{ab} \sim \mathcal{N}(0,1)$ (diagonal types: uniform$(0,1)$ entries; spherical: uniform$(0,1) + 1$) |
| tensor | the given values (shape per `covariance_type`) |

For the **tied** types, `'empirical'` pools the *within-cluster* scatter
around each cluster's own mean and divides by $n$ (the biased/maximum-
likelihood normalization):
$\Sigma = \frac{1}{n}\sum_k \sum_{x\in C_k}(x - \bar x_{C_k})(x - \bar x_{C_k})^\top + \texttt{reg\_covar}\,I$.
For `diag` the diagonal of the corresponding matrix is kept; for
`spherical` the mean of that diagonal. With the 8 non-classical GPCM codes
(see [GPCM Covariance Models](gpcm-covariance-models.md)), `'empirical'` and
`'global'` are projected onto the requested parameterization, and `'eye'`/
`'random'` just start from plain identity/random full matrices.

```python
# Estimated from data assigned to each component (default)
gmm = GaussianMixture(n_components=3, n_features=2, init_covariances='empirical')

# Identity-like matrices, scaled by (1 + reg_covar)
gmm = GaussianMixture(n_components=3, n_features=2, init_covariances='eye')

# Random positive semi-definite matrices
gmm = GaussianMixture(n_components=3, n_features=2, init_covariances='random')

# Shared global covariance of the whole dataset
gmm = GaussianMixture(n_components=3, n_features=2, init_covariances='global')

# Explicit covariances
init_covs = torch.eye(2).unsqueeze(0).repeat(3, 1, 1) * 0.5
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='full', init_covariances=init_covs)
```

## Other families

`StudentTMixture` uses the same `init_means` / `init_weights` /
`init_covariances` options (its `init_covariances` has no `'random'`
option; the covariance starts as the *scale matrix*, and `dof_init` sets
the initial degrees of freedom). `VonMisesFisherMixture` instead
initializes mean *directions* by spherical k-means and sets every
$\kappa_k$ to `kappa_init`; see [Other Mixture
Families](other-mixture-families.md).

## References

- Arthur, D. & Vassilvitskii, S. (2007). k-means++: the advantages of
  careful seeding. *SODA*.
- Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Trans.
  Information Theory*, 28(2), 129-137.

## Complete API Reference

For full details, see the [API Reference](../api/gmm-initializer.md).
