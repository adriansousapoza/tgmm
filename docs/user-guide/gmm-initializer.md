# GMM Initializer

The `GMMInitializer` class provides various strategies for initializing Gaussian Mixture Models. Good initialization is crucial for the EM algorithm to converge to a good solution.

`GMMInitializer` is a collection of `@staticmethod`s -- there is no instance to construct. Each
method takes the data (and sometimes the current means) directly and returns a tensor.

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

`GMMInitializer` is used internally by `GaussianMixture`, but its static methods can also be
called directly to build custom initial parameters:

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
(no scikit-learn dependency):

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

Smart initialization that spreads centers apart:

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

Greedily select centers to maximize minimum distance:

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_means='maxdist'
)
```

## Weight Initialization

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

## Complete API Reference

For full details, see the [API Reference](../api/gmm-initializer.md).
