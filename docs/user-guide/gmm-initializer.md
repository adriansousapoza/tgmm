# GMM Initializer

The `GMMInitializer` class provides various strategies for initializing Gaussian Mixture Models. Good initialization is crucial for the EM algorithm to converge to a good solution.

## Overview

Available initialization methods:

- **kmeans**: Use K-means clustering (recommended)
- **kpp**: K-means++ initialization
- **random**: Random selection of data points
- **points**: Use specific provided points
- **maxdist**: Maximize distance between initial centers

## Usage

The initializer is used internally by `GaussianMixture`, but can also be used standalone:

```python
from tgmm import GMMInitializer
import torch

# Create initializer
initializer = GMMInitializer(
    n_components=3,
    n_features=2,
    device='cuda'
)

# Initialize means using K-means
means = initializer.initialize_means(X, method='kmeans')

# Initialize weights (uniform by default)
weights = initializer.initialize_weights(method='uniform')

# Initialize covariances
covariances = initializer.initialize_covariances(
    X, means, method='data', covariance_type='full'
)
```

## Initialization Methods

### K-means Initialization

Uses scikit-learn's K-means algorithm:

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_means='kmeans',
    random_state=42
)
```

**Advantages**: Fast, usually provides good starting points  
**Disadvantages**: Requires scikit-learn

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

Randomly selects K data points:

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

Weights can be initialized as:

- **uniform**: All components have equal weight (default)
- **data**: Based on proximity of data to initial means

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_weights='uniform'  # or 'data'
)
```

## Covariance Initialization

Covariances can be initialized as:

- **data**: Estimated from data assigned to each component
- **identity**: Identity matrix
- **custom**: Provide specific covariance matrices

```python
# Data-based initialization
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_covariances='data'
)

# Identity initialization
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_covariances='identity'
)

# Custom covariances
init_covs = torch.eye(2).unsqueeze(0).repeat(3, 1, 1) * 0.5
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    init_covariances=init_covs
)
```

## Complete API Reference

For full details, see the [API Reference](../api/gmm-initializer.md).
