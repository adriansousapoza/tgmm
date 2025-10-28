# Gaussian Mixture Model

The `GaussianMixture` class is the core component of tgmm, implementing the Expectation-Maximization (EM) algorithm for fitting Gaussian Mixture Models with optional Bayesian priors (MAP estimation).

## Overview

A Gaussian Mixture Model represents a probability distribution as a weighted sum of Gaussian components:

$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
$$

where:

- $K$ is the number of components
- $\pi_k$ are the mixing weights ($\sum_k \pi_k = 1$)
- $\mu_k$ are the component means
- $\Sigma_k$ are the component covariances

## Basic Usage

```python
from tgmm import GaussianMixture
import torch

# Create GMM
gmm = GaussianMixture(
    n_components=3,      # Number of Gaussian components
    n_features=2,        # Data dimensionality
    covariance_type='full'  # Type of covariance matrix
)

# Fit to data
gmm.fit(X)

# Make predictions
labels = gmm.predict(X)              # Hard assignments
probabilities = gmm.predict_proba(X)  # Soft assignments
log_likelihood = gmm.score(X)         # Average log-likelihood
```

## Covariance Types

tgmm supports six different covariance types, offering trade-offs between flexibility and computational cost:

### Component-Specific Covariances

Each component has its own covariance matrix:

#### Full Covariance

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='full')
```

- **Shape**: $(K, d, d)$ where $K$ is number of components, $d$ is dimensionality
- **Parameters**: $K \cdot d \cdot (d+1)/2$ (due to symmetry)
- **Use**: Most flexible; clusters can have arbitrary orientation and shape
- **Computational Cost**: Highest

#### Diagonal Covariance

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='diag')
```

- **Shape**: $(K, d)$
- **Parameters**: $K \cdot d$
- **Use**: Axis-aligned ellipsoids; assumes feature independence
- **Computational Cost**: Medium

#### Spherical Covariance

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='spherical')
```

- **Shape**: $(K,)$ (single variance per component)
- **Parameters**: $K$
- **Use**: Spherical/circular clusters; all features have same variance
- **Computational Cost**: Lowest

### Tied Covariances

All components share the same covariance structure:

#### Tied Full

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='tied_full')
```

- **Shape**: $(d, d)$
- **Parameters**: $d \cdot (d+1)/2$
- **Use**: All clusters have same shape/orientation, different centers

#### Tied Diagonal

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='tied_diag')
```

- **Shape**: $(d,)$
- **Parameters**: $d$
- **Use**: All clusters are axis-aligned ellipsoids with same shape

#### Tied Spherical

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='tied_spherical')
```

- **Shape**: Scalar
- **Parameters**: 1
- **Use**: All clusters are spheres with same radius

## Estimation Methods

### Maximum Likelihood Estimation (MLE)

Standard EM algorithm without priors:

```python
gmm = GaussianMixture(n_components=3, n_features=2)
gmm.fit(X)
```

### Maximum A Posteriori (MAP) with Priors

Bayesian approach with conjugate priors:

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    # Dirichlet prior on weights
    weight_concentration_prior=torch.ones(3) * 2.0,
    # Gaussian prior on means
    mean_prior=torch.zeros(3, 2),
    mean_precision_prior=0.01,
    # Inverse-Wishart prior on covariances
    covariance_prior=torch.eye(2).unsqueeze(0).repeat(3, 1, 1),
    degrees_of_freedom_prior=3.0
)
gmm.fit(X)
```

See the [Priors Tutorial](../notebooks/priors.ipynb) for details.

### Normal-Inverse-Wishart (NIW) Conjugate Prior

When both mean and covariance priors are specified, tgmm uses the NIW conjugate prior:

```python
# NIW automatically used when both priors are set
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    mean_prior=torch.zeros(3, 2),
    mean_precision_prior=0.1,
    covariance_prior=torch.eye(2).unsqueeze(0).repeat(3, 1, 1),
    degrees_of_freedom_prior=5.0
)
```

## Initialization Strategies

Good initialization is critical for EM convergence:

### K-means Initialization (Recommended)

```python
gmm = GaussianMixture(n_components=3, n_features=2, init_means='kmeans')
```

Uses scikit-learn's K-means to initialize component means.

### K-means++ Initialization

```python
gmm = GaussianMixture(n_components=3, n_features=2, init_means='kpp')
```

More sophisticated initialization that spreads initial centers apart.

### Random Initialization

```python
gmm = GaussianMixture(n_components=3, n_features=2, init_means='random')
```

Randomly selects data points as initial means.

### Custom Initialization

```python
initial_means = torch.tensor([[0, 0], [5, 5], [0, 5]], dtype=torch.float32)
gmm = GaussianMixture(n_components=3, n_features=2, init_means=initial_means)
```

### Multiple Random Restarts

To avoid local minima:

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    n_init=10,  # Run EM 10 times with different initializations
    random_state=42
)
```

## Classification EM (CEM)

Standard EM uses soft assignments. CEM uses hard assignments:

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    cem=True  # Enable Classification EM
)
gmm.fit(X)
```

CEM tends to converge faster but may be more prone to local minima.

## Convergence Control

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    max_iter=200,     # Maximum EM iterations
    tol=1e-4,         # Convergence tolerance on log-likelihood change
    verbose=True,     # Print progress
    verbose_interval=10  # Print every 10 iterations
)
gmm.fit(X)

# Check convergence
print(f"Converged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")
print(f"Final log-likelihood: {gmm.lower_bound_:.2f}")
```

## Sampling from the Model

Generate new samples from the fitted distribution:

```python
# Basic sampling
samples, component_ids = gmm.sample(100)

# Sample from specific component
samples, _ = gmm.sample(50, component=0)

# Sample within confidence region
samples, _ = gmm.sample(100, confidence=0.95)

# Sample within standard deviation range
samples, _ = gmm.sample(100, std_range=(1.0, 2.0))

# Sample outliers (beyond 3 std)
outliers, _ = gmm.sample(20, std_range=(3.0, float('inf')))
```

## Model Persistence

Save and load trained models:

```python
# Save model
gmm.save('my_model.pth')

# Load model
loaded_gmm = GaussianMixture.load('my_model.pth', device='cuda')

# Or use state dict (PyTorch-style)
state_dict = gmm.save_state_dict()
torch.save(state_dict, 'model_state.pth')

# Load state dict
new_gmm = GaussianMixture(n_components=3, n_features=2)
new_gmm.load_state_dict(torch.load('model_state.pth'))
```

## GPU Acceleration

```python
# Automatic device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    device=device
)

# Data is automatically moved to the correct device
gmm.fit(X)  # X can be on CPU or GPU
```

## Attributes

After fitting, the model exposes:

- `gmm.weights_`: Mixing weights, shape $(K,)$
- `gmm.means_`: Component means, shape $(K, d)$
- `gmm.covariances_`: Component covariances (shape depends on type)
- `gmm.converged_`: Whether EM converged
- `gmm.n_iter_`: Number of EM iterations performed
- `gmm.lower_bound_`: Final average log-likelihood
- `gmm.fitted_`: Whether model has been fitted

## Complete API Reference

For full details on all parameters and methods, see the [API Reference](../api/gaussian-mixture.md).
