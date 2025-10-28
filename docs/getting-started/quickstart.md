# Quick Start Guide

This guide will get you up and running with tgmm in minutes.

## Basic Workflow

The typical workflow with tgmm consists of four steps:

1. **Create** a GaussianMixture instance
2. **Fit** the model to your data
3. **Predict** cluster assignments
4. **Evaluate** or visualize results

## Your First GMM

Let's fit a simple Gaussian Mixture Model:

```python
import torch
import numpy as np
from tgmm import GaussianMixture

# Generate some synthetic data
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [0, 0],     # Cluster 1
    np.random.randn(100, 2) + [5, 5],     # Cluster 2
    np.random.randn(100, 2) + [0, 5]      # Cluster 3
])
X = torch.tensor(X, dtype=torch.float32)

# Create and fit GMM
gmm = GaussianMixture(n_components=3, n_features=2)
gmm.fit(X)

# Get cluster assignments
labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)

print(f"Model converged: {gmm.converged_}")
print(f"Final log-likelihood: {gmm.lower_bound_:.2f}")
print(f"Cluster assignments: {labels[:10]}")
```

## Choosing Covariance Type

The `covariance_type` parameter controls the shape of clusters:

```python
# Full covariance - arbitrary ellipses (most flexible)
gmm_full = GaussianMixture(n_components=3, n_features=2, covariance_type='full')

# Diagonal covariance - axis-aligned ellipses
gmm_diag = GaussianMixture(n_components=3, n_features=2, covariance_type='diag')

# Spherical covariance - circles (least flexible, fastest)
gmm_sph = GaussianMixture(n_components=3, n_features=2, covariance_type='spherical')

# Tied full - all clusters share the same full covariance
gmm_tied = GaussianMixture(n_components=3, n_features=2, covariance_type='tied_full')
```

### When to Use Each Type

| Type | Use When | Advantages | Disadvantages |
|------|----------|------------|---------------|
| `full` | Clusters can have any shape/orientation | Most flexible | Most parameters, slowest |
| `diag` | Features are independent | Faster than full | Cannot capture correlations |
| `spherical` | Clusters are roughly circular | Fastest, fewest parameters | Very restrictive |
| `tied_*` | All clusters have similar shapes | Fewer parameters | Less flexible |

## Initialization Methods

Good initialization is crucial for EM convergence:

```python
# K-means initialization (recommended, default)
gmm = GaussianMixture(n_components=3, n_features=2, init_means='kmeans')

# K-means++ initialization (better for difficult cases)
gmm = GaussianMixture(n_components=3, n_features=2, init_means='kpp')

# Random initialization
gmm = GaussianMixture(n_components=3, n_features=2, init_means='random')

# Custom initialization with provided means
initial_means = torch.tensor([[0, 0], [5, 5], [0, 5]], dtype=torch.float32)
gmm = GaussianMixture(n_components=3, n_features=2, init_means=initial_means)
```

## Multiple Random Restarts

To avoid local minima, use multiple initializations:

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    n_init=10,  # Try 10 different initializations
    random_state=42  # For reproducibility
)
gmm.fit(X)

# The model will automatically select the best result
print(f"Best log-likelihood: {gmm.lower_bound_:.2f}")
```

## GPU Acceleration

Enable GPU acceleration for large datasets:

```python
# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create GMM on GPU
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    device=device
)

# Fit on GPU
gmm.fit(X)  # X will be automatically moved to GPU

# Predictions on GPU
labels = gmm.predict(X)
```

## Monitoring Convergence

Track the training process:

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    verbose=True,          # Enable progress output
    verbose_interval=5,    # Print every 5 iterations
    max_iter=200,          # Maximum iterations
    tol=1e-4              # Convergence tolerance
)
gmm.fit(X)

# Check convergence
if gmm.converged_:
    print(f"Converged in {gmm.n_iter_} iterations")
else:
    print(f"Did not converge after {gmm.max_iter} iterations")
```

## Generating Samples

Generate new samples from the fitted model:

```python
# Generate 100 new samples
new_samples, component_ids = gmm.sample(100)

print(f"Generated {new_samples.shape[0]} samples")
print(f"Sample from component: {component_ids[:10]}")

# Sample from a specific component
samples_from_comp_0, _ = gmm.sample(50, component=0)

# Sample within confidence region
samples_95, _ = gmm.sample(100, confidence=0.95)
```

## Model Selection

Compare models with different numbers of components:

```python
from tgmm import ClusteringMetrics

metrics = ClusteringMetrics()
best_k = None
best_score = -float('inf')

for k in range(2, 10):
    gmm = GaussianMixture(n_components=k, n_features=2, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    
    # Use Silhouette score for model selection
    score = metrics.silhouette_score(X, labels)
    print(f"k={k}: Silhouette={score:.3f}, BIC={gmm.lower_bound_:.2f}")
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nBest number of components: {best_k}")
```

## Saving and Loading Models

Persist trained models:

```python
# Save model
gmm.save('my_gmm_model.pth')

# Load model later
from tgmm import GaussianMixture

loaded_gmm = GaussianMixture.load('my_gmm_model.pth')

# Use loaded model
predictions = loaded_gmm.predict(X_new)
```

## Common Patterns

### Pattern 1: Fit and Predict

```python
gmm = GaussianMixture(n_components=3, n_features=2)
gmm.fit(X_train)
labels = gmm.predict(X_test)
```

### Pattern 2: Soft Clustering

```python
# Get probability of belonging to each cluster
probabilities = gmm.predict_proba(X)

# Find samples uncertain about their cluster
uncertainty = 1 - probabilities.max(dim=1)[0]
uncertain_samples = X[uncertainty > 0.4]
```

### Pattern 3: Anomaly Detection

```python
# Fit GMM on normal data
gmm.fit(X_normal)

# Compute likelihood of new samples
log_likelihood = gmm.score_samples(X_new)

# Low likelihood → potential anomaly
threshold = -10  # Tune based on your data
anomalies = X_new[log_likelihood < threshold]
```

## Next Steps

Now that you know the basics:

- **[User Guide](../user-guide/gaussian-mixture.md)** - Deep dive into GMM features
- **[Tutorials](../tutorials/index.md)** - Interactive examples with visualization
- **[API Reference](../api/gaussian-mixture.md)** - Complete documentation
- **[Bayesian Priors](../notebooks/priors.ipynb)** - Learn about MAP estimation
