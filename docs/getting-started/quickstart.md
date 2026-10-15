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

# Any of the 14 mclust/GPCM codes, e.g. shared orientation, per-cluster volume and shape
gmm_eve = GaussianMixture(n_components=3, n_features=2, covariance_type='VVE')
```

### When to Use Each Type

| Type | Use When | Advantages | Disadvantages |
|------|----------|------------|---------------|
| `full` | Clusters can have any shape/orientation | Most flexible | Most parameters, slowest |
| `diag` | Features are independent | Faster than full | Cannot capture correlations |
| `spherical` | Clusters are roughly circular | Fastest, fewest parameters | Very restrictive |
| `tied_*` | All clusters have similar shapes | Fewer parameters | Less flexible |
| GPCM codes (`'EVE'`, `'VEV'`, ...) | Something in between: e.g. same orientation but different sizes | Systematic family between `tied_full` and `full` | MLE only (no priors, no Gibbs mode) for the 8 codes without a classic name |

The number of free covariance parameters grows from 1 (`tied_spherical`) to $K\,d(d+1)/2$ (`full`); see
the [Gaussian Mixture Model guide](../user-guide/gaussian-mixture.md#m-step-maximum-likelihood-updates)
and [GPCM Covariance Models](../user-guide/gpcm-covariance-models.md) for the exact update equations and
parameter counts, and the [Covariance Structures tutorial](../notebooks/15_covariance_models.ipynb) for a visual comparison.

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

Compare models with the information criteria. BIC $=-2\ell+p\ln n$ and AIC $=-2\ell+2p$ trade the
log-likelihood $\ell$ against the number of free parameters $p$ (lower is better); ICL additionally
penalizes overlapping clusters:

```python
for k in range(2, 10):
    gmm = GaussianMixture(n_components=k, n_features=2, random_state=42)
    gmm.fit(X)
    print(f"k={k}: BIC={gmm.bic(X):.1f}  AIC={gmm.aic(X):.1f}  ICL={gmm.icl(X):.1f}")
```

`select_model` does the whole sweep over `n_components` **and** covariance structures (all 14 GPCM codes
by default) and returns the best fit:

```python
from tgmm import select_model

result = select_model(X, n_components=range(1, 8), criterion='bic', random_state=42)
print(result.best_.covariance_type_, result.best_.n_components)
result.to_dataframe().sort_values('bic').head()
```

Label-free validity indices such as the silhouette score are also available:

```python
from tgmm import ClusteringMetrics

labels = gmm.predict(X)
score = ClusteringMetrics.silhouette_score(X, labels, n_components=gmm.n_components)
```

If you do not want to choose $K$ at all, `GaussianMixture(n_components=None, ...)` infers it with a
Dirichlet-process Gibbs sampler (see the [user guide](../user-guide/gaussian-mixture.md#gibbs-sampling-inferring-the-number-of-components)).

## Other Fitting Options

```python
# Stochastic EM / SAEM / classification EM: other variants of the fitting loop
gmm = GaussianMixture(n_components=3, n_features=2, algorithm='saem')

# Deterministic annealing: smooth the likelihood early on to avoid poor local optima
gmm = GaussianMixture(n_components=3, n_features=2, temperature=(0.1, 30))

# Mini-batch / online EM for large or streaming data
gmm = GaussianMixture(n_components=3, n_features=2, batch_size=256, step_size='auto', max_iter=10)

# Partially labeled data: -1 marks unknown labels (semi-supervised EM)
gmm = GaussianMixture(n_components=3, n_features=2).fit(X, labels=y_partial)
```

## Other Mixture Families

```python
from tgmm import StudentTMixture, VonMisesFisherMixture

# Robust to outliers: heavy-tailed components
stm = StudentTMixture(n_components=3, n_features=2).fit(X)

# Directions on the unit sphere (rows of X_unit must have unit L2 norm)
vmf = VonMisesFisherMixture(n_components=5).fit(X_unit)
```

Both share `GaussianMixture`'s `fit` / `predict` / `bic` / algorithm options; see
[Other Mixture Families](../user-guide/other-mixture-families.md). For positive-unlabeled data see
`PositiveUnlabeledMixture` in [Semi-Supervised and PU Fitting](../user-guide/semi-supervised-pu.md).

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

- **[User Guide](../user-guide/gaussian-mixture.md)** - What EM computes: equations for every update, priors, algorithm variants
- **[GPCM Covariance Models](../user-guide/gpcm-covariance-models.md)** - The 14 covariance structures
- **[Tutorials](../tutorials/index.md)** - Interactive examples with visualization
- **[Bayesian Priors](../notebooks/09_priors.ipynb)** - Learn about MAP estimation
- **[API Reference](../api/gaussian-mixture.md)** - Complete documentation
