<div align="center" markdown="1">

<img src="assets/tgmm-logo.png" alt="TorchGMM Logo" width="300"/>

</div>

# Gaussian Mixture Models in PyTorch

**tgmm** is a flexible, GPU-accelerated implementation of Gaussian Mixture Models (GMM) in PyTorch, featuring:

- **EM & MAP Estimation** - Maximum Likelihood and Bayesian approaches
- **Multiple Covariance Types** - Full, diagonal, spherical, and tied variants
- **GPU Acceleration** - Seamless CPU/CUDA support via PyTorch
- **Comprehensive Metrics** - Supervised and unsupervised clustering evaluation
- **Rich Visualization** - Beautiful plotting utilities for GMM analysis
- **Flexible Initialization** - K-means, K-means++, random, and custom methods

## Quick Start

```python
import torch
from tgmm import GaussianMixture

# Fit a GMM with 3 components (showing all default parameters)
gmm = GaussianMixture(
    # Core model parameters
    n_components=3,
    n_features=2,
    covariance_type='full',
    
    # Convergence and training parameters
    max_iter=1000,
    tol=1e-4,
    reg_covar=1e-6,
    n_init=1,
    
    # Initialization parameters
    init_means='kmeans',
    init_weights='uniform',
    init_covariances='empirical',
    
    # Random state and restart options
    random_state=None,
    warm_start=False,
    
    # Algorithm options
    cem=False,
    
    # Prior parameters for MAP estimation
    weight_concentration_prior=None,
    mean_prior=None,
    mean_precision_prior=None,
    covariance_prior=None,
    degrees_of_freedom_prior=None,
    
    # Output and device options
    verbose=False,
    verbose_interval=10,
    device=None
)

gmm.fit(X)

# Make predictions
labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)

# Score samples
log_likelihood_per_sample = gmm.score_samples(X)
average_log_likelihood = gmm.score(X)

# Generate new samples (showing all default parameters)
samples, component_ids = gmm.sample(
    n_samples=1,
    component=None,
    std_radius=None,
    std_range=None,
    confidence=None,
    confidence_range=None,
    center_point=None,
    center_radius=None,
    max_attempts_per_sample=1000
)

# Save and load models
gmm.save('my_gmm_model.pth')
loaded_gmm = GaussianMixture.load('my_gmm_model.pth', device=None)
```

## Key Features

### 1. Gaussian Mixture Model

The core `GaussianMixture` class supports:

- **Covariance Types**: `'full'`, `'diag'`, `'spherical'`, `'tied_full'`, `'tied_diag'`, `'tied_spherical'`
- **Estimation Methods**: MLE (Maximum Likelihood) or MAP (Maximum A Posteriori) with priors
- **Algorithms**: Standard EM or CEM (Classification EM) for hard assignments
- **Initialization**: Multiple strategies via `GMMInitializer`

### 2. Bayesian Inference with Priors

Support for conjugate priors enables proper Bayesian inference:

- **Weight Prior**: Dirichlet distribution
- **Mean Prior**: Gaussian distribution  
- **Covariance Prior**: Wishart/Inverse-Wishart distribution
- **NIW Conjugate Prior**: Normal-Inverse-Wishart for joint mean-covariance updates

### 3. Clustering Metrics

Comprehensive evaluation with `ClusteringMetrics`:

**Unsupervised Metrics** (no ground truth needed):
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- BIC, AIC

**Supervised Metrics** (with ground truth labels):
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Purity, Accuracy
- Confusion Matrix
- F1 Score

### 4. Visualization Tools

Beautiful plotting utilities in `tgmm.plotting`:

- Component ellipses and contours
- Sample scatter plots with cluster coloring
- PCA projections for high-dimensional data
- Responsibility heatmaps

## Installation

=== "PyPI (Stable)"

    ```bash
    pip install tgmm
    ```

=== "Development (Latest)"

    ```bash
    git clone https://github.com/adriansousapoza/TorchGMM.git
    cd TorchGMM
    pip install -e .
    ```

**Requirements**: Python 3.8+ and PyTorch 1.0+

For GPU support, install CUDA-enabled PyTorch following the [official instructions](https://pytorch.org/get-started/locally/).

## Documentation Structure

- **[Getting Started](getting-started/installation.md)** - Installation and quick start guide
- **[User Guide](user-guide/gaussian-mixture.md)** - Detailed explanations of each component
- **Check out the [Tutorials](tutorials/index.md) to see TorchGMM in action!** - Interactive Jupyter notebooks
- **[API Reference](api/gaussian-mixture.md)** - Complete API documentation

## Example: Fitting a GMM

```python
import torch
import numpy as np
from tgmm import GaussianMixture
import matplotlib.pyplot as plt

# Generate synthetic data with 3 clusters
np.random.seed(42)
X = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 300),
    np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], 300),
    np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 2]], 200)
])
X = torch.tensor(X, dtype=torch.float32)

# Fit GMM
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    covariance_type='full',
    init_means='kmeans',
    random_state=42
)
gmm.fit(X)

# Predict clusters
labels = gmm.predict(X)
print(f"Converged: {gmm.converged_}")
print(f"Log-likelihood: {gmm.lower_bound_:.2f}")
```

## Example: Using Priors (MAP Estimation)

```python
from tgmm import GaussianMixture

# Define priors
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    covariance_type='full',
    # Dirichlet prior on weights (encourages balanced clusters)
    weight_concentration_prior=torch.ones(3) * 2.0,
    # Gaussian prior on means (weak regularization toward origin)
    mean_prior=torch.zeros(3, 2),
    mean_precision_prior=0.01,
    # Inverse-Wishart prior on covariances
    covariance_prior=torch.eye(2).unsqueeze(0).repeat(3, 1, 1),
    degrees_of_freedom_prior=3.0
)

gmm.fit(X)
```

## Example: Clustering Metrics

```python
from tgmm import ClusteringMetrics

# Assuming you have true labels
metrics = ClusteringMetrics()

# Unsupervised metrics
silhouette = metrics.silhouette_score(X, labels)
davies_bouldin = metrics.davies_bouldin_score(X, labels)

# Supervised metrics (with ground truth)
ari = metrics.adjusted_rand_index(true_labels, labels)
nmi = metrics.normalized_mutual_info(true_labels, labels)
purity = metrics.purity(true_labels, labels)

print(f"Silhouette: {silhouette:.3f}")
print(f"ARI: {ari:.3f}")
print(f"NMI: {nmi:.3f}")
```

## Citation

If you use tgmm in your research, please cite:

```bibtex
@software{tgmm2025,
  title = {tgmm: Gaussian Mixture Models in PyTorch},
  author = {Sousa-Poza, Adrián A.},
  year = {2025},
  url = {https://github.com/adriansousapoza/TorchGMM}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please see [Contributing Guide](contributing.md) for details.
