# tgmm: Gaussian Mixture Models in PyTorch

[![PyPI version](https://badge.fury.io/py/tgmm.svg)](https://badge.fury.io/py/tgmm)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://adriansousapoza.github.io/tgmm/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-orange.svg)](https://pytorch.org/)

<div align="center">

<img src="docs/assets/tgmm-logo.png" alt="tgmm Logo" width="300"/>

</div>

**tgmm** is a flexible, GPU-accelerated implementation of Gaussian Mixture Models (GMM) in
PyTorch, supporting EM and MAP estimation, multiple covariance types, several initialization
strategies, and a comprehensive suite of clustering metrics and visualization tools.

## Installation

Install the latest stable version from PyPI:

```bash
pip install tgmm
```

For development installation:

```bash
git clone https://github.com/adriansousapoza/TorchGMM.git
cd TorchGMM
pip install -e .
```

**Requirements:** Python 3.8+ and PyTorch 2.5+. For GPU usage, install the CUDA-enabled version of PyTorch as per the [official instructions](https://pytorch.org/get-started/locally/).

## Documentation

Full documentation is hosted at **[adriansousapoza.github.io/tgmm](https://adriansousapoza.github.io/tgmm/)**:

- **[Getting Started](https://adriansousapoza.github.io/tgmm/getting-started/installation/)** - Installation and quick start guide
- **[User Guide](https://adriansousapoza.github.io/tgmm/user-guide/gaussian-mixture/)** - Detailed explanations of each component
- **[Tutorials](https://adriansousapoza.github.io/tgmm/tutorials/)** - Interactive Jupyter notebooks covering every feature
- **[API Reference](https://adriansousapoza.github.io/tgmm/api/gaussian-mixture/)** - Complete API documentation

## Quick Start

```python
import torch
import numpy as np
from tgmm import GaussianMixture

# Generate sample data
np.random.seed(42)
X = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 300),
    np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], 300),
    np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 2]], 200)
])
X_tensor = torch.tensor(X, dtype=torch.float32)

# Create and fit GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_tensor)

# Make predictions
labels = gmm.predict(X_tensor)
print(f"Converged: {gmm.converged_}, Iterations: {gmm.n_iter_}")
```

See the [Tutorials](https://adriansousapoza.github.io/tgmm/tutorials/) for runnable, end-to-end
examples of every feature -- covariance types, all initialization strategies, MAP estimation
with priors, Classification EM, constrained sampling, save/load, and more.

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
- Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Dunn Index, BIC, AIC

**Supervised Metrics** (with ground truth labels):
- Adjusted Rand Index (ARI), Normalized/Adjusted Mutual Information, Purity
- Confusion Matrix and per-class Classification Report (precision, recall, F1, ROC-AUC)

### 4. Visualization Tools

Flexible plotting utilities in `tgmm.plotting`:

- Component ellipses at multiple confidence levels
- Cluster coloring, ground-truth comparison, and log-likelihood coloring
- Initial vs. final mean trajectories and weight-scaled markers

## Citation

TBA

## License

Released under the MIT License.
© 2025, Adrián A. Sousa-Poza

## Contributing

tgmm isn't yet set up to accept external contributions. If you're interested in contributing,
please reach out to [asp@di.ku.dk](mailto:asp@di.ku.dk).
