# tgmm: A Gaussian Mixture Model Implementation with PyTorch

[![PyPI version](https://badge.fury.io/py/tgmm.svg)](https://badge.fury.io/py/tgmm)
[![Documentation Status](https://readthedocs.org/projects/tgmm/badge/?version=latest)](https://tgmm.readthedocs.io/en/latest/?badge=latest)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-orange.svg)](https://pytorch.org/)

**tgmm** is a flexible implementation of Gaussian Mixture Models in PyTorch, supporting:

- EM Algorithm
- MAP Estimation with Priors
- Multiple Covariance Types
- Various Initialization Methods
- Comprehensive Clustering & Evaluation Metrics

## Features

1. **GaussianMixture**  
   - Full, diag, spherical, tied covariances  
   - MLE or MAP estimation with weight, mean, or covariance priors  

2. **GMMInitializer**  
   - `kmeans`, `kpp` (k-means++), `random`, `points`, `maxdist`  

3. **ClusteringMetrics**  
   - Unsupervised metrics (Silhouette, Davies-Bouldin, etc.)  
   - Supervised metrics (ARI, NMI, Purity, Confusion Matrix, etc.)  

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

**Requirements:** Python 3.8+ and PyTorch. For GPU usage, install the CUDA-enabled version of PyTorch as per the [official instructions](https://pytorch.org/get-started/locally/).

## Documentation
We use Sphinx to build documentation. The generated HTML pages live under docs/_build/html/. You can also read the online documentation at [tgmm.readthedocs.io](https://tgmm.readthedocs.io/).

```bash
cd docs
make clean
make html
# Open _build/html/index.html in a browser
# Linux
xdg-open _build/html/index.html 
```

The docs include:

- **API Reference** for all modules (see GaussianMixture, GMMInitializer, and ClusteringMetrics)
- **Tutorials** that walk through different usage scenarios (basic GMM, metrics, using priors)
## Tutorials
We provide Jupyter notebooks in the notebooks/ folder:

- **GMM Tutorial** (`gmm.ipynb`): Basic usage of the GaussianMixture class.
- **Metrics Tutorial** (`metrics.ipynb`): Demonstrates ClusteringMetrics and how to compare models.
- **Priors Tutorial** (`priors.ipynb`): Shows how to use weight/mean/covariance priors (MAP).
- **CEM Tutorial** (`cem.ipynb`): Cross-entropy method for initialization.
- **Visualisation Tutorial** (`visualise.ipynb`): Plotting and visualization examples.

To view or run them locally, just open them in Jupyter or VS Code.

## Usage Example

Here's a comprehensive example showing the main features:

```python
import torch
import numpy as np
from tgmm import GaussianMixture

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generate sample data
np.random.seed(42)
X = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 300),
    np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], 300),
    np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 2]], 200)
])
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

# Create GMM with full parameter specification
gmm = GaussianMixture(
    n_components=3,                    # Number of mixture components
    covariance_type='full',            # 'full', 'diag', 'spherical', 'tied'
    tol=1e-6,                         # Convergence tolerance
    reg_covar=1e-6,                   # Regularization for covariance
    max_iter=100,                     # Maximum EM iterations
    init_params='kmeans',             # Initialization method
    n_init=1,                         # Number of random initializations
    random_state=42,                  # For reproducibility
    warm_start=False,                 # Use previous fit as initialization
    verbose=True,                     # Print convergence info
    verbose_interval=10,              # Print every N iterations
    device=device                     # GPU/CPU device
)

# Fit the model
gmm.fit(X_tensor)

# Make predictions
labels = gmm.predict(X_tensor)              # Hard clustering
probabilities = gmm.predict_proba(X_tensor) # Soft clustering
log_likelihood = gmm.score(X_tensor)        # Data log-likelihood
sample_scores = gmm.score_samples(X_tensor) # Per-sample log-likelihood

# Generate new samples
new_samples, sample_labels = gmm.sample(100)

# Print results
print(f"Converged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")
print(f"Log-likelihood: {log_likelihood:.4f}")
print(f"Component weights: {gmm.weights_}")
print(f"Component means shape: {gmm.means_.shape}")
print(f"Component covariances shape: {gmm.covariances_.shape}")
```

### Advanced Features

```python
# Using different initialization methods
from tgmm import GMMInitializer

# Custom initialization
initializer = GMMInitializer(
    init_method='kpp',  # 'kmeans', 'kpp', 'random', 'points', 'maxdist'
    random_state=42
)
weights_init, means_init, covariances_init = initializer.initialize(
    X_tensor, n_components=3, covariance_type='full'
)

# Use custom initialization
gmm_custom = GaussianMixture(
    n_components=3,
    weights_init=weights_init,
    means_init=means_init,
    covariances_init=covariances_init,
    device=device
)
gmm_custom.fit(X_tensor)
```

## Contributing

1. Fork the repository and create your feature branch
2. Make changes and add tests or notebooks as appropriate
3. Submit a pull request (PR) for review

We welcome improvements to both the code and the documentation.

## License
Released under the MIT License.
© 2025, Adrián A. Sousa-Poza
