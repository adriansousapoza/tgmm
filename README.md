# tgmm: A Gaussian Mixture Model based on Expectation-Maximisation implemented with PyTorch

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




## Tutorials
We provide Jupyter notebooks in the notebooks/ folder:

- **GMM Tutorial** (`gmm.ipynb`): Basic usage of the GaussianMixture class.
- **Metrics Tutorial** (`metrics.ipynb`): Demonstrates ClusteringMetrics and how to compare models.
- **Priors Tutorial** (`priors.ipynb`): Shows how to use weight/mean/covariance priors (MAP).
- **CEM Tutorial** (`cem.ipynb`): Cross-entropy method for initialization.
- **Visualisation Tutorial** (`visualise.ipynb`): Plotting and visualization examples.

To view or run them locally, just open them in Jupyter or VS Code.

## Usage Examples

### Basic Usage

Here's a simple example to get started:

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

# Create and fit GMM
gmm = GaussianMixture(n_components=3, random_state=42, device=device)
gmm.fit(X_tensor)

# Make predictions
labels = gmm.predict(X_tensor)
print(f"Converged: {gmm.converged_}, Iterations: {gmm.n_iter_}")
```

### Complete Parameter Specification

Here's a comprehensive example showing **all available parameters**:

```python
import torch
import numpy as np
from tgmm import GaussianMixture

# Set device and random seed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data
X = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 300),
    np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], 300),
    np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 2]], 200)
])
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

# Create GMM with ALL parameters specified
gmm = GaussianMixture(
    # === Core Architecture ===
    n_components=3,                              # Number of mixture components
    n_features=None,                             # Auto-inferred from data
    
    # === Covariance Configuration ===
    covariance_type='full',                      # Options: 'full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical'
    
    # === Convergence Control ===
    tol=1e-6,                                   # Convergence tolerance (relative improvement)
    max_iter=1000,                              # Maximum EM iterations
    
    # === Numerical Stability ===
    reg_covar=1e-6,                             # Regularization added to covariance diagonal
    
    # === Initialization ===
    init_params='kmeans',                        # Options: 'kmeans', 'kpp', 'random', 'points', 'maxdist'
    cov_init_method='eye',                       # Options: 'eye', 'random', 'empirical'
    weights_init=None,                           # Custom initial weights (n_components,)
    means_init=None,                             # Custom initial means (n_components, n_features)
    covariances_init=None,                       # Custom initial covariances (depends on type)
    
    # === Multiple Initialization ===
    n_init=5,                                   # Number of random initializations (best kept)
    random_state=42,                            # Random seed for reproducibility
    
    # === Training Control ===
    warm_start=False,                           # Use previous fit as initialization
    verbose=True,                               # Print training progress
    verbose_interval=10,                        # Print every N iterations
    
    # === Algorithm Variant ===
    cem=False,                                  # Use Classification EM instead of standard EM
    
    # === Bayesian Priors (MAP Estimation) ===
    weight_concentration_prior=None,             # Dirichlet prior for mixture weights
    mean_prior=None,                            # Prior means (n_components, n_features)
    mean_precision_prior=None,                  # Precision for mean prior (scalar)
    covariance_prior=None,                      # Prior for covariances (shape depends on type)
    degrees_of_freedom_prior=None,              # Degrees of freedom for Wishart prior
    
    # === Hardware ===
    device=device                               # 'cpu', 'cuda', or torch.device object
)

# Fit the model
gmm.fit(X_tensor)

# === Comprehensive Results ===
print("=== Model Convergence ===")
print(f"Converged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")
print(f"Log-likelihood: {gmm.score(X_tensor):.4f}")
print(f"Lower bound: {gmm.lower_bound_:.4f}")

print("\n=== Component Parameters ===")
print(f"Weights: {gmm.weights_}")
print(f"Means shape: {gmm.means_.shape}")
print(f"Covariances shape: {gmm.covariances_.shape}")

print("\n=== Initial vs Final ===")
print(f"Initial weights: {gmm.initial_weights_}")
print(f"Final weights: {gmm.weights_}")

# === All Prediction Methods ===
labels = gmm.predict(X_tensor)                    # Hard clustering (argmax)
probabilities = gmm.predict_proba(X_tensor)       # Soft clustering (responsibilities)
log_likelihoods = gmm.score_samples(X_tensor)     # Per-sample log-likelihood
mean_log_likelihood = gmm.score(X_tensor)         # Mean log-likelihood

# === Sample Generation ===
new_samples, sample_labels = gmm.sample(100)      # Generate 100 new samples
specific_samples, _ = gmm.sample(50, component=1)  # Sample only from component 1

print(f"\n=== Generated Samples ===")
print(f"New samples shape: {new_samples.shape}")
print(f"Sample component labels: {sample_labels[:10]}")  # First 10 labels
```

### Advanced Features

#### Custom Initialization

```python
from tgmm import GMMInitializer

# Different initialization methods
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

#### MAP Estimation with Bayesian Priors

```python
import torch

# === Example 1: Weight Priors (Dirichlet) ===
# Favor uniform weights vs. concentrated weights
weight_prior = torch.tensor([1.0, 1.0, 1.0])  # Uniform prior
# weight_prior = torch.tensor([10.0, 1.0, 1.0])  # Favor first component

gmm_weight_prior = GaussianMixture(
    n_components=3,
    weight_concentration_prior=weight_prior,
    random_state=42,
    device=device
)
gmm_weight_prior.fit(X_tensor)

# === Example 2: Mean Priors (Gaussian) ===
# Prior belief about where component means should be
mean_prior = torch.tensor([
    [0.0, 0.0],   # Prior for component 1 mean
    [2.0, 2.0],   # Prior for component 2 mean  
    [-1.0, 1.0]   # Prior for component 3 mean
])
mean_precision = 0.1  # Low precision = weak prior, high precision = strong prior

gmm_mean_prior = GaussianMixture(
    n_components=3,
    mean_prior=mean_prior,
    mean_precision_prior=mean_precision,
    random_state=42,
    device=device
)
gmm_mean_prior.fit(X_tensor)

# === Example 3: Covariance Priors (Inverse Wishart) ===
# Prior belief about covariance structure
n_features = X_tensor.shape[1]
covariance_prior = torch.eye(n_features) * 0.5  # Prior covariance matrix
degrees_of_freedom = n_features + 1  # Minimum value for valid prior

gmm_cov_prior = GaussianMixture(
    n_components=3,
    covariance_type='full',
    covariance_prior=covariance_prior,
    degrees_of_freedom_prior=degrees_of_freedom,
    random_state=42,
    device=device
)
gmm_cov_prior.fit(X_tensor)

# === Example 4: Complete MAP with All Priors ===
gmm_full_map = GaussianMixture(
    n_components=3,
    covariance_type='full',
    # Weight prior (Dirichlet)
    weight_concentration_prior=torch.tensor([2.0, 2.0, 2.0]),
    # Mean prior (Gaussian)
    mean_prior=torch.tensor([[0.0, 0.0], [3.0, 3.0], [-2.0, 2.0]]),
    mean_precision_prior=0.1,
    # Covariance prior (Inverse Wishart)
    covariance_prior=torch.eye(2) * 1.0,
    degrees_of_freedom_prior=3.0,
    random_state=42,
    device=device
)
gmm_full_map.fit(X_tensor)

print("MAP vs MLE comparison:")
print(f"MLE weights: {gmm.weights_}")
print(f"MAP weights: {gmm_full_map.weights_}")
```


#### Save and Load model

```python
# Save model
gmm.save('my_gmm_model.pth')

# Load model
from tgmm import GaussianMixture
loaded_gmm = GaussianMixture.load('my_gmm_model.pth', device=device)

# Or use state dict (PyTorch style)
state_dict = gmm.save_state_dict()
new_gmm = GaussianMixture()
new_gmm.load_state_dict(state_dict)
```

### Visualization

The package includes plotting capabilities:

```python
import matplotlib.pyplot as plt
from tgmm import plot_gmm

# === Basic Plot ===
plt.figure(figsize=(10, 8))
plot_gmm(X, gmm, 
         title='GMM Results',
         show_ellipses=True, 
         show_means=True)
plt.show()

# === Advanced Visualization Options ===
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Color by cluster predictions
plot_gmm(X, gmm,
         ax=axes[0,0],
         color_by_cluster=True,
         show_ellipses=True,
         ellipse_std_devs=[1, 2],  # Show 1 and 2 standard deviation ellipses
         title='Colored by Cluster')

# Plot 2: Show prediction probabilities
log_probs = gmm.score_samples(X_tensor)
plot_gmm(X, gmm,
         ax=axes[0,1],
         log_probs=log_probs.cpu().numpy(),
         colormap='viridis',
         show_ellipses=False,
         title='Colored by Log Probability')

# Plot 3: Compare with true labels (if available)
# Assuming you have true_labels
plot_gmm(X, gmm,
         ax=axes[1,0],
         true_labels=true_labels,  # Your ground truth labels
         color_by_cluster=True,
         match_labels_to_true=True,
         show_incorrect_predictions=True,
         title='Predictions vs Truth')

# Plot 4: Show initial vs final means
plot_gmm(X, gmm,
         ax=axes[1,1],
         show_initial_means=True,
         show_means=True,
         ellipse_std_devs=[1],
         title='Initial vs Final Means')

plt.tight_layout()
plt.show()

# === Custom Styling ===
plot_gmm(X, gmm,
         # Point styling
         point_size=8,
         point_alpha=0.7,
         
         # Ellipse styling  
         ellipse_std_devs=[1, 2, 3],
         ellipse_alpha=0.3,
         ellipse_fill=True,
         ellipse_line_style='--',
         
         # Mean markers
         mean_marker='*',
         mean_size=100,
         mean_color='red',
         
         # Scale by component weights
         scale_alpha_by_weight=True,
         scale_size_by_weight=True,
         
         title='Custom Styled GMM')
plt.show()
```


## Contributing

1. Fork the repository and create your feature branch
2. Make changes and add tests or notebooks as appropriate
3. Submit a pull request (PR) for review

We welcome improvements to both the code and the documentation.

## License
Released under the MIT License.
© 2025, Adrián A. Sousa-Poza
