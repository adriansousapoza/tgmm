# tgmm: Gaussian Mixture Models in PyTorch

[![PyPI version](https://badge.fury.io/py/tgmm.svg)](https://badge.fury.io/py/tgmm)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://adriansousapoza.github.io/tgmm/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-orange.svg)](https://pytorch.org/)

<div align="center">

<img src="docs/assets/tgmm-logo.png" alt="tgmm Logo" width="300"/>

</div>

**tgmm** is a flexible, GPU-accelerated implementation of Gaussian Mixture Models (GMM) in PyTorch.
It supports EM, MAP and Gibbs-sampling (Dirichlet-process) estimation, 14 covariance structures,
several EM variants, fixed values and bounds on the parameters, semi-supervised and
positive-unlabeled learning, a uniform noise component, missing-value EM, marginal/conditional
distributions, several initialisation strategies (k-means, k-means++, HDBSCAN, ...), model selection,
and a comprehensive suite of clustering metrics and visualization tools.

A mixture model writes the density of the data as a weighted sum of components,
p(x) = Σₖ πₖ fₖ(x | θₖ); EM alternates computing each component's responsibility for every point
(E-step) with re-estimating the parameters from those soft assignments (M-step). The
[Introduction tutorial](https://adriansousapoza.github.io/tgmm/notebooks/01_introduction/)
derives every update tgmm implements.

## Installation

Install the latest stable version from PyPI:

```bash
pip install tgmm
```

For development installation:

```bash
git clone https://github.com/adriansousapoza/tgmm.git
cd tgmm
pip install -e .
```

**Requirements:** Python 3.9+ and PyTorch 2.5+. For GPU usage, install the CUDA-enabled version of PyTorch as per the [official instructions](https://pytorch.org/get-started/locally/).

## Documentation

Full documentation is hosted at **[adriansousapoza.github.io/tgmm](https://adriansousapoza.github.io/tgmm/)**:

- **[Getting Started](https://adriansousapoza.github.io/tgmm/getting-started/installation/)** - Installation and quick start guide
- **[Tutorials](https://adriansousapoza.github.io/tgmm/tutorials/)** - marimo notebooks covering every feature, with the mathematics; each has an interactive version (molab)
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
examples of every feature -- covariance structures, initialization strategies, MAP estimation
with priors, EM variants, model selection, semi-supervised fitting, constrained sampling,
save/load, and more.

## Key Features

### 1. Gaussian Mixture Model

The core `GaussianMixture` class supports:

- **Covariance Types**: `'full'`, `'diag'`, `'spherical'`, `'tied_full'`, `'tied_diag'`, `'tied_spherical'`,
  or any of the 14 Celeux & Govaert GPCM models via their mclust codes (`'VEI'`, `'EVI'`, `'VEE'`,
  `'EVE'`, `'EEV'`, `'VVE'`, `'VEV'`, `'EVV'`, plus mclust aliases for the 6 above)
- **Estimation Methods**: MLE (Maximum Likelihood) or MAP (Maximum A Posteriori) with priors
- **Algorithms**: `algorithm='em'` (standard), `'cem'` (Classification EM), `'sem'` (Stochastic
  EM), or `'saem'` (Stochastic Approximation EM); `temperature` for deterministic annealing EM
  on top of any of those
- **Scalable fitting**: optional `batch_size` chunks the E-step and sufficient-statistics
  accumulation over rows, bounding peak memory on large datasets; `step_size` switches to
  mini-batch/online EM (plus `partial_fit` for streaming); optional Aitken-accelerated
  stopping rule via `convergence='aitken'`
- **Dirichlet-process mixtures**: `GaussianMixture(n_components=None, ...)` runs a collapsed Gibbs
  sampler that infers the number of components from the data
- **Initialization**: Multiple strategies via `GMMInitializer`, including HDBSCAN (`init_means='hdbscan'`: means, weights and covariances from the cluster assignments)
- **Model Selection**: `bic`/`aic`/`icl` methods for comparing fits at different `n_components`,
  or sweep a grid automatically with `select_model`

### 2. Bayesian Inference with Priors

Support for conjugate priors enables proper Bayesian inference:

- **Weight Prior**: Dirichlet distribution
- **Mean Prior**: Gaussian distribution
- **Covariance Prior**: Wishart/Inverse-Wishart distribution
- **NIW Conjugate Prior**: Normal-Inverse-Wishart for joint mean-covariance updates

### 3. Clustering Metrics

Comprehensive evaluation with `ClusteringMetrics`:

**Unsupervised Metrics** (no ground truth needed; `metric`, `noise_label`, and optional
`n_components` shared across all of them):
- Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Dunn Index
- WB-Index, Wemmert-Gancarski Index, DBCV -- top performers in the Simpson, Campello &
  Stojanovski (2025) clustering-validity benchmark
  (BIC/AIC/ICL are `bic`/`aic`/`icl` methods on `GaussianMixture` directly -- including `GaussianMixture(n_components=None, ...)` -- not `ClusteringMetrics`)

**Supervised Metrics** (with ground truth labels):
- Adjusted Rand Index (ARI), Normalized/Adjusted Mutual Information, Purity
- Homogeneity, Completeness, V-Measure (weighted via `beta`), Normalized Information Distance
- Confusion Matrix and per-class Classification Report (precision, recall, F1, ROC-AUC)

### 4. Visualization Tools

Flexible plotting utilities in `tgmm.plotting`:

- Component ellipses at multiple confidence levels
- Cluster coloring, ground-truth comparison, and log-likelihood coloring
- Initial vs. final mean trajectories and weight-scaled markers

### 5. Semi-Supervised and Positive-Unlabeled Fitting

`GaussianMixture` supports `fit(X, labels=...)` with `-1` for unknown rows -- fully labeled
runs a single closed-form fit (optionally with `components_per_class` for a multi-component
class); any `-1` present runs semi-supervised EM (`unlabeled_weight`, `label_confidence`).
`predict_class(X)` returns class labels, as opposed to `predict(X)`'s raw component indices.

`PositiveUnlabeledMixture` wraps this for positive-unlabeled (PU) learning (Elkan & Noto,
2008): a labeled positive subset, everything else unlabeled. Estimates the class prior and a
per-row positive probability under the SCAR assumption, with `anchor in ('fixed', 'prior', 'free')`
controlling identifiability.

See the [Semi-Supervised and PU Fitting](https://adriansousapoza.github.io/tgmm/notebooks/15_semi_supervised_pu/) tutorial.

### 6. Density-based clustering and building blocks

- **`HDBSCAN`**: a PyTorch implementation of hierarchical density-based clustering (no `K`
  needed, non-convex clusters, explicit noise points).
- **`SufficientStats`** (the additive per-component statistics behind every M-step, used by
  chunked, online and streaming fitting) and **`BaseMixture`** (the EM machinery shared by
  `GaussianMixture`, `ConditionalGaussianMixture` and `GenericMixture`) are public.

## Citation

TBA

## License

Released under the MIT License.
© 2025, Adrián A. Sousa-Poza

## Contributing

tgmm isn't yet set up to accept external contributions. If you're interested in contributing,
please reach out to [asp@di.ku.dk](mailto:asp@di.ku.dk).
