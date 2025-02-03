# TorchGMM: A Gaussian Mixture Model Implementation with PyTorch

![TorchGMM logo](docs/_static/logo-no-background.png)

**TorchGMM** is a flexible implementation of Gaussian Mixture Models in PyTorch, supporting:

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

```bash
git clone https://github.com/YourUser/TorchGMM.git
cd TorchGMM
pip install -r requirements.txt
```

Make sure you have PyTorch installed. For GPU usage, install the CUDA-enabled version of PyTorch as per the official instructions.

## Documentation
We use Sphinx to build documentation. The generated HTML pages live under docs/_build/html/. You can also read them online if you host them (e.g., on GitHub Pages).

```bash
cd docs
make clean
make html
# Open _build/html/index.html in a browser
# Linux
xdg-open _build/html/index.html 
```

The docs include:

API Reference for all modules (see GaussianMixture, GMMInitializer, and ClusteringMetrics).
Tutorials that walk through different usage scenarios (basic GMM, metrics, using priors).
Tutorials
We provide three Jupyter notebooks in the tutorials/ folder:

GMM Tutorial: Basic usage of the GaussianMixture class.
Metrics Tutorial: Demonstrates ClusteringMetrics and how to compare models.
Priors Tutorial: Shows how to use weight/mean/covariance priors (MAP).
To view or run them locally, just open them in Jupyter or VS Code.

## Basic Usage Example
Here’s a short snippet:

```python
import torch
from utils.gmm import GaussianMixture

# Generate random 2D data
X = torch.randn(500, 2)

# Create and fit the GMM
gmm = GaussianMixture(
    n_features=2,
    n_components=3,
    covariance_type='full',
    max_iter=200
)
gmm.fit(X)

print("Converged?:", gmm.converged_)
print("Cluster Weights:", gmm.weights_)
print("Cluster Means:", gmm.means_)
```

You can also run on GPU by specifying device='cuda' in the GaussianMixture constructor (assuming a CUDA-capable device).

## Contributing
Fork the repository and create your feature branch.
Make changes and add tests or notebooks as appropriate.
Submit a pull request (PR) for review.
We welcome improvements to both the code and the documentation.

## License
Released under the MIT License.
© 2025, Adrián A. Sousa-Poza
