# Tutorials

Interactive Jupyter notebook tutorials demonstrating various features of TorchGMM.

## Getting Started

- [**Gaussian Mixture Models (GMM)**](../notebooks/gmm.ipynb) - Basic introduction to fitting GMMs with TorchGMM, including a comparison of mean/weight/covariance initialization strategies
- [**EM Algorithm Walkthrough**](../notebooks/em_algorithm.ipynb) - Step-by-step visualization of how the EM algorithm updates responsibilities and parameters each iteration

## Advanced Topics

- [**Classification EM (CEM)**](../notebooks/cem.ipynb) - Comparing the hard-assignment CEM algorithm against standard EM
- [**Supervised Fitting**](../notebooks/supervised_gmm.ipynb) - Comparing EM, CEM, and label-supervised fitting on the same synthetic data
- [**Prior Distributions**](../notebooks/priors.ipynb) - Using priors for regularization and MAP estimation
- [**NIW Priors Comparison**](../notebooks/niw_priors_comparison.ipynb) - Detailed comparison of Normal-Inverse-Wishart priors
- [**Sampling from GMMs**](../notebooks/sampling.ipynb) - Generating synthetic data from fitted models
- [**Dirichlet Process Mixture (Gibbs Sampling)**](../notebooks/dpgmm.ipynb) - Inferring the number of components with `GaussianMixture(n_components=None, ...)`, compared against classical EM
- [**Gibbs Sampling Internals**](../notebooks/dpgmm_gibbs_sampling.ipynb) - Step-by-step look at the collapsed Gibbs sampler itself

## Visualization

- [**PCA Plotting**](../notebooks/pca_plotting.ipynb) - Visualizing high-dimensional GMM results with PCA
- [**Visualization Techniques**](../notebooks/visualise.ipynb) - Comprehensive guide to plotting GMM results

## Model Evaluation

- [**Clustering Metrics**](../notebooks/metrics.ipynb) - Evaluating GMM performance with various metrics

## Running Tutorials

### Option 1: Google Colab

Click the "Open in Colab" button at the top of each notebook.

### Option 2: Local Jupyter

Install Jupyter:

```bash
pip install jupyter
```

Launch Jupyter:

```bash
cd docs/notebooks
jupyter notebook
```

### Option 3: JupyterLab

```bash
pip install jupyterlab
jupyter lab
```

## Requirements

All tutorials require:

```bash
pip install torch numpy matplotlib scipy
pip install tgmm  # or install from source
```

Some tutorials may have additional dependencies listed at the top of the notebook.
