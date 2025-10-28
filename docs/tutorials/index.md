# Tutorials

Interactive Jupyter notebook tutorials demonstrating various features of TorchGMM.

## Getting Started

- [**Gaussian Mixture Models (GMM)**](../notebooks/gmm.ipynb) - Basic introduction to fitting GMMs with TorchGMM
- [**Initialization Methods**](../notebooks/cem.ipynb) - Comparison of initialization strategies including CEM

## Advanced Topics

- [**Prior Distributions**](../notebooks/priors.ipynb) - Using priors for regularization and MAP estimation
- [**NIW Priors Comparison**](../notebooks/niw_priors_comparison.ipynb) - Detailed comparison of Normal-Inverse-Wishart priors
- [**Sampling from GMMs**](../notebooks/sampling.ipynb) - Generating synthetic data from fitted models

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
