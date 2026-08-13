# Tutorials

Interactive Jupyter notebook tutorials demonstrating various features of tgmm, numbered in the
suggested reading order.

## 1. Foundations

- [**1. Gaussian Mixture Models (GMM)**](../notebooks/01_gmm.ipynb) - Basic introduction to fitting GMMs with tgmm, including a comparison of mean/weight/covariance initialization strategies
- [**2. EM Algorithm Walkthrough**](../notebooks/02_em_algorithm.ipynb) - Step-by-step visualization of how the EM algorithm updates responsibilities and parameters each iteration
- [**3. Visualization Techniques**](../notebooks/03_visualise.ipynb) - Comprehensive guide to `plot_gmm`, the toolkit every later tutorial uses

## 2. Fitting Variants

- [**4. Classification EM (CEM)**](../notebooks/04_cem.ipynb) - Comparing the hard-assignment CEM algorithm against standard EM
- [**5. Supervised Fitting**](../notebooks/05_supervised_gmm.ipynb) - Comparing EM, CEM, and label-supervised fitting on the same synthetic data

## 3. Evaluation & Utilities

- [**6. Clustering Metrics**](../notebooks/06_metrics.ipynb) - Evaluating GMM performance with various metrics
- [**7. Sampling from GMMs**](../notebooks/07_sampling.ipynb) - Generating synthetic data from fitted models
- [**8. PCA Plotting**](../notebooks/08_pca_plotting.ipynb) - Visualizing high-dimensional GMM results with PCA

## 4. Bayesian Priors

- [**9. Prior Distributions**](../notebooks/09_priors.ipynb) - Using priors for regularization and MAP estimation
- [**10. NIW Priors Comparison**](../notebooks/10_niw_priors_comparison.ipynb) - Detailed comparison of Normal-Inverse-Wishart priors

## 5. Advanced / Nonparametric Methods

- [**11. Dirichlet Process Mixtures: Stick-Breaking and Gibbs Sampling**](../notebooks/11_dpgmm.ipynb) - The math (stick-breaking, Chinese Restaurant Process), a step-by-step sampler walkthrough, then validating `GaussianMixture(n_components=None, ...)` against classical EM at scale
- [**12. HDBSCAN Clustering**](../notebooks/12_hdbscan.ipynb) - Density-based clustering with a PyTorch HDBSCAN implementation, including using it to remove noise before fitting a GMM

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
