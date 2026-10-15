# Tutorials

Interactive Jupyter notebook tutorials demonstrating various features of tgmm, numbered in the
suggested reading order.

## 1. Foundations

- [**1. Gaussian Mixture Models (GMM)**](../notebooks/01_gmm.ipynb) - Basic introduction to fitting GMMs with tgmm: initialization strategies (means, weights, covariances), training controls (`n_init`, `warm_start`, ...), MAP priors, constrained sampling and save/load. How the *covariance structure* is chosen and estimated now has its own tutorial, [15. Covariance Structures](../notebooks/15_covariance_models.ipynb)
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

## 6. Model Selection and Algorithms

- [**13. Model Selection (BIC, AIC, ICL)**](../notebooks/13_model_selection.ipynb) - Choosing the number of components and the covariance structure with `select_model`, and how the three criteria differ on overlapping clusters
- [**14. Stochastic EM, Annealing & SAEM**](../notebooks/14_stochastic_em.ipynb) - Escaping poor local optima: SEM, deterministic annealing (DAEM) and SAEM compared against EM and CEM from the same bad initialization

## 7. Covariance Structures

- [**15. Covariance Structures: the 6 classic types and the 14 GPCM models**](../notebooks/15_covariance_models.ipynb) - What each covariance type assumes, how its parameters are estimated in the M-step (the equations), the Celeux & Govaert volume/shape/orientation decomposition behind the 14 mclust models, and model selection among all 14

## 8. Other Families and Supervision

- [**16. Student-t Mixtures**](../notebooks/16_student_t.ipynb) - Heavy-tailed components, the scale-mixture view and robustness weights $u_{ik}$, and resisting contamination with `StudentTMixture`
- [**17. Von Mises-Fisher Mixtures**](../notebooks/17_von_mises_fisher.ipynb) - Directional data: the vMF distribution on the circle and sphere, the EM updates for $\mu_k$ and $\kappa_k$, `plot_vmf` / `plot_vmf_cosine`, and clustering unit-norm embeddings with `VonMisesFisherMixture`
- [**18. Semi-Supervised and Positive-Unlabeled Fitting**](../notebooks/18_semi_supervised_pu.ipynb) - Fitting with partially known labels and `PositiveUnlabeledMixture`

## 9. Around the Fit

- [**19. Noise, Missing Values and Conditioning**](../notebooks/19_noise_missing_conditioning.ipynb) - A uniform noise component (`noise=True`), EM with missing values and `impute`, `marginal` / `condition` and Gaussian mixture regression, `frozen_components` and `sample_weight`, and `GenericMixture` with your own component classes

## Interactive explainers

Four short pages teach the ideas behind tgmm with sliders and buttons that run **in your browser** (no install, no tgmm
or PyTorch involved):

- [GPCM Covariance Shapes](../explainers/gpcm-shapes.md) -- volume, shape and orientation, and the mclust code they spell
- [EM Step by Step](../explainers/em-stepper.md) -- the E- and M-steps of EM, CEM and SEM on a 1-D mixture
- [Deterministic Annealing](../explainers/annealing.md) -- how a temperature escapes a collapsed initialization
- [Clustering Metrics and Model Selection](../explainers/metrics-intuition.md) -- internal indices, BIC and ICL live

## Running Tutorials

The tutorials are [marimo](https://marimo.io) notebooks (`notebooks/NN_name.py`), published here as static
pages. Each page has an **Open in molab** badge at the top that runs the real notebook, PyTorch included, on a
free hosted server (no setup; the full-size runs, not the shortened ones used to build this site).

### Locally

```bash
git clone https://github.com/adriansousapoza/tgmm.git && cd tgmm
pip install -e . marimo
marimo edit notebooks/01_gmm.py        # interactive editor
python notebooks/01_gmm.py             # or run it as a plain script
```

Or, without installing anything yourself, in an isolated environment built from the notebook's own header:
`uvx marimo edit --sandbox notebooks/01_gmm.py`. See `notebooks/README.md` for the conventions every notebook follows.

## Requirements

All tutorials require:

```bash
pip install torch numpy scipy matplotlib seaborn tqdm
pip install tgmm  # or, from a source checkout: pip install -e .
```

Some tutorials additionally use scikit-learn (datasets, PCA), pandas (`select_model(...).to_dataframe()`) or `mpltern`; see [Installation](../getting-started/installation.md#optional-dependencies). The notebooks declare all of their dependencies in a header, so `marimo edit --sandbox` installs them for you.
