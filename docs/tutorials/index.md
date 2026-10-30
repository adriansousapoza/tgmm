# Tutorials

The tutorials are the documentation: each one is a [marimo](https://marimo.io) notebook with the mathematics,
the code and the figures together, numbered in the suggested reading order. Every page has an **Open the
interactive version** link at the top, which opens the same notebook with real tgmm so you can change the data
and the settings (the first tutorial lets you move, reshape, add and remove the Gaussians). Locally:
`uvx marimo edit --sandbox notebooks/01_introduction.py`.

## Core

1. [**Introduction: Gaussian Mixtures and EM**](../notebooks/01_introduction.ipynb): build the data (and design your own), the model, the EM algorithm, fitting and inspecting a `GaussianMixture`
2. [**Covariance Types**](../notebooks/02_covariance_types.ipynb): the six classic types and the 14 GPCM models, and how each is estimated
3. [**Initialization**](../notebooks/03_initialization.ipynb): means, covariances, weights and `n_init`
4. [**Sampling**](../notebooks/04_sampling.ipynb): drawing from a fitted mixture, with and without constraints

## Fitting and evaluation

5. [**Visualization**](../notebooks/05_visualise.ipynb): `plot_gmm` and friends
6. [**Classification EM (CEM)**](../notebooks/06_cem.ipynb)
7. [**Supervised Fitting**](../notebooks/07_supervised_gmm.ipynb)
8. [**Metrics and Evaluation**](../notebooks/08_metrics.ipynb)
9. [**PCA Plotting**](../notebooks/09_pca_plotting.ipynb)

## Priors, nonparametric and robust methods

10. [**Priors (MAP)**](../notebooks/10_priors.ipynb)
11. [**NIW Priors Comparison**](../notebooks/11_niw_priors_comparison.ipynb)
12. [**Dirichlet Process Mixtures**](../notebooks/12_dpgmm.ipynb): stick-breaking and Gibbs sampling
13. [**HDBSCAN**](../notebooks/13_hdbscan.ipynb)
14. [**Model Selection (BIC, AIC, ICL)**](../notebooks/14_model_selection.ipynb)
15. [**Stochastic EM, Annealing and SAEM**](../notebooks/15_stochastic_em.ipynb)

## Other mixture families and features

16. [**Student-t Mixtures**](../notebooks/16_student_t.ipynb)
17. [**Von Mises-Fisher Mixtures**](../notebooks/17_von_mises_fisher.ipynb)
18. [**Semi-Supervised and PU Fitting**](../notebooks/18_semi_supervised_pu.ipynb)
19. [**Noise, Missing Values and Conditioning**](../notebooks/19_noise_missing_conditioning.ipynb)
20. [**Constraints, Frozen Components and Sample Weights**](../notebooks/20_constraints_and_weights.ipynb)
