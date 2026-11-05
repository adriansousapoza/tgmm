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
4. [**Priors (MAP and NIW)**](../notebooks/04_priors.ipynb): regularising the fit with prior distributions
5. [**Sampling**](../notebooks/05_sampling.ipynb): drawing from a fitted mixture, with and without constraints

## Fitting and evaluation

6. [**Visualization**](../notebooks/06_visualise.ipynb): `plot_gmm` and friends
7. [**Classification EM (CEM)**](../notebooks/07_cem.ipynb)
8. [**Supervised Fitting**](../notebooks/08_supervised_gmm.ipynb)
9. [**Metrics and Evaluation**](../notebooks/09_metrics.ipynb)
10. [**PCA Plotting**](../notebooks/10_pca_plotting.ipynb)

## Nonparametric and robust methods

11. [**Dirichlet Process Mixtures**](../notebooks/11_dpgmm.ipynb): stick-breaking and Gibbs sampling
12. [**HDBSCAN**](../notebooks/12_hdbscan.ipynb)
13. [**Model Selection (BIC, AIC, ICL)**](../notebooks/13_model_selection.ipynb)
14. [**Stochastic EM, Annealing and SAEM**](../notebooks/14_stochastic_em.ipynb)

## Other mixture families and features

15. [**Student-t Mixtures**](../notebooks/15_student_t.ipynb)
16. [**Von Mises-Fisher Mixtures**](../notebooks/16_von_mises_fisher.ipynb)
17. [**Semi-Supervised and PU Fitting**](../notebooks/17_semi_supervised_pu.ipynb)
18. [**Noise, Missing Values and Conditioning**](../notebooks/18_noise_missing_conditioning.ipynb)
19. [**Constraints, Frozen Components and Sample Weights**](../notebooks/19_constraints_and_weights.ipynb)

## Algorithm walk-throughs

20. [**k-means++ Seeding Step by Step**](../notebooks/20_kmeans_plus_plus.ipynb): how `init_means='kpp'` picks its centres, with the sampling probabilities drawn at every step
21. [**k-means (Lloyd's Algorithm) Step by Step**](../notebooks/21_kmeans.ipynb): how `init_means='kmeans'` iterates, with the Voronoi regions, the centroid paths and the inertia
