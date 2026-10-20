# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "tgmm",
#     "torch; sys_platform != 'emscripten'",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
#     "scipy",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    # docs: hide-cell
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # docs: hide-cell
    import os as _os

    # TGMM_DOCS_FAST=1 (set by CI) shrinks the slow calls; the defaults below are the realistic local ones.
    FAST = _os.environ.get("TGMM_DOCS_FAST") == "1"
    MAX_ITER = 100 if FAST else 1000
    N_INIT = 1 if FAST else 5
    return FAST, MAX_ITER


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NIW Conjugate Prior vs. Separate Priors

    When both a mean prior and a covariance prior are supplied, `GaussianMixture` uses the joint **Normal-Inverse-Wishart (NIW)** conjugate update rather than treating the two priors independently. This notebook explains how the NIW update differs from putting a prior on only the means or only the covariances, and then probes how sensitive a fit is to each NIW hyperparameter ($\lambda_0$, $\nu_0$, $\boldsymbol{\Psi}_0$, $\boldsymbol{m}_0$) and to the amount of data per component.

    **What you'll learn**

    - what the NIW prior is, and why its posterior update couples the mean and covariance
    - how fits with a mean prior only, a covariance prior only and the full NIW prior differ
    - how $\lambda_0$, $\nu_0$, $\boldsymbol{\Psi}_0$ and $\boldsymbol{m}_0$ interact, over a grid of values
    - how the prior's influence fades as the number of points per component grows, and how it behaves for unequal component sizes

    **Prerequisites:** [09 -- Priors](09_priors.ipynb) (which introduces each prior separately).
    See also the [Gaussian Mixture Model user guide](../user-guide/gaussian-mixture.md).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    The next cell imports what the notebook needs and fixes the random seeds. It expects `tgmm` to be importable: `pip install tgmm`, or `pip install -e .` from a clone of the repository. Opened with `marimo edit --sandbox`, the dependencies listed in the header of this file are installed automatically.
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from tgmm import GaussianMixture
    from tgmm.plotting import plot_gmm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f'Using device: {device}')
    return GaussianMixture, device, np, plot_gmm, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Normal-Inverse-Wishart (NIW) Conjugate Prior Mathematical Background

    The Normal-Inverse-Wishart (NIW) prior is a **conjugate prior** for the parameters of a multivariate Gaussian distribution when both the mean $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ are unknown. This is fundamentally different from specifying separate priors for means and covariances.

    ### Mathematical Formulation

    For a multivariate Gaussian component with parameters $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, the NIW prior is defined as:

    $$p(\boldsymbol{\mu}, \boldsymbol{\Sigma}^{-1}) = \mathcal{N}\!\left(\boldsymbol{\mu} \,\middle|\, \boldsymbol{m}_0, \tfrac{1}{\lambda_0}\boldsymbol{\Sigma}\right) \cdot \mathcal{W}(\boldsymbol{\Sigma}^{-1} | \nu_0, \boldsymbol{\Psi}_0^{-1})$$

    Where:
    - $\boldsymbol{m}_0$: prior mean (hyperparameter)
    - $\lambda_0$: prior precision parameter (`mean_precision_prior`; the prior covariance of $\boldsymbol{\mu}$ is $\boldsymbol{\Sigma}/\lambda_0$)
    - $\nu_0$: degrees of freedom for the Wishart distribution (`degrees_of_freedom_prior`)
    - $\boldsymbol{\Psi}_0$: scale matrix for the Wishart distribution (`covariance_prior`)

    ### Key Properties of NIW Prior

    1. **Conjugacy**: The posterior distribution is also NIW, making Bayesian inference analytically tractable.

    2. **Coupling**: The mean and covariance are coupled through the precision parameter $\lambda_0$. When $\boldsymbol{\Sigma}$ is large (uncertain), the mean prior becomes less informative automatically.

    3. **Natural Parameterization**: The NIW prior naturally captures the relationship between mean uncertainty and covariance uncertainty.

    ### NIW Posterior Updates

    Given data points assigned to component $k$, the NIW posterior parameters are:

    $$\lambda_n = \lambda_0 + n_k$$

    $$\nu_n = \nu_0 + n_k$$

    $$\boldsymbol{m}_n = \frac{\lambda_0 \boldsymbol{m}_0 + n_k \bar{\boldsymbol{x}}_k}{\lambda_0 + n_k}$$

    $$\boldsymbol{\Psi}_n = \boldsymbol{\Psi}_0 + \mathbf{S}_k + \frac{\lambda_0 n_k}{\lambda_0 + n_k}(\bar{\boldsymbol{x}}_k - \boldsymbol{m}_0)(\bar{\boldsymbol{x}}_k - \boldsymbol{m}_0)^T$$

    where $\bar{\boldsymbol{x}}_k$ is the responsibility-weighted mean, $n_k$ the effective sample count and $\mathbf{S}_k$ the scatter matrix of component $k$ about $\bar{\boldsymbol{x}}_k$. tgmm sets $\boldsymbol{\mu}_k = \boldsymbol{m}_n$ and $\boldsymbol{\Sigma}_k = \boldsymbol{\Psi}_n / \nu_n$ -- *not* the textbook inverse-Wishart posterior mean $\boldsymbol{\Psi}_n/(\nu_n - d - 1)$, which is undefined once $\nu_n \le d + 1$. Note that $\boldsymbol{\Psi}_0/\nu_0$ is therefore the prior's best guess for the covariance.

    ### Why NIW ≠ Separate Mean + Covariance Priors

    When using **separate priors**:
    - Mean prior: $p(\boldsymbol{\mu}) = \mathcal{N}(\boldsymbol{\mu} | \boldsymbol{m}_0, \tau_0^2 \mathbf{I})$
    - Covariance prior: $p(\boldsymbol{\Sigma}^{-1}) = \mathcal{W}(\boldsymbol{\Sigma}^{-1} | \nu_0, \boldsymbol{\Psi}_0^{-1})$

    The key differences are:

    1. **Independence vs. coupling.** Separate priors assume independence between mean and covariance uncertainty; NIW ties the mean's prior covariance to the component covariance, $\boldsymbol{\Sigma}/\lambda_0$, so a broad component automatically has a less informative mean prior.
    2. **Prior-strength interpretation.** In NIW, $\lambda_0$ and $\nu_0$ are pseudo-observation counts for the mean and the covariance respectively, and the covariance update gains an extra term $\frac{\lambda_0 n_k}{\lambda_0 + n_k}(\bar{\boldsymbol{x}}_k - \boldsymbol{m}_0)(\bar{\boldsymbol{x}}_k - \boldsymbol{m}_0)^\top$ that grows when the data mean disagrees with the prior mean.
    3. **Updates.** NIW updates both parameters jointly from the same posterior; in tgmm, supplying only `mean_prior` leaves the covariance update as maximum likelihood, supplying only `covariance_prior` leaves the mean update as maximum likelihood, and supplying both switches to the joint NIW update.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate Synthetic Data

    We'll create synthetic 2D data with four well-separated Gaussian components to clearly observe how the prior configurations differ.
    """)
    return


@app.cell
def _(device, np, plot_gmm, plt, torch):
    # Create synthetic 2D data with clear structure
    _n_samples_per_component = 100
    true_means = np.array([[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0]])
    true_covs = np.array([[[0.5, 0.2], [0.2, 0.3]], [[0.8, -0.1], [-0.1, 0.4]], [[0.3, 0.0], [0.0, 0.6]], [[0.4, 0.1], [0.1, 0.5]]])
    _X_list = []
    for _i, (_mean, _cov) in enumerate(zip(true_means, true_covs)):
        _X_component = np.random.multivariate_normal(_mean, _cov, _n_samples_per_component)
        _X_list.append(_X_component)
    X = np.vstack(_X_list)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    n_samples, n_features = X.shape
    n_components = 4
    plt.figure()
    plot_gmm(X_tensor.cpu().numpy(), true_labels=np.repeat(np.arange(n_components), _n_samples_per_component), title='Generated Data with True Labels')
    # Generate data
    print(f'Generated {n_samples} samples with {n_features} features')
    print(f'Data shape: {X.shape}')
    # Plot the generated data
    print(f'Data range: X1=[{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], X2=[{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]')
    plt.gcf()
    return X, X_tensor, n_components, n_features, true_covs, true_means


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define Prior Hyperparameters

    We'll use the same prior hyperparameters for both approaches for every configuration, to ensure a fair comparison.
    """)
    return


@app.cell
def _(device, n_components, n_features, torch):
    # Define NIW hyperparameters
    lambda0 = 20  # Prior precision parameter - moderate coupling
    nu0 = n_features + 100  # Degrees of freedom (weak prior)
    m0 = torch.zeros(n_components, n_features, device=device)  # Prior mean at origin

    # Scale matrix (we expect unit covariances)
    Psi0 = torch.eye(n_features, device=device).unsqueeze(0).expand(n_components, -1, -1) * 100.0

    print("Prior Hyperparameters:")
    print(f"λ₀ (precision parameter): {lambda0}")
    print(f"ν₀ (degrees of freedom): {nu0}")
    print(f"m₀ (prior mean): {m0[0].cpu().numpy()}")
    print(f"Ψ₀ (scale matrix): \n{Psi0[0].cpu().numpy()}")
    print(f"\nNote: All four components use the same prior hyperparameters")
    return Psi0, lambda0, m0, nu0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison: Different Prior Configurations

    Now we'll fit the same data using three different configurations:
    1. **Mean Prior Only**: Prior on means, MLE for covariances
    2. **Covariance Prior Only**: Prior on covariances, MLE for means
    3. **NIW Conjugate Prior**: Coupled prior on both parameters

    This will demonstrate how priors affect the fitted parameters.
    """)
    return


@app.cell
def _(
    GaussianMixture,
    MAX_ITER,
    Psi0,
    X,
    X_tensor,
    device,
    lambda0,
    m0,
    n_components,
    n_features,
    nu0,
    plot_gmm,
    plt,
):
    # Set up the comparison
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 6))
    _fig.suptitle("Comparison: Different Prior Configurations", fontsize=16, y=1.02)

    _models = {}

    # 1. Mean Prior Only
    print("Fitting Mean Prior Only...")
    _ax = _axes[0]
    _models['mean_only'] = GaussianMixture(
        n_features=n_features,
        n_components=n_components,
        covariance_type='full',
        max_iter=MAX_ITER,
        device=device,
        mean_prior=m0,
        mean_precision_prior=lambda0,
    )
    _models['mean_only'].fit(X_tensor)

    plot_gmm(
        X=X,
        gmm=_models['mean_only'],
        ax=_ax,
        title=f"Mean Prior Only\nLL: {_models['mean_only'].lower_bound_:.2f}",
    )

    # 2. Covariance Prior Only
    print("Fitting Covariance Prior Only...")
    _ax = _axes[1]
    _models['cov_only'] = GaussianMixture(
        n_features=n_features,
        n_components=n_components,
        covariance_type='full',
        max_iter=MAX_ITER,
        device=device,
        covariance_prior=Psi0,
        degrees_of_freedom_prior=nu0,
    )
    _models['cov_only'].fit(X_tensor)

    plot_gmm(
        X=X,
        gmm=_models['cov_only'],
        ax=_ax,
        title=f"Covariance Prior Only\nLL: {_models['cov_only'].lower_bound_:.2f}",
    )

    # 3. NIW Conjugate Prior
    print("Fitting NIW Conjugate Prior...")
    _ax = _axes[2]
    _models['niw'] = GaussianMixture(
        n_features=n_features,
        n_components=n_components,
        covariance_type='full',
        max_iter=MAX_ITER,
        device=device,
        mean_prior=m0,
        mean_precision_prior=lambda0,
        covariance_prior=Psi0,
        degrees_of_freedom_prior=nu0,
    )
    _models['niw'].fit(X_tensor)

    plot_gmm(
        X=X,
        gmm=_models['niw'],
        ax=_ax,
        title=f"NIW Conjugate Prior\nLL: {_models['niw'].lower_bound_:.2f}",
    )

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(GaussianMixture, np, plt, torch):
    def analyze_niw_hyperparameters(X_tensor, n_components, n_features, device, title_suffix="", fast=False):
        """Run NIW hyperparameter sensitivity analysis on given data.

        Args:
            X_tensor: Input data tensor
            n_components: Number of mixture components
            n_features: Number of features
            device: torch device
            title_suffix: Additional text for plot title
            fast: use coarser hyperparameter grids (docs build)
        """
        # Define parameter grids (coarser when `fast`)
        lambda_grid = [0.1, 10.0, 1000] if fast else [0.1, 1.0, 10.0, 100.0, 1000]
        nu_grid = [n_features, 50, 500] if fast else [n_features, 10, 50, 100, 500, 1000]
        psi_scales = [0.1, 10.0, 1000.0] if fast else [0.1, 1.0, 10.0, 100.0, 1000.0]
        m0_scales = [0, 4, 16] if fast else [0, 2, 4, 8, 16]
        signs = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], device=device)
        psi_grid = [
            torch.eye(n_features, device=device).unsqueeze(0).expand(n_components, -1, -1) * scale
            for scale in psi_scales
        ]
        m0_grid = [signs * float(m) for m in m0_scales]  # 0 -> all zeros, else corners at +-m

        # Default values
        lambda_default = 1.0
        nu_default = n_features + 10
        m0_default = torch.zeros(n_components, n_features, device=device)
        psi_default = torch.eye(n_features, device=device).unsqueeze(0).expand(n_components, -1, -1) * 1.0

        # Define all 6 parameter combinations
        param_combinations = [
            {'x': 'nu', 'y': 'lambda', 'x_grid': nu_grid, 'y_grid': lambda_grid, 
             'x_label': 'ν₀ (Degrees of Freedom)', 'y_label': 'λ₀ (Coupling Parameter)',
             'fixed': {'m0': m0_default, 'psi': psi_default}},

            {'x': 'psi', 'y': 'lambda', 'x_grid': list(range(len(psi_grid))), 'y_grid': lambda_grid,
             'x_label': 'Ψ₀ Scale', 'y_label': 'λ₀ (Coupling Parameter)',
             'fixed': {'nu': nu_default, 'm0': m0_default}},

            {'x': 'm0', 'y': 'lambda', 'x_grid': list(range(len(m0_grid))), 'y_grid': lambda_grid,
             'x_label': 'm₀ Configuration', 'y_label': 'λ₀ (Coupling Parameter)',
             'fixed': {'nu': nu_default, 'psi': psi_default}},

            {'x': 'psi', 'y': 'nu', 'x_grid': list(range(len(psi_grid))), 'y_grid': nu_grid,
             'x_label': 'Ψ₀ Scale', 'y_label': 'ν₀ (Degrees of Freedom)',
             'fixed': {'lambda': lambda_default, 'm0': m0_default}},

            {'x': 'm0', 'y': 'nu', 'x_grid': list(range(len(m0_grid))), 'y_grid': nu_grid,
             'x_label': 'm₀ Configuration', 'y_label': 'ν₀ (Degrees of Freedom)',
             'fixed': {'lambda': lambda_default, 'psi': psi_default}},

            {'x': 'm0', 'y': 'psi', 'x_grid': list(range(len(m0_grid))), 'y_grid': list(range(len(psi_grid))),
             'x_label': 'm₀ Configuration', 'y_label': 'Ψ₀ Scale',
             'fixed': {'lambda': lambda_default, 'nu': nu_default}},
        ]

        # Labels for categorical grids
        psi_labels = [f'{v:g}' for v in psi_scales]
        m0_labels = ['0' if m == 0 else f'±{m}' for m in m0_scales]

        # Create 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for idx, combo in enumerate(param_combinations):    
            # Initialize results matrix
            ll_matrix = np.zeros((len(combo['y_grid']), len(combo['x_grid'])))

            # Iterate over all combinations
            for i, y_val in enumerate(combo['y_grid']):
                for j, x_val in enumerate(combo['x_grid']):
                    # Set up parameters
                    params = combo['fixed'].copy()
    
                    # Set y parameter (look up from grid if it's m0 or psi)
                    if combo['y'] == 'm0':
                        params['m0'] = m0_grid[y_val]
                    elif combo['y'] == 'psi':
                        params['psi'] = psi_grid[y_val]
                    else:
                        params[combo['y']] = y_val
    
                    # Set x parameter (look up from grid if it's m0 or psi)
                    if combo['x'] == 'm0':
                        params['m0'] = m0_grid[x_val]
                    elif combo['x'] == 'psi':
                        params['psi'] = psi_grid[x_val]
                    else:
                        params[combo['x']] = x_val
    
                    # Handle proper parameter names
                    mean_prior = params.get('m0', m0_default)
                    mean_precision_prior = params.get('lambda', lambda_default)
                    covariance_prior = params.get('psi', psi_default)
                    degrees_of_freedom_prior = params.get('nu', nu_default)
    
                    model = GaussianMixture(
                        n_features=n_features,
                        n_components=n_components,
                        covariance_type='full',
                        max_iter=1000,
                        device=device,
                        mean_prior=mean_prior,
                        mean_precision_prior=mean_precision_prior,
                        covariance_prior=covariance_prior,
                        degrees_of_freedom_prior=degrees_of_freedom_prior,
                    )
    
                    model.fit(X_tensor)
    
                    # Assert convergence
                    assert model.converged_, f"Model did not converge for {combo['y']}={y_val}, {combo['x']}={x_val}"
    
                    ll_matrix[i, j] = model.lower_bound_

            # Plot heatmap
            ax = axes[idx]
            im = ax.imshow(ll_matrix, cmap='RdYlGn', aspect='auto')
            ax.set_title(f'{combo["y_label"]} vs {combo["x_label"]}', fontsize=11, pad=10)
            ax.set_xlabel(combo['x_label'], fontsize=9)
            ax.set_ylabel(combo['y_label'], fontsize=9)

            # Set tick labels
            ax.set_xticks(range(len(combo['x_grid'])))
            if combo['x'] == 'm0':
                ax.set_xticklabels(m0_labels, fontsize=8)
            elif combo['x'] == 'psi':
                ax.set_xticklabels(psi_labels, fontsize=8)
            else:
                ax.set_xticklabels([f'{v}' for v in combo['x_grid']], fontsize=8)

            ax.set_yticks(range(len(combo['y_grid'])))
            if combo['y'] == 'm0':
                ax.set_yticklabels(m0_labels, fontsize=8)
            elif combo['y'] == 'psi':
                ax.set_yticklabels(psi_labels, fontsize=8)
            else:
                ax.set_yticklabels([f'{v}' for v in combo['y_grid']], fontsize=8)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('LL', rotation=270, labelpad=12, fontsize=8)

            # Add text annotations
            for i in range(len(combo['y_grid'])):
                for j in range(len(combo['x_grid'])):
                    ax.text(j, i, f'{ll_matrix[i, j]:.3f}', 
                            ha='center', va='center', fontweight='bold', fontsize=8)

        plt.suptitle(f'NIW Prior Hyperparameter Interactions: Log-Likelihood Heatmaps{title_suffix}', 
                     fontsize=14, y=0.995)
        plt.tight_layout()
        return fig

    return (analyze_niw_hyperparameters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameter Sensitivity Analysis Setup

    Define reusable parameter grids and analysis function for NIW hyperparameter experiments.
    """)
    return


@app.cell
def _(
    FAST,
    analyze_niw_hyperparameters,
    device,
    n_components,
    n_features,
    np,
    torch,
    true_covs,
    true_means,
):
    _n_samples_per_component = 100
    _X_list = []
    for _i, (_mean, _cov) in enumerate(zip(true_means, true_covs)):
        _X_component = np.random.multivariate_normal(_mean, _cov, _n_samples_per_component)
        _X_list.append(_X_component)
    X_1 = np.vstack(_X_list)
    X_tensor_1 = torch.tensor(X_1, dtype=torch.float32, device=device)
    print(f'Experiment 1: {_n_samples_per_component} samples per component')
    analyze_niw_hyperparameters(X_tensor_1, n_components, n_features, device, title_suffix=' (n=100/comp)', fast=FAST)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What happens if we use 1000 instead of 100 points per component?

    (In the fast docs build, `TGMM_DOCS_FAST=1`, 300 points per component and coarser hyperparameter grids are used.)
    """)
    return


@app.cell
def _(
    FAST,
    analyze_niw_hyperparameters,
    device,
    n_components,
    n_features,
    np,
    torch,
    true_covs,
    true_means,
):
    _n_samples_per_component = 300 if FAST else 1000
    _X_list = []
    for _i, (_mean, _cov) in enumerate(zip(true_means, true_covs)):
        _X_component = np.random.multivariate_normal(_mean, _cov, _n_samples_per_component)
        _X_list.append(_X_component)
    X_2 = np.vstack(_X_list)
    X_tensor_2 = torch.tensor(X_2, dtype=torch.float32, device=device)
    print(f'Experiment 2: {_n_samples_per_component} samples per component')
    analyze_niw_hyperparameters(X_tensor_2, n_components, n_features, device, title_suffix=f' (n={_n_samples_per_component}/comp)', fast=FAST)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What happens if we use diffenrent number of points per component?
    """)
    return


@app.cell
def _(
    FAST,
    analyze_niw_hyperparameters,
    device,
    n_components,
    n_features,
    np,
    torch,
    true_covs,
    true_means,
):
    _n_samples_per_component = [100, 200, 300, 500] if FAST else [100, 500, 1000, 5000]
    _X_list = []
    for _i, (_mean, _cov, n_s) in enumerate(zip(true_means, true_covs, _n_samples_per_component)):
        _X_component = np.random.multivariate_normal(_mean, _cov, n_s)
        _X_list.append(_X_component)
    X_3 = np.vstack(_X_list)
    X_tensor_3 = torch.tensor(X_3, dtype=torch.float32, device=device)
    print(f'Experiment 3: {_n_samples_per_component} samples per component (unequal)')
    analyze_niw_hyperparameters(X_tensor_3, n_components, n_features, device, title_suffix=f' (n={_n_samples_per_component})', fast=FAST)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - The NIW prior is conjugate: the posterior is again NIW, with $\lambda_n=\lambda_0+n_k$, $\nu_n=\nu_0+n_k$, $\boldsymbol{m}_n$ a weighted mean of $\boldsymbol{m}_0$ and $\bar{\boldsymbol{x}}_k$, and $\boldsymbol{\Psi}_n$ gaining a term that grows when the data mean disagrees with $\boldsymbol{m}_0$.
    - Using only a mean prior or only a covariance prior regularizes one parameter set while the other stays maximum likelihood; the NIW prior updates both jointly.
    - The hyperparameters trade off against each other: $\lambda_0$ and $\nu_0$ act as pseudo-observation counts, while $\boldsymbol{\Psi}_0/\nu_0$ and $\boldsymbol{m}_0$ set where the prior pulls.
    - A prior's influence fades as $n_k$ grows, so priors matter most for small or unbalanced components.

    **Next:** [11 -- Dirichlet Process Mixtures](11_dpgmm.ipynb), where the same NIW prior serves as the base measure of a nonparametric sampler.
    """)
    return


if __name__ == "__main__":
    app.run()
