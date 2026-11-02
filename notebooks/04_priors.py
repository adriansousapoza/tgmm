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
#     "mpltern",
# ]
# ///

import marimo

__generated_with = "0.25.2"
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
    # Priors: MAP Estimation and the Normal-Inverse-Wishart Prior

    Priors turn the maximum-likelihood M-step of EM into a **maximum a posteriori (MAP)** update: each parameter estimate is pulled toward a prior guess by an amount that depends on how much data supports the component. This regularizes the model, prevents degenerate (collapsed) components, and lets you encode domain knowledge. This notebook builds up the three priors `tgmm` supports, deriving the distributions involved and showing how each one changes a fit, and then studies the conjugate **Normal-Inverse-Wishart (NIW)** prior that tgmm uses when a mean prior and a covariance prior are given together.

    **What you'll learn**

    - how a Dirichlet prior on the weights (`weight_concentration_prior`) shapes the mixing proportions, and the MAP weight update it produces
    - how a Gaussian prior on the means (`mean_prior`, `mean_precision_prior`) acts as pseudo-observations
    - why an inverse-Wishart (or inverse-Gamma) prior is the natural choice for covariances, and how `covariance_prior` and `degrees_of_freedom_prior` change a fit
    - how the NIW prior couples means and covariances, the exact posterior update tgmm uses (for every covariance type), and how a fit depends on each NIW hyperparameter and on the amount of data per component

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb) (the model and the EM updates this notebook modifies) and [03 -- Initialization](03_initialization.ipynb) (how EM is started).
    See also [06 -- Visualization](06_visualise.ipynb) (for `plot_gmm`) and the [`GaussianMixture` API reference](../api/gaussian-mixture.md).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The priors at a glance

    All priors are conjugate, and each one only changes the corresponding M-step update (the E-step is untouched; see [01](01_introduction.ipynb)). Throughout, $n_k=\sum_i r_{ik}$ is the effective size of component $k$, $\bar y_k=\frac1{n_k}\sum_i r_{ik}x_i$ its responsibility-weighted mean and $W_k=\sum_i r_{ik}(x_i-\bar y_k)(x_i-\bar y_k)^\top$ its scatter.

    | parameter | prior | constructor arguments | MAP update (`covariance_type='full'`) |
    | --- | --- | --- | --- |
    | weights $\pi$ | $\operatorname{Dirichlet}(\alpha_1,\dots,\alpha_K)$ | `weight_concentration_prior` $=\alpha$ | $\pi_k=\dfrac{n_k+\alpha_k-1}{n+\sum_j\alpha_j-K}$ |
    | mean $\mu_k$ | $\mathcal N(\mu_{0,k},\,\Sigma_k/\kappa_0)$ | `mean_prior` $=\mu_0$, `mean_precision_prior` $=\kappa_0$ | $\mu_k=\dfrac{n_k\bar y_k+\kappa_0\mu_{0,k}}{n_k+\kappa_0}$ |
    | covariance $\Sigma_k$ | $\operatorname{Inverse\text{-}Wishart}(\Psi_0,\nu_0)$ | `covariance_prior` $=\Psi_0$, `degrees_of_freedom_prior` $=\nu_0$ | $\Sigma_k=\dfrac{\Psi_0+W_k+\varepsilon I}{\nu_0+n_k}$ |

    A prior is only active when *both* of its arguments are given (`mean_prior` needs `mean_precision_prior`, `covariance_prior` needs `degrees_of_freedom_prior`). Passing **both** a mean and a covariance prior switches to the *joint* Normal-Inverse-Wishart update, which is the subject of the NIW section below. `weight_concentration_prior` is independent of all this and works with every covariance type, including the 14 GPCM codes; `mean_prior` / `covariance_prior` work with the six classic types only (the 8 GPCM-only codes raise `NotImplementedError`).
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
    import warnings
    import scipy.special
    from scipy.stats import wishart, invgamma
    from scipy.ndimage import gaussian_filter


    from tgmm import GaussianMixture, GMMInitializer, dynamic_figsize, generate_gmm_data, plot_gmm
    from tutorial_data import default_blobs, make_default_data

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plt.rcParams['figure.dpi'] = 45  # tgmm's theme uses 300 dpi; keep embedded figure outputs small

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f'Using device: {device}')
    return (
        GMMInitializer,
        GaussianMixture,
        SEED,
        default_blobs,
        device,
        dynamic_figsize,
        gaussian_filter,
        generate_gmm_data,
        invgamma,
        make_default_data,
        np,
        plot_gmm,
        plt,
        scipy,
        torch,
        warnings,
        wishart,
    )


@app.cell
def _(SEED, device, make_default_data):
    # The default four blobs of 01 -- Introduction (see tutorial_data.py)
    X_tensor, _ = make_default_data(SEED, device=device)
    X = X_tensor.cpu().numpy()
    legend_labels = [f'Component {i + 1}' for i in range(4)]
    n_features = X.shape[1]
    n_components = 4
    return X, X_tensor, legend_labels, n_components, n_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dirichlet Distribution: Definition and Derivation from the Beta Distribution

    The Dirichlet distribution is a multivariate generalization of the Beta distribution. For a vector
    $$
    \mathbf{x} = (x_1, x_2, \dots, x_K)
    $$
    that lies in the $(K-1)$-simplex, i.e.,
    $$
    \sum_{i=1}^K x_i = 1 \quad \text{and} \quad x_i \ge 0 \quad \text{for all } i,
    $$
    the Dirichlet distribution with parameter vector
    $$
    \boldsymbol{\alpha} = (\alpha_1, \alpha_2, \dots, \alpha_K) \quad \text{with} \quad \alpha_i > 0,
    $$
    is defined by its probability density function
    $$
    f(x_1, \dots, x_K; \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{i=1}^K x_i^{\alpha_i - 1},
    $$
    where the normalizing constant $B(\boldsymbol{\alpha})$ is the multivariate Beta function:
    $$
    B(\boldsymbol{\alpha}) = \frac{\prod_{i=1}^K \Gamma(\alpha_i)}{\Gamma\left(\sum_{i=1}^K \alpha_i\right)}.
    $$

    ### Derivation from the Beta Distribution

    For the special case $K=2$, let
    $$
    x_1 = x \quad \text{and} \quad x_2 = 1 - x.
    $$
    Then the Dirichlet density becomes
    $$
    f(x; \alpha_1, \alpha_2) = \frac{1}{B(\alpha_1, \alpha_2)} x^{\alpha_1 - 1} (1 - x)^{\alpha_2 - 1},
    $$
    with
    $$
    B(\alpha_1, \alpha_2) = \frac{\Gamma(\alpha_1)\Gamma(\alpha_2)}{\Gamma(\alpha_1 + \alpha_2)}.
    $$
    This is exactly the density of the Beta distribution, denoted as
    $$
    \operatorname{Beta}(\alpha_1,\alpha_2).
    $$

    ### Symmetric Dirichlet Distribution
    In its most general form, the Dirichlet distribution has a parameter vector $\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \dots, \alpha_K)$, with each $\alpha_i$ playing a role similar to that in the Beta distribution. However, in many practical applications—especially when there is no prior reason to favor one category over another—a **symmetric Dirichlet distribution** is used. In the symmetric case, all parameters are set equal, i.e.,
    $$
    \alpha_1 = \alpha_2 = \cdots = \alpha_K = \alpha.
    $$
    This reduces the number of free parameters from $K$ to one, and the density function simplifies to
    $$
    f(x_1,\dots,x_K; \alpha) = \frac{\Gamma(K\alpha)}{[\Gamma(\alpha)]^K} \prod_{i=1}^K x_i^{\alpha - 1}.
    $$
    Thus, while the general Dirichlet has $K$ parameters, the symmetric case is completely described by a single concentration parameter $\alpha$.

    ### How Does $\alpha$ Affect the Dirichlet Distribution?

    - **Small $\alpha$ ($\alpha \ll 1$):**
      When $\alpha$ is much less than 1, the distribution is *sparse*, favoring configurations where one or a few weights are near 1 while the others are close to 0. This tends to lead to “hard” clustering where only a few components dominate.

    - **Uniform Case ($\alpha = 1$):**
      For $\alpha = 1$, the Dirichlet distribution is uniform over the simplex, meaning that all weight configurations are equally likely a priori.

    - **Large $\alpha$ ($\alpha \gg 1$):**
      With a large $\alpha$, the distribution becomes more concentrated around the center of the simplex, promoting balanced weights across all components.

    The plots below visualize how different $\alpha$ values affect the Dirichlet density over a ternary plot.
    """)
    return


@app.cell
def _(dynamic_figsize, np, plt, scipy):
    import mpltern  # noqa: F401  (importing it registers matplotlib's 'ternary' projection)

    def dirichlet_pdf(x, y, z, alpha):
        """
        Compute the Dirichlet density for point (x, y, z)
        given parameters alpha = (α1, α2, α3).
        """
        a1, a2, a3 = alpha
        B = scipy.special.gamma(a1) * scipy.special.gamma(a2) * scipy.special.gamma(a3) / scipy.special.gamma(a1 + a2 + a3)
        return x ** (a1 - 1) * y ** (a2 - 1) * z ** (a3 - 1) / B
    alpha_list = [((2, 2, 2), 'Dirichlet Density (α=2,2,2)'), ((8, 8, 8), 'Dirichlet Density (α=8,8,8)'), ((1, 1, 2), 'Dirichlet Density (α=1,1,2)'), ((1, 2, 2), 'Dirichlet Density (α=1,2,2)'), ((2, 4, 8), 'Dirichlet Density (α=2,4,8)'), ((1, 1, 1), 'Dirichlet Density (α=1,1,1)')]
    N = 200
    a_vals = np.linspace(0.0001, 0.9999, N)
    b_vals = np.linspace(0.0001, 0.9999, N)
    A, B_mesh = np.meshgrid(a_vals, b_vals)
    C = 1 - A - B_mesh
    mask = C > 0
    x_grid = A[mask]
    y_grid = B_mesh[mask]
    z_grid = C[mask]
    _nrows, _ncols = (3, 2)
    _fig, _axes = plt.subplots(nrows=_nrows, ncols=_ncols, subplot_kw={'projection': 'ternary'}, figsize=dynamic_figsize(_nrows, _ncols))
    for _ax, (_alpha, _title) in zip(_axes.flat, alpha_list):
        densities = dirichlet_pdf(x_grid, y_grid, z_grid, _alpha)
        vertices = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
        _ax.plot([v[0] for v in vertices], [v[1] for v in vertices], [v[2] for v in vertices], color='k', lw=2)
        triang = _ax.tricontourf(x_grid, y_grid, z_grid, densities, levels=15, cmap='viridis')
        _ax.tricontour(x_grid, y_grid, z_grid, densities, levels=triang.levels, colors='k', linewidths=0.5)
        _ax.grid(False)
        _ax.set_title(_title)
    plt.suptitle('Different Dirichlet Distributions', y=1.05)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weight Priors in GMM: Small vs. Large $\alpha$

    Weight priors directly influence the mixing proportions of the GMM. Here, we compare the effects of setting a very small versus a very large $\alpha$ value:

    - **Small $\alpha$ ($<1$):** removes about $1-\alpha_k$ pseudo-observations from each component. That only matters for sparsely populated components, so with well-populated components like these (hundreds of points each) the weights stay at the maximum-likelihood values; the prior sparsifies a fit only where a component is already close to empty.
    - **Large $\alpha$:** pulls the weights toward equal values whatever the data say, so all components end up with roughly the same weight.

    With a Dirichlet prior $\mathrm{Dir}(\alpha_1,\dots,\alpha_K)$ (`weight_concentration_prior`), tgmm's M-step replaces the maximum-likelihood weights $\pi_k = n_k / n$ by the MAP estimate

    $$
    \pi_k = \frac{n_k + \alpha_k - 1}{n + \sum_j \alpha_j - K}.
    $$

    Each $\alpha_k$ therefore acts like $\alpha_k - 1$ extra pseudo-observations assigned to component $k$. For $\alpha_k < 1$ the numerator can become negative for sparsely populated components; tgmm clamps the weights to a tiny positive floor so they stay valid.

    Each subplot shows the fitted GMM components (ellipses and means) along with the log-likelihood (LL) achieved.
    """)
    return


@app.cell
def _(
    GaussianMixture,
    MAX_ITER,
    SEED,
    X,
    X_tensor,
    device,
    dynamic_figsize,
    n_components,
    n_features,
    np,
    plot_gmm,
    plt,
    torch,
):
    weigths = np.array([0.0001, 10000.0])
    _weights = [torch.tensor([weight], device=device) for weight in weigths]
    _nrows, _ncols = (2, 1)
    _figsize = dynamic_figsize(_nrows, _ncols)
    _fig, _axs = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=_figsize)
    _fig.suptitle('Different Priors: Small vs Large Weight')
    for _ax, _weight in zip(_axs, weigths):
        _gmm_weight_prior = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=MAX_ITER, device=device, weight_concentration_prior=_weight, random_state=SEED, init_means='random')
        _gmm_weight_prior.fit(X_tensor)
        plot_gmm(X=X, gmm=_gmm_weight_prior, ax=_ax, title=f'alpha={_weight.item()}\nLL={_gmm_weight_prior.lower_bound_:.4f}', scale_alpha_by_weight=True, ellipse_line_style='dashed', ellipse_alpha=0.7, scale_size_by_weight=False, ellipse_colors='yellow', ellipse_std_devs=[2], mean_color='red')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unbalanced Weight Priors: Dominant Component Effect

    In this example, we apply unbalanced weight priors by assigning one component a significantly higher concentration parameter than the others. As expected, the dominant components capture a larger share of the data than they would without the prior (the ellipses are drawn with transparency scaled by the fitted weight). The log-likelihood in each title is lower than for the unconstrained fit, because the prior pushes the weights away from the data's own proportions, but it does not simply fall as more components become dominant: which local maximum EM reaches matters as well.
    """)
    return


@app.cell
def _(
    GaussianMixture,
    MAX_ITER,
    SEED,
    X,
    X_tensor,
    device,
    dynamic_figsize,
    n_components,
    n_features,
    plot_gmm,
    plt,
    torch,
):
    # Example: One dominant prior and three nearly zero priors.
    # Here, we expect the first component to dominate.
    weight1 = torch.tensor([1000.0, 1, 1, 1], device=device)
    weight2 = torch.tensor([1000.0, 1000.0, 1, 1], device=device)
    weight3 = torch.tensor([1000.0, 1000.0, 1000.0, 1], device=device)
    _weights = [weight1, weight2, weight3]
    _nrows, _ncols = (len(_weights), 1)
    _figsize = dynamic_figsize(_nrows, _ncols)
    _fig, _axs = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=_figsize)
    _fig.suptitle('Different Priors: Unbalanced Weights')
    for _ax, _weight in zip(_axs, _weights):
        _gmm_weight_prior = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=MAX_ITER, device=device, weight_concentration_prior=_weight, random_state=SEED, init_means='random')
        _gmm_weight_prior.fit(X_tensor)
        plot_gmm(X=X, gmm=_gmm_weight_prior, ax=_ax, title=f'alpha={_weight.tolist()}\nLL={_gmm_weight_prior.lower_bound_:.4f}', scale_alpha_by_weight=True, scale_size_by_weight=False, ellipse_colors='yellow', ellipse_std_devs=[2], ellipse_line_style='dashed', ellipse_alpha=0.7, mean_color='red')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior on the Means in Gaussian Mixture Models

    A **Gaussian prior** on the component means regularizes the model and guides Maximum a Posteriori (MAP) estimation. It keeps the estimated means from drifting too far from a reasonable location, which is particularly useful when the data are noisy or scarce.

    ### Gaussian Prior on the Means

    tgmm uses the form of the mean prior that is conjugate to a Gaussian with covariance $\Sigma_k$:

    $$
    p(\mu_k \mid \Sigma_k) = \mathcal{N}\!\left(\mu_k;\ \mu_0,\ \tfrac{1}{\kappa_0}\,\Sigma_k\right),
    $$

    where:
    - $\mu_0$ is the **prior mean** (`mean_prior`, often set from a K-means initialization),
    - $\kappa_0$ is the **mean precision** (`mean_precision_prior`).

    Written this way, the prior's spread scales with the component's own covariance and $\kappa_0$ acts as a number of *pseudo-observations* sitting at $\mu_0$. The MAP M-step update for the mean is

    $$
    \mu_k = \frac{\sum_i r_{ik}\,x_i + \kappa_0\,\mu_0}{n_k + \kappa_0}
          = \frac{n_k\,\bar y_k + \kappa_0\,\mu_0}{n_k + \kappa_0},
    $$

    a weighted average of the data mean $\bar y_k$ (weight $n_k$) and the prior mean $\mu_0$ (weight $\kappa_0$).

    With a mean prior *only*, the covariance update stays maximum likelihood, but its scatter is taken about the *shrunk* mean $\mu_k$, not about $\bar y_k$. By the parallel-axis identity $\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top=W_k+n_k(\bar y_k-\mu_k)(\bar y_k-\mu_k)^\top$, so a component whose mean is pulled away from its data gets a slightly wider covariance.

    ### Impact of the Mean Precision Prior

    - **Weak prior (small $\kappa_0$).** The prior has little influence and the means are determined mostly by the data. This is flexible but can overfit when data are noisy.
    - **Strong prior (large $\kappa_0$).** The estimated means are pulled strongly toward $\mu_0$. This helps with limited data or when you have prior knowledge about where the clusters sit, but a very strong prior can prevent the model from adapting to the data.
    """)
    return


@app.cell
def _(
    GMMInitializer,
    X_tensor,
    device,
    dynamic_figsize,
    n_components,
    np,
    plt,
):
    # Visualize the mean prior distributions as 1D Gaussians
    _precision_values = [1, 10, 1000]
    _init_means_kmeans = GMMInitializer.kmeans(X_tensor.cpu(), k=n_components)
    # Create the initial means using K-means on the CPU
    _mean_prior = _init_means_kmeans.clone().to(device)
    x_range = np.linspace(-3, 5, 1000)
    _nrows, _ncols = (len(_precision_values), 1)
    # Create a range for plotting the Gaussian distributions
    _figsize = dynamic_figsize(_nrows, _ncols)
    _fig, _axs = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=_figsize)
    _fig.suptitle('Mean Prior Distributions (1D Visualization)', y=1)
    colors = ['red', 'blue', 'green', 'orange']
    component_labels = [f'Component {i + 1}' for i in range(n_components)]
    for _ax, _precision in zip(_axs, _precision_values):
        prior_std = 1.0 / np.sqrt(_precision)
        for _i in range(n_components):
            mean_val = _mean_prior[_i, 0].cpu().numpy()
            gaussian_pdf = 1 / (prior_std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_range - mean_val) / prior_std) ** 2)
            _ax.plot(x_range, gaussian_pdf, color=colors[_i], linewidth=2, label=f'{component_labels[_i]}')
            _ax.axvline(mean_val, color=colors[_i], linestyle='--', alpha=0.7)  # Standard deviation of the prior (inverse of precision)
        _ax.set_title(f'Precision λ = {_precision} (σ = {prior_std:.4f})')
        _ax.set_xlabel('Mean Value (First Dimension)')
        _ax.set_ylabel('Prior Density')  # Plot Gaussian distributions for each component's mean prior
        _ax.legend(loc='upper right')
        _ax.grid(True, alpha=0.3)  # Use the first dimension of the mean for visualization
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(
    GMMInitializer,
    GaussianMixture,
    MAX_ITER,
    SEED,
    X,
    X_tensor,
    device,
    dynamic_figsize,
    n_components,
    n_features,
    plot_gmm,
    plt,
):
    # Define a list of precision values to compare
    _precision_values = [1, 100.0, 1000.0]
    _init_means_kmeans = GMMInitializer.kmeans(X_tensor.cpu(), k=n_components)
    # Create the initial means using K-means on the CPU
    _mean_prior = _init_means_kmeans.clone().to(device)
    _nrows, _ncols = (len(_precision_values), 1)
    _figsize = dynamic_figsize(_nrows, _ncols)
    _fig, _axs = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=_figsize)
    for _ax, _precision in zip(_axs, _precision_values):
        gmm_mean_prior = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=MAX_ITER, device=device, mean_prior=_mean_prior, mean_precision_prior=_precision, random_state=SEED, init_means=_mean_prior)
        gmm_mean_prior.fit(X_tensor)
        _title = f'Mean Prior (precision={_precision})\nLL={gmm_mean_prior.lower_bound_:.2f}'
        plot_gmm(X=X, gmm=gmm_mean_prior, ax=_ax, title=_title, ellipse_alpha=0.5, ellipse_line_style='dashed', show_initial_means=True, ellipse_colors='blue', ellipse_std_devs=[2], mean_color='yellow', mean_size=50, initial_mean_size=50, scale_alpha_by_weight=True, scale_size_by_weight=False)  # Create a new GMM instance for each precision value
        _ax.legend()
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Gamma Distribution and its Inverse

    The Gamma distribution is given by the density function

    $$
    f(x; \alpha, \theta) = \frac{1}{\Gamma(\alpha)\,\theta^\alpha}\, x^{\alpha-1} \exp\left(-\frac{x}{\theta}\right), \quad x > 0,
    $$

    where:
    - $\alpha$ is the **shape parameter**,
    - $\theta$ is the **scale parameter**,
    - $\Gamma(\alpha)$ is the gamma function.

    The mean of the Gamma distribution is

    $$
    E[X] = \alpha \theta.
    $$

    If we define

    $$
    Y = \frac{1}{X},
    $$

    then using the standard transformation method the density of $Y$ becomes

    $$
    \begin{aligned}
    f_Y(y) &= f_X\left(\frac{1}{y}\right) \left|\frac{d}{dy}\left(\frac{1}{y}\right)\right| \\
    &= \frac{1}{\Gamma(\alpha)\,\theta^\alpha} \left(\frac{1}{y}\right)^{\alpha-1} \exp\left(-\frac{1}{\theta y}\right) \frac{1}{y^2} \\
    &= \frac{1}{\Gamma(\alpha)\,\theta^\alpha}\, y^{-\alpha-1} \exp\left(-\frac{1}{\theta y}\right), \quad y > 0.
    \end{aligned}
    $$

    This density is of the form of an inverse Gamma distribution. A standard parameterization for the inverse Gamma distribution is:

    $$
    f_Y(y; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}\, y^{-\alpha-1} \exp\left(-\frac{\beta}{y}\right), \quad y > 0,
    $$

    where $\beta > 0$ is the **scale parameter** of the inverse Gamma. By comparing the two forms, we identify

    $$
    \beta^\alpha = \frac{1}{\theta^\alpha} \quad \Longrightarrow \quad \beta = \frac{1}{\theta}.
    $$

    ### Mean of the Inverse Gamma Distribution

    For the inverse Gamma distribution with parameters $\alpha$ and $\beta$, the mean exists for $\alpha > 1$ and is given by

    $$
    E[Y] = \frac{\beta}{\alpha - 1}.
    $$

    Substituting $\beta = \frac{1}{\theta}$ into the formula, we get

    $$
    E[Y] = \frac{1/\theta}{\alpha - 1} = \frac{1}{\theta (\alpha - 1)}.
    $$

    ### Impact of the Parameters

    - **Shape ($\alpha$)**:
      The shape parameter $\alpha$ affects the tail behavior and the concentration of the distribution. Increasing $\alpha$ (with $\beta$ fixed) causes both the mean and the mode to decrease:

      $$
      E[Y] = \frac{\beta}{\alpha - 1}, \qquad \text{mode} = \frac{\beta}{\alpha + 1}.
      $$

      Thus, a higher $\alpha$ results in a distribution that is more concentrated near zero and has lighter tails, while a lower $\alpha$ produces heavier tails.

    - **Scale ($\beta$)**:
      The scale parameter $\beta$ shifts the distribution. A larger $\beta$ leads to higher values of the mean and mode. In many Bayesian contexts, the inverse gamma is obtained by inverting a Gamma-distributed variable with scale parameter $\theta$, where $\beta = \frac{1}{\theta}$. In that case, increasing $\theta$ (making the Gamma distribution more spread out) results in a smaller $\beta$, which in turn shifts the inverse gamma distribution toward lower values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the Inverse Gamma Distribution

    The three panels below demonstrate the following scenarios:

    1. **Varying $\alpha$ with Fixed $\theta$:**
       How changes in the shape parameter alter the spread and concentration of the distribution.

    2. **Fixed $\alpha$ with Varying $\theta$:**
       The effect of the scale parameter on the location and spread of the distribution.

    3. **Varying Both to Maintain the Same Mode:**
       How different combinations of $\alpha$ and $\theta$ can yield the same mode, illustrating the trade-off between the parameters.
    """)
    return


@app.cell
def _(invgamma, np, plt):
    _x = np.linspace(0.001, 2, 1000)
    _fig, _axes = plt.subplots(3, 1, figsize=(10, 18))
    fixed_theta = 1
    alphas = [1, 2, 3, 7]
    # -------------------------------
    # Plot 1: Varying $\alpha$, fixed $\theta$
    for _alpha in alphas:
        beta = 1 / fixed_theta
        pdf_vals = invgamma.pdf(_x, a=_alpha, scale=beta)
        mode = beta / (_alpha + 1)
        line, = _axes[0].plot(_x, pdf_vals, label=f'$\\alpha={_alpha},\\,\\theta={fixed_theta},\\,\\mathrm{{mode}}={mode:.3f}$')
        _axes[0].axvline(mode, linestyle='dashed', color=line.get_color(), linewidth=1.5)
    _axes[0].set_title('Inverse Gamma PDF (Varying $\\alpha$, Fixed $\\theta$)')
    _axes[0].set_xlabel('$x$')
    _axes[0].set_ylabel('Density')
    _axes[0].legend()
    _axes[0].grid(True)
    fixed_alpha = 3
    thetas = [0.25, 0.5, 1, 2]
    for theta in thetas:
        beta = 1 / theta
        pdf_vals = invgamma.pdf(_x, a=fixed_alpha, scale=beta)
        mode = beta / (fixed_alpha + 1)
        line, = _axes[1].plot(_x, pdf_vals, label=f'$\\alpha={fixed_alpha},\\,\\theta={theta},\\,\\mathrm{{mode}}={mode:.3f}$')
        _axes[1].axvline(mode, linestyle='dashed', color=line.get_color(), linewidth=1.5)
    # Plot 2: Fixed $\alpha$, varying $\theta$
    _axes[1].set_title('Inverse Gamma PDF (Fixed $\\alpha$, Varying $\\theta$)')
    _axes[1].set_xlabel('$x$')
    _axes[1].set_ylabel('Density')
    _axes[1].legend()
    _axes[1].grid(True)
    alphas_equal = [1, 3, 7, 15]
    thetas_equal = [1, 0.5, 0.25, 0.125]
    for _alpha, theta in zip(alphas_equal, thetas_equal):
        beta = 1 / theta
        pdf_vals = invgamma.pdf(_x, a=_alpha, scale=beta)
        mode = beta / (_alpha + 1)
        line, = _axes[2].plot(_x, pdf_vals, label=f'$\\alpha={_alpha},\\,\\theta={theta},\\,\\mathrm{{mode}}={mode:.3f}$')
        _axes[2].axvline(mode, linestyle='dashed', color=line.get_color(), linewidth=1.5)
    _axes[2].set_title('Inverse Gamma PDF (Varying both $\\alpha$ and $\\theta$ to get the same mode)')
    _axes[2].set_xlabel('$x$')
    _axes[2].set_ylabel('Density')
    _axes[2].legend()
    _axes[2].grid(True)
    plt.tight_layout()
    # Plot 3: Varying both $\alpha$ and $\theta$ to get the same mode
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Wishart Distribution and its Inverse

    In Bayesian modeling of Gaussian mixtures, a key step is placing a prior on the covariance matrices of each Gaussian component. Since these matrices must be positive-definite (i.e., all eigenvalues are positive), the prior must respect this constraint. While the (non-inverse) Wishart distribution is a natural multivariate analogue of the Gamma distribution, it has characteristics that are not ideal for covariance priors in a MAP estimation context. Instead, the **inverse Wishart** (and its one-dimensional counterpart, the **inverse Gamma**) is commonly used. Let’s explore why.

    ### From the Gamma to the Wishart Distribution

    #### The Gamma Distribution (Scalar Case)

    The Gamma distribution is defined as

    $$
    f(x; \alpha, \theta) = \frac{1}{\Gamma(\alpha)\,\theta^\alpha}\, x^{\alpha-1} \exp\left(-\frac{x}{\theta}\right), \quad x > 0,
    $$

    where:
    - $ \alpha $ is the **shape parameter**,
    - $ \theta $ is the **scale parameter**, and
    - $ \Gamma(\alpha) $ is the Gamma function.

    For many Bayesian models, the Gamma distribution is used as a conjugate prior for positive parameters (e.g. precision). Notice that—even though certain parameter choices can put high density near zero—its support is strictly $ (0, \infty) $, so the probability of $ x=0 $ is exactly zero.

    #### The Wishart Distribution (Multivariate Case)

    When we generalize to matrices, the Wishart distribution arises naturally. Its density over $ d \times d $ positive-definite matrices is

    $$
    p(\mathbf{S}; \nu, \Psi) \propto |\mathbf{S}|^{\frac{\nu-d-1}{2}} \exp\left(-\frac{1}{2}\operatorname{tr}\left(\Psi^{-1} \mathbf{S}\right)\right),
    $$

    where:
    - $ \mathbf{S} $ is a positive-definite matrix,
    - $ \nu $ is the **degrees of freedom** (playing a role similar to the shape parameter),
    - $ \Psi $ is the **scale matrix**, and
    - $ d $ is the dimension.

    For $ d = 1 $, the Wishart reduces to the Gamma distribution (up to constants), which shows their close connection.

    ### Why Use the Inverse Wishart and Inverse Gamma?

    There are two main reasons for preferring the inverse distributions as priors on covariance matrices in a GMM:

    **The Density Vanishes at Zero**

    A Gamma density with shape $\alpha \le 1$ (or a Wishart density with $\nu \le d+1$) does not vanish near zero -- it can even diverge there -- so a Gamma or Wishart prior placed on a variance or covariance can favor degenerate, near-zero values. The inverse Gamma and inverse Wishart densities go to zero as $y \to 0$ (exponentially fast, through the $\exp(-\beta/y)$ factor; for matrices, as $\Sigma$ approaches singularity). The prior itself therefore rules out collapsed components, which is crucial for well-defined Gaussian likelihoods.

    **Longer Right-Hand Tail**

    The inverse distributions typically have a heavier (longer) tail on the right-hand side. This longer tail means that while the prior penalizes extremely small covariances (preventing singularity), it is more tolerant of large covariance estimates.

    For covariance matrices, this is important because the empirical covariance can vary widely—especially in high-dimensional settings. The long tail ensures that the prior does not overly constrain large values, thus preserving flexibility. In 2D, for example, the determinant of the covariance matrix (which is proportional to the area of the corresponding ellipse) can be thought of as capturing the “size” of the uncertainty. A heavy right tail in the inverse Wishart better accommodates the possibility of large uncertainty, similar in spirit (though not strictly proportional) to how the surface area of a circle increases with its radius.
    """)
    return


@app.cell
def _(dynamic_figsize, gaussian_filter, np, plt, wishart):
    covariance_matrix = np.array([[1, 0.5], [0.5, 1]])
    dfs = [2, 3, 4]
    scales = [0.5, 1, 2]
    scale_matrices = [covariance_matrix * scale for scale in scales]
    inv_wishart_samples_list = []
    for df in dfs:
        for scale_matrix in scale_matrices:
            _samples = wishart.rvs(df=df, scale=scale_matrix, size=1000000)
            inv_samples = np.linalg.inv(_samples)
            inv_wishart_samples_list.append((df, scale_matrix, inv_samples))
    xmin, xmax = (-0.01, 2)
    ymin, ymax = (-0.01, 2)
    _nrows, _ncols = (len(dfs), len(scale_matrices))
    _figsize = dynamic_figsize(_nrows, _ncols, base_width=7, base_height=6)
    _fig, _axes = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=_figsize)
    _fig.suptitle('Hexbin Plots with Smoothed Gradient Contours\nof 2D Inverse Wishart Distributions', fontsize=16)
    idx = 0
    for _i, df in enumerate(dfs):
        for _j, scale_matrix in enumerate(scale_matrices):
            df_current, scale_matrix_current, inv_samples = inv_wishart_samples_list[idx]
            idx = idx + 1
            _x = inv_samples[:, 0, 0]
            y = inv_samples[:, 1, 1]
            _ax = _axes[_i, _j]
            hb = _ax.hexbin(_x, y, gridsize=30, cmap='PiYG', extent=[xmin, xmax, ymin, ymax])
            _fig.colorbar(hb, ax=_ax, orientation='vertical')
            H, xedges, yedges = np.histogram2d(_x, y, bins=20, range=[[xmin, xmax], [ymin, ymax]])
            H_smooth = gaussian_filter(H, sigma=1)
            Xc = (xedges[:-1] + xedges[1:]) / 2
            Yc = (yedges[:-1] + yedges[1:]) / 2
            X_mesh, Y_mesh = np.meshgrid(Xc, Yc)
            _ax.contour(X_mesh, Y_mesh, H_smooth.T, colors='k', linewidths=1, alpha=1, levels=10)
            _ax.set_aspect('equal')
            _ax.set_xlim([xmin, xmax])
            _ax.set_ylim([ymin, ymax])
            _ax.set_title(f'DF={df}, Scale={np.round(scale_matrix[0, 0], 1)}')
            _ax.set_xlabel('Diagonal Element 1')
            _ax.set_ylabel('Diagonal Element 2')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Covariance Priors in GMM

    Next, we demonstrate how covariance priors affect the GMM. In the following three sections, we vary:

    1. **Prior Strength:** The covariance prior is scaled by different factors.
    2. **Degrees of Freedom (DOF):** The DOF of the prior distribution is varied.
    3. **Strength and DOF:** Varying both the prior strength and the DOF.

    ### 1. Varying Prior Strength

    Here, the DOF is fixed and the covariance prior strength is varied.
    ### How the covariance prior enters the M-step

    With an inverse-Wishart prior $\Sigma_k \sim \mathcal{IW}(\Psi_0, \nu_0)$ (`covariance_prior` $=\Psi_0$, `degrees_of_freedom_prior` $=\nu_0$) and **no** mean prior, tgmm's MAP covariance update for `covariance_type='full'` is

    $$
    \Sigma_k = \frac{\Psi_0 + W_k + \varepsilon I}{\nu_0 + n_k},
    $$

    where $W_k$ is the scatter about the (unshrunk) mean and $\varepsilon$ is `reg_covar`. Read it as a weighted average: the prior contributes $\nu_0$ pseudo-observations with total scatter $\Psi_0$, so the prior's best guess for $\Sigma_k$ is $\Psi_0/\nu_0$. To encode a prior covariance guess $\Sigma^{\ast}$ with strength $\nu_0$, pass `covariance_prior`$=\nu_0\,\Sigma^{\ast}$.

    The other covariance types use the same recipe in structured form (`covariance_prior` has the shape of `covariances_`; $\nu_0$ is a float):

    | `covariance_type` | $\Sigma$ |
    | --- | --- |
    | `full` | $(\Psi_0+W_k+\varepsilon I)/(\nu_0+n_k)$ |
    | `diag` | per coordinate $j$: $(\psi_{0,j}+(W_k)_{jj}+\varepsilon)/(\nu_0+n_k)$ |
    | `spherical` | $\sigma_k^2=(\psi_0+\operatorname{tr}W_k+\varepsilon)/\big((\nu_0+n_k)\,d\big)$ |
    | `tied_full`, `tied_diag` | the same with $W_k$ summed over $k$ and $\nu_0+\sum_kn_k$ in the denominator |
    | `tied_spherical` | $(\psi_0+\sum_k\operatorname{tr}W_k+\varepsilon)/\big((\nu_0+\sum_kn_k)\,d\big)$ |

    The cells below pass only `covariance_prior` and `degrees_of_freedom_prior`, so they exercise exactly this update.

    !!! note
        The denominator is $\nu_0 + n_k$, not the textbook inverse-Wishart posterior mode $\nu_0 + n_k + d + 1$ (nor the mean's $\nu_0 + n_k - d - 1$). tgmm deliberately uses the simple ratio $\Psi_n/\nu_n$: unlike the textbook formulas it stays finite and positive for every $\nu_n > 0$, and the two agree closely when $\nu_n \gg d$.
    """)
    return


@app.cell
def _(
    GaussianMixture,
    MAX_ITER,
    X,
    X_tensor,
    device,
    dynamic_figsize,
    n_components,
    n_features,
    np,
    plot_gmm,
    plt,
    torch,
):
    _data_covariance = np.cov(X_tensor.cpu().numpy(), rowvar=False)  # shape (2, 2)
    _data_covariance = torch.tensor(_data_covariance, dtype=torch.float32, device=device)
    _data_covariance = _data_covariance.unsqueeze(0).expand(n_components, -1, -1)
    degrees_of_freedom_prior = float(n_features + 2)
    _prior_strengths = [1, 5, 25, 100]
    _nrows, _ncols = (2, 2)
    _figsize = dynamic_figsize(_nrows, _ncols, base_width=5, base_height=4)
    _fig, _axs = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=_figsize, dpi=50)
    _fig.suptitle('Covariance Priors: Varying Strength')
    for _ax, _strength in zip(_axs.ravel(), _prior_strengths):
        _cov_prior = _data_covariance * _strength
        _gmm_cov_prior = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=10 * MAX_ITER, device=device, covariance_prior=_cov_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
        _gmm_cov_prior.fit(X_tensor)
        _title = f'DOF={degrees_of_freedom_prior}, Strength={_strength}\nLL={_gmm_cov_prior.lower_bound_:.2f}'
        plot_gmm(X=X, gmm=_gmm_cov_prior, ax=_ax, title=_title, ellipse_colors='green', ellipse_std_devs=[1, 2, 3])
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Varying Degrees of Freedom (DOF)

    Here, the covariance prior strength is fixed and we vary the DOF.

    Note: For 2D, DOF must be greater than 1.
    """)
    return


@app.cell
def _(
    GaussianMixture,
    MAX_ITER,
    X,
    X_tensor,
    device,
    dynamic_figsize,
    n_components,
    n_features,
    np,
    plot_gmm,
    plt,
    torch,
):
    _data_covariance = np.cov(X_tensor.cpu().numpy(), rowvar=False)  # shape (2, 2)
    _data_covariance = torch.tensor(_data_covariance, dtype=torch.float32, device=device)
    _data_covariance = _data_covariance.unsqueeze(0).expand(n_components, -1, -1)
    _dof_values = [n_features + 8, n_features + 98, n_features + 498, n_features + 1998]
    # Instead of varying strength, we fix the strength and vary DOF.
    fixed_strength = 5
    _nrows, _ncols = (2, 2)
    _figsize = dynamic_figsize(_nrows, _ncols, base_width=5, base_height=4)
    _fig, _axs = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=_figsize, dpi=50)
    _fig.suptitle('Covariance Priors: Varying DOF')
    for _ax, _dof in zip(_axs.ravel(), _dof_values):
        _cov_prior = _data_covariance * fixed_strength
        _gmm_cov_prior = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=10 * MAX_ITER, device=device, covariance_prior=_cov_prior, degrees_of_freedom_prior=_dof)
        _gmm_cov_prior.fit(X_tensor)
        _title = f'DOF={_dof}, Strength={fixed_strength}\nLL={_gmm_cov_prior.lower_bound_:.2f}'
        plot_gmm(X=X, gmm=_gmm_cov_prior, ax=_ax, title=_title, ellipse_colors='red', ellipse_std_devs=[1, 2, 3])
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Varying Strength and Degrees of Freedom

    Here, both the covariance prior strength and the DOF are varied.
    """)
    return


@app.cell
def _(
    GaussianMixture,
    MAX_ITER,
    X,
    X_tensor,
    device,
    dynamic_figsize,
    n_components,
    n_features,
    np,
    plot_gmm,
    plt,
    torch,
):
    # Compute data covariance from the tensor data.
    _data_covariance = np.cov(X_tensor.cpu().numpy(), rowvar=False)  # shape (2, 2)
    _data_covariance = torch.tensor(_data_covariance, dtype=torch.float32, device=device)
    _data_covariance = _data_covariance.unsqueeze(0).expand(n_components, -1, -1)
    _prior_strengths = [25, 10, 5]
    # Define the lists of prior strengths and degrees of freedom to explore.
    _dof_values = [n_features + 8, n_features + 98, n_features + 498]
    _nrows, _ncols = (len(_dof_values), len(_prior_strengths))
    _figsize = dynamic_figsize(_nrows, _ncols, base_width=6, base_height=6)
    # Create a 3x3 grid.
    _fig, _axs = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=_figsize)
    _fig.suptitle('Covariance Priors: Varying Strength and DOF')
    for _i, _dof in enumerate(_dof_values):
        for _j, _strength in enumerate(_prior_strengths):
            _ax = _axs[_i, _j]
    # Loop over DOF values (rows) and prior strengths (columns)
            _cov_prior = _data_covariance * _strength
            _gmm_cov_prior = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=10 * MAX_ITER, device=device, covariance_prior=_cov_prior, degrees_of_freedom_prior=_dof)  # Compute covariance prior for this combination.
            _gmm_cov_prior.fit(X_tensor)
            _title = f'DOF={_dof}, Strength={_strength}\nLL={_gmm_cov_prior.lower_bound_:.2f}'
            plot_gmm(X=X, gmm=_gmm_cov_prior, ax=_ax, title=_title, ellipse_colors='orange', ellipse_std_devs=[1, 2, 3])
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All Three Priors Together

    Give a weight prior, a mean prior and a covariance prior at once and `tgmm` applies the Dirichlet MAP update to the weights and the *joint* Normal-Inverse-Wishart update to means and covariances (the mean and covariance priors are not applied independently once both are present; the NIW section below explains the difference).

    To see what that does, the cell below fits the default blobs twice from the same k-means starting centres, so that any difference comes from the priors alone: by maximum likelihood (no priors) and with a prior worth **50 pseudo-points per component** for each ingredient:

    - **weights:** `weight_concentration_prior` $\alpha_k=50$ for every component pulls the mixing weights toward equal values;
    - **means:** `mean_prior` $\mu_0$ = the overall data mean with `mean_precision_prior` $\kappa_0=50$ pulls every mean toward the centre, by the fraction $\kappa_0/(\kappa_0+n_k)$;
    - **covariances:** `covariance_prior` $\Psi_0=50\,I$ with `degrees_of_freedom_prior` $\nu_0=50$ says "a component's covariance is probably about $\Psi_0/\nu_0=I$, and I trust that as much as 50 observations".
    """)
    return


@app.cell
def _(
    GaussianMixture,
    MAX_ITER,
    SEED,
    X,
    X_tensor,
    device,
    dynamic_figsize,
    n_components,
    n_features,
    plot_gmm,
    plt,
    torch,
):
    # Both fits start from the same k-means centres, so the differences come from the priors alone
    _common = dict(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=10 * MAX_ITER, tol=1e-9, device=device, random_state=SEED, init_means='kmeans')
    gmm_mle = GaussianMixture(**_common).fit(X_tensor)
    gmm_map = GaussianMixture(
        **_common,
        weight_concentration_prior=torch.full((n_components,), 50.0, device=device),                                          # alpha_k = 50
        mean_prior=X_tensor.mean(0, keepdim=True).repeat(n_components, 1), mean_precision_prior=50.0,                          # mu_0 = data mean, kappa_0 = 50
        covariance_prior=50.0 * torch.eye(n_features, device=device).unsqueeze(0).repeat(n_components, 1, 1), degrees_of_freedom_prior=50.0,  # Psi_0 = 50 I, nu_0 = 50
    ).fit(X_tensor)
    print(f'MLE: converged={gmm_mle.converged_}, mean log-likelihood = {gmm_mle.score(X_tensor):.4f}')
    print(f'MAP: converged={gmm_map.converged_}, mean log-likelihood = {gmm_map.score(X_tensor):.4f}')
    _fig, _axs = plt.subplots(1, 2, figsize=dynamic_figsize(1, 2, base_width=5.5, base_height=4.5), dpi=50)
    for _ax, _gmm, _name in zip(_axs, (gmm_mle, gmm_map), ('MLE (no priors)', 'MAP (weight + NIW priors)')):
        plot_gmm(X, gmm=_gmm, ax=_ax, title=f'{_name}\nmean log-lik = {_gmm.score(X_tensor):.3f}', legend=False)
    plt.tight_layout()
    plt.gcf()
    return gmm_mle, gmm_map


@app.cell(hide_code=True)
def _(X_tensor, gmm_map, gmm_mle, mo):
    _w0, _w1 = gmm_mle.weights_.cpu().numpy(), gmm_map.weights_.cpu().numpy()
    _pull = 50.0 / (50.0 + gmm_mle.weights_.cpu().numpy() * X_tensor.shape[0])  # share of the prior in the mean update, from the MLE sizes
    _rows = '\n'.join(f'| {_k + 1} | {_w0[_k]:.3f} | {_w1[_k]:.3f} | {100 * _pull[_k]:.0f}% |' for _k in range(len(_w0)))
    mo.md(f"""
    | component | MLE weight | MAP weight | share of the prior in the mean update $\\kappa_0/(\\kappa_0+n_k)$ |
    |---|---|---|---|
    {_rows}

    The MAP fit has a *lower* log-likelihood on the training data ({gmm_map.score(X_tensor):.3f} against {gmm_mle.score(X_tensor):.3f}). That is by construction: MAP maximises likelihood times prior, not the likelihood, and the prior here is deliberately not the truth (equal weights, one shared centre, unit covariances). What it buys is regularisation, and how much that matters depends on how much data there is: the next sections quantify it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Normal-Inverse-Wishart Prior

    The mean prior $\mathcal N(\mu_0,\Sigma_k/\kappa_0)$ is written in terms of the component covariance $\Sigma_k$, and the covariance prior is an inverse Wishart on that same $\Sigma_k$. Together they form the **Normal-Inverse-Wishart (NIW)** prior, the conjugate prior for a Gaussian whose mean *and* covariance are both unknown:

    $$
    p(\mu,\Sigma)=\mathcal N\!\left(\mu\,\middle|\,\mu_0,\tfrac1{\kappa_0}\Sigma\right)\cdot\mathcal{IW}(\Sigma\mid\Psi_0,\nu_0),
    $$

    with $\mu_0$ = `mean_prior`, $\kappa_0$ = `mean_precision_prior`, $\Psi_0$ = `covariance_prior` and $\nu_0$ = `degrees_of_freedom_prior`. With **both** a mean prior and a covariance prior given, `GaussianMixture` uses the closed-form NIW posterior in the M-step, instead of applying the two priors independently.

    **Properties.**

    1. *Conjugacy:* the posterior is again NIW, so the update is analytic.
    2. *Coupling:* the prior covariance of $\mu$ is $\Sigma/\kappa_0$, so a broad component automatically has a less informative mean prior.
    3. *Pseudo-observations:* $\kappa_0$ counts pseudo-observations for the mean and $\nu_0$ for the covariance.

    **Posterior update.** Given the effective size $n_k=\sum_ir_{ik}$, the weighted mean $\bar y_k$ and the scatter $W_k$ of component $k$ (see the table at the top):

    $$
    \kappa_n=\kappa_0+n_k,\qquad \nu_n=\nu_0+n_k,\qquad
    \mu_n=\frac{\kappa_0\mu_0+n_k\bar y_k}{\kappa_0+n_k},
    $$

    $$
    \Psi_n=\Psi_0+W_k+\frac{\kappa_0n_k}{\kappa_0+n_k}(\bar y_k-\mu_0)(\bar y_k-\mu_0)^{\top}.
    $$

    tgmm sets $\mu_k=\mu_n$ and $\Sigma_k=\Psi_n/\nu_n$, *not* the textbook inverse-Wishart posterior mean $\Psi_n/(\nu_n-d-1)$, which is undefined once $\nu_n\le d+1$. So $\Psi_0/\nu_0$ is the prior's best guess for the covariance. The last term of $\Psi_n$ is the new ingredient compared with using a covariance prior alone: it grows when the data mean disagrees with $\mu_0$. It vanishes as $n_k\to\infty$ or $\kappa_0\to0$, and with $\kappa_0\to0$, $\Psi_0\to0$ the update reduces to the maximum-likelihood $W_k/n_k$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\Psi_n$ also receives `reg_covar` ($\varepsilon I$). The other covariance types use the same $\Psi_n$ in structured form:

    | `covariance_type` | $\Psi_n$ | $\Sigma$ |
    | --- | --- | --- |
    | `full` | $\Psi_0+W_k+\frac{\kappa_0n_k}{\kappa_n}(\bar y_k-\mu_{0,k})(\bar y_k-\mu_{0,k})^\top+\varepsilon I$ | $\Psi_n/\nu_n$ |
    | `diag` | per coordinate: $\psi_{0,j}+(W_k)_{jj}+\frac{\kappa_0n_k}{\kappa_n}(\bar y_{kj}-\mu_{0,kj})^2+\varepsilon$ | $\Psi_n/\nu_n$ per coordinate |
    | `spherical` | $\psi_0+\operatorname{tr}W_k+\frac{\kappa_0n_k}{\kappa_n}\lVert\bar y_k-\mu_{0,k}\rVert^2+d\,\varepsilon$ | $\sigma_k^2=\Psi_n/(\nu_n\,d)$ |
    | `tied_full` | $\Psi_0+\sum_kW_k+\sum_k\frac{\kappa_0n_k}{\kappa_n}(\bar y_k-\mu_{0,k})(\bar y_k-\mu_{0,k})^\top+\varepsilon I$ | $\Psi_n/(\nu_0+\sum_kn_k)$ |
    | `tied_diag`, `tied_spherical` | the tied analogues (sums over $k$) | $\Psi_n/(\nu_0+\sum_kn_k)$, divided by $d$ for spherical |

    (For `tied_spherical` the code adds $\varepsilon\,d$ once to the trace, as for `spherical`.) The weight prior is independent of all this and is applied on top.

    !!! note "Restrictions"
        The mean/covariance priors (hence NIW) are implemented for the six classic covariance types only; the 8 GPCM-only codes raise `NotImplementedError`, because a conjugate prior would have to be derived per parameterization. `GaussianMixture.suggest_priors(X, n_components, covariance_type)` returns a data-driven starting point `(mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior)`; it is required in Gibbs mode ([11 -- Dirichlet Process Mixtures](11_dpgmm.ipynb)).

    ### NIW versus a single prior

    tgmm has no "independent mean and covariance priors" variant: both together always select the joint NIW update, so "separate" here means supplying *only one* of the two. That coupling is what makes the pair conjugate, and so the update closed-form: the mean prior $\mathcal N(\mu_0,\Sigma_k/\kappa_0)$ is scaled by the component's own covariance, which is also why $\Psi_n$ gains the cross term below. An *independent* Gaussian prior $\mu_k\sim\mathcal N(\mu_0,P_0^{-1})$ would make the mean update depend on $\Sigma_k$, $\mu_k=(n_k\Sigma_k^{-1}+P_0)^{-1}(n_k\Sigma_k^{-1}\bar y_k+P_0\mu_0)$, and is not available.

    - **Mean prior only.** The mean is shrunk, $\mu_k=(n_k\bar y_k+\kappa_0\mu_0)/(n_k+\kappa_0)$, and the covariance stays maximum likelihood, but its scatter is taken about the shrunk mean.
    - **Covariance prior only.** The mean stays maximum likelihood, $\mu_k=\bar y_k$, and $\Sigma_k=(\Psi_0+W_k+\varepsilon I)/(\nu_0+n_k)$.
    - **NIW (both).** Both are updated jointly, and $\Psi_n$ gains the extra term $\frac{\kappa_0n_k}{\kappa_0+n_k}(\bar y_k-\mu_0)(\bar y_k-\mu_0)^\top$.

    The cells below fit all three, and the maximum-likelihood fit for reference, on a small dataset where the differences are visible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data and hyperparameters for the NIW experiments

    The experiments below use a second, smaller dataset (four well-separated corner blobs with 100 points each, drawn from fixed covariances) so that the effect of each prior is easy to see. One shared set of NIW hyperparameters is used for every configuration, for a fair comparison: $\kappa_0=20$, $\nu_0=d+100$, $\mu_0=0$ and $\Psi_0=100\,I$, so the prior's best guess for the covariance is $\Psi_0/\nu_0\approx0.98\,I$ (about unit covariances).
    """)
    return


@app.cell
def _(SEED, device, np, torch):
    niw_true_means = np.array([[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0]])
    niw_true_covs = np.array([[[0.5, 0.2], [0.2, 0.3]], [[0.8, -0.1], [-0.1, 0.4]], [[0.3, 0.0], [0.0, 0.6]], [[0.4, 0.1], [0.1, 0.5]]])
    _rng = np.random.default_rng(SEED)
    niw_X = np.vstack([_rng.multivariate_normal(_m, _c, 100) for _m, _c in zip(niw_true_means, niw_true_covs)])
    niw_X_tensor = torch.tensor(niw_X, dtype=torch.float32, device=device)
    print(f'NIW experiments data: {niw_X.shape[0]} samples with {niw_X.shape[1]} features (100 per component)')

    # Shared NIW hyperparameters
    niw_kappa0 = 20  # mean precision (pseudo-observations for the mean): moderate coupling
    niw_nu0 = niw_X.shape[1] + 100  # degrees of freedom (pseudo-observations for the covariance)
    niw_mu0 = torch.zeros(4, niw_X.shape[1], device=device)  # prior mean at the origin
    niw_Psi0 = torch.eye(niw_X.shape[1], device=device).unsqueeze(0).expand(4, -1, -1) * 100.0  # scale matrix: prior guess Psi0/nu0 ~ unit covariance
    print(f'kappa0={niw_kappa0}, nu0={niw_nu0}, mu0=0, Psi0=100*I  ->  prior guess for the covariance: Psi0/nu0 = {100.0 / niw_nu0:.3f} * I')
    return (
        niw_Psi0,
        niw_X,
        niw_X_tensor,
        niw_kappa0,
        niw_mu0,
        niw_nu0,
        niw_true_covs,
        niw_true_means,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mean prior only, covariance prior only, and NIW

    Four fits of the same data and hyperparameters: no prior (maximum likelihood) for reference, a prior on the means only (covariances maximum likelihood), a prior on the covariances only (means maximum likelihood), and both together (the joint NIW update).
    """)
    return


@app.cell
def _(
    GaussianMixture,
    MAX_ITER,
    device,
    n_components,
    n_features,
    niw_Psi0,
    niw_X,
    niw_X_tensor,
    niw_kappa0,
    niw_mu0,
    niw_nu0,
    plot_gmm,
    plt,
):
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 8), dpi=50)
    _fig.suptitle('Comparison: Different Prior Configurations', fontsize=14)
    _configs = {
        'No Prior (MLE)': dict(),
        'Mean Prior Only': dict(mean_prior=niw_mu0, mean_precision_prior=niw_kappa0),
        'Covariance Prior Only': dict(covariance_prior=niw_Psi0, degrees_of_freedom_prior=niw_nu0),
        'NIW Conjugate Prior': dict(mean_prior=niw_mu0, mean_precision_prior=niw_kappa0, covariance_prior=niw_Psi0, degrees_of_freedom_prior=niw_nu0),
    }
    for _ax, (_name, _kw) in zip(_axes.ravel(), _configs.items()):
        _gmm = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=MAX_ITER, device=device, **_kw)
        _gmm.fit(niw_X_tensor)
        plot_gmm(X=niw_X, gmm=_gmm, ax=_ax, title=f'{_name}\nLL: {_gmm.lower_bound_:.2f}')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sensitivity to the NIW Hyperparameters

    How much does the fit depend on each hyperparameter, and on how much data there is? The function below fits the NIW model over a grid of two hyperparameters at a time (the others at defaults: $\kappa_0=1$, $\nu_0=d+10$, $\mu_0=0$, $\Psi_0=I$) and shows the final log-likelihood as a heatmap, for all six pairs of $\kappa_0$, $\nu_0$, $\Psi_0$ (a scale times the identity) and $\mu_0$ (the corners at $\pm m$). (In the fast docs build, `TGMM_DOCS_FAST=1`, the grids are coarser and the larger datasets smaller.)
    """)
    return


@app.cell
def _(GaussianMixture, np, plt, torch):
    def analyze_niw_hyperparameters(X_tensor, n_components, n_features, device, title_suffix="", fast=False):
        """Run NIW hyperparameter sensitivity analysis on given data (coarser grids when `fast`)."""
        kappa_grid = [0.1, 10.0, 1000] if fast else [0.1, 1.0, 10.0, 100.0, 1000]
        nu_grid = [n_features, 50, 500] if fast else [n_features, 10, 50, 100, 500, 1000]
        psi_scales = [0.1, 10.0, 1000.0] if fast else [0.1, 1.0, 10.0, 100.0, 1000.0]
        m0_scales = [0, 4, 16] if fast else [0, 2, 4, 8, 16]
        signs = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], device=device)
        psi_grid = [torch.eye(n_features, device=device).unsqueeze(0).expand(n_components, -1, -1) * scale for scale in psi_scales]
        m0_grid = [signs * float(m) for m in m0_scales]  # 0 -> all zeros, else corners at +-m

        # Default values
        kappa_default = 1.0
        nu_default = n_features + 10
        m0_default = torch.zeros(n_components, n_features, device=device)
        psi_default = torch.eye(n_features, device=device).unsqueeze(0).expand(n_components, -1, -1) * 1.0

        # The 6 parameter pairs
        param_combinations = [
            {'x': 'nu', 'y': 'kappa', 'x_grid': nu_grid, 'y_grid': kappa_grid, 'x_label': 'ν₀ (Degrees of Freedom)', 'y_label': 'κ₀ (Mean Precision)', 'fixed': {'m0': m0_default, 'psi': psi_default}},
            {'x': 'psi', 'y': 'kappa', 'x_grid': list(range(len(psi_grid))), 'y_grid': kappa_grid, 'x_label': 'Ψ₀ Scale', 'y_label': 'κ₀ (Mean Precision)', 'fixed': {'nu': nu_default, 'm0': m0_default}},
            {'x': 'm0', 'y': 'kappa', 'x_grid': list(range(len(m0_grid))), 'y_grid': kappa_grid, 'x_label': 'μ₀ Configuration', 'y_label': 'κ₀ (Mean Precision)', 'fixed': {'nu': nu_default, 'psi': psi_default}},
            {'x': 'psi', 'y': 'nu', 'x_grid': list(range(len(psi_grid))), 'y_grid': nu_grid, 'x_label': 'Ψ₀ Scale', 'y_label': 'ν₀ (Degrees of Freedom)', 'fixed': {'kappa': kappa_default, 'm0': m0_default}},
            {'x': 'm0', 'y': 'nu', 'x_grid': list(range(len(m0_grid))), 'y_grid': nu_grid, 'x_label': 'μ₀ Configuration', 'y_label': 'ν₀ (Degrees of Freedom)', 'fixed': {'kappa': kappa_default, 'psi': psi_default}},
            {'x': 'm0', 'y': 'psi', 'x_grid': list(range(len(m0_grid))), 'y_grid': list(range(len(psi_grid))), 'x_label': 'μ₀ Configuration', 'y_label': 'Ψ₀ Scale', 'fixed': {'kappa': kappa_default, 'nu': nu_default}},
        ]
        psi_labels = [f'{v:g}' for v in psi_scales]
        m0_labels = ['0' if m == 0 else f'±{m}' for m in m0_scales]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=70)
        axes = axes.flatten()
        for idx, combo in enumerate(param_combinations):
            ll_matrix = np.zeros((len(combo['y_grid']), len(combo['x_grid'])))
            for i, y_val in enumerate(combo['y_grid']):
                for j, x_val in enumerate(combo['x_grid']):
                    params = combo['fixed'].copy()
                    for _axis, _val in (('y', y_val), ('x', x_val)):  # look up m0 / psi from their grids, take the others as they are
                        if combo[_axis] == 'm0':
                            params['m0'] = m0_grid[_val]
                        elif combo[_axis] == 'psi':
                            params['psi'] = psi_grid[_val]
                        else:
                            params[combo[_axis]] = _val
                    model = GaussianMixture(
                        n_features=n_features, n_components=n_components, covariance_type='full', max_iter=1000, device=device,
                        mean_prior=params.get('m0', m0_default), mean_precision_prior=params.get('kappa', kappa_default),
                        covariance_prior=params.get('psi', psi_default), degrees_of_freedom_prior=params.get('nu', nu_default),
                    )
                    model.fit(X_tensor)
                    assert model.converged_, f"Model did not converge for {combo['y']}={y_val}, {combo['x']}={x_val}"
                    ll_matrix[i, j] = model.lower_bound_

            ax = axes[idx]
            im = ax.imshow(ll_matrix, cmap='RdYlGn', aspect='auto')
            ax.set_title(f'{combo["y_label"]} vs {combo["x_label"]}', fontsize=11, pad=10)
            ax.set_xlabel(combo['x_label'], fontsize=9)
            ax.set_ylabel(combo['y_label'], fontsize=9)
            ax.set_xticks(range(len(combo['x_grid'])))
            ax.set_xticklabels(m0_labels if combo['x'] == 'm0' else psi_labels if combo['x'] == 'psi' else [f'{v}' for v in combo['x_grid']], fontsize=8)
            ax.set_yticks(range(len(combo['y_grid'])))
            ax.set_yticklabels(m0_labels if combo['y'] == 'm0' else psi_labels if combo['y'] == 'psi' else [f'{v}' for v in combo['y_grid']], fontsize=8)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('LL', rotation=270, labelpad=12, fontsize=8)
            for i in range(len(combo['y_grid'])):
                for j in range(len(combo['x_grid'])):
                    ax.text(j, i, f'{ll_matrix[i, j]:.3f}', ha='center', va='center', fontweight='bold', fontsize=8)

        plt.suptitle(f'NIW Prior Hyperparameter Interactions: Log-Likelihood Heatmaps{title_suffix}', fontsize=14, y=0.995)
        plt.tight_layout()
        return fig

    return (analyze_niw_hyperparameters,)


@app.cell
def _(
    FAST,
    analyze_niw_hyperparameters,
    device,
    n_components,
    n_features,
    niw_X_tensor,
    plt,
):
    print('Experiment 1: 100 samples per component (the NIW data above)')
    analyze_niw_hyperparameters(niw_X_tensor, n_components, n_features, device, title_suffix=' (n=100/comp)', fast=FAST)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What happens with more points per component?

    The prior counts as $\kappa_0$ and $\nu_0$ pseudo-observations, so with more data its influence fades: with 1000 points per component (300 in the fast docs build) the heatmaps flatten. In a full run, the spread of the log-likelihood over the $\kappa_0\times\nu_0$ grid (with $\Psi_0=I$, $\mu_0=0$) falls from about 7.2 at 100 points per component to about 0.9 at 1000.
    """)
    return


@app.cell
def _(
    FAST,
    SEED,
    analyze_niw_hyperparameters,
    device,
    n_components,
    n_features,
    niw_true_covs,
    niw_true_means,
    np,
    plt,
    torch,
):
    _n_per = 300 if FAST else 1000
    _rng = np.random.default_rng(SEED + 1)
    _X = np.vstack([_rng.multivariate_normal(_m, _c, _n_per) for _m, _c in zip(niw_true_means, niw_true_covs)])
    print(f'Experiment 2: {_n_per} samples per component')
    analyze_niw_hyperparameters(torch.tensor(_X, dtype=torch.float32, device=device), n_components, n_features, device, title_suffix=f' (n={_n_per}/comp)', fast=FAST)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What happens with different numbers of points per component?

    With unequal component sizes the small components feel the prior much more strongly than the large ones.
    """)
    return


@app.cell
def _(
    FAST,
    SEED,
    analyze_niw_hyperparameters,
    device,
    n_components,
    n_features,
    niw_true_covs,
    niw_true_means,
    np,
    plt,
    torch,
):
    _n_per = [100, 200, 300, 500] if FAST else [100, 500, 1000, 5000]
    _rng = np.random.default_rng(SEED + 2)
    _X = np.vstack([_rng.multivariate_normal(_m, _c, _n) for _m, _c, _n in zip(niw_true_means, niw_true_covs, _n_per)])
    print(f'Experiment 3: {_n_per} samples per component (unequal)')
    analyze_niw_hyperparameters(torch.tensor(_X, dtype=torch.float32, device=device), n_components, n_features, device, title_suffix=f' (n={_n_per})', fast=FAST)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Prior and the Amount of Data: What It Means for New Data

    The heatmaps above show one thing again and again: **how much a prior matters depends on how many points each component has.** The M-step formulas say why. In the update of component $k$ the prior is worth $\kappa_0$ pseudo-observations for the mean and $\nu_0$ for the covariance, next to the $n_k$ real ones:

    $$
    \mu_k=\frac{n_k\bar y_k+\kappa_0\mu_0}{n_k+\kappa_0},\qquad
    \Sigma_k=\frac{\Psi_0+W_k+\dots}{\nu_0+n_k},
    $$

    so the prior's share of the mean is $\kappa_0/(\kappa_0+n_k)$ and of the covariance $\nu_0/(\nu_0+n_k)$. With $n_k$ in the single digits the prior dominates; with $n_k$ in the hundreds it is negligible.

    This matters most when a fitted mixture is applied to **new, unseen data**. A maximum-likelihood fit on a handful of points can place a component on a few observations and give it a tiny, nearly singular covariance, which scores well on the training data and terribly on anything else. A prior is a regulariser against exactly that. The experiment below measures it: fit by maximum likelihood and with a weakly informative NIW prior (set from the training data only: $\mu_0$ the training mean, $\kappa_0=1$, $\nu_0=d+2$, and a prior covariance guess of 10% of the training covariance divided by $K$) on training sets of growing size drawn from the default blobs, and score both on a large held-out set from the same blobs.
    """)
    return


@app.cell
def _(
    FAST,
    GaussianMixture,
    SEED,
    default_blobs,
    device,
    dynamic_figsize,
    generate_gmm_data,
    n_components,
    np,
    plt,
    torch,
    warnings,
):
    _centers, _covs, _ns = default_blobs()
    _props = np.array(_ns) / sum(_ns)

    def draw_blobs(n, seed):
        """n points from the default four blobs, in the proportions of the tutorial data."""
        _counts = [max(int(round(n * _p)), 2) for _p in _props]
        return generate_gmm_data(_centers, _covs, _counts, device=device, random_state=seed)[0]

    X_heldout = draw_blobs(30000, 10000 + SEED)

    def heldout_ll(X_train, with_prior):
        """Mean log-likelihood on the held-out set of a fit to X_train (maximum likelihood, or with the weakly informative NIW prior)."""
        _kw = {}
        if with_prior:
            _nu0 = float(X_train.shape[1] + 2)
            _S = torch.cov(X_train.T)
            _kw = dict(mean_prior=X_train.mean(0, keepdim=True).repeat(n_components, 1), mean_precision_prior=1.0,
                       covariance_prior=(_nu0 * 0.1 * _S / n_components).unsqueeze(0).repeat(n_components, 1, 1), degrees_of_freedom_prior=_nu0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # tiny training sets make tgmm warn about near-zero weights: expected here
                _g = GaussianMixture(n_components=n_components, n_features=X_train.shape[1], covariance_type='full', init_means='kmeans', random_state=0, max_iter=500, device=device, **_kw).fit(X_train)
            return float(_g.score(X_heldout))
        except Exception:  # a degenerate (singular) fit counts as a failed run
            return float('nan')

    train_sizes = [20, 30, 60, 100, 300, 1000, 3000]  # cheap, so the docs build runs the full experiment too
    n_repeats = 10
    ll_mle = np.zeros((len(train_sizes), n_repeats))
    ll_map = np.zeros((len(train_sizes), n_repeats))
    for _i, _n in enumerate(train_sizes):
        for _r in range(n_repeats):
            _X_train = draw_blobs(_n, _r)  # the same training set for both fits
            ll_mle[_i, _r] = heldout_ll(_X_train, False)
            ll_map[_i, _r] = heldout_ll(_X_train, True)
    gap = ll_map - ll_mle  # paired difference per training set: positive means the prior helped on new data
    print(f'{len(train_sizes)} training sizes x {n_repeats} training sets; failed fits: MLE {int(np.isnan(ll_mle).sum())}, MAP {int(np.isnan(ll_map).sum())}')

    _fig, _axs = plt.subplots(1, 2, figsize=dynamic_figsize(1, 2, base_width=5.5, base_height=4), dpi=60)
    for _ll, _label, _color in ((ll_mle, 'maximum likelihood', 'tab:red'), (ll_map, 'NIW prior (MAP)', 'tab:green')):
        _axs[0].errorbar(train_sizes, np.nanmean(_ll, axis=1), yerr=np.nanstd(_ll, axis=1), marker='o', capsize=3, color=_color, label=_label)
    _axs[0].set_xscale('log')
    _axs[0].set_xlabel('training points $n$')
    _axs[0].set_ylabel('mean log-likelihood on held-out data')
    _axs[0].set_title('Held-out log-likelihood')
    _axs[0].legend()
    _axs[1].errorbar(train_sizes, np.nanmean(gap, axis=1), yerr=np.nanstd(gap, axis=1), marker='o', capsize=3, color='k')
    _axs[1].axhline(0, color='gray', ls='--', lw=1)
    _axs[1].set_xscale('log')
    _axs[1].set_xlabel('training points $n$')
    _axs[1].set_ylabel('MAP minus MLE')
    _axs[1].set_title('Gain from the prior (above 0: it helps)')
    plt.tight_layout()
    plt.gcf()
    return gap, ll_map, ll_mle, train_sizes


@app.cell(hide_code=True)
def _(gap, ll_map, ll_mle, mo, np, train_sizes):
    _rows = '\n'.join(
        f'| {_n} | {_n // 4} | {np.nanmean(ll_mle[_i]):.3f} ± {np.nanstd(ll_mle[_i]):.3f} | {np.nanmean(ll_map[_i]):.3f} ± {np.nanstd(ll_map[_i]):.3f} | {np.nanmean(gap[_i]):+.3f} |'
        for _i, _n in enumerate(train_sizes)
    )
    mo.md(f"""
    | training points $n$ | average points per component $n/K$ | MLE (held-out) | MAP (held-out) | MAP − MLE |
    |---|---|---|---|---|
    {_rows}

    (Mean ± standard deviation over the training sets; the right column is the paired difference.) With only {train_sizes[0] // 4} points per component on average (the smallest blob gets about 1 in 15) the maximum-likelihood fit generalises poorly (held-out {np.nanmean(ll_mle[0]):.2f}, and it varies a lot from one training set to the next) and the average gain from the prior is {np.nanmean(gap[0]):+.2f} (positive: the prior helps). By $n={train_sizes[-1]}$ the two fits are practically the same ({np.nanmean(gap[-1]):+.3f}): the prior has washed out.

    **Take-aways for using a fitted mixture on new data.**

    - The prior's weight is $\\kappa_0/(\\kappa_0+n_k)$ and $\\nu_0/(\\nu_0+n_k)$: it regularises a fit with few points per component and disappears with many. The same hyperparameters therefore behave very differently on a small and a large training set.
    - A prior is a guess, and a wrong guess costs something while the data are still few enough to feel it. Here the prior covariance and the shared prior mean are only roughly right for these blobs, and the gap in the middle rows shows the price. Keep $\\kappa_0$ and $\\nu_0$ small compared with the component sizes you expect (a few observations, not hundreds) unless you know the structure.
    - Judge the choice on held-out data, not on the training log-likelihood: the MAP fit always has the lower training log-likelihood (it optimises likelihood times prior), but that is not the number that matters for new data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - A prior converts each EM update into a MAP update: weights $\pi_k = (n_k + \alpha_k - 1)/(n + \sum_j \alpha_j - K)$, means $\mu_k = (n_k \bar y_k + \kappa_0 \mu_0)/(n_k + \kappa_0)$, and covariances $\Sigma_k = (\Psi_0 + \text{scatter} + \dots)/(\nu_0 + n_k)$.
    - Small Dirichlet concentrations favor a few dominant components; large ones favor balanced weights; unbalanced concentrations push the fit toward the favored component.
    - The mean precision $\kappa_0$ (and the covariance prior's degrees of freedom $\nu_0$) behave like pseudo-observation counts: the larger they are relative to $n_k$, the stronger the pull toward the prior, and the weaker it is for large components.
    - The inverse-Wishart / inverse-Gamma family is used for covariances because its density vanishes at singular covariances and keeps a heavy right tail. The covariance prior's denominator is $\nu_0+n_k$, not a textbook posterior mode or mean.
    - A mean prior alone or a covariance prior alone changes just that update; given both, tgmm uses the coupled NIW update with $\mu_k=\mu_n$, $\Sigma_k = \Psi_n/\nu_n$, where $\Psi_n$ gains a term that grows when the data mean disagrees with $\mu_0$. An independent mean prior is not available.
    - How much any prior matters depends on the component sizes: its weight is $\kappa_0/(\kappa_0+n_k)$ for the mean and $\nu_0/(\nu_0+n_k)$ for the covariance. With few points per component it can rescue a fit on new data; with many it washes out; a wrong prior costs something in between.
    - `mean_prior` / `covariance_prior` support the six classic covariance types; `weight_concentration_prior` works with all 14 GPCM models too.

    **Next:** [05 -- Sampling](05_sampling.ipynb) draws from a fitted mixture. See also [11 -- Dirichlet Process Mixtures](11_dpgmm.ipynb), where the same NIW prior is the base measure of a nonparametric sampler.
    """)
    return


if __name__ == "__main__":
    app.run()
