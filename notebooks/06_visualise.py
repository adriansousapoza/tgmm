# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "tgmm",
#     "torch; sys_platform != 'emscripten'",
#     "numpy",
#     "matplotlib",
#     "scipy",
# ]
#
# [tool.marimo.display]
# dataframes = "plain"
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
    return MAX_ITER, N_INIT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `plot_gmm`: Visualizing Gaussian Mixtures

    `plot_gmm` is the plotting workhorse of `tgmm` and is used by nearly every tutorial that follows. This notebook goes through its parameters one group at a time -- point styling, cluster coloring, confidence ellipses, mean markers, weight-based scaling and continuous log-likelihood coloring -- showing how each changes the figure. It then covers the helpers around it (`confidence_ellipse_std_devs`, `match_predicted_to_true_labels`, `gmm_proxy`) and the directional counterparts `plot_vmf` and `plot_vmf_cosine`.

    `plot_gmm` accepts any fitted `tgmm` mixture: it only needs `means_`, `covariances_`, `weights_`, `covariance_type` and `predict`, so it also draws every GPCM covariance code from [02 -- Covariance Types](02_covariance_types.ipynb). It draws **two-dimensional** data only (`X.shape[1] == 2`); for more features, project to 2-D first (see [10 -- PCA Plotting](10_pca_plotting.ipynb)).

    **What you'll learn**

    - Plot raw data, fitted models, and true generating components with `plot_gmm`
    - Color points by true labels, predicted clusters, or per-sample log-likelihood
    - Control confidence ellipses (including ellipses that enclose a chosen probability), mean markers and initial-mean markers
    - Scale markers or transparency by component weight, highlight misclassified points, and align cluster indices with the truth
    - Visualize von Mises-Fisher mixtures on the circle, the sphere and in high dimension

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb) (the data and the fit used throughout), and [05 -- Sampling](05_sampling.ipynb) for the von Mises-Fisher section.
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


    from scipy.stats import chi2
    from tutorial_data import default_blobs, make_default_data
    from tgmm import (
        CATEGORICAL_PALETTE, GaussianMixture, SEQUENTIAL_CMAP, VonMisesFisherMixture, confidence_ellipse_std_devs, gmm_proxy, match_predicted_to_true_labels, plot_gmm, plot_vmf, plot_vmf_cosine,
        vmf_angle_density, vmf_cap_angle,
    )

    # --- figure style (see notebooks/README.md) ---
    DOCS_RC = {'figure.dpi': 100, 'figure.titlesize': 12, 'axes.titlesize': 10.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
               'legend.fontsize': 8, 'lines.linewidth': 1.4, 'lines.markersize': 5}
    plt.rcParams.update(DOCS_RC)
    plt.rcParams['figure.dpi'] = 70  # lower than the style's 100 only to keep the exported page under 8 MB
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (
        CATEGORICAL_PALETTE, GaussianMixture, SEED, SEQUENTIAL_CMAP, VonMisesFisherMixture, chi2, confidence_ellipse_std_devs, default_blobs, device,
        gmm_proxy, make_default_data, match_predicted_to_true_labels, np, plot_gmm, plot_vmf,
        plot_vmf_cosine, plt, torch, vmf_angle_density, vmf_cap_angle,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview of Parameters

    The `plot_gmm` function provides extensive control over GMM visualization through several parameter categories. The module also exports the colour constants used as its defaults: `CATEGORICAL_PALETTE` (cluster colours), `SEQUENTIAL_CMAP` (continuous values) and `STATUS_CRITICAL` (misclassified points).

    **Core Data Parameters**
    - **`X`**: Input data points to plot (array-like, shape (n_samples, 2))
    - **`gmm`**: Fitted mixture with `predict()`, `means_`, `covariances_`, `weights_` and `covariance_type` (optional; `gmm_proxy` builds one from raw parameters)

    **Data Point Styling**
    - **`show_points`** *(bool, default=True)*: Whether to display the data points
    - **`point_size`** *(float, default=5)*: Size of individual data points
    - **`point_alpha`** *(float, default=0.5)*: Transparency of points (0=invisible, 1=opaque)
    - **`point_color`** *(str/array, default='auto')*: Point coloring strategy:
      - `'auto'`: Smart coloring based on available cluster info
      - `'black'`: All points in black
      - Colormap name: Use matplotlib colormap (e.g., 'viridis', 'plasma')
      - Single color: Use one color for all points (e.g., 'red', '#FF0000')
      - Array-like: Custom colors/values for each point

    **Cluster Visualization**
    - **`color_by_cluster`** *(bool, default=False)*: Enable cluster-based point coloring
    - **`true_labels`** *(array-like, optional)*: Ground truth cluster labels for comparison
    - **`match_labels_to_true`** *(bool, default=False)*: Remap predicted labels to match true labels using Hungarian algorithm
    - **`cluster_colors`** *(str/list, default=`CATEGORICAL_PALETTE`)*: Cluster color specification:
      - Matplotlib colormap name (e.g., 'turbo', 'viridis')
      - Single color for all clusters
      - List of specific colors (e.g., ['red', 'blue', 'green'])
    - **`show_incorrect_predictions`** *(bool, default=False)*: Highlight correct vs incorrect predictions (requires `true_labels`; misclassified points use `incorrect_color`)
    - **`incorrect_color`** *(str, default=`STATUS_CRITICAL`)*: Color of misclassified points

    **Continuous Coloring**
    - **`log_probs`** *(array-like, optional)*: Continuous values for color mapping (e.g., log-probabilities)
    - **`colormap`** *(str/Colormap, default=`SEQUENTIAL_CMAP`)*: Matplotlib colormap for continuous values
    - **`colorbar_label`** *(str, default='Log Probability')*: Label for the colorbar

    **Component Ellipses**
    - **`show_ellipses`** *(bool, default=True)*: Display confidence ellipses for GMM components
    - **`ellipse_std_devs`** *(list, default=[1, 2, 3])*: Mahalanobis radii of the ellipses, in standard deviations (see "Ellipses that enclose a given probability" below for what they enclose)
    - **`ellipse_alpha`** *(float, default=0.5)*: Fill transparency of ellipses
    - **`ellipse_colors`** *(str/list, default='auto')*: Ellipse color specification (same options as cluster_colors)
    - **`ellipse_fill`** *(bool, default=True)*: Whether ellipses should be filled or outline-only
    - **`ellipse_line_style`** *(str, default='dotted')*: Line style ('solid', 'dashed', 'dotted')
    - **`ellipse_line_width`** *(float, default=2)*: Thickness of ellipse boundaries
    - **`ellipse_line_color`** *(str, default='#0b0b0b')*: Color of ellipse boundary lines
    - **`ellipse_line_alpha`** *(float, default=0.5)*: Transparency of ellipse boundaries

    **Component Centers/Means**
    - **`show_means`** *(bool, default=True)*: Display component center points
    - **`mean_marker`** *(str, default='h')*: Marker style for means (any matplotlib marker: 'x', 'o', 'D', 'h', 'H', etc.)
    - **`mean_size`** *(float, default=25)*: Size of mean markers
    - **`mean_color`** *(str, default='black')*: Color of mean markers

    **Initial Means** (if available)
    - **`show_initial_means`** *(bool, default=False)*: Show the starting positions (`gmm.initial_means_`) EM was initialized from
    - **`initial_mean_marker`** *(str, default='H')*: Marker style for initial means
    - **`initial_mean_size`** *(float, default=25)*: Size of initial mean markers
    - **`initial_mean_color`** *(str, default='red')*: Color of initial mean markers

    **Weight-Based Scaling**
    - **`scale_alpha_by_weight`** *(bool, default=False)*: Scale ellipse transparency by component weight
    - **`scale_size_by_weight`** *(bool, default=False)*: Scale marker size by component weight

    **Plot Styling & Layout**
    - **`ax`** *(matplotlib.Axes, optional)*: Axes object to plot on (creates new if None)
    - **`title`** *(str, default='GMM Visualization')*: Plot title
    - **`xlabel`** *(str, default='Feature 1')*: X-axis label
    - **`ylabel`** *(str, default='Feature 2')*: Y-axis label
    - **`legend`** *(bool, default=True)*: Whether to show legend
    - **`legend_labels`** *(list, optional)*: Custom labels for legend entries
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A Fitted Model

    A `GaussianMixture` fitted to the default four blobs of [1. Introduction](01_introduction.ipynb). The fitted model, its per-sample log-likelihoods and its predictions feed every plot in the rest of the notebook.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, N_INIT, SEED, default_blobs, device, make_default_data, np, torch):
    centers, covs, n_samples = default_blobs()  # the true blobs (used again for the "truth" plot below)
    X_tensor, _y = make_default_data(SEED, device=device)
    X = X_tensor.cpu().numpy()
    labels = _y.cpu().numpy()

    n_components = 4

    # Initialize the GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        tol=1e-6,
        reg_covar=1e-10,
        max_iter=MAX_ITER,
        init_means='random',
        init_covariances='eye',
        n_init=N_INIT,
        random_state=SEED,
        warm_start=False,
        verbose=False,
        verbose_interval=100,
        device='cpu',
    )
    # Fit the GMM
    gmm.fit(X_tensor)

    # Get predictions
    y_pred = gmm.predict(X_tensor)

    # Compute per-sample log-likelihoods
    log_probs = gmm.score_samples(X_tensor)

    # Get probabilities for each sample for each component
    probs = gmm.predict_proba(X_tensor)
    probs = probs.detach()

    # Generate new samples
    n_samples_to_generate = len(X)
    gmm_samples, gmm_labels = gmm.sample(n_samples_to_generate)
    gmm_samples = gmm_samples.detach().cpu().numpy()
    gmm_labels = gmm_labels.detach().cpu().numpy()

    # Compute probabilities for each generated sample
    generated_probs = gmm.predict_proba(torch.tensor(gmm_samples, dtype=torch.float32).to(device))
    generated_probs = generated_probs.detach().cpu().numpy()
    return X, X_tensor, centers, covs, gmm, labels, log_probs, n_samples, y_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Default Parameters

    Every argument written out at its default value (with no `gmm`, so only the points are drawn):
    """)
    return


@app.cell
def _(CATEGORICAL_PALETTE, SEQUENTIAL_CMAP, X, plot_gmm, plt):
    _nrows, _ncols = (1, 1)
    _figsize = (6.5, 4.5)
    _fig, _ax = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=_figsize)
    plot_gmm(X, gmm=None, show_points=True, point_size=5, point_alpha=0.5, point_color='auto', color_by_cluster=False, true_labels=None, match_labels_to_true=False, cluster_colors=CATEGORICAL_PALETTE, show_incorrect_predictions=False, log_probs=None, colormap=SEQUENTIAL_CMAP, colorbar_label='Log Probability', show_ellipses=True, ellipse_std_devs=[1, 2, 3], ellipse_alpha=0.5, ellipse_colors='auto', ellipse_fill=True, ellipse_line_style='dotted', ellipse_line_width=2, ellipse_line_color='#0b0b0b', ellipse_line_alpha=0.5, show_means=True, mean_marker='h', mean_size=25, mean_color='black', show_initial_means=False, initial_mean_marker='H', initial_mean_size=25, initial_mean_color='red', scale_alpha_by_weight=False, scale_size_by_weight=False, ax=_ax, title='GMM Visualization', xlabel='Feature 1', ylabel='Feature 2', legend=True, legend_labels=None)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple Data Points Without GMM
    """)
    return


@app.cell
def _(X, plot_gmm, plt):
    _nrows, _ncols = (1, 2)
    _figsize = (11, 4.5)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=_figsize)
    _axes = _axes.flatten()
    _fig.suptitle('Different Point Styling Options without True Labels')
    point_color_options = [('auto', 'Auto (black by default)', {}), ('red', 'Single color (red)', {})]
    # Different point colors
    for _i, (color_option, _title, extra_params) in enumerate(point_color_options):
        plot_gmm(X, gmm=None, point_color=color_option, point_size=8, point_alpha=0.7, title=_title, show_ellipses=False, show_means=False, ax=_axes[_i], **extra_params)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cluster Coloring with True Labels
    """)
    return


@app.cell
def _(X, labels, plot_gmm, plt):
    _nrows, _ncols = (2, 2)
    _figsize = (11, 9)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=_figsize)
    _axes = _axes.flatten()
    _fig.suptitle('Cluster Visualization with True Labels')
    cluster_options = [(True, 'black', 'Cluster coloring (black)'), (True, 'turbo', 'Cluster coloring (turbo)'), (True, 'viridis', 'Cluster coloring (viridis)'), (True, ['red', 'blue', 'green', 'orange'], 'Custom cluster colors')]
    for _i, (color_by_cluster, cluster_colors, _title) in enumerate(cluster_options):
        plot_gmm(X, gmm=None, color_by_cluster=color_by_cluster, true_labels=labels, cluster_colors=cluster_colors, title=_title, show_ellipses=False, show_means=False, ax=_axes[_i])
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confidence Ellipses
    """)
    return


@app.cell
def _(X, gmm, plot_gmm, plt):
    _nrows, _ncols = (2, 2)
    _figsize = (11, 9)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=_figsize)
    _axes = _axes.flatten()
    _fig.suptitle('GMM Visualization with Different Ellipse Options')
    plot_gmm(X, gmm=gmm, color_by_cluster=True, title='GMM with Ellipses (Default)', ax=_axes[0])
    # Basic GMM with ellipses
    plot_gmm(X, gmm=gmm, color_by_cluster=True, ellipse_std_devs=[2], title='2σ Ellipses Only', ax=_axes[1], legend=False)
    plot_gmm(X, gmm=gmm, color_by_cluster=True, ellipse_std_devs=[1, 2, 3], title='1σ, 2σ and 3σ Ellipses', ax=_axes[2], legend=False)
    plot_gmm(X, gmm=gmm, color_by_cluster=True, ellipse_std_devs=[0.5, 1, 1.5, 2, 2.5, 3], title='Multiple Ellipse Levels', ax=_axes[3], legend=False)
    plt.tight_layout()
    # Different ellipse standard deviations
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ellipse Styling Options
    """)
    return


@app.cell
def _(X, gmm, plot_gmm, plt):
    _nrows, _ncols = (3, 2)
    _figsize = (11, 13.5)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=_figsize)
    _axes = _axes.flatten()
    ellipse_styles = [{'ellipse_fill': True, 'ellipse_alpha': 0.3, 'title': 'Filled Ellipses'}, {'ellipse_fill': False, 'ellipse_line_width': 2, 'title': 'Outline Only'}, {'ellipse_line_style': 'dashed', 'ellipse_line_width': 1.4, 'title': 'Dashed Lines'}, {'ellipse_line_color': 'red', 'ellipse_line_alpha': 1.0, 'title': 'Red Borders'}, {'ellipse_colors': 'Set3', 'title': 'Different Ellipse Colors'}, {'ellipse_alpha': 0.8, 'ellipse_line_width': 2, 'ellipse_line_color': 'black', 'title': 'High Contrast'}]
    # Different ellipse styling options
    for _i, style_options in enumerate(ellipse_styles):
        _title = style_options.pop('title')
        plot_gmm(X, gmm=gmm, color_by_cluster=True, ellipse_std_devs=[1, 2], title=_title, ax=_axes[_i], legend=(_i == 0), **style_options)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ellipses That Enclose a Given Probability

    `ellipse_std_devs` is a Mahalanobis radius $r$ in units of standard deviations: the ellipse is the set $\{x : (x-\mu)^{\top}\Sigma^{-1}(x-\mu) = r^2\}$. For a Gaussian in $d$ dimensions the squared Mahalanobis distance follows a $\chi^2_d$ distribution, so in $d=2$ (all that `plot_gmm` draws)

    $$
    P\big(\text{inside radius } r\big)=P\big(\chi^2_2\le r^2\big)=1-e^{-r^2/2}
    $$

    which is 39.3% for $r=1$, 86.5% for $r=2$ and 98.9% for $r=3$ -- *not* the 68/95/99.7% of the one-dimensional rule. To draw ellipses with a chosen coverage $q$, convert it through the $\chi^2_2$ quantile, $r=\sqrt{\chi^2_{2,q}}$. `confidence_ellipse_std_devs(q)` does exactly that and returns values to pass as `ellipse_std_devs`. The cell below checks the formula on a million standard-normal draws.
    """)
    return


@app.cell
def _(chi2, confidence_ellipse_std_devs, np):
    _z = np.random.default_rng(0).standard_normal((1_000_000, 2))
    _d2 = (_z ** 2).sum(1)  # squared Mahalanobis distance to the mean (Sigma = I)
    print(f"{'r':>3} {'1 - exp(-r^2/2)':>16} {'empirical':>10} {'1-D rule':>9}")
    for _r, _one_d in zip((1, 2, 3), (0.6827, 0.9545, 0.9973)):
        print(f"{_r:>3} {1 - np.exp(-_r ** 2 / 2):>16.4f} {(_d2 <= _r ** 2).mean():>10.4f} {_one_d:>9.4f}")
    _q = [0.5, 0.95]
    _r_q = confidence_ellipse_std_devs(_q)
    print(f"confidence_ellipse_std_devs({_q}) = {[round(_v, 4) for _v in _r_q]}  (sqrt of chi2.ppf: {[round(float(np.sqrt(chi2.ppf(_v, 2))), 4) for _v in _q]})")
    print(f"empirical coverage at those radii: {[round(float((_d2 <= _v ** 2).mean()), 4) for _v in _r_q]}")
    return


@app.cell
def _(X, confidence_ellipse_std_devs, gmm, plot_gmm, plt, torch):
    def coverage_figure(q):
        """Draw the fitted GMM with ellipses enclosing probability q; returns the figure and the share of each component's assigned points inside its ellipse."""
        _r = confidence_ellipse_std_devs(q)
        _fig, _ax = plt.subplots(figsize=(6.5, 4.5))
        _Xt = torch.as_tensor(X, dtype=gmm.means_.dtype, device=gmm.means_.device)
        _lab = gmm.predict(_Xt)
        _inside = []
        for _k in range(gmm.n_components):
            _dx = _Xt[_lab == _k] - gmm.means_[_k]
            _inside.append(float(((_dx @ torch.linalg.inv(gmm.covariances_[_k]) * _dx).sum(1) <= _r[0] ** 2).float().mean()))
        plot_gmm(X, gmm=gmm, color_by_cluster=True, ellipse_std_devs=_r, ellipse_alpha=0.3, title=f'{q:.0%} confidence ellipses (r = {_r[0]:.3f})', ax=_ax)
        return _fig, _inside

    _fig, _inside = coverage_figure(0.95)
    print(f"share of assigned points inside each component's ellipse: {[round(_v, 3) for _v in _inside]}")
    plt.gcf()
    return (coverage_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log Probability Visualization
    """)
    return


@app.cell
def _(X, gmm, log_probs, plot_gmm, plt):
    _nrows, _ncols = (2, 2)
    _figsize = (11, 9)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=_figsize)
    _axes = _axes.flatten()
    _fig.suptitle('Continuous Coloring with Log Probabilities')
    colormaps = ['viridis', 'plasma', 'coolwarm', 'RdYlBu']
    # Different continuous coloring options
    for _i, cmap in enumerate(colormaps):
        plot_gmm(X, gmm=gmm, log_probs=log_probs, colormap=cmap, colorbar_label='Log Probability', title=f'Log Probabilities ({cmap})', show_ellipses=False, ax=_axes[_i])
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean Markers and Initial Means

    `mean_marker`, `mean_size` and `mean_color` style the fitted means, and `initial_mean_*` the starting points EM began from (a fitted model remembers them in `initial_means_`). Any matplotlib marker works and larger sizes are possible, but the tutorials keep markers small (`mean_size` at most 40) so they do not hide the data: component means are black hexagons (`'h'`, `s=25`, the default), initial guesses are red crosses (`'x'`, `s=30`). The panels below show other options at `s=40`.
    """)
    return


@app.cell
def _(X, gmm, plot_gmm, plt):
    _nrows, _ncols = (2, 2)
    _figsize = (11, 9)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=_figsize)
    _axes = _axes.flatten()
    _fig.suptitle('Component Means and Initial Means Visualization')
    plot_gmm(X, gmm=gmm, color_by_cluster=True, show_means=True, mean_marker='X', mean_size=40, mean_color='black', title='X Markers for Means', ax=_axes[0])
    # Different mean marker options
    plot_gmm(X, gmm=gmm, color_by_cluster=True, show_means=True, mean_marker='D', mean_size=40, mean_color='white', title='White Diamond Markers', ax=_axes[1], legend=False)
    plot_gmm(X, gmm=gmm, color_by_cluster=True, show_means=True, show_initial_means=True, initial_mean_marker='x', initial_mean_size=30, initial_mean_color='red', title='Final (black) and Initial (red) Means', ax=_axes[2])
    plot_gmm(X, gmm=gmm, color_by_cluster=True, show_means=False, show_ellipses=True, ellipse_alpha=0.2, title='Ellipses Only (No Mean Markers)', ax=_axes[3], legend=False)
    plt.tight_layout()
    # Show initial means if available
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Component Weight Visualization
    """)
    return


@app.cell
def _(X, gmm, plot_gmm, plt):
    _nrows, _ncols = (1, 3)
    _figsize = (11, 3.8)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=_figsize)
    _axes = _axes.flatten()
    _fig.suptitle('Weight-Based Scaling of Ellipses')
    plot_gmm(X, gmm=gmm, color_by_cluster=True, title='No Weight Scaling', ax=_axes[0])
    # Regular visualization
    plot_gmm(X, gmm=gmm, color_by_cluster=True, scale_alpha_by_weight=True, title='Alpha Scaled by Weight', ax=_axes[1], legend=False)
    plot_gmm(X, gmm=gmm, color_by_cluster=True, scale_size_by_weight=True, title='Mean Size Scaled by Weight', ax=_axes[2], mean_size=40, legend=False)
    plt.tight_layout()
    # Print component weights for reference
    print(f'Component weights: {gmm.weights_.detach().cpu().numpy()}')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prediction Accuracy Assessment
    """)
    return


@app.cell
def _(X, gmm, labels, plot_gmm, plt):
    _nrows, _ncols = (1, 3)
    _figsize = (11, 3.8)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=_figsize)
    _axes = _axes.flatten()
    _fig.suptitle('Prediction Accuracy Comparison')
    plot_gmm(X, gmm=None, color_by_cluster=True, true_labels=labels, title='True Labels', show_ellipses=False, ax=_axes[0])
    # Show true labels
    plot_gmm(X, gmm=gmm, color_by_cluster=True, title='Predicted Labels', show_ellipses=False, ax=_axes[1])
    plot_gmm(X, gmm=gmm, true_labels=labels, show_incorrect_predictions=True, match_labels_to_true=True, title='Correct (Green) vs Incorrect (Red)', show_ellipses=False, ax=_axes[2])
    plt.tight_layout()
    # Show predicted labels
    # Show correct vs incorrect predictions
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matching Predicted Labels to the Truth

    Cluster indices returned by `predict()` are arbitrary: EM may call the true blob 0 "component 3". Before computing supervised metrics, or coloring a prediction consistently with the truth, remap them with `match_predicted_to_true_labels` (a Hungarian assignment that maximises the number of agreeing points). `plot_gmm(..., match_labels_to_true=True)` does this internally whenever both `true_labels` and a fitted `gmm` are given, as in the correct-vs-incorrect panel above.
    """)
    return


@app.cell
def _(labels, match_predicted_to_true_labels, torch, y_pred):
    _true = torch.as_tensor(labels, dtype=torch.long)
    _raw = y_pred.cpu().long()
    aligned_labels = match_predicted_to_true_labels(_true, _raw)
    print(f"agreement with the true labels, raw predict():   {(_raw == _true).float().mean():.3f}")
    print(f"agreement after match_predicted_to_true_labels:  {(aligned_labels == _true).float().mean():.3f}")
    print(f"predicted index -> true index: { {int(_k): int(aligned_labels[_raw == _k][0]) for _k in _raw.unique()} }")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing Models Side by Side

    Every call takes `ax=`, so comparing models is a loop with one axes per panel. Here the same data are fitted with three classic covariance types and one GPCM code (`VEI`: varying volume, equal axis-aligned shape; see [02 -- Covariance Types](02_covariance_types.ipynb)); `plot_gmm` draws all of them the same way.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, SEED, X, X_tensor, plot_gmm, plt):
    _cov_types = ['full', 'diag', 'spherical', 'VEI']
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 9))
    for _i, (_ax, _ct) in enumerate(zip(_axes.ravel(), _cov_types)):
        _m = GaussianMixture(n_components=4, covariance_type=_ct, max_iter=MAX_ITER, random_state=SEED).fit(X_tensor)
        plot_gmm(X, gmm=_m, color_by_cluster=True, ellipse_std_devs=[1, 2], ax=_ax, legend=(_i == 0), title=f"covariance_type='{_ct}'   (BIC = {_m.bic(X_tensor):.0f})")
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting Distributions That Were Never Fitted: `gmm_proxy`

    `gmm_proxy(means, covariances, weights, covariance_type, predicted_labels)` wraps raw parameters in the minimal object `plot_gmm` expects. It is the way to draw the *true* generating components behind synthetic data, or parameters that were projected to 2-D by hand. When you color by cluster, supply `predicted_labels` (the per-point assignment `predict()` should return).
    """)
    return


@app.cell
def _(X, centers, covs, gmm_proxy, labels, n_samples, np, plot_gmm, plt):
    _truth = gmm_proxy(means=np.stack(centers), covariances=np.stack(covs), weights=np.array(n_samples) / sum(n_samples), covariance_type='full', predicted_labels=labels.astype(int))
    _fig, _ax = plt.subplots(figsize=(6.5, 4.5))
    plot_gmm(X, gmm=_truth, color_by_cluster=True, ellipse_std_devs=[1, 2], scale_alpha_by_weight=True, title='True generating components (via gmm_proxy)', ax=_ax)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## High-Dimensional Data

    `plot_gmm` requires exactly two features. Project higher-dimensional data down first -- for example with PCA -- fit in 2-D, and pass the projected coordinates:

    ```python
    from sklearn.decomposition import PCA

    X_2d = PCA(n_components=2).fit_transform(X_hd.cpu().numpy())
    gmm = GaussianMixture(n_components=5, n_features=2).fit(torch.tensor(X_2d, dtype=torch.float32))
    plot_gmm(X_2d, gmm, show_ellipses=True, title='PCA projection with clusters')
    ```

    [10 -- PCA Plotting](10_pca_plotting.ipynb) shows the better route: fit in the original high-dimensional space and only project the fitted means and covariances for display.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview of Different Visualization Combinations
    """)
    return


@app.cell
def _(X, gmm, log_probs, plot_gmm, plt):
    _nrows, _ncols = (2, 2)
    _figsize = (11, 9)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=_figsize)
    _axes = _axes.flatten()
    _fig.suptitle('Comprehensive GMM Visualization Examples')
    custom_labels = ['Cluster A', 'Cluster B', 'Cluster C', 'Cluster D']
    # Custom legend labels
    plot_gmm(X, gmm=gmm, show_points=True, point_size=8, point_alpha=0.6, point_color='auto', color_by_cluster=True, cluster_colors='Set2', legend_labels=custom_labels, show_ellipses=True, ellipse_std_devs=[1, 2, 3], ellipse_alpha=0.4, ellipse_fill=True, ellipse_line_style='solid', ellipse_line_width=1.5, ellipse_line_color='darkblue', ellipse_line_alpha=0.8, show_means=True, mean_marker='H', mean_size=40, mean_color='black', show_initial_means=True, initial_mean_marker='x', initial_mean_size=30, initial_mean_color='red', title='Complete GMM Visualization', xlabel='Feature Dimension 1', ylabel='Feature Dimension 2', ax=_axes[0])
    plot_gmm(X, gmm=gmm, show_points=True, point_size=5, point_alpha=0.8, point_color='black', show_ellipses=False, show_means=False, legend=False, title='Minimal: Points Only', ax=_axes[1])
    # Full-featured visualization
    plot_gmm(X, gmm=gmm, show_points=False, show_ellipses=True, ellipse_std_devs=[1, 2, 3], ellipse_alpha=0.6, ellipse_colors='viridis', show_means=True, mean_marker='X', mean_size=40, title='Ellipses and Means Only', ax=_axes[2])
    plot_gmm(X, gmm=gmm, log_probs=log_probs, colormap='coolwarm', colorbar_label='Log-Likelihood', show_ellipses=True, ellipse_std_devs=[2], ellipse_fill=False, ellipse_line_width=2, ellipse_line_color='black', title='Log Probabilities + Ellipses', ax=_axes[3])
    plt.tight_layout()
    # Minimal visualization
    # Ellipses only
    # Continuous coloring with ellipses
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Von Mises-Fisher Mixtures: `plot_vmf` and `plot_vmf_cosine`

    `plot_gmm` draws Euclidean ellipses, so it cannot represent a `VonMisesFisherMixture` ([16 -- von Mises-Fisher](16_von_mises_fisher.ipynb)), whose data live on the unit sphere and whose components are described by a mean *direction* $\mu_k$ and a concentration $\kappa_k$. Two functions cover that case.

    **`plot_vmf(X, vmf, ax=None, ...)`** draws unit-norm data on the circle ($d=2$) or the sphere ($d=3$; pass a 3-D axes, or omit `ax` to create one). Points are colored by predicted component (or by `true_labels=`); each mean direction is an arrow from the origin; each concentration is drawn as nested **spherical caps**, the vMF analogue of confidence ellipses. A cap of angular radius $\theta_m$ around $\mu_k$ contains probability $m$ of the component, where $\theta_m$ solves $P(\theta\le\theta_m)=m$ under the angle density

    $$
    f_\kappa(\theta)\;\propto\;e^{\kappa\cos\theta}\sin^{d-2}\theta,\qquad\theta=\arccos(\mu^{\top}x)\in[0,\pi].
    $$

    The $\sin^{d-2}\theta$ factor is the surface area of the ring at angle $\theta$; the density is normalised numerically. On the circle the caps are arcs just outside the unit circle, on the sphere they are rings of constant angular distance from $\mu_k$. `show_means`, `show_caps`, `cap_masses` (default `(0.5, 0.9)`), `legend_labels`, `view_elev` and `view_azim` control the rest. `vmf_angle_density(kappa, n_features, theta)` and `vmf_cap_angle(kappa, n_features, mass)` expose the density and the cap radius directly.
    """)
    return


@app.cell
def _(SEED, VonMisesFisherMixture, np, torch):
    _g = np.random.default_rng(SEED)
    _dirs = np.array([[1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 0.0, -1.0]])
    _dirs /= np.linalg.norm(_dirs, axis=1, keepdims=True)
    X_sphere = np.vstack([_d + 0.35 * _g.standard_normal((250, 3)) for _d in _dirs])
    X_sphere /= np.linalg.norm(X_sphere, axis=1, keepdims=True)  # unit-norm rows
    vmf_sphere = VonMisesFisherMixture(n_components=3, random_state=SEED).fit(torch.tensor(X_sphere, dtype=torch.float32))
    return X_sphere, vmf_sphere


@app.cell
def _(X_sphere, plot_vmf, plt, vmf_sphere):
    _fig = plt.figure(figsize=(6.5, 4.5))
    _ax = _fig.add_subplot(projection='3d')
    plot_vmf(X_sphere, vmf_sphere, ax=_ax, cap_masses=(0.5, 0.9), title='plot_vmf on the sphere: means, 50% and 90% caps')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`plot_vmf_cosine(X, vmf)`** works in any dimension. For every component it histograms the angle between each assigned point and the component's mean direction and overlays the fitted $f_{\kappa_k}(\theta)$ (dashed); if the vMF assumption holds, histogram and curve agree. In high dimension the typical angle is far from $0^\circ$ even for large $\kappa$, because the mode of $f_\kappa$ solves $\kappa\sin^2\theta=(d-2)\cos\theta$. Below: two components in $d=20$.
    """)
    return


@app.cell
def _(SEED, VonMisesFisherMixture, np, torch):
    _g = np.random.default_rng(SEED + 1)
    _mu = _g.standard_normal((2, 20))
    _mu /= np.linalg.norm(_mu, axis=1, keepdims=True)
    X_hd = np.vstack([_m + 0.18 * _g.standard_normal((400, 20)) for _m in _mu])
    X_hd /= np.linalg.norm(X_hd, axis=1, keepdims=True)
    vmf_hd = VonMisesFisherMixture(n_components=2, random_state=SEED).fit(torch.tensor(X_hd, dtype=torch.float32))
    return X_hd, vmf_hd


@app.cell
def _(X_hd, plot_vmf_cosine, plt, vmf_hd):
    _fig, _ax = plt.subplots(figsize=(6.5, 4.5))
    plot_vmf_cosine(X_hd, vmf_hd, ax=_ax, title='plot_vmf_cosine in d = 20: angle to the mean direction')
    plt.gcf()
    return


@app.cell
def _(X_hd, np, torch, vmf_angle_density, vmf_cap_angle, vmf_hd):
    from scipy.optimize import brentq as _brentq
    _d = X_hd.shape[1]
    _Xt = torch.tensor(X_hd, dtype=vmf_hd.means_.dtype, device=vmf_hd.means_.device)
    _lab = vmf_hd.predict(_Xt)
    _grid = np.linspace(1e-3, np.pi - 1e-3, 20001)
    for _k in range(vmf_hd.n_components):
        _kappa = float(vmf_hd.kappa_[_k])
        _ang = torch.arccos((_Xt[_lab == _k] @ vmf_hd.means_[_k]).clamp(-1, 1)).cpu().numpy()
        _cap = vmf_cap_angle(_kappa, _d, 0.9)
        _mode = _brentq(lambda t: _kappa * np.sin(t) ** 2 - (_d - 2) * np.cos(t), 1e-6, np.pi / 2)
        _mode_num = _grid[np.argmax(vmf_angle_density(_kappa, _d, _grid))]
        print(f"component {_k}: kappa = {_kappa:.1f}; 90% cap radius {np.degrees(_cap):.1f} deg holds {(_ang <= _cap).mean():.3f} of its points; "
              f"mode of f: {np.degrees(_mode):.1f} deg from kappa sin^2 = (d-2) cos, {np.degrees(_mode_num):.1f} deg numerically")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tips for Effective Visualization

    1. **Project first.** `plot_gmm` needs two features; use PCA (or another method) for more.
    2. **Adjust transparency.** `point_alpha` and `ellipse_alpha` reveal overlapping clusters.
    3. **Use several ellipse levels.** `ellipse_std_devs=[1, 2, 3]` shows the spread at a glance; use `confidence_ellipse_std_devs` when you need a stated coverage.
    4. **Compare models side by side.** Loop over `covariance_type` or the initialization method with one `ax` per panel.
    5. **Check initial against final means.** `show_initial_means=True` helps debug poor convergence ([03 -- Initialization](03_initialization.ipynb)).
    6. **Align labels before judging a fit.** Use `match_predicted_to_true_labels` or `match_labels_to_true=True`.

    For the full signatures, see the [plotting API reference](../api/plotting.md).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - `plot_gmm(X, gmm=...)` draws the data, the fitted component means and nested covariance ellipses in one call; `gmm=None` plots data only.
    - Points can be colored by true labels, predicted clusters, correctness (`show_incorrect_predictions`) or a continuous value such as `log_probs`.
    - Ellipse levels (`ellipse_std_devs`), fill, line style, mean markers and initial-mean markers are all configurable.
    - `scale_alpha_by_weight` / `scale_size_by_weight` make component weights visible, and every call accepts `ax=` so panels compose into grids.
    - `ellipse_std_devs` is a Mahalanobis radius: 1, 2, 3 standard deviations enclose 39.3%, 86.5% and 98.9% in 2-D; `confidence_ellipse_std_devs(q)` gives the radius for a chosen coverage.
    - `match_predicted_to_true_labels` aligns arbitrary cluster indices with the truth, `gmm_proxy` plots parameters that were never fitted, and `plot_vmf` / `plot_vmf_cosine` cover directional data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it yourself (interactive version)

    The page you are reading shows the default setting of each control below. Open the interactive version (the link at the top of the page: molab, or `marimo edit` on your machine) to change them.

    ### Choose the enclosed probability

    Move the slider to redraw the confidence ellipses with a different coverage $q$ (the radius is $r=\sqrt{\chi^2_{2,q}}$, as above). The title also reports the share of each component's assigned points that falls inside its ellipse.
    """)
    return


@app.cell
def _(mo):
    # docs: hide-cell
    coverage = mo.ui.slider(start=0.05, stop=0.99, step=0.01, value=0.95, label='enclosed probability q', show_value=True)
    coverage
    return (coverage,)


@app.cell
def _(coverage, coverage_figure, plt):
    # docs: hide-cell  (the static page already shows the 95% default above)
    _fig, _inside = coverage_figure(coverage.value)
    _fig.axes[0].set_title(_fig.axes[0].get_title() + f"\nshare of assigned points inside: {[round(_v, 2) for _v in _inside]}")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Next:** [07 -- Classification EM](07_cem.ipynb), or [10 -- PCA Plotting](10_pca_plotting.ipynb) for plotting high-dimensional models.
    """)
    return


if __name__ == "__main__":
    app.run()
