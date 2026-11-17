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
    return (MAX_ITER,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction: Gaussian Mixtures and the EM Algorithm

    This is the starting point for `tgmm`. It explains what a Gaussian mixture model (GMM) is and how the Expectation-Maximization (EM) algorithm fits one, first on a small 1-D example and then on a 2-D dataset of four Gaussians, and then fits a `GaussianMixture` and goes through everything you do with a fitted model: predicting labels and soft assignments, scoring, sampling, inspecting the learned parameters, plotting, and saving.

    The data on this page are fixed defaults, but they are also *yours to change*: in the interactive version of this notebook (see the link at the top of the page) the last section lets you define your own 2-D blobs (the centre, covariance and number of points of each blob) and shows a fit to them. The two EM videos are rendered ahead of time and only loaded here.

    **What you'll learn**

    - How a mixture of Gaussians generates data, and why its maximum-likelihood fit has no closed form
    - The E-step (responsibilities) and the M-step (weighted maximum-likelihood updates), and why each EM iteration never lowers the log-likelihood
    - How a poor start is repaired iteration by iteration in 1-D, and how responsibilities and covariance ellipses evolve in 2-D (two videos you can play, pause and scrub)
    - Fit a `GaussianMixture` and use `predict`, `predict_proba`, `score_samples`, `score`, `bic`, `aic`, `icl` and `sample`
    - Inspect `weights_`, `means_` and `covariances_`, visualize them with `plot_gmm`, and `save`/`load` a model
    - Define your own 2-D blobs in the interactive version and fit them

    **Prerequisites:** none.

    **Where the details live:** the six classic and fourteen GPCM covariance structures are in [02 -- Covariance Types](02_covariance_types.ipynb); how EM is started (means, weights, covariances, `n_init`) in [03 -- Initialization](03_initialization.ipynb); drawing samples in [05 -- Sampling](05_sampling.ipynb).
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
    import base64
    import warnings

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import norm

    from tgmm import GaussianMixture, plot_gmm, generate_gmm_data

    # --- figure style (see notebooks/README.md) ---
    DOCS_RC = {'figure.dpi': 100, 'figure.titlesize': 12, 'axes.titlesize': 10.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
               'legend.fontsize': 8, 'lines.linewidth': 1.4, 'lines.markersize': 5}
    plt.rcParams.update(DOCS_RC)
    plt.rcParams['figure.dpi'] = 75  # lower than the style's 100 only to keep the export under 8 MB (the page embeds two videos and four large figures)
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convergence warnings are expected below: several cells deliberately stop EM after one or a few iterations.
    warnings.filterwarnings('ignore', message='.*EM did not converge.*')
    return (
        GaussianMixture,
        SEED,
        base64,
        device,
        generate_gmm_data,
        linear_sum_assignment,
        norm,
        np,
        plot_gmm,
        plt,
        torch,
    )


@app.cell
def _(base64, mo):
    # docs: hide-cell
    import json as _json

    _RAW = 'https://raw.githubusercontent.com/adriansousapoza/tgmm/main/notebooks/assets/'  # where a hosted copy of this notebook finds the files

    def asset_info(name):
        """The numbers saved next to notebooks/assets/<name>.mp4 (``{}`` if the file is not there)."""
        _dir = mo.notebook_dir()
        _path = _dir / 'assets' / f'{name}.json' if _dir is not None else None
        return _json.loads(_path.read_text()) if _path is not None and _path.exists() else {}

    def show_video(name):
        """Show the video notebooks/assets/<name>.mp4 (with its poster frame). The videos are rendered ahead of time by scripts/make_assets/."""
        _dir = mo.notebook_dir()
        _mp4 = _dir / 'assets' / f'{name}.mp4' if _dir is not None else None
        if _mp4 is not None and _mp4.exists():  # embedded, so the exported docs page is self-contained
            _src = 'data:video/mp4;base64,' + base64.b64encode(_mp4.read_bytes()).decode()
            _jpg = _mp4.with_suffix('.jpg')
            _poster = 'data:image/jpeg;base64,' + base64.b64encode(_jpg.read_bytes()).decode() if _jpg.exists() else ''
        else:
            _src, _poster = _RAW + f'{name}.mp4', _RAW + f'{name}.jpg'
        return mo.Html(f'<video controls loop muted playsinline preload="metadata" poster="{_poster}" style="width:100%;height:auto">'
                       f'<source src="{_src}" type="video/mp4"></video>')

    return asset_info, show_video


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The Model: A Mixture of Gaussians

    A Gaussian mixture says that each data point was generated in two stages. First pick one of $K$ components at random, then draw the point from that component's Gaussian. Formally there is a hidden (latent) label $z_i\in\{1,\dots,K\}$ for each point $x_i\in\mathbb R^d$ with

    $$
    p(z_i = k) = \pi_k, \qquad \pi_k\ge 0,\quad \sum_{k=1}^K \pi_k = 1,
    $$

    and, given the label,

    $$
    p(x_i \mid z_i = k) = \mathcal{N}(x_i \mid \mu_k, \Sigma_k),
    \qquad
    \mathcal{N}(x\mid\mu,\Sigma)=\frac{1}{(2\pi)^{d/2}\,|\Sigma|^{1/2}}\exp\!\Big(-\tfrac12 (x-\mu)^\top\Sigma^{-1}(x-\mu)\Big).
    $$

    Marginalizing out the label we cannot observe gives the density that is actually fitted to the data:

    $$
    p(x_i \mid \theta) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k),
    \qquad \theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K .
    $$

    The weights $\pi_k$ are the mixing proportions (the fraction of points each component generates, $n_k/n$ for the examples below), $\mu_k$ the centres and $\Sigma_k$ the shapes. *Clustering* means inferring the hidden label $z_i$ of each point.

    **Maximum likelihood** asks for the parameters that make the $n$ observed points most probable:

    $$
    \theta^\star=\arg\max_\theta\ \ell(\theta),\qquad
    \ell(\theta)=\sum_{i=1}^n \log \sum_{k=1}^K \pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k).
    $$

    The sum *inside* the logarithm is the obstacle: unlike for a single Gaussian there is no closed-form maximizer. If the labels *were* observed, the **complete-data** log-likelihood

    $$
    \ell_c(\theta)=\sum_{i,k}\mathbb 1[z_i=k]\,\big(\log\pi_k+\log\mathcal N(x_i\mid\mu_k,\Sigma_k)\big)
    $$

    would split into one independent, closed-form problem per component. **Expectation-Maximization** exploits exactly this: it alternates between guessing the labels (softly) and solving the easy complete-data problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The EM Algorithm

    ### Notation

    With $r_{ik}$ the responsibility of component $k$ for sample $i$ (defined next), the M-step needs only three summaries of the data per component, its **sufficient statistics**:

    $$
    n_k=\sum_i r_{ik},\qquad
    \bar y_k=\frac{1}{n_k}\sum_i r_{ik}\,x_i,\qquad
    W_k=\sum_i r_{ik}\,(x_i-\bar y_k)(x_i-\bar y_k)^\top .
    $$

    $n_k$ is the "effective number of points" of component $k$, $\bar y_k$ the responsibility-weighted mean and $W_k$ the responsibility-weighted scatter matrix.

    ### E-step: soft-assign points to components

    Using Bayes' rule, compute the posterior probability ("responsibility") that component $k$ generated point $x_i$ under the *current* parameters:

    $$
    r_{ik}=p(z_i=k\mid x_i,\theta)=
    \frac{\pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_{j}\pi_j\,\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}
    =\operatorname{softmax}_k\!\big(\log\pi_k+\log\mathcal{N}(x_i\mid\mu_k,\Sigma_k)\big).
    $$

    Each row of $r$ sums to 1: a *soft*, probabilistic assignment, not a hard one. `tgmm` evaluates this in log space (a Cholesky factor $\Sigma_k=L_kL_k^\top$ gives $\log|\Sigma_k|=2\sum\log\operatorname{diag}L_k$ and the Mahalanobis term by a triangular solve; the normalizer is a `logsumexp`), which keeps it stable even when a point is far from every component. The same normalizer is the per-sample log-likelihood $\log p(x_i\mid\theta)$, whose mean is `lower_bound_` (and what `score` returns).

    ### M-step: re-estimate the parameters from the soft assignments

    Treat $r_{ik}$ as weights and refit each component to the weighted points:

    $$
    \pi_k=\frac{n_k}{n},\qquad
    \mu_k=\bar y_k,\qquad
    \Sigma_k=\frac{W_k}{n_k}\ \ (+\ \varepsilon I),
    $$

    where $\varepsilon$ is `reg_covar`, a small value added to keep $\Sigma_k$ positive definite. These are the ordinary Gaussian maximum-likelihood formulas (sample mean, sample covariance, class proportion), each point weighted by $r_{ik}$ instead of being hard-assigned to one component.

    Why $\Sigma_k=W_k/n_k$: the $\Sigma_k$-dependent part of the objective is $-\tfrac12\big[n_k\log|\Sigma_k|+\operatorname{tr}(\Sigma_k^{-1}W_k)\big]$ at $\mu_k=\bar y_k$, and its maximizer over all positive definite matrices is the weighted sample covariance. Constraining $\Sigma_k$ to a smaller family (diagonal, spherical, shared between components, ...) changes only this one update; see [02 -- Covariance Types](02_covariance_types.ipynb).

    ### Repeat until convergence

    Alternate the two steps until the log-likelihood $\ell(\theta)$ improves by less than `tol` (relative), or `max_iter` iterations have run. `converged_` tells you which happened, and `n_iter_` how many iterations were used.

    ### Parameter counts

    The number of free parameters, which `bic` and `aic` use, is $p=(K-1)+Kd+p_\Sigma$: $K-1$ weights, $Kd$ mean coordinates, and $p_\Sigma$ covariance parameters, which for `covariance_type='full'` is $K\,d(d+1)/2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why does it work?

    For any distribution $q_{ik}$ over the labels, Jensen's inequality gives a lower bound on the log-likelihood (the *evidence lower bound*, ELBO):

    $$
    \ell(\theta)\ \ge\ \mathcal F(q,\theta)=\sum_{i,k}q_{ik}\log\frac{\pi_k\,\mathcal N(x_i\mid\mu_k,\Sigma_k)}{q_{ik}},
    $$

    with equality exactly when $q_{ik}=p(z_i=k\mid x_i,\theta)$. EM alternates

    1. **E-step:** set $q$ to that posterior, which makes the bound *tight* at the current parameters $\theta^{(t)}$;
    2. **M-step:** maximize $\mathcal F(q,\theta)$ over $\theta$.

    Then

    $$
    \ell(\theta^{(t+1)})\ \ge\ \mathcal F(q,\theta^{(t+1)})\ \ge\ \mathcal F(q,\theta^{(t)})\ =\ \ell(\theta^{(t)}),
    $$

    so **the log-likelihood never decreases** from one iteration to the next. That guarantees convergence to a stationary point, in practice a **local** maximum, so the result depends on where EM starts. That is why initialization and `n_init` restarts matter; they are the subject of [03 -- Initialization](03_initialization.ipynb). To make the dependence, and the improvement itself, visible, the next two sections start EM from a deliberately poor guess and watch it recover: first in one dimension, where the fitted curves can be drawn directly against the data, then in two dimensions on four blobs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A 1-D Example

    We start with the simplest case: one feature ($d=1$), generated from three true Gaussian components so that there is ground truth to compare the fit with. (The EM runs in this section and the next were recorded ahead of time, one E-step and one M-step at a time with `tgmm`'s internal `_e_step` / `_m_step` methods, so that every iteration can be shown as a video; [06 -- Visualization](06_visualise.ipynb) introduces `plot_gmm`, which is used for the rest of this notebook.)
    """)
    return


@app.cell
def _(SEED, norm, np, plt):
    # Generate sample data with real means and stds. The draws are a fixed slice of a seeded stream of standard normals, so the data
    # never depend on what other cells draw (and the design-your-own panel at the end of the notebook starts from exactly this mixture).
    real_means = [-2, 2, 6]
    real_stds = [2, 1, 3]
    cluster_sizes = [2000, 1000, 3000]
    _z = np.random.RandomState(SEED).standard_normal(12000)[6000:]
    _data1 = real_means[0] + real_stds[0] * _z[:2000]
    _data2 = real_means[1] + real_stds[1] * _z[2000:3000]
    _data3 = real_means[2] + real_stds[2] * _z[3000:6000]
    data = np.concatenate([_data1, _data2, _data3])
    _total_size = sum(cluster_sizes)
    # True mixing coefficients based on the actual data proportions
    true_taus = [size / _total_size for size in cluster_sizes]
    plt.figure(figsize=(6.5, 4.5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data histogram')
    _x = np.linspace(min(data), max(data), 1000)
    # Plotting the histogram and the weighted Gaussians
    _colors = ['r', 'b', 'g']
    _gaussian_real = 0
    for _real_mean, _real_std, _true_tau, _color in zip(real_means, real_stds, true_taus, _colors):
        plt.plot(_x, _true_tau * norm.pdf(_x, _real_mean, _real_std), color=_color, linestyle='--', label='Gaussian (real)')
        _gaussian_real += _true_tau * norm.pdf(_x, _real_mean, _real_std)
    plt.plot(_x, _gaussian_real, 'k--', label='GMM (real)')
    plt.title('Data and Gaussian Distributions')
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 0: a (deliberately) bad initialization

    We initialize all three components close together around $9$ to $10$, even though the true components are centred at $-2$, $2$ and $6$, and give them a shared, overly wide initial standard deviation of $2.5$. No EM iteration has run yet: these are just the starting parameters, and the E-step computes the responsibilities $r_{ik}$ from the formula above using only this initial guess.

    Because the initial means sit together and far from where two of the three true clusters are, the responsibilities are not yet informative. This is the starting point EM has to improve on (it is the first frame of the video below).
    """)
    return


@app.cell(hide_code=True)
def _(asset_info, mo):
    _i = asset_info('em_1d')
    _f = lambda key, spec='': format(_i[key], spec) if key in _i else 'n/a'
    mo.md(rf"""
    ### Full EM run: watching convergence

    From this start, EM was run to convergence one E-step and one M-step at a time, recording $(\mu_k,\sigma_k,\pi_k)$ and the average log-likelihood after every iteration. The run needed **{_f('iterations')} iterations**: the average log-likelihood rose from {_f('ll_first', '.2f')} for the initial guess to {_f('ll_last', '.4f')}, and it never decreased (the largest drop between two iterations was {_f('largest_drop', '.1e')}). The fitted means ended at {', '.join(f'{_m:.2f}' for _m in _i.get('final_means', []))}, close to the true $-2$, $2$ and $6$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The whole run, iteration by iteration

    The video replays the run: **press play, or drag the timeline to scrub through the iterations**. Iteration $0$ is the bad initial guess and the last frame the converged fit (the frames are denser at the start, where EM changes fastest). At iteration $t$ the top row shows the fitted components against the real ones **(left)**, each point's responsibility for each component **(middle)**, and the average log-likelihood climbing over the iterations so far **(right)**; the bottom row shows, for each component, the **log-likelihood surface** over its $(\text{mean},\text{std})$ with the path EM has taken up to $t$ and the current estimate. The surface is the one EM is climbing: the other components are held at their converged values, and each estimate walks uphill to the maximum.
    """)
    return


@app.cell(hide_code=True)
def _(mo, show_video):
    mo.vstack([
        show_video('em_1d'),
        mo.md("<small>This video was rendered ahead of time by [`scripts/make_assets/make_em_videos.py`](https://github.com/adriansousapoza/tgmm/blob/main/scripts/make_assets/make_em_videos.py) (`make assets` regenerates it); the notebook only loads it from `notebooks/assets/`.</small>"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. EM in 2-D on Four Blobs

    The same algorithm works in $d>1$ dimensions; only the Gaussian density changes, to the multivariate normal of section 1, with each $\Sigma_k$ now a full $d\times d$ covariance matrix. The E-step and M-step formulas are unchanged.

    ### The data: four Gaussians

    The synthetic 2-D dataset combines four Gaussian blobs, drawn with `generate_gmm_data` (which samples $x=\mu_k+L_kz$ with $z\sim\mathcal N(0,I)$ and $L_kL_k^\top=\Sigma_k$, a Cholesky factor, so the covariance you request is the covariance you get). Each blob is described by a centre, a covariance matrix and a number of points. Two are spherical (the second has fewer points), one is axis-aligned and elongated (diagonal covariance), and one is tilted (full covariance, correlated features):
    """)
    return


@app.cell
def _(np):
    default_n_samples = [800, 200, 1000, 1000]
    default_centers = [np.array([0, 2]),
                       np.array([2, -2]),
                       np.array([0, 0]),
                       np.array([2, 2])]
    default_covs = [
        np.eye(2),                              # spherical covariance
        0.25 * np.eye(2),                       # spherical covariance, small and with few points
        np.array([[4.0, 0.0], [0.0, 0.25]]),    # diagonal covariance, elongated along x
        np.array([[0.29, 1.1], [1.1, 4.25]]),   # full covariance: very elongated and tilted (correlation 0.99)
    ]
    return default_centers, default_covs, default_n_samples


@app.cell
def _(
    SEED,
    default_centers,
    default_covs,
    default_n_samples,
    device,
    generate_gmm_data,
):
    centers, covs, n_samples = default_centers, default_covs, default_n_samples

    # Generate the synthetic data (a seeded draw: the same blobs always give the same points)
    X_tensor, y_tensor = generate_gmm_data(centers, covs, n_samples, device=device, random_state=SEED)

    # Convert to numpy for plotting
    X = X_tensor.cpu().numpy()
    labels = y_tensor.cpu().numpy()
    n_components = len(n_samples)
    legend_labels = [f'Component {i+1}' for i in range(n_components)]
    return (
        X,
        X_tensor,
        centers,
        covs,
        labels,
        legend_labels,
        n_components,
        n_samples,
    )


@app.cell
def _(X, labels, legend_labels, plot_gmm, plt):
    plot_gmm(X=X, true_labels=labels, title='Original Data', legend_labels=legend_labels)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EM on these blobs

    We fit one component per blob to the data above. To make the dynamics visible, the run starts from a deliberately uninformative guess: all four components start close together around $(0,0)$ (the origin plus a little seeded noise), each with the covariance of the whole data set and equal weights. The state was recorded after every EM iteration, and the video below shows three panels for each of them: **(left)** each point's responsibility for each component (the points are in the order of the blobs they were drawn from), **(middle)** the hard assignment (arg-max responsibility, used only for colouring) in data space, and **(right)** the data with the trajectory of each component's mean (dotted line, from the initial means marked `x` to the current means `o`) and the dashed ellipse that encloses $95\%$ of its Gaussian (a radius of $\sqrt{\chi^2_{2,\,0.95}}\approx 2.45$ standard deviations in 2-D; in 1-D the same level is about $1.96$, i.e. roughly two standard deviations).
    """)
    return


@app.cell(hide_code=True)
def _(asset_info, mo):
    _i = asset_info('em_2d')
    _f = lambda key, spec='': format(_i[key], spec) if key in _i else 'n/a'
    mo.md(rf"""
    ### Watching EM separate the blobs

    EM converged after **{_f('iterations')} iterations**; the average log-likelihood rose from {_f('ll_first', '.2f')} to {_f('ll_last', '.4f')}. **Press play, or drag the timeline to scrub through the iterations**: watch how the soft responsibilities sharpen into confident assignments, and how the four components, which start on top of each other, spread out and settle onto the blobs.
    """)
    return


@app.cell(hide_code=True)
def _(mo, show_video):
    mo.vstack([
        show_video('em_2d'),
        mo.md("<small>This video was rendered ahead of time by [`scripts/make_assets/make_em_videos.py`](https://github.com/adriansousapoza/tgmm/blob/main/scripts/make_assets/make_em_videos.py) (`make assets` regenerates it); the notebook only loads it from `notebooks/assets/`.</small>"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Fitting with `GaussianMixture`

    Everything above can be done with one call. `GaussianMixture(n_components=K).fit(X)` runs EM for you. Here we choose `n_components` to be the number of blobs and seed the starting means with k-means++ (`init_means='kpp'`), which spreads them out over the data so that EM is less likely to end in a poor local optimum. `verbose=True` and `verbose_interval=10` print the progress. (How the starting means, weights and covariances can be chosen is the subject of [03 -- Initialization](03_initialization.ipynb).)
    """)
    return


@app.cell
def _(GaussianMixture, X_tensor, device, n_components, np, torch):
    gmm = GaussianMixture(n_components=n_components, init_means='kpp', verbose=True, verbose_interval=10)
    gmm.fit(X_tensor)
    y_pred = gmm.predict(X_tensor)
    log_probs = gmm.score_samples(X_tensor)
    probs = gmm.predict_proba(X_tensor)
    probs = probs.detach()
    n_samples_to_generate = len(X_tensor)
    gmm_samples, gmm_labels = gmm.sample(n_samples_to_generate)
    gmm_samples = gmm_samples.detach().cpu().numpy()
    gmm_labels = gmm_labels.detach().cpu().numpy()
    generated_probs = gmm.predict_proba(torch.tensor(gmm_samples, dtype=torch.float32).to(device))
    generated_probs = generated_probs.detach().cpu().numpy()
    print('Mean per-sample log-likelihood:                        ', gmm.score(X_tensor))
    print('Mean per-sample log-likelihood (torch.mean(log_probs)):', torch.mean(log_probs).item())
    print('Lower bound:                                           ', gmm.lower_bound_)
    for _i in range(probs.shape[0]):
        assert np.isclose(torch.sum(probs[_i]).item(), 1.0), f'Probabilities for sample {_i} do not sum to 1'
    print('Number of iterations:                                  ', gmm.n_iter_)
    print('Converged:                                             ', gmm.converged_)
    return gmm, gmm_samples, log_probs, probs, y_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Inspecting the Learned Parameters

    For each component the fitted model stores a **weight** $\pi_k$ (`weights_`), a **mean** $\mu_k$ (`means_`) and a **covariance** $\Sigma_k$ (`covariances_`; for other covariance types a vector or scalar, see [02 -- Covariance Types](02_covariance_types.ipynb)). Component numbering is arbitrary (EM may order the blobs differently from the way they were defined), so the table below pairs each fitted component with the closest true blob.
    """)
    return


@app.cell
def _(centers, covs, gmm, linear_sum_assignment, n_samples, np):
    _fit_means = gmm.means_.detach().cpu().numpy()
    _true_means = np.stack(centers).astype(float)
    _cost = np.linalg.norm(_fit_means[:, None] - _true_means[None], axis=2)  # fitted x true distances
    _fit_idx, _true_idx = linear_sum_assignment(_cost)  # best one-to-one pairing of fitted and true components
    print(f'{"fitted":>6} {"true":>5} {"weight (fit / true)":>21} {"mean (fit)":>20} {"mean (true)":>14}')
    for _f, _t in zip(_fit_idx, _true_idx):
        _w_true = n_samples[_t] / sum(n_samples)
        print(f'{_f + 1:>6} {_t + 1:>5} {gmm.weights_[_f].item():>10.3f} /{_w_true:>8.3f}   {np.round(_fit_means[_f], 2)!s:>18} {np.round(_true_means[_t], 2)!s:>14}')
    print()
    for _f, _t in zip(_fit_idx, _true_idx):
        print(f'Component {_f + 1} (true blob {_t + 1}) covariance, fitted vs true:')
        print(np.round(gmm.covariances_[_f].detach().cpu().numpy(), 3))
        print(np.round(np.asarray(covs[_t], float), 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Visualizing the Fitted Model: One Command per View

    Each panel is the output of one method of the fitted `gmm`, drawn with `plot_gmm` (introduced properly in [06 -- Visualization](06_visualise.ipynb)):

    - **`gmm.fit(X)`** (top left): the learned `means_` and `covariances_`, drawn as the fitted covariance ellipses (1, 2 and 3 standard deviations) over the data, with the means as black hexagons. The box in its corner shows the information criteria `gmm.bic(X)`, `gmm.aic(X)` and `gmm.icl(X)` (lower is better; defined below the plots).
    - **`gmm.score_samples(X)`** (top right): each point coloured by its log-likelihood $\log p(x_i\mid\theta)$; points far from every component are the unlikely ones. The title also shows its average, `gmm.score(X)`.
    - **`gmm.predict(X)`** (bottom left): the hard cluster label of every point.
    - **`gmm.sample(n)`** (bottom right): new points drawn from the fitted mixture, in the same grey as the data in the first panel and without the ellipses, so you can compare them with the original data (more in [05 -- Sampling](05_sampling.ipynb)). Unlike the predictions, which colour the *real* points by their cluster, the samples carry no label here: if the model is good the grey cloud looks like the data.

    Below them, `gmm.predict_proba(X)` is drawn one component at a time, for the observed data and for the generated samples. The information criteria are defined and checked right after the plots.
    """)
    return


@app.cell
def _(
    X,
    X_tensor,
    gmm,
    gmm_samples,
    legend_labels,
    log_probs,
    n_components,
    plot_gmm,
    plt,
    y_pred,
):
    _fig, _axs = plt.subplots(2, 2, figsize=(11, 9))
    _means = dict(mean_marker='h', mean_size=25, mean_color='black')  # the component means are small black hexagons, the points are coloured or grey
    plot_gmm(X=X, gmm=gmm, ax=_axs[0, 0], title='gmm.fit(X)\nmeans_ and covariances_', legend_labels=legend_labels, point_color='#6b6b6b', **_means)
    plot_gmm(X=X, ax=_axs[0, 1], title=f'gmm.score_samples(X)\nmean = gmm.score(X) = {gmm.score(X_tensor):.3f}', log_probs=log_probs, colorbar_label='Log-Likelihood', colormap='viridis')
    plot_gmm(X=X, gmm=gmm, show_ellipses=False, ax=_axs[1, 0], true_labels=y_pred, title='gmm.predict(X)', legend_labels=[f'Cluster {i}' for i in range(n_components)])
    plot_gmm(X=gmm_samples, gmm=gmm, ax=_axs[1, 1], title='gmm.sample(n)', show_ellipses=False, point_color='#6b6b6b', **_means)  # one solid grey like the data in the first panel (a colormap name such as 'gray' would shade the points)
    _ic = f'gmm.bic(X) = {gmm.bic(X_tensor):.1f}\ngmm.aic(X) = {gmm.aic(X_tensor):.1f}\ngmm.icl(X) = {gmm.icl(X_tensor):.1f}'  # lower is better
    _axs[0, 0].text(0.02, 0.02, _ic, transform=_axs[0, 0].transAxes, ha='left', va='bottom', fontsize=plt.rcParams['legend.fontsize'], family='monospace', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='0.7'))
    _fig.tight_layout()
    plt.gcf()
    return


@app.cell
def _(X, n_components, plot_gmm, plt, probs):
    _nrows = (n_components + 1) // 2
    _fig, _axs = plt.subplots(_nrows, 2, figsize=(11, 4.5 * _nrows), squeeze=False)
    for _k, _ax in enumerate(_axs.ravel()):
        if _k < n_components:
            plot_gmm(X=X, ax=_ax, title=f'Component {_k + 1} Probability', log_probs=probs[:, _k], colormap='RdBu', colorbar_label='Probability')
        else:
            _ax.axis('off')
    plt.suptitle('gmm.predict_proba(X): responsibilities of the observed data')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Fitted Attributes, Devices and Persistence

    After `fit`, the model exposes (among others):

    | Attribute | Meaning |
    | --- | --- |
    | `weights_` | mixing weights $\pi$, shape $(K,)$ |
    | `means_` | component means, shape $(K,d)$ |
    | `covariances_` | covariances; shape depends on `covariance_type` (for `'full'`: $(K,d,d)$) |
    | `converged_`, `n_iter_` | whether EM converged, and in how many iterations |
    | `lower_bound_` | final average log-likelihood of the best run |
    | `log_likelihood_history_` | average log-likelihood after each iteration |
    | `best_random_state_` | seed of the winning run when `n_init > 1` |
    | `initial_weights_`, `initial_means_`, `initial_covariances_` | the parameters EM started from |

    The complete list is in the [API reference](../api/gaussian-mixture.md).

    **Devices.** `GaussianMixture(..., device='cuda')` runs everything on the GPU; `X` may live on the CPU or the GPU and is moved to `device`. This notebook picks `'cuda'` if available and falls back to `'cpu'` (the `device` variable in the setup cell).

    **Saving and loading.** `GaussianMixture` has four methods for persistence, which preserve all parameters, configuration, training state and prior settings:

    1. **`save(filepath)`**: save the complete model to a file;
    2. **`load(filepath, device=None)`**: class method that loads a complete model from a file;
    3. **`save_state_dict()`**: return a dictionary with all the model state (as in PyTorch);
    4. **`load_state_dict(_state_dict)`**: load the state from a dictionary into a model with the same `n_components` and `n_features`.
    """)
    return


@app.cell
def _(GaussianMixture, X_tensor, device, gmm, n_components, torch):
    import tempfile as _tempfile
    from pathlib import Path as _Path

    print('Original model:')
    print(f'  Number of components: {gmm.n_components}, covariance type: {gmm.covariance_type}')
    print(f'  Converged: {gmm.converged_}, iterations: {gmm.n_iter_}, log-likelihood: {gmm.score(X_tensor):.4f}')
    print()
    print('=== Method 1: complete model save/load ===')
    with _tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as _f:
        _model_path = _f.name
    gmm.save(_model_path)
    _loaded_gmm = GaussianMixture.load(_model_path, device=device)
    print(f'  Saved to and loaded from {_model_path}')
    print(f'  Loaded log-likelihood: {_loaded_gmm.score(X_tensor):.4f}, converged: {_loaded_gmm.converged_}')
    print(f'  Predictions match: {torch.equal(gmm.predict(X_tensor), _loaded_gmm.predict(X_tensor))}')
    _Path(_model_path).unlink()
    print()
    print('=== Method 2: state dictionary save/load ===')
    _state_dict = gmm.save_state_dict()
    print(f'  The state dict has {len(_state_dict)} keys, e.g. {list(_state_dict.keys())[:4]}')
    _new_gmm = GaussianMixture(n_components=n_components, n_features=2, device=device)
    _new_gmm.load_state_dict(_state_dict)
    print(f'  Loaded log-likelihood: {_new_gmm.score(X_tensor):.4f}, converged: {_new_gmm.converged_}')
    print(f'  Predictions match: {torch.equal(gmm.predict(X_tensor), _new_gmm.predict(X_tensor))}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - A GMM generates a point by drawing a hidden component $z_i\sim\pi$ and then $x_i\sim\mathcal N(\mu_{z_i},\Sigma_{z_i})$; maximum likelihood has no closed form because of the sum inside the logarithm.
    - EM alternates the E-step (responsibilities $r_{ik}$ from the current parameters) and the M-step ($r$-weighted maximum-likelihood updates). Each iteration is guaranteed not to decrease the log-likelihood, but only a local optimum is reached, so the result depends on the initialization.
    - Even from a deliberately poor start the 1-D example recovers the true components, and in 2-D the responsibilities sharpen into confident assignments while the covariance ellipses settle onto the blobs.
    - `GaussianMixture(n_components=K).fit(X)` is all that is needed for a first fit; `predict`, `predict_proba`, `score_samples`, `score`, `bic`, `aic`, `icl` and `sample` give labels, soft assignments, per-point log-likelihoods, information criteria and new data.
    - `plot_gmm` overlays the fitted means and covariance ellipses on the data and can colour points by cluster, responsibility or log-likelihood; `save`/`load` and `save_state_dict`/`load_state_dict` persist a fitted model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it yourself (interactive version)

    The page you are reading shows the fixed default data. To specify the mixture yourself, open the interactive version of this notebook (the molab link at the top of the page, or `marimo edit` locally): the number fields below only work there, and all of this notebook's interactive controls live in this last section. The result is static: one `GaussianMixture` fit to your blobs and the same views as in section 7 (`score_samples`, `predict`, `sample`, `predict_proba`), with no video.

    For every blob you type the centre $(x, y)$, the covariance matrix

    $$
    \Sigma = \begin{pmatrix}\Sigma_{11} & \Sigma_{12}\\ \Sigma_{12} & \Sigma_{22}\end{pmatrix}
    $$

    (symmetric, so you enter $\Sigma_{11}$, $\Sigma_{12}=\Sigma_{21}$ and $\Sigma_{22}$) and the number of points $n$. A valid covariance has $\Sigma_{11}>0$, $\Sigma_{22}>0$ and $\Sigma_{12}^2<\Sigma_{11}\Sigma_{22}$ (positive definite); the panel shows the standard deviations $\sqrt{\Sigma_{11}},\sqrt{\Sigma_{22}}$ and the correlation $\rho=\Sigma_{12}/\sqrt{\Sigma_{11}\Sigma_{22}}$, and warns you if the matrix is not valid (it then uses the closest valid one). Equal variances and $\Sigma_{12}=0$ give a spherical blob, $\Sigma_{12}=0$ with unequal variances an axis-aligned one, and $\Sigma_{12}\neq 0$ a tilted ("full") one. The **number of blobs** field ($K = 1,\dots,6$) adds or removes blobs, and **Reset** restores the four default blobs of section 4. The fit uses the default constructor, `GaussianMixture(n_components=K)`.
    """)
    return


@app.cell
def _(default_centers, default_covs, default_n_samples, mo, np):
    # docs: hide-cell
    # (x, y, S11, S12, S22, n) per blob: the four defaults, then two spare blobs for "add a blob"
    DEFAULT_ROWS = [(*[float(_v) for _v in _c], float(_S[0, 0]), float(_S[0, 1]), float(_S[1, 1]), float(_n))
                    for _c, _S, _n in zip(default_centers, default_covs, default_n_samples)]
    DEFAULT_ROWS += [(-3.0, -3.0, 0.36, 0.0, 0.36, 300.0), (4.0, 0.0, 0.25, 0.0, 0.25, 300.0)]

    def valid_cov(_s11, _s12, _s22):
        """The symmetric matrix [[s11, s12], [s12, s22]] if it is positive definite, else the closest one (eigenvalues floored)."""
        _S = np.array([[_s11, _s12], [_s12, _s22]], float)
        _w, _V = np.linalg.eigh(_S)
        if _w.min() > 1e-6:
            return _S, True
        return (_V * np.maximum(_w, 1e-3)) @ _V.T, False

    def blob_spec(rows, k):
        """(centers, covs, n_samples) for the first k blobs. Untouched defaults give the exact default arrays."""
        if k == len(default_n_samples) and [list(_r) for _r in rows[:k]] == [list(_r) for _r in DEFAULT_ROWS[:k]]:
            return default_centers, default_covs, default_n_samples
        return ([np.array(_r[:2]) for _r in rows[:k]], [valid_cov(*_r[2:5])[0] for _r in rows[:k]], [max(int(_r[5]), 1) for _r in rows[:k]])

    reset_blobs = mo.ui.button(label='Reset to the default four blobs')
    return DEFAULT_ROWS, blob_spec, reset_blobs, valid_cov


@app.cell
def _(DEFAULT_ROWS, mo, reset_blobs):
    # docs: hide-cell
    _ = reset_blobs.value  # pressing Reset re-runs this cell, which recreates every field at its default
    _spec = [('x', -10, 10, 0.1), ('y', -10, 10, 0.1), ('Σ₁₁', 0.0, 20, 0.01), ('Σ₁₂', -20, 20, 0.01), ('Σ₂₂', 0.0, 20, 0.01), ('n', 1, 20000, 1)]
    # mo.ui.array (not a plain list): marimo only re-runs dependent cells when a field's value changes if the field sits in an array
    blob_fields = mo.ui.array([mo.ui.array([mo.ui.number(start=_lo, stop=_hi, step=_st, value=_v, label=f'{_nm}', debounce=True)
                                            for (_nm, _lo, _hi, _st), _v in zip(_spec, DEFAULT_ROWS[_i])]) for _i in range(6)])
    k_blobs = mo.ui.number(start=1, stop=6, step=1, value=4, label='number of blobs K', debounce=True)
    return blob_fields, k_blobs


@app.cell
def _(blob_fields, k_blobs, mo, np, reset_blobs, valid_cov):
    # docs: hide-cell
    def _blob_panel(_i):
        _x, _y, _s11, _s12, _s22, _n = blob_fields.value[_i]
        _, _ok = valid_cov(_s11, _s12, _s22)
        _f = blob_fields[_i]
        _info = (f'std devs {np.sqrt(_s11):.3f}, {np.sqrt(_s22):.3f}, correlation ρ = {_s12 / np.sqrt(_s11 * _s22):.3f}' if _ok
                 else '⚠️ this matrix is not positive definite (need Σ₁₁>0, Σ₂₂>0, Σ₁₂² < Σ₁₁Σ₂₂); the closest valid covariance is used')
        return mo.vstack([mo.md(f'**Blob {_i + 1}**  ·  centre (x, y), covariance Σ (Σ₂₁ = Σ₁₂), points n'),
                          mo.hstack([_f[0], _f[1], _f[2], _f[3], _f[4], _f[5]], justify='start'),
                          mo.md(f'<small>{_info}</small>')])

    mo.vstack([
        mo.hstack([k_blobs, reset_blobs], justify='start', align='end'),
        *[_blob_panel(_i) for _i in range(int(k_blobs.value))],
    ])
    return


@app.cell
def _(SEED, blob_fields, blob_spec, device, generate_gmm_data, k_blobs):
    custom_centers, custom_covs, custom_n_samples = blob_spec(blob_fields.value, int(k_blobs.value))
    custom_X_tensor, custom_y_tensor = generate_gmm_data(custom_centers, custom_covs, custom_n_samples, device=device, random_state=SEED)
    custom_X = custom_X_tensor.cpu().numpy()
    return custom_X, custom_X_tensor, custom_n_samples


@app.cell
def _(
    GaussianMixture,
    MAX_ITER,
    SEED,
    custom_X,
    custom_X_tensor,
    custom_n_samples,
    mo,
    plot_gmm,
    plt,
):
    _K = len(custom_n_samples)
    try:
        custom_gmm = GaussianMixture(n_components=_K, random_state=SEED, max_iter=MAX_ITER)
        custom_gmm.fit(custom_X_tensor)
    except Exception as _e:  # e.g. too few points for the number of components
        custom_gmm = None
        _out = mo.md(f'⚠️ the fit failed for these blobs ({_e}); try more points or larger covariances')
    else:
        print(f'gmm.score(X) = {custom_gmm.score(custom_X_tensor):.4f}   (mean log-likelihood per sample)')
        print(f'gmm.bic(X)   = {custom_gmm.bic(custom_X_tensor):.1f}')
        print(f'gmm.aic(X)   = {custom_gmm.aic(custom_X_tensor):.1f}')
        print(f'gmm.icl(X)   = {custom_gmm.icl(custom_X_tensor):.1f}')
        _log_probs = custom_gmm.score_samples(custom_X_tensor).detach().cpu()
        _y_pred = custom_gmm.predict(custom_X_tensor).detach().cpu().numpy()
        _samples = custom_gmm.sample(len(custom_X_tensor))[0].detach().cpu().numpy()
        _names = [f'Component {_i + 1}' for _i in range(_K)]
        _means = dict(mean_marker='h', mean_size=25, mean_color='black')
        _fig, _axs = plt.subplots(2, 2, figsize=(11, 9))
        plot_gmm(X=custom_X, gmm=custom_gmm, ax=_axs[0, 0], title='gmm.fit(X)\nmeans_ and covariances_', legend_labels=_names, point_color='#6b6b6b', **_means)
        plot_gmm(X=custom_X, ax=_axs[0, 1], title=f'gmm.score_samples(X)\nmean = gmm.score(X) = {custom_gmm.score(custom_X_tensor):.3f}', log_probs=_log_probs, colorbar_label='Log-Likelihood', colormap='viridis')
        plot_gmm(X=custom_X, gmm=custom_gmm, show_ellipses=False, ax=_axs[1, 0], true_labels=_y_pred, title='gmm.predict(X)', legend_labels=[f'Cluster {_i}' for _i in range(_K)], **_means)
        plot_gmm(X=_samples, gmm=custom_gmm, ax=_axs[1, 1], title='gmm.sample(n)', show_ellipses=False, point_color='#6b6b6b', **_means)
        _fig.tight_layout()
        _out = plt.gcf()
    _out
    return (custom_gmm,)


@app.cell
def _(custom_X, custom_X_tensor, custom_gmm, plot_gmm, plt):
    if custom_gmm is None:
        _out = None
    else:
        _K = custom_gmm.n_components
        _probs = custom_gmm.predict_proba(custom_X_tensor).detach().cpu()
        _nrows = (_K + 1) // 2
        _fig, _axs = plt.subplots(_nrows, 2, figsize=(11, 4.5 * _nrows), squeeze=False)
        for _k, _ax in enumerate(_axs.ravel()):
            if _k < _K:
                plot_gmm(X=custom_X, ax=_ax, title=f'Component {_k + 1} Probability', log_probs=_probs[:, _k], colormap='RdBu', colorbar_label='Probability')
            else:
                _ax.axis('off')
        plt.suptitle('gmm.predict_proba(X): responsibilities of your data')
        plt.tight_layout()
        _out = plt.gcf()
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Where to go next.** This notebook fitted one kind of mixture (`covariance_type='full'`, maximum likelihood, EM from a k-means++ start). Each of the following changes one ingredient:

    - **The shape of each component:** diagonal, spherical, shared (`tied_*`) and the 14 GPCM structures, in [02 -- Covariance Types](02_covariance_types.ipynb).
    - **The starting point and restarts:** `init_means`, `init_weights`, `init_covariances`, `n_init`, `warm_start`, in [03 -- Initialization](03_initialization.ipynb).
    - **Drawing new data, including constrained sampling:** [05 -- Sampling](05_sampling.ipynb).
    - **Hard assignments:** `algorithm='cem'` (Classification EM), in [07 -- CEM](07_cem.ipynb).
    - **Regularizing the estimates:** priors and MAP estimation, in [04 -- Priors](04_priors.ipynb) (MAP and the NIW prior).
    - **Choosing $K$:** [13 -- Model Selection](13_model_selection.ipynb).

    **Next:** [02 -- Covariance Types](02_covariance_types.ipynb), then [03 -- Initialization](03_initialization.ipynb) and [05 -- Sampling](05_sampling.ipynb).
    """)
    return


if __name__ == "__main__":
    app.run()
