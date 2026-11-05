# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "tgmm",
#     "torch; sys_platform != 'emscripten'",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "pillow",
#     "imageio-ffmpeg",
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
    return FAST, MAX_ITER


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction: Gaussian Mixtures and the EM Algorithm

    This is the starting point for `tgmm`. It explains what a Gaussian mixture model (GMM) is and how the Expectation-Maximization (EM) algorithm fits one, first on a small 1-D example and then on a 2-D dataset of four Gaussians, and then fits a `GaussianMixture` and goes through everything you do with a fitted model: predicting labels and soft assignments, scoring, sampling, inspecting the learned parameters, plotting, and saving.

    The data on this page are fixed defaults, but they are also *yours to change*: in the interactive version of this notebook (see the link at the top of the page) the last section lets you define your own 1-D mixture (the mean, standard deviation and number of points of each component) and your own 2-D blobs (the centre, covariance and number of points of each blob) and shows a fit to them.

    **What you'll learn**

    - How a mixture of Gaussians generates data, and why its maximum-likelihood fit has no closed form
    - The E-step (responsibilities) and the M-step (weighted maximum-likelihood updates), and why each EM iteration never lowers the log-likelihood
    - How a poor start is repaired iteration by iteration in 1-D, and how responsibilities and covariance ellipses evolve in 2-D (two videos you can play, pause and scrub)
    - Fit a `GaussianMixture` and use `predict`, `predict_proba`, `score_samples`, `score`, `bic`, `aic`, `icl` and `sample`
    - Inspect `weights_`, `means_` and `covariances_`, visualize them with `plot_gmm`, and `save`/`load` a model
    - Define your own 1-D and 2-D mixtures in the interactive version and fit them

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
    import io
    import warnings

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import norm

    from tgmm import GaussianMixture, plot_gmm, generate_gmm_data

    plt.rcParams['figure.dpi'] = 75  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    COLORS = ['#e41a1c', '#ff7f00', '#984ea3', '#4daf4a', '#377eb8', '#a65628']  # one colour per blob / component (up to 6)

    # Convergence warnings are expected below: several cells deliberately stop EM after one or a few iterations.
    warnings.filterwarnings('ignore', message='.*EM did not converge.*')
    return (
        COLORS,
        Ellipse,
        GaussianMixture,
        Image,
        SEED,
        base64,
        device,
        generate_gmm_data,
        io,
        linear_sum_assignment,
        norm,
        np,
        plot_gmm,
        plt,
        torch,
    )


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

    We start with the simplest case: one feature ($d=1$), generated from three true Gaussian components so that there is ground truth to compare the fit with. (This section and the next use `tgmm`'s internal `_e_step` / `_m_step` methods so each iteration is visible, with hand-rolled plotting so you can see exactly what is drawn; [06 -- Visualization](06_visualise.ipynb) introduces `plot_gmm`, which is used for the rest of this notebook.)
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
    plt.figure(figsize=(8, 4.5))
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
    return data, real_means, real_stds, true_taus


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 0: a (deliberately) bad initialization

    We initialize all three components close together around $9$ to $10$, even though the true components are centred at $-2$, $2$ and $6$, and give them a shared, overly wide initial standard deviation of $2.5$. No EM iteration has run yet: these are just the starting parameters, and the E-step computes the responsibilities $r_{ik}$ from the formula above using only this initial guess.

    Because the initial means sit together and far from where two of the three true clusters are, the responsibilities are not yet informative. This is the starting point EM has to improve on (it is the first frame of the video below).
    """)
    return


@app.cell
def _(data, torch):
    # Convert data to a torch tensor
    X_1d = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)
    initial_means = torch.tensor([[9.5], [10.0], [10.5]], dtype=torch.float32)
    # Initial parameters for the Gaussian Mixture Model
    initial_covariances = torch.tensor([[[2.5 ** 2]], [[2.5 ** 2]], [[2.5 ** 2]]], dtype=torch.float32)
    initial_weights = torch.tensor([0.33, 0.33, 0.33], dtype=torch.float32)
    return X_1d, initial_covariances, initial_means, initial_weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Full EM run: watching convergence

    Now we run EM to (near) convergence, one E-step and M-step at a time, recording $(\mu_k,\sigma_k,\pi_k)$ and the log-likelihood after every iteration.
    """)
    return


@app.cell
def _(
    FAST,
    GaussianMixture,
    MAX_ITER,
    X_1d,
    initial_covariances,
    initial_means,
    initial_weights,
    np,
    torch,
):
    _gmm_full = GaussianMixture(n_features=1, n_components=3, covariance_type='full', init_means=initial_means, init_covariances=initial_covariances, init_weights=initial_weights, device='cpu')
    _gmm_full._allocate_parameters(X_1d, set_random_state=False)
    means = [initial_means.squeeze().cpu().numpy()]  # parameters after iteration 0 (the initial guess), 1, 2, ...
    stds = [np.sqrt(initial_covariances[:, 0, 0].cpu().numpy())]
    taus = [initial_weights.cpu().numpy()]
    _resp, _lp = _gmm_full._e_step(X_1d)
    ll_history = [_lp.mean().item()]  # average log-likelihood of the initial guess
    for _step in range(MAX_ITER):
        _gmm_full._m_step(X_1d, _resp)  # M-step from the current responsibilities ...
        _resp, _lp = _gmm_full._e_step(X_1d)  # ... then the E-step under the updated parameters
        means.append(_gmm_full.means_.squeeze().cpu().numpy())
        stds.append(torch.sqrt(_gmm_full.covariances_[:, 0, 0]).cpu().numpy())
        taus.append(_gmm_full.weights_.cpu().numpy())
        ll_history.append(_lp.mean().item())
        if abs(ll_history[-1] - ll_history[-2]) < 1e-9:
            break
    means, stds, taus = np.array(means), np.array(stds), np.array(taus)
    ll_history = np.array(ll_history)
    GRID = 25 if FAST else 50
    print(f'EM iterations run: {len(ll_history) - 1}')
    print(f'Average log-likelihood: {ll_history[0]:.4f} (initial guess) -> {ll_history[-1]:.4f} (final)')
    print(f'Never decreases (largest drop between iterations: {max(0.0, -np.diff(ll_history).min()):.2e})')
    return GRID, ll_history, means, stds, taus


@app.cell
def _(GRID, data, means, norm, np, stds, taus):
    def log_likelihood(data, means_np, stds_np, taus_np):
        likelihood = np.zeros((data.shape[0], len(means_np)))
        for i in range(len(means_np)):
            likelihood[:, i] = norm.pdf(data, means_np[i], stds_np[i])
        total_likelihood = np.dot(likelihood, taus_np)
        return np.sum(np.log(total_likelihood))

    mean_range = np.linspace(min(data), max(data), GRID)
    std_range = np.linspace(0.01, 5, GRID)
    surfaces = [np.zeros((len(std_range), len(mean_range))) for _ in range(3)]
    for _i, _m in enumerate(mean_range):
        for _j, _s in enumerate(std_range):
            for _k in range(3):  # vary component _k's (mean, std); the others stay at their converged values
                _means_k = list(means[-1])
                _stds_k = list(stds[-1])
                _means_k[_k], _stds_k[_k] = _m, _s
                surfaces[_k][_j, _i] = log_likelihood(data, _means_k, _stds_k, taus[-1])
    return mean_range, std_range, surfaces


@app.cell
def _(FAST, Image, base64, io, mo, np, plt):
    ANIM_RC = {'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'figure.titlesize': 11, 'savefig.bbox': 'standard'}
    ANIM_DPI = 150  # a figure of 11 inches is 1650 px wide: crisp when the page scales it down to the notebook column
    ANIM_FPS = 4
    ANIM_HOLD = 8  # the last frame is repeated, so the end of the run stays on screen for a moment

    def pick_frames(n_iterations):
        """Iterations to animate, 0..n_iterations, denser early on (where EM changes fastest); always includes both ends."""
        _m = min(40 if FAST else 60, n_iterations + 1)
        return [int(_i) for _i in np.unique(np.round(n_iterations * np.linspace(0, 1, _m) ** 2).astype(int))]

    def _encode_mp4(imgs):
        """H.264 video (yuv420p, constant frame rate) of the frames, with the ffmpeg binary that ships with the imageio-ffmpeg package."""
        import pathlib
        import subprocess
        import tempfile

        import imageio_ffmpeg

        _w, _h = imgs[0].size
        with tempfile.TemporaryDirectory() as _dir:
            _path = pathlib.Path(_dir) / 'anim.mp4'  # a file, not a pipe: '+faststart' moves the index to the front, so the browser can start playing at once
            _cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{_w}x{_h}', '-r', str(ANIM_FPS), '-i', '-',
                    '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264', '-preset', 'slow', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(_path)]
            _proc = subprocess.Popen(_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for _img in imgs + [imgs[-1]] * ANIM_HOLD:
                _proc.stdin.write(_img.tobytes())
            _proc.stdin.close()
            _err = _proc.stderr.read().decode(errors='replace')
            if _proc.wait() != 0:
                raise RuntimeError(f'ffmpeg failed: {_err}')
            return _path.read_bytes()

    def anim_from_frames(draw, frames):
        """Render ``draw(t)`` (it returns a matplotlib figure) for every t in ``frames`` into a video the reader can play, pause and scrub.

        Every frame has the same size (the tgmm theme saves with a tight bounding box, which would crop each frame differently). The video is
        embedded in the page as a ``<video controls>`` (no autoplay). If no ffmpeg is available it falls back to a lossless animated WebP.
        """
        _imgs = []
        for _t in frames:
            with plt.rc_context(ANIM_RC):  # compact fonts for the small figure; no tight bbox
                _fig = draw(_t)
                _buf = io.BytesIO()
                _fig.savefig(_buf, format='png', dpi=ANIM_DPI)
            plt.close(_fig)
            _imgs.append(Image.open(_buf).convert('RGB'))
        try:
            _data = _encode_mp4(_imgs)
            _poster = io.BytesIO()
            _imgs[0].save(_poster, format='JPEG', quality=85)
            _html = (f'<video controls loop muted playsinline preload="metadata" poster="data:image/jpeg;base64,{base64.b64encode(_poster.getvalue()).decode()}" '
                     f'style="width:100%;height:auto"><source src="data:video/mp4;base64,{base64.b64encode(_data).decode()}" type="video/mp4"></video>')
        except (ImportError, OSError, RuntimeError):  # no usable ffmpeg: a lossless animated WebP instead
            _out = io.BytesIO()
            _imgs[0].save(_out, format='WEBP', save_all=True, append_images=_imgs[1:], lossless=True, method=4, loop=0, duration=[1000 // ANIM_FPS] * (len(_imgs) - 1) + [2500])
            _data = _out.getvalue()
            _html = f'<img src="data:image/webp;base64,{base64.b64encode(_data).decode()}" style="width:100%;height:auto">'
        return mo.Html(_html), len(_imgs), len(_data)

    return anim_from_frames, pick_frames


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The whole run, iteration by iteration

    The video replays the run above: **press play, or drag the timeline to scrub through the iterations**. Iteration $0$ is the bad initial guess and the last frame the converged fit (the frames are denser at the start, where EM changes fastest). At iteration $t$ the top row shows the fitted components against the real ones **(left)**, each point's responsibility for each component **(middle)**, and the average log-likelihood climbing over the iterations so far **(right)**; the bottom row shows, for each component, the **log-likelihood surface** over its $(\text{mean},\text{std})$ with the path EM has taken up to $t$ and the current estimate. The surface is the one EM is climbing: the other components are held at their converged values, and each estimate walks uphill to the maximum.
    """)
    return


@app.cell
def _(
    anim_from_frames,
    data,
    ll_history,
    mean_range,
    means,
    norm,
    np,
    pick_frames,
    plt,
    real_means,
    real_stds,
    std_range,
    stds,
    surfaces,
    taus,
    true_taus,
):
    def draw_1d_frame(_t):
        _colors = ['r', 'b', 'g']
        _fig = plt.figure(figsize=(11, 6.4))
        _gs = _fig.add_gridspec(2, 3)
        _ax_fit = _fig.add_subplot(_gs[0, 0])
        _ax_resp = _fig.add_subplot(_gs[0, 1])
        _ax_ll = _fig.add_subplot(_gs[0, 2])
        _x = np.linspace(min(data), max(data), 1000)
        _ax_fit.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data histogram')
        _est = 0
        _real = 0
        for _k in range(3):
            _comp = taus[_t, _k] * norm.pdf(_x, means[_t, _k], stds[_t, _k])
            _ax_fit.plot(_x, _comp, color=_colors[_k], linestyle='-', label=f'Gaussian {_k + 1} (estimated)')
            _ax_fit.plot(_x, true_taus[_k] * norm.pdf(_x, real_means[_k], real_stds[_k]), color=_colors[_k], linestyle='--', label=f'Gaussian {_k + 1} (real)')
            _est = _est + _comp
            _real = _real + true_taus[_k] * norm.pdf(_x, real_means[_k], real_stds[_k])
        _ax_fit.plot(_x, _est, 'k-', label='GMM (estimated)')
        _ax_fit.plot(_x, _real, 'k--', label='GMM (real)')
        _ax_fit.set_title('Data and Gaussian Distributions')
        _ax_fit.set_ylim(0, 1.7 * max(np.histogram(data, bins=30, density=True)[0].max(), _real.max()))  # headroom for the legend
        _ax_fit.legend(ncol=2, loc='upper left', handlelength=1.6, columnspacing=0.9)
        # responsibilities r_ik of the parameters after iteration t (the E-step formula, evaluated directly)
        _dens = taus[_t] * norm.pdf(data[:, None], means[_t], stds[_t])
        _r = _dens / np.maximum(_dens.sum(axis=1, keepdims=True), 1e-300)
        for _k in range(3):
            _ax_resp.scatter(data, _r[:, _k], color=_colors[_k], label=f'Responsibility of cluster {_k + 1}', alpha=0.6, marker='x', s=12)
        _ax_resp.set_title('Responsibilities')
        _ax_resp.legend()
        # average log-likelihood after each iteration so far (the quantity EM never decreases)
        _ax_ll.plot(ll_history, color='lightgray', linewidth=1)
        _ax_ll.plot(np.arange(_t + 1), ll_history[:_t + 1], color='k', linewidth=1.5, label='Up to this iteration')
        _ax_ll.scatter([_t], [ll_history[_t]], color='magenta', edgecolor='white', s=80, zorder=5, label='Current iteration')
        _lo, _hi = ll_history[min(2, len(ll_history) - 1):].min(), ll_history.max()  # zoom past the (terrible) initial guess so the slow climb is visible
        _pad = 0.1 * max(_hi - _lo, 1e-3)
        _ax_ll.set_ylim(_lo - _pad, _hi + _pad)
        _ax_ll.set_xlim(0, len(ll_history) - 1)
        if ll_history[0] < _lo - _pad:
            _ax_ll.text(0.07, 0.04, f'iteration 0: {ll_history[0]:.2f} (off scale)', transform=_ax_ll.transAxes, fontsize=7, va='bottom')
        _ax_ll.set_title('Average log-likelihood')
        _ax_ll.set_xlabel('EM iteration')
        _ax_ll.set_ylabel('Average log-likelihood')
        _ax_ll.legend(loc='lower right')
        for _k in range(3):
            _ax = _fig.add_subplot(_gs[1, _k])
            _cs = _ax.contourf(mean_range, std_range, surfaces[_k], levels=15, cmap='viridis')
            _fig.colorbar(_cs, ax=_ax).ax.set_ylabel('Log-Likelihood')
            _ax.scatter(real_means[_k], real_stds[_k], color='red', label='Real mean and std', s=50, marker='o')
            _ax.scatter(means[0, _k], stds[0, _k], color='red', label='Initial mean and std', s=100, marker='*')
            _ax.plot(means[:_t + 1, _k], stds[:_t + 1, _k], color='magenta', label='Path up to this iteration')
            _ax.scatter(means[_t, _k], stds[_t, _k], color='magenta', edgecolor='white', s=80, zorder=5, label='Current estimate')
            _ax.set_title(f'Log-Likelihood vs. Mean{_k + 1} and Std{_k + 1}')
            _ax.set_xlabel('Mean')
            _ax.set_ylabel('Standard Deviation')
            _ax.legend(loc='lower left')
        _fig.suptitle(f'EM iteration {_t} of {len(ll_history) - 1}: average log-likelihood {ll_history[_t]:.4f}')
        _fig.tight_layout(rect=[0, 0, 1, 0.96])
        return _fig

    anim_1d, n_frames_1d, anim_1d_bytes = anim_from_frames(draw_1d_frame, pick_frames(len(ll_history) - 1))
    print(f'1-D video: {n_frames_1d} frames, {anim_1d_bytes / 1e6:.2f} MB')
    anim_1d
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
def _(SEED, default_centers, default_covs, default_n_samples, device, generate_gmm_data):
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

    We fit one component per blob to the data above, starting from means placed on a horizontal line *above* the data, and record the state after every EM iteration. The helper below draws the three panels for a given iteration: **(left)** each point's responsibility per component, **(middle)** the hard assignment (arg-max responsibility, used only for colouring) in data space, and **(right)** the trajectory of each component's mean and its current $1\sigma$ covariance ellipse, with the initial means marked `x` and the current means `o`.
    """)
    return


@app.cell
def _(COLORS, Ellipse, torch):
    def plot_2d_gmm_resp(snapshot, X, em_means, axs, row):
        """One row of the 2-D EM figure: responsibilities, hard assignments, means trace and covariance ellipses."""
        _means, _covs, _resp, _iteration = snapshot
        _K = _means.shape[0]
        for _k in range(_K):
            axs[row, 0].plot(_resp[:, _k].cpu().numpy(), label=f'Cluster {_k + 1}', color=COLORS[_k], marker='.', markersize=2, linestyle='None', alpha=0.5)
        axs[row, 0].legend(markerscale=4)
        axs[row, 0].set_title(f'Responsibilities, iteration {_iteration}')
        axs[row, 0].set_xlabel('Data point index')
        axs[row, 0].set_ylabel('Probability')

        # colour each point by the component with the highest responsibility
        _assignment = _resp.cpu().max(dim=1)[1]
        axs[row, 1].scatter(X[:, 0], X[:, 1], c=[COLORS[_c] for _c in _assignment], s=10, alpha=0.5)
        axs[row, 1].set_title(f'Hard assignments, iteration {_iteration}')
        axs[row, 1].set_xlabel('X1')
        axs[row, 1].set_ylabel('X2')

        # data, trace of the means, and a 1-sigma ellipse per component
        axs[row, 2].scatter(X[:, 0], X[:, 1], c='gray', s=5, alpha=0.5)
        for _k in range(_K):
            _eigvals, _eigvecs = torch.linalg.eigh(_covs[_k].cpu())
            _angle = torch.atan2(_eigvecs[1, 0], _eigvecs[0, 0]).item() * 180.0 / torch.pi
            _width, _height = 2 * torch.sqrt(_eigvals).cpu().numpy()
            axs[row, 2].add_patch(Ellipse(_means[_k].cpu().numpy(), _width, _height, angle=_angle, edgecolor=COLORS[_k], facecolor='none', lw=2))
            _trace = torch.stack([_m[_k] for _m in em_means[:_iteration + 1]]).cpu().numpy()
            axs[row, 2].plot(_trace[:, 0], _trace[:, 1], color=COLORS[_k], linestyle='--')
            axs[row, 2].scatter(_trace[-1, 0], _trace[-1, 1], color=COLORS[_k], marker='o', s=50)
            axs[row, 2].scatter(_trace[0, 0], _trace[0, 1], color=COLORS[_k], marker='x', s=100)
        axs[row, 2].set_title(f'GMM, iteration {_iteration}')
        axs[row, 2].set_xlabel('X1')
        axs[row, 2].set_ylabel('X2')
        axs[row, 2].scatter([], [], color='black', marker='x', s=100, label='Initial means')
        axs[row, 2].scatter([], [], color='black', marker='o', s=50, label='Current means')
        axs[row, 2].plot([], [], color='black', linestyle='-', label='Gaussian components')
        axs[row, 2].plot([], [], color='black', linestyle='--', label='Trace of means')
        axs[row, 2].legend()

    return (plot_2d_gmm_resp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Running EM and recording every iteration

    One E-step and M-step at a time, we store the state after every iteration (convergence, or the iteration cap in the fast docs build). The video below replays it: **press play, or drag the timeline to scrub through the iterations**, and watch how the soft responsibilities sharpen into confident assignments as the ellipses settle onto the blobs.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, X_tensor, device, n_components, torch):
    # initial means: n_components points on a horizontal line above the data
    means_init = torch.stack([torch.linspace(-1.0, 2.0, n_components), torch.full((n_components,), 5.0)], dim=1).to(device)
    gmm_steps = GaussianMixture(n_features=2, n_components=n_components, covariance_type='full', init_means=means_init, device=device)
    gmm_steps._allocate_parameters(X_tensor)
    em_means = [gmm_steps.means_.clone()]  # means after iteration 0 (initial), 1, 2, ...
    _resp, _lp = gmm_steps._e_step(X_tensor)
    _ll_prev = _lp.mean().item()
    em_states = {0: (gmm_steps.means_.clone(), gmm_steps.covariances_.clone(), _resp.clone(), 0)}  # state after every iteration (0 = initial guess)
    em_ll = {0: _ll_prev}  # average log-likelihood after every iteration
    for _it in range(1, MAX_ITER + 1):
        gmm_steps._m_step(X_tensor, _resp)
        _resp, _lp = gmm_steps._e_step(X_tensor)
        em_means.append(gmm_steps.means_.clone())
        _ll = _lp.mean().item()
        em_states[_it] = (gmm_steps.means_.clone(), gmm_steps.covariances_.clone(), _resp.clone(), _it)
        em_ll[_it] = _ll
        if abs(_ll - _ll_prev) < 1e-6:
            break
        _ll_prev = _ll
    print(f'EM iterations recorded: {max(em_states)}')
    return em_ll, em_means, em_states


@app.cell
def _(
    X,
    anim_from_frames,
    em_ll,
    em_means,
    em_states,
    pick_frames,
    plot_2d_gmm_resp,
    plt,
):
    def draw_2d_frame(_t):
        _fig, _axs = plt.subplots(1, 3, figsize=(11, 3.9), squeeze=False)
        plot_2d_gmm_resp(em_states[_t], X, em_means, _axs, 0)
        _fig.suptitle(f'EM iteration {_t} of {max(em_states)}: average log-likelihood {em_ll[_t]:.4f}')
        _fig.tight_layout(rect=[0, 0, 1, 0.94])
        return _fig

    anim_2d, n_frames_2d, anim_2d_bytes = anim_from_frames(draw_2d_frame, pick_frames(max(em_states)))
    print(f'2-D video: {n_frames_2d} frames, {anim_2d_bytes / 1e6:.2f} MB')
    anim_2d
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
    return generated_probs, gmm, gmm_samples, log_probs, probs, y_pred


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

    - **`gmm.fit(X)`** (top left): the learned `means_` and `covariances_`, drawn as the fitted covariance ellipses over the data. The box in its corner shows the information criteria `gmm.bic(X)`, `gmm.aic(X)` and `gmm.icl(X)` (lower is better; defined below the plots).
    - **`gmm.score_samples(X)`** (top right): each point coloured by its log-likelihood $\log p(x_i\mid\theta)$; points far from every component are the unlikely ones. The title also shows its average, `gmm.score(X)`.
    - **`gmm.predict(X)`** (bottom left): the hard cluster label of every point.
    - **`gmm.sample(n)`** (bottom right): new points drawn from the fitted mixture (shown without the ellipses, so you can compare their spread with the data; more in [05 -- Sampling](05_sampling.ipynb)). If the model is good they look like the original data.

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
    plot_gmm(X=X, gmm=gmm, ax=_axs[0, 0], title='gmm.fit(X)\nmeans_ and covariances_', legend_labels=legend_labels)
    plot_gmm(X=X, ax=_axs[0, 1], title=f'gmm.score_samples(X)\nmean = gmm.score(X) = {gmm.score(X_tensor):.3f}', log_probs=log_probs, colorbar_label='Log-Likelihood')
    plot_gmm(X=X, ax=_axs[1, 0], true_labels=y_pred, title='gmm.predict(X)', legend_labels=[f'Cluster {i}' for i in range(n_components)])
    plot_gmm(X=gmm_samples, gmm=gmm, ax=_axs[1, 1], title='gmm.sample(n)', show_ellipses=False)
    _ic = f'gmm.bic(X) = {gmm.bic(X_tensor):.1f}\ngmm.aic(X) = {gmm.aic(X_tensor):.1f}\ngmm.icl(X) = {gmm.icl(X_tensor):.1f}'  # lower is better
    _axs[0, 0].text(0.02, 0.02, _ic, transform=_axs[0, 0].transAxes, ha='left', va='bottom', fontsize=8, family='monospace', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='0.7'))
    _fig.tight_layout()
    plt.gcf()
    return


@app.cell
def _(X, n_components, plot_gmm, plt, probs):
    _nrows = (n_components + 1) // 2
    _fig, _axs = plt.subplots(_nrows, 2, figsize=(9, 3.8 * _nrows), squeeze=False)
    for _k, _ax in enumerate(_axs.ravel()):
        if _k < n_components:
            plot_gmm(X=X, ax=_ax, title=f'Component {_k + 1} Probability', log_probs=probs[:, _k], colormap='RdBu', colorbar_label='Probability')
        else:
            _ax.axis('off')
    plt.suptitle('gmm.predict_proba(X): responsibilities of the observed data', fontsize=11)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(generated_probs, gmm, gmm_samples, n_components, plot_gmm, plt):
    _nrows = (n_components + 1) // 2
    _fig, _axs = plt.subplots(_nrows, 2, figsize=(9, 3.8 * _nrows), squeeze=False)
    for _k, _ax in enumerate(_axs.ravel()):
        if _k < n_components:
            plot_gmm(X=gmm_samples, gmm=gmm, ax=_ax, show_ellipses=False, title=f'Component {_k + 1} Probability', log_probs=generated_probs[:, _k], colormap='PRGn', colorbar_label='Probability')
        else:
            _ax.axis('off')
    _fig.suptitle('gmm.predict_proba(gmm_samples): responsibilities of the generated samples', fontsize=11)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Information criteria

    The box in the top-left panel above shows three information criteria of the fitted model (lower is better). With $n\,\text{score}(X)$ the maximized log-likelihood and $p$ the parameter count from section 2: **BIC** is $-2\,n\,\text{score}+p\log n$ (`bic(X)`), **AIC** is $-2\,n\,\text{score}+2p$ (`aic(X)`), and **ICL**, the integrated completed likelihood (`icl(X, mode='hard' or 'soft')`), is BIC plus the entropy of the assignments, $-2\sum_i\log r_{i\hat z_i}$ (hard) or $-2\sum_{i,k}r_{ik}\log r_{ik}$ (soft), so it also penalizes overlapping components. How to use them to choose $K$ is the subject of [13 -- Model Selection](13_model_selection.ipynb); new data from the fitted mixture come from `sample(n)` ([05 -- Sampling](05_sampling.ipynb)). The cell below checks the three criteria against their definitions.
    """)
    return


@app.cell
def _(X_tensor, gmm, n_components, np, probs):
    _n, _d = X_tensor.shape
    _p = (n_components - 1) + n_components * _d + n_components * _d * (_d + 1) // 2  # full covariance
    _bic = -2 * _n * gmm.score(X_tensor) + _p * np.log(_n)
    _r = np.clip(probs.detach().cpu().numpy().astype(np.float64), 1e-300, 1.0)  # responsibilities r_ik
    _icl_hard = _bic - 2 * np.log(_r.max(axis=1)).sum()    # BIC - 2 sum_i log r_{i, z_hat_i}
    _icl_soft = _bic - 2 * (_r * np.log(_r)).sum()         # BIC - 2 sum_ik r_ik log r_ik
    print(f'free parameters p = (K-1) + K*d + K*d*(d+1)/2 = {_p}')
    print(f'bic  from the definition: {_bic:.3f}   gmm.bic(X) = {gmm.bic(X_tensor):.3f}')
    print(f'aic  from the definition: {-2 * _n * gmm.score(X_tensor) + 2 * _p:.3f}   gmm.aic(X) = {gmm.aic(X_tensor):.3f}')
    print(f'icl (hard) from the definition: {_icl_hard:.3f}   gmm.icl(X, "hard") = {gmm.icl(X_tensor, "hard"):.3f}')
    print(f'icl (soft) from the definition: {_icl_soft:.3f}   gmm.icl(X, "soft") = {gmm.icl(X_tensor, "soft"):.3f}')
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
    ## 9. Design Your Own Mixture (interactive version)

    The page you are reading shows the fixed default data. To specify the mixture yourself, open the interactive version of this notebook (the molab link at the top of the page, or `marimo edit` locally): the number fields below only work there, and all of this notebook's interactive controls live in this last section. Each designer shows static results, no videos: one `GaussianMixture` fit to your data and its plot.

    ### A 1-D mixture of your own

    For every component you type its **mean** $\mu_k$, its **standard deviation** $\sigma_k$ and its **number of points** $n_k$ (so the true mixing weight is $\pi_k=n_k/n$). The **number of components** field ($K=2,\dots,4$) adds or removes components, and **Reset** restores the defaults, the three components of section 3. The standard deviation must be positive and $n_k\ge 1$; otherwise the panel warns you and uses the closest valid value. The result is one fit with `GaussianMixture(n_components=K, n_features=1)` drawn over the histogram of your data, next to the true components.
    """)
    return


@app.cell
def _(SEED, mo, np):
    # docs: hide-cell
    # (mean, std, n) per component: the three defaults, then a spare fourth for "add a component"
    DEFAULT_ROWS_1D = [(-2.0, 2.0, 2000.0), (2.0, 1.0, 1000.0), (6.0, 3.0, 3000.0), (10.0, 1.5, 1000.0)]
    # Fixed standard normals, so that changing one field never reshuffles the other components' points. Each slot reads its own slice of
    # the pool (the defaults read the slices the original example always used) and, when n exceeds the slot's default size, a second fixed stream.
    NOISE_POOL_1D = np.random.RandomState(SEED).standard_normal(13000)
    SLOT_OFFSETS_1D = [6000, 8000, 9000, 12000]
    SLOT_CAPS_1D = [2000, 1000, 3000, 1000]

    def draw_component_1d(k, mean, std, n):
        """n points of component k: mean + std * (fixed standard normals)."""
        _z = NOISE_POOL_1D[SLOT_OFFSETS_1D[k]:SLOT_OFFSETS_1D[k] + min(n, SLOT_CAPS_1D[k])]
        if n > SLOT_CAPS_1D[k]:
            _z = np.concatenate([_z, np.random.RandomState(SEED + 1 + k).standard_normal(n - SLOT_CAPS_1D[k])])
        return mean + std * _z

    reset_1d = mo.ui.button(label='Reset to the default 1-D mixture')
    return DEFAULT_ROWS_1D, draw_component_1d, reset_1d


@app.cell
def _(DEFAULT_ROWS_1D, mo, reset_1d):
    # docs: hide-cell
    _ = reset_1d.value  # pressing Reset re-runs this cell, which recreates every field at its default
    _spec = [('mean', -50, 50, 0.1), ('std', 0.0, 30, 0.1), ('n', 1, 20000, 1)]
    # mo.ui.array (not a plain list): marimo only re-runs dependent cells when a field's value changes if the field sits in an array
    fields_1d = mo.ui.array([mo.ui.array([mo.ui.number(start=_lo, stop=_hi, step=_st, value=_v, label=_nm, debounce=True)
                                          for (_nm, _lo, _hi, _st), _v in zip(_spec, DEFAULT_ROWS_1D[_i])]) for _i in range(4)])
    k_1d = mo.ui.number(start=2, stop=4, step=1, value=3, label='number of components K', debounce=True)
    return fields_1d, k_1d


@app.cell
def _(fields_1d, k_1d, mo, reset_1d):
    # docs: hide-cell
    def _panel(_i):
        _mean, _std, _n = fields_1d.value[_i]
        _f = fields_1d[_i]
        _note = '' if (_std > 0 and _n >= 1) else '<small>⚠️ need std > 0 and n ≥ 1; the closest valid value is used</small>'
        return mo.vstack([mo.md(f'**Component {_i + 1}**  ·  mean, standard deviation, number of points n'),
                          mo.hstack([_f[0], _f[1], _f[2]], justify='start'),
                          mo.md(_note)])

    mo.vstack([
        mo.hstack([k_1d, reset_1d], justify='start', align='end'),
        *[_panel(_i) for _i in range(int(k_1d.value))],
    ])
    return


@app.cell
def _(GaussianMixture, MAX_ITER, SEED, draw_component_1d, fields_1d, k_1d, mo, norm, np, plt, torch):
    # The fields' values, clipped to valid ones
    _rows = [(float(_m), max(float(_s), 0.05), max(int(_n), 1)) for _m, _s, _n in fields_1d.value[:int(k_1d.value)]]
    _K = len(_rows)
    custom_data_1d = np.concatenate([draw_component_1d(_k, *_r) for _k, _r in enumerate(_rows)])
    _true_w = np.array([_r[2] for _r in _rows], float) / sum(_r[2] for _r in _rows)
    _X = torch.tensor(custom_data_1d, dtype=torch.float32).reshape(-1, 1)
    try:
        custom_gmm_1d = GaussianMixture(n_components=_K, n_features=1, random_state=SEED, max_iter=MAX_ITER, device='cpu')
        custom_gmm_1d.fit(_X)
    except Exception as _e:  # e.g. too few points for the number of components
        custom_gmm_1d = None
        _out = mo.md(f'⚠️ the fit failed for this mixture ({_e}); try more points or a larger standard deviation')
    else:
        _w = custom_gmm_1d.weights_.cpu().numpy()
        _mu = custom_gmm_1d.means_.cpu().numpy()[:, 0]
        _sd = np.sqrt(custom_gmm_1d.covariances_.cpu().numpy()[:, 0, 0])
        print(f'fitted GMM: BIC = {custom_gmm_1d.bic(_X):.1f}, AIC = {custom_gmm_1d.aic(_X):.1f}')
        for _k in np.argsort(_mu):
            print(f'  weight {_w[_k]:.3f}  mean {_mu[_k]:7.3f}  std {_sd[_k]:6.3f}')
        print('true mixture:')
        for _r, _tw in zip(_rows, _true_w):
            print(f'  weight {_tw:.3f}  mean {_r[0]:7.3f}  std {_r[1]:6.3f}')
        _x = np.linspace(custom_data_1d.min(), custom_data_1d.max(), 800)
        plt.figure(figsize=(8, 4.5))
        plt.hist(custom_data_1d, bins=40, density=True, alpha=0.5, color='g', label='your data')
        for _k in np.argsort(_mu):
            plt.plot(_x, _w[_k] * norm.pdf(_x, _mu[_k], _sd[_k]), color='C0', label='fitted component' if _k == np.argsort(_mu)[0] else None)
        plt.plot(_x, sum(_w[_k] * norm.pdf(_x, _mu[_k], _sd[_k]) for _k in range(_K)), 'k-', label='fitted GMM')
        plt.plot(_x, sum(_tw * norm.pdf(_x, _r[0], _r[1]) for _r, _tw in zip(_rows, _true_w)), 'k--', label='true GMM')
        plt.title(f'1-D mixture of your own: K = {_K} fitted components')
        plt.legend()
        _out = plt.gcf()
    _out
    return custom_data_1d, custom_gmm_1d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Your own 2-D blobs

    For every blob you type the centre $(x, y)$, the covariance matrix

    $$
    \Sigma = \begin{pmatrix}\Sigma_{11} & \Sigma_{12}\\ \Sigma_{12} & \Sigma_{22}\end{pmatrix}
    $$

    (symmetric, so you enter $\Sigma_{11}$, $\Sigma_{12}=\Sigma_{21}$ and $\Sigma_{22}$) and the number of points $n$. A valid covariance has $\Sigma_{11}>0$, $\Sigma_{22}>0$ and $\Sigma_{12}^2<\Sigma_{11}\Sigma_{22}$ (positive definite); the panel shows the standard deviations $\sqrt{\Sigma_{11}},\sqrt{\Sigma_{22}}$ and the correlation $\rho=\Sigma_{12}/\sqrt{\Sigma_{11}\Sigma_{22}}$, and warns you if the matrix is not valid (it then uses the closest valid one). Equal variances and $\Sigma_{12}=0$ give a spherical blob, $\Sigma_{12}=0$ with unequal variances an axis-aligned one, and $\Sigma_{12}\neq 0$ a tilted ("full") one. The **number of blobs** field ($K = 1,\dots,6$) adds or removes blobs, and **Reset** restores the four default blobs of section 4. The result is one fit with the default constructor, `GaussianMixture(n_components=K)`: its plot next to your data and its `score`, `bic`, `aic` and `icl`.
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
    custom_labels = custom_y_tensor.cpu().numpy()
    return custom_X, custom_X_tensor, custom_labels, custom_n_samples


@app.cell
def _(GaussianMixture, MAX_ITER, SEED, custom_X, custom_X_tensor, custom_labels, custom_n_samples, mo, plot_gmm, plt):
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
        _fig, _axs = plt.subplots(1, 2, figsize=(11, 4.6))
        plot_gmm(X=custom_X, true_labels=custom_labels, ax=_axs[0], title='Your data', legend_labels=[f'Blob {_i + 1}' for _i in range(_K)])
        plot_gmm(X=custom_X, gmm=custom_gmm, ax=_axs[1], title=f'gmm.fit(X) with K = {_K}')
        _fig.tight_layout()
        _out = plt.gcf()
    _out
    return custom_gmm,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to Go Next

    This notebook fitted one kind of mixture (`covariance_type='full'`, maximum likelihood, EM from a k-means++ start). Each of the following changes one ingredient:

    - **The shape of each component:** diagonal, spherical, shared (`tied_*`) and the 14 GPCM structures, in [02 -- Covariance Types](02_covariance_types.ipynb).
    - **The starting point and restarts:** `init_means`, `init_weights`, `init_covariances`, `n_init`, `warm_start`, in [03 -- Initialization](03_initialization.ipynb).
    - **Drawing new data, including constrained sampling:** [05 -- Sampling](05_sampling.ipynb).
    - **Hard assignments:** `algorithm='cem'` (Classification EM), in [07 -- CEM](07_cem.ipynb).
    - **Regularizing the estimates:** priors and MAP estimation, in [04 -- Priors](04_priors.ipynb) (MAP and the NIW prior).
    - **Choosing $K$:** [13 -- Model Selection](13_model_selection.ipynb).
    """)
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

    **Next:** [02 -- Covariance Types](02_covariance_types.ipynb), then [03 -- Initialization](03_initialization.ipynb) and [05 -- Sampling](05_sampling.ipynb).
    """)
    return


if __name__ == "__main__":
    app.run()
