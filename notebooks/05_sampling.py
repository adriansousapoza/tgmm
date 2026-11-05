# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "tgmm",
#     "torch; sys_platform != 'emscripten'",
#     "numpy",
#     "matplotlib",
# #     "scipy",
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
    N_DRAW = 40 if FAST else 500  # samples drawn per constrained-sampling example
    N_BIG = 4000 if FAST else 50000  # samples drawn for the "does the sample match the model" checks
    return MAX_ITER, N_BIG, N_DRAW, N_INIT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sampling from Gaussian Mixture Models

    A fitted mixture is also a **generative model**: it says how data like yours could have been produced, so you can ask it for more. `GaussianMixture.sample` draws new points by first picking a component according to the mixture weights and then sampling from that Gaussian. It can also restrict the draw to one component, to a shell of standard deviations, to a confidence region, or to a ball around a point you choose, using rejection sampling.

    **What you'll learn**

    - The two-stage (ancestral) sampling scheme, and a check that `sample` really implements it
    - How `sample` handles every covariance type, including the eight GPCM-only models
    - What `sample` returns, and how to draw from one component only
    - Constraints by standard-deviation radius or range, by confidence level and by distance from a point, and how their rejection sampling behaves
    - Why "1 standard deviation" captures far less mass in high dimensions, and how to choose a radius for a target probability

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb) (the model and EM), [02 -- Covariance Types](02_covariance_types.ipynb) and [03 -- Initialization](03_initialization.ipynb).

    **See also:** [`GaussianMixture.sample`](../api/gaussian-mixture.md) in the API reference; [06 -- Visualization](06_visualise.ipynb) for the plotting tools used below.
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

    from tgmm import GaussianMixture, dynamic_figsize, plot_gmm
    from tutorial_data import make_default_data

    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; keep embedded figure outputs small

    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return GaussianMixture, SEED, chi2, device, dynamic_figsize, make_default_data, np, plot_gmm, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A Fitted Model

    We fit a 4-component full-covariance model to the default four blobs of [1. Introduction](01_introduction.ipynb). Everything below samples from this fitted model, so the "ground truth" for the checks is the model's own `weights_`, `means_` and `covariances_`.
    """)
    return


@app.cell
def _(SEED, device, make_default_data):
    X_default, _ = make_default_data(SEED, device=device)
    X = X_default.cpu().numpy()
    legend_labels = [f'Component {_i + 1}' for _i in range(4)]
    return X, legend_labels


@app.cell
def _(GaussianMixture, MAX_ITER, N_INIT, SEED, X, device, torch):
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    gmm = GaussianMixture(n_components=4, covariance_type='full', max_iter=MAX_ITER, init_means='random', n_init=N_INIT, random_state=SEED, device=device)
    gmm.fit(X_tensor)
    print('Converged:', gmm.converged_, f'after {gmm.n_iter_} iterations')
    print('weights_:', [round(_w, 3) for _w in gmm.weights_.tolist()])
    return (gmm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How Sampling Works

    A Gaussian mixture is a two-stage process, and sampling just runs it forwards (this is called *ancestral* sampling):

    1. Draw a component index $z \sim \mathrm{Categorical}(\pi_1,\dots,\pi_K)$.
    2. Draw $x \mid z=k \sim \mathcal{N}(\mu_k, \Sigma_k)$, i.e. $x = \mu_k + L_k\, g$ with $g \sim \mathcal{N}(0, I)$ and $L_k L_k^\top = \Sigma_k$ (a Cholesky factor).

    Then $(z, x)$ has the joint density $\pi_z\,\mathcal{N}(x;\mu_z,\Sigma_z)$, so $x$ alone is distributed as the mixture $\sum_k \pi_k \mathcal{N}(x;\mu_k,\Sigma_k)$, and $z$ is the *true* component that produced $x$.

    `sample(n_samples)` returns `(samples, indices)`: a tensor of shape `(n_samples, n_features)` and a long tensor of shape `(n_samples,)` holding $z$ for every point. Unlike `predict`, these indices are not a guess from the posterior: they are the component the point was actually drawn from.

    The cell below implements the two steps by hand with NumPy and compares the result with `gmm.sample` on a large draw. The per-component frequencies should match $\pi_k$, and the per-component sample means and covariances should match $\mu_k$ and $\Sigma_k$, up to Monte-Carlo error of order $1/\sqrt{n}$.
    """)
    return


@app.cell
def _(N_BIG, SEED, gmm, np):
    _pi = gmm.weights_.cpu().numpy().astype(float)
    _mu = gmm.means_.cpu().numpy().astype(float)
    _S = gmm.covariances_.cpu().numpy().astype(float)
    _K, _d = _mu.shape

    _rng = np.random.default_rng(SEED)
    _z = _rng.choice(_K, size=N_BIG, p=_pi / _pi.sum())                       # step 1
    _L = np.linalg.cholesky(_S)                                                # (K, d, d)
    _by_hand = _mu[_z] + np.einsum('nij,nj->ni', _L[_z], _rng.standard_normal((N_BIG, _d)))  # step 2

    _drawn, _idx = gmm.sample(N_BIG)
    _drawn, _idx = _drawn.cpu().numpy(), _idx.cpu().numpy()

    print(f'Drawing {N_BIG} points: by hand vs gmm.sample')
    print('component   pi_k   freq (hand)  freq (sample)   max|mean err| (sample)   max|cov err| (sample)')
    for _k in range(_K):
        _sel = _drawn[_idx == _k]
        print(f'{_k:^9d}  {_pi[_k]:.3f}   {np.mean(_z == _k):^10.3f}   {len(_sel) / N_BIG:^12.3f}   '
              f'{np.abs(_sel.mean(0) - _mu[_k]).max():^22.3f}   {np.abs(np.cov(_sel.T) - _S[_k]).max():^20.3f}')
    print('hand-rolled sample, overall mean error vs mixture mean:',
          np.abs(_by_hand.mean(0) - (_pi[:, None] * _mu).sum(0)).max().round(3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Every covariance type samples the same way

    Step 2 only needs a matrix square root of the component covariance, and every covariance structure provides one:

    | `covariance_type` | stored `covariances_` | covariance used for $x \mid z=k$ |
    |---|---|---|
    | `'full'` | `(K, d, d)` | $\Sigma_k$ |
    | `'diag'` | `(K, d)` | $\operatorname{diag}(\sigma_k^2)$ |
    | `'spherical'` | `(K,)` | $\sigma_k^2 I$ |
    | `'tied_full'` | `(d, d)` | $\Sigma$, shared by all components |
    | `'tied_diag'` | `(d,)` | $\operatorname{diag}(\sigma^2)$, shared |
    | `'tied_spherical'` | scalar | $\sigma^2 I$, shared |
    | the 8 GPCM-only codes `VEI`, `EVI`, `VEE`, `EVE`, `EEV`, `VVE`, `VEV`, `EVV` | `(K, d, d)` | $\Sigma_k = \lambda_k D_k A_k D_k^\top$ assembled as a dense matrix (see [02 -- Covariance Types](02_covariance_types.ipynb)) |

    For the diagonal and spherical types the Cholesky factor is just the square root of the entries, so `sample` computes $x=\mu_k+\sigma_k\odot g$ directly instead of building $d\times d$ matrices. In every case only the stored parameters of the fitted model matter: a model with a tied or constrained structure *samples* from that structure, not from the data's empirical covariance.

    (Before version 0.4.2, `sample` failed for the eight GPCM-only models; they are covered in the check below.)

    The cell fits one model per type on the same data, draws a large sample from each, and compares the covariance of the points labelled $k$ with the covariance the model says component $k$ has.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, N_BIG, SEED, X, device, np, torch):
    _Xt = torch.tensor(X, dtype=torch.float32, device=device)
    _d = _Xt.shape[1]

    def _full_cov(_c, _k, _name):
        _c = _c.detach().cpu().numpy().astype(float)
        if _name == 'diag':
            return np.diag(_c[_k])
        if _name == 'spherical':
            return _c[_k] * np.eye(_d)
        if _name == 'tied_full':
            return _c
        if _name == 'tied_diag':
            return np.diag(_c)
        if _name == 'tied_spherical':
            return float(_c) * np.eye(_d)
        return _c[_k]  # 'full' and the GPCM codes are stored as (K, d, d)

    print(f'{"covariance_type":<16}{"stored shape":<16}{"max |sample cov - model cov|":>30}')
    for _name in ['full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical', 'VEI', 'EVV']:
        _g = GaussianMixture(n_components=4, covariance_type=_name, init_means='kmeans', max_iter=MAX_ITER, random_state=SEED, device=device).fit(_Xt)
        _s, _i = _g.sample(N_BIG)
        _s, _i = _s.cpu().numpy(), _i.cpu().numpy()
        _err = max(np.abs(np.cov(_s[_i == _k].T) - _full_cov(_g.covariances_, _k, _name)).max() for _k in range(4) if (_i == _k).sum() > 10)
        print(f'{_name:<16}{str(tuple(_g.covariances_.shape)):<16}{_err:>30.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sampling from Specific Components

    The `component` argument fixes step 1 instead of drawing it:

    - **`component=None` (default):** $z$ is drawn from the mixture weights, so a draw of $n$ points contains about $n\pi_k$ points from component $k$.
    - **`component=k`:** every point comes from component $k$ ($k\in\{0,\dots,K-1\}$) and `indices` is constant.

    That is useful for looking at one fitted component in isolation, for generating data with known labels, and for balanced synthetic data (draw the same number from every component).
    """)
    return


@app.cell
def _(N_DRAW, gmm, np):
    component_samples = {}
    for _k in range(gmm.n_components):
        _s, _i = gmm.sample(N_DRAW, component=_k)
        component_samples[_k] = _s.detach().cpu().numpy()
        print(f'component={_k}: {len(_s)} samples, unique indices {np.unique(_i.cpu().numpy())}, weight pi_k = {gmm.weights_[_k].item():.3f}')
    mixed_samples, _mixed_idx = gmm.sample(N_DRAW * gmm.n_components)
    mixed_samples = mixed_samples.detach().cpu().numpy()
    print()
    print('component=None, label counts: ', np.bincount(_mixed_idx.cpu().numpy(), minlength=gmm.n_components))
    print('expected from the weights:     ', (gmm.weights_.cpu().numpy() * len(mixed_samples)).round().astype(int))
    return component_samples, mixed_samples


@app.cell
def _(component_samples, dynamic_figsize, gmm, legend_labels, mixed_samples, plot_gmm, plt, X):
    _nrows, _ncols = (3, 2)
    _fig, _axs = plt.subplots(_nrows, _ncols, figsize=dynamic_figsize(_nrows, _ncols))
    _axs = _axs.flatten()
    _colors = ['C0', 'C1', 'C2', 'C3']
    for _k in range(gmm.n_components):
        plot_gmm(X=component_samples[_k], gmm=gmm, ax=_axs[_k], title=f'component={_k}', legend_labels=[f'Component {_k + 1}'], ellipse_fill=False, color_by_cluster=True, point_color=_colors[_k])
    plot_gmm(X=mixed_samples, gmm=gmm, ax=_axs[4], title='component=None (mixture)', legend_labels=legend_labels, ellipse_fill=False)
    plot_gmm(X=X, gmm=gmm, ax=_axs[5], title='Original Data', legend_labels=legend_labels, ellipse_fill=False)
    plt.suptitle('Component-specific vs mixed sampling')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Constrained Sampling

    Besides plain sampling, `sample` can keep only points that satisfy a distance constraint. The distance is the **Mahalanobis distance** to the component the point was drawn from,

    $$
    \delta_k(x) = \sqrt{(x-\mu_k)^\top \Sigma_k^{-1} (x-\mu_k)},
    $$

    which is the number of standard deviations along the direction of $x-\mu_k$: it equals 1 on the $1\sigma$ ellipse that `plot_gmm` draws. The available constraints are

    | argument | keeps points with | meaning |
    |---|---|---|
    | `std_radius=r` | $\delta_k \le r$ | inside the $r\sigma$ ellipse |
    | `std_range=(a, b)` | $a \le \delta_k \le b$ | a shell; `b=float('inf')` gives the outliers beyond $a\sigma$ |
    | `confidence=p` | $\delta_k \le \sqrt{\chi^2_{d,p}}$ | inside the ellipse holding probability mass $p$ |
    | `confidence_range=(p_lo, p_hi)` | between the two confidence radii | a shell by probability mass |
    | `center_point=c, center_radius=R` | $\lVert x-c\rVert \le R$ | a Euclidean ball around any location, ignoring $\Sigma_k$ |

    Three behaviours are worth knowing, because they follow directly from how the constraint is implemented:

    - **One constraint at a time.** Combining `std_radius`, `std_range`, `confidence`, `confidence_range` and the centre constraint raises `ValueError`; `center_point` and `center_radius` must be given together.
    - **Each point is drawn, then accepted or rejected.** For every output point `sample` first picks its component (from the weights, or from `component=`) and then redraws from that component until the constraint holds. The returned labels therefore still follow $\pi_k$, **not** the probability mass each component has inside the constrained region. That matters for `center_point`: a component that mostly lies outside the ball is still chosen with probability $\pi_k$ and must be redrawn many times.
    - **Rejection can fail.** If a point is not accepted after `max_attempts_per_sample` draws (default 1000) a `RuntimeError` is raised. Raise the limit for thin or far-out regions.

    With `noise=True` only plain `sample(n)` is available (noise points get index `-1`); every constraint raises `NotImplementedError`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Radius, range and confidence

    Each panel samples one component only (`component=k`) so the effect of the constraint on one ellipse is visible. The first plot uses `std_radius=1`.
    """)
    return


@app.cell
def _(N_DRAW, X, dynamic_figsize, gmm, legend_labels, plot_gmm, plt):
    _fig, _axs = plt.subplots(3, 2, figsize=dynamic_figsize(3, 2))
    _axs = _axs.flatten()
    _colors = ['C0', 'C1', 'C2', 'C3']
    for _k in range(gmm.n_components):
        _s, _ = gmm.sample(N_DRAW, component=_k, std_radius=1.0)
        plot_gmm(X=_s.detach().cpu().numpy(), gmm=gmm, ax=_axs[_k], title=f'Component {_k + 1}: within 1σ', legend_labels=[f'Component {_k + 1} (1σ)'], ellipse_fill=False, color_by_cluster=False, point_color=_colors[_k])
    _s, _ = gmm.sample(N_DRAW * gmm.n_components, std_radius=1.0)
    plot_gmm(X=_s.detach().cpu().numpy(), gmm=gmm, ax=_axs[4], title='Mixture: within 1σ', legend_labels=legend_labels, ellipse_fill=False)
    plot_gmm(X=X, gmm=gmm, ax=_axs[5], title='Original Data', legend_labels=legend_labels, ellipse_fill=False)
    plt.suptitle('std_radius=1.0')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Shells and confidence regions, one per component (components are numbered in fit order): a $2$--$3\sigma$ shell, a $1$--$2\sigma$ shell, the 95% confidence region, and the tail beyond $3\sigma$ (an "outlier" generator; `max_attempts_per_sample` is raised because few draws land that far out).
    """)
    return


@app.cell
def _(N_DRAW, X, dynamic_figsize, gmm, legend_labels, plot_gmm, plt):
    _specs = [
        ('Component 1: std_range=(2, 3)', dict(component=0, std_range=(2, 3))),
        ('Component 2: std_range=(1, 2)', dict(component=1, std_range=(1, 2))),
        ('Component 3: confidence=0.95', dict(component=2, confidence=0.95)),
        ('Component 4: std_range=(3, inf)', dict(component=3, std_range=(3, float('inf')), max_attempts_per_sample=5000)),
    ]
    _fig, _axs = plt.subplots(3, 2, figsize=dynamic_figsize(3, 2))
    _axs = _axs.flatten()
    _colors = ['C0', 'C1', 'C2', 'C3']
    for _k, (_title, _kw) in enumerate(_specs):
        _s, _ = gmm.sample(N_DRAW, **_kw)
        plot_gmm(X=_s.detach().cpu().numpy(), gmm=gmm, ax=_axs[_k], title=_title, legend_labels=[f'Component {_k + 1}'], ellipse_fill=False, color_by_cluster=False, point_color=_colors[_k])
    _s, _ = gmm.sample(N_DRAW * gmm.n_components, confidence_range=(0.68, 0.95))
    plot_gmm(X=_s.detach().cpu().numpy(), gmm=gmm, ax=_axs[4], title='Mixture: confidence_range=(0.68, 0.95)', legend_labels=legend_labels, ellipse_fill=False)
    plot_gmm(X=X, gmm=gmm, ax=_axs[5], title='Original Data', legend_labels=legend_labels, ellipse_fill=False)
    plt.suptitle('Shells and confidence regions')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling around a fixed centre point

    `center_point` and `center_radius` restrict samples to a Euclidean ball around an arbitrary location, for example to generate synthetic data near a point of interest. The red circle in each panel is the constraint. Component indices follow the fit, not the order in which the blobs were generated: the component centred near $(2,-2)$ lies just outside the ball around $(1,0)$ of radius 2 (its centre is about 2.2 away, so only about 30% of its draws are accepted), yet it is still picked with its full probability $\pi_k$ in the mixture draw.
    """)
    return


@app.cell
def _(N_DRAW, X, dynamic_figsize, gmm, legend_labels, plot_gmm, plt, torch):
    center_point = torch.tensor([1.0, 0.0])
    center_radius = 2.0
    _fig, _axs = plt.subplots(3, 2, figsize=dynamic_figsize(3, 2))
    _axs = _axs.flatten()
    _colors = ['C0', 'C1', 'C2', 'C3']
    _c = center_point.cpu().numpy()
    for _k in range(gmm.n_components):
        _s, _ = gmm.sample(N_DRAW, component=_k, center_point=center_point, center_radius=center_radius, max_attempts_per_sample=10000)
        plot_gmm(X=_s.detach().cpu().numpy(), gmm=gmm, ax=_axs[_k], title=f'Component {_k + 1}', legend_labels=[f'Component {_k + 1}'], ellipse_fill=False, color_by_cluster=False, point_color=_colors[_k])
    _s, _ = gmm.sample(N_DRAW * gmm.n_components, center_point=center_point, center_radius=center_radius, max_attempts_per_sample=10000)
    plot_gmm(X=_s.detach().cpu().numpy(), gmm=gmm, ax=_axs[4], title='Mixture', legend_labels=legend_labels, ellipse_fill=False)
    for _ax in _axs[:5]:
        _ax.add_patch(plt.Circle(_c, center_radius, fill=False, color='red', linewidth=2, linestyle='--', alpha=0.8))
        _ax.plot(_c[0], _c[1], 'r*', markersize=12)
    plot_gmm(X=X, gmm=gmm, ax=_axs[5], title='Original Data (no constraint)', legend_labels=legend_labels, ellipse_fill=False)
    plt.suptitle(f'center_point={_c.tolist()}, center_radius={center_radius}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How Expensive Is Rejection? The Curse of Dimensionality

    **Why does `std_radius=1.0` become expensive in high dimensions?**

    In high dimensions "1 standard deviation" captures very little probability mass. For a standard multivariate Gaussian $\mathbf{X} \sim \mathcal{N}(0, I_d)$ the squared distance from the origin is a sum of $d$ squared standard normals,

    $$
    \|\mathbf{X}\|^2 = X_1^2 + X_2^2 + \cdots + X_d^2 \sim \chi^2(d),
    $$

    and the same holds for any Gaussian once distances are measured by the Mahalanobis distance $\delta_k$: $\delta_k^2 \sim \chi^2(d)$. So the probability of a draw landing within radius $r$ is

    $$
    P(\delta_k \leq r) = P(\chi^2(d) \leq r^2) = F_{\chi^2(d)}(r^2).
    $$

    This is exactly the **acceptance probability** of one rejection-sampling attempt, so a point costs on average $1/F_{\chi^2(d)}(r^2)$ draws, and the chance that all `max_attempts_per_sample` attempts fail is $\bigl(1-F_{\chi^2(d)}(r^2)\bigr)^{\text{max\_attempts}}$. In one dimension $r=1$ keeps 68%, in two dimensions 39%, and in ten dimensions only about 0.02% (about 5800 draws per accepted point, more than the default limit of 1000).

    To choose the radius that contains a target mass $p$ invert the CDF: $r_p = \sqrt{F_{\chi^2(d)}^{-1}(p)}$. This is what `confidence=p` does for you. The cell below computes the table.
    """)
    return


@app.cell
def _(chi2, mo, np):
    _rows = []
    for _d in (1, 2, 3, 5, 10):
        _rows.append(f'| {_d} | {100 * chi2.cdf(1, _d):.2f}% | {100 * chi2.cdf(4, _d):.2f}% | {np.sqrt(chi2.ppf(0.95, _d)):.2f} | {1 / chi2.cdf(1, _d):.0f} |')
    mo.md('| Dimension $d$ | within $1\\sigma$ | within $2\\sigma$ | radius for 95% ($r_{95}$) | mean draws per accepted point at $r=1$ |\n|---|---|---|---|---|\n' + '\n'.join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check by simulation: draw standard normals in $d=2$ and $d=10$, count the fraction that lands inside $r=1$ and inside the 95% radius, and compare with the $\chi^2$ prediction (this is exactly the acceptance rate of one rejection attempt).
    """)
    return


@app.cell
def _(N_BIG, SEED, chi2, np):
    _rng = np.random.default_rng(SEED)
    print(f'{"d":>3}  {"r":>5}  {"predicted":>10}  {"simulated":>10}')
    for _d in (2, 10):
        _dist = np.linalg.norm(_rng.standard_normal((N_BIG, _d)), axis=1)
        for _r in (1.0, float(np.sqrt(chi2.ppf(0.95, _d)))):
            print(f'{_d:>3}  {_r:>5.2f}  {chi2.cdf(_r ** 2, _d):>10.4f}  {np.mean(_dist <= _r):>10.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other Model Families

    Every mixture in tgmm shares the same two-stage `sample(n)` (components drawn from the weights, then each component's own draw), returning `(samples, indices)`. The constraint arguments above exist only on `GaussianMixture`.

    - **Student-t** mixtures draw $x=\mu_k+z/\sqrt{w}$ with $z\sim\mathcal{N}(0,\Sigma_k)$ and $w\sim\mathrm{Gamma}(\nu_k/2,\nu_k/2)$ (the scale-mixture form), so the tails are heavier than a Gaussian's: see [15 -- Student-t Mixtures](15_student_t.ipynb). Note that for these, `covariances_` is the scale matrix.
    - **von Mises-Fisher** mixtures sample directions on the unit sphere: see [16 -- von Mises-Fisher Mixtures](16_von_mises_fisher.ipynb).
    - With a **noise component** (`noise=True`, see [18 -- Noise, Missing Values and Conditioning](18_noise_missing_conditioning.ipynb)) plain sampling also emits noise points, labelled `-1`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - `sample(n)` runs the generative process forwards: $z\sim\mathrm{Categorical}(\pi)$, then $x=\mu_z+L_z g$ with $L_zL_z^\top=\Sigma_z$; it returns the points and the **true** component indices. For every covariance type, including the eight GPCM-only ones, the draw follows the model's stored covariance structure.
    - `component=k` restricts the draw to one component.
    - `std_radius`, `std_range` and `confidence` / `confidence_range` constrain the Mahalanobis distance to the drawing component; `center_point` / `center_radius` constrain the Euclidean distance to a location. Only one constraint can be used at a time.
    - Constraints are enforced by rejection per point. Labels still follow $\pi_k$, and very restrictive regions need a larger `max_attempts_per_sample` (or raise `RuntimeError`).
    - In $d$ dimensions the squared Mahalanobis distance is $\chi^2_d$-distributed, so use a confidence level (or the $\chi^2$-quantile radius) rather than a fixed number of standard deviations; the acceptance probability of one attempt is $F_{\chi^2(d)}(r^2)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it yourself (interactive version)

    The page you are reading shows the default setting of each control below. Open the interactive version (the link at the top of the page: molab, or `marimo edit` on your machine) to change them.

    ### Sample from the fitted mixture

    Choose a component (or the whole mixture), a number of points and a constraint. With these controls at their defaults the plot shows the unconstrained mixture; moving a control re-draws the figure.
    """)
    return


@app.cell
def _(mo):
    # docs: hide-cell
    n_widget = mo.ui.slider(50, 3000, value=800, step=50, label='number of samples')
    component_widget = mo.ui.dropdown(['mixture', '1', '2', '3', '4'], value='mixture', label='component')
    mode_widget = mo.ui.dropdown(['none', 'std_radius', 'confidence'], value='none', label='constraint')
    radius_widget = mo.ui.slider(0.25, 4.0, value=1.0, step=0.25, label='std_radius (σ)')
    confidence_widget = mo.ui.slider(0.1, 0.99, value=0.95, step=0.01, label='confidence')
    mo.vstack([mo.hstack([n_widget, component_widget], justify='start'), mo.hstack([mode_widget, radius_widget, confidence_widget], justify='start')])
    return component_widget, confidence_widget, mode_widget, n_widget, radius_widget


@app.cell
def _(X, component_widget, confidence_widget, gmm, legend_labels, mode_widget, n_widget, plot_gmm, plt, radius_widget):
    _kw = {}
    if component_widget.value != 'mixture':
        _kw['component'] = int(component_widget.value) - 1
    if mode_widget.value == 'std_radius':
        _kw['std_radius'] = radius_widget.value
    elif mode_widget.value == 'confidence':
        _kw['confidence'] = confidence_widget.value
    _s, _i = gmm.sample(n_widget.value, max_attempts_per_sample=5000, **_kw)
    _fig, _ax = plt.subplots(figsize=(6, 5))
    plot_gmm(X=_s.detach().cpu().numpy(), gmm=gmm, ax=_ax, title=f'{n_widget.value} samples, ' + (', '.join(f'{_a}={_b}' for _a, _b in _kw.items()) or 'unconstrained'), true_labels=_i.cpu().numpy(), legend_labels=legend_labels, ellipse_fill=False)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Next:** [06 -- Visualization](06_visualise.ipynb) for the plotting tools, or [10 -- PCA Plotting](10_pca_plotting.ipynb) for visualizing high-dimensional samples.
    """)
    return


if __name__ == "__main__":
    app.run()
