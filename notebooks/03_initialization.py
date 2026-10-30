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
    N_SEEDS = 20 if FAST else 60  # number of independent starts in the sweeps below
    return MAX_ITER, N_INIT, N_SEEDS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialization: Means, Weights, Covariances and `n_init`

    EM climbs the likelihood from wherever it starts and stops at the nearest local maximum, so the starting parameters $(\pi_k, \mu_k, \Sigma_k)$ decide which maximum it finds. This tutorial shows how much that matters on the four-blob dataset from [1. Introduction](01_introduction.ipynb), what each initialization strategy does mathematically, and how `n_init` restarts guard against a bad start.

    **What you'll learn**

    - Why initialization matters: the log-likelihood is non-convex, and EM only ever goes uphill
    - The five mean initializers (`random`, `points`, `kpp`, `kmeans`, `maxdist`) and the math behind them
    - How initial weights and covariances are derived, and how little or how much they change the result
    - How `n_init` works and how many restarts you need (best-of-$n$ selection)
    - Training controls (`max_iter`, `tol`, `reg_covar`, `warm_start`, per-call overrides) and custom tensor initialization with `GMMInitializer`

    **Prerequisites:** [1. Introduction](01_introduction.ipynb).

    **See also:** [2. Covariance Structures](02_covariance_types.ipynb) for the shape constraints placed on $\Sigma_k$, and the [`GMMInitializer` API reference](../api/gmm-initializer.md).
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
    import warnings

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    from tgmm import GaussianMixture, GMMInitializer, plot_gmm
    from tutorial_data import make_default_data

    warnings.filterwarnings('ignore')  # short max_iter budgets below intentionally stop before convergence
    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (
        Ellipse,
        GMMInitializer,
        GaussianMixture,
        SEED,
        device,
        make_default_data,
        np,
        plot_gmm,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data are the default four blobs of [1. Introduction](01_introduction.ipynb). They overlap, which is what makes the starting point matter.
    """)
    return


@app.cell
def _(SEED, device, make_default_data):
    X_tensor, _ = make_default_data(SEED, device=device)
    X = X_tensor.cpu().numpy()
    legend_labels = [f'Component {i + 1}' for i in range(4)]
    return X, X_tensor, legend_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why Initialization Matters

    The mixture log-likelihood

    $$
    \ell(\theta) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k\, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)
    $$

    is non-convex in $\theta = (\pi_k, \mu_k, \Sigma_k)_k$ and has many local maxima: a component can cover two blobs while another is split across a third, for example. Each EM iteration is guaranteed not to decrease $\ell$, so EM converges to a *local* maximum that is determined by the starting point. (There are also degenerate maxima where one component collapses onto a single point and $\ell \to \infty$; `reg_covar` keeps the covariances away from them.)

    The initial parameters enter through the first E-step, which computes the responsibilities

    $$
    r_{ik} = \frac{\pi_k\, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j} \pi_j\, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
    $$

    from them. The initial means matter most (they decide *where* the components sit), the initial covariances and weights mostly decide how soft that first assignment is.

    To see this, the next cell fits `GaussianMixture(n_components=4)` many times with `n_init=1` and a different seed each time, for each of the five `init_means` methods, and records the final mean log-likelihood per sample (`lower_bound_`).
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, N_SEEDS, X_tensor, device, np):
    MEAN_METHODS = ['random', 'points', 'kpp', 'kmeans', 'maxdist']
    sweep = {
        _method: np.array([
            float(GaussianMixture(n_components=4, init_means=_method, random_state=_s, max_iter=MAX_ITER, device=device).fit(X_tensor).lower_bound_)
            for _s in range(N_SEEDS)
        ])
        for _method in MEAN_METHODS
    }
    BEST_LL = max(_v.max() for _v in sweep.values())  # best final log-likelihood found by any run
    TOL_HIT = 1e-3  # a run "reaches the best optimum" if it ends within this of BEST_LL
    return BEST_LL, MEAN_METHODS, TOL_HIT, sweep


@app.cell
def _(BEST_LL, MEAN_METHODS, N_SEEDS, SEED, np, plt, sweep):
    _rng = np.random.default_rng(SEED)
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _i, _method in enumerate(MEAN_METHODS):
        _ax.scatter(_i + _rng.uniform(-0.25, 0.25, N_SEEDS), sweep[_method], s=14, alpha=0.6)
    _ax.axhline(BEST_LL, color='k', ls='--', lw=1, label='best log-likelihood found')
    _ax.set_xticks(range(len(MEAN_METHODS)))
    _ax.set_xticklabels(MEAN_METHODS)
    _ax.set_ylabel('final mean log-likelihood per sample')
    _ax.set_title(f'{N_SEEDS} single runs (n_init=1) per initialization method')
    _ax.legend(loc='lower right')
    plt.gcf()
    return


@app.cell
def _(BEST_LL, MEAN_METHODS, N_SEEDS, TOL_HIT, mo, sweep):
    _rows = [
        f"| `{_m}` | {sweep[_m].max():.4f} | {sweep[_m].min():.4f} | {(sweep[_m] > BEST_LL - TOL_HIT).mean():.0%} |"
        for _m in MEAN_METHODS
    ]
    mo.md(
        f"Over {N_SEEDS} seeds each, the best final log-likelihood is **{BEST_LL:.4f}**. A run counts as reaching it when it ends within {TOL_HIT:g} of that value:\n\n"
        "| `init_means` | best run | worst run | runs reaching the best optimum |\n|---|---|---|---|\n" + "\n".join(_rows)
        + "\n\nRuns that stop clearly below the best value have converged to a different local maximum. "
        "Starting from k-means centres reaches the best optimum most often; the other methods more often end in a clearly worse one."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial Means

    `init_means` accepts a method name or a `(K, d)` tensor. The named methods are static methods of `GMMInitializer`, so you can also call them yourself. With $N$ data points $x_j$ and $K$ components:

    **`'random'`.** Draw each centre from a Gaussian fitted to the data's empirical mean $\bar x$ and covariance $\hat\Sigma = L L^\top$ (Cholesky):
    $$
    \mu_i = \bar x + L z, \qquad z \sim \mathcal{N}(0, I_d).
    $$
    Centres follow the global data distribution but are not data points.

    **`'points'`.** Pick $K$ distinct data points uniformly at random: $\mu_i = x_{s_i}$ with $S = \{s_1,\dots,s_K\}$ a random subset of $\{1,\dots,N\}$.

    **`'kpp'` (k-means++).** The first centre is a uniformly random data point. Each next centre is a data point $x_j$ drawn with probability proportional to its squared distance to the nearest centre already chosen ($D^2$-sampling):
    $$
    P(x_j) = \frac{D(x_j)}{\sum_{j'=1}^{N} D(x_{j'})}, \qquad D(x_j) = \min_{l < i} \lVert x_j - \mu_l \rVert^2 .
    $$
    Far-away points are likely to be chosen, so the centres spread out. Arthur & Vassilvitskii (2007) show the resulting k-means objective is within $O(\log K)$ of the optimum in expectation.

    **`'kmeans'` (the default).** Start from k-means++ and run Lloyd's algorithm: alternate
    $$
    c_j = \arg\min_i \lVert x_j - \mu_i \rVert^2 \quad\text{(assign)}, \qquad \mu_i = \frac{1}{|C_i|}\sum_{x_j \in C_i} x_j \quad\text{(update)}
    $$
    until the centres stop moving. Each step does not increase $\sum_j \lVert x_j - \mu_{c_j}\rVert^2$, so this converges to a local minimum of the k-means objective, which is usually a good place for EM to start.

    **`'maxdist'`.** Farthest-first traversal. After a random first pick, repeatedly choose
    $$
    \mu_i = \arg\max_{x} \min_{l < i} \lVert x - \mu_l \rVert,
    $$
    then re-select the first centre as $\mu_1 = \arg\max_x \min_{l \ge 2} \lVert x - \mu_l \rVert$ so the arbitrary first pick does not bias the result. This spreads centres maximally but is sensitive to outliers, which are by definition the farthest points.

    The next figure shows where each method puts its four initial centres (red stars) for one seed. `random`, `points` and `maxdist` use the global torch seed; `kpp` and `kmeans` take a `random_state` argument.
    """)
    return


@app.cell
def _(GMMInitializer, MEAN_METHODS, SEED, X, X_tensor, plt, torch):
    _fig, _axs = plt.subplots(1, len(MEAN_METHODS), figsize=(15, 3.4), sharex=True, sharey=True)
    for _ax, _method in zip(_axs, MEAN_METHODS):
        torch.manual_seed(SEED)
        if _method in ('kpp', 'kmeans'):
            _means = getattr(GMMInitializer, _method)(X_tensor, 4, random_state=SEED)
        else:
            _means = getattr(GMMInitializer, _method)(X_tensor, 4)
        _m = _means.cpu().numpy()
        _ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.2, color='gray')
        _ax.scatter(_m[:, 0], _m[:, 1], marker='*', s=220, color='red', edgecolor='k', zorder=3)
        _ax.set_title(f"init_means='{_method}'")
        _ax.set_aspect('equal')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it: pick a method and a seed

    Choose an initializer and a seed. The cell below calls the matching `GMMInitializer` method, passes the resulting tensor as `init_means`, runs EM and plots the result (red markers are the initial means). The title reports the final log-likelihood and whether this run reached the best optimum found in the sweep above. Different seeds with `random` or `points` regularly end in different optima; try a few. (This control is live in `marimo edit` or on molab; the static docs page shows the default.)
    """)
    return


@app.cell
def _(MEAN_METHODS, mo):
    init_method = mo.ui.dropdown(MEAN_METHODS, value='random', label='init_means')
    init_seed = mo.ui.slider(0, 29, value=0, label='seed')
    mo.hstack([init_method, init_seed], justify='start')
    return init_method, init_seed


@app.cell
def _(
    BEST_LL,
    GMMInitializer,
    GaussianMixture,
    MAX_ITER,
    TOL_HIT,
    X,
    X_tensor,
    device,
    init_method,
    init_seed,
    legend_labels,
    plot_gmm,
    plt,
    torch,
):
    _method, _seed = init_method.value, int(init_seed.value)
    torch.manual_seed(_seed)
    if _method in ('kpp', 'kmeans'):
        _init = getattr(GMMInitializer, _method)(X_tensor, 4, random_state=_seed)
    else:
        _init = getattr(GMMInitializer, _method)(X_tensor, 4)
    _gmm = GaussianMixture(n_components=4, init_means=_init, random_state=_seed, max_iter=MAX_ITER, device=device).fit(X_tensor)
    _ll = float(_gmm.lower_bound_)
    _verdict = 'reaches the best optimum' if _ll > BEST_LL - TOL_HIT else 'stuck in a worse local optimum'
    _fig, _ax = plt.subplots(figsize=(6.5, 5))
    plot_gmm(X, gmm=_gmm, ax=_ax, show_initial_means=True, legend_labels=legend_labels,
             title=f"init_means='{_method}', seed {_seed}: log-lik {_ll:.4f}\n{_verdict} ({_gmm.n_iter_} iterations)")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial Weights

    `init_weights` sets the starting mixing proportions $\pi_k$:

    | `init_weights` | Initial $\pi_k$ |
    | --- | --- |
    | `'uniform'` / `'equal'` (default) | $1/K$ for every component |
    | `'random'` | one draw from $\mathrm{Dirichlet}(1,\dots,1)$, i.e. uniform over the simplex |
    | `'kmeans'` | $n_k / n$, where $n_k$ counts the points whose nearest initial mean is $\mu_k$ |
    | tensor | the given values, renormalized to sum to 1 |

    The weights only enter the first E-step as the prior factor in $r_{ik} \propto \pi_k\,\mathcal{N}(x_i \mid \mu_k,\Sigma_k)$. Once the means are sensible, they are the least influential of the three initializations: the M-step recomputes $\pi_k = \frac{1}{n}\sum_i r_{ik}$ immediately. The next cell fixes the initial means (k-means, seed 0) and varies only the weights; `initial_weights_` records what each run started from.
    """)
    return


@app.cell
def _(GMMInitializer, GaussianMixture, MAX_ITER, SEED, X_tensor, device, mo, torch):
    _means = GMMInitializer.kmeans(X_tensor, 4, random_state=SEED)
    _rows = []
    for _method in ['uniform', 'random', 'kmeans']:
        torch.manual_seed(SEED)
        _gmm = GaussianMixture(n_components=4, init_means=_means, init_weights=_method, random_state=SEED, max_iter=MAX_ITER, device=device).fit(X_tensor)
        _w0 = ', '.join(f'{_w:.2f}' for _w in _gmm.initial_weights_.cpu().tolist())
        _w1 = ', '.join(f'{_w:.2f}' for _w in _gmm.weights_.cpu().tolist())
        _rows.append(f"| `{_method}` | [{_w0}] | [{_w1}] | {float(_gmm.lower_bound_):.4f} | {_gmm.n_iter_} |")
    mo.md(
        "| `init_weights` | initial $\\pi$ | fitted $\\pi$ | log-likelihood | EM iterations |\n|---|---|---|---|---|\n" + "\n".join(_rows)
        + "\n\nThe fitted weights and log-likelihood hardly depend on the starting weights; only the number of EM iterations changes, and only slightly."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial Covariances

    `init_covariances` sets the starting $\Sigma_k$ (and the starting scale of the soft assignments). Written for `covariance_type='full'`:

    | `init_covariances` | Initial $\Sigma_k$ |
    | --- | --- |
    | `'empirical'` (default) | assign every point to its nearest initial mean to get clusters $C_k$; $\Sigma_k = \widehat{\mathrm{Cov}}(C_k) + \texttt{reg\_covar}\, I$ (unbiased, $n_k - 1$); an empty cluster gets $(1+\texttt{reg\_covar})\,I$ |
    | `'global'` | the sample covariance of the whole dataset $+\ \texttt{reg\_covar}\,I$, the same for every component |
    | `'eye'` | $(1+\texttt{reg\_covar})\,I$ |
    | `'random'` | $A A^\top + \texttt{reg\_covar}\, I$ with $A_{ab} \sim \mathcal{N}(0,1)$ |
    | tensor | the given values (shape per `covariance_type`) |

    For the diagonal, spherical and tied covariance types the corresponding diagonal, mean-of-diagonal or pooled form is used (see [2. Covariance Structures](02_covariance_types.ipynb)). For the `tied_*` types `'empirical'` pools the within-cluster scatter, $\Sigma = \frac1n \sum_k \sum_{x\in C_k}(x - \bar x_{C_k})(x - \bar x_{C_k})^\top + \texttt{reg\_covar}\, I$.

    The figure draws the 2σ ellipses each method starts from (around the same k-means initial means), then the table compares the resulting fits.
    """)
    return


@app.cell
def _(GMMInitializer, GaussianMixture, MAX_ITER, SEED, X_tensor, device):
    COV_METHODS = ['empirical', 'global', 'eye', 'random']
    _means = GMMInitializer.kmeans(X_tensor, 4, random_state=SEED)
    cov_runs = {
        _method: GaussianMixture(n_components=4, init_means=_means, init_covariances=_method, random_state=SEED, max_iter=MAX_ITER, device=device).fit(X_tensor)
        for _method in COV_METHODS
    }
    return COV_METHODS, cov_runs


@app.cell
def _(COV_METHODS, Ellipse, X, cov_runs, np, plt):
    def cov_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
        w, V = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        ax.add_patch(Ellipse(mean, 2 * n_std * np.sqrt(w[1]), 2 * n_std * np.sqrt(w[0]), angle=angle, fill=False, **kwargs))

    _fig, _axs = plt.subplots(1, len(COV_METHODS), figsize=(15, 3.6), sharex=True, sharey=True)
    for _ax, _method in zip(_axs, COV_METHODS):
        _g = cov_runs[_method]
        _mu, _S = _g.initial_means_.cpu().numpy(), _g.initial_covariances_.cpu().numpy()
        _ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.2, color='gray')
        for _k in range(4):
            cov_ellipse(_ax, _mu[_k], _S[_k], color=f'C{_k}', lw=2)
        _ax.scatter(_mu[:, 0], _mu[:, 1], marker='*', s=120, color='red', edgecolor='k', zorder=3)
        _ax.set_title(f"init_covariances='{_method}'")
        _ax.set_aspect('equal')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(COV_METHODS, cov_runs, mo):
    _rows = [f"| `{_m}` | {float(cov_runs[_m].lower_bound_):.4f} | {cov_runs[_m].n_iter_} | {cov_runs[_m].converged_} |" for _m in COV_METHODS]
    mo.md(
        "| `init_covariances` | final log-likelihood | EM iterations | converged |\n|---|---|---|---|\n" + "\n".join(_rows)
        + "\n\nWith good initial means all four reach practically the same log-likelihood. A `random` start (matrices unrelated to the data's scale) can need many more iterations."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Restarts: `n_init`

    Suppose a single run from a random start reaches the best optimum with probability $p$ (independent across seeds). Running $n$ independent starts and keeping the one with the highest final log-likelihood finds the best optimum with probability

    $$
    P(\text{best of } n \text{ is optimal}) = 1 - (1 - p)^n .
    $$

    `n_init=n` does exactly that: run $i$ ($i = 0,\dots,n-1$) uses seed `random_state + i`, and the run with the highest `lower_bound_` is kept; its seed is stored in `best_random_state_`. The cost grows linearly in $n$, while the failure probability shrinks geometrically.

    The next cell estimates $p$ from the single `init_means='random'` runs above and compares the formula with what actually happens when we take the best of $n$ consecutive seeds.
    """)
    return


@app.cell
def _(BEST_LL, N_SEEDS, TOL_HIT, np, plt, sweep):
    _ll = sweep['random']
    _p = float((_ll > BEST_LL - TOL_HIT).mean())
    _ns = list(range(1, 9))
    _emp = [np.mean([_ll[_b:_b + _n].max() > BEST_LL - TOL_HIT for _b in range(N_SEEDS - _n + 1)]) for _n in _ns]
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.plot(_ns, [1 - (1 - _p) ** _n for _n in _ns], 'o--', label=f'formula $1-(1-p)^n$, $p$ = {_p:.2f}')
    _ax.plot(_ns, _emp, 's-', label=f'observed (best of $n$ consecutive seeds)')
    _ax.set_xlabel('n_init')
    _ax.set_ylabel('probability of reaching the best optimum')
    _ax.set_ylim(0, 1.05)
    _ax.set_title("init_means='random'")
    _ax.legend(loc='lower right')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it: choose `n_init`

    The slider sets `n_init` for a `GaussianMixture(init_means='random', random_state=0)`. The line below it checks the definition against the single-run results from the sweep: the winner is the best of the single runs with seeds $0,\dots,n-1$.
    """)
    return


@app.cell
def _(mo):
    n_init_slider = mo.ui.slider(1, 10, value=3, label='n_init')
    n_init_slider
    return (n_init_slider,)


@app.cell
def _(GaussianMixture, MAX_ITER, X_tensor, device, mo, n_init_slider, np, sweep):
    _n = int(n_init_slider.value)
    _gmm = GaussianMixture(n_components=4, init_means='random', n_init=_n, random_state=0, max_iter=MAX_ITER, device=device).fit(X_tensor)
    _singles = sweep['random'][:_n]
    mo.md(
        f"`n_init={_n}`: best log-likelihood **{float(_gmm.lower_bound_):.4f}**, winning seed `best_random_state_` = **{_gmm.best_random_state_}**.\n\n"
        f"Single runs with seeds 0..{_n - 1}: " + ', '.join(f'{_v:.4f}' for _v in _singles)
        + f". Best single run is seed {int(np.argmax(_singles))}, matching the winner: **{int(np.argmax(_singles)) == _gmm.best_random_state_}**."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Controls

    Beyond the initialization, a few parameters control the EM optimization itself:

    - `max_iter`, `tol`: stop after this many iterations, or once the relative log-likelihood improvement drops below `tol`.
    - `reg_covar`: regularization added to the covariance diagonal, needed to avoid singular or degenerate covariances.
    - `n_init`: number of restarts, as above.
    - `warm_start`: reuse the previous fit's parameters as the starting point for the next call to `fit`, instead of re-initializing.
    - `random_state`: seed for reproducibility. With `n_init > 1`, run `i` uses `random_state + i`.

    `max_iter`, `tol`, `random_state` and `warm_start` can also be overridden on a single call to `fit(...)` without changing the model's own defaults.
    """)
    return


@app.cell
def _(GaussianMixture, N_INIT, X_tensor):
    # n_init keeps the best of several random restarts, and records which seed won
    gmm_multi = GaussianMixture(n_components=4, n_init=N_INIT, random_state=0, max_iter=200)
    gmm_multi.fit(X_tensor)
    print(f'Best log-likelihood: {gmm_multi.lower_bound_:.4f}')
    print(f'Winning seed: {gmm_multi.best_random_state_}')

    # warm_start: continue optimizing an already-fitted model instead of restarting from scratch
    gmm_warm = GaussianMixture(n_components=4, warm_start=True, max_iter=5, random_state=0)
    gmm_warm.fit(X_tensor)
    print(f'\nAfter 5 iterations: log-likelihood={gmm_warm.lower_bound_:.4f}, converged={gmm_warm.converged_}')
    gmm_warm.fit(X_tensor)  # picks up where it left off instead of re-initializing
    print(f'After 5 more (warm-started) iterations: log-likelihood={gmm_warm.lower_bound_:.4f}, converged={gmm_warm.converged_}')

    # fit() accepts overrides for max_iter/tol/random_state/warm_start without touching the model's own defaults
    gmm_override = GaussianMixture(n_components=4, max_iter=1000, random_state=0)
    gmm_override.fit(X_tensor, max_iter=10, tol=0.01)
    print(f'\nOverridden fit (max_iter=10, tol=1e-2): n_iter_={gmm_override.n_iter_}, converged={gmm_override.converged_}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom (Tensor) Initialization

    `init_means`, `init_weights` and `init_covariances` each also accept an explicit `torch.Tensor`, so you can seed the model with domain knowledge, a previous run's fitted parameters, or the output of `GMMInitializer` itself. The static methods are:

    ```python
    means = GMMInitializer.kmeans(X, k=4)                                   # or .kpp, .random, .points, .maxdist
    weights = GMMInitializer.init_weights_from_clusters(X, means)            # n_k / n by nearest mean
    covariances = GMMInitializer.init_covariances_empirical(X, means, covariance_type='full', reg_covar=1e-6)
    ```

    The cell below compares four starts for the same data: the default, the tensors built from the three calls above (which should reproduce the default), a start from rough prior knowledge of the blob locations, and a deliberately poor start with all four means on a horizontal line far above the data.
    """)
    return


@app.cell
def _(GMMInitializer, GaussianMixture, MAX_ITER, SEED, X_tensor, device, mo, torch):
    _means = GMMInitializer.kmeans(X_tensor, 4, random_state=SEED)
    _weights = GMMInitializer.init_weights_from_clusters(X_tensor, _means)
    _covs = GMMInitializer.init_covariances_empirical(X_tensor, _means, covariance_type='full', reg_covar=1e-6)
    _prior_means = torch.tensor([[0.0, 2.0], [2.0, -2.0], [0.0, 0.0], [2.0, 2.0]], device=device) + 0.5  # rough guess: true centres, shifted by 0.5
    _bad_means = torch.tensor([[-1.0, 5.0], [0.0, 5.0], [1.0, 5.0], [2.0, 5.0]], device=device)
    _starts = {
        'default (`kmeans`, uniform, empirical)': dict(random_state=SEED),
        'tensors from `GMMInitializer`': dict(init_means=_means, init_weights=_weights, init_covariances=_covs),
        'prior-knowledge means (identity covariances)': dict(init_means=_prior_means, init_covariances=torch.eye(2, device=device).repeat(4, 1, 1)),
        'poor means on a line above the data': dict(init_means=_bad_means, init_weights=torch.tensor([0.4, 0.2, 0.2, 0.2], device=device), init_covariances=torch.eye(2, device=device).repeat(4, 1, 1)),
    }
    _rows = []
    for _name, _kw in _starts.items():
        _gmm = GaussianMixture(n_components=4, covariance_type='full', max_iter=MAX_ITER, device=device, **{'random_state': SEED, **_kw}).fit(X_tensor)
        _rows.append(f"| {_name} | {float(_gmm.lower_bound_):.4f} | {_gmm.n_iter_} | {_gmm.converged_} |")
    mo.md("| start | final log-likelihood | EM iterations | converged |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other Mixture Families

    `StudentTMixture` uses the same `init_means` / `init_weights` / `init_covariances` options (the covariance starts as the scale matrix, and `dof_init` sets the initial degrees of freedom), see [16. Student-t Mixtures](16_student_t.ipynb). `VonMisesFisherMixture` initializes mean directions by spherical k-means and sets every $\kappa_k$ to `kappa_init`, see [17. Von Mises-Fisher Mixtures](17_von_mises_fisher.ipynb).

    ## Summary

    - EM only goes uphill, so it ends in the local maximum nearest its start; initialization decides which one.
    - `init_means` matters most. `kmeans` (the default) is the most reliable; `random`, `points`, `kpp` and `maxdist` land in worse optima more often.
    - `init_weights` and `init_covariances` mostly affect speed once the means are good; badly scaled covariances can cost many extra iterations.
    - `n_init=n` keeps the best of $n$ runs (seeds `random_state + i`), failing with probability about $(1-p)^n$; the winner's seed is in `best_random_state_`.
    - `warm_start` and per-call `fit(...)` overrides control optimization; all three initializers also accept tensors, and `GMMInitializer` builds them directly.

    **Next:** [4. Sampling](04_sampling.ipynb).

    **References:** Arthur & Vassilvitskii (2007), *k-means++: the advantages of careful seeding*, SODA. Lloyd (1982), *Least squares quantization in PCM*, IEEE Trans. Information Theory 28(2).
    """)
    return


if __name__ == "__main__":
    app.run()
