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
    N_SEEDS = 20 if FAST else 60  # number of independent starts in the sweeps below
    return MAX_ITER, N_SEEDS


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
    - Training controls (`max_iter`, `tol`, `reg_covar`) and how to start EM from parameters you give it: `init_*` tensors, `GMMInitializer`, and `warm_start`

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

    from tgmm import GaussianMixture, GMMInitializer, generate_gmm_data
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
        generate_gmm_data,
        make_default_data,
        np,
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
    return X, X_tensor


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

    To see this, the next cell fits `GaussianMixture(n_components=4)` many times with `n_init=1` and a different seed each time, for each of the five `init_means` methods, and records the final mean log-likelihood per sample (`lower_bound_`). The share of runs that reach the best optimum is the last column of the table in *Initial Means* below.
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
    then re-select the first centre as $\mu_1 = \arg\max_x \min_{l \ge 2} \lVert x - \mu_l \rVert$ so the arbitrary first pick does not bias the result. This is Gonzalez's (1985) farthest-first traversal, the classical 2-approximation for the *k-center* objective (minimise the largest distance to a centre), not for the k-means objective or the likelihood. It spreads centres maximally, but outliers are by definition the farthest points, so it is sensitive to them; its only randomness is the first pick (torch's global RNG).

    The next figure shows where each method puts its four initial centres (red stars) for one seed. `random`, `points` and `maxdist` use the global torch seed; `kpp` and `kmeans` take a `random_state` argument. In the interactive version the **seed** slider changes the seed for all five methods at once: `random`, `points` and `maxdist` move around a lot from seed to seed, while `kmeans` varies much less (it ends on one of only a few different solutions). (The static docs page shows seed 0.)
    """)
    return


@app.cell
def _(SEED, mo):
    # docs: hide-cell
    init_seed = mo.ui.slider(0, 29, step=1, value=SEED, label='seed', show_value=True)
    init_seed
    return (init_seed,)


@app.cell
def _(GMMInitializer, MEAN_METHODS, X, X_tensor, init_seed, plt, torch):
    _seed = int(init_seed.value)
    _fig, _axs = plt.subplots(3, 2, figsize=(8, 11), sharex=True, sharey=True, dpi=80)
    _flat = _axs.ravel()
    for _ax, _method in zip(_flat, MEAN_METHODS):
        torch.manual_seed(_seed)
        if _method in ('kpp', 'kmeans'):
            _means = getattr(GMMInitializer, _method)(X_tensor, 4, random_state=_seed)
        else:
            _means = getattr(GMMInitializer, _method)(X_tensor, 4)
        _m = _means.cpu().numpy()
        _ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.2, color='gray')
        _ax.scatter(_m[:, 0], _m[:, 1], marker='*', s=220, color='red', edgecolor='k', zorder=3)
        _ax.set_title(f"init_means='{_method}'")
        _ax.set_aspect('equal')
    _key = _flat[-1]  # the sixth cell holds the legend
    _key.axis('off')
    _key.scatter([], [], s=12, color='gray', label='data')
    _key.scatter([], [], marker='*', s=220, color='red', edgecolor='k', label='initial centre')
    _key.legend(loc='center', frameon=False)
    _key.set_title(f'seed = {_seed}')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(BEST_LL, MEAN_METHODS, N_SEEDS, TOL_HIT, mo, sweep):
    _hit = {_m: f"{(sweep[_m] > BEST_LL - TOL_HIT).mean():.0%}" for _m in MEAN_METHODS}
    mo.md(f"""
    ### Which `init_means` when?

    | `init_means` | What it does | Pros | Cons | Good for / bad for | Runs reaching the best optimum ({N_SEEDS} single runs, default blobs) |
    |---|---|---|---|---|---|
    | `random` | draws centres from $\\mathcal N(\\bar x,\\hat\\Sigma)$ (not data points) | cheapest; centres follow the global spread | centres can fall in empty space or pile into one cluster; ignores local structure | **good:** a quick baseline together with `n_init` restarts. **bad:** well separated or many clusters | {_hit['random']} |
    | `points` | $K$ distinct data points, uniformly at random | cheap; centres sit on real data, more often in dense regions | two points can come from one cluster and none from a small one; ignores spread | **good:** clusters of similar size. **bad:** small clusters (rarely hit) | {_hit['points']} |
    | `kpp` | first point uniform, then each next point drawn with probability $\\propto D^2$ to the nearest centre | spreads centres yet is randomized, so a lone outlier is rarely picked; $O(\\log K)$-competitive for the k-means objective | still seed dependent; no refinement step | **good:** a general-purpose start, mild outliers, many clusters. **bad:** nothing in particular; use `n_init` for unlucky draws | {_hit['kpp']} |
    | `kmeans` (default) | k-means++ followed by Lloyd iterations until the centres stop moving | centres are already cluster means; reaches the best optimum most often here | costs a k-means run; favours round, similar-sized clusters and can cut an elongated cluster in two (EM usually repairs this) | **good:** compact, roughly round clusters, the safe default. **bad:** very elongated or very unequal clusters | {_hit['kmeans']} |
    | `maxdist` | farthest-first traversal: random first point, then always the point farthest from all centres so far | no iterations; deterministic apart from the first point; finds one point per cluster when clusters are further apart than their diameter | picks extreme points: outliers, heavy tails or elongated clusters steal centres; ignores density; can leave a true cluster without a centre | **good:** compact, well separated clusters *without* outliers. **bad:** noisy data, outliers, overlapping or elongated clusters | {_hit['maxdist']} |
    | tensor `(K, d)` | your own centres | uses prior knowledge or a previous fit; the same start every run | a wrong guess is a bad start, and `n_init` only repeats it | **good:** known cluster positions (see *Starting from Given Parameters*) | n/a |
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, N_SEEDS, device, generate_gmm_data, mo, np, torch):
    # a small experiment on the claim in the table: four well separated, compact clusters, with and without 1% uniform outliers
    _centres = np.array([(0, 0), (8, 0), (0, 8), (8, 8)], dtype=float)
    _X_clean, _ = generate_gmm_data([_c for _c in _centres], [np.eye(2)] * 4, [300] * 4, device=device, random_state=1)
    _out = torch.tensor(np.random.default_rng(0).uniform(-12, 20, (12, 2)), dtype=_X_clean.dtype, device=_X_clean.device)
    _X_noisy = torch.cat([_X_clean, _out])

    def _recovers(_X, _method, _seed):
        """True when every true centre has a fitted mean within 1.5 (a crashed fit counts as a failure)."""
        torch.manual_seed(_seed)
        try:
            _g = GaussianMixture(n_components=4, init_means=_method, random_state=_seed, max_iter=MAX_ITER, device=device).fit(_X)
        except ValueError:
            return False
        _d = np.linalg.norm(_centres[:, None] - _g.means_.detach().cpu().numpy()[None], axis=2)
        return bool((_d.min(axis=1) < 1.5).all())

    _rows = []
    for _method in ['maxdist', 'kmeans', 'kpp']:
        _clean = sum(_recovers(_X_clean, _method, _s) for _s in range(N_SEEDS))
        _noisy = sum(_recovers(_X_noisy, _method, _s) for _s in range(N_SEEDS))
        _rows.append(f"| `{_method}` | {_clean}/{N_SEEDS} | {_noisy}/{N_SEEDS} |")
    mo.md(
        "On four well separated, compact clusters, the number of single runs (of "
        f"{N_SEEDS}) in which every true cluster ends up with a fitted component:\n\n"
        "| `init_means` | clean data | plus 12 uniform outliers (1%) |\n|---|---|---|\n" + "\n".join(_rows)
        + "\n\n`maxdist` is excellent on the clean data and collapses as soon as a few outliers appear, because the outliers are the farthest points and take the centres. "
        "`kpp` samples with probability $\\propto D^2$ instead of taking the maximum, so a lone outlier is rarely chosen."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Which one to use.** `kmeans` (the default) is the robust choice: it is cheap compared with EM and puts the centres where the clusters are. `kpp` is the randomized, outlier-tolerant version of farthest-first and a good lighter-weight alternative. Use `maxdist` only for clean, well separated data. The random methods (`random`, `points`, `kpp`) should be combined with `n_init` (below). Variants such as *greedy* k-means++ (several candidates per step, keep the best) and robust farthest-first traversals (trim outliers first, or start from the densest point) reduce these weaknesses further; tgmm does not implement them, but you can build such centres yourself and pass them as a tensor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial Weights

    `init_weights` sets the starting mixing proportions $\pi_k$:

    | `init_weights` | Initial $\pi_k$ | Pros | Cons | Good for / bad for |
    | --- | --- | --- | --- | --- |
    | `'uniform'` / `'equal'` (default) | $1/K$ for every component | never favours a component; always valid; nothing to tune | far from the truth when cluster sizes differ a lot (small clusters start over-weighted) | **good:** roughly balanced clusters. **bad:** very unequal sizes |
    | `'random'` | one draw from $\mathrm{Dirichlet}(1,\dots,1)$, i.e. uniform over the simplex | different seeds start differently, which diversifies restarts | can start with a near-zero weight for a real cluster and adds noise | **good:** restarts with `n_init` when you want diversity. **bad:** a single run |
    | `'kmeans'` | $n_k / n$, where $n_k$ counts the points whose nearest initial mean is $\mu_k$ | starts at the sizes the data suggest | only as good as the means; a component with no nearest points starts at weight $\approx 10^{-10}$ | **good:** unequal cluster sizes with sensible means. **bad:** poor or random means |
    | tensor | the given values, renormalized to sum to 1 | uses known proportions | a wrong guess is a bad start | **good:** known class proportions |

    The weights only enter the first E-step as the prior factor in $r_{ik} \propto \pi_k\,\mathcal{N}(x_i \mid \mu_k,\Sigma_k)$. Once the means are sensible, they are the least influential of the three initializations: the M-step recomputes $\pi_k = \frac{1}{n}\sum_i r_{ik}$ immediately. The cell below fixes the initial means (k-means, seed 0) and varies only the weights; `initial_weights_` records what each run started from.
    """)
    return


@app.cell
def _(GMMInitializer, GaussianMixture, MAX_ITER, SEED, X_tensor, device, mo, torch):
    _means = GMMInitializer.kmeans(X_tensor, 4, random_state=SEED)
    _res = {}
    for _method in ['uniform', 'random', 'kmeans']:
        torch.manual_seed(SEED)
        _gmm = GaussianMixture(n_components=4, init_means=_means, init_weights=_method, random_state=SEED, max_iter=MAX_ITER, device=device).fit(X_tensor)
        _res[_method] = (float(_gmm.lower_bound_), _gmm.n_iter_)
    mo.md(
        "From the same k-means means, the three weight starts end at log-likelihood "
        + ", ".join(f"{_ll:.4f} (`{_m}`, {_it} EM iterations)" for _m, (_ll, _it) in _res.items())
        + ": the starting weights barely matter."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial Covariances

    `init_covariances` sets the starting $\Sigma_k$ (and the starting scale of the soft assignments). Written for `covariance_type='full'`:

    | `init_covariances` | Initial $\Sigma_k$ | Pros | Cons | Good for / bad for |
    | --- | --- | --- | --- | --- |
    | `'empirical'` (default) | assign every point to its nearest initial mean to get clusters $C_k$; $\Sigma_k = \widehat{\mathrm{Cov}}(C_k) + \texttt{reg\_covar}\, I$ (unbiased, $n_k - 1$); an empty cluster gets $(1+\texttt{reg\_covar})\,I$ | adapts each component to its cluster's size and shape; the first E-step is already informative | needs sensible means; unreliable when a cluster has few points (singular when $n_k \le d$) | **good:** many points per cluster ($n_k \gg d$), clusters of different shapes. **bad:** high dimension with small clusters |
    | `'global'` | the sample covariance of the whole dataset $+\ \texttt{reg\_covar}\,I$, the same for every component | always well conditioned; works with few points per cluster | too wide for tight clusters, so the first responsibilities are very soft; ignores shapes | **good:** high dimension or small clusters. **bad:** clusters much tighter than the whole data |
    | `'eye'` | $(1+\texttt{reg\_covar})\,I$ | no data needed; trivially valid | wrong scale unless the data are standardized; ignores shape | **good:** standardized data, a neutral start. **bad:** features on very different scales |
    | `'random'` | $A A^\top + \texttt{reg\_covar}\, I$ with $A_{ab} \sim \mathcal{N}(0,1)$ | different seeds start differently, which diversifies restarts | unrelated to the data's scale and shape, so it can start far from any good solution | **good:** restarts with many `n_init`. **bad:** a single run |
    | tensor | the given values (shape per `covariance_type`) | uses prior knowledge or a previous fit | a wrong guess is a bad start | **good:** known cluster shapes |

    For the diagonal, spherical and tied covariance types the corresponding diagonal, mean-of-diagonal or pooled form is used (see [2. Covariance Structures](02_covariance_types.ipynb)). For the `tied_*` types `'empirical'` pools the within-cluster scatter, $\Sigma = \frac1n \sum_k \sum_{x\in C_k}(x - \bar x_{C_k})(x - \bar x_{C_k})^\top + \texttt{reg\_covar}\, I$.

    The figure draws the 2σ ellipses each method starts from (around the same k-means initial means); the sentence below it compares the resulting fits.
    """)
    return


@app.cell
def _(GMMInitializer, GaussianMixture, MAX_ITER, N_SEEDS, SEED, X_tensor, device, np):
    COV_METHODS = ['empirical', 'global', 'eye', 'random']
    _means = GMMInitializer.kmeans(X_tensor, 4, random_state=SEED)
    cov_runs = {
        _method: GaussianMixture(n_components=4, init_means=_means, init_covariances=_method, random_state=SEED, max_iter=MAX_ITER, device=device).fit(X_tensor)
        for _method in COV_METHODS
    }
    # the random covariance start depends on the seed: repeat it, with the same (k-means) initial means
    _rand = [GaussianMixture(n_components=4, init_means=_means, init_covariances='random', random_state=_s, max_iter=MAX_ITER, device=device).fit(X_tensor) for _s in range(N_SEEDS)]
    cov_random_ll = np.array([float(_g.lower_bound_) for _g in _rand])
    cov_random_iters = np.array([_g.n_iter_ for _g in _rand])
    return COV_METHODS, cov_random_iters, cov_random_ll, cov_runs


@app.cell
def _(COV_METHODS, Ellipse, X, cov_runs, np, plt):
    def cov_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
        w, V = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        ax.add_patch(Ellipse(mean, 2 * n_std * np.sqrt(w[1]), 2 * n_std * np.sqrt(w[0]), angle=angle, fill=False, **kwargs))

    _fig, _axs = plt.subplots(2, 2, figsize=(8, 7.2), sharex=True, sharey=True, dpi=80)
    for _ax, _method in zip(_axs.ravel(), COV_METHODS):
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
def _(BEST_LL, COV_METHODS, N_SEEDS, TOL_HIT, cov_random_iters, cov_random_ll, cov_runs, mo):
    _bad = float((cov_random_ll < BEST_LL - TOL_HIT).mean())
    _data_driven = [_m for _m in COV_METHODS if _m != 'random']
    mo.md(
        "From the same k-means means, the data-driven starts end at log-likelihood "
        + ", ".join(f"{float(cov_runs[_m].lower_bound_):.4f} (`{_m}`, {cov_runs[_m].n_iter_} EM iterations)" for _m in _data_driven)
        + ". A `random` start (matrices unrelated to the data's scale) is the risky one: "
        f"over {N_SEEDS} seeds it ended in a clearly worse local maximum in {_bad:.0%} of the runs (worst log-likelihood {cov_random_ll.min():.4f}), and the number of EM iterations ranged from {cov_random_iters.min()} to {cov_random_iters.max()}."
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
    ## Training Controls

    Beyond the initialization, a few parameters control the EM optimization itself:

    - `max_iter`, `tol`: stop after this many iterations, or once the relative log-likelihood improvement drops below `tol`.
    - `reg_covar`: regularization added to the covariance diagonal, needed to avoid singular or degenerate covariances.
    - `n_init`: number of restarts, as above.
    - `random_state`: seed for reproducibility. With `n_init > 1`, run `i` uses `random_state + i`.

    `max_iter`, `tol`, `random_state` and `warm_start` can also be overridden on a single call to `fit(...)`, without changing the model's own defaults. The next section uses that.

    ## Starting from Given Parameters

    There are two ways to hand EM a starting point that you chose yourself:

    - **Tensors.** `init_means`, `init_weights` and `init_covariances` each also accept an explicit `torch.Tensor` (or only some of them: the rest is derived as usual). This is a *fresh* fit that starts there.
    - **`warm_start=True`.** `fit` does not re-initialize; it continues EM from the model's current `means_`, `weights_` and `covariances_`.

    Are they the same thing? For the optimization, yes: continuing from a fitted model is exactly a fresh fit whose `init_*` tensors are that model's `means_`, `weights_` and `covariances_`. The cell below checks it. A first fit is cut short by a per-call `max_iter=5` override, then continued with `fit(..., warm_start=True)` under the same `max_iter=5` budget. A second model is started fresh from the tensors the first one held after its short run, with the same budget. The last two rows continue to convergence instead, and the last one gives the tensor fit a different `random_state`.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, X_tensor, device, mo):
    _fresh = dict(n_components=4, max_iter=MAX_ITER, device=device)
    # stage 1: a short run. max_iter=5 is overridden for this call only; the model's own max_iter stays at MAX_ITER
    gmm_stage1 = GaussianMixture(random_state=0, **_fresh).fit(X_tensor, max_iter=5)
    _m, _w, _c = gmm_stage1.means_.clone(), gmm_stage1.weights_.clone(), gmm_stage1.covariances_.clone()
    _init = dict(init_means=_m, init_weights=_w, init_covariances=_c)

    def _row(_name, _a, _b):
        _d = lambda _x, _y: float((_x - _y).abs().max())
        return (f"| {_name} | {_d(_a.means_, _b.means_):.1e} | {_d(_a.weights_, _b.weights_):.1e} | {_d(_a.covariances_, _b.covariances_):.1e} "
                f"| {float(_a.lower_bound_):.6f} / {float(_b.lower_bound_):.6f} | {_a.n_iter_} / {_b.n_iter_} |")

    _warm5 = GaussianMixture(random_state=0, **_fresh).fit(X_tensor, max_iter=5)
    _warm5.fit(X_tensor, max_iter=5, warm_start=True)
    _tens5 = GaussianMixture(random_state=0, **_fresh, **_init).fit(X_tensor, max_iter=5)
    _warmC = GaussianMixture(random_state=0, **_fresh).fit(X_tensor, max_iter=5)
    _warmC.fit(X_tensor, warm_start=True)
    _tensC = GaussianMixture(random_state=0, **_fresh, **_init).fit(X_tensor)
    _tensC7 = GaussianMixture(random_state=7, **_fresh, **_init).fit(X_tensor)

    _a0 = float((_warm5.initial_means_ - _m).abs().max())
    _b0 = float((_tens5.initial_means_ - _m).abs().max())
    mo.md(
        "| warm-started vs. fresh from tensors | max abs diff `means_` | `weights_` | `covariances_` | log-likelihood (warm / fresh) | `n_iter_` (warm / fresh) |\n|---|---|---|---|---|---|\n"
        + "\n".join([_row('continued with `max_iter=5`', _warm5, _tens5), _row('to convergence', _warmC, _tensC), _row('to convergence, fresh fit with `random_state=7`', _warmC, _tensC7)])
        + "\n\nThe fitted parameters, the log-likelihood and the iteration count agree (the differences are zero, or at the level of floating-point rounding). What differs is only bookkeeping. "
        f"`initial_means_` records where the *first* fit began (a warm-started `fit` does not overwrite it, it is {_a0:.2f} away from the tensors above), whereas the fresh fit records the tensors you passed ({_b0:.1e} away). "
        "`n_iter_` counts the iterations of the latest call only. And with all three tensors given nothing random is left, so `random_state` has no effect (third row), and extra restarts (`n_init > 1`) just repeat the same run; "
        "`warm_start=True` with `n_init > 1` warns about exactly that."
    )
    return (gmm_stage1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Any start you like

    So `warm_start` is the right tool when you already have a fitted model and want to refine it (for example after raising `max_iter`, or on updated data). The `init_*` tensors are more general: they need no previous fit, so they can encode *any* start, for example domain knowledge, or the output of `GMMInitializer` itself. The static methods are:

    ```python
    means = GMMInitializer.kmeans(X, k=4)                                   # or .kpp, .random, .points, .maxdist
    weights = GMMInitializer.init_weights_from_clusters(X, means)            # n_k / n by nearest mean
    covariances = GMMInitializer.init_covariances_empirical(X, means, covariance_type='full', reg_covar=1e-6)
    ```

    The cell below compares four starts for the same data: the default, the tensors built from the three calls above (which reach the same optimum as the default, in about the same number of iterations), a start from rough prior knowledge of the blob locations, and a deliberately poor start with all four means on a horizontal line far above the data.
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

    `StudentTMixture` uses the same `init_means` / `init_weights` / `init_covariances` options (the covariance starts as the scale matrix, and `dof_init` sets the initial degrees of freedom), see [15. Student-t Mixtures](15_student_t.ipynb). `VonMisesFisherMixture` initializes mean directions by spherical k-means and sets every $\kappa_k$ to `kappa_init`, see [16. Von Mises-Fisher Mixtures](16_von_mises_fisher.ipynb).

    ## Summary

    - EM only goes uphill, so it ends in the local maximum nearest its start; initialization decides which one.
    - `init_means` matters most. `kmeans` (the default) is the most reliable; `random`, `points`, `kpp` and `maxdist` land in worse optima more often. `maxdist` is only for clean, well separated data: a few outliers take its centres.
    - `init_weights` and `init_covariances` mostly affect speed once the means are good; a `random` covariance start is the exception and can cost many extra iterations or end in a worse optimum.
    - `n_init=n` keeps the best of $n$ runs (seeds `random_state + i`), failing with probability about $(1-p)^n$; the winner's seed is in `best_random_state_`.
    - To start from parameters you choose, pass `init_*` tensors (any start, `GMMInitializer` builds them) or use `warm_start=True` to continue a fitted model; continuing is exactly a fresh fit started from that model's parameters. Per-call `fit(...)` overrides (`max_iter`, `tol`, `random_state`, `warm_start`) leave the model's defaults untouched.

    **Next:** [4. Priors](04_priors.ipynb).

    **References:** Arthur & Vassilvitskii (2007), *k-means++: the advantages of careful seeding*, SODA. Lloyd (1982), *Least squares quantization in PCM*, IEEE Trans. Information Theory 28(2). Gonzalez (1985), *Clustering to minimize the maximum intercluster distance*, Theoretical Computer Science 38.
    """)
    return


if __name__ == "__main__":
    app.run()
