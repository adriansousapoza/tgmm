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
    N_SEEDS = 20 if FAST else 60  # number of independent starts in the sweeps below
    return FAST, MAX_ITER, N_SEEDS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialization: Means, Weights, Covariances and `n_init`

    EM climbs the likelihood from wherever it starts and stops at the nearest local maximum, so the starting parameters $(\pi_k, \mu_k, \Sigma_k)$ decide which maximum it finds. This tutorial shows how much that matters on the four-blob dataset from [1. Introduction](01_introduction.ipynb), what each initialization strategy does mathematically, and how `n_init` restarts guard against a bad start.

    **What you'll learn**

    - Why initialization matters: the log-likelihood is non-convex, and EM only ever goes uphill
    - The five mean initializers (`random`, `points`, `kpp`, `kmeans`, `maxdist`) and the math behind them, plus an HDBSCAN start that estimates the means, weights and covariances from its cluster assignments
    - How initial weights and covariances are derived, and how little or how much they change the result
    - How `n_init` works and how many restarts you need (best-of-$n$ selection), also when *everything* is initialized at random
    - Training controls (`max_iter`, `tol`, `reg_covar`) and how to start EM from parameters you give it: `init_*` tensors, `GMMInitializer`, and `warm_start`

    **Prerequisites:** [1. Introduction](01_introduction.ipynb).

    **See also:** [2. Covariance Structures](02_covariance_types.ipynb) for the shape constraints placed on $\Sigma_k$, [18 -- k-means++ Seeding](18_kmeans_plus_plus.ipynb) and [19 -- k-means (Lloyd's Algorithm)](19_kmeans.ipynb) for step-by-step visualisations of `kpp` and `kmeans`, and the [`GMMInitializer` API reference](../api/gmm-initializer.md).
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

    from scipy.stats import chi2

    from tgmm import CATEGORICAL_PALETTE, GaussianMixture, GMMInitializer, HDBSCAN, generate_gmm_data
    from tutorial_data import make_default_data

    warnings.filterwarnings('ignore')  # short max_iter budgets below intentionally stop before convergence
    # --- figure style (see notebooks/README.md) ---
    DOCS_RC = {'figure.dpi': 100, 'figure.titlesize': 12, 'axes.titlesize': 10.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
               'legend.fontsize': 8, 'lines.linewidth': 1.4, 'lines.markersize': 5}
    plt.rcParams.update(DOCS_RC)
    N_STD95 = float(np.sqrt(chi2.ppf(0.95, 2)))  # a 2-D Gaussian's 95% region is this many standard deviations out (about 2.45, not 2)
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (
        CATEGORICAL_PALETTE,
        Ellipse,
        GMMInitializer,
        GaussianMixture,
        HDBSCAN,
        N_STD95,
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
    Far-away points are likely to be chosen, so the centres spread out. Arthur & Vassilvitskii (2007) show the resulting k-means objective is within $O(\log K)$ of the optimum in expectation. (Step by step in [18 -- k-means++ Seeding](18_kmeans_plus_plus.ipynb).)

    **`'kmeans'` (the default).** Start from k-means++ and run Lloyd's algorithm: alternate
    $$
    c_j = \arg\min_i \lVert x_j - \mu_i \rVert^2 \quad\text{(assign)}, \qquad \mu_i = \frac{1}{|C_i|}\sum_{x_j \in C_i} x_j \quad\text{(update)}
    $$
    until the centres stop moving. Each step does not increase $\sum_j \lVert x_j - \mu_{c_j}\rVert^2$, so this converges to a local minimum of the k-means objective, which is usually a good place for EM to start. (Step by step in [19 -- k-means (Lloyd's Algorithm)](19_kmeans.ipynb).)

    **`'maxdist'`.** Farthest-first traversal. After a random first pick, repeatedly choose
    $$
    \mu_i = \arg\max_{x} \min_{l < i} \lVert x - \mu_l \rVert,
    $$
    then re-select the first centre as $\mu_1 = \arg\max_x \min_{l \ge 2} \lVert x - \mu_l \rVert$ so the arbitrary first pick does not bias the result. This is Gonzalez's (1985) farthest-first traversal, the classical 2-approximation for the *k-center* objective (minimise the largest distance to a centre), not for the k-means objective or the likelihood. It spreads centres maximally, but outliers are by definition the farthest points, so it is sensitive to them; its only randomness is the first pick (torch's global RNG).

    **`hdbscan` (from init tensors).** Not a built-in `init_means` name, but HDBSCAN's cluster assignments give all three starting parameters at once, so it is used here through the three init tensors. It needs no $K$ and ignores noise; the math, the helper and a figure are in the section *HDBSCAN: Initialise Everything from the Assignments* below.

    The next figure shows where each method puts its four initial centres (red crosses) for one seed (the static docs page shows seed 0; the interactive version at the end of the notebook has a **seed** slider). `random`, `points` and `maxdist` use the global torch seed; `kpp` and `kmeans` take a `random_state` argument; the `hdbscan` start is deterministic here (the seed would only place padding centres). `random`, `points` and `maxdist` move around a lot from seed to seed, while `kmeans` varies much less (it ends on one of only a few different solutions).
    """)
    return


@app.cell
def _(
    GMMInitializer,
    MEAN_METHODS,
    X,
    X_tensor,
    hd_labels,
    hdbscan_init,
    plt,
    torch,
):
    GALLERY_METHODS = MEAN_METHODS + ['hdbscan']

    def draw_means_gallery(seed):
        """The initial centres of the six start methods for one seed, as a 3 x 2 grid."""
        _fig, _axs = plt.subplots(3, 2, figsize=(11, 13.5), sharex=True, sharey=True)
        for _ax, _method in zip(_axs.ravel(), GALLERY_METHODS):
            torch.manual_seed(seed)
            if _method == 'hdbscan':
                _means = hdbscan_init(X_tensor, 4, seed=seed, labels=hd_labels)[0]
                _title = 'HDBSCAN init tensors'
            elif _method in ('kpp', 'kmeans'):
                _means = getattr(GMMInitializer, _method)(X_tensor, 4, random_state=seed)
                _title = f"init_means='{_method}'"
            else:
                _means = getattr(GMMInitializer, _method)(X_tensor, 4)
                _title = f"init_means='{_method}'"
            _m = _means.cpu().numpy()
            _h_data = _ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.5, color='gray', label='data')
            _h_ctr = _ax.scatter(_m[:, 0], _m[:, 1], marker='H', s=30, linewidths=1.2, color='red', zorder=3, label='initial centre')
            _ax.set_title(_title)
            _ax.set_aspect('equal')
        _fig.legend(handles=[_h_data, _h_ctr], loc='lower center', ncol=2, frameon=False)
        _fig.suptitle(f'seed = {seed}')
        _fig.tight_layout(rect=[0, 0.025, 1, 0.98])
        return _fig

    return (draw_means_gallery,)


@app.cell
def _(SEED, draw_means_gallery, plt):
    draw_means_gallery(SEED)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(
    BEST_LL,
    MEAN_METHODS,
    N_SEEDS,
    TOL_HIT,
    X_tensor,
    hd_found,
    hd_noise,
    hdbscan_ll,
    mo,
    np,
    sweep,
):
    _hit = {_m: f"{(sweep[_m] > BEST_LL - TOL_HIT).mean():.0%}" for _m in MEAN_METHODS}
    _hit['hdbscan'] = f"{(hdbscan_ll > BEST_LL - TOL_HIT).mean():.0%} (deterministic: {len(np.unique(np.round(hdbscan_ll, 5)))} distinct outcome(s))"
    mo.md(f"""
    ### Which `init_means` when?

    | `init_means` | What it does | Pros | Cons | Good for / bad for | Runs reaching the best optimum ({N_SEEDS} single runs, default blobs) |
    |---|---|---|---|---|---|
    | `random` | draws centres from $\\mathcal N(\\bar x,\\hat\\Sigma)$ (not data points) | cheapest; centres follow the global spread | centres can fall in empty space or pile into one cluster; ignores local structure | **good:** a quick baseline together with `n_init` restarts. **bad:** well separated or many clusters | {_hit['random']} |
    | `points` | $K$ distinct data points, uniformly at random | cheap; centres sit on real data, more often in dense regions | two points can come from one cluster and none from a small one; ignores spread | **good:** clusters of similar size. **bad:** small clusters (rarely hit) | {_hit['points']} |
    | `kpp` | first point uniform, then each next point drawn with probability $\\propto D^2$ to the nearest centre | spreads centres yet is randomized, so a lone outlier is rarely picked; $O(\\log K)$-competitive for the k-means objective | still seed dependent; no refinement step | **good:** a general-purpose start, mild outliers, many clusters. **bad:** nothing in particular; use `n_init` for unlucky draws | {_hit['kpp']} |
    | `kmeans` (default) | k-means++ followed by Lloyd iterations until the centres stop moving | centres are already cluster means; reaches the best optimum most often here | costs a k-means run; favours round, similar-sized clusters and can cut an elongated cluster in two (EM usually repairs this) | **good:** compact, roughly round clusters, the safe default. **bad:** very elongated or very unequal clusters | {_hit['kmeans']} |
    | `maxdist` | farthest-first traversal: random first point, then always the point farthest from all centres so far | no iterations; deterministic apart from the first point; finds one point per cluster when clusters are further apart than their diameter | picks extreme points: outliers, heavy tails or elongated clusters steal centres; ignores density; can leave a true cluster without a centre | **good:** compact, well separated clusters *without* outliers. **bad:** noisy data, outliers, overlapping or elongated clusters | {_hit['maxdist']} |
    | `hdbscan` (init tensors) | HDBSCAN clustering, noise dropped; cluster means, size fractions and covariances become `init_means`, `init_weights`, `init_covariances` | needs no $K$ to find the dense clusters; noise and outliers are ignored, so they cannot take a centre; copes with non-convex or very unequal clusters; supplies weights and covariances too | needs `min_cluster_size` (different values find a different number of clusters); may find fewer or more than $K$ (on the default blobs: {hd_found} clusters and {hd_noise} of {len(X_tensor)} points called noise{', so a centre is padded' if hd_found < 4 else ''}); can call a real cluster noise; $O(n^2)$ memory and time in this implementation; deterministic, so `n_init` does not diversify it | **good:** noisy data with outliers, non-convex or very unequal clusters. **bad:** heavily overlapping Gaussian clusters (the overlap between the peaks becomes noise), very large $n$ | {_hit['hdbscan']} |
    | tensor `(K, d)` | your own centres | uses prior knowledge or a previous fit; the same start every run | a wrong guess is a bad start, and `n_init` only repeats it | **good:** known cluster positions (see *Starting from Given Parameters*) | n/a |
    """)
    return


@app.cell
def _(
    GaussianMixture,
    HDBSCAN,
    MAX_ITER,
    N_SEEDS,
    device,
    generate_gmm_data,
    hdbscan_init,
    mo,
    np,
    torch,
):
    # a small experiment on the claims in the table: four well separated, compact clusters, with and without 1% uniform outliers
    _centres = np.array([(0, 0), (8, 0), (0, 8), (8, 8)], dtype=float)
    _X_clean, _ = generate_gmm_data([_c for _c in _centres], [np.eye(2)] * 4, [300] * 4, device=device, random_state=1)
    _out = torch.tensor(np.random.default_rng(0).uniform(-12, 20, (12, 2)), dtype=_X_clean.dtype, device=_X_clean.device)
    _X_noisy = torch.cat([_X_clean, _out])

    def _hd_labels(_X):
        _n = len(_X)
        return HDBSCAN(min_cluster_size=max(10, int(0.02 * _n)), min_samples=max(5, int(0.02 * _n) // 2), device=device).fit(_X).labels_

    _labels = {id(_X_clean): _hd_labels(_X_clean), id(_X_noisy): _hd_labels(_X_noisy)}

    def _recovers(_X, _method, _seed):
        """True when every true centre has a fitted mean within 1.5 (a crashed fit counts as a failure)."""
        torch.manual_seed(_seed)
        try:
            if _method == 'hdbscan':
                _m, _w, _c = hdbscan_init(_X, 4, seed=_seed, labels=_labels[id(_X)])
                _g = GaussianMixture(n_components=4, init_means=_m, init_weights=_w, init_covariances=_c, random_state=_seed, max_iter=MAX_ITER, device=device).fit(_X)
            else:
                _g = GaussianMixture(n_components=4, init_means=_method, random_state=_seed, max_iter=MAX_ITER, device=device).fit(_X)
        except ValueError:
            return False
        _d = np.linalg.norm(_centres[:, None] - _g.means_.detach().cpu().numpy()[None], axis=2)
        return bool((_d.min(axis=1) < 1.5).all())

    _rows = []
    _res = {}
    for _method in ['maxdist', 'kmeans', 'kpp', 'hdbscan']:
        _clean = sum(_recovers(_X_clean, _method, _s) for _s in range(N_SEEDS))
        _noisy = sum(_recovers(_X_noisy, _method, _s) for _s in range(N_SEEDS))
        _res[_method] = (_clean, _noisy)
        _rows.append(f"| `{_method}` | {_clean}/{N_SEEDS} | {_noisy}/{N_SEEDS} |")
    _lab_n = _labels[id(_X_noisy)]
    mo.md(
        "On four well separated, compact clusters, the number of single runs (of "
        f"{N_SEEDS}) in which every true cluster ends up with a fitted component:\n\n"
        "| `init_means` | clean data | plus 12 uniform outliers (1%) |\n|---|---|---|\n" + "\n".join(_rows)
        + "\n\n`maxdist` is excellent on the clean data and collapses as soon as a few outliers appear, because the outliers are the farthest points and take the centres. "
        "`kpp` samples with probability $\\propto D^2$ instead of taking the maximum, so a lone outlier is rarely chosen. "
        f"The HDBSCAN start recovers the clusters in {_res['hdbscan'][0]}/{N_SEEDS} and {_res['hdbscan'][1]}/{N_SEEDS} runs: on the noisy data it found {int(_lab_n.max()) + 1} clusters and called {int((_lab_n == -1).sum())} points noise, "
        "so the outliers are dropped before any centre is chosen (it is deterministic, so every run is the same start)."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Which one to use.** `kmeans` (the default) is the robust choice: it is cheap compared with EM and puts the centres where the clusters are ([19 -- k-means (Lloyd's Algorithm)](19_kmeans.ipynb) shows it step by step). `kpp` is the randomized, outlier-tolerant version of farthest-first and a good lighter-weight alternative ([18 -- k-means++ Seeding](18_kmeans_plus_plus.ipynb)). Use `maxdist` only for clean, well separated data. The HDBSCAN start is worth a look when the data have outliers or non-convex clusters, and it needs no $K$ to find them; it costs an $O(n^2)$ clustering and a `min_cluster_size` choice, and it is a poor match for heavily overlapping Gaussian clusters. The random methods (`random`, `points`, `kpp`) should be combined with `n_init` (below). Variants such as *greedy* k-means++ (several candidates per step, keep the best) and robust farthest-first traversals (trim outliers first, or start from the densest point) reduce these weaknesses further; tgmm does not implement them, but you can build such centres yourself and pass them as a tensor.
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
    | tensor from HDBSCAN (`hdbscan_init`, see below) | $\|C_k\|/n$ of each HDBSCAN cluster, noise left out, renormalized; padded components get a share of the noise fraction | consistent with the assignments that give the means and covariances; follows the density structure instead of the nearest mean | depends on `min_cluster_size`; clusters that HDBSCAN merges or splits give wrong sizes | **good:** unequal sizes together with noise or outliers. **bad:** overlapping blobs (much of the data becomes noise) |
    | tensor | the given values, renormalized to sum to 1 | uses known proportions | a wrong guess is a bad start | **good:** known class proportions |

    The weights only enter the first E-step as the prior factor in $r_{ik} \propto \pi_k\,\mathcal{N}(x_i \mid \mu_k,\Sigma_k)$. Once the means are sensible, they are the least influential of the three initializations: the M-step recomputes $\pi_k = \frac{1}{n}\sum_i r_{ik}$ immediately. The cell below fixes the initial means (k-means, seed 0) and varies only the weights; `initial_weights_` records what each run started from.
    """)
    return


@app.cell
def _(
    GMMInitializer,
    GaussianMixture,
    MAX_ITER,
    SEED,
    X_tensor,
    device,
    mo,
    torch,
):
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
    | tensor from HDBSCAN (`hdbscan_init`, see below) | $\widehat{\mathrm{Cov}}(C_k) + \texttt{reg\_covar}\, I$ of each HDBSCAN cluster, noise left out; padded components get the global covariance | shapes come from a density-based assignment, so outliers do not inflate them | needs $n_k > d$ points per cluster; merged or split clusters give a wrong shape; overlapping blobs give too-tight clusters | **good:** noisy or non-convex clusters. **bad:** heavily overlapping Gaussian clusters |
    | tensor | the given values (shape per `covariance_type`) | uses prior knowledge or a previous fit | a wrong guess is a bad start | **good:** known cluster shapes |

    For the diagonal, spherical and tied covariance types the corresponding diagonal, mean-of-diagonal or pooled form is used (see [2. Covariance Structures](02_covariance_types.ipynb)). For the `tied_*` types `'empirical'` pools the within-cluster scatter, $\Sigma = \frac1n \sum_k \sum_{x\in C_k}(x - \bar x_{C_k})(x - \bar x_{C_k})^\top + \texttt{reg\_covar}\, I$.

    The figure draws the 95% ellipses each method starts from (around the same k-means initial means); the sentence below it compares the resulting fits.
    """)
    return


@app.cell
def _(
    GMMInitializer,
    GaussianMixture,
    MAX_ITER,
    N_SEEDS,
    SEED,
    X_tensor,
    device,
    np,
):
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
def _(
    CATEGORICAL_PALETTE,
    COV_METHODS,
    Ellipse,
    N_STD95,
    X,
    cov_runs,
    np,
    plt,
):
    def cov_ellipse(ax, mean, cov, n_std=N_STD95, **kwargs):
        w, V = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        ax.add_patch(Ellipse(mean, 2 * n_std * np.sqrt(w[1]), 2 * n_std * np.sqrt(w[0]), angle=angle, fill=False, **kwargs))

    _fig, _axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for _ax, _method in zip(_axs.ravel(), COV_METHODS):
        _g = cov_runs[_method]
        _mu, _S = _g.initial_means_.cpu().numpy(), _g.initial_covariances_.cpu().numpy()
        _ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.5, color='gray')
        for _k in range(4):
            cov_ellipse(_ax, _mu[_k], _S[_k], color=CATEGORICAL_PALETTE[_k])
        _ax.scatter(_mu[:, 0], _mu[:, 1], marker='x', s=30, linewidths=1.2, color='red', zorder=3)
        _ax.set_title(f"init_covariances='{_method}'")
        _ax.set_aspect('equal')
    plt.tight_layout()
    plt.gcf()
    return (cov_ellipse,)


@app.cell
def _(
    BEST_LL,
    COV_METHODS,
    N_SEEDS,
    TOL_HIT,
    cov_random_iters,
    cov_random_ll,
    cov_runs,
    mo,
):
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
    ## HDBSCAN: Initialise Everything from the Assignments

    The other initializers choose centres first and derive weights and covariances from "nearest centre" assignments. tgmm's `HDBSCAN` class ([12 -- HDBSCAN](12_hdbscan.ipynb)) works the other way round: it returns *cluster assignments* $z_i \in \{1,\dots,m\}$ (and $-1$ for noise) without being told $K$, so everything the first E-step needs can be estimated from them. For each cluster $C_k = \{i : z_i = k\}$

    $$
    \mu_k = \frac{1}{|C_k|}\sum_{i\in C_k} x_i, \qquad \pi_k = \frac{|C_k|}{n}, \qquad \Sigma_k = \widehat{\mathrm{Cov}}(C_k) + \texttt{reg\_covar}\, I,
    $$

    with the weights renormalized to sum to 1. The noise points are left out of all three, so outliers cannot distort any of them. HDBSCAN may find $m \neq K$ clusters: the helper below keeps the $K$ largest when $m > K$ and, when $m < K$, adds the missing centres by $D^2$-sampling from the noise points, giving them the global covariance and a share of the noise fraction as weight. It uses `min_cluster_size` equal to 2% of the points and `min_samples` half of that, a rule of thumb that is not tuned (the sensitivity is measured at the end of this section).

    This works today through the init tensors `init_means`, `init_weights` and `init_covariances`; a built-in `init_means='hdbscan'` option exists on a library branch awaiting release.
    """)
    return


@app.cell
def _(HDBSCAN, torch):
    def hdbscan_init(X, K, min_cluster_size=None, min_samples=None, reg_covar=1e-6, seed=0, labels=None):
        """Init tensors ``(means, weights, covariances)`` from an HDBSCAN clustering of ``X`` (``(n, d)``), for ``covariance_type='full'``.

        Noise points (label -1) are ignored. Each of the K largest clusters gives a mean, a size fraction and a regularised covariance.
        Fewer than K clusters: the missing centres are D^2-sampled from the noise points (or from all points if there are too few),
        with the global covariance and a share of the noise fraction as weight. ``labels`` lets you reuse an HDBSCAN fit.
        """
        _n, _d = X.shape
        _mcs = min_cluster_size or max(10, int(0.02 * _n))
        _ms = min_samples or max(5, _mcs // 2)
        if labels is None:
            labels = HDBSCAN(min_cluster_size=_mcs, min_samples=_ms, device=X.device.type).fit(X).labels_
        labels = labels.to(X.device)
        _m = int(labels.max()) + 1 if labels.numel() else 0
        _ids = sorted([_k for _k in range(_m) if int((labels == _k).sum()) > 0], key=lambda _k: -int((labels == _k).sum()))[:K]
        _eye = torch.eye(_d, device=X.device, dtype=X.dtype)
        _means, _covs, _weights = [], [], []
        for _k in _ids:
            _P = X[labels == _k]
            _means.append(_P.mean(0))
            _covs.append(torch.cov(_P.T) + reg_covar * _eye)
            _weights.append(len(_P) / _n)
        _n_noise = int((labels == -1).sum())
        _n_extra = K - len(_ids)
        if _n_extra > 0:
            _gen = torch.Generator(device=X.device)
            _gen.manual_seed(int(seed))
            _pool = X[labels == -1] if _n_noise >= _n_extra else X  # the points no cluster claims first
            _centres = torch.stack(_means) if _means else None
            _extra = []
            for _ in range(_n_extra):
                if _centres is None:
                    _idx = torch.randint(0, len(_pool), (1,), device=X.device, generator=_gen)
                else:
                    _d2 = torch.cdist(_pool, _centres).pow(2).min(dim=1).values
                    _p = _d2 / _d2.sum() if float(_d2.sum()) > 0 else torch.full_like(_d2, 1.0 / len(_d2))
                    _idx = torch.multinomial(_p, 1, generator=_gen)
                _extra.append(_pool[_idx[0]])
                _centres = torch.stack(_extra) if _centres is None else torch.cat([_centres, _extra[-1][None]])
            _glob = torch.cov(X.T) + reg_covar * _eye
            _means += _extra
            _covs += [_glob] * _n_extra
            _weights += [max(_n_noise / _n, 1e-3) / _n_extra] * _n_extra
        _w = torch.tensor(_weights, device=X.device, dtype=X.dtype)
        return torch.stack(_means), _w / _w.sum(), torch.stack(_covs)

    return (hdbscan_init,)


@app.cell
def _(
    GaussianMixture,
    HDBSCAN,
    MAX_ITER,
    N_SEEDS,
    X_tensor,
    device,
    hdbscan_init,
    np,
):
    # one HDBSCAN fit (it is deterministic), then the same single-run sweep as for the other methods
    _n = len(X_tensor)
    hd_labels = HDBSCAN(min_cluster_size=max(10, int(0.02 * _n)), min_samples=max(5, int(0.02 * _n) // 2), device=device).fit(X_tensor).labels_
    hdbscan_ll = np.array([
        float(GaussianMixture(n_components=4, init_means=_m, init_weights=_w, init_covariances=_c, random_state=_s, max_iter=MAX_ITER, device=device).fit(X_tensor).lower_bound_)
        for _s in range(N_SEEDS)
        for _m, _w, _c in [hdbscan_init(X_tensor, 4, seed=_s, labels=hd_labels)]
    ])
    hd_found = int(hd_labels.max()) + 1
    hd_noise = int((hd_labels == -1).sum())
    return hd_found, hd_labels, hd_noise, hdbscan_ll


@app.cell
def _(
    CATEGORICAL_PALETTE,
    GaussianMixture,
    MAX_ITER,
    N_STD95,
    X,
    X_tensor,
    cov_ellipse,
    device,
    hd_found,
    hd_labels,
    hd_noise,
    hdbscan_init,
    plt,
):
    _K = 4
    _m, _w, _c = hdbscan_init(X_tensor, _K, seed=0, labels=hd_labels)
    _fit = GaussianMixture(n_components=_K, init_means=_m, init_weights=_w, init_covariances=_c, random_state=0, max_iter=MAX_ITER, device=device).fit(X_tensor)
    _mu, _S, _pi = _m.cpu().numpy(), _c.cpu().numpy(), _w.cpu().numpy()
    _lab = hd_labels.cpu().numpy()
    _order = sorted(range(hd_found), key=lambda _k: -int((_lab == _k).sum()))[:_K]  # cluster ids by decreasing size = the order of the init components
    _fig, _axs = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    _axs[0].scatter(X[_lab == -1, 0], X[_lab == -1, 1], s=4, alpha=0.5, color='lightgray', label='noise')
    for _j, _k in enumerate(_order):
        _axs[0].scatter(X[_lab == _k, 0], X[_lab == _k, 1], s=4, alpha=0.5, color=CATEGORICAL_PALETTE[_j], label=f'cluster {_j + 1}')
    _axs[0].set_title(f'HDBSCAN assignments: {hd_found} clusters, {hd_noise} noise points')
    _axs[0].legend(markerscale=3, loc='upper left')
    _axs[1].scatter(X[:, 0], X[:, 1], s=4, alpha=0.5, color='lightgray')
    for _j in range(_K):
        _padded = _j >= len(_order)  # a centre added by D^2-sampling from the noise, with the global covariance
        cov_ellipse(_axs[1], _mu[_j], _S[_j], n_std=N_STD95, color=CATEGORICAL_PALETTE[_j], ls=':' if _padded else '-')
        _axs[1].annotate(f'$\\pi$={_pi[_j]:.2f}' + (' (padded)' if _padded else ''), _mu[_j], textcoords='offset points', xytext=(6, 6), color=CATEGORICAL_PALETTE[_j])
        cov_ellipse(_axs[1], _fit.means_[_j].detach().cpu().numpy(), _fit.covariances_[_j].detach().cpu().numpy(), n_std=N_STD95, color='black', ls='--', lw=0.8)
    _axs[1].scatter(_mu[:, 0], _mu[:, 1], marker='x', s=30, linewidths=1.2, color='red', zorder=3, label='initial mean')
    _fm = _fit.means_.detach().cpu().numpy()
    _axs[1].scatter(_fm[:, 0], _fm[:, 1], marker='h', s=25, color='black', zorder=3, label='fitted mean')
    _axs[1].plot([], [], color='gray', label='initial Gaussians (95%)')
    _axs[1].plot([], [], color='black', ls='--', lw=0.8, label='fitted GMM (95%)')
    _axs[1].set_title('Initial Gaussians from the assignments')
    _axs[1].legend(loc='upper left')
    for _ax in _axs:
        _ax.set_aspect('equal')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(
    GaussianMixture,
    MAX_ITER,
    X_tensor,
    device,
    hd_found,
    hd_labels,
    hd_noise,
    hdbscan_init,
    mo,
):
    _found, _noise = hd_found, hd_noise
    _m, _w, _c = hdbscan_init(X_tensor, 4, seed=0, labels=hd_labels)
    _starts = {
        'default: `kmeans` means, uniform weights, empirical covariances': dict(),
        'HDBSCAN means only': dict(init_means=_m),
        'HDBSCAN means, weights and covariances (all from the assignments)': dict(init_means=_m, init_weights=_w, init_covariances=_c),
    }
    _rows, _res = [], {}
    for _name, _kw in _starts.items():
        _g = GaussianMixture(n_components=4, random_state=0, max_iter=MAX_ITER, device=device, **_kw).fit(X_tensor)
        _res[_name] = (float(_g.lower_bound_), _g.n_iter_)
        _rows.append(f"| {_name} | {float(_g.lower_bound_):.4f} | {_g.n_iter_} | {_g.converged_} |")
    _lls = [_v[0] for _v in _res.values()]
    _its = [_v[1] for _v in _res.values()]
    mo.md(
        "| start | final log-likelihood | EM iterations | converged |\n|---|---|---|---|\n" + "\n".join(_rows)
        + f"\n\nThe three starts end within {max(_lls) - min(_lls):.1e} of each other in log-likelihood; they differ in the number of EM iterations "
        f"({_its[0]} for the default start, {_its[1]} and {_its[2]} for the HDBSCAN starts: they begin at the dense cores of the clusters, so EM has little left to do). "
        + (f"HDBSCAN finds {_found} clusters here (so no centre is padded), but {_noise} of the {len(X_tensor)} points are called noise: the overlap between the blobs is low-density, so the clusters it does find are the dense cores, and their covariances are smaller than the blobs' own."
           if _found >= 4 else
           f"HDBSCAN finds only {_found} clusters here, so {4 - _found} starting component(s) are padded ones with the global covariance, which EM has to move into place.")
    )
    return


@app.cell
def _(
    BEST_LL,
    FAST,
    GaussianMixture,
    HDBSCAN,
    MAX_ITER,
    TOL_HIT,
    X_tensor,
    device,
    hdbscan_init,
    mo,
    np,
):
    # how sensitive is the HDBSCAN start to min_cluster_size? (min_samples = half of it)
    _seeds = 5 if FAST else 20
    _rows = []
    for _mcs in ([30, 60, 100, 150] if FAST else [15, 30, 60, 100, 150]):
        _ms = max(5, _mcs // 2)
        _lab = HDBSCAN(min_cluster_size=_mcs, min_samples=_ms, device=device).fit(X_tensor).labels_
        _ll = np.array([
            float(GaussianMixture(n_components=4, init_means=_m, init_weights=_w, init_covariances=_c, random_state=_s, max_iter=MAX_ITER, device=device).fit(X_tensor).lower_bound_)
            for _s in range(_seeds)
            for _m, _w, _c in [hdbscan_init(X_tensor, 4, min_cluster_size=_mcs, min_samples=_ms, seed=_s, labels=_lab)]
        ])
        _rows.append(f"| {_mcs} | {_ms} | {int(_lab.max()) + 1} | {int((_lab == -1).sum())} of {len(X_tensor)} | {(_ll > BEST_LL - TOL_HIT).mean():.0%} |")
    mo.md(
        "**Sensitivity of the HDBSCAN start to `min_cluster_size`** (default blobs, $K=4$):\n\n"
        f"| `min_cluster_size` | `min_samples` | clusters found | noise points | runs reaching the best optimum ({_seeds} seeds) |\n|---|---|---|---|---|\n" + "\n".join(_rows)
        + "\n\nThe number of clusters HDBSCAN finds changes a lot with `min_cluster_size` (and on these overlapping blobs a large share of the points is called noise, since the overlap between the peaks is low-density), "
        "but the padding and truncation in the helper keep the start usable: EM repairs a start with too few clusters as long as the padded centres land somewhere sensible."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The strength of the HDBSCAN start is not these overlapping blobs but data with outliers or noise and with clusters that are not round (see the clean-versus-outliers experiment in *Initial Means*): the assignments it gives are exactly what the weights and the covariances should be estimated from, because the outliers never enter any of them.
    """)
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

    So far only the means were random. The experiment below initializes *everything* at random: `init_means='random'`, `init_weights='random'` (a Dirichlet draw) and `init_covariances='random'` (matrices unrelated to the data), nothing data-driven. The next cell runs the single runs, estimates $p$ from them, and compares the formula with what actually happens when we take the best of $n$ consecutive seeds.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, N_SEEDS, X_tensor, device, np):
    # every part of the start at random: means, weights and covariances
    sweep_all_random = np.array([
        float(GaussianMixture(n_components=4, init_means='random', init_weights='random', init_covariances='random', random_state=_s, max_iter=MAX_ITER, device=device).fit(X_tensor).lower_bound_)
        for _s in range(N_SEEDS)
    ])
    return (sweep_all_random,)


@app.cell
def _(BEST_LL, N_SEEDS, TOL_HIT, np, plt, sweep_all_random):
    _ll = sweep_all_random
    _p = float((_ll > BEST_LL - TOL_HIT).mean())
    _ns = list(range(1, 9))
    _emp = [np.mean([_ll[_b:_b + _n].max() > BEST_LL - TOL_HIT for _b in range(N_SEEDS - _n + 1)]) for _n in _ns]
    _fig, _ax = plt.subplots(figsize=(6.5, 4.5))
    _ax.plot(_ns, [1 - (1 - _p) ** _n for _n in _ns], 'o--', label=f'formula $1-(1-p)^n$, $p$ = {_p:.2f}')
    _ax.plot(_ns, _emp, 's-', label='observed (best of $n$ consecutive seeds)')
    _ax.set_xlabel('n_init')
    _ax.set_ylabel('probability of reaching the best optimum')
    _ax.set_ylim(0, 1.05)
    _ax.set_title("means, weights and covariances all random")
    _ax.legend(loc='lower right')
    plt.gcf()
    return


@app.cell
def _(BEST_LL, N_SEEDS, TOL_HIT, mo, np, sweep, sweep_all_random):
    _p_means = float((sweep['random'] > BEST_LL - TOL_HIT).mean())
    _p_all = float((sweep_all_random > BEST_LL - TOL_HIT).mean())
    _se = float(np.sqrt(max(_p_means * (1 - _p_means), _p_all * (1 - _p_all)) / N_SEEDS))  # standard error of a proportion from N_SEEDS runs
    mo.md(
        f"Over {N_SEEDS} single runs, a start with only the means at random reaches the best optimum with $p$ = {_p_means:.2f}, and a start with *everything* at random with $p$ = {_p_all:.2f}: "
        f"a difference of {abs(_p_all - _p_means) * 100:.0f} percentage points, against a standard error of about {_se * 100:.0f} points for each estimate from {N_SEEDS} runs, "
        "so these runs cannot tell the two apart much. The covariance experiment above is still a reminder that a random covariance start can cost iterations or end in a worse optimum, "
        f"and the worst fully random run here ends at {sweep_all_random.min():.4f}, against a best of {BEST_LL:.4f}. In both cases the formula above says how many restarts make up for the failures."
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
    return


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
def _(
    GMMInitializer,
    GaussianMixture,
    MAX_ITER,
    SEED,
    X_tensor,
    device,
    mo,
    torch,
):
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
    - `init_means` matters most. `kmeans` (the default) is the most reliable; `random`, `points`, `kpp` and `maxdist` land in worse optima more often. `maxdist` is only for clean, well separated data: a few outliers take its centres. An HDBSCAN start (built from init tensors) estimates the means, weights and covariances from its cluster assignments, so noise and outliers never enter them.
    - `init_weights` and `init_covariances` mostly affect speed once the means are good; a `random` covariance start is the exception and can cost many extra iterations or end in a worse optimum.
    - `n_init=n` keeps the best of $n$ runs (seeds `random_state + i`), failing with probability about $(1-p)^n$, also when means, weights and covariances are all random; the winner's seed is in `best_random_state_`.
    - To start from parameters you choose, pass `init_*` tensors (any start, `GMMInitializer` builds them) or use `warm_start=True` to continue a fitted model; continuing is exactly a fresh fit started from that model's parameters. Per-call `fit(...)` overrides (`max_iter`, `tol`, `random_state`, `warm_start`) leave the model's defaults untouched.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it yourself (interactive version)

    The page you are reading shows the default setting (seed 0). In the interactive version of this notebook (the link at the top of the page, on molab, or `marimo edit` locally) the **seed** slider below redraws where all six start methods put their initial centres for any seed from 0 to 29: `random`, `points` and `maxdist` jump around, `kmeans` and `hdbscan` hardly move.
    """)
    return


@app.cell
def _(SEED, mo):
    # docs: hide-cell
    init_seed = mo.ui.slider(0, 29, step=1, value=SEED, label='seed', show_value=True)
    init_seed
    return (init_seed,)


@app.cell
def _(SEED, draw_means_gallery, init_seed, mo, plt):
    # docs: hide-cell
    if int(init_seed.value) == SEED:
        _out = mo.md(f'*Move the slider to redraw the gallery for another seed (seed {SEED} is the figure above).*')
    else:
        draw_means_gallery(int(init_seed.value))
        _out = plt.gcf()
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Next:** [4. Priors](04_priors.ipynb).

    **References:** Arthur & Vassilvitskii (2007), *k-means++: the advantages of careful seeding*, SODA. Lloyd (1982), *Least squares quantization in PCM*, IEEE Trans. Information Theory 28(2). Gonzalez (1985), *Clustering to minimize the maximum intercluster distance*, Theoretical Computer Science 38. Campello, Moulavi & Sander (2013), *Density-based clustering based on hierarchical density estimates*, PAKDD. McInnes, Healy & Astels (2017), *hdbscan: Hierarchical density based clustering*, Journal of Open Source Software 2(11).
    """)
    return


if __name__ == "__main__":
    app.run()
