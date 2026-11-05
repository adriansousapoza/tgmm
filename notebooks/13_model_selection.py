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
    return FAST, MAX_ITER


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Selection: BIC, AIC, and ICL

    How many components should a mixture have, and which covariance structure? `tgmm.select_model` fits a `GaussianMixture` for every requested `n_components` (and, optionally, every `covariance_type`), records `bic`, `aic` and `icl` for each fit, and returns the best one. This notebook compares the three criteria on a dataset built to make them disagree, including the Integrated Completed Likelihood, `GaussianMixture.icl` (Biernacki, Celeux & Govaert, 2000).

    BIC and AIC score a fitted mixture purely on log-likelihood plus a complexity penalty -- they don't care whether the mixture's components are cleanly separated or heavily overlapping, as long as the extra components earn their keep in likelihood. ICL adds an entropy penalty for *ambiguous* assignments (`-2 * sum_i log(responsibility of the point's most likely component)`), so on data with overlapping clusters it tends to prefer **fewer, better-separated** components than BIC does on the same data, even though both coincide when clusters are perfectly separated (one-hot responsibilities).

    **What you'll learn**

    - how to sweep `n_components` with `select_model` and read its results table
    - how BIC, AIC and ICL differ, and why ICL prefers well-separated partitions
    - how to compute `icl` (hard and soft) on a single fitted model
    - the formulas behind AIC, BIC and ICL, checked numerically against the library
    - how to sweep the covariance structure together with `n_components`, and how cluster separation and background noise move the choice

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb) (fitting a mixture, the log-likelihood), [02 -- Covariance Types](02_covariance_types.ipynb) (the parameter counts), [09 -- Clustering Metrics](09_metrics.ipynb) (label-free validity indices, the other way to compare clusterings).
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

    from sklearn.datasets import load_iris

    from tgmm import GaussianMixture, generate_gmm_data, select_model
    from tgmm.covariance.gpcm import GPCM_CODES
    from tgmm.plotting import CATEGORICAL_PALETTE, plot_gmm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f'Using device: {device}')
    return CATEGORICAL_PALETTE, GPCM_CODES, GaussianMixture, SEED, generate_gmm_data, load_iris, np, plot_gmm, plt, select_model, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The three criteria

    Let $\hat\ell=\sum_i\log p(x_i\mid\hat\theta)$ be the maximised log-likelihood of $n$ points and $p$ the number of free parameters. Then

    $$
    \mathrm{AIC}=-2\hat\ell+2p,\qquad
    \mathrm{BIC}=-2\hat\ell+p\ln n,\qquad
    \mathrm{ICL}=\mathrm{BIC}+2\,\mathrm{EN}.
    $$

    All three are *lower is better* (`gmm.aic(X)`, `gmm.bic(X)`, `gmm.icl(X)`). BIC charges $\ln n$ per parameter, AIC only $2$, so BIC prefers smaller models once $n>e^2\approx7$. The entropy term $\mathrm{EN}\ge0$ is the classification-entropy penalty of Biernacki, Celeux & Govaert (2000). With `mode='hard'` (default)

    $$
    \mathrm{EN}=-\sum_i\log\hat\tau_i,\qquad \hat\tau_i=\max_k r_{ik},
    $$

    the log posterior probability of the cluster each point is assigned to; with `mode='soft'` it is the full responsibility entropy $-\sum_{i,k}r_{ik}\log r_{ik}$. $\mathrm{EN}=0$ exactly when every responsibility is one-hot, so ICL then equals BIC; the more the components overlap, the larger EN and the further ICL rises above BIC.

    **Parameter count.** $p=(K-1)+Kd+p_\Sigma$: the weights (they sum to 1), the means, and the covariance parameters $p_\Sigma$, which depend on `covariance_type`:

    | `covariance_type` | $p_\Sigma$ |
    |---|---|
    | `full` | $K\,d(d+1)/2$ |
    | `diag` | $K\,d$ |
    | `spherical` | $K$ |
    | `tied_full` | $d(d+1)/2$ |
    | `tied_diag` | $d$ |
    | `tied_spherical` | $1$ |

    The 14 GPCM models have their own counts, listed in [02 -- Covariance Types](02_covariance_types.ipynb). A fitted noise component adds its weight $\pi_0$ to $p$, and parameters you freeze or fix (`fixed_weights`, `frozen_means`, `frozen_covariances`) are not counted.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A dataset with overlapping blobs

    Five true components in two dimensions: two pairs that overlap substantially (`(0, 0)`/`(2.2,
    0)` and `(6, 6)`/`(8.2, 6)`), plus one isolated component (`(0, 2.2)`). Each pair is close enough
    relative to its spread that a 2- or 3-component mixture can already explain most of the
    likelihood in that region -- the point of this dataset is to make BIC and ICL disagree about
    how many of the five "true" components are worth resolving separately.
    """)
    return


@app.cell
def _(CATEGORICAL_PALETTE, SEED, generate_gmm_data, np, plt):
    centers = [
        np.array([0.0, 0.0]),
        np.array([2.2, 0.0]),
        np.array([0.0, 2.2]),
        np.array([6.0, 6.0]),
        np.array([8.2, 6.0]),
    ]
    covs = [np.eye(2) for _ in centers]
    n_samples = [150] * len(centers)

    X, true_labels = generate_gmm_data(centers, covs, n_samples, random_state=SEED)

    plt.figure(figsize=(5, 5))
    for _k in range(len(centers)):
        _mask = true_labels == _k
        plt.scatter(X[_mask, 0], X[_mask, 1], s=10, color=CATEGORICAL_PALETTE[_k % len(CATEGORICAL_PALETTE)],
                    label=f'true component {_k}')
    plt.legend(fontsize=8)
    plt.title('Overlapping blobs (true generating components)')
    plt.gca().set_aspect('equal')
    plt.gcf()
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sweep `n_components` with `select_model`

    `select_model` fits a `GaussianMixture` for every requested `n_components`, recording
    `bic`, `aic`, `icl`, the final log-likelihood, and whether each fit converged.
    """)
    return


@app.cell
def _(FAST, MAX_ITER, X, select_model):
    result = select_model(
        X,
        n_components=range(1, 9),
        covariance_types=('full',),
        criterion='bic',
        n_init=1 if FAST else 3,
        random_state=0,
        max_iter=min(200, MAX_ITER),
    )

    for row in result.table:
        print(f"K={row['n_components']}  bic={row['bic']:.1f}  aic={row['aic']:.1f}  "
              f"icl={row['icl']:.1f}  icl-bic={row['icl'] - row['bic']:.1f}  converged={row['converged']}")
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BIC vs. ICL curves

    Both criteria are "lower is better." Watch where each curve's minimum sits.
    """)
    return


@app.cell
def _(np, plt, result):
    ks = [row['n_components'] for row in result.table]
    bics = [row['bic'] for row in result.table]
    icls = [row['icl'] for row in result.table]

    best_bic_k = ks[int(np.argmin(bics))]
    best_icl_k = ks[int(np.argmin(icls))]

    plt.figure(figsize=(6, 4))
    plt.plot(ks, bics, marker='o', label='BIC')
    plt.plot(ks, icls, marker='s', label='ICL')
    plt.axvline(best_bic_k, color='C0', linestyle='--', alpha=0.5)
    plt.axvline(best_icl_k, color='C1', linestyle='--', alpha=0.5)
    plt.xlabel('n_components (K)')
    plt.ylabel('score (lower is better)')
    plt.title('BIC vs ICL over K on overlapping blobs')
    plt.legend()
    print(f'BIC-optimal K = {best_bic_k}')
    print(f'ICL-optimal K = {best_icl_k}')
    plt.gcf()
    return (best_bic_k,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As expected on this overlapping dataset, **ICL settles on fewer components than BIC**: BIC
    keeps adding components as long as they buy a bit more likelihood, while ICL's entropy penalty
    charges for the ambiguity introduced by splitting an overlapping pair into two components whose
    responsibilities are never cleanly one-hot. Neither is "wrong" -- they answer different
    questions (best density estimate vs. best *well-separated partition*).

    `result.best_` is the fitted `GaussianMixture` that minimised the sweep's `criterion` (`'bic'`
    here); `result.to_dataframe()` gives the same table as a `pandas.DataFrame` for further analysis.
    """)
    return


@app.cell
def _(result):
    print(type(result.best_))
    print('best_ n_components:', result.best_.n_components)

    df = result.to_dataframe()
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `icl()` on a single fitted model

    `icl(X, mode='hard')` (the default) uses the most-likely component per point;
    `mode='soft'` uses the full responsibility-entropy penalty instead. Both agree exactly with
    `bic()` whenever responsibilities are one-hot (e.g. far-apart, well-separated components), and
    diverge more the more overlapping/ambiguous the fit is -- which is exactly what drives the gap
    in the curves above.
    """)
    return


@app.cell
def _(X, best_bic_k, result):
    gmm = result.models_[(best_bic_k, 'full')]
    print('bic:      ', gmm.bic(X))
    print('icl hard: ', gmm.icl(X, mode='hard'))
    print('icl soft: ', gmm.icl(X, mode='soft'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checking the formulas

    The parameter count and the entropy term are not exposed directly, so we recover them from the public scores. For a model with mean log-likelihood `gmm.score(X)` we have $p=(\mathrm{AIC}+2n\,\overline{\ell})/2$, which we compare with the table above, and we rebuild ICL from the responsibilities.
    """)
    return


@app.cell
def _(GaussianMixture, X, np, torch):
    _n, _d, _K = X.shape[0], X.shape[1], 3
    _formula = {'full': _K * _d * (_d + 1) / 2, 'diag': _K * _d, 'spherical': _K,
                'tied_full': _d * (_d + 1) / 2, 'tied_diag': _d, 'tied_spherical': 1}
    print(f"{'covariance_type':15s} {'p from AIC':>10s} {'p (table)':>10s}   BIC ok   ICL hard ok   ICL soft ok")
    for _t, _pc in _formula.items():
        _g = GaussianMixture(n_components=_K, covariance_type=_t, random_state=0).fit(X)
        _ll = _g.score(X)
        _p_aic = (_g.aic(X) + 2 * _n * _ll) / 2
        _p = _pc + _K * _d + _K - 1
        _r = _g.predict_proba(X).double()
        _en_hard = -torch.log(_r.max(1).values).sum().item()
        _en_soft = -torch.special.xlogy(_r, _r).sum().item()
        _bic_ok = np.isclose(_g.bic(X), -2 * _ll * _n + _p * np.log(_n), rtol=1e-4)
        _hard_ok = np.isclose(_g.icl(X), _g.bic(X) + 2 * _en_hard, rtol=1e-4)
        _soft_ok = np.isclose(_g.icl(X, mode='soft'), _g.bic(X) + 2 * _en_soft, rtol=1e-4)
        print(f'{_t:15s} {_p_aic:10.1f} {_p:10.1f}   {str(bool(_bic_ok)):6s}   {str(bool(_hard_ok)):11s}   {bool(_soft_ok)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sweeping the covariance structure too

    `select_model` takes the Cartesian product of `n_components` and `covariance_types`. By default it tries **all 14 GPCM codes** (`tgmm.covariance.gpcm.GPCM_CODES`), the mclust-style workflow; names such as `'full'` or `'tied_full'` and mclust codes such as `'VVV'` are both accepted. A combination whose fit raises is recorded with every score `None` and skipped, so one singular fit does not abort the sweep. `result.best_` is the winner by the sweep's `criterion`, `result.models_[(K, type)]` any fitted model, and `result.to_dataframe()` the table (columns `n_components`, `covariance_type`, `bic`, `aic`, `icl`, `log_likelihood`, `converged`).

    The full 14-code grid is the default when you run this notebook yourself. The docs build uses a four-code subset (spherical, diagonal, tied and full) to stay within its time budget.
    """)
    return


@app.cell
def _(FAST, GPCM_CODES, MAX_ITER, X, select_model):
    grid_codes = ('VII', 'VVI', 'EEE', 'VVV') if FAST else GPCM_CODES
    grid = select_model(X, n_components=range(1, 7), covariance_types=grid_codes, n_init=1,
                        random_state=0, max_iter=min(200, MAX_ITER))
    grid_df = grid.to_dataframe()
    print(f'{len(grid_codes)} covariance codes x 6 values of K = {len(grid_df)} fits, {int(grid_df.bic.isna().sum())} failed')
    for _c in ('bic', 'aic', 'icl'):
        _b = grid_df.loc[grid_df[_c].idxmin()]
        print(f'best by {_c}: K = {int(_b.n_components)}, covariance_type = {_b.covariance_type}  ({_c} = {_b[_c]:.1f})')
    return (grid_df,)


@app.cell
def _(grid_df):
    grid_df.pivot(index='covariance_type', columns='n_components', values='bic').round(0).astype(int)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The table is BIC for every pair. Moving down a column changes only the covariance structure at a fixed $K$: a cheaper structure wins when it loses little likelihood, an expensive one when the clusters really differ in shape. The best cell can lie at a $K$ that the `covariance_type='full'` sweep above would not choose, which is why sweeping both together is worthwhile.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A real dataset: iris petal measurements

    Petal length and width are the two iris features that most cleanly separate the three species -- small enough to sweep all 14 models for several $K$ quickly, and two-dimensional so that the winner can be plotted directly. This sweep always uses all 14 codes, also in the docs build, because the data are tiny.
    """)
    return


@app.cell
def _(load_iris, plt, torch):
    iris = load_iris()
    X_iris = torch.tensor(iris.data[:, [2, 3]], dtype=torch.float64)  # petal length, petal width

    plt.figure(figsize=(5, 4))
    plt.scatter(X_iris[:, 0], X_iris[:, 1], c=iris.target, cmap='viridis', s=15)
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.title('Iris petal measurements (true species)')
    plt.gcf()
    return X_iris, iris


@app.cell
def _(GPCM_CODES, SEED, X_iris, select_model):
    iris_result = select_model(
        X_iris, n_components=range(1, 6), covariance_types=GPCM_CODES, criterion='bic',
        random_state=SEED, n_init=1, max_iter=200,
    )

    iris_df = iris_result.to_dataframe().sort_values('bic')
    iris_df.head(10)
    return iris_df, iris_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One line per covariance code, BIC against $K$ -- lower is better.
    """)
    return


@app.cell
def _(GPCM_CODES, iris_df, plt):
    plt.figure(figsize=(9, 5))
    for _code_name in GPCM_CODES:
        _sub = iris_df[iris_df['covariance_type'] == _code_name].sort_values('n_components')
        _sub = _sub.dropna(subset=['bic'])
        if len(_sub) == 0:
            continue
        plt.plot(_sub['n_components'], _sub['bic'], marker='o', markersize=3, label=_code_name, alpha=0.7)
    plt.xlabel('n_components (K)')
    plt.ylabel('BIC (lower is better)')
    plt.title('BIC across all 14 GPCM models, iris petal data')
    plt.legend(ncol=2, fontsize=8)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ellipses show what the selected structure looks like on the data. When the winner is one of the 8 non-classic codes it carries a decomposition, so `volume_`, `shape_` and `orientation_` say *what kind* of parsimony the data supported -- shared or per-component size, elongation and rotation -- rather than merely that the model won.
    """)
    return


@app.cell
def _(X_iris, iris, iris_result, plot_gmm, plt):
    iris_best = iris_result.best_
    print(f'Winning model: {iris_best.covariance_type_} (K={iris_best.n_components}), BIC={iris_best.bic(X_iris):.2f}')
    plt.figure(figsize=(6, 5))
    plot_gmm(X_iris.cpu().numpy(), gmm=iris_best, true_labels=iris.target, color_by_cluster=True)
    plt.title(f'{iris_best.covariance_type_} (K={iris_best.n_components}) fit to iris petal measurements')
    plt.gcf()
    return (iris_best,)


@app.cell
def _(iris_best):
    print('covariance_type_:', iris_best.covariance_type_)
    for _name in ('volume_', 'shape_', 'orientation_'):
        _value = getattr(iris_best, _name)
        print(f'{_name}:', None if _value is None else _value.cpu().numpy().round(3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A 10-D latent-space-like dataset

    Four clusters in 10 dimensions with deliberately varied covariance structure (different volumes, some axis-aligned, some correlated and elongated) -- the situation a sweep over the 14 structures is actually *for*: no single covariance type is obviously right, and the full model spends $K\,d(d+1)/2=220$ covariance parameters at $K=4$ (see the parameter counts in [02 -- Covariance Types](02_covariance_types.ipynb)).
    """)
    return


@app.cell
def _(SEED, torch):
    def make_latent_cluster(n, d, seed, volume, elongation=None, rotate=False):
        _gen = torch.Generator().manual_seed(seed)
        _base = torch.randn(n, d, generator=_gen, dtype=torch.float64)
        _scale = torch.ones(d, dtype=torch.float64)
        if elongation is not None:
            _scale[: len(elongation)] = torch.tensor(elongation, dtype=torch.float64)
        _Z = _base * _scale.sqrt() * volume
        if rotate:
            _Q, _ = torch.linalg.qr(torch.randn(d, d, generator=_gen, dtype=torch.float64))
            _Z = _Z @ _Q.T
        return _Z


    torch.manual_seed(SEED)
    d_latent = 10
    _centres = torch.randn(4, d_latent, dtype=torch.float64) * 6
    _elong = [4, 4, 1, 1, 1, 1, 1, 1, 1, 1]
    X_latent = torch.cat([
        make_latent_cluster(150, d_latent, seed=1, volume=1.0) + _centres[0],
        make_latent_cluster(150, d_latent, seed=2, volume=1.8) + _centres[1],
        make_latent_cluster(150, d_latent, seed=3, volume=1.0, elongation=_elong) + _centres[2],
        make_latent_cluster(150, d_latent, seed=4, volume=1.0, elongation=_elong, rotate=True) + _centres[3],
    ], dim=0)
    print(X_latent.shape)
    return (X_latent,)


@app.cell
def _(GPCM_CODES, SEED, X_latent, select_model):
    latent_result = select_model(
        X_latent, n_components=range(2, 6), covariance_types=GPCM_CODES, criterion='bic',
        random_state=SEED, n_init=1, max_iter=100,
    )

    latent_df = latent_result.to_dataframe().sort_values('bic')
    latent_df.head(10)
    return latent_df, latent_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BIC rewards fit and penalises parameters; ICL additionally penalises overlap between clusters. Both criteria, compared across covariance structures at the BIC-winning $K$:
    """)
    return


@app.cell
def _(X_latent, latent_df, latent_result):
    latent_best = latent_result.best_
    latent_k = latent_best.n_components
    print(f'Winning model: {latent_best.covariance_type_} (K={latent_k}), '
          f'BIC={latent_best.bic(X_latent):.2f}, ICL={latent_best.icl(X_latent):.2f}')

    _at_k = latent_df[latent_df['n_components'] == latent_k].dropna(subset=['bic', 'icl']).sort_values('bic')
    _at_k[['covariance_type', 'bic', 'icl', 'converged']].head(8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How separation and noise move the choice

    Three Gaussian blobs of unit variance whose centres are spread by `separation` (the corners of a triangle with side `separation`). In the interactive version the controls below set it, and whether 30 uniform background points are added; the curves are BIC and ICL for $K=1,\dots,6$ with full covariances. When the blobs overlap, ICL's entropy penalty is large and it prefers fewer components than BIC; as they separate the two curves merge, because the responsibilities become one-hot. Background noise is not Gaussian, so the mixture tends to spend an extra component on it unless you model it explicitly (see [18 -- Noise, Missing Values and Conditioning](18_noise_missing_conditioning.ipynb), and [12 -- HDBSCAN](12_hdbscan.ipynb) for removing it first).
    """)
    return


@app.cell
def _(mo):
    # docs: hide-cell
    sep_slider = mo.ui.slider(1.0, 8.0, step=0.5, value=4.0, label='cluster separation', debounce=True, show_value=True)
    noise_switch = mo.ui.switch(label='add 30 noise points')
    mo.hstack([sep_slider, noise_switch], justify='start')
    return noise_switch, sep_slider


@app.cell
def _(np, torch):
    def three_blobs(separation, noise):
        _rng = np.random.default_rng(0)
        _centres = np.array([[0, 0], [1, 0], [0.5, 0.87]]) * separation
        _Z = np.vstack([_rng.normal(_c, 1.0, (80, 2)) for _c in _centres])
        if noise:
            _Z = np.vstack([_Z, _rng.uniform(_Z.min(0) - 3, _Z.max(0) + 3, (30, 2))])
        return torch.tensor(_Z, dtype=torch.float32)
    return (three_blobs,)


@app.cell
def _(FAST, noise_switch, np, plt, sep_slider, select_model, three_blobs):
    _res = select_model(three_blobs(sep_slider.value, noise_switch.value), n_components=range(1, 7),
                        covariance_types=('full',), n_init=1 if FAST else 3, random_state=0)
    _ks = [r['n_components'] for r in _res.table]
    _bic = np.array([r['bic'] for r in _res.table])
    _icl = np.array([r['icl'] for r in _res.table])
    plt.figure(figsize=(6, 4))
    for _y, _lab, _m in ((_bic, 'BIC', 'o'), (_icl, 'ICL', 's')):
        plt.plot(_ks, _y, marker=_m, label=_lab)
        plt.plot(_ks[int(_y.argmin())], _y.min(), 'k*', ms=14)
    plt.xlabel('n_components (K)')
    plt.ylabel('criterion (lower is better)')
    plt.title(f'separation {sep_slider.value}: BIC picks K = {_ks[int(_bic.argmin())]}, ICL picks K = {_ks[int(_icl.argmin())]}')
    plt.legend()
    plt.gcf()
    return


@app.cell
def _(FAST, np, select_model, three_blobs):
    print(f"{'separation':>10s} {'noise':>6s} {'BIC K':>6s} {'ICL K':>6s} {'ICL-BIC at K=3':>15s}")
    for _sep in (1, 2, 3, 4, 6):
        for _noise in (False, True):
            _r = select_model(three_blobs(_sep, _noise), n_components=range(1, 7), covariance_types=('full',),
                              n_init=1 if FAST else 3, random_state=0)
            _k = [t['n_components'] for t in _r.table]
            _at3 = next(t for t in _r.table if t['n_components'] == 3)
            print(f"{_sep:10d} {str(_noise):>6s} {_k[int(np.argmin([t['bic'] for t in _r.table]))]:6d} "
                  f"{_k[int(np.argmin([t['icl'] for t in _r.table]))]:6d} {_at3['icl'] - _at3['bic']:15.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The last column is $2\,\mathrm{EN}$ at the true $K=3$: it is large when the blobs overlap and shrinks toward $0$ as they separate. Comparing each pair of rows, BIC picks at least one more component when the 30 noise points are present, because a Gaussian has to be spent on the background.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - `select_model` sweeps `n_components` (and optionally covariance types) and returns the best fit by the chosen criterion, together with the full table of `bic`, `aic`, `icl`, log-likelihood and convergence for every candidate.
    - BIC and AIC measure fit versus complexity; ICL adds an entropy penalty, so on overlapping clusters it favors fewer, better-separated components.
    - Neither criterion is wrong -- they answer different questions: the best density estimate versus the best-separated partition.
    - `icl(X, mode='hard' | 'soft')` can be computed on any fitted model; it equals BIC when responsibilities are one-hot.
    - The formulas $\mathrm{AIC}=-2\hat\ell+2p$, $\mathrm{BIC}=-2\hat\ell+p\ln n$ and $\mathrm{ICL}=\mathrm{BIC}+2\,\mathrm{EN}$ match the library to numerical precision, with $p=(K-1)+Kd+p_\Sigma$.
    - `select_model` sweeps $K$ and the covariance structure together (all 14 GPCM codes by default); on the iris petals BIC picks `VVE` with $K=3$ (per-component volume and shape, one shared orientation) and on the 10-D dataset `VVI` with $K=4$ (per-component volume and shape, axis-aligned) -- neither is the default `full` model -- and `volume_` / `shape_` / `orientation_` say which kind of parsimony a non-classic winner uses. Overlap widens the BIC-ICL gap and background noise adds a spurious component.

    **Next:** [14 -- Stochastic EM, Annealing and SAEM](14_stochastic_em.ipynb) for fitting algorithms that escape poor local optima, and [02 -- Covariance Types](02_covariance_types.ipynb) for what each of the 14 covariance models assumes.
    """)
    return


if __name__ == "__main__":
    app.run()
