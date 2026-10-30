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
    return (FAST,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Covariance Structures: From Full Matrices to the 14 GPCM Models

    Every component of a Gaussian mixture owns a covariance matrix $\Sigma_k$, and it is by far the
    most expensive and most consequential part of the model: a full covariance in $d$ dimensions has
    $d(d+1)/2$ free parameters per component, so a mixture can easily spend more parameters describing
    the *shapes* of its clusters than their locations. `covariance_type` controls how much freedom each
    $\Sigma_k$ gets, and with it the trade-off between flexibility (can the model follow
    elongated, rotated clusters?) and parsimony (can it be estimated reliably from limited data?).

    `tgmm` offers two layers of choices. The six **classic** types (`'full'`, `'diag'`, `'spherical'`
    and their `tied_` versions) are the ones scikit-learn offers. On top of them, `GaussianMixture`
    supports all 14 **Gaussian Parsimonious Clustering Models** (GPCM) of Celeux & Govaert (1995) --
    the family `mclust` implements -- which write each covariance as a product of a *volume*, a *shape*
    and an *orientation* and let you share or free each factor independently.

    This notebook shows what each choice looks like, derives **how EM estimates every one of them**
    (the M-step), checks the derivations numerically against the library, and compares all 14 models
    on the four-cluster data.

    **What you'll learn**

    - how the covariance enters EM's M-step, and why every covariance type is the same optimisation
      problem over a different feasible set;
    - the closed-form estimates for the six classic types, and their number of free parameters;
    - the volume / shape / orientation decomposition behind the 14 GPCM models and its mclust naming;
    - how each of the 14 models is estimated: closed forms, alternating updates, and the one step
      that needs a numerical (Stiefel-manifold) optimisation;
    - how to read `volume_`, `shape_`, `orientation_` on a fitted model, and how the number of free
      parameters of each model grows with the number of components $K$ and the dimension $d$.

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb), which builds the four-cluster data and
    explains the mixture model and EM (the E-step / M-step vocabulary used throughout).

    **See also:** [03 -- Initialisation](03_initialization.ipynb) for how the starting covariances are
    chosen, and [`tgmm.covariance.gpcm`](../api/covariance-gpcm.md) for the API of the M-step functions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    One imports-and-seeds cell, then the dataset.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt

    from tgmm import GaussianMixture
    from tutorial_data import make_default_data
    from tgmm.covariance.gpcm import GPCM_CODES, mstep_gpcm
    from tgmm.plotting import plot_gmm, dynamic_figsize

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plt.rcParams['figure.dpi'] = 75  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f'Using device: {device}')
    return (
        GPCM_CODES,
        GaussianMixture,
        SEED,
        device,
        dynamic_figsize,
        make_default_data,
        mstep_gpcm,
        np,
        pd,
        plot_gmm,
        plt,
        torch,
    )


@app.cell
def _(SEED, device, make_default_data):
    # The default four blobs of the Introduction (spherical, small spherical, diagonal, tilted/full), from tutorial_data.py
    X, y_true = make_default_data(SEED, device=device)
    X = X.double()  # float64: the numerical checks below compare against closed forms to ~1e-6
    n, d = X.shape
    K = 4
    legend_labels = [f'Component {k + 1}' for k in range(K)]
    return K, X, d, legend_labels, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The six classic covariance types

    Each type constrains the matrices $\Sigma_1,\dots,\Sigma_K$ differently. A covariance is *tied* if
    all components share it, *spherical* if it is a multiple of the identity, *diagonal* if it is
    axis-aligned, and *full* otherwise:

    | `covariance_type` | constraint | free covariance parameters |
    | --- | --- | --- |
    | `'tied_spherical'` | $\Sigma_k = \sigma^2 I$ for all $k$ | $1$ |
    | `'spherical'` | $\Sigma_k = \sigma_k^2 I$ | $K$ |
    | `'tied_diag'` | $\Sigma_k = \mathrm{diag}(\sigma_1^2,\dots,\sigma_d^2)$ for all $k$ | $d$ |
    | `'diag'` | $\Sigma_k = \mathrm{diag}(\sigma_{k1}^2,\dots,\sigma_{kd}^2)$ | $Kd$ |
    | `'tied_full'` | $\Sigma_k = \Sigma$ for all $k$, any symmetric positive-definite matrix | $d(d+1)/2$ |
    | `'full'` | each $\Sigma_k$ any symmetric positive-definite matrix | $K\,d(d+1)/2$ |

    The grid below fits each one to the default four blobs of [1. Introduction](01_introduction.ipynb). The ellipses are the fitted components' 1-,
    2- and 3-standard-deviation contours: spherical types can only draw circles, diagonal types only
    axis-aligned ellipses, tied types give every component the *same* ellipse, and only `'full'` can
    follow the tilted cluster *and* give each cluster its own size.
    """)
    return


@app.cell
def _(
    GaussianMixture,
    K,
    SEED,
    X,
    device,
    dynamic_figsize,
    legend_labels,
    plot_gmm,
    plt,
):
    classic_types = ['tied_spherical', 'spherical', 'tied_diag', 'diag', 'tied_full', 'full']
    _fig, _axs = plt.subplots(3, 2, figsize=dynamic_figsize(3, 2), dpi=75)
    for _ax, _cov_type in zip(_axs.ravel(), classic_types):
        _gmm = GaussianMixture(n_components=K, covariance_type=_cov_type, init_means='kmeans', random_state=SEED, device=device)
        _gmm.fit(X)
        plot_gmm(X.cpu().numpy(), gmm=_gmm, ax=_ax, legend_labels=legend_labels, title=f"covariance_type='{_cov_type}'   (mean log-lik = {float(_gmm.score(X)):.3f})")
    plt.suptitle('The six classic covariance types')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.gcf()
    return (classic_types,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Per-component types (`'full'`, `'diag'`, `'spherical'`) let each cluster have its own size, shape
    and orientation to the degree the type allows. The `tied_*` types give every cluster the same
    covariance (only the centres differ), which saves parameters and stabilises fits when there are few
    points per cluster. Selecting one is a single argument:

    ```python
    GaussianMixture(n_components=4, covariance_type='full')            # arbitrary ellipsoids
    GaussianMixture(n_components=4, covariance_type='diag')            # axis-aligned ellipsoids
    GaussianMixture(n_components=4, covariance_type='spherical')       # one variance per cluster
    GaussianMixture(n_components=4, covariance_type='tied_full')       # one shared full covariance
    GaussianMixture(n_components=4, covariance_type='tied_diag')       # one shared diagonal covariance
    GaussianMixture(n_components=4, covariance_type='tied_spherical')  # one shared variance
    ```

    `'tied'` is accepted as an alias of `'tied_full'` and `'isotropic'` of `'spherical'`. The total
    number of free parameters of a fitted model is
    $p=(K-1)+Kd+p_\Sigma$: the $K-1$ free weights, the $Kd$ mean coordinates and the covariance count
    $p_\Sigma$ from the last column of the table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How EM estimates a covariance: the M-step

    Given the responsibilities $r_{ik}$ from the E-step (see the [introduction](01_introduction.ipynb)),
    the M-step maximises the expected complete-data log-likelihood

    $$
    Q(\theta)=\sum_{i=1}^{n}\sum_{k=1}^{K} r_{ik}\Big[\log\pi_k-\tfrac d2\log 2\pi-\tfrac12\log|\Sigma_k|
    -\tfrac12 (x_i-\mu_k)^{\top}\Sigma_k^{-1}(x_i-\mu_k)\Big].
    $$

    Write $n_k=\sum_i r_{ik}$ for the effective number of points in component $k$. Maximising over
    $\mu_k$ gives the responsibility-weighted mean $\bar y_k=\frac1{n_k}\sum_i r_{ik}x_i$. Since

    $$
    \sum_i r_{ik}(x_i-\mu_k)^{\top}\Sigma_k^{-1}(x_i-\mu_k)
    =\mathrm{tr}\Big(\Sigma_k^{-1}\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^{\top}\Big),
    $$

    and at $\mu_k=\bar y_k$ the inner sum is the **weighted scatter matrix**

    $$
    W_k=\sum_{i=1}^{n} r_{ik}\,(x_i-\bar y_k)(x_i-\bar y_k)^{\top},
    $$

    the only part of $Q$ that depends on the covariances is, up to a constant factor $-\tfrac12$,

    $$
    \boxed{\;J(\Sigma_1,\dots,\Sigma_K)=\sum_{k=1}^{K}\Big[\,n_k\log|\Sigma_k|+\mathrm{tr}\big(\Sigma_k^{-1}W_k\big)\Big]\;}
    \qquad\text{(to be minimised).}
    $$

    This is the key fact of the whole notebook: **every covariance type is the same objective $J$
    minimised over a different feasible set.** The M-step needs the data only through $(n_k, W_k)$ --
    these are exactly the sufficient statistics `tgmm` accumulates (`SufficientStats.nk` and `.s2`) --
    and the covariance types differ only in which matrices $\Sigma_k$ they allow. (`reg_covar` is
    added to the diagonal of the result afterwards, to keep matrices positive definite; the formulas
    below are the unregularised solutions.)

    ### The six classic closed forms

    Substituting $\Omega_k=\Sigma_k^{-1}$, the full-covariance term is
    $J_k=-n_k\log|\Omega_k|+\mathrm{tr}(\Omega_kW_k)$, whose gradient $-n_k\Omega_k^{-1}+W_k$ vanishes at
    $\Sigma_k=W_k/n_k$. The constrained types follow by restricting the feasible set:

    - **Diagonal**, $\Sigma_k=\mathrm{diag}(\sigma_{kj}^2)$: $J$ separates over coordinates,
      $J_k=\sum_j\big[n_k\log\sigma_{kj}^2+(W_k)_{jj}/\sigma_{kj}^2\big]$, minimised at
      $\sigma_{kj}^2=(W_k)_{jj}/n_k$.
    - **Spherical**, $\Sigma_k=\sigma_k^2I$: $J_k=n_kd\log\sigma_k^2+\mathrm{tr}(W_k)/\sigma_k^2$, minimised at
      $\sigma_k^2=\mathrm{tr}(W_k)/(d\,n_k)$.
    - **Tied** versions: the same $\Sigma$ appears in every term, so $n_k$ and $W_k$ are simply pooled
      across components -- $n=\sum_kn_k$ and $\sum_kW_k$ take their place.

    | type | estimate |
    | --- | --- |
    | `full` | $\Sigma_k=W_k/n_k$ |
    | `diag` | $\Sigma_k=\mathrm{diag}(W_k)/n_k$ |
    | `spherical` | $\Sigma_k=\dfrac{\mathrm{tr}(W_k)}{d\,n_k}\,I$ |
    | `tied_full` | $\Sigma=\dfrac1n\sum_kW_k$ |
    | `tied_diag` | $\Sigma=\mathrm{diag}\Big(\sum_kW_k\Big)\big/n$ |
    | `tied_spherical` | $\Sigma=\dfrac{\mathrm{tr}\big(\sum_kW_k\big)}{d\,n}\,I$ |

    At a converged EM fit the stored covariances must be a fixed point of these formulas applied to the
    final responsibilities. The cell below checks that: it recomputes $n_k$, $\bar y_k$ and $W_k$ from
    `predict_proba` and compares the closed forms with `covariances_`.
    """)
    return


@app.cell
def _(GaussianMixture, K, SEED, X, classic_types, device, pd, torch):
    def e_step_statistics(gmm, X):
        """n_k, ybar_k and the scatter matrices W_k implied by the model's responsibilities."""
        r = gmm.predict_proba(X).detach()  # (n, K)   r_ik
        nk = r.sum(dim=0)  # (K,)     n_k
        ybar = r.T @ X / nk[:, None]  # (K, d)   weighted means
        diff = X[None, :, :] - ybar[:, None, :]  # (K, n, d)
        W = torch.einsum('ik,kid,kie->kde', r, diff, diff)  # (K, d, d) scatter matrices
        return (nk, ybar, W)

    def classic_mle(cov_type, nk, W):
        """The closed-form covariance estimates from the table above."""
        n, d = (nk.sum(), W.shape[-1])
        diag_W = torch.diagonal(W, dim1=-2, dim2=-1)
        return {'full': lambda: W / nk[:, None, None], 'diag': lambda: diag_W / nk[:, None], 'spherical': lambda: diag_W.sum(-1) / (nk * d), 'tied_full': lambda: W.sum(0) / n, 'tied_diag': lambda: diag_W.sum(0) / n, 'tied_spherical': lambda: diag_W.sum() / (n * d)}[cov_type]()  # (K, d)
    _rows = []
    for _cov_type in classic_types:
        _gmm = GaussianMixture(n_components=K, covariance_type=_cov_type, init_means='kmeans', random_state=SEED, device=device, max_iter=2000, tol=1e-12, reg_covar=0.0)
        _gmm.fit(X)
        nk, ybar, W = e_step_statistics(_gmm, X)
        gap = (_gmm.covariances_ - classic_mle(_cov_type, nk, W)).abs().max().item()
        _rows.append({'covariance_type': _cov_type, 'EM iterations': _gmm.n_iter_, 'max |covariances_ - closed form|': gap})
    pd.DataFrame(_rows)  # reg_covar=0 and a tight tolerance so the fixed-point property holds to numerical precision
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The gaps are below $2\times10^{-6}$ for all six types (EM stops at a finite tolerance, so the fixed point
    is reached only approximately): the library's M-step *is* the table above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Volume, shape and orientation: the GPCM family

    The six classic types are a rather coarse menu: nothing sits between "every component has its own
    unconstrained matrix" and "everything is tied or axis-aligned". Banfield & Raftery (1993) and
    Celeux & Govaert (1995) proposed to take apart each covariance through its eigendecomposition

    $$
    \boxed{\;\Sigma_k=\lambda_k\,D_k\,A_k\,D_k^{\top}\;},\qquad
    \det A_k=1,\quad D_k^{\top}D_k=I,\quad A_k=\mathrm{diag}(a_{k1}\ge\dots\ge a_{kd}>0),
    $$

    into three geometrically meaningful pieces:

    - **volume** $\lambda_k=|\Sigma_k|^{1/d}>0$ -- how large the cluster is (its ellipsoid's volume
      scales as $\lambda_k^{d/2}$);
    - **shape** $A_k$ -- how elongated it is: the relative lengths of its axes (only ratios matter,
      because $\det A_k=1$ moves all the scale into $\lambda_k$);
    - **orientation** $D_k$ -- which way the axes point (the columns of $D_k$ are the principal
      directions).

    The semi-axes of the cluster's one-standard-deviation ellipse are $\sqrt{\lambda_k a_{kj}}$ along
    the $j$-th column of $D_k$. The figure varies one factor at a time.
    """)
    return


@app.cell
def _(np, plt):
    def ellipse(volume, shape_diag, angle_deg, n_points=200):
        """Unit-Mahalanobis contour of Sigma = volume * D diag(shape) D^T, D = rotation by `angle`."""
        theta = np.deg2rad(angle_deg)
        D = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = np.linspace(0, 2 * np.pi, n_points)
        circle = np.stack([np.cos(t), np.sin(t)])
        return D @ np.diag(np.sqrt(volume * np.asarray(shape_diag))) @ circle

    def unit_det(a, b):
        """A = diag(a, b) rescaled to determinant 1."""
        s = np.sqrt(a * b)
        return (a / s, b / s)
    base = dict(volume=1.0, shape=unit_det(4.0, 1.0), angle=30)
    _fig, _axs = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    panels = [('volume $\\lambda$ varies', [dict(base, volume=v) for v in (0.25, 1.0, 2.5)], lambda c: f"$\\lambda$={c['volume']}"), ('shape $A$ varies', [dict(base, shape=unit_det(a, 1.0)) for a in (1.0, 4.0, 12.0)], lambda c: f"$a_1/a_2$={c['shape'][0] / c['shape'][1]:.0f}"), ('orientation $D$ varies', [dict(base, angle=a) for a in (0, 30, 90)], lambda c: f"angle={c['angle']}$^\\circ$")]
    for _ax, (title, configs, label) in zip(_axs, panels):
        for cfg in configs:
            pts = ellipse(cfg['volume'], cfg['shape'], cfg['angle'])
            _ax.plot(pts[0], pts[1], label=label(cfg))
        _ax.set_title(title)
        _ax.set_aspect('equal')
        _ax.axhline(0, color='0.85', lw=0.8)
        _ax.axvline(0, color='0.85', lw=0.8)
        _ax.legend(loc='upper right', fontsize=8)
    plt.suptitle('Changing one factor of $\\Sigma=\\lambda D A D^{\\top}$ at a time')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The 14 models and their names

    Each factor can be **E**qual across components, **V**ary from component to component, or -- for
    shape and orientation only -- be the **I**dentity ($A_k=I$: spherical; $D_k=I$: axis-aligned, no
    rotation). A model's mclust code reads *Volume–Shape–Orientation*, e.g. `EVE` = equal volume,
    varying shape, equal orientation. Of the $2\times3\times3$ letter combinations, 14 are valid:

    | code | volume | shape | orientation | `tgmm` alias | free covariance parameters |
    | --- | --- | --- | --- | --- | --- |
    | `EII` | equal | identity | identity | `'tied_spherical'` | $1$ |
    | `VII` | varying | identity | identity | `'spherical'` | $K$ |
    | `EEI` | equal | equal | identity | `'tied_diag'` | $d$ |
    | `VEI` | varying | equal | identity | -- | $K+(d-1)$ |
    | `EVI` | equal | varying | identity | -- | $1+K(d-1)$ |
    | `VVI` | varying | varying | identity | `'diag'` | $Kd$ |
    | `EEE` | equal | equal | equal | `'tied_full'` | $d(d+1)/2$ |
    | `VEE` | varying | equal | equal | -- | $K+(d-1)+d(d-1)/2$ |
    | `EVE` | equal | varying | equal | -- | $1+K(d-1)+d(d-1)/2$ |
    | `VVE` | varying | varying | equal | -- | $Kd+d(d-1)/2$ |
    | `EEV` | equal | equal | varying | -- | $d+K\,d(d-1)/2$ |
    | `VEV` | varying | equal | varying | -- | $K+(d-1)+K\,d(d-1)/2$ |
    | `EVV` | equal | varying | varying | -- | $1+K(d-1)+K\,d(d-1)/2$ |
    | `VVV` | varying | varying | varying | `'full'` | $K\,d(d+1)/2$ |

    The parameter count is simply the sum of three independent contributions:

    | factor | `E` | `V` | `I` |
    | --- | --- | --- | --- |
    | volume | $1$ | $K$ | -- |
    | shape (free ratios of $A$) | $d-1$ | $K(d-1)$ | $0$ |
    | orientation (free angles of $D$) | $d(d-1)/2$ | $K\,d(d-1)/2$ | $0$ |

    Because the count is a sum over the three factors, it is easy to see how each model scales. Volume
    costs $1$ or $K$ parameters, shape $d-1$ or $K(d-1)$, orientation $d(d-1)/2$ or $K\,d(d-1)/2$. Only
    a *per-component orientation* (`EEV`, `VEV`, `EVV`, `VVV`) makes the count grow like $K\,d^2$. A
    *shared* orientation (`EEE`, `VEE`, `EVE`, `VVE`) adds a $d^2$ term that does not depend on $K$ (the
    volume and shape terms still grow with $K$, at most like $K\,d$); with the identity orientation the
    count grows at most like $K\,d$. The cells below first check the formula against fitted models, then
    compare all 14 models for several numbers of components $K$ and dimensions $d$. The last two columns of
    the table are the measured growth exponents, $a=\log_2$ of the ratio of the counts at $K=16$ and $K=8$
    (with $d=20$) and $b=\log_2$ of the ratio at $d=40$ and $d=20$ (with $K=8$): $a\approx1$ means the count
    doubles with $K$, $b\approx2$ that it quadruples when $d$ doubles.
    """)
    return


@app.cell
def _(GPCM_CODES, GaussianMixture, torch):
    ALIASES = {'EII': 'tied_spherical', 'VII': 'spherical', 'EEI': 'tied_diag', 'VVI': 'diag', 'EEE': 'tied_full', 'VVV': 'full'}

    def n_cov_params(code, K, d):
        """Free covariance parameters of a GPCM model: volume + shape + orientation."""
        volume = {'E': 1, 'V': K}[code[0]]
        shape = {'I': 0, 'E': d - 1, 'V': K * (d - 1)}[code[1]]
        orientation = {'I': 0, 'E': d * (d - 1) // 2, 'V': K * d * (d - 1) // 2}[code[2]]
        return volume + shape + orientation

    def count_from_fit(model, K, d):
        """Free covariance parameters read off a fitted model's stored factors, without using the formula above."""
        if model.volume_ is None:  # the six classic types store covariances_ directly
            _t = model.covariance_type_
            _full = {'EEE': d * (d + 1) // 2, 'VVV': K * d * (d + 1) // 2}
            return _full.get(_t, model.covariances_.numel())  # scalar / (K,) / (d,) / (K, d): every entry is free
        _n_shape_vectors = model.shape_.shape[0] if model.shape_.dim() == 2 else 1
        _n_orient = model.orientation_.shape[0] if model.orientation_.dim() == 3 else 1
        _is_identity = torch.allclose(model.orientation_.reshape(-1, d, d)[0], torch.eye(d, dtype=model.orientation_.dtype))
        return (model.volume_.numel()                                   # one scale per volume entry
                + model.shape_.numel() - _n_shape_vectors               # each shape vector has unit determinant
                + (0 if _is_identity else _n_orient * d * (d - 1) // 2))  # each rotation has d(d-1)/2 angles

    _rows = []
    for _K, _d in ((3, 4), (2, 5)):
        _gen = torch.Generator().manual_seed(_K * 10 + _d)
        _X = torch.randn(400, _d, generator=_gen, dtype=torch.float64) * torch.linspace(0.5, 3, _d, dtype=torch.float64)
        for _code_name in GPCM_CODES:
            _m = GaussianMixture(n_components=_K, covariance_type=_code_name, random_state=0, max_iter=20).fit(_X)
            _rows.append((_K, _d, _code_name, n_cov_params(_code_name, _K, _d), count_from_fit(_m, _K, _d)))
    _bad = [r for r in _rows if r[3] != r[4]]
    print(f'checked {len(_rows)} fits (14 models x (K, d) in {{(3, 4), (2, 5)}}): formula == count read from the fitted model for all: {not _bad}')
    assert not _bad, _bad
    return ALIASES, n_cov_params


@app.cell
def _(ALIASES, GPCM_CODES, n_cov_params, np, pd):
    Ks, ds = (2, 4, 8), (2, 5, 10, 20)
    _table = pd.DataFrame(
        [[n_cov_params(_c, _K, _d) for _K in Ks for _d in ds] for _c in GPCM_CODES],
        index=pd.Index([f"{_c}  ({ALIASES[_c]})" if _c in ALIASES else _c for _c in GPCM_CODES], name='model'),
        columns=pd.MultiIndex.from_tuples([(f'K={_K}', f'd={_d}') for _K in Ks for _d in ds]),
    )
    # growth exponents: count ~ K^a d^b. Measured between K=8 -> 16 at d=20 and d=20 -> 40 at K=8.
    _a = [np.log2(n_cov_params(_c, 16, 20) / n_cov_params(_c, 8, 20)) for _c in GPCM_CODES]
    _b = [np.log2(n_cov_params(_c, 8, 40) / n_cov_params(_c, 8, 20)) for _c in GPCM_CODES]
    _table[('grows like', 'K^a, a ≈')] = np.round(_a, 1)
    _table[('grows like', 'd^b, b ≈')] = np.round(_b, 1)
    param_table = _table
    param_table
    return Ks, ds


@app.cell
def _(GPCM_CODES, Ks, ds, n_cov_params, plt):
    _fig, _axs = plt.subplots(1, 2, figsize=(11, 4.2), dpi=75)
    _dgrid = list(range(2, 41))
    _kgrid = [1, 2, 4, 8, 16, 32]
    for _c in GPCM_CODES:
        _col = {'E': '#2a78d6', 'V': '#eb6834'}[_c[0]]
        _ls = {'I': '-', 'E': '--', 'V': ':'}[_c[2]]  # orientation: identity solid, shared dashed, per component dotted
        _axs[0].plot(_dgrid, [n_cov_params(_c, 4, _d) for _d in _dgrid], color=_col, ls=_ls, label=_c)
        _axs[1].plot(_kgrid, [n_cov_params(_c, _k, 10) for _k in _kgrid], color=_col, ls=_ls, marker='o', ms=3)
    _axs[0].set_xscale('log'); _axs[0].set_yscale('log')
    _axs[1].set_xscale('log', base=2); _axs[1].set_yscale('log')
    _axs[0].set_xticks([2, 5, 10, 20, 40]); _axs[0].set_xticklabels(['2', '5', '10', '20', '40']); _axs[0].minorticks_off()
    _axs[1].set_xticks(_kgrid); _axs[1].set_xticklabels([str(_k) for _k in _kgrid]); _axs[1].minorticks_off()
    _axs[0].set_xlabel('dimension $d$ (K = 4)'); _axs[1].set_xlabel('number of components $K$ (d = 10)')
    for _a in _axs:
        _a.set_ylabel('free covariance parameters'); _a.grid(alpha=0.3)
    _axs[1].legend(*_axs[0].get_legend_handles_labels(), ncol=1, fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5),
                   title='orange: varying volume\nblue: equal volume\nsolid: identity orientation\ndashed: shared orientation\ndotted: per-component', title_fontsize=7)
    _fig.suptitle('Free covariance parameters of the 14 models')
    _fig.subplots_adjust(left=0.07, right=0.85, wspace=0.25, top=0.88)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it: build a model from its three factors

    Two components in two dimensions. For each factor choose whether the components share it (**E**:
    component 2 copies component 1), let it **V**ary (each component has its own slider) or, for shape
    and orientation, fix it to the **I**dentity (a circle, or no rotation). The three letters are the
    mclust code, and the cell below reports whether that combination is one of the 14 models and how
    many covariance parameters it has. Component 2's sliders are ignored whenever its factor is **E**,
    both shape sliders are ignored for shape **I**, and both angle sliders for orientation **I**.
    Interactive when you open the notebook (see the link at the top of the page); the page you are
    reading shows the default setting, `VEV`.
    """)
    return


@app.cell
def _(mo):
    gp_vol = mo.ui.dropdown(['E', 'V'], value='V', label='volume')
    gp_shape = mo.ui.dropdown(['E', 'V', 'I'], value='E', label='shape')
    gp_orient = mo.ui.dropdown(['E', 'V', 'I'], value='V', label='orientation')
    gp_lam1 = mo.ui.slider(0.2, 3.0, step=0.1, value=1.0, label='λ₁')
    gp_lam2 = mo.ui.slider(0.2, 3.0, step=0.1, value=2.0, label='λ₂')
    gp_ratio1 = mo.ui.slider(1.0, 12.0, step=0.5, value=4.0, label='a₁/a₂ (comp. 1)')
    gp_ratio2 = mo.ui.slider(1.0, 12.0, step=0.5, value=4.0, label='a₁/a₂ (comp. 2)')
    gp_ang1 = mo.ui.slider(0, 180, step=5, value=30, label='θ₁ (°)')
    gp_ang2 = mo.ui.slider(0, 180, step=5, value=120, label='θ₂ (°)')
    mo.vstack([
        mo.hstack([gp_vol, gp_shape, gp_orient], justify='start'),
        mo.hstack([gp_lam1, gp_ratio1, gp_ang1], justify='start'),
        mo.hstack([gp_lam2, gp_ratio2, gp_ang2], justify='start'),
    ])
    return (
        gp_ang1,
        gp_ang2,
        gp_lam1,
        gp_lam2,
        gp_orient,
        gp_ratio1,
        gp_ratio2,
        gp_shape,
        gp_vol,
    )


@app.cell
def _(
    GPCM_CODES,
    gp_ang1,
    gp_ang2,
    gp_lam1,
    gp_lam2,
    gp_orient,
    gp_ratio1,
    gp_ratio2,
    gp_shape,
    gp_vol,
    mo,
    n_cov_params,
    np,
):
    _v, _s, _o = gp_vol.value, gp_shape.value, gp_orient.value
    gp_code = _v + _s + _o
    _l1 = gp_lam1.value
    _l2 = _l1 if _v == 'E' else gp_lam2.value
    _r1 = 1.0 if _s == 'I' else gp_ratio1.value
    _r2 = 1.0 if _s == 'I' else (_r1 if _s == 'E' else gp_ratio2.value)
    _t1 = 0.0 if _o == 'I' else gp_ang1.value
    _t2 = 0.0 if _o == 'I' else (_t1 if _o == 'E' else gp_ang2.value)

    def _make_cov(lam, ratio, deg):
        # Sigma = lam * D diag(a) D^T with a = (sqrt(r), 1/sqrt(r)), so det A = 1 and a1/a2 = r
        _a = np.array([np.sqrt(ratio), 1 / np.sqrt(ratio)])
        _c, _sn = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        _D = np.array([[_c, -_sn], [_sn, _c]])
        return lam * _D @ np.diag(_a) @ _D.T

    gp_covs = [_make_cov(_l1, _r1, _t1), _make_cov(_l2, _r2, _t2)]
    _verdict = ('one of the 14 GPCM models' if gp_code in GPCM_CODES else
                '**not one of the 14 models**: with a spherical (identity) shape the orientation is meaningless, so use orientation I')
    mo.md(f"**mclust code `{gp_code}`** -- {_verdict}.\n\n"
          f"Free covariance parameters for $d=2$, $K=2$: **{n_cov_params(gp_code, 2, 2)}** "
          f"(volume + shape + orientation contributions; "
          f"$\\det\\Sigma_1={np.linalg.det(gp_covs[0]):.2f}$, $\\det\\Sigma_2={np.linalg.det(gp_covs[1]):.2f}$).")
    return gp_code, gp_covs


@app.cell
def _(gp_code, gp_covs, np, plt):
    _t = np.linspace(0, 2 * np.pi, 200)
    _circle = np.stack([np.cos(_t), np.sin(_t)])
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _S, _c, _x0 in zip(gp_covs, ['#2a78d6', '#eb6834'], [-3.0, 3.0]):
        _L = np.linalg.cholesky(_S)
        for _m in (1, 2):
            _ax.plot(_x0 + _m * (_L @ _circle)[0], _m * (_L @ _circle)[1], color=_c, lw=2 if _m == 1 else 1)
        _ax.plot(_x0, 0, 'o', color=_c)
    _ax.set_aspect('equal')
    _ax.set_xlim(-7, 7)
    _ax.set_ylim(-3.5, 3.5)
    _ax.set_title(f'1σ and 2σ contours of Σ₁ (blue) and Σ₂ (orange), model {gp_code}')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to try: `EII` and `VII` give circles (identity shape and orientation), `EEE` shares everything
    (the same ellipse twice), and `VVV` frees all three factors. The parameter count is why constrained models can be estimated
    reliably from little data: `VVV` needs $K\,d(d+1)/2$ covariance parameters, `EII` only one.

    ### Reading the decomposition off a fitted model

    Pass either the tgmm name or the mclust code to `covariance_type`; mclust codes are
    case-insensitive, and `covariance_type_` on the model always reports the canonical code. For the 8
    models without a classic alias, `GaussianMixture` exposes the fitted decomposition as
    `volume_`, `shape_` and `orientation_`; `covariances_` always holds the assembled
    $(K, d, d)$ matrices, so everything else (`predict`, `sample`, `plot_gmm`, `save`, ...) is
    oblivious to which model was used. (The six classic types do not compute a decomposition, so these
    attributes are `None` for them.) Below, `EVE` -- one shared volume and one shared orientation,
    but a separate shape per component -- is fitted to the data and its covariances are rebuilt from
    the three factors.
    """)
    return


@app.cell
def _(GaussianMixture, K, SEED, X, d, device, torch):
    gmm_eve = GaussianMixture(n_components=K, covariance_type='EVE', init_means='kmeans',
                              random_state=SEED, device=device)
    gmm_eve.fit(X)

    print('volume_      (shared, scalar):', gmm_eve.volume_.item())
    print('orientation_ (shared D):\n', gmm_eve.orientation_.cpu().numpy().round(3))
    print('shape_ (one row of a_1 >= a_2 per component, each with a_1 * a_2 = 1):')
    print(gmm_eve.shape_.cpu().numpy().round(3))

    # Sigma_k = lambda * D diag(a_k) D^T (+ reg_covar * I), exactly as the decomposition says
    D, lam = gmm_eve.orientation_, gmm_eve.volume_
    rebuilt = lam * D @ torch.diag_embed(gmm_eve.shape_) @ D.T
    rebuilt = rebuilt + gmm_eve.reg_covar * torch.eye(d, dtype=rebuilt.dtype, device=rebuilt.device)
    print('\nmax |rebuilt - covariances_| =', (rebuilt - gmm_eve.covariances_).abs().max().item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deriving the M-step for all 14 models

    With the decomposition, $|\Sigma_k|=\lambda_k^d$ (because $\det D_k=\pm1$, $\det A_k=1$) and
    $\Sigma_k^{-1}=\lambda_k^{-1}D_kA_k^{-1}D_k^{\top}$, so by the cyclic property of the trace

    $$
    J=\sum_{k=1}^{K}\Big[\,n_k\,d\,\log\lambda_k+\frac1{\lambda_k}\,\mathrm{tr}\big(A_k^{-1}D_k^{\top}W_kD_k\big)\Big].
    $$

    Every model fixes some of the factors to be shared or trivial; the M-step minimises $J$ over the
    rest. Two small lemmas do almost all the work.

    **Lemma 1 (shape).** Let $B$ be symmetric positive definite. Among positive-definite $C$ with
    $\det C=1$,
    $$\mathrm{tr}(C^{-1}B)\;\ge\;d\,(\det B)^{1/d},\qquad\text{with equality iff } C=B/(\det B)^{1/d}.$$
    *Proof.* The eigenvalues $\mu_1,\dots,\mu_d$ of $C^{-1/2}BC^{-1/2}$ have product
    $\det B/\det C=\det B$. By the AM–GM inequality
    $\mathrm{tr}(C^{-1}B)=\sum_j\mu_j\ge d(\prod_j\mu_j)^{1/d}=d(\det B)^{1/d}$, with equality iff all
    $\mu_j$ coincide, i.e. $B=\mu C$; $\det C=1$ then forces $\mu=(\det B)^{1/d}$. $\square$
    For diagonal $C$ and diagonal $B$ this reads $a_j\propto b_j$, normalised to unit product.

    **Lemma 2 (orientation).** Let $W$ be symmetric with eigenvalues $\omega_1\ge\dots\ge\omega_d$ and
    let $a_1\ge\dots\ge a_d>0$. Over orthogonal $D$,
    $$\mathrm{tr}\big(A^{-1}D^{\top}WD\big)\;\ge\;\sum_j\frac{\omega_j}{a_j},$$
    attained when the columns of $D$ are the eigenvectors of $W$, ordered by decreasing eigenvalue.
    *Proof sketch.* $\mathrm{tr}(A^{-1}D^{\top}WD)=\sum_j(D^{\top}WD)_{jj}/a_j$ and the diagonal of
    $D^{\top}WD$ is majorised by $(\omega_j)$ (Schur–Horn); a linear functional with increasing weights
    $1/a_j$ is smallest when the largest $\omega$ is paired with the largest $a$ (the smallest weight).
    $\square$ This is why `tgmm` keeps every shape sorted in decreasing order: component $k$'s first
    axis is its longest one, and orderings stay consistent when eigenvalues are summed across components.

    ### Group 1 -- closed forms (`EVI`, `EEV`, `EVV`, and the six classic ones)

    The six classic types were solved above (they are `EII`, `VII`, `EEI`, `VVI`, `EEE`, `VVV`). Three
    of the new models are also available in closed form:

    - **`EVI`** ($\Sigma_k=\lambda A_k$, axis-aligned). Only the diagonal $B_k=\mathrm{diag}(W_k)$ matters
      in $\mathrm{tr}(A_k^{-1}W_k)$. For any $\lambda$, Lemma 1 gives
      $A_k=B_k/|B_k|^{1/d}$ ($|B_k|=\prod_j(W_k)_{jj}$) and $\mathrm{tr}(A_k^{-1}B_k)=d|B_k|^{1/d}$, so
      $J=nd\log\lambda+\frac d\lambda\sum_k|B_k|^{1/d}$, minimised at
      $$\lambda=\frac1n\sum_k|B_k|^{1/d}.$$
    - **`EEV`** ($\Sigma_k=\lambda D_kAD_k^{\top}$). For any $A$, Lemma 2 sends each $D_k$ to the
      eigenvectors of $W_k$ (eigenvalues $\Omega_k$, decreasing), leaving
      $J=nd\log\lambda+\frac1\lambda\mathrm{tr}(A^{-1}\Omega)$ with $\Omega=\sum_k\Omega_k$. Lemma 1 then gives
      $$A=\Omega/|\Omega|^{1/d},\qquad\lambda=|\Omega|^{1/d}/n.$$
    - **`EVV`** ($\Sigma_k=\lambda C_k$ with $C_k=D_kA_kD_k^{\top}$ a free unit-determinant matrix).
      Lemma 1 with $B=W_k$ gives $C_k=W_k/|W_k|^{1/d}$ and, as for `EVI`,
      $$\lambda=\frac1n\sum_k|W_k|^{1/d}.$$

    ### Group 2 -- alternating closed-form updates (`VEI`, `VEE`, `VEV`)

    When the *volumes* vary but the *shape* is shared, the two coupled factors have no joint closed
    form -- but each has one *given the other*, so `tgmm` alternates:

    - *Given the shape*, $J$ separates over $k$ in $\lambda_k$:
      $\;\lambda_k=\mathrm{tr}\big(A^{-1}D_k^{\top}W_kD_k\big)/(n_kd)$.
    - *Given the volumes*, the shape minimises $\mathrm{tr}(A^{-1}B)$ with
      $B=\sum_kD_k^{\top}W_kD_k/\lambda_k$, which Lemma 1 solves: $A=B/|B|^{1/d}$.

    | model | what is shared / varies | $B$ in the shape step | shape |
    | --- | --- | --- | --- |
    | `VEI` ($\Sigma_k=\lambda_kA$) | $D_k=I$ | $\mathrm{diag}\big(\sum_kW_k/\lambda_k\big)$ | $A=B/\lvert B\rvert^{1/d}$ |
    | `VEE` ($\Sigma_k=\lambda_kC$, $C=DAD^{\top}$) | $D_k=D$, shape and orientation together as $C$ | $\sum_kW_k/\lambda_k$ | $C=B/\lvert B\rvert^{1/d}$ |
    | `VEV` ($\Sigma_k=\lambda_kD_kAD_k^{\top}$) | $D_k$ = eigenvectors of $W_k$ (Lemma 2, once and for all) | $\sum_k\Omega_k/\lambda_k$ | $A=B/\lvert B\rvert^{1/d}$ |

    Each sub-step is an exact minimiser, so $J$ never increases (coordinate descent), and the loop stops
    when its relative change drops below `gpcm_tol` or after `gpcm_max_iter` sweeps. Because the
    alternation is *warm-started* from the previous EM iteration's decomposition, it usually needs only
    a sweep or two to re-settle.

    ### Group 3 -- shared orientation, varying shape (`EVE`, `VVE`)

    Here the unknowns are a shared rotation $D$ plus per-component shapes, and (for `VVE`)
    per-component volumes. *Given $D$*, everything else is in closed form: with
    $B_k=\mathrm{diag}(D^{\top}W_kD)$,
    $$A_k=B_k/|B_k|^{1/d},\qquad
    \lambda=\frac1n\sum_k|B_k|^{1/d}\ \ (\texttt{EVE}),\qquad
    \lambda_k=\frac{|B_k|^{1/d}}{n_k}\ \ (\texttt{VVE}).$$
    *Given the shapes and volumes*, the best $D$ minimises
    $\sum_k\lambda_k^{-1}\mathrm{tr}(A_k^{-1}D^{\top}W_kD)$ over orthogonal matrices. For $K=1$ this is
    Lemma 2, but for several components with different shapes **there is no closed form** -- it is the
    classic *common principal components* problem (Flury, 1984). In fact, profiling out $\lambda_k$ and
    $A_k$ from `VVE` gives exactly Flury's criterion,
    $$J(D)=\sum_kn_k\log\big|\mathrm{diag}(D^{\top}S_kD)\big|+nd,\qquad S_k=W_k/n_k,$$
    which by Hadamard's inequality is never below the unconstrained optimum $\sum_kn_k\log|S_k|+nd$, with equality iff one
    $D$ diagonalises every $S_k$ -- i.e. `VVE` coincides with `VVV` exactly when the components truly
    share principal axes.

    `tgmm` therefore optimises $D$ numerically: it runs a few L-BFGS steps over the manifold of
    orthogonal matrices (PyTorch's `orthogonal` parametrisation), warm-started at the previous
    $D$, and **accepts the result only if it lowers the objective**. This makes the M-step a
    *generalised* EM step (Dempster, Laird & Rubin, 1977): $Q$ is improved rather than exactly
    maximised, which is all EM needs to keep the log-likelihood monotone.

    ### Summary of the M-steps

    | model | M-step |
    | --- | --- |
    | `EII` `VII` `EEI` `VVI` `EEE` `VVV` | closed form (table above) |
    | `EVI` `EEV` `EVV` | closed form (Group 1) |
    | `VEI` `VEE` `VEV` | alternate closed-form $\lambda_k$ and shape updates (Group 2) |
    | `EVE` `VVE` | closed-form $\lambda$/$A_k$ given $D$, plus a numerical orthogonal-manifold step for $D$ (Group 3) |

    The `mstep_gpcm` function in `tgmm.covariance.gpcm` implements all 14 through one interface: it takes
    $(n_k, W_k)$ and returns the decomposition and the assembled covariances. (`GaussianMixture` only
    calls it for the 8 new models; the six classic types keep their own dedicated code paths, which
    the check above confirmed agree with the closed forms.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checking the derivations numerically

    Derivations deserve a test. The cell below builds scatter matrices $W_k$ with deliberately varied
    volumes, shapes and orientations, calls `mstep_gpcm` for all 14 codes, and compares the objective
    $J$ it reaches with an **independent brute-force minimiser**: L-BFGS over an unconstrained
    parametrisation of each model's feasible set (log-volumes, log-shapes mean-centred so that
    $\det A=1$, and orthogonal matrices from the matrix exponential of a skew-symmetric one), restarted
    from several random points.
    """)
    return


@app.cell
def _(FAST, GPCM_CODES, mstep_gpcm, pd, torch):
    torch.manual_seed(0)
    Kt, dt = (3, 3)
    nk_test = torch.tensor([120.0, 80.0, 200.0], dtype=torch.float64)
    W_test = []
    for k in range(Kt):
        Q, _ = torch.linalg.qr(torch.randn(dt, dt, dtype=torch.float64))
        eig = torch.tensor([4.0, 1.0, 0.5], dtype=torch.float64) * (1 + k) ** 0.7 * (torch.rand(dt, dtype=torch.float64) + 0.5)
        W_test.append(nk_test[k] * (Q @ torch.diag(eig) @ Q.T))
    W_test = torch.stack(W_test)

    def objective_J(cov, nk=nk_test, W=W_test):
        """J = sum_k n_k log|Sigma_k| + tr(Sigma_k^{-1} W_k)"""
        return sum((nk[k] * torch.logdet(cov[k]) + torch.trace(torch.linalg.solve(cov[k], W[k])) for k in range(len(nk))))

    def assemble(code_name, params):
        """Sigma_k = lambda_k D_k diag(a_k) D_k^T from unconstrained parameters, respecting the code."""
        v, s, o = code_name
        lam = torch.exp(params['volume']).expand(Kt)
        if s == 'I':
            a = torch.ones(Kt, dt, dtype=torch.float64)
        else:
            u = params['shape'] - params['shape'].mean(-1, keepdim=True)
            a = torch.exp(u).expand(Kt, dt)
        if o == 'I':
            D = torch.eye(dt, dtype=torch.float64).expand(Kt, dt, dt)  # centred logs => det A = 1
        else:
            S = params['orientation'] - params['orientation'].transpose(-1, -2)
            D = torch.linalg.matrix_exp(S).expand(Kt, dt, dt)
        return lam.view(Kt, 1, 1) * D @ torch.diag_embed(a) @ D.transpose(-1, -2)
      # skew-symmetric
    def brute_force_minimum(code_name, restarts=2 if FAST else 6):  # orthogonal
        v, s, o = code_name
        best = float('inf')
        for _ in range(restarts):
            params = {'volume': torch.randn(Kt if v == 'V' else 1, dtype=torch.float64).requires_grad_(), 'shape': torch.randn(dt if s == 'E' else (Kt, dt), dtype=torch.float64).requires_grad_(), 'orientation': torch.randn(*((dt, dt) if o == 'E' else (Kt, dt, dt)), dtype=torch.float64).requires_grad_()}
            free = [params['volume']] + ([params['shape']] if s != 'I' else []) + ([params['orientation']] if o != 'I' else [])
            opt = torch.optim.LBFGS(free, max_iter=400, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-14)

            def closure():
                opt.zero_grad()
                loss = objective_J(assemble(code_name, params))
                loss.backward()
                return loss
            for _ in range(4):
                opt.step(closure)
            best = min(best, objective_J(assemble(code_name, params)).item())
        return best
    _rows = []
    for _code_name in GPCM_CODES:
        res = mstep_gpcm(_code_name, W_test, nk_test, dt, gpcm_tol=1e-12, gpcm_max_iter=500)
        j_mstep = objective_J(res.cov_full).item()
        j_brute = brute_force_minimum(_code_name)
        _rows.append({'code': _code_name, 'J (mstep_gpcm)': j_mstep, 'J (brute force)': j_brute, 'mstep - brute': j_mstep - j_brute})
    df_check = pd.DataFrame(_rows)
    df_check.style.format({'J (mstep_gpcm)': '{:.6f}', 'J (brute force)': '{:.6f}', 'mstep - brute': '{:+.1e}'})
    return (df_check,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `mstep_gpcm` matches the brute-force minimum for all 14 codes (the largest difference is about
    $2\times10^{-9}$ on objective values in the thousands, i.e. the optimiser tolerance), so the formulas above -- including the alternating and Stiefel-manifold
    models -- reach the true minimum of $J$ on this problem.

    The 14 models are also **nested**: constraining a factor can only increase the minimum of $J$.
    Calling a model *more general* than another if it is at least as free in every one of the three
    factors ($V$ before $E$ before $I$), we can check that ordering on the same data:
    """)
    return


@app.cell
def _(GPCM_CODES, df_check):
    FREEDOM = {'V': 0, 'E': 1, 'I': 2}   # lower = freer; volume has no 'I'


    def at_least_as_general(a, b):
        '''True if model `a` contains model `b` as a special case.'''
        return a != b and all(FREEDOM[x] <= FREEDOM[y] for x, y in zip(a, b))


    J_by_code = df_check.set_index('code')['J (mstep_gpcm)']
    pairs = [(a, b) for a in GPCM_CODES for b in GPCM_CODES if at_least_as_general(a, b)]
    violations = [(a, b) for a, b in pairs if J_by_code[a] > J_by_code[b] + 1e-6]
    print(f'{len(pairs)} nested pairs (general, constrained); violations of J_general <= J_constrained: {violations}')

    J_by_code.sort_values().to_frame('J').T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All 14 models on the four-cluster data

    Back to the dataset. The grid arranges the 14 models by *orientation* (columns: identity /
    shared / per-component), so that moving down a column relaxes volume and shape. Compare the
    panels with the true covariances of the four clusters (two spherical of different size, one
    axis-aligned, one tilted). The table under the grid lists each model's log-likelihood next to its
    number of free parameters.
    """)
    return


@app.cell
def _(GaussianMixture, K, SEED, X, device, dynamic_figsize, plot_gmm, plt):
    columns = {'orientation = I  (axis-aligned)': ['EII', 'VII', 'EEI', 'VEI', 'EVI', 'VVI'], 'orientation = E  (shared)': ['EEE', 'VEE', 'EVE', 'VVE'], 'orientation = V  (per component)': ['EEV', 'VEV', 'EVV', 'VVV']}
    fits = {}
    _fig, _axs = plt.subplots(6, 3, figsize=dynamic_figsize(6, 3, base_width=5.5, base_height=3.6), dpi=70)
    for col, (col_title, codes) in enumerate(columns.items()):
        for row in range(6):
            _ax = _axs[row, col]
            if row >= len(codes):
                _ax.axis('off')
                continue
            _code_name = codes[row]
            _gmm = GaussianMixture(n_components=K, covariance_type=_code_name, init_means='kmeans', random_state=SEED, device=device)
            _gmm.fit(X)
            fits[_code_name] = _gmm
            plot_gmm(X.cpu().numpy(), gmm=_gmm, ax=_ax, title=f'{_code_name}   mean log-lik = {float(_gmm.score(X)):.3f}')
            if row == 0:
                _ax.annotate(col_title, xy=(0.5, 1.28), xycoords='axes fraction', ha='center', fontsize=11, weight='bold')
    plt.tight_layout()
    plt.gcf()
    return (fits,)


@app.cell
def _(ALIASES, K, X, d, fits, n, n_cov_params, pd):
    summary = pd.DataFrame([
        {'code': c, 'alias': ALIASES.get(c, ''), 'log-likelihood': g.score(X) * n,
         'covariance params': n_cov_params(c, K, d), 'total free params': n_cov_params(c, K, d) + K * d + K - 1,
         'converged': g.converged_}
        for c, g in fits.items()
    ]).sort_values('log-likelihood', ascending=False).reset_index(drop=True)
    summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    More free parameters always fit the four clusters at least as well, so the log-likelihood alone cannot say
    which structure a dataset supports. That question, together with the number of components, is the subject of
    [13 -- Model Selection](13_model_selection.ipynb), which sweeps all 14 models on real and synthetic data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practical notes and limitations

    - **Default.** `covariance_type='full'` (`VVV`) is the default and the most flexible; prefer a
      constrained model when $n_k$ is small relative to $d(d+1)/2$, or when you believe clusters share
      an orientation or a shape.
    - **MLE only for the 8 new codes.** `mean_prior` / `covariance_prior` (MAP / NIW estimation) raise
      `NotImplementedError` for them -- a conjugate-prior derivation specific to each parameterisation is
      not provided. The six classic types still support the full prior machinery
      ([priors tutorial](04_priors.ipynb)); `weight_concentration_prior` works with all 14.
    - **No Gibbs mode.** `n_components=None` (Dirichlet-process Gibbs sampling) is not available for
      the 8 new codes, as for `tied_*`: a covariance that couples components does not factor per
      component the way the collapsed sampler needs.
    - **Iterative M-steps.** `VEI`, `VEE`, `VEV`, `EVE` and `VVE` alternate until the relative change in
      $J$ drops below `gpcm_tol` (default `1e-6`) or `gpcm_max_iter` (default `20`) sweeps are done; set
      `gpcm_tol` / `gpcm_max_iter` on the `GaussianMixture` to tighten or loosen this, e.g.
      `GaussianMixture(n_components=3, covariance_type='VEE', gpcm_tol=1e-8, gpcm_max_iter=50)`. Each EM
      iteration's alternation is warm-started from the previous iteration's decomposition, so in practice
      it usually needs only one or two sweeps to re-settle.
    - **Initial covariances.** With `init_covariances='empirical'` or `'global'`, the starting
      covariances are projected onto the requested model (its constrained MLE); `'eye'` / `'random'`
      simply start from identity / random matrices.
    - **`reg_covar`** is added to the *assembled* covariances, not to the decomposition, so `volume_`,
      `shape_` and `orientation_` describe the unregularised fit.

    ## Summary

    - Every covariance type is the same M-step objective $J=\sum_k[n_k\log|\Sigma_k|+\mathrm{tr}(\Sigma_k^{-1}W_k)]$
      minimised over a different feasible set; the M-step only needs the sufficient statistics $n_k$ and $W_k$.
    - The six classic types have simple closed forms (pooled for the `tied_` versions), verified here against
      `GaussianMixture`'s converged fits.
    - The GPCM family writes $\Sigma_k=\lambda_kD_kA_kD_k^{\top}$ and lets volume, shape and orientation each be equal,
      varying or (for shape/orientation) the identity -- 14 models whose parameter counts add up factor by factor.
    - Three of the new models have closed-form M-steps, three alternate closed-form sub-steps, and the two with a
      shared orientation and varying shape (`EVE`, `VVE`) need a numerical orthogonal-manifold step -- all matching an
      independent brute-force minimiser.
    - The number of free covariance parameters is a sum over the three factors: only a per-component orientation
      grows like $K\,d^2$, a shared orientation adds a $K$-independent $d^2$ term, and the identity orientation
      grows at most like $K\,d$ -- the formula is checked against fitted models and compared for
      $K\in\{2,4,8\}$ and $d$ up to 20.
    - `volume_` / `shape_` / `orientation_` expose the fitted decomposition of the eight non-classic models.

    **Next:** [03 -- Initialization](03_initialization.ipynb) for how the starting means, covariances and
    weights are chosen, [04 -- Priors](04_priors.ipynb) for regularising covariance estimates (classic types), and
    [13 -- Model Selection](13_model_selection.ipynb) for choosing the covariance structure and the number of
    components for your data.

    ## References

    - Celeux, G. & Govaert, G. (1995). Gaussian parsimonious clustering models. *Pattern Recognition*, 28(5), 781-793.
    - Banfield, J. D. & Raftery, A. E. (1993). Model-based Gaussian and non-Gaussian clustering. *Biometrics*, 49(3), 803-821.
    - Flury, B. N. (1984). Common principal components in $k$ groups. *JASA*, 79(388), 892-898.
    - Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS B*, 39(1), 1-38.
    - Scrucca, L., Fop, M., Murphy, T. B. & Raftery, A. E. (2016). mclust 5: clustering, classification and density estimation using Gaussian finite mixture models. *The R Journal*, 8(1).

    `mclust`'s naming and model definitions are followed for compatibility, but every M-step in `tgmm` is
    derived independently from the papers' equations (`tgmm` is MIT-licensed; `mclust` is GPL, and its
    source was not read or ported).
    """)
    return


if __name__ == "__main__":
    app.run()
