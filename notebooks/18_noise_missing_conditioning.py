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
    # Noise, Missing Values and Conditioning

    A fitted mixture is more than a clustering. This tutorial shows five things you can do *around* the EM fit once the
    model is a proper probability density: give stray points somewhere to go (a **uniform noise component**), fit data
    with holes in it (**missing-value EM**), ask questions of the fitted density (**marginals** and **conditionals**, hence
    Gaussian mixture regression), control what EM is allowed to change (**frozen components** and **sample weights**), and
    bring your own component families (**the component protocol** and `GenericMixture`).

    **What you'll learn**

    - how `noise=True` adds a parameter-free uniform component, what it does to the fitted means, and how `predict` and `noise_proba` report it;
    - how `missing='em'` fits incomplete rows by maximising the likelihood of the *observed* entries, and what `impute` returns;
    - how `marginal` and `condition` turn a fitted `GaussianMixture` into a regressor whose conditional distributions can be bimodal;
    - how `frozen_components` pins a known component and how integer `sample_weight`s equal duplicating rows;
    - how to plug a hand-written component (here an exponential) into `GenericMixture` by implementing three methods.

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb) (the model and EM) and [02 -- Covariance Types](02_covariance_types.ipynb).

    **Notation:** $n$ samples, $d$ features, $K$ components, responsibilities $r_{ik}$, $n_k=\sum_i r_{ik}$.

    **See also:** the API pages for [`ConditionalGaussianMixture`](../api/conditional-gaussian-mixture.md) and [`GenericMixture`](../api/generic-mixture.md).
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
    import math

    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from tgmm import GaussianMixture, GenericMixture, HDBSCAN
    from tgmm.metrics import ClusteringMetrics
    from tgmm.plotting import CATEGORICAL_PALETTE

    # --- figure style (see notebooks/README.md) ---
    DOCS_RC = {'figure.dpi': 100, 'figure.titlesize': 12, 'axes.titlesize': 10.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
               'legend.fontsize': 8, 'lines.linewidth': 1.4, 'lines.markersize': 5}
    plt.rcParams.update(DOCS_RC)

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    return (
        CATEGORICAL_PALETTE,
        ClusteringMetrics,
        GaussianMixture,
        GenericMixture,
        HDBSCAN,
        SEED,
        math,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. A uniform noise component

    Real data often contain points no cluster explains: background clutter, sensor glitches, mislabelled rows. A Gaussian
    component asked to cover them has to widen and shift. The classical remedy (Banfield & Raftery, 1993) is to add one more
    mixture component that is **uniform over a region of volume $V$** and has no parameters except its weight $\pi_0$:

    $$
    p(x)=\sum_{k=1}^{K}\pi_k f_k(x)+\frac{\pi_0}{V},\qquad \sum_k\pi_k+\pi_0=1 .
    $$

    The E-step simply appends a column $\log\pi_0-\log V$ to the component log-densities, so the responsibilities become
    $(n,K+1)$; the last column is the noise responsibility $\tau_{i0}$. The M-step sets $\pi_0=\sum_i\tau_{i0}/n$ and updates the $K$
    components from the first $K$ columns as usual. By default $V$ is the volume of the bounding box of the data
    (`noise_volume='bbox'`).

    Below: three Gaussian blobs plus 60 points sprinkled uniformly over a larger rectangle.
    """)
    return


@app.cell
def _(SEED, plt, torch):
    _g = torch.Generator().manual_seed(SEED)
    _centers = torch.tensor([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    _blobs = torch.cat([_c + 0.7 * torch.randn(200, 2, generator=_g) for _c in _centers])
    _lo, _hi = _blobs.min(0).values - 2, _blobs.max(0).values + 2
    _background = _lo + (_hi - _lo) * torch.rand(60, 2, generator=_g)

    X_noise = torch.cat([_blobs, _background])
    y_noise = torch.cat([torch.arange(3).repeat_interleave(200), torch.full((60,), -1)])  # -1 = background

    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(X_noise[:, 0], X_noise[:, 1], c=y_noise, cmap='viridis', s=8, alpha=0.5)
    plt.title('Three blobs plus uniform background (yellow)')
    plt.gcf()
    return X_noise, y_noise


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fit an ordinary `GaussianMixture` and one with `noise=True`. With the noise component on, `predict` returns **`-1`** for the
    points whose most probable "component" is the noise column -- the same convention as `HDBSCAN` and
    `ClusteringMetrics(noise_label=-1)` -- and `weights_` holds the $K$ *component* weights, which now sum to
    $1-\pi_0$ (`noise_weight_` is $\pi_0$).
    """)
    return


@app.cell
def _(
    ClusteringMetrics,
    GaussianMixture,
    MAX_ITER,
    N_INIT,
    SEED,
    X_noise,
    torch,
    y_noise,
):
    gmm_plain = GaussianMixture(n_components=3, random_state=SEED, max_iter=MAX_ITER, n_init=N_INIT).fit(X_noise)
    gmm_noise = GaussianMixture(n_components=3, noise=True, random_state=SEED, max_iter=MAX_ITER, n_init=N_INIT).fit(X_noise)

    pred_noise = gmm_noise.predict(X_noise).cpu()
    _is_noise, _flagged = (y_noise == -1), (pred_noise == -1)
    _kept = ~_is_noise & ~_flagged
    print(f'estimated noise weight pi_0 = {float(gmm_noise.noise_weight_):.3f}  (true fraction {float(_is_noise.float().mean()):.3f})')
    print(f'component weights sum to {float(gmm_noise.weights_.sum()):.3f}')
    print(f'background points flagged as noise: {int((_is_noise & _flagged).sum())} of {int(_is_noise.sum())}')
    print(f'ARI on the non-noise points: {ClusteringMetrics.adjusted_rand_score(y_noise[_kept], pred_noise[_kept]):.3f}')
    print('mean covariance trace, plain vs noise model:',
          f'{float(torch.diagonal(gmm_plain.covariances_, dim1=-2, dim2=-1).sum(-1).mean()):.2f}',
          f'{float(torch.diagonal(gmm_noise.covariances_, dim1=-2, dim2=-1).sum(-1).mean()):.2f}')
    return gmm_noise, gmm_plain


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plain model inflates its covariances to swallow the background (left); the noise model keeps them tight (right) (true trace of each
    covariance is $2\times0.7^2\approx0.98$). The figure draws the 95% ellipse of every component for both fits and colours
    the points by the noise model's `noise_proba(X)`, the posterior probability that a point belongs to the uniform component.
    """)
    return


@app.cell
def _(CATEGORICAL_PALETTE, X_noise, gmm_noise, gmm_plain, math, plt, torch):
    def _ellipse(ax, mean, cov, color):
        mean, cov = mean.cpu(), cov.cpu()  # the fitted model may live on the GPU; the ellipse is drawn on the CPU
        _vals, _vecs = torch.linalg.eigh(cov)
        _t = torch.linspace(0, 2 * math.pi, 100)
        _pts = (_vecs @ torch.diag(2.4477 * _vals.sqrt()) @ torch.stack([_t.cos(), _t.sin()])).T + mean  # the 95% region: sqrt(chi2.ppf(0.95, 2)) = 2.4477
        ax.plot(_pts[:, 0], _pts[:, 1], color=color)


    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    _axes[0].scatter(X_noise[:, 0], X_noise[:, 1], s=6, color='0.6', alpha=0.5)
    for _k in range(3):
        _ellipse(_axes[0], gmm_plain.means_[_k], gmm_plain.covariances_[_k], CATEGORICAL_PALETTE[_k])
    _axes[0].set_title('GaussianMixture (plain)')

    _sc = _axes[1].scatter(X_noise[:, 0], X_noise[:, 1], c=gmm_noise.noise_proba(X_noise).detach().cpu().numpy(), cmap='viridis_r', s=8, alpha=0.6, vmin=0, vmax=1)
    for _k in range(3):
        _ellipse(_axes[1], gmm_noise.means_[_k], gmm_noise.covariances_[_k], CATEGORICAL_PALETTE[_k])
    _axes[1].set_title('noise=True, colour = noise_proba(X)')
    _fig.colorbar(_sc, ax=_axes[1], label='noise probability')
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Starting from a density-based guess

    `noise_init` seeds the noise column from a per-point flag. A boolean tensor means "`True` = noise"; any other dtype is read as
    cluster labels with `-1` = noise, so the output of `HDBSCAN` can be passed as it is. The initial means and covariances are then
    computed from the unflagged points only, and `noise_prior` is replaced by the flagged fraction.
    """)
    return


@app.cell
def _(GaussianMixture, HDBSCAN, MAX_ITER, SEED, X_noise, torch):
    _hdb = HDBSCAN(min_cluster_size=30).fit(X_noise)
    _labels = torch.as_tensor(_hdb.labels_)
    gmm_hdb_init = GaussianMixture(n_components=3, noise=True, noise_init=_labels, random_state=SEED,
                                   max_iter=MAX_ITER).fit(X_noise)
    print(f'HDBSCAN flagged {int((_labels == -1).sum())} points; fitted pi_0 = {float(gmm_hdb_init.noise_weight_):.3f}')
    print('noise_proba shape:', tuple(gmm_hdb_init.noise_proba(X_noise).shape),
          ' predict_proba shape:', tuple(gmm_hdb_init.predict_proba(X_noise).shape))
    _s, _idx = gmm_hdb_init.sample(2000)
    print(f'sample(): {int((_idx == -1).sum())} of 2000 draws are noise (index -1), uniform in the fit bounding box')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Arguments, families and limits

    The noise log-density is the same constant $-\log V$ at every $x$, **also outside the data's bounding box**: a new point far from everything is
    given to the noise component, not rejected. The M-step is $\pi_0=\sum_i\tau_{i0}/n$, and `weights_` $=n_k/n$ keeps shape $(K,)$ and sums to
    $1-\pi_0$, which is why `sample`, `plot_gmm`, `plot_vmf` and `save`/`load` work unchanged.

    | argument | default | meaning |
    | --- | --- | --- |
    | `noise` | `False` | switch the component on; with `False` every fit is bit-identical to a model without the feature |
    | `noise_volume` | `'bbox'` | `'bbox'`: volume of the axis-aligned bounding box of the fit data, $\prod_j(\max_j-\min_j)$ (each extent floored at $10^{-12}$), stored in `noise_bounds_`; or a positive float, a hypervolume you supply |
    | `noise_prior` | `0.05` | the initial $\pi_0\in(0,1)$ |
    | `noise_init` | `None` | per-sample initial flag of length $n$: a `bool` tensor is a mask (`True` = noise), any other dtype is read as cluster labels with `-1` = noise. It overrides `noise_prior`, and the initial means and covariances come from the unflagged rows only |

    **Families.** The noise log-density is $-\log V$ for `GaussianMixture` (all covariance types and GPCM codes) and `StudentTMixture`. For `VonMisesFisherMixture`
    it is $-\log A_d$ with $A_d=2\pi^{d/2}/\Gamma(d/2)$ the surface area of the unit sphere $S^{d-1}$, i.e. the uniform density on the sphere, independent of
    `noise_volume`. Families without bounded continuous support raise `ValueError`.

    **Behaviour.** `predict_proba(X)` has shape $(n,K+1)$. `score_samples`, `bic`, `aic`, `icl` and `select_model` include the noise term, which costs **one extra
    parameter**. `sample` draws a noise point with probability $\pi_0$, uniformly in the stored bounding box (uniformly on the sphere for vMF), with component
    index `-1`. It works with `em`, `cem`, `sem`, `saem`, `temperature`, `batch_size`, `n_init`, `warm_start` and persistence. The cell below checks the
    parameter count, the density outside the box, `noise_volume` and the vMF constant.

    **Unsupported combinations (they raise):** Gibbs mode (`n_components=None`); `fit(X, labels=...)` (a point cannot be labelled "known noise", since `-1` already means
    unlabelled); online EM (`step_size`) and `partial_fit`; `weight_concentration_prior`; the constrained `GaussianMixture.sample` variants (`component=`, `confidence=`, ...);
    and `missing='em'`.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, SEED, X_noise, gmm_noise, gmm_plain, math, torch):
    from tgmm import VonMisesFisherMixture as _VMF

    _n = X_noise.shape[0]
    _implied = lambda g: (g.bic(X_noise) + 2 * g.score_samples(X_noise).sum().item()) / math.log(_n)
    print(f'free parameters implied by bic(): plain {_implied(gmm_plain):.1f}, noise model {_implied(gmm_noise):.1f}  (one more: pi_0)')

    _V = float((gmm_noise.noise_bounds_[1] - gmm_noise.noise_bounds_[0]).prod())
    print(f'bounding-box volume V = {_V:.1f}; noise density 1/V = {1 / _V:.4f} everywhere')
    print('noise_proba at (100, 100), far outside the box:', float(gmm_noise.noise_proba(torch.tensor([[100.0, 100.0]]))))

    _small_V = GaussianMixture(n_components=3, noise=True, noise_volume=50.0, noise_prior=0.1, random_state=SEED, max_iter=MAX_ITER).fit(X_noise)
    print(f'pi_0 with the default bbox volume {float(gmm_noise.noise_weight_):.3f}; with noise_volume=50 (denser noise) {float(_small_V.noise_weight_):.3f}')

    _U = torch.nn.functional.normalize(torch.randn(300, 3, generator=torch.Generator().manual_seed(SEED)), dim=1)
    _vmf = _VMF(n_components=2, noise=True, random_state=SEED).fit(_U)
    print(f'vMF noise log-density {float(_vmf._noise_log_density(_U[:1])):.4f}  =  -log(4 pi) = {-math.log(4 * math.pi):.4f}  (A_3 = 4 pi, the area of S^2)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Missing values

    With `missing='em'`, `NaN` entries are treated as **missing at random** and the likelihood is maximised over the entries that
    *were* observed. For a row with observed block $o$ and missing block $m$, and component $k$:

    - the E-step uses the **marginal** density $\mathcal N(x_o;\mu_{k,o},\Sigma_{k,oo})$;
    - the M-step fills the hole with the **conditional mean**
      $\hat x_m=\mu_{k,m}+\Sigma_{k,mo}\Sigma_{k,oo}^{-1}(x_o-\mu_{k,o})$ and adds the **conditional covariance**
      $C_k=\Sigma_{k,mm}-\Sigma_{k,mo}\Sigma_{k,oo}^{-1}\Sigma_{k,om}$ to the second moment, so the variance is not underestimated.

    In full, with $A_k=\Sigma_{k,mo}\Sigma_{k,oo}^{-1}$ and $p=|o|$ (a Gaussian-mixture treatment going back to Ghahramani & Jordan, 1994, and Hunt & Jorgensen, 2003):

    - **E-step (marginal likelihood):** $\log p(x_i\mid k)=-\tfrac12\big[p\log2\pi+\log|\Sigma_{k,oo}|+\delta_o\big]$ with
      $\delta_o=(x_o-\mu_{k,o})^\top\Sigma_{k,oo}^{-1}(x_o-\mu_{k,o})$.
    - **Conditional mean and covariance of the hole:** $\hat x_m=\mu_{k,m}+A_k(x_o-\mu_{k,o})$ and $C_k=\Sigma_{k,mm}-A_k\Sigma_{k,om}$.
    - **Completed statistics.** Let $\tilde x_{ik}$ be the row with the hole replaced by $\hat x_m$. Then $n_k=\sum_ir_{ik}$, $s_{1,k}=\sum_ir_{ik}\tilde x_{ik}$, $\bar y_k=s_{1,k}/n_k$ and

    $$
    s_{2,k}=\sum_i r_{ik}(\tilde x_{ik}-\bar y_k)(\tilde x_{ik}-\bar y_k)^\top+\sum_i r_{ik}\,C_{k,\text{pattern}(i)}\qquad(C\text{ placed in the }m\times m\text{ block}),
    $$

      which is exactly $\mathbb E\big[\sum_ir_{ik}(x_i-\bar y_k)(x_i-\bar y_k)^\top\mid x_o,\theta^{\text{old}}\big]$. Every M-step formula of
      [01 -- Introduction](01_introduction.ipynb), [02 -- Covariance Types](02_covariance_types.ipynb) and [04 -- Priors](04_priors.ipynb) (MLE, MAP/NIW, the 14 GPCM models) reads
      only $(n_k,s_1,s_2)$, so it applies unchanged.

    Rows are grouped by **missingness pattern** (`torch.unique(torch.isnan(X), dim=0)`), so the cost is one batched solve per pattern per component, not per row (a
    `UserWarning` is raised when there are more than 1000 distinct patterns). For `diag` and `spherical` types $\Sigma_{mo}=0$, hence $\hat x_m=\mu_m$ and
    $C=\operatorname{diag}\Sigma_{mm}$: no solves at all. Initialisation runs on the column-mean-imputed data. Below, 20% of the entries
    of a two-cluster dataset are deleted completely at random (keeping at least one observed entry per row: a row with nothing observed
    carries no information and is rejected).
    """)
    return


@app.cell
def _(SEED, torch):
    _g = torch.Generator().manual_seed(SEED + 1)
    _a = torch.randn(300, 2, generator=_g) * 0.8
    _b = torch.randn(300, 2, generator=_g) * 0.8 + torch.tensor([4.0, 3.0])
    X_complete = torch.cat([_a, _b])
    y_missing = torch.arange(2).repeat_interleave(300)

    holes_mask = torch.rand(X_complete.shape, generator=_g) < 0.2
    holes_mask[holes_mask.all(1), 0] = False                 # keep one observed entry in every row
    X_holes = X_complete.clone()
    X_holes[holes_mask] = float('nan')
    print(f'{int(holes_mask.sum())} of {X_holes.numel()} entries are missing; {int(holes_mask.any(1).sum())} rows are incomplete')
    return X_complete, X_holes, holes_mask, y_missing


@app.cell
def _(
    ClusteringMetrics,
    GaussianMixture,
    MAX_ITER,
    N_INIT,
    SEED,
    X_complete,
    X_holes,
    y_missing,
):
    gmm_complete = GaussianMixture(n_components=2, random_state=SEED, max_iter=MAX_ITER, n_init=N_INIT).fit(X_complete)
    gmm_missing = GaussianMixture(n_components=2, missing='em', random_state=SEED, max_iter=MAX_ITER, n_init=N_INIT).fit(X_holes)

    print('means, complete data :', gmm_complete.means_.cpu().numpy().round(2).tolist())
    print('means, 20% missing   :', gmm_missing.means_.cpu().numpy().round(2).tolist())
    print(f'ARI, complete data : {ClusteringMetrics.adjusted_rand_score(y_missing, gmm_complete.predict(X_complete)):.3f}')
    print(f'ARI, 20% missing   : {ClusteringMetrics.adjusted_rand_score(y_missing, gmm_missing.predict(X_holes)):.3f}   (predict accepts rows with NaN)')
    return (gmm_missing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `impute(X)` returns `X` with every `NaN` replaced by the posterior-weighted conditional mean $\sum_k\tau_{ik}\hat x_{ik}$; observed entries
    are untouched. Compared with filling each column with its mean, the model-based imputation uses the row's *other* coordinates and the cluster
    structure.
    """)
    return


@app.cell
def _(
    CATEGORICAL_PALETTE,
    X_complete,
    X_holes,
    gmm_missing,
    holes_mask,
    plt,
    torch,
):
    X_imputed = gmm_missing.impute(X_holes).cpu()
    _col_mean = torch.nanmean(X_holes, dim=0).expand_as(X_holes)
    _rmse_model = float(((X_imputed - X_complete)[holes_mask] ** 2).mean().sqrt())
    _rmse_mean = float(((_col_mean - X_complete)[holes_mask] ** 2).mean().sqrt())
    print(f'RMSE of the {int(holes_mask.sum())} imputed entries: model-based {_rmse_model:.2f}  vs  column mean {_rmse_mean:.2f}')

    _incomplete = holes_mask.any(1)
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(X_complete[~_incomplete, 0], X_complete[~_incomplete, 1], s=8, color='0.75', alpha=0.5, label='complete rows')
    plt.scatter(X_imputed[_incomplete, 0], X_imputed[_incomplete, 1], s=8, color=CATEGORICAL_PALETTE[1], alpha=0.6, label='imputed rows')
    plt.legend()
    plt.title('Imputed rows sit on the cluster structure')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Student-t, defaults and limits

    **Student-t.** `missing='em'` is available for `GaussianMixture` (all six classic covariance types and all 14 GPCM codes) and `StudentTMixture`. With the scale-mixture form
    ($w\sim\mathrm{Gamma}(\nu/2,\nu/2)$, $x\mid w\sim\mathcal N(\mu,\Sigma/w)$) the observed block is a $t_p$ and the weights
    $u_{ik}=(\nu_k+p)/(\nu_k+\delta_o)$ use the number $p$ of *observed* entries. The conditional mean does not depend on $w$, so
    $\mathbb E[w\,x_m\mid x_o]=u\,\hat x_m$ and $\mathbb E[w(x_m-c)(x_m-c)^\top\mid x_o]=u(\hat x_m-c)(\hat x_m-c)^\top+C_k$: the conditional covariance carries **no** factor $u$. The
    degrees of freedom use the textbook update per row with that row's own $p$, and `scale_update='ecme'` is not available together with `missing='em'`.

    **Defaults.** With `missing='error'` (the default) a `NaN` raises a `ValueError` that points at `missing='em'`; earlier versions silently propagated `NaN` into the parameters.
    `missing='em'` on data without `NaN` gives results identical to `'error'`.

    **What raises.** Rows with no observed entry and entirely-`NaN` columns (`ValueError`: they carry no information); `noise=True`, Gibbs mode, `labels=`, online EM and
    `partial_fit` (`NotImplementedError`); `VonMisesFisherMixture` (`ValueError`).
    """)
    return


@app.cell
def _(GaussianMixture, X_holes, gmm_missing, holes_mask):
    try:
        GaussianMixture(n_components=2).fit(X_holes)
    except ValueError as _err:
        print('missing="error" (default):', _err)

    _X_imputed = gmm_missing.impute(X_holes).cpu()
    print('impute leaves observed entries untouched:', bool((_X_imputed[~holes_mask] == X_holes[~holes_mask]).all()), '| NaN left:', bool(_X_imputed.isnan().any()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Marginals and conditionals: Gaussian mixture regression

    A fitted `GaussianMixture` is a joint density, so it can answer any question about subsets of coordinates.

    - `gmm.marginal(dims)` returns a new `GaussianMixture` over the chosen coordinates (diagonal, spherical and tied types keep their type; the
      parametrised GPCM codes become `full`).
    - `gmm.condition(x_obs, obs_dims)` returns a `ConditionalGaussianMixture`: for every row of `x_obs`, the distribution of the *remaining* coordinates,

    $$
    p(y\mid x_{o,i})=\sum_{k}w_{ik}\,\mathcal N\big(y;\ m_{ik},\ C_k\big),\qquad
    w_{ik}\propto\pi_k\mathcal N(x_{o,i};\mu_{k,o},\Sigma_{k,oo}),
    $$

    with $m_{ik}=\mu_{k,m}+\Sigma_{k,mo}\Sigma_{k,oo}^{-1}(x_{o,i}-\mu_{k,o})$ and $C_k=\Sigma_{k,mm}-\Sigma_{k,mo}\Sigma_{k,oo}^{-1}\Sigma_{k,om}$.
    The weights depend on the observed values, the conditional covariances do not. `mean()` is **Gaussian mixture regression**:
    $\mathbb E[y\mid x_o]=\sum_kw_{ik}m_{ik}$.

    Fit eight components to noisy samples of $y=\sin 2x$ and regress $y$ on $x$.
    """)
    return


@app.cell
def _(
    CATEGORICAL_PALETTE,
    GaussianMixture,
    MAX_ITER,
    N_INIT,
    SEED,
    plt,
    torch,
):
    _g = torch.Generator().manual_seed(SEED + 2)
    _x = torch.rand(600, 1, generator=_g) * 6 - 3
    _y = torch.sin(2 * _x) + 0.15 * torch.randn(600, 1, generator=_g)
    XY_sine = torch.cat([_x, _y], dim=1)

    gmm_joint = GaussianMixture(n_components=8, random_state=SEED, max_iter=MAX_ITER, n_init=N_INIT).fit(XY_sine)

    x_grid = torch.linspace(-3, 3, 200)[:, None]
    cond_sine = gmm_joint.condition(x_grid, [0])          # condition on coordinate 0 (x)
    gmr_mean = cond_sine.mean()[:, 0].cpu()
    _rmse = float(((gmr_mean - torch.sin(2 * x_grid[:, 0])) ** 2).mean().sqrt())
    print(f'weights {tuple(cond_sine.weights_.shape)}, means {tuple(cond_sine.means_.shape)}, covariances {tuple(cond_sine.covariances_.shape)}')
    print(f'RMSE of the GMR curve against the noiseless sin(2x): {_rmse:.3f}')

    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(XY_sine[:, 0], XY_sine[:, 1], s=6, color='0.7', alpha=0.5, label='data')
    plt.plot(x_grid[:, 0], torch.sin(2 * x_grid[:, 0]), 'k--', label='sin(2x)')
    plt.plot(x_grid[:, 0], gmr_mean, color=CATEGORICAL_PALETTE[1], label='E[y | x] (GMR)')
    plt.legend()
    plt.title('Gaussian mixture regression with 8 components')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The conditional is a whole *distribution*, not just a mean, and it can be multimodal. On an X-shaped dataset ($y=\pm x$ with equal
    probability) the regression mean at $x>0$ is close to zero -- a value the data almost never take -- while the conditional density has two clear
    peaks. `log_prob(Y)` evaluates it on a grid and `as_gmm(i)` hands over the conditional of row `i` as an ordinary one-dimensional `GaussianMixture`.
    """)
    return


@app.cell
def _(
    CATEGORICAL_PALETTE,
    GaussianMixture,
    MAX_ITER,
    N_INIT,
    SEED,
    plt,
    torch,
):
    _g = torch.Generator().manual_seed(SEED + 3)
    _t = torch.rand(800, 1, generator=_g) * 4 - 2
    _sign = torch.where(torch.rand(800, 1, generator=_g) < 0.5, 1.0, -1.0)
    XY_cross = torch.cat([_t, _sign * _t + 0.1 * torch.randn(800, 1, generator=_g)], dim=1)

    gmm_cross = GaussianMixture(n_components=6, random_state=SEED, max_iter=MAX_ITER, n_init=N_INIT).fit(XY_cross)
    _x_query = torch.tensor([[0.5], [1.5]])
    cond_cross = gmm_cross.condition(_x_query, [0])        # one conditional mixture per query row

    y_axis = torch.linspace(-3, 3, 300)[:, None]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _axes[0].scatter(XY_cross[:, 0], XY_cross[:, 1], s=5, color='0.7', alpha=0.5)
    for _i, _xq in enumerate(_x_query[:, 0].tolist()):
        _axes[0].axvline(_xq, color=CATEGORICAL_PALETTE[_i])
        _single = gmm_cross.condition(torch.tensor([[_xq]]), [0])      # a single-row conditional broadcasts against the whole grid
        _axes[1].plot(y_axis[:, 0], _single.log_prob(y_axis).exp().cpu(), color=CATEGORICAL_PALETTE[_i], label=f'x = {_xq}')
    _axes[0].set_title('X-shaped data: y = ±x')
    _axes[1].set_title('conditional density p(y | x): two peaks')
    _axes[1].set_xlabel('y')
    _axes[1].legend()
    plt.tight_layout()
    _fig
    return (cond_cross,)


@app.cell
def _(cond_cross):
    _conditional_at_1_5 = cond_cross.as_gmm(1)
    print('as_gmm(1):', type(_conditional_at_1_5).__name__, 'with', _conditional_at_1_5.n_components, 'components over the missing coordinate')
    print('regression mean at x = 1.5 :', f'{float(cond_cross.mean()[1, 0]):+.2f}   (mixture weights {cond_cross.weights_[1].cpu().numpy().round(2).tolist()})')
    print('conditional component means:', cond_cross.means_[1, :, 0].cpu().numpy().round(2).tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Details and limits

    **`marginal(dims)`**: component $k$ becomes $\mathcal N(\mu_k[\text{dims}],\Sigma_k[\text{dims},\text{dims}])$ with the same weights. `diag`, `spherical` and `tied_*` keep their type; `full`
    and the GPCM codes without a classic name return `full`, because the volume/shape/orientation decomposition does not survive restriction. `StudentTMixture.marginal` is exact too
    (same $\nu$, restricted scale matrix).

    **`condition(x_obs, obs_dims)`**: all steps are batched Cholesky solves, with no loop over samples. `obs_dims` keeps the order you give (column $j$ of `x_obs` is coordinate
    `obs_dims[j]`); the conditional is over the **sorted complement**. The result carries `weights_` $(n,K)$, `means_` $(n,K,d_m)$, `covariances_` $(K,d_m,d_m)$, `obs_dims` and
    `missing_dims`. Besides `mean()`, `log_prob(Y)` (a single-row conditional broadcasts against many `Y`) and `as_gmm(i)`, `sample(n_draws)` returns $(n,n_\text{draws},d_m)$.

    **Not supported:** Gibbs-mode and `noise=True` models (a noise component has no Gaussian-mixture form after conditioning); `StudentTMixture.condition` (the conditional $t$ has
    $\nu+d_o$ degrees of freedom and a sample-dependent scale, so it is not a member of this class); `VonMisesFisherMixture`.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, SEED, torch):
    _g = torch.Generator().manual_seed(SEED + 6)
    _X3 = torch.cat([torch.randn(200, 3, generator=_g), torch.randn(200, 3, generator=_g) + 3.0])
    _gmm3 = GaussianMixture(n_components=3, random_state=SEED, max_iter=MAX_ITER).fit(_X3)
    _c = _gmm3.condition(torch.tensor([[0.5, 0.1]]), obs_dims=[2, 0])        # column 0 of x_obs is coordinate 2, column 1 is coordinate 0
    print('obs_dims', _c.obs_dims, '| missing_dims', _c.missing_dims, '(sorted complement)')
    print('weights_', tuple(_c.weights_.shape), '| means_', tuple(_c.means_.shape), '| covariances_', tuple(_c.covariances_.shape))
    _gd = GaussianMixture(n_components=3, covariance_type='diag', random_state=SEED, max_iter=MAX_ITER).fit(_X3)
    print('marginal types:', f"full -> {_gmm3.marginal([0, 2]).covariance_type}", '|', f"diag -> {_gd.marginal([0, 1]).covariance_type}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Controlling what EM changes: `frozen_components` and `sample_weight`

    **Frozen components.** `frozen_components=[k, ...]` keeps those components' means and covariances (or `dof_` / `kappa_` for the other families)
    at their **initial** values for the whole fit; the mixing weights still update (and so does the noise weight). Use it to pin a component you already
    know -- a calibrated background, a reference population -- and let the others adapt around it. Priors never shrink a frozen component and SEM never
    re-seeds it. The initial values come from `init_means` / `init_covariances` or `warm_start`; $\sum_k\pi_k+\pi_0=1$ is preserved. Not supported (`ValueError`): tied covariance types and GPCM codes
    without a classic name, `dof='shared'`, `concentration='shared'` and Gibbs mode. `PositiveUnlabeledMixture(anchor='fixed')`, in
    [17 -- Semi-Supervised and Positive-Unlabeled Fitting](17_semi_supervised_pu.ipynb), is built on it.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, SEED, torch):
    _g = torch.Generator().manual_seed(SEED + 4)
    X_pair = torch.cat([torch.randn(300, 2, generator=_g) * 0.7, torch.randn(300, 2, generator=_g) * 0.7 + torch.tensor([4.0, 0.0])])
    init_pair = torch.tensor([[0.0, 0.0], [4.5, 0.5]])          # slightly off for the second component

    gmm_frozen = GaussianMixture(n_components=2, init_means=init_pair, frozen_components=[0], random_state=SEED,
                                 max_iter=MAX_ITER).fit(X_pair)
    print('component 0 (frozen)  mean:', gmm_frozen.means_[0].tolist(), '  -> exactly the initial value')
    print('component 1 (free)    mean:', gmm_frozen.means_[1].cpu().numpy().round(2).tolist(), '  (moved from', init_pair[1].tolist(), 'towards the data)')
    print('mixing weights still update:', gmm_frozen.weights_.cpu().numpy().round(3).tolist(), ' sum =', round(float(gmm_frozen.weights_.sum()), 6))
    return X_pair, init_pair


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Sample weights.** `fit(X, sample_weight=w)` with $w\in\mathbb R_{\ge0}^n$ counts row $i$ as $w_i$ observations: responsibilities are scaled by $w_i$
    before the sufficient statistics, so $n_k$, $s_1$, $s_2$ and the mixing weights are weighted. Integer weights are therefore *equivalent to duplicating
    rows* -- a handy way to fit pre-aggregated data -- and a weight of 0 removes a row. Per-feature weights of shape $(n,d)$ are accepted by the models whose
    likelihood factorises over features (`'diag'` / `'spherical'` Gaussians, diagonal Student-t); a zero entry hides one feature of one row.

    In detail, **row weights** (shape $(n,)$) make the effective sample count $n_\text{eff}=\sum_iw_i$: the mixing weights are $\pi_k=n_k/n_\text{eff}$, MAP priors use $n_\text{eff}$, and
    `lower_bound_` is $\sum_iw_i\log p(x_i)/\sum_iw_i$. Hence a weight of 0 drops the row, a constant rescaling changes nothing, and all-ones is bit-identical to no weights. Initialisation
    ignores the weights (except to exclude zero-weight rows), and `bic`, `aic`, `icl` and `score` take only `X`. Row weights work with `em`/`cem`/`sem`/`saem`, `batch_size`, `n_init`,
    `warm_start`, `noise=True` and `missing='em'` (each on its own: the last two cannot be combined); they raise `NotImplementedError` in Gibbs mode, with `labels=` and with online EM.

    **Per-feature weights** (shape $(n,d)$) are for maximum likelihood only. A weight $w_{ij}=0$ hides feature $j$ of row $i$ (exact marginalisation). For `'diag'`:

    $$
    \log p(x_i\mid k)=-\tfrac12\sum_jw_{ij}\Big[\log2\pi+\log s_{kj}+\frac{(x_{ij}-m_{kj})^2}{s_{kj}}\Big],\qquad
    m_{kj}=\frac{\sum_ir_{ik}w_{ij}x_{ij}}{N_{kj}},\quad s_{kj}=\frac{\sum_ir_{ik}w_{ij}(x_{ij}-m_{kj})^2}{N_{kj}},
    $$

    with $N_{kj}=\sum_ir_{ik}w_{ij}$. Other models raise `ValueError`. This is the same likelihood as `missing='em'` for a diagonal model, which the last cell of this section checks.

    pomegranate's `inertia` is deliberately not added: `step_size` (online EM / SAEM) is the same control on the statistics.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, SEED, X_pair, init_pair, torch):
    _w = torch.ones(len(X_pair))
    _w[:300] = 3.0                                          # the first blob counts three times
    gmm_weighted = GaussianMixture(n_components=2, init_means=init_pair, random_state=SEED, max_iter=MAX_ITER).fit(X_pair, sample_weight=_w)
    gmm_duplicated = GaussianMixture(n_components=2, init_means=init_pair, random_state=SEED, max_iter=MAX_ITER).fit(
        torch.cat([X_pair[:300]] * 3 + [X_pair[300:]]))

    print('weights with sample_weight :', gmm_weighted.weights_.cpu().numpy().round(4).tolist())
    print('weights with duplicated rows:', gmm_duplicated.weights_.cpu().numpy().round(4).tolist())
    print('largest difference in the fitted means:', f'{float((gmm_weighted.means_ - gmm_duplicated.means_).abs().max()):.1e}', '(float32 round-off)')
    return


@app.cell
def _(GaussianMixture, SEED, torch):
    _g = torch.Generator().manual_seed(SEED + 7)
    _X = torch.cat([torch.randn(200, 2, generator=_g) * 0.7, torch.randn(200, 2, generator=_g) * 0.7 + torch.tensor([4.0, 3.0])])
    _hole = torch.rand(_X.shape, generator=_g) < 0.2
    _hole[_hole.all(1), 0] = False
    _X_nan = _X.clone()
    _X_nan[_hole] = float('nan')
    _start = torch.tensor([[0.0, 0.0], [4.0, 3.0]])
    _kw = dict(n_components=2, covariance_type='diag', init_means=_start, random_state=SEED, tol=1e-8, max_iter=500)

    _em = GaussianMixture(missing='em', **_kw).fit(_X_nan)
    _pf = GaussianMixture(**_kw).fit(torch.where(_hole, torch.zeros_like(_X), _X), sample_weight=(~_hole).float())   # w_ij = 0 hides entry (i, j)
    print('diag model: per-feature weights vs missing="em" on the same holes')
    print('  largest difference in means      :', f'{float((_em.means_ - _pf.means_).abs().max()):.1e}')
    print('  largest difference in variances  :', f'{float((_em.covariances_ - _pf.covariances_).abs().max()):.1e}')

    _w = torch.rand(len(_X), generator=_g) + 0.5
    _gw = GaussianMixture(n_components=2, random_state=SEED).fit(_X, sample_weight=_w)
    print(f'lower_bound_ {_gw.lower_bound_:.5f}  vs  sum(w * log p) / sum(w) = {float((_w * _gw.score_samples(_X).cpu()).sum() / _w.sum()):.5f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Bring your own component: the protocol and `GenericMixture`

    Every family in `tgmm` is a `BaseMixture`, and `GenericMixture` is the one that takes its components from you. A component is any object that
    implements three methods (the shape pomegranate popularised):

    | method | job |
    | --- | --- |
    | `log_probability(X)` | return $\log f_k(x_i)$, shape `(n,)` |
    | `summarize(X, sample_weight)` | **accumulate** weighted statistics of one chunk of data (`sample_weight` = responsibility $\times$ row weight) |
    | `from_summaries()` | update the parameters from everything summarised, then reset the accumulators |

    The M-step calls `summarize` once per data chunk and `from_summaries` once, so `batch_size` works for free. Optional members are `sample(n)`,
    `n_parameters()` (needed for `bic`/`aic`/`icl`) and a `frozen` flag. A plain `torch.nn.Module` with a `log_prob(X)` method is also accepted: it is
    updated by a few L-BFGS steps that are kept only if they improve the expected complete-data log-likelihood.

    **M-step.** For every data chunk and every non-frozen component, `summarize(X_chunk, resp_chunk[:, k])` is called, then `from_summaries()` once. Because `summarize` accumulates, chunked equals
    unchunked; mixing weights are $\pi_k=n_k/n$ as in every family. The protocol is `tgmm.mixtures.protocol.Component`, a runtime-checkable `typing.Protocol` whose shape is borrowed from
    pomegranate 1.x (MIT); tgmm does not import pomegranate. Optional members: `sample(n)`, `n_parameters()` (fallback: the number of scalar entries in the object's public tensors), `frozen` and `reseed(x)`.

    **Autograd components.** A `torch.nn.Module` with learnable parameters and a `log_prob(X)` method (and no `summarize`) is wrapped automatically. Its M-step maximises
    $Q_k(\theta_k)=\sum_iw_{ik}\log f_k(x_i\mid\theta_k)\big/\sum_iw_{ik}$ with a few L-BFGS steps and **keeps the update only if $Q_k$ increases**, so the log-likelihood stays monotone (a generalised EM step).

    **Decisions.** The components you pass *are* the starting point (no k-means) and `n_init` must be 1; frozen parameters still count in BIC; `em`, `cem`, `sem` (with `sem_estimate='best'`) and
    `temperature` work, while `saem`, online EM and `partial_fit` raise `NotImplementedError` (opaque `summarize` state cannot be blended); `noise=True` needs a float `noise_volume`; per-feature
    weights and `missing='em'` raise `ValueError`; and `save`/`load` pickle the components, so their classes must be importable.

    Here is a tiny mixture of an **exponential** (waiting times) and a **normal** (a burst around 6) over positive data -- neither is a `GaussianMixture` component.
    """)
    return


@app.cell
def _(math, torch):
    class Exponential:
        """f(x) = rate * exp(-rate x) on x >= 0; the weighted M-step is rate = sum(w) / sum(w x)."""

        def __init__(self, rate):
            self.rate = torch.tensor(float(rate), dtype=torch.float64)
            self._reset()

        def _reset(self):
            self._sw = torch.zeros((), dtype=torch.float64)
            self._sx = torch.zeros((), dtype=torch.float64)

        def log_probability(self, X):
            _x = X[:, 0]
            return torch.where(_x >= 0, torch.log(self.rate) - self.rate * _x, torch.full_like(_x, -1e30))

        def summarize(self, X, sample_weight):
            _w = sample_weight.to(torch.float64)
            self._sw = self._sw + _w.sum()
            self._sx = self._sx + (_w * X[:, 0]).sum()

        def from_summaries(self):
            if float(self._sw) > 1e-12:
                self.rate = self._sw / self._sx.clamp(min=1e-12)
            self._reset()

        def sample(self, n):
            return torch.distributions.Exponential(self.rate).sample((n, 1))

        def n_parameters(self):
            return 1


    class Normal1D:
        """N(mean, var) with the usual weighted-moment M-step."""

        def __init__(self, mean, var):
            self.mean = torch.tensor([float(mean)], dtype=torch.float64)
            self.var = torch.tensor([float(var)], dtype=torch.float64)
            self._reset()

        def _reset(self):
            self._sw = torch.zeros((), dtype=torch.float64)
            self._s1 = torch.zeros(1, dtype=torch.float64)
            self._s2 = torch.zeros(1, dtype=torch.float64)

        def log_probability(self, X):
            return (-0.5 * ((X - self.mean) ** 2 / self.var + torch.log(2 * math.pi * self.var))).sum(1)

        def summarize(self, X, sample_weight):
            _w = sample_weight.to(torch.float64)
            self._sw = self._sw + _w.sum()
            self._s1 = self._s1 + (_w[:, None] * X).sum(0)
            self._s2 = self._s2 + (_w[:, None] * X * X).sum(0)

        def from_summaries(self):
            if float(self._sw) > 1e-12:
                self.mean = self._s1 / self._sw
                self.var = (self._s2 / self._sw - self.mean ** 2).clamp(min=0) + 1e-6
            self._reset()

        def sample(self, n):
            return self.mean + self.var.sqrt() * torch.randn(n, 1, dtype=torch.float64)

        def n_parameters(self):
            return 2

    return Exponential, Normal1D


@app.cell
def _(
    CATEGORICAL_PALETTE,
    Exponential,
    GenericMixture,
    MAX_ITER,
    Normal1D,
    SEED,
    plt,
    torch,
):
    _g = torch.Generator().manual_seed(SEED + 5)
    _waits = torch.distributions.Exponential(torch.tensor(1.0, dtype=torch.float64)).sample((600, 1))
    _burst = 6 + 0.7 * torch.randn(300, 1, generator=_g, dtype=torch.float64)
    X_wait = torch.cat([_waits, _burst])

    generic = GenericMixture(components=[Exponential(0.5), Normal1D(5.0, 1.0)], random_state=SEED, max_iter=MAX_ITER, device='cpu').fit(X_wait)  # the components hold CPU tensors
    print(f'weights {generic.weights_.cpu().numpy().round(3).tolist()}   (true 0.667 / 0.333)')
    print(f'exponential rate {float(generic.components_[0].rate):.2f} (true 1.0);  normal mean {float(generic.components_[1].mean):.2f}, '
          f'variance {float(generic.components_[1].var):.2f} (true 6.0 / 0.49)')
    print(f'converged in {generic.n_iter_} iterations; BIC = {generic.bic(X_wait):.1f}; {generic.sample(3)[0].shape[0]} samples drawn from the fitted components')

    _grid = torch.linspace(0, 9, 300, dtype=torch.float64)[:, None]
    _pdf_total = generic.score_samples(_grid).exp()
    plt.figure(figsize=(6.5, 4.5))
    plt.hist(X_wait[:, 0], bins=50, density=True, color='0.8', label='data')
    for _k, _name in enumerate(['exponential', 'normal']):
        plt.plot(_grid[:, 0], generic.weights_[_k] * generic.components_[_k].log_probability(_grid).exp(),
                 color=CATEGORICAL_PALETTE[_k], label=f'{_name} (weighted)')
    plt.plot(_grid[:, 0], _pdf_total, 'k', label='mixture')
    plt.legend()
    plt.title('GenericMixture of an exponential and a normal')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - `noise=True` adds a uniform component $\pi_0/V$: `predict` returns `-1` for noise, `noise_proba(X)` gives the probability, `weights_` sums to $1-\pi_0$, and
      the component covariances are no longer inflated by background points. `noise_init` accepts a boolean mask or `HDBSCAN` labels.
    - `missing='em'` maximises the likelihood of the observed entries: marginal densities in the E-step, conditional means and covariances in the M-step, `impute`
      for the filled-in data, and every consumer (`predict`, `score`, `bic`, ...) accepts rows with `NaN`.
    - `marginal(dims)` and `condition(x_obs, obs_dims)` turn a fitted `GaussianMixture` into marginal and conditional distributions; `mean()` is Gaussian mixture
      regression and the conditional can be multimodal where a regression mean is misleading.
    - `frozen_components` pins components at their initial parameters, and `sample_weight` counts rows (integer weights equal duplication; per-feature weights for factorising models).
    - `GenericMixture` fits any family that implements `log_probability`, `summarize` and `from_summaries` (or any `nn.Module` with `log_prob`), with `batch_size`,
      `sample_weight`, frozen components and information criteria included.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Next:** [17 -- Semi-Supervised and Positive-Unlabeled Fitting](17_semi_supervised_pu.ipynb) for another way of steering the fit, or back to the [tutorial overview](../tutorials/index.md).

    ## References

    - Banfield, J. D. & Raftery, A. E. (1993). Model-based Gaussian and non-Gaussian clustering. *Biometrics*, 49(3).
    - Ghahramani, Z. & Jordan, M. I. (1994). Supervised learning from incomplete data via an EM approach. *NeurIPS 6*.
    - Hunt, L. & Jorgensen, M. (2003). Mixture model clustering for mixed data with missing information. *CSDA*, 41.
    - Lin, T. I., Lee, J. C. & Ho, H. J. (2006). On fast supervised learning for normal mixture models with missing information. *Pattern Recognition*, 39.
    - McLachlan, G. & Krishnan, T. (2008). *The EM Algorithm and Extensions*, ch. 5.
    - Sung, H. G. (2004). *Gaussian Mixture Regression and Classification*. PhD thesis, Rice University.
    """)
    return


if __name__ == "__main__":
    app.run()
