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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Student-t Mixtures: Robust Clustering with Heavy Tails

    A Gaussian mixture is only as robust as its Gaussian components. Each component's mean and covariance are
    (weighted) averages of the points assigned to it, so a few far-away points -- contamination, sensor glitches,
    a stray cluster nobody asked for -- can drag a component's mean and inflate its covariance without limit.
    `StudentTMixture` (Peel & McLachlan, 2000) replaces every Gaussian component by a multivariate **Student-t**
    distribution. The t has the same bell-shaped core but polynomially decaying tails, and EM then *automatically
    downweights* points that are far from their component: a per-sample weight $u_{ik}$ shrinks toward zero with
    Mahalanobis distance.

    The model is built on the same `BaseMixture` machinery as `GaussianMixture` (so `fit`, `predict`, `score`,
    `bic`/`aic`/`icl`, `algorithm`, `temperature`, `step_size`, `n_init`, semi-supervised `labels=` all work the
    same way), so it is a drop-in replacement whenever you suspect heavy-tailed noise.

    **What you'll learn**

    - how a Student-t density differs from a Gaussian, and what the degrees of freedom $\nu$ control;
    - the scale-mixture representation that makes EM tractable, and the robustness weights $u_{ik}$ it produces;
    - the ECM updates for $\mu_k$, $\Sigma_k$ and $\nu_k$ that `StudentTMixture` performs;
    - how much better the model resists contamination than `GaussianMixture`, and how to flag outliers with `sample_weights`;
    - how to recover a known degrees-of-freedom value, choose between `per_component` / `shared` / fixed `dof`, and
      compare information criteria with a Gaussian fit.

    **Prerequisites:** [01 -- Gaussian Mixture Models](01_gmm.ipynb) and [02 -- The EM Algorithm](02_em_algorithm.ipynb).
    See also the [Other Mixture Families](../user-guide/other-mixture-families.md) user guide.
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
    from scipy import stats

    from tgmm import GaussianMixture, StudentTMixture, plot_gmm
    from tgmm.metrics import ClusteringMetrics
    from tgmm.plotting import CATEGORICAL_PALETTE

    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    return (
        CATEGORICAL_PALETTE,
        ClusteringMetrics,
        GaussianMixture,
        SEED,
        StudentTMixture,
        np,
        plot_gmm,
        plt,
        stats,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Heavy tails: Student-t versus Gaussian

    In $d$ dimensions the Student-t distribution with mean $\mu$, scale matrix $\Sigma$ and $\nu>0$ degrees of freedom has density

    $$
    t(x\mid\mu,\Sigma,\nu)=
    \frac{\Gamma\!\big(\tfrac{\nu+d}{2}\big)}{\Gamma\!\big(\tfrac{\nu}{2}\big)\,(\nu\pi)^{d/2}\,|\Sigma|^{1/2}}
    \Big(1+\frac{\delta}{\nu}\Big)^{-\frac{\nu+d}{2}},\qquad
    \delta=(x-\mu)^{\top}\Sigma^{-1}(x-\mu).
    $$

    Compare with the Gaussian, whose density decays like $e^{-\delta/2}$: the t decays only like a *power* of
    $\delta$. Small $\nu$ means very heavy tails ($\nu=1$ is the Cauchy distribution, with no mean); as
    $\nu\to\infty$ the t converges to the Gaussian. The plots show the one-dimensional densities, linearly and on a
    log scale, where the tail behaviour is unmistakable.

    !!! note "Scale matrix versus covariance"
        $\Sigma$ is the **scale matrix**, which is what `covariances_` stores. For $\nu>2$ the actual covariance of the
        distribution is $\frac{\nu}{\nu-2}\Sigma$; for $\nu\le2$ it is infinite.
    """)
    return


@app.cell
def _(CATEGORICAL_PALETTE, np, plt, stats):
    x = np.linspace(-8, 8, 801)
    nus = [1, 3, 10, np.inf]
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for _nu, _color in zip(nus, CATEGORICAL_PALETTE):
        pdf = stats.norm.pdf(x) if np.isinf(_nu) else stats.t.pdf(x, df=_nu)
        label = 'Gaussian ($\\nu=\\infty$)' if np.isinf(_nu) else f'$\\nu={_nu}$'
        _axes[0].plot(x, pdf, color=_color, label=label)
        _axes[1].semilogy(x, pdf, color=_color, label=label)
    _axes[0].set_title('Density')
    _axes[1].set_title('Log density: the Gaussian tail collapses, the t tails do not')
    _axes[1].set_ylim(1e-12, 1)
    for _ax in _axes:
        _ax.set_xlabel('$x$')
        _ax.legend()
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The scale-mixture representation

    The key to fitting a t-mixture is that a Student-t is a *Gaussian with a random precision*. Draw a latent
    multiplier $w\sim\mathrm{Gamma}(\nu/2,\ \nu/2)$ (shape / rate, so $\mathbb{E}[w]=1$) and then
    $x\mid w\sim\mathcal{N}(\mu,\ \Sigma/w)$; marginalising over $w$ gives exactly the t density above. A small $w$
    inflates the variance: those are the far-out points. `StudentTMixture.sample` uses this recipe directly.

    The scatter plots draw 600 points from a Gaussian and from a t with $\nu=2$ (same scale matrix), and colour the t
    points by their latent $\log w$. The t cloud has the same dense core but a halo of distant points, and the
    distant ones are precisely those with a small $w$.
    """)
    return


@app.cell
def _(CATEGORICAL_PALETTE, SEED, np, plt):
    rng = np.random.default_rng(SEED)
    n_demo, nu_demo = (600, 2.0)
    z = rng.standard_normal((n_demo, 2))
    w = rng.gamma(shape=nu_demo / 2, scale=2 / nu_demo, size=n_demo)  # rate nu/2 -> scale 2/nu
    x_gauss = z
    x_t = z / np.sqrt(w)[:, None]
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    _axes[0].scatter(*x_gauss.T, s=8, color=CATEGORICAL_PALETTE[0])
    _axes[0].set_title('Gaussian, $\\Sigma=I$')
    _sc = _axes[1].scatter(*x_t.T, s=8, c=np.log(w), cmap='viridis_r', vmin=-4, vmax=1.5)
    _axes[1].set_title(f'Student-t, $\\nu={nu_demo:g}$, $\\Sigma=I$ (colour: latent $\\log w$)')
    _fig.colorbar(_sc, ax=_axes[1], label='$\\log w$')
    for _ax in _axes:
        _ax.set_xlim(-12, 12)
        _ax.set_ylim(-12, 12)
        _ax.set_aspect('equal')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EM for a Student-t mixture

    **E-step.** Besides the usual responsibilities $r_{ik}$, the latent multiplier has a closed-form posterior given
    that $x_i$ came from component $k$: $w_{ik}\mid x_i\sim\mathrm{Gamma}\big(\tfrac{\nu_k+d}{2},\ \tfrac{\nu_k+\delta_{ik}}{2}\big)$,
    with mean

    $$
    u_{ik}=\mathbb{E}[w_{ik}\mid x_i,z_i=k]=\frac{\nu_k+d}{\nu_k+\delta_{ik}},\qquad
    \delta_{ik}=(x_i-\mu_k)^{\top}\Sigma_k^{-1}(x_i-\mu_k).
    $$

    This is the **robustness weight**: points at the centre ($\delta\ll\nu$) get $u>1$, points far away ($\delta\gg\nu$)
    get $u\to0$. The plot shows $u$ as a function of the Mahalanobis distance for several $\nu$ (here $d=2$): small
    $\nu$ means a faster fall-off, i.e. more suspicion of distant points.
    """)
    return


@app.cell
def _(CATEGORICAL_PALETTE, np, plt):
    delta = np.linspace(0, 60, 400)
    d_demo = 2
    plt.figure(figsize=(7, 4.5))
    for _nu, _color in zip([1, 3, 10, 100], CATEGORICAL_PALETTE):
        plt.plot(delta, (_nu + d_demo) / (_nu + delta), color=_color, label=f'$\\nu={_nu}$')
    plt.axhline(1.0, color='0.6', lw=1)
    plt.xlabel('squared Mahalanobis distance $\\delta$')
    plt.ylabel('robustness weight $u$')
    plt.title(f'$u=(\\nu+d)/(\\nu+\\delta)$ for $d={d_demo}$')
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **M-step (ECM).** With $n_k=\sum_ir_{ik}$ and $m_k=\sum_ir_{ik}u_{ik}$:

    $$
    \pi_k=\frac{n_k}{n},\qquad
    \mu_k=\frac{\sum_ir_{ik}u_{ik}\,x_i}{m_k},\qquad
    \Sigma_k=\frac{\sum_ir_{ik}u_{ik}\,(x_i-\mu_k)(x_i-\mu_k)^{\top}}{n_k}.
    $$

    These are the Gaussian updates with an extra weight $u_{ik}$ inside every sum, so a distant point contributes less
    the further away it is. As $\nu\to\infty$, $u_{ik}\to1$ and the updates reduce to the Gaussian ones. Each update
    uses the other parameters' values from the same iteration, which makes this an *ECM* (expectation conditional
    maximisation) algorithm. With `scale_update='ecme'` the covariance is divided by $m_k$ instead of $n_k$; at a
    fixed point the two coincide, because stationarity forces $\sum_ir_{ik}u_{ik}\delta_{ik}=d\,n_k$ and hence $m_k=n_k$.

    **Degrees of freedom.** For `dof='per_component'` each $\nu_k$ solves, every M-step, the one-dimensional equation

    $$
    1-\psi\!\Big(\frac{\nu}{2}\Big)+\ln\!\Big(\frac{\nu}{2}\Big)+c_k
    +\psi\!\Big(\frac{\nu+d}{2}\Big)-\ln\!\Big(\frac{\nu+d}{2}\Big)=0,\qquad
    c_k=\frac1{n_k}\sum_ir_{ik}\big(\ln u_{ik}-u_{ik}\big),
    $$

    ($\psi$ the digamma function), clamped to `dof_bounds`. `dof='shared'` pools $c$ across components and solves for one
    $\nu$; a numeric `dof` fixes it. The details, including where tgmm's step differs from the textbook one while
    sharing its fixed point, are in the [user guide](../user-guide/other-mixture-families.md).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contaminated data: `StudentTMixture` versus `GaussianMixture`

    Two well-separated clusters, plus 5% uniform-noise outliers scattered across the plot. A Gaussian component's
    covariance gets dragged toward any outliers it is forced to "explain"; a Student-t component's heavier tails let it
    downweight them instead.
    """)
    return


@app.cell
def _(SEED, plt, torch):
    _gen = torch.Generator().manual_seed(SEED)
    n_per = 250
    true_centers = torch.tensor([[5.0, 5.0], [-5.0, -5.0]])
    X_clean = torch.cat([torch.randn(n_per, 2, generator=_gen) * 0.6 + true_centers[0], torch.randn(n_per, 2, generator=_gen) * 0.6 + true_centers[1]])
    n_out = int(0.05 * 2 * n_per)
    outliers = (torch.rand(n_out, 2, generator=_gen) - 0.5) * 40
    X = torch.cat([X_clean, outliers])
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:-n_out, 0], X[:-n_out, 1], s=10, label='clean')
    plt.scatter(X[-n_out:, 0], X[-n_out:, 1], s=10, c='red', marker='x', label='outliers')
    plt.legend()
    plt.title(f'{2 * n_per} clean points + {n_out} outliers (5%)')
    plt.gcf()
    return X, n_out, n_per, true_centers


@app.cell
def _(GaussianMixture, SEED, StudentTMixture, X, torch, true_centers):
    gmm = GaussianMixture(n_components=2, random_state=SEED, n_init=5)
    gmm.fit(X)

    stm = StudentTMixture(n_components=2, covariance_type='full', dof='per_component',
                          random_state=SEED, n_init=5)
    stm.fit(X)


    def match_err(means, true_centers=true_centers):
        '''Mean distance between fitted and true centres, under the best matching.'''
        d = torch.cdist(means, true_centers)
        return min(d[0, 0] + d[1, 1], d[0, 1] + d[1, 0]).item() / 2


    print(f'GaussianMixture mean error from truth:  {match_err(gmm.means_):.3f}')
    print(f'StudentTMixture mean error from truth:  {match_err(stm.means_):.3f}')
    print(f'StudentTMixture fitted dof_: {stm.dof_.tolist()}')
    return gmm, match_err, stm


@app.cell
def _(X, gmm, match_err, plot_gmm, plt, stm):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_gmm(X, gmm=gmm, ax=_axes[0])
    _axes[0].set_title(f'GaussianMixture (mean error {match_err(gmm.means_):.2f})')
    plot_gmm(X, gmm=stm, ax=_axes[1])
    _axes[1].set_title(f'StudentTMixture (mean error {match_err(stm.means_):.2f})')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the ellipses: the Gaussian fit stretches both components to swallow the outliers, while the t fit stays
    on the clusters. Note the fitted `dof_` -- small values say the model found heavy tails in this data. Because
    `covariances_` is the *scale matrix*, the t ellipses describe the bulk of each cluster, not its (much larger)
    actual covariance.

    ### Flagging outliers with `sample_weights`

    `sample_weights(X)` returns each point's robustness weight $u_{i\hat z_i}$ at its predicted component -- small
    values are points the t-mixture effectively treated as outliers.
    """)
    return


@app.cell
def _(X, plt, stm):
    weights = stm.sample_weights(X)
    plt.figure(figsize=(5.5, 5))
    _sc = plt.scatter(X[:, 0], X[:, 1], c=weights, cmap='viridis_r', s=12, vmin=0, vmax=1)
    plt.colorbar(_sc, label='sample weight $u_{ik}$ (low = likely outlier)')
    plt.title('StudentTMixture outlier weights')
    plt.gcf()
    return (weights,)


@app.cell
def _(n_out, weights):
    print(f'Mean weight, clean points:      {weights[:-n_out].mean():.3f}')
    print(f'Mean weight, injected outliers: {weights[-n_out:].mean():.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How much contamination can it take?

    Repeat the experiment while increasing the fraction of uniform outliers from 0% to 20% and track how far each
    model's fitted centres end up from the truth. With no contamination the two models are equally good; as the
    fraction grows, the Gaussian centres drift while the t centres stay close to the truth.
    """)
    return


@app.cell
def _(
    GaussianMixture,
    SEED,
    StudentTMixture,
    match_err,
    n_per,
    plt,
    torch,
    true_centers,
):
    fractions = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
    err_gauss, err_t = ([], [])
    for _frac in fractions:
        _gen = torch.Generator().manual_seed(1)
        _clean = torch.cat([torch.randn(n_per, 2, generator=_gen) * 0.6 + true_centers[0], torch.randn(n_per, 2, generator=_gen) * 0.6 + true_centers[1]])
        _n_o = int(_frac * 2 * n_per)
        _noise = (torch.rand(_n_o, 2, generator=_gen) - 0.5) * 40
        _X_f = torch.cat([_clean, _noise])
        _g = GaussianMixture(n_components=2, random_state=SEED, n_init=3).fit(_X_f)
        _t = StudentTMixture(n_components=2, random_state=SEED, n_init=3).fit(_X_f)
        err_gauss.append(match_err(_g.means_))
        err_t.append(match_err(_t.means_))
    plt.figure(figsize=(7, 4.5))
    plt.plot([100 * f for f in fractions], err_gauss, marker='o', label='GaussianMixture')
    plt.plot([100 * f for f in fractions], err_t, marker='o', label='StudentTMixture')
    plt.xlabel('fraction of uniform outliers (%)')
    plt.ylabel('mean error of fitted centres')
    plt.title('Robustness to contamination')
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recovering the degrees of freedom

    Because `dof_` is estimated by maximum likelihood, it should recover the tail heaviness the data was generated
    with. We simulate two clusters whose points are genuine Student-t draws with $\nu=3$ (via the scale-mixture
    recipe) and fit `per_component`, `shared`, and a fixed near-Gaussian `dof=1e4`. We also fit clean Gaussian data, where the estimate
    should drift towards the upper bound.
    """)
    return


@app.cell
def _(SEED, StudentTMixture, torch, true_centers):
    def sample_t_clusters(nu, n_each, scale, seed):
        gen = torch.Generator().manual_seed(seed)
        parts, labels = ([], [])
        for k, c in enumerate(true_centers):
            z = torch.randn(n_each, 2, generator=gen)
            w = torch._standard_gamma(torch.full((n_each, 1), nu / 2), generator=gen) * (2.0 / nu)
            parts.append(c + scale * z / w.sqrt())
            labels.append(torch.full((n_each,), k))
        return (torch.cat(parts), torch.cat(labels))
    X_t3, y_t3 = sample_t_clusters(nu=3.0, n_each=1000, scale=0.6, seed=3)
    X_gauss, y_gauss = sample_t_clusters(nu=1000000.0, n_each=1000, scale=0.6, seed=4)  # effectively Gaussian
    _rows = []
    for _name, _data in [('t, nu = 3', X_t3), ('Gaussian', X_gauss)]:
        for _dof in ['per_component', 'shared', 10000.0]:
            _m = StudentTMixture(n_components=2, dof=_dof, random_state=SEED, n_init=2).fit(_data)
            _rows.append((_name, _dof, _m.dof_.numpy().round(2), round(_m.bic(_data), 1)))
    for _r in _rows:
        print(f'{_r[0]:<10} dof={str(_r[1]):<14} fitted dof_={_r[2]}   BIC={_r[3]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On the $\nu=3$ data the fitted `dof_` is close to the true value, and the free-$\nu$ models reach a lower BIC than
    the fixed near-Gaussian one. On Gaussian data `dof_` runs to large values (tens or more, drifting towards `dof_bounds[1]`): the model
    reports that the component is indistinguishable from a Gaussian, and `dof=1e4` is then an equally good fit with
    fewer parameters.

    ## ECM versus ECME

    The `scale_update` option chooses how the scale matrix is normalised ($n_k$ for `'ecm'`, $m_k$ for `'ecme'`). Both reach the same
    fixed point; they take different paths to it.
    """)
    return


@app.cell
def _(SEED, StudentTMixture, X):
    for _update in ['ecm', 'ecme']:
        _m = StudentTMixture(n_components=2, scale_update=_update, random_state=SEED, tol=1e-08, max_iter=500).fit(X)
        print(f"scale_update='{_update}': iterations={_m.n_iter_:>3}  log-likelihood/sample={_m.lower_bound_:.6f}  dof_={_m.dof_.numpy().round(3)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing with a Gaussian fit

    Gaussian and Student-t mixtures both define densities on $\mathbb{R}^d$ with respect to the same (Lebesgue)
    measure, so their log-likelihoods and information criteria **are** directly comparable. (This is not true across
    families with different base measures, such as the von Mises-Fisher mixture on the sphere -- see the
    [next tutorial](17_von_mises_fisher.ipynb).) On the contaminated data the t model is favoured even after paying
    for its extra $K$ degrees-of-freedom parameters:
    """)
    return


@app.cell
def _(ClusteringMetrics, X, gmm, n_per, stm, torch):
    print(f'GaussianMixture   BIC = {gmm.bic(X):10.1f}   AIC = {gmm.aic(X):10.1f}')
    print(f'StudentTMixture   BIC = {stm.bic(X):10.1f}   AIC = {stm.aic(X):10.1f}')

    ari_g = ClusteringMetrics.adjusted_rand_score(torch.cat([torch.zeros(n_per), torch.ones(n_per)]).long(), gmm.predict(X)[:2 * n_per])
    ari_t = ClusteringMetrics.adjusted_rand_score(torch.cat([torch.zeros(n_per), torch.ones(n_per)]).long(), stm.predict(X)[:2 * n_per])
    print(f'ARI on the clean points:  Gaussian {ari_g:.3f}   Student-t {ari_t:.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - A Student-t is a Gaussian with a random precision; its heavy tails are controlled by the degrees of freedom $\nu$
      ($\nu\to\infty$ recovers the Gaussian).
    - In EM this appears as a per-sample, per-component weight $u_{ik}=(\nu_k+d)/(\nu_k+\delta_{ik})$ that shrinks with
      Mahalanobis distance; the M-step is the Gaussian one with $u_{ik}$ inside the sums, and each $\nu_k$ is found by a
      one-dimensional root-finding step.
    - `StudentTMixture` is a drop-in, more outlier-robust alternative to `GaussianMixture`: its centres stay on the clusters as
      contamination grows, and `sample_weights` points out which observations it distrusted.
    - `dof_` recovers the tail heaviness of the data; a large `dof_` says the data are effectively Gaussian.
    - `covariances_` holds the scale matrix, not the covariance; information criteria are comparable with a
      `GaussianMixture` fit on the same data.

    **Next:** [17 -- Von Mises-Fisher Mixtures](17_von_mises_fisher.ipynb) for directional data, or
    [18 -- Semi-Supervised and Positive-Unlabeled Fitting](18_semi_supervised_pu.ipynb).
    """)
    return


if __name__ == "__main__":
    app.run()
