# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "tgmm",
#     "torch; sys_platform != 'emscripten'",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "pandas",
#     "seaborn",
#     "tqdm",
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
    return (MAX_ITER,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Constraints, Frozen Components and Sample Weights

    Sometimes part of the model is *known*: the class proportions come from a census, a reference population has a known mean, the means
    must stay inside the measurement range, the variances must stay below a noise ceiling. This tutorial shows how to tell
    `GaussianMixture` what you know, why EM still works when you do (the log-likelihood stays monotone), and how to fit a *weighted* sample.

    **What you'll learn**

    1. the six constructor arguments `fixed_weights`, `weight_bounds`, `mean_bounds`, `covariance_bounds`, `frozen_means`, `frozen_covariances` (and the older `frozen_components`),
    2. why the constrained M-step is an **ECM** step with *exact* block updates: water-filling for the weights, clipping or coordinate ascent for the means, variance / eigenvalue clipping for the covariances,
    3. how to read `active_constraints_`, and how `bic` / `aic` count fixed parameters,
    4. `fit(X, sample_weight=w)`: integer weights are row duplication, weight $0$ is dropping a row, and an $(n, d)$ weight matrix marginalises a feature out of a row.

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb) (EM, the E- and M-step), [02 -- Covariance Types](02_covariance_types.ipynb).
    Constraints are not available in Gibbs mode ([11 -- Dirichlet Process Mixtures](11_dpgmm.ipynb)), nor together with a mean or covariance prior ([04 -- Priors](04_priors.ipynb)).
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
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy import optimize

    from tgmm import GaussianMixture, plot_gmm
    from tutorial_data import default_blobs, make_default_data
    from tgmm.constraints import box_qp_coordinate_ascent, clip_eigenvalues, water_fill_weights

    # Expected here: FAST builds cap max_iter, the stepping cell runs one EM iteration per fit, and the example bounds
    # deliberately exclude the true starting parameters (tgmm then warns that it projected them into the feasible set).
    warnings.filterwarnings('ignore', category=UserWarning, module='tgmm')

    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float64)
    # marimo embeds figures at 2 x figure.dpi, and tgmm's plot theme sets figure.dpi=300; keep outputs small.
    plt.rcParams['figure.dpi'] = 75
    return (
        GaussianMixture,
        Rectangle,
        SEED,
        box_qp_coordinate_ascent,
        clip_eigenvalues,
        default_blobs,
        make_default_data,
        np,
        optimize,
        pd,
        plot_gmm,
        plt,
        torch,
        water_fill_weights,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The data and an unconstrained reference fit

    We use the default four blobs of the earlier tutorials. To make "component $k$" mean "blob $k$" (so that a constraint on component $k$ is a statement about blob $k$),
    every fit below starts from the true centres, covariances and weights, passed as tensors. The reference model is the plain maximum-likelihood fit.
    """)
    return


@app.cell
def _(SEED, default_blobs, make_default_data, np, torch):
    centers, covs, n_samples = default_blobs()  # the true blobs, used below as the starting parameters
    _X, _y = make_default_data(SEED)
    X = _X.double()
    X_np = X.numpy()
    y_true = _y.numpy()
    n_total = X.shape[0]
    init_means = torch.tensor(np.stack(centers), dtype=torch.float64)
    init_covs = torch.tensor(np.stack(covs), dtype=torch.float64)
    init_weights = torch.tensor(n_samples, dtype=torch.float64) / n_total
    return X, X_np, init_covs, init_means, init_weights, n_total, y_true


@app.cell
def _(GaussianMixture, MAX_ITER, X, init_covs, init_means, init_weights):
    def fit_gmm(X_fit=X, **kwargs):
        """Fit a 4-component model started from the true parameters; extra keyword arguments add constraints."""
        options = dict(n_components=4, init_means=init_means, init_covariances=init_covs, init_weights=init_weights,
                       max_iter=MAX_ITER, tol=1e-9)
        options.update(kwargs)
        return GaussianMixture(**options).fit(X_fit)

    gmm_free = fit_gmm()
    return fit_gmm, gmm_free


@app.cell
def _(gmm_free, np, pd):
    pd.DataFrame({
        'weight': gmm_free.weights_.numpy().round(3),
        'mean x': gmm_free.means_[:, 0].numpy().round(3),
        'mean y': gmm_free.means_[:, 1].numpy().round(3),
        'covariance eigenvalues': [np.linalg.eigvalsh(c).round(3).tolist() for c in gmm_free.covariances_.numpy()],
    }, index=pd.RangeIndex(1, 5, name='component'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why EM still works: ECM with exact block updates

    The E-step is unchanged. With responsibilities $r_{ik}$, write $n_k=\sum_ir_{ik}$, $\bar y_k=\sum_ir_{ik}x_i/n_k$ and $W_k=\sum_ir_{ik}(x_i-\bar y_k)(x_i-\bar y_k)^\top$.
    The expected complete-data log-likelihood is

    $$
    Q=\sum_k n_k\log\pi_k-\tfrac12\sum_k\Big[n_k(\mu_k-\bar y_k)^\top\Sigma_k^{-1}(\mu_k-\bar y_k)+n_k\log|\Sigma_k|+\operatorname{tr}\big(\Sigma_k^{-1}W_k\big)\Big]+\text{const}.
    $$

    $Q$ splits into one term per parameter block: the weights $\pi$, the means $\mu_k$ and the covariances $\Sigma_k$. A constrained M-step maximises $Q$ over **one block at a time, each exactly,
    given the others** (a *Conditional* Maximisation, hence ECM). Each block update cannot decrease $Q$, so neither does the observed-data log-likelihood. A *fixed* value is simply a block that is not updated.
    The three exact solutions are derived next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weights: `fixed_weights` and `weight_bounds`

    For the weights the problem is $\max_\pi\sum_kn_k\log\pi_k$ subject to $\sum_k\pi_k=1$ and $lo_k\le\pi_k\le hi_k$. The Lagrangian (multiplier $\nu$ for the sum constraint) gives

    $$
    \pi_k=\operatorname{clip}\!\big(n_k/\nu,\ lo_k,\ hi_k\big),\qquad \nu>0\ \text{such that}\ \sum_k\operatorname{clip}(n_k/\nu,lo_k,hi_k)=1 .
    $$

    The left-hand side is decreasing in $\nu$, so the root is bracketed by doubling and found by bisection (the "water level" $1/\nu$ rises until the total is 1). If the unconstrained answer $n_k/n$ is already feasible
    it is returned unchanged. `tgmm.constraints.water_fill_weights` implements exactly this, and the cell below checks it against a general-purpose optimiser on counts that make some bounds bind:
    """)
    return


@app.cell
def _(n_samples, np, optimize, torch, water_fill_weights):
    _counts = torch.tensor(n_samples, dtype=torch.float64)
    _lo = torch.full((4,), 0.10)
    _hi = torch.full((4,), 0.32)
    pi_exact = water_fill_weights(_counts, _lo, _hi).numpy()
    _res = optimize.minimize(lambda p: -(np.array(n_samples) * np.log(p)).sum(), x0=np.full(4, 0.25),
                             bounds=[(0.10, 0.32)] * 4, constraints=[{'type': 'eq', 'fun': lambda p: p.sum() - 1}],
                             method='SLSQP', options={'ftol': 1e-14})
    print('counts n_k          :', n_samples, ' (unconstrained pi = n_k / n =', np.round(np.array(n_samples) / sum(n_samples), 3), ')')
    print('water-filling pi    :', pi_exact.round(6), ' sum =', round(float(pi_exact.sum()), 12))
    print('SLSQP (generic)     :', _res.x.round(6))
    print('max |difference|    :', f'{np.abs(pi_exact - _res.x).max():.2e}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now inside a fit. `weight_bounds=(0.1, 0.32)` forces every weight into $[0.1,0.32]$, and `fixed_weights` holds the weights at given values for the whole fit
    (positive, summing to one, and not combined with `weight_bounds`). `active_constraints_` reports which bounds bind at the solution.
    The band is chosen so that the problem is not degenerate (with four components, $0.1+3\times0.3=1$ would leave a single feasible point): the small blob is lifted to the lower bound,
    the largest blobs are pushed against the upper bound (the table shows which bounds actually bind in the fit), and the others share what is left.
    """)
    return


@app.cell
def _(fit_gmm, gmm_free, pd, torch):
    gmm_wb = fit_gmm(weight_bounds=(0.1, 0.32))
    gmm_fw = fit_gmm(fixed_weights=torch.tensor([0.25, 0.25, 0.25, 0.25]))
    _active = gmm_wb.active_constraints_
    pd.DataFrame({
        'free ML fit': gmm_free.weights_.numpy().round(4),
        'weight_bounds=(0.1, 0.32)': gmm_wb.weights_.numpy().round(4),
        'at lower bound': _active['weights_at_lower'].numpy(),
        'at upper bound': _active['weights_at_upper'].numpy(),
        'fixed_weights=0.25': gmm_fw.weights_.numpy().round(4),
    }, index=pd.RangeIndex(1, 5, name='component'))
    return gmm_fw, gmm_wb


@app.cell
def _(gmm_free, gmm_fw, gmm_wb, plt):
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _w = 0.26
    for _i, (_name, _g) in enumerate([('free', gmm_free), ('weight_bounds=(0.1, 0.32)', gmm_wb), ('fixed_weights=0.25', gmm_fw)]):
        _ax.bar([k + (_i - 1) * _w for k in range(4)], _g.weights_.numpy(), width=_w, label=_name)
    _ax.axhspan(0.1, 0.32, color='grey', alpha=0.12, label='allowed band')
    _ax.set_xticks(range(4))
    _ax.set_xticklabels([f'component {k + 1}' for k in range(4)])
    _ax.set_ylabel('mixing weight $\\pi_k$')
    _ax.legend(fontsize=8)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Means: `mean_bounds`

    The part of $Q$ that involves $\mu_k$ is $-\tfrac12n_k(\mu_k-\bar y_k)^\top\Sigma_k^{-1}(\mu_k-\bar y_k)$, to be maximised over the box $lo\le\mu_k\le hi$ (each bound a scalar, a $(d,)$ or a $(K,d)$ tensor; use $\pm\infty$ for one side).

    * For **diagonal and spherical** covariances $\Sigma_k^{-1}$ is diagonal, the problem separates by coordinate, and the exact solution is $\mu_k=\operatorname{clip}(\bar y_k,lo,hi)$.
    * For **full** covariances it is a box-constrained quadratic program. With $P=\Sigma_k^{-1}$, the exact conditional maximiser in coordinate $j$ is
      $\mu_j=\operatorname{clip}\!\big(\bar y_j-\sum_{l\ne j}P_{jl}(\mu_l-\bar y_l)/P_{jj},\ lo_j,\ hi_j\big)$, so a few coordinate-ascent sweeps never decrease $Q$ and converge to the box-QP optimum.

    The cell checks `box_qp_coordinate_ascent` against `scipy`'s bounded L-BFGS-B on a strongly correlated 3-d covariance where the clipped point is *not* the optimum:
    """)
    return


@app.cell
def _(box_qp_coordinate_ascent, np, optimize, torch):
    _rng = np.random.default_rng(1)
    _A = _rng.standard_normal((3, 3))
    _P = _A @ _A.T + 0.3 * np.eye(3)                      # a positive-definite precision matrix with strong correlations
    _ybar = np.array([2.0, -1.5, 0.5])
    _lo, _hi = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    _obj = lambda x: 0.5 * (x - _ybar) @ _P @ (x - _ybar)  # = -Q up to a constant, to be minimised
    _ours = box_qp_coordinate_ascent(torch.tensor(_ybar)[None], torch.tensor(_P)[None], torch.tensor(_lo)[None],
                                     torch.tensor(_hi)[None], torch.zeros(1, 3), max_sweeps=500)[0].numpy()
    _ref = optimize.minimize(_obj, x0=np.zeros(3), bounds=list(zip(_lo, _hi)), method='L-BFGS-B', options={'ftol': 1e-15, 'gtol': 1e-12}).x
    print('coordinate ascent   :', _ours.round(5), ' objective =', round(float(_obj(_ours)), 8))
    print('L-BFGS-B (generic)  :', _ref.round(5), ' objective =', round(float(_obj(_ref)), 8))
    print('plain clipping      :', np.clip(_ybar, _lo, _hi).round(5), ' objective =', round(float(_obj(np.clip(_ybar, _lo, _hi))), 8), ' (worse)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a fit, `mean_bounds=(lo, hi)` constrains every component. Here every mean must lie in $[-1,1.5]^2$ -- the true centres of three blobs are outside that box:
    """)
    return


@app.cell
def _(fit_gmm, pd, torch):
    MEAN_BOX = (torch.tensor([-1.0, -1.0]), torch.tensor([1.5, 1.5]))
    gmm_mb = fit_gmm(mean_bounds=MEAN_BOX)
    _active = gmm_mb.active_constraints_
    pd.DataFrame({
        'mean x': gmm_mb.means_[:, 0].numpy().round(3),
        'mean y': gmm_mb.means_[:, 1].numpy().round(3),
        'x at upper': _active['means_at_upper'][:, 0].numpy(),
        'y at upper': _active['means_at_upper'][:, 1].numpy(),
        'y at lower': _active['means_at_lower'][:, 1].numpy(),
    }, index=pd.RangeIndex(1, 5, name='component'))
    return MEAN_BOX, gmm_mb


@app.cell
def _(MEAN_BOX, Rectangle, X_np, gmm_free, gmm_mb, plot_gmm, plt):
    _fig, _axs = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    plot_gmm(X_np, gmm=gmm_free, ax=_axs[0], title='free maximum-likelihood fit', ellipse_std_devs=[2], point_size=3, legend=False)
    plot_gmm(X_np, gmm=gmm_mb, ax=_axs[1], title='mean_bounds: box [-1, 1.5]^2', ellipse_std_devs=[2], point_size=3, legend=False)
    _lo, _hi = (b.numpy() for b in MEAN_BOX)
    _axs[1].add_patch(Rectangle(_lo, *(_hi - _lo), fill=False, ls='--', lw=1.5, color='k'))
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Means that want to leave the box sit on its boundary (the table above), and the weights and covariances are re-estimated around the constrained means.
    A bounded fit is a **constrained maximum-likelihood** estimate: where a bound binds it is not an unbiased estimate of the unconstrained value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Covariances: `covariance_bounds`

    `covariance_bounds=(lo, hi)` bounds the **variances** for `diag`, `spherical`, `tied_diag` and `tied_spherical`, and the **eigenvalues** of $\Sigma_k$ for `full` and `tied_full` ($lo\ge0$).
    With $T_k$ the scatter about the *new* mean (so the order matters: the mean block is updated first), the unconstrained estimate is $T_k/n_k$ and `reg_covar` is added first, so `lo` is a hard floor.

    * **A scalar variance** $s$ minimises $n\log s+t/s$, which is unimodal with minimum at $t/n$; the bounded optimum is therefore $\operatorname{clip}(t/n,lo,hi)$.
    * **A matrix.** Minimise $n\log|\Sigma|+\operatorname{tr}(\Sigma^{-1}T)$ subject to $lo\,I\preceq\Sigma\preceq hi\,I$. By von Neumann's trace inequality the minimiser shares the eigenvectors of $T$, so the problem reduces to the scalar one
      for each eigenvalue: **clip the eigenvalues** of $T/n$ to $[lo,hi]$ (see also Won, Lim, Kim & Rubin, *Condition-number-regularized covariance estimation*, 2013).

    Both claims can be checked numerically. First the scalar case on a grid; then the matrix case against a generic constrained optimiser over all $2\times2$ symmetric matrices:
    """)
    return


@app.cell
def _(clip_eigenvalues, np, optimize, torch):
    _n, _t, _lo, _hi = 50.0, 120.0, 0.5, 2.0               # t/n = 2.4 lies above the upper bound
    _grid = np.linspace(_lo, _hi, 200001)
    _s_grid = _grid[np.argmin(_n * np.log(_grid) + _t / _grid)]
    print(f'scalar: grid minimiser = {_s_grid:.5f}   clip(t/n) = {np.clip(_t / _n, _lo, _hi):.5f}')

    _rot = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    _T = _rot @ np.diag([0.2, 5.0]) @ _rot.T               # a scatter matrix with a tiny and a large eigenvalue
    _n = 10.0
    _S = _T / _n
    _lo_m, _hi_m = 0.1, 0.35
    _cost = lambda s: _n * np.log(max(s[0] * s[2] - s[1] ** 2, 1e-12)) + np.trace(np.linalg.inv([[s[0], s[1]], [s[1], s[2]]]) @ _T)
    def _eig(s):                                            # eigenvalues of the symmetric matrix [[a, b], [b, c]]
        _m, _h = (s[0] + s[2]) / 2, np.sqrt(((s[0] - s[2]) / 2) ** 2 + s[1] ** 2)
        return np.array([_m - _h, _m + _h])
    _cons = [{'type': 'ineq', 'fun': lambda s: _eig(s) - _lo_m}, {'type': 'ineq', 'fun': lambda s: _hi_m - _eig(s)}]
    _ref = optimize.minimize(_cost, x0=[0.2, 0.0, 0.2], constraints=_cons, method='SLSQP', options={'ftol': 1e-14, 'maxiter': 500})
    _ours = clip_eigenvalues(torch.tensor(_S), _lo_m, _hi_m).numpy()
    print('matrix: eigenvalues of T/n          :', np.linalg.eigvalsh(_S).round(4), ' -> bounds [', _lo_m, ',', _hi_m, ']')
    print('        clip_eigenvalues eigenvalues:', np.linalg.eigvalsh(_ours).round(4), ' cost =', round(float(_cost([_ours[0, 0], _ours[0, 1], _ours[1, 1]])), 6))
    print('        generic SLSQP     eigenvalues:', _eig(_ref.x).round(4), ' cost =', round(float(_ref.fun), 6))
    return


@app.cell(hide_code=True)
def _(init_covs, mo, np):
    _eig = [np.linalg.eigvalsh(_c).round(3).tolist() for _c in init_covs.numpy()]
    mo.md(f"""
    In a fit, `covariance_bounds=(0.05, 0.5)` confines every eigenvalue of every full covariance to $[0.05,0.5]$. The true eigenvalues of the four blobs are {_eig[0]}, {_eig[1]}, {_eig[2]} and {_eig[3]}.
    Eigenvalues above the cap (component 1, and the long axes of components 3 and 4) are clipped to $0.5$, and the nearly degenerate direction of component 4 (true eigenvalue {_eig[3][0]}) is lifted to the lower bound:
    """)
    return


@app.cell
def _(fit_gmm, np, pd):
    gmm_cb = fit_gmm(covariance_bounds=(0.05, 0.5))
    _active = gmm_cb.active_constraints_
    pd.DataFrame({
        'eigenvalues of Sigma_k': [np.linalg.eigvalsh(c).round(3).tolist() for c in gmm_cb.covariances_.numpy()],
        'at upper bound (0.5)': [m.tolist() for m in _active['covariances_at_upper'].numpy()],
        'at lower bound (0.05)': [m.tolist() for m in _active['covariances_at_lower'].numpy()],
    }, index=pd.RangeIndex(1, 5, name='component'))
    return (gmm_cb,)


@app.cell(hide_code=True)
def _(X, gmm_cb, gmm_free, init_covs, mo, np):
    _lo, _hi = 0.05, 0.5
    _true = [np.linalg.eigvalsh(_c) for _c in init_covs.numpy()]
    _fit = [np.linalg.eigvalsh(_c) for _c in gmm_cb.covariances_.numpy()]
    _assigned = np.bincount(gmm_cb.predict(X).numpy(), minlength=4)
    _inside = [_k for _k in range(4) if _true[_k].min() >= _lo and _true[_k].max() <= _hi and _fit[_k].max() >= _hi - 1e-6]
    _msg = ""
    for _k in _inside:
        _msg += (f"Component {_k + 1} is the instructive case: its true eigenvalues {_true[_k].round(3).tolist()} lie inside the band, yet it ends at the cap too, and its weight grows from {gmm_free.weights_[_k]:.3f} to "
                 f"{gmm_cb.weights_[_k]:.3f} ({_assigned[_k]} points are assigned to it, against {int(round(float(gmm_free.weights_[_k]) * X.shape[0]))} in the free fit). With its neighbours capped it takes over the points they can no longer explain: "
                 f"a bound changes the whole constrained maximum-likelihood solution, not only the component it touches. ")
    mo.md(_msg)
    return


@app.cell
def _(X_np, gmm_cb, gmm_free, plot_gmm, plt):
    _fig, _axs = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    plot_gmm(X_np, gmm=gmm_free, ax=_axs[0], title='free maximum-likelihood fit', ellipse_std_devs=[2], point_size=3, legend=False)
    plot_gmm(X_np, gmm=gmm_cb, ax=_axs[1], title='covariance_bounds=(0.05, 0.5)', ellipse_std_devs=[2], point_size=3, legend=False)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frozen means, covariances and components

    Instead of bounding a block you can **hold it at its initial value**. `frozen_means=[k, ...]` keeps those means at `init_means` (a tensor), `frozen_covariances=[k, ...]` keeps those covariances at `init_covariances`, and the older
    `frozen_components=[k, ...]` freezes both (the mixing weights of frozen components still update; priors never shrink them). This anchors a known "reference" population while the rest of the mixture is learned; it is what
    `PositiveUnlabeledMixture(anchor='fixed')` uses ([17 -- Semi-Supervised and PU Fitting](17_semi_supervised_pu.ipynb)).

    A frozen mean is restored *before* the covariance update, so that covariance is estimated from the scatter about the fixed mean, by the parallel-axis identity

    $$T_k=W_k+n_k(\bar y_k-\mu_k)(\bar y_k-\mu_k)^\top,\qquad \Sigma_k=T_k/n_k .$$

    A frozen covariance leaves the mean update unchanged. Below, component 1 has a frozen mean that is deliberately *wrong* (the blob is centred at $(0,2)$), component 3 a frozen covariance, and component 2 is frozen completely.
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, X, init_covs, init_means, init_weights, np, pd, torch):
    _start_means = init_means.clone()
    _start_means[0] = torch.tensor([0.5, 1.5])                 # a deliberately wrong, frozen mean for component 1
    gmm_frozen = GaussianMixture(n_components=4, init_means=_start_means, init_covariances=init_covs, init_weights=init_weights,
                                 max_iter=MAX_ITER, tol=1e-12, frozen_means=(0,), frozen_covariances=(2,), frozen_components=(1,)).fit(X)
    pd.DataFrame({
        'frozen': ['mean', 'mean + covariance', 'covariance', '-'],
        'mean': [m.round(3).tolist() for m in gmm_frozen.means_.numpy()],
        'initial mean': [m.round(3).tolist() for m in _start_means.numpy()],
        'covariance diagonal': [np.diag(c).round(3).tolist() for c in gmm_frozen.covariances_.numpy()],
        'initial covariance diagonal': [np.diag(c).round(3).tolist() for c in init_covs.numpy()],
        'weight': gmm_frozen.weights_.numpy().round(3),
    }, index=pd.RangeIndex(1, 5, name='component'))
    return (gmm_frozen,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The frozen parameters are exactly their initial values; the covariance of component 1 (frozen mean, free covariance) is larger than the blob's true $I$ because it is measured about the wrong centre.
    The parallel-axis identity can be checked directly from the final responsibilities: the fitted covariance of component 1 equals $T_k/n_k$ computed about the frozen mean plus the small `reg_covar` ridge (default $10^{-6}$); the printed gap is at that level.
    """)
    return


@app.cell
def _(X, gmm_frozen, torch):
    _r = gmm_frozen.predict_proba(X)[:, 0:1]                   # responsibilities of component 1
    _d = X - gmm_frozen.means_[0]
    _T = (_r * _d).T @ _d                                      # scatter about the FROZEN mean
    _gap = (gmm_frozen.covariances_[0] - _T / _r.sum() - gmm_frozen.reg_covar * torch.eye(2, dtype=torch.float64)).abs().max().item()
    print(f'max |Sigma_1 - T_1 / n_1 - reg_covar I| = {_gap:.2e}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The log-likelihood stays monotone

    The claim of the ECM argument is testable: run EM **one iteration at a time** (`max_iter=1`, `warm_start=True`) with *all three* kinds of bounds on at once and record the data log-likelihood after every step.
    Starting from a random initialisation, the sequence must never decrease:
    """)
    return


@app.cell
def _(GaussianMixture, SEED, X, n_total, np, torch):
    _stepper = GaussianMixture(n_components=4, init_means='random', max_iter=1, tol=1e-30, warm_start=True, random_state=SEED,
                               weight_bounds=(0.1, 0.4), mean_bounds=(torch.tensor([-1.0, -1.0]), torch.tensor([1.5, 1.5])),
                               covariance_bounds=(0.05, 0.5))
    loglik_trace = []
    for _step in range(40):
        _stepper.fit(X)
        loglik_trace.append(float(_stepper.score(X)) * n_total)
    loglik_trace = np.array(loglik_trace)
    print(f'{len(loglik_trace)} constrained EM steps: log-likelihood {loglik_trace[0]:.2f} -> {loglik_trace[-1]:.2f}; smallest step = {np.diff(loglik_trace).min():+.2e} ' + ('(never negative)' if np.diff(loglik_trace).min() >= 0 else '(a decrease this small is floating-point rounding of a sum over all rows, not a real decrease)'))
    return (loglik_trace,)


@app.cell
def _(loglik_trace, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ax.plot(range(1, len(loglik_trace) + 1), loglik_trace, 'o-', ms=3)
    _ax.set_xlabel('EM step (one iteration each, all constraints on)')
    _ax.set_ylabel('log-likelihood')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Counting parameters: `bic` and `aic`

    Fixed parameters are not free, so `bic` / `aic` discount them: `fixed_weights` removes $K-1$, each frozen mean $d$ and each frozen covariance its own parameter count; bounds do *not* change the count
    (`frozen_components` keeps its earlier accounting). Recovering the implied number of free parameters $p$ from $\mathrm{BIC}=-2\ell+p\ln n$ shows it:
    """)
    return


@app.cell
def _(GaussianMixture, MAX_ITER, X, fit_gmm, gmm_free, init_covs, init_means, init_weights, n_total, np, pd, torch):
    def implied_parameters(g):
        return round((g.bic(X) + 2 * n_total * g.score(X)) / np.log(n_total), 6)

    _kw = dict(n_components=4, init_means=init_means, init_covariances=init_covs, init_weights=init_weights, max_iter=MAX_ITER, tol=1e-9)
    _models = {
        'free fit': gmm_free,
        'fixed_weights': fit_gmm(fixed_weights=torch.tensor([0.25] * 4)),
        'weight / mean / covariance bounds': fit_gmm(weight_bounds=(0.1, 0.4), mean_bounds=(-5.0, 5.0), covariance_bounds=(0.01, 2.0)),
        'frozen_means=(0,)': GaussianMixture(**_kw, frozen_means=(0,)).fit(X),
        'frozen_covariances=(2,)': GaussianMixture(**_kw, frozen_covariances=(2,)).fit(X),
    }
    pd.DataFrame({'implied free parameters': {k: implied_parameters(v) for k, v in _models.items()},
                  'expected': {'free fit': 23, 'fixed_weights': 23 - 3, 'weight / mean / covariance bounds': 23,
                               'frozen_means=(0,)': 23 - 2, 'frozen_covariances=(2,)': 23 - 3}})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (A free 4-component full-covariance model in 2-d has $(K-1)+Kd+Kd(d+1)/2=3+8+12=23$ parameters.)

    ## Try your own bounds

    In the interactive version of this notebook (see the link at the top of the page) the controls below let you set the weight band, the box for the means and the variance cap, and the fit and picture
    update. Statically you see the defaults: a weight band no blob touches, the mean box of the earlier section, and a variance cap of 3 that only the two elongated blobs reach.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # docs: hide-cell
    w_lo = mo.ui.slider(0.0, 0.25, step=0.01, value=0.05, label='weight lower bound', show_value=True)
    w_hi = mo.ui.slider(0.25, 1.0, step=0.01, value=0.40, label='weight upper bound', show_value=True)
    box_hi = mo.ui.slider(-1.0, 3.0, step=0.1, value=1.5, label='upper bound of both mean coordinates', show_value=True)
    var_hi = mo.ui.slider(0.1, 6.0, step=0.05, value=3.0, label='variance (eigenvalue) cap', show_value=True)
    mo.vstack([mo.hstack([w_lo, w_hi], justify='start'), mo.hstack([box_hi, var_hi], justify='start')])
    return box_hi, var_hi, w_hi, w_lo


@app.cell
def _(X_np, box_hi, fit_gmm, plot_gmm, plt, torch, var_hi, w_hi, w_lo):
    _upper = max(box_hi.value, -0.99)
    gmm_user = fit_gmm(weight_bounds=(w_lo.value, w_hi.value), mean_bounds=(torch.tensor([-1.0, -1.0]), torch.tensor([_upper, _upper])),
                       covariance_bounds=(0.05, var_hi.value))
    _fig, _ax = plt.subplots(figsize=(5.2, 4.4))
    plot_gmm(X_np, gmm=gmm_user, ax=_ax, ellipse_std_devs=[2], point_size=3, legend=False,
             title=f'weights {gmm_user.weights_.numpy().round(2).tolist()}')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample weights

    `fit(X, sample_weight=w)` fits a *weighted* sample. With $w$ of shape $(n,)$, row $i$ counts $w_i$ times: every responsibility is multiplied by $w_i$ before the M-step statistics, the mixing weights use
    $n_\mathrm{eff}=\sum_iw_i$ as the sample size, and convergence is judged on the weighted mean log-likelihood. Two consequences can be tested directly:

    * **integer weights are row duplication**, and
    * **weight $0$ is dropping the row.**
    """)
    return


@app.cell
def _(GaussianMixture, X, init_covs, init_means, init_weights, torch, y_true):
    _kw = dict(n_components=4, init_means=init_means, init_covariances=init_covs, init_weights=init_weights, max_iter=60, tol=1e-30)
    _labels = torch.as_tensor(y_true)
    _w = torch.ones(X.shape[0])
    _w[_labels == 1] = 2.0                                          # every point of blob 2 counts twice
    _weighted = GaussianMixture(**_kw).fit(X, sample_weight=_w)
    _duplicated = GaussianMixture(**_kw).fit(torch.cat([X, X[_labels == 1]]))
    print('weight 2 vs duplicated rows : max |means|', f'{(_weighted.means_ - _duplicated.means_).abs().max().item():.1e}',
          '| |covariances|', f'{(_weighted.covariances_ - _duplicated.covariances_).abs().max().item():.1e}',
          '| |weights|', f'{(_weighted.weights_ - _duplicated.weights_).abs().max().item():.1e}')

    _w0 = torch.ones(X.shape[0])
    _w0[_labels == 3] = 0.0                                         # blob 4 gets weight zero ...
    _zeroed = GaussianMixture(**_kw).fit(X, sample_weight=_w0)
    _keep = _labels != 3                                             # ... which must equal a 4-component fit on X without blob 4
    _dropped = GaussianMixture(**_kw).fit(X[_keep])
    print('weight 0 vs dropped rows    : max |means|', f'{(_zeroed.means_ - _dropped.means_).abs().max().item():.1e}',
          '| |covariances|', f'{(_zeroed.covariances_ - _dropped.covariances_).abs().max().item():.1e}',
          '| |weights|', f'{(_zeroed.weights_ - _dropped.weights_).abs().max().item():.1e}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Per-feature weights.** For `covariance_type='diag'` or `'spherical'` (maximum likelihood only) $w$ may have shape $(n,d)$: $w_{ij}$ then weights feature $j$ of row $i$ in *both* steps,

    $$
    \mu_{kj}=\frac{\sum_i r_{ik}w_{ij}x_{ij}}{\sum_i r_{ik}w_{ij}},\qquad
    \sigma^2_{kj}=\frac{\sum_i r_{ik}w_{ij}(x_{ij}-\mu_{kj})^2}{\sum_i r_{ik}w_{ij}},
    $$

    and $w_{ij}=0$ makes that entry invisible -- exact marginalisation of an independent feature. A convenient use is a corrupted or missing measurement: give it weight $0$ and its value is irrelevant.
    The cell corrupts the second coordinate of every point of blob 3 twice, once with $+100$ and once with $-100$, and gives those entries weight zero; the two fits are identical:
    """)
    return


@app.cell
def _(GaussianMixture, X, init_covs, init_means, init_weights, torch, y_true):
    _kw = dict(n_components=4, covariance_type='diag', init_means=init_means, init_covariances=torch.diagonal(init_covs, dim1=1, dim2=2),
               init_weights=init_weights, max_iter=60, tol=1e-30)
    _blob3 = torch.as_tensor(y_true) == 2
    _W = torch.ones_like(X)
    _W[_blob3, 1] = 0.0
    _fits = []
    for _bad in (100.0, -100.0):
        _Xc = X.clone()
        _Xc[_blob3, 1] = _bad
        _fits.append(GaussianMixture(**_kw).fit(_Xc, sample_weight=_W))
    print('corrupted +100 vs -100: max |means| difference', f'{(_fits[0].means_ - _fits[1].means_).abs().max().item():.1e}',
          '| max |variances| difference', f'{(_fits[0].covariances_ - _fits[1].covariances_).abs().max().item():.1e}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is (not) supported

    Constraints work with `em`, `cem`, `sem`, `saem`, `batch_size`, online EM / `partial_fit`, row `sample_weight`, `missing='em'`, `noise=True` (mean and covariance bounds only), `select_model(..., **kwargs)` and save / load.
    Initial parameters are projected into the feasible set (a `UserWarning` says so when a tensor you passed had to change). Each of the following raises a clear error:

    - Gibbs mode (`n_components=None`);
    - mean or covariance bounds, or frozen values, together with a mean or covariance prior;
    - `covariance_bounds` for the 8 GPCM codes without a classic name, and `frozen_covariances` / `frozen_components` for the tied types and those 8 codes (one factor is shared across components);
    - weight constraints together with `noise=True` or `labels=`;
    - any of these with per-feature $(n,d)$ `sample_weight`.

    ## Summary

    - Known values and bounds are ordinary constructor arguments, off by default, and a fit without them is unchanged.
    - EM remains valid: the constrained M-step is an ECM step whose block updates are each exact -- water-filling for the weights, clipping (diagonal / spherical) or coordinate ascent (full) for the means, variance or eigenvalue clipping for the covariances -- so the log-likelihood never decreases.
    - Where a bound binds the estimate sits on the boundary (see `active_constraints_`); `bic` / `aic` discount fixed parameters but not bounds.
    - `sample_weight` generalises row duplication (integer weights) and deletion (zero), and $(n,d)$ weights marginalise single features out of diagonal and spherical models.

    **Next:** [17 -- Semi-Supervised and PU Fitting](17_semi_supervised_pu.ipynb) uses `frozen_components` as an anchor, and [13 -- Model Selection](13_model_selection.ipynb) shows how to choose among constrained and unconstrained models.
    """)
    return


if __name__ == "__main__":
    app.run()
