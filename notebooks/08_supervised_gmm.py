# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "tgmm",
#     "torch; sys_platform != 'emscripten'",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Supervised Fitting vs. EM vs. CEM

    This notebook compares three ways of fitting a `GaussianMixture` on the same synthetic data:

    - **EM**: standard soft-assignment fitting, no label information.
    - **CEM**: Classification EM -- hard assignment via `argmax(resp)` each iteration, no label information.
    - **Supervised**: `fit(X, labels=y)` -- the assignment comes from the true label instead of `argmax(resp)`, converging in a single M-step (the approach follows Celeux & Govaert 1992, McLachlan & Peel 2000 and Fraley & Raftery 2002).

    The data reuses all 4 Gaussian components from [01 -- Introduction](01_introduction.ipynb) (same centers, covariances and sample counts), but two of them share one label -- so there are only 3 classes even though there are 4 physical blobs. That is deliberate: EM and CEM don't see labels, so with `n_components=3` they are free to find whichever 3-way split fits the data best (which turns out *not* to match the label grouping), while the supervised fit is bound by the label and has to force a single Gaussian across two blobs it is told belong together. This is what makes the three methods diverge visibly instead of agreeing trivially.

    **What you'll learn**

    - How `fit(X, labels=y)` differs from unsupervised EM/CEM, and why it needs no iteration
    - Why unsupervised components need not line up with class labels (and how `match_labels_to_true` handles arbitrary component indices)
    - How to read misclassifications and 95% confidence regions in `plot_gmm` panels
    - What the labelled fit computes (the equations), `predict` vs `predict_class`, `components_per_class`, and how a *partially* labelled fit blends labels with the data (`unlabeled_weight`, `label_confidence`)

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb) and [07 -- Classification EM](07_cem.ipynb).

    **See also:** [15 -- Semi-Supervised and Positive-Unlabeled Fitting](15_semi_supervised_pu.ipynb), where only some labels are known.
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
    import time

    import numpy as np
    import torch
    import matplotlib.pyplot as plt


    from tgmm import (
        GaussianMixture, plot_gmm, generate_gmm_data,
        gmm_proxy, confidence_ellipse_std_devs, ClusteringMetrics,
    )
    from tutorial_data import default_blobs

    # --- figure style (see notebooks/README.md) ---
    DOCS_RC = {'figure.dpi': 100, 'figure.titlesize': 12, 'axes.titlesize': 10.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
               'legend.fontsize': 8, 'lines.linewidth': 1.4, 'lines.markersize': 5}
    plt.rcParams.update(DOCS_RC)
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (
        GaussianMixture,
        SEED,
        confidence_ellipse_std_devs,
        default_blobs,
        device,
        generate_gmm_data,
        gmm_proxy,
        np,
        plot_gmm,
        plt,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data: 4 Gaussians from `01_introduction.ipynb`, 2 sharing one label

    Same 4 centers/covariances/sample counts as in [01 -- Introduction](01_introduction.ipynb). The
    two spatially farthest-apart components -- `(2, -2)` and `(2, 2)`,
    distance 4.0 -- share `Class 2`; `(0, 2)` and `(0, 0)` keep their own
    labels. An unsupervised 3-component fit has no reason to follow that
    grouping: it has to cover four blobs with three Gaussians, and the wide,
    flat blob at `(0, 0)` (a large variance along the first axis) overlaps its
    neighbours so much that the fit tends to split it, or to absorb the small blob at `(2, -2)`
    into it. The table after the fits shows which component each blob ends up in.
    """)
    return


@app.cell
def _(SEED, default_blobs, device, generate_gmm_data, torch):
    # The default blobs of tutorial_data.py, reordered as (spherical, diagonal, small spherical, full) so that the two
    # farthest-apart blobs sit at positions 2 and 3 and can share a label.
    _centers, _covs, _ns = default_blobs()
    _order = [0, 2, 1, 3]
    centers = [_centers[_i] for _i in _order]
    covs = [_covs[_i] for _i in _order]
    n_samples = [_ns[_i] for _i in _order]

    # Components 2 and 3 (the two farthest apart, distance 4.0) share one
    # label -- physical_to_label[i] is the class label for physical blob i.
    physical_to_label = torch.tensor([0, 1, 2, 2])
    n_components = 3
    legend_labels = [f'Class {i}' for i in range(n_components)]

    X_tensor, y_physical = generate_gmm_data(centers, covs, n_samples, device=device, random_state=SEED)
    y_tensor = physical_to_label.to(y_physical.device)[y_physical]
    X = X_tensor.cpu().numpy()
    y_true = y_tensor.cpu().numpy()
    return (
        X,
        X_tensor,
        centers,
        covs,
        n_components,
        physical_to_label,
        y_physical,
        y_tensor,
        y_true,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting all three models

    EM and CEM don't see `y_tensor` at all; only the supervised fit does.
    """)
    return


@app.cell
def _(GaussianMixture, SEED, X_tensor, n_components, time, y_tensor):
    gmm_em = GaussianMixture(n_components=n_components, random_state=SEED)
    t0 = time.time()
    gmm_em.fit(X_tensor)
    time_em = time.time() - t0

    gmm_cem = GaussianMixture(n_components=n_components, random_state=SEED, algorithm='cem')
    t0 = time.time()
    gmm_cem.fit(X_tensor)
    time_cem = time.time() - t0

    gmm_sup = GaussianMixture(n_components=n_components)
    t0 = time.time()
    gmm_sup.fit(X_tensor, labels=y_tensor)
    time_sup = time.time() - t0

    print(f"EM:         iters={gmm_em.n_iter_:4d}  converged={gmm_em.converged_}  time={time_em:.4f}s  mean log-lik={gmm_em.score(X_tensor):.4f}")
    print(f"CEM:        iters={gmm_cem.n_iter_:4d}  converged={gmm_cem.converged_}  time={time_cem:.4f}s  mean log-lik={gmm_cem.score(X_tensor):.4f}")
    print(f"Supervised: iters={gmm_sup.n_iter_:4d}  converged={gmm_sup.converged_}  time={time_sup:.4f}s  mean log-lik={gmm_sup.score(X_tensor):.4f}")
    print(f"\nSupervised gmm.classes_: {gmm_sup.classes_.cpu().tolist()}")
    return gmm_cem, gmm_em, gmm_sup


@app.cell(hide_code=True)
def _(centers, gmm_cem, gmm_em, mo, np, X_tensor, y_physical):
    _yp = y_physical.cpu().numpy()
    _blob = [f"({_c[0]}, {_c[1]})" for _c in centers]
    _counts = {}
    for _name, _g in (('EM', gmm_em), ('CEM', gmm_cem)):
        _p = _g.predict(X_tensor).cpu().numpy()
        _counts[_name] = np.array([[int(((_yp == _b) & (_p == _k)).sum()) for _k in range(_g.n_components)] for _b in range(len(centers))])
    _lines = ['| blob (centre) | label | n | ' + ' | '.join(f'EM comp. {_k}' for _k in range(3)) + ' | ' + ' | '.join(f'CEM comp. {_k}' for _k in range(3)) + ' |', '|---|---|---|' + '---|' * 6]
    for _b in range(len(centers)):
        _lines.append(f"| {_blob[_b]} | {[0, 1, 2, 2][_b]} | {int((_yp == _b).sum())} | " + ' | '.join(str(_v) for _v in _counts['EM'][_b]) + ' | ' + ' | '.join(str(_v) for _v in _counts['CEM'][_b]) + ' |')
    _dom = _counts['EM'].argmax(1)
    _split = [_blob[_b] for _b in range(len(centers)) if _counts['EM'][_b].max() < 0.8 * _counts['EM'][_b].sum()]
    _together = [(_blob[_a], _blob[_b]) for _a in range(len(centers)) for _b in range(_a + 1, len(centers)) if _dom[_a] == _dom[_b]]
    _txt = "\n".join(_lines) + "\n\n"
    _txt += ("Blobs `" + "`, `".join(_split) + "` are spread over several EM components. " if _split else "")
    _txt += ("The blobs " + ", ".join(f"`{_a}` and `{_b}`" for _a, _b in _together) + " mostly share one EM component. " if _together else "")
    _txt += f"The two blobs that carry the same label, `{_blob[2]}` and `{_blob[3]}`, have their points mostly in " + ("the same EM component." if _dom[2] == _dom[3] else "different EM components.")
    mo.md(_txt)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visual comparison

    2x2 grid: true classes, EM, CEM, supervised. All four panels use the same 3
    blue shades for `Class 0/1/2`, drawn entirely through `plot_gmm` -- no
    hand-rolled matplotlib. The "True Classes" panel uses `gmm_proxy` to feed
    the 4 true generating Gaussians (not a fitted model) into `plot_gmm`, with
    `cluster_colors` repeating Class 2's blue for the two merged physical
    components. EM/CEM component indices are arbitrary, so `match_labels_to_true`
    Hungarian-matches them to the true labels for both point and ellipse
    coloring; `show_incorrect_predictions` then colors correct points by their
    matched cluster color and incorrect ones red -- so misclassifications are
    directly visible, concentrated where the forced 3-way split disagrees with
    the (arbitrary) label grouping. The supervised fit needs no matching, since
    its component index already *is* the label (`classes_ == [0, 1, 2]`).
    Ellipses are 95% confidence regions via `confidence_ellipse_std_devs(0.95)`
    (Mahalanobis radius `sqrt(chi2.ppf(0.95, df=2))`, *not* "2 standard
    deviations" -- that's a 1D interval, not a 2D confidence region), and
    markers show each component's mean.
    """)
    return


@app.cell
def _(
    X,
    centers,
    confidence_ellipse_std_devs,
    covs,
    gmm_cem,
    gmm_em,
    gmm_proxy,
    gmm_sup,
    n_components,
    np,
    physical_to_label,
    plot_gmm,
    plt,
    y_physical,
    y_true,
):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    BLUE_SHADES = ['#9ecae1', '#4292c6', '#08519c']  # Class 0, 1, 2
    WRONG_COLOR = '#d62728'                          # incorrect predictions (EM/CEM only)
    CONF_95 = confidence_ellipse_std_devs(0.95)      # chi2-based 95% CI multiplier, not "2 std devs"

    ellipse_kwargs = dict(show_ellipses=True, ellipse_std_devs=CONF_95, ellipse_fill=True, ellipse_alpha=0.15)
    mean_kwargs = dict(show_means=True, mean_marker='h', mean_color='black', mean_size=25)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle('True labels vs. EM vs. CEM vs. Supervised')

    # True classes: the 4 physical Gaussians via gmm_proxy (a stand-in `gmm=`
    # object built from the raw generating parameters, not a fitted model),
    # colored by the label each was folded into -- physical components 2 and 3
    # both get Class 2's blue.
    true_proxy = gmm_proxy(
        means=np.stack(centers), covariances=np.stack(covs),
        covariance_type='full', predicted_labels=y_physical,
    )
    true_cluster_colors = [BLUE_SHADES[label] for label in physical_to_label.tolist()]
    plot_gmm(X=X, gmm=true_proxy, ax=axes[0, 0], title='True Classes',
             color_by_cluster=True, cluster_colors=true_cluster_colors,
             point_size=8, legend=False, **ellipse_kwargs, **mean_kwargs)

    # EM / CEM: match_labels_to_true Hungarian-matches the arbitrary component
    # indices to true labels for both point and ellipse coloring;
    # show_incorrect_predictions colors correct points by their matched cluster
    # color and incorrect ones in WRONG_COLOR.
    plot_gmm(X=X, gmm=gmm_em, ax=axes[0, 1], title=f'EM\n({gmm_em.n_iter_} iters)',
             color_by_cluster=True, true_labels=y_true, match_labels_to_true=True,
             cluster_colors=BLUE_SHADES, show_incorrect_predictions=True, incorrect_color=WRONG_COLOR,
             point_size=8, legend=False, **ellipse_kwargs, **mean_kwargs)

    plot_gmm(X=X, gmm=gmm_cem, ax=axes[1, 0], title=f'CEM\n({gmm_cem.n_iter_} iters)',
             color_by_cluster=True, true_labels=y_true, match_labels_to_true=True,
             cluster_colors=BLUE_SHADES, show_incorrect_predictions=True, incorrect_color=WRONG_COLOR,
             point_size=8, legend=False, **ellipse_kwargs, **mean_kwargs)

    # Supervised: component index already is the label (classes_ == [0, 1, 2]),
    # no Hungarian matching needed. Not correctness-colored (only EM/CEM get the
    # red/blue treatment) -- points are colored by gmm_sup's own predict(),
    # which by construction of the supervised fit matches the given label
    # almost everywhere.
    plot_gmm(X=X, gmm=gmm_sup, ax=axes[1, 1], title='Supervised\n(1 iter)',
             color_by_cluster=True, cluster_colors=BLUE_SHADES,
             point_size=8, legend=False, **ellipse_kwargs, **mean_kwargs)

    legend_handles = (
        [Patch(facecolor=BLUE_SHADES[i], edgecolor='none', label=f'Class {i}') for i in range(n_components)]
        + [Patch(facecolor=WRONG_COLOR, edgecolor='none', label='Incorrect (EM/CEM)'),
           Line2D([0], [0], marker='h', color='w', markerfacecolor='black', markersize=5, label='Component mean'),
           Line2D([0], [0], color='gray', linestyle='dotted', linewidth=1.4, label='95% confidence region')]
    )
    fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the labelled fit computes

    Write $c_i$ for the known class of point $i$ and give each class its own component, $k = c_i$. The responsibilities are then not estimated, they are *given*:

    $$
    r_{ik} = \mathbb{1}[c_i = k].
    $$

    With one-hot $r_{ik}$ the E-step no longer depends on the parameters, so the M-step is a fixed point and a single pass finishes the fit. For `covariance_type='full'` it reads

    $$
    \pi_k = \frac{n_k}{n},\qquad
    \mu_k = \frac{1}{n_k}\sum_{i:\,c_i=k} x_i,\qquad
    \Sigma_k = \frac{1}{n_k}\sum_{i:\,c_i=k}(x_i-\mu_k)(x_i-\mu_k)^{\top},
    $$

    which maximises the *complete-data* log-likelihood $\sum_i \log \pi_{c_i}\,\mathcal N(x_i\mid\mu_{c_i},\Sigma_{c_i})$ of a generative classifier (quadratic discriminant analysis for `full`, linear discriminant analysis for `tied_full`). Other covariance types use their usual M-step on the same one-hot responsibilities (see [02 -- Covariance Types](02_covariance_types.ipynb)). This is exactly where CEM ([07 -- CEM](07_cem.ipynb)) ends up when it is started from the correct partition. Priors, if given, are applied in the same M-step.

    The next cell checks the closed form against the fitted model. The differences are of the order of `reg_covar` (default $10^{-6}$), which is added to every covariance diagonal for numerical stability.
    """)
    return


@app.cell
def _(gmm_sup, np, y_true, X):
    _y = y_true
    _pi = np.array([(_y == _c).mean() for _c in range(3)])
    _mu = np.stack([X[_y == _c].mean(0) for _c in range(3)])
    _S = np.stack([np.cov(X[_y == _c].T, bias=True) for _c in range(3)])
    print('max |pi_k    - n_k/n|      =', abs(gmm_sup.weights_.cpu().numpy() - _pi).max())
    print('max |mu_k    - class mean| =', abs(gmm_sup.means_.cpu().numpy() - _mu).max())
    print('max |Sigma_k - class cov|  =', abs(gmm_sup.covariances_.cpu().numpy() - _S).max())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Class labels, component indices and `components_per_class`

    `n_components` must equal the number of distinct labels times `components_per_class` (an `int`, or a `dict` from class label to a number of components). Label values need not be contiguous; `classes_` maps the internal class index back to the value you passed. `predict` returns a *component* index, `predict_class` sums the responsibilities over each class's block of components and returns the original *label*.

    A class that is really two blobs (here Class 2, made of the blobs at $(2,-2)$ and $(2,2)$) is badly described by one Gaussian. Giving it two components, `components_per_class={0: 1, 1: 1, 2: 2}`, fits each class's block by ordinary EM on that class's rows only (weights scaled by the class fraction), so the labels still decide which rows belong to which class. Tied covariance types are not supported in this mode, because one shared covariance cannot be assembled from independently fitted blocks.
    """)
    return


@app.cell
def _(GaussianMixture, X_tensor, gmm_sup, torch, y_tensor):
    gmm_cpc = GaussianMixture(n_components=4, components_per_class={0: 1, 1: 1, 2: 2}, random_state=0, n_init=3)
    gmm_cpc.fit(X_tensor, labels=y_tensor)

    print('class_of_component_      :', gmm_cpc.class_of_component_.tolist(), '(components 2 and 3 both belong to the class with index 2)')
    print('mean log-lik, 1 comp/class:', round(gmm_sup.score(X_tensor), 4))
    print('mean log-lik, class 2 x 2 :', round(gmm_cpc.score(X_tensor), 4))
    print('predict (component)      :', gmm_cpc.predict(X_tensor[-5:]).tolist())
    print('predict_class (label)    :', gmm_cpc.predict_class(X_tensor[-5:]).tolist())

    # non-contiguous label values: classes_ records them, predict_class returns them
    gmm_lab = GaussianMixture(n_components=3).fit(X_tensor, labels=torch.tensor([10, 20, 30], device=y_tensor.device)[y_tensor])
    print('classes_ for labels {10,20,30}:', gmm_lab.classes_.tolist(), ' predict_class:', gmm_lab.predict_class(X_tensor[:3]).tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partially labelled data

    Mark an unknown label with `-1`. Let $\mathcal L$ be the labelled rows and $\mathcal U$ the rest. Semi-supervised EM maximises

    $$
    \sum_{i\in\mathcal L}\log\!\!\sum_{k\in\text{block}(c_i)}\!\!\pi_k\,\mathcal N(x_i\mid\mu_k,\Sigma_k)\;+\;\sum_{i\in\mathcal U}\log\sum_{k=1}^{K}\pi_k\,\mathcal N(x_i\mid\mu_k,\Sigma_k),
    $$

    by *clamping* each labelled row's responsibility to its class's block of components (zero outside, renormalised inside) and leaving unlabelled rows with the full posterior. Two knobs change that:

    - `unlabeled_weight` $u\ge0$ multiplies the unlabelled rows' responsibilities before the M-step. $u=0$ fits from the labelled rows alone, $u=1$ is the likelihood above, and for $u\ne1$ the update is a weighted likelihood (the weights are renormalised to sum to 1).
    - `label_confidence` $c<1$ keeps mass $c$ on the class block and gives $1-c$ to the unclamped posterior *outside* the block, which tolerates noisy labels.

    Only `algorithm='em'` and `'cem'` support partial labels (`'sem'`/`'saem'` raise `NotImplementedError`). The class-to-blob structure here is deliberately awkward (Class 2 is two distant blobs), so we can watch the two sources of information compete: with 10% of the labels known, the unlabelled rows pull the components toward the data's own geometry as $u$ grows.
    """)
    return


@app.cell
def _(GaussianMixture, X_tensor, n_components, torch, y_tensor):
    _gen = torch.Generator().manual_seed(0)
    _known = (torch.rand(len(y_tensor), generator=_gen) < 0.1).to(y_tensor.device)
    y_partial = torch.where(_known, y_tensor, torch.full_like(y_tensor, -1))
    print(f'labelled rows: {int(_known.sum())} of {len(y_tensor)}')
    print(f"{'setting':22s} {'iters':>5s} {'accuracy (all rows)':>20s} {'mean log-lik':>13s}")
    for _name, _kw in [('u = 0 (labelled only)', dict(unlabeled_weight=0.0)), ('u = 0.2', dict(unlabeled_weight=0.2)),
                       ('u = 1 (full likelihood)', dict()), ('u = 1, confidence 0.8', dict(label_confidence=0.8))]:
        _g = GaussianMixture(n_components=n_components, random_state=0, n_init=3, **_kw).fit(X_tensor, labels=y_partial)
        _acc = (_g.predict_class(X_tensor) == y_tensor).float().mean().item()
        print(f'{_name:22s} {_g.n_iter_:5d} {_acc:20.3f} {_g.score(X_tensor):13.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Accuracy is measured against the full labels. Adding the unlabelled rows raises the likelihood but can lower the agreement with the labels, because the data's natural structure disagrees with the labelling; a smaller $u$ trusts the labels more. For noisy labels, positive-unlabelled problems and the details of every option see [15 -- Semi-Supervised and Positive-Unlabeled Fitting](15_semi_supervised_pu.ipynb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - Supervised fitting (`fit(X, labels=y)`) takes the assignment from the labels, so one M-step already gives the per-class mean, covariance and weight; `n_iter_` is always 1.
    - EM and CEM never see the labels, so with a label grouping that disagrees with the data's geometry they find a different 3-way split.
    - Unsupervised component indices are arbitrary; `match_labels_to_true=True` Hungarian-matches them to the labels so errors can be visualized.
    - Supervised fits trade the flexibility of unsupervised fitting for fidelity to the labels -- useful when the labels are trusted.
    - The labelled M-step is the class-conditional maximum likelihood (QDA/LDA-style), checked numerically above; `predict_class` returns labels, `predict` component indices, and `components_per_class` gives a class several components.
    - With partial labels, `unlabeled_weight` and `label_confidence` set how strongly the labels compete with the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Next:** [15 -- Semi-Supervised and Positive-Unlabeled Fitting](15_semi_supervised_pu.ipynb) for the full treatment of partially labeled data and `PositiveUnlabeledMixture`, or [09 -- Clustering Metrics](09_metrics.ipynb) to quantify agreement with labels.
    """)
    return


if __name__ == "__main__":
    app.run()
