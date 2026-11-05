# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "tgmm",
#     "torch; sys_platform != 'emscripten'",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
#     "pandas",
#     "scipy",
#     "seaborn",
#     "tqdm",
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
    return (MAX_ITER,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering Metrics: Evaluating Gaussian Mixture Models

    This notebook evaluates clustering performance with both **unsupervised** indices (which need only the data and the assignments) and **supervised** scores (which compare against ground-truth labels), using `ClusteringMetrics` and the information criteria built into `GaussianMixture`. It covers the mathematical definitions, how each index behaves (separation, the choice of $K$, noise points), and how to use it. `ClusteringMetrics` is a collection of static methods: call them on the class, there is no instance to construct.

    **What you'll learn**

    - Choose the number of components with Silhouette, Davies-Bouldin, Calinski-Harabasz, WB, Wemmert-Gancarski, DBCV, Dunn, BIC, AIC and ICL, and see which of them are fooled by overlap
    - Use the shared options of the unsupervised indices: `metric`, `noise_label` and `chunk_size`
    - Compare supervised scores (Rand/ARI, MI/NMI/AMI, FMI, homogeneity/completeness/V-measure, NID, purity) against scikit-learn
    - Build a confusion matrix and classification report after matching cluster labels to true classes
    - Measure the divergence between two fitted mixtures with a Monte Carlo KL estimate

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb) (the data and the fitted model) and [02 -- Covariance Types](02_covariance_types.ipynb) (the covariance parameter counts that BIC and AIC use).

    **See also:** [13 -- Model Selection](13_model_selection.ipynb) for `select_model` and the BIC-vs-ICL comparison, and the [`ClusteringMetrics` API reference](../api/clustering-metrics.md).
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
    import pandas as pd
    import seaborn as sns
    import torch
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.cluster import KMeans

    from tgmm import GaussianMixture, ClusteringMetrics, dynamic_figsize, plot_gmm, match_predicted_to_true_labels
    from tutorial_data import make_default_data

    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; keep embedded figure outputs small

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (
        ClusteringMetrics,
        GaussianMixture,
        KMeans,
        SEED,
        device,
        dynamic_figsize,
        make_default_data,
        match_predicted_to_true_labels,
        np,
        pd,
        plot_gmm,
        plt,
        sns,
        torch,
        tqdm,
    )


@app.cell
def _(SEED, device, make_default_data):
    # The default four blobs of 01 -- Introduction (see tutorial_data.py)
    X_tensor, y_tensor = make_default_data(SEED, device=device)
    n_features = X_tensor.shape[1]
    n_components = 4
    return X_tensor, n_components, n_features, y_tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unsupervised Cluster Indices

    These indices judge a partition from the data and the assignments alone. They apply to any clustering (here: the hard assignments `gmm.predict` of a fitted mixture) and take `labels` and, optionally, `n_components`.

    **Notation.** A clustering assigns each of $n$ points $x_i$ to one of $K$ clusters $C_1,\dots,C_K$ with sizes $n_k=|C_k|$ and centroids $c_k=\frac1{n_k}\sum_{x\in C_k}x$. This is the plain coordinate-wise mean of the *hard* assignment, not the mixture's `means_`. $d(\cdot,\cdot)$ is the chosen `metric` (Euclidean by default), and $\bar x$ is the global mean. With

    $$
    \mathrm{SSW}=\sum_{k}\sum_{x\in C_k} d(x,c_k)^2,\qquad
    \mathrm{SSB}=\sum_{k} n_k\, d(c_k,\bar x)^2
    $$

    the within- and between-cluster sums of squared distances (for the Euclidean metric $\mathrm{SSW}+\mathrm{SSB}$ is the total sum of squares).

    ### Silhouette Score

    For a point $i$ in cluster $C_{k(i)}$,

    $$
    a(i)=\frac{1}{n_{k(i)}-1}\sum_{j\in C_{k(i)},\,j\ne i} d(i,j),\qquad
    b(i)=\min_{l\ne k(i)}\frac1{n_l}\sum_{j\in C_l} d(i,j),
    $$

    the mean distance to its own cluster and to the *nearest other* cluster. Then

    $$
    s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}},\qquad \text{Silhouette}=\frac1n\sum_i s(i),
    $$

    with $s(i)=0$ for singleton clusters. It lies in $[-1,1]$ (higher is better; 0 means overlapping clusters) and needs at least two clusters.

    ### Davies-Bouldin Index

    With the within-cluster scatter $S_k=\frac1{n_k}\sum_{x\in C_k}d(x,c_k)$ (mean distance to the centroid) and the centroid separation $M_{kl}=d(c_k,c_l)$,

    $$
    \mathrm{DB}=\frac1K\sum_{k=1}^K\max_{l\ne k}\frac{S_k+S_l}{M_{kl}},
    $$

    the average over clusters of the similarity to the *most similar* other cluster: large when two clusters are wide and close. It lies in $[0,\infty)$ and **lower is better**.

    ### Calinski-Harabasz Score

    The variance-ratio (pseudo-$F$) statistic

    $$
    \mathrm{CH}=\frac{\mathrm{SSB}/(K-1)}{\mathrm{SSW}/(n-K)},
    $$

    in $[0,\infty)$, **higher is better**.

    ### WB-Index

    $$
    \mathrm{WB}=K\,\frac{\mathrm{SSW}}{\mathrm{SSB}}
    $$

    (Zhao & Fränti, 2014): the same SSW/SSB as Calinski-Harabasz, with the number of clusters as a built-in penalty. In $[0,\infty)$, **lower is better**.

    ### Wemmert-Gancarski Index

    Compare each point's distance to its own centroid with its distance to the *nearest other* centroid:

    $$
    R(x)=\frac{d(x,c_{k(x)})}{\min_{l\ne k(x)}d(x,c_l)},\qquad
    J_k=\max\Big(0,\,1-\frac1{n_k}\sum_{x\in C_k}R(x)\Big),\qquad
    \mathrm{WG}=\frac1n\sum_k n_kJ_k .
    $$

    A cluster contributes $0$ when, on average, its points are no closer to their own centroid than to another's. In $[0,\infty)$ (in practice $[0,1]$), **higher is better**.

    ### DBCV (Density-Based Clustering Validation)

    DBCV (Moulavi, Jaskowiak, Campello, Zimek & Sander, 2014) is the natural index for density-based partitions such as `HDBSCAN` output (see [12 -- HDBSCAN](12_hdbscan.ipynb)), where centroid-based indices like Davies-Bouldin make no sense for non-convex clusters. For every cluster $C$, with $d_f$ the number of features:

    1. The **all-points core distance** of $x\in C$ is $a_{\text{core}}(x)=\Big(\frac1{|C|-1}\sum_{y\in C,\,y\ne x}d(x,y)^{-d_f}\Big)^{-1/d_f}$.
    2. The **mutual reachability** is $d_{\text{mreach}}(x,y)=\max\{a_{\text{core}}(x),\,a_{\text{core}}(y),\,d(x,y)\}$.
    3. A **minimum spanning tree** (MST) over $C$'s mutual-reachability graph gives the **density sparseness** $\mathrm{DSC}(C)$, the largest MST edge weight.
    4. The **density separation** $\mathrm{DSPC}(C_i,C_j)$ is the smallest mutual-reachability distance between an *internal* node (MST degree $\ge2$) of $C_i$ and one of $C_j$.

    The per-cluster validity and the index are

    $$
    V(C_i)=\frac{\min_{j\ne i}\mathrm{DSPC}(C_i,C_j)-\mathrm{DSC}(C_i)}{\max\{\min_{j\ne i}\mathrm{DSPC}(C_i,C_j),\,\mathrm{DSC}(C_i)\}},\qquad
    \mathrm{DBCV}=\sum_i\frac{n_i}{n}V(C_i),
    $$

    where $n$ **includes the noise points**, which add to the denominator only. That is DBCV's own noise handling, so it does not use the generic `noise_label` rescaling described below. Clusters with fewer than 2 points contribute $0$. The core distance here is the all-points one of the DBCV paper, not the $k$-NN core distance `HDBSCAN` uses internally. In $[-1,1]$, **higher is better**.

    ### Dunn Index

    $$
    D=\frac{\min_{k\ne l}\,d(C_k,C_l)}{\max_{k}\,\mathrm{diam}(C_k)},\qquad
    d(C_k,C_l)=\min_{x\in C_k,\,y\in C_l}d(x,y),\quad
    \mathrm{diam}(C_k)=\max_{x,y\in C_k}d(x,y),
    $$

    the smallest between-cluster gap over the largest within-cluster diameter. It depends on only two pairs of points, so it is very sensitive to outliers. In $[0,\infty)$, **higher is better**.

    > **Which to prefer.** In the clustering-validity benchmark of Simpson, Campello & Stojanovski (2025), the WB-Index, Wemmert-Gancarski and DBCV ranked near the top, while the Dunn Index ranked 25th of 26. `tgmm` keeps Dunn for completeness and familiarity, but prefer the other three.

    ### Information criteria (BIC, AIC, ICL)

    BIC and AIC only need a model's own log-likelihood and parameter count, so they are `bic`/`aic`/`icl` *methods on every fitted mixture* (`GaussianMixture`, including `n_components=None` in Gibbs mode, `StudentTMixture`, `VonMisesFisherMixture`), not `ClusteringMetrics` calls. With $\ell(\hat\theta)=\sum_i\log p(x_i\mid\hat\theta)$ the maximized *total* log-likelihood and $p$ the number of free parameters,

    $$
    \mathrm{BIC}=-2\,\ell(\hat\theta)+p\ln n,\qquad \mathrm{AIC}=-2\,\ell(\hat\theta)+2p,
    $$

    both "lower is better". BIC's $\ln n$ penalty is heavier than AIC's $2$ once $n>7$, so BIC prefers smaller models and is consistent for the true $K$ when the model family is correct, while AIC targets predictive accuracy and tends to pick more components. For `GaussianMixture`, $p=(K-1)+Kd+p_\Sigma$: $K-1$ free weights, $Kd$ means, and $p_\Sigma$ covariance parameters, which depend on the covariance structure (the table of $p_\Sigma$ for all 20 types is in [02 -- Covariance Types](02_covariance_types.ipynb)). In Gibbs mode $K$ is the number of *active* components.

    **ICL** (Biernacki, Celeux & Govaert, 2000) adds an entropy penalty for ambiguous assignments. With responsibilities $r_{ik}$ and $\hat z_i=\arg\max_k r_{ik}$,

    $$
    \mathrm{ICL}_{\text{hard}}=\mathrm{BIC}-2\sum_i\ln r_{i\hat z_i},\qquad
    \mathrm{ICL}_{\text{soft}}=\mathrm{BIC}-2\sum_{i,k}r_{ik}\ln r_{ik}.
    $$

    Both penalties are $\ge0$, so $\mathrm{ICL}\ge\mathrm{BIC}$, and the two coincide when the responsibilities are one-hot (perfectly separated clusters). ICL is BIC applied to the *completed* (data plus labels) likelihood, so it penalizes solutions whose components overlap and tends to prefer fewer, better-separated components. The worked BIC-vs-ICL comparison, and `select_model` (which sweeps `n_components` and `covariance_type` in one call), are in [13 -- Model Selection](13_model_selection.ipynb).

    **Reading the sweep below.** In practice the ideal number of clusters is where the Silhouette, Calinski-Harabasz, Wemmert-Gancarski, DBCV and Dunn indices are *maximal* and the Davies-Bouldin, WB, AIC and BIC values are *minimal*.
    """)
    return


@app.cell
def _(
    ClusteringMetrics,
    GaussianMixture,
    MAX_ITER,
    SEED,
    X_tensor,
    device,
    n_components,
    n_features,
    np,
    torch,
    tqdm,
):
    components_range = np.arange(2, 11)
    _nk = len(components_range)
    silhouette_vals = torch.zeros(_nk, device=device)
    davies_vals = torch.zeros(_nk, device=device)
    calinski_vals = torch.zeros(_nk, device=device)
    dunn_vals = torch.zeros(_nk, device=device)
    wb_vals = torch.zeros(_nk, device=device)
    wg_vals = torch.zeros(_nk, device=device)
    dbcv_vals = torch.zeros(_nk, device=device)
    bic_vals = torch.zeros(_nk, device=device)
    aic_vals = torch.zeros(_nk, device=device)
    labels_by_k = {}
    for _i, _n in tqdm(enumerate(components_range), total=_nk, desc='Evaluating range'):
        _gmm = GaussianMixture(n_features=n_features, n_components=_n, covariance_type='full', max_iter=MAX_ITER, init_means='kmeans', device=device, random_state=SEED, tol=1e-05, reg_covar=1e-07)
        _gmm.fit(X_tensor)
        _lab = _gmm.predict(X_tensor)
        labels_by_k[int(_n)] = _lab
        silhouette_vals[_i] = ClusteringMetrics.silhouette_score(X_tensor, _lab, n_components=_n)
        davies_vals[_i] = ClusteringMetrics.davies_bouldin_index(X_tensor, _lab, n_components=_n)
        calinski_vals[_i] = ClusteringMetrics.calinski_harabasz_score(X_tensor, _lab, n_components=_n)
        dunn_vals[_i] = ClusteringMetrics.dunn_index(X_tensor, _lab, n_components=_n)
        wb_vals[_i] = ClusteringMetrics.wb_index(X_tensor, _lab, n_components=_n)
        wg_vals[_i] = ClusteringMetrics.wemmert_gancarski_index(X_tensor, _lab, n_components=_n)
        dbcv_vals[_i] = ClusteringMetrics.dbcv(X_tensor, _lab)
        bic_vals[_i] = _gmm.bic(X_tensor)
        aic_vals[_i] = _gmm.aic(X_tensor)
    sweep = {
        'Silhouette': (silhouette_vals, 'max', 'o-b'),
        'Davies-Bouldin': (davies_vals, 'min', 'o-g'),
        'Calinski-Harabasz': (calinski_vals, 'max', 'o-y'),
        'Dunn': (dunn_vals, 'max', 'o-'),
        'WB-Index': (wb_vals, 'min', 'o-r'),
        'Wemmert-Gancarski': (wg_vals, 'max', 'o-k'),
        'DBCV': (dbcv_vals, 'max', 'o-'),
        'BIC': (bic_vals, 'min', 'o-m'),
        'AIC': (aic_vals, 'min', 'o-c'),
    }
    best_k = {_name: int(components_range[(_v.argmax() if _w == 'max' else _v.argmin()).item()]) for _name, (_v, _w, _f) in sweep.items()}
    return best_k, components_range, labels_by_k, sweep


@app.cell
def _(best_k, components_range, dynamic_figsize, n_components, plt, sweep):
    _names = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Dunn', 'BIC', 'AIC']
    _fig, _axs = plt.subplots(3, 2, figsize=dynamic_figsize(3, 2))
    for _ax, _name in zip(_axs.ravel(), _names):
        _vals, _dir, _fmt = sweep[_name]
        _ax.plot(components_range, _vals.cpu(), _fmt)
        _ax.axvline(x=n_components, color='r', linestyle='--', label='True # of Components')
        _ax.axvline(x=best_k[_name], color='k', linestyle=':', label='Predicted Ideal # of Components')
        _ax.set_title(f"{_name} ({'higher' if _dir == 'max' else 'lower'} is better)")
        _ax.set_xlabel('Number of Components')
        _ax.grid(True)
        _ax.legend(loc='upper right')
    plt.suptitle('Unsupervised Clustering Metrics for a GMM')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.gcf()
    return


@app.cell
def _(best_k, components_range, n_components, plt, sweep):
    _names = ['WB-Index', 'Wemmert-Gancarski', 'DBCV']
    _fig, _axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for _ax, _name in zip(_axs, _names):
        _vals, _dir, _fmt = sweep[_name]
        _ax.plot(components_range, _vals.cpu(), _fmt)
        _ax.axvline(x=n_components, color='r', linestyle='--', label='True # of Components')
        _ax.axvline(x=best_k[_name], color='k', linestyle=':', label='Predicted Ideal # of Components')
        _ax.set_title(f"{_name} ({'higher' if _dir == 'max' else 'lower'} is better)")
        _ax.set_xlabel('Number of Components')
        _ax.grid(True)
    _axs[0].legend(loc='upper right')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(best_k, pd):
    print('=== Best Number of Components According to Each Metric (true K = 4) ===')
    pd.Series(best_k, name='best K').to_frame()
    return


@app.cell(hide_code=True)
def _(best_k, components_range, labels_by_k, mo, n_components, np, y_tensor):
    _top = int(components_range.max())
    _lab_top = labels_by_k[_top].cpu().numpy()
    _y = y_tensor.cpu().numpy()
    _pieces = np.zeros(n_components, int)  # components of the largest-K fit whose majority comes from each true blob
    for _c in np.unique(_lab_top):
        _pieces[np.bincount(_y[_lab_top == _c], minlength=n_components).argmax()] += 1

    _right = [_n for _n, _k in best_k.items() if _k == n_components]
    _edge = [_n for _n, _k in best_k.items() if _k >= _top - 1]
    _rest = {_n: _k for _n, _k in best_k.items() if _n not in _right and _n not in _edge}
    _msg = f"With the true $K={n_components}$, {' and '.join(_right) if _right else 'no index'} recover{'s' if len(_right) == 1 else ''} it in this run. "
    if _edge:
        _msg += (f"{', '.join(_edge)} run to the top of the range ($K\\ge{_top - 1}$ of at most {_top}): these indices keep rewarding finer partitions of this data, most likely because the elongated blobs "
                 f"are cut into several compact pieces that a centroid-based index scores better than one stretched cluster, whereas a mixture likelihood describes the elongation with a single covariance "
                 f"(in the $K={_top}$ fit the blobs are covered by {', '.join(str(int(_p)) for _p in _pieces)} components, in the order of the true labels: the wide flat blob and the long thin tilted one are the ones cut up). ")
    if _rest:
        _msg += "The rest pick " + ", ".join(f"{_n} $K={_k}$" for _n, _k in _rest.items()) + ". "
    _msg += "No single index should be trusted alone. (Results in the fast docs build, which caps EM at 100 iterations, can differ slightly.)"
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Shared options: `metric`, `noise_label`, `chunk_size`

    `silhouette_score`, `davies_bouldin_index`, `calinski_harabasz_score`, `dunn_index`, `wb_index` and `wemmert_gancarski_index` share the same extra keyword arguments:

    - **`n_components`** is optional: if omitted it is inferred from the distinct values in `labels`, which then need not be contiguous or 0-indexed. Passing it explicitly (the historical required argument) behaves exactly as before.
    - **`metric`** is `'euclidean'` (default), `'manhattan'`, `'cosine'`, `'mahalanobis'` (pass the precision matrix as `VI`) or, for `silhouette_score` and `dunn_index` only, `'precomputed'` (pass an $(n,n)$ distance matrix as `X`). The other four indices average points into a cluster centroid, which a precomputed distance matrix does not support, and raise a clear `ValueError`. Passing the pooled precision of a fitted `GaussianMixture` (`torch.linalg.inv` of the weighted average of `covariances_`) as `VI` gives the GMM-matched version of an index.
    - **`noise_label`** marks noise points (for example `-1` for `HDBSCAN` output). Noise rows are dropped before the index is computed, and the result is then penalized for having discarded points. With $f=(n-n_{\text{noise}})/n$, indices where higher is better (Silhouette, CH, Dunn, WG) are *multiplied* by $f$, and indices where lower is better (Davies-Bouldin, WB) are *divided* by $f$.
    - **`chunk_size`** (silhouette and Dunn only) is the row-chunk size of the pairwise-distance computation. It bounds peak memory regardless of the number of samples (default `4096`), and chunking is exact, not an approximation.

    `dbcv` has its own simpler signature (`metric`, `noise_label=-1`). The cell below checks the noise rule and the other options on the fitted four-component mixture, flagging the 5% least likely points as noise.
    """)
    return


@app.cell
def _(ClusteringMetrics, GaussianMixture, MAX_ITER, SEED, X_tensor, device, n_components, n_features, pd, torch):
    gmm_ref = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=MAX_ITER, init_means='kmeans', device=device, random_state=SEED).fit(X_tensor)
    labels_ref = gmm_ref.predict(X_tensor)
    _ll = gmm_ref.score_samples(X_tensor)
    labels_noisy = torch.where(_ll < torch.quantile(_ll, 0.05), torch.full_like(labels_ref, -1), labels_ref)
    _keep = labels_noisy != -1
    _f = _keep.float().mean().item()
    _rows = []
    for _name, _fn, _up in [('silhouette_score', ClusteringMetrics.silhouette_score, True), ('calinski_harabasz_score', ClusteringMetrics.calinski_harabasz_score, True), ('dunn_index', ClusteringMetrics.dunn_index, True), ('wemmert_gancarski_index', ClusteringMetrics.wemmert_gancarski_index, True), ('davies_bouldin_index', ClusteringMetrics.davies_bouldin_index, False), ('wb_index', ClusteringMetrics.wb_index, False)]:
        _with_noise = _fn(X_tensor, labels_noisy, noise_label=-1)
        _dropped = _fn(X_tensor[_keep], labels_noisy[_keep])
        _expected = _dropped * _f if _up else _dropped / _f
        _rows.append({'index': _name, 'higher is better': _up, 'noise_label=-1': _with_noise, 'drop noise, then x f (or / f)': _expected, 'difference': abs(_with_noise - _expected)})
    pd.DataFrame(_rows)
    return gmm_ref, labels_noisy, labels_ref


@app.cell
def _(ClusteringMetrics, X_tensor, gmm_ref, labels_ref, torch):
    _pooled = (gmm_ref.weights_[:, None, None] * gmm_ref.covariances_).sum(0)
    _VI = torch.linalg.inv(_pooled)
    print('Chunking is exact, and other metrics plug in through the same call:')
    print(f'silhouette, default chunk_size       : {ClusteringMetrics.silhouette_score(X_tensor, labels_ref):.12f}')
    print(f'silhouette, chunk_size=256           : {ClusteringMetrics.silhouette_score(X_tensor, labels_ref, chunk_size=256):.12f}')
    print(f"silhouette, metric='manhattan'       : {ClusteringMetrics.silhouette_score(X_tensor, labels_ref, metric='manhattan'):.4f}")
    print(f"silhouette, metric='mahalanobis' (pooled GMM precision as VI): {ClusteringMetrics.silhouette_score(X_tensor, labels_ref, metric='mahalanobis', VI=_VI):.4f}")
    return


@app.cell
def _(gmm_ref, X_tensor, torch):
    _bic = gmm_ref.bic(X_tensor)
    print(f'BIC            : {_bic:.2f}')
    print(f"ICL (hard)     : {gmm_ref.icl(X_tensor):.2f}   (ICL - BIC = {gmm_ref.icl(X_tensor) - _bic:.2f} >= 0)")
    print(f"ICL (soft)     : {gmm_ref.icl(X_tensor, mode='soft'):.2f}")
    _n = X_tensor.shape[0]
    _ll_total = gmm_ref.score(X_tensor) * _n
    _K, _d = gmm_ref.n_components, gmm_ref.n_features
    _p = (_K - 1) + _K * _d + _K * _d * (_d + 1) // 2  # full covariance: K d (d+1)/2
    print(f'BIC from -2*loglik + p*ln(n) with p = {_p}: {-2 * _ll_total + _p * torch.log(torch.tensor(float(_n))):.2f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it: how the indices react to separation, $K$ and noise

    Three Gaussian blobs, a $k$-means clustering with the $K$ you choose, and every index above recomputed live with `ClusteringMetrics`. Move the **separation** slider to watch the indices degrade as the blobs overlap, change $K$ to see which indices are fooled, and switch on **noise** (30 uniform background points). In the interactive version of this notebook (see the link at the top of the page) the three controls are live; on the static page you see the default state.

    `tgmm` treats noise by *dropping* the points labelled as noise and scaling the index by $f=(n-n_{\text{noise}})/n$ (multiplied for higher-is-better, divided for lower-is-better indices). The last column of the table does that; the third lets $k$-means swallow the noise into the clusters. The right panel fits a full-covariance `GaussianMixture` for $K=1,\dots,6$ on all points and compares BIC with ICL.

    What to look for:

    * **Separation.** As the blobs overlap, Silhouette and Calinski-Harabasz fall and Davies-Bouldin and WB rise (Silhouette 0.76 at separation 8, 0.54 at 4, 0.34 at 1). Dunn is the most erratic (it rests on one closest pair and one widest cluster, and is not even monotone: 0.018, 0.009, 0.037 at separations 1, 2, 3), which is why it is ranked behind the others.
    * **$K$.** At separation 8 all six indices peak at the true $K=3$. When the blobs overlap (separation 1 or 2), Silhouette, Calinski-Harabasz and Wemmert-Gancarski still point at $K=3$, while Davies-Bouldin, WB and Dunn wander (they pick $K=6$ to $8$).
    * **Noise.** At the default separation 4 with $K=3$, dropping the known noise and scaling by $f$ improves Calinski-Harabasz (238 to 307), WB (1.68 to 1.16) and Dunn (0.019 to 0.034) a lot, while Silhouette (0.486 to 0.477) and Davies-Bouldin (0.689 to 0.694) barely move. The mixture cannot drop noise: it spends extra components on the background, and BIC and ICL both pick $K=5$ instead of 3 once noise is switched on.
    * **BIC vs. ICL.** ICL adds $-2\sum_i\log\max_k r_{ik}$ to BIC, which is near 0 for well-separated components and large when they overlap. At separation 3, BIC picks $K=2$ but ICL already prefers $K=1$.
    """)
    return


@app.cell
def _(mo):
    # docs: hide-cell
    ix_sep = mo.ui.slider(1.0, 8.0, step=0.5, value=4.0, label='cluster separation', debounce=True, show_value=True)
    ix_k = mo.ui.slider(2, 8, step=1, value=3, label='K for k-means', debounce=True, show_value=True)
    ix_noise = mo.ui.switch(label='add noise points')
    mo.hstack([ix_sep, ix_k, ix_noise], justify='start')
    return ix_k, ix_noise, ix_sep


@app.cell
def _(
    ClusteringMetrics,
    GaussianMixture,
    KMeans,
    MAX_ITER,
    SEED,
    ix_k,
    ix_noise,
    ix_sep,
    np,
    torch,
):
    _rng = np.random.default_rng(0)
    ix_X = np.vstack([_rng.normal(_c, 1.0, (80, 2)) for _c in np.array([[0, 0], [1, 0], [0.5, 0.87]]) * ix_sep.value])
    ix_y = np.repeat([0, 1, 2], 80)
    if ix_noise.value:
        ix_X = np.vstack([ix_X, _rng.uniform(ix_X.min(0) - 3, ix_X.max(0) + 3, (30, 2))])
        ix_y = np.r_[ix_y, -np.ones(30, int)]
    ix_lab = KMeans(n_clusters=ix_k.value, n_init=1, random_state=1).fit_predict(ix_X)
    ix_lab_noise = np.where(ix_y == -1, -1, ix_lab)
    _Xt = torch.tensor(ix_X, dtype=torch.float32)

    def _indices(_labels, _noise_label=None):
        _l = torch.tensor(_labels)
        _kw = dict(noise_label=_noise_label) if _noise_label is not None else {}
        return {
            'Silhouette': ClusteringMetrics.silhouette_score(_Xt, _l, **_kw),
            'Calinski-Harabasz': ClusteringMetrics.calinski_harabasz_score(_Xt, _l, **_kw),
            'Dunn': ClusteringMetrics.dunn_index(_Xt, _l, **_kw),
            'Wemmert-Gancarski': ClusteringMetrics.wemmert_gancarski_index(_Xt, _l, **_kw),
            'Davies-Bouldin': ClusteringMetrics.davies_bouldin_index(_Xt, _l, **_kw),
            'WB-Index': ClusteringMetrics.wb_index(_Xt, _l, **_kw),
        }

    ix_plain = _indices(ix_lab)
    ix_dropped = _indices(ix_lab_noise, -1)
    ix_crit = np.zeros((6, 2))
    for _kk in range(1, 7):
        _g = GaussianMixture(n_components=_kk, covariance_type='full', init_means='kmeans', max_iter=MAX_ITER, random_state=SEED).fit(_Xt)
        ix_crit[_kk - 1] = (_g.bic(_Xt), _g.icl(_Xt))
    return ix_X, ix_crit, ix_dropped, ix_lab_noise, ix_plain


@app.cell
def _(ix_X, ix_crit, ix_k, ix_lab_noise, ix_noise, plt):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 4.4))
    _axs[0].scatter(*ix_X.T, c=ix_lab_noise, cmap='viridis', s=12)
    _axs[0].set_aspect('equal')
    _axs[0].set_title(f'k-means, K = {ix_k.value}' + (' (noise: label -1)' if ix_noise.value else ''))
    for _j, _name in enumerate(('BIC', 'ICL')):
        _axs[1].plot(range(1, 7), ix_crit[:, _j], marker='o', label=_name)
        _axs[1].plot(1 + ix_crit[:, _j].argmin(), ix_crit[:, _j].min(), 'k*', ms=14)
    _axs[1].set_xlabel('number of GMM components K')
    _axs[1].set_ylabel('criterion (lower is better)')
    _axs[1].legend()
    _axs[1].set_title(f'BIC picks K = {1 + ix_crit[:, 0].argmin()}, ICL picks K = {1 + ix_crit[:, 1].argmin()}')
    plt.gcf()
    return


@app.cell
def _(ix_crit, ix_dropped, ix_plain, mo):
    _arrow = {'Silhouette': 'higher', 'Calinski-Harabasz': 'higher', 'Dunn': 'higher', 'Wemmert-Gancarski': 'higher', 'Davies-Bouldin': 'lower', 'WB-Index': 'lower'}
    mo.md('| index | better | noise swallowed by clusters | noise labelled -1 (dropped, scaled) |\n|---|---|---|---|\n' + '\n'.join(f'| {_n} | {_arrow[_n]} | {ix_plain[_n]:.3f} | {ix_dropped[_n]:.3f} |' for _n in ix_plain) + f'\n\nICL - BIC at K = 3: {ix_crit[2, 1] - ix_crit[2, 0]:.1f} (twice the assignment entropy; it grows with overlap).')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supervised Cluster Scores

    When ground-truth labels are available, we can evaluate a clustering with scores that measure the agreement between a predicted partition $V$ (`labels_pred`) and a reference partition $U$ (`labels_true`). Most are built from the **contingency table**

    $$
    n_{ij}=|\{x:\ u(x)=i,\ v(x)=j\}|,\qquad a_i=\sum_j n_{ij},\quad b_j=\sum_i n_{ij},\quad n=\sum_{ij}n_{ij},
    $$

    with row sums $a_i$ and column sums $b_j$. Cluster labels are arbitrary, so every score below is invariant to relabeling the clusters.

    > **Label format.** The supervised scores build the contingency table by indexing, so they expect **non-negative, 0-indexed integer labels** (for example from `gmm.predict`). To score density-based output that contains a `-1` noise label, drop or remap the noise rows first.

    ### Rand Index (RI) and Adjusted Rand Index (ARI)

    Count pairs of points: $TP$ are the pairs together in both partitions, $TN$ the pairs apart in both, and $FP$/$FN$ the disagreements (together only in the predicted / only in the true partition). Then

    $$
    \mathrm{RI}=\frac{TP+TN}{\binom n2}\in[0,1].
    $$

    The **Adjusted Rand Index** subtracts the Rand index's expected value under random labeling with the same cluster sizes (the generalized hypergeometric model) and rescales:

    $$
    \mathrm{ARI}=\frac{\sum_{ij}\binom{n_{ij}}2-\Big[\sum_i\binom{a_i}2\sum_j\binom{b_j}2\Big]\Big/\binom n2}{\tfrac12\Big[\sum_i\binom{a_i}2+\sum_j\binom{b_j}2\Big]-\Big[\sum_i\binom{a_i}2\sum_j\binom{b_j}2\Big]\Big/\binom n2}.
    $$

    ARI is at most 1 (perfect agreement), about 0 for a random labeling, and negative for worse-than-chance agreement.

    ### Mutual Information (MI), NMI and AMI

    With $p_{ij}=n_{ij}/n$, $p_i=a_i/n$, $p_j=b_j/n$ and entropies $H(U)=-\sum_ip_i\ln p_i$, $H(V)=-\sum_jp_j\ln p_j$ (all in nats),

    $$
    \mathrm{MI}(U,V)=\sum_{ij}p_{ij}\ln\frac{p_{ij}}{p_i\,p_j},\qquad
    \mathrm{NMI}(U,V)=\frac{2\,\mathrm{MI}(U,V)}{H(U)+H(V)}\in[0,1].
    $$

    MI is non-negative and bounded by $\min\{H(U),H(V)\}$, not by 1. The **Adjusted Mutual Information** corrects MI for chance,

    $$
    \mathrm{AMI}=\frac{\mathrm{MI}-\mathbb E[\mathrm{MI}]}{\tfrac12\big(H(U)+H(V)\big)-\mathbb E[\mathrm{MI}]},
    $$

    where the expectation is the **exact** one under the permutation (hypergeometric) model of Vinh, Epps & Bailey (2010):

    $$
    \mathbb E[\mathrm{MI}]=\sum_{i,j}\ \sum_{n_{ij}=\max(1,\,a_i+b_j-n)}^{\min(a_i,\,b_j)}
    \frac{n_{ij}}{n}\ln\frac{n\,n_{ij}}{a_ib_j}\;
    \frac{a_i!\,b_j!\,(n-a_i)!\,(n-b_j)!}{n!\,n_{ij}!\,(a_i-n_{ij})!\,(b_j-n_{ij})!\,(n-a_i-b_j+n_{ij})!}.
    $$

    The summand depends only on $(a_i,b_j)$, so `tgmm` groups equal cluster sizes and evaluates each distinct pair once, vectorised over $n_{ij}$ with log-gamma in float64 (adapted from scikit-learn's `expected_mutual_information`, BSD-3). The exact expectation is exposed as `ClusteringMetrics.expected_mutual_information(contingency)`. `adjusted_mutual_info_score` and `normalized_mutual_info_score` take `average_method` (`'arithmetic'` by default, `'min'`, `'max'`, `'geometric'`) and agree with scikit-learn to about $10^{-8}$ or better. NMI is $1$ for perfect agreement and $0$ for independent labelings; AMI is nominally at most 1 and can be slightly negative.

    ### Fowlkes-Mallows Index (FMI)

    The geometric mean of pairwise precision and recall,

    $$
    \mathrm{FMI}=\sqrt{\frac{TP}{TP+FP}\cdot\frac{TP}{TP+FN}},\qquad TP=\sum_{ij}\binom{n_{ij}}2\ .
    $$

    ### Homogeneity, Completeness and V-Measure

    Homogeneity asks "does each cluster contain only one true class?", and completeness asks "is each true class kept in one cluster?" (Rosenberg & Hirschberg, 2007):

    $$
    h=1-\frac{H(U\mid V)}{H(U)},\qquad
    c=1-\frac{H(V\mid U)}{H(V)},\qquad
    V_\beta=\frac{(1+\beta)\,h\,c}{\beta\,h+c},
    $$

    with the conventions $h=1$ when $H(U)=0$ and $c=1$ when $H(V)=0$. The V-measure is their weighted harmonic mean. `beta=1.0` (the default) weighs them equally and gives $2hc/(h+c)$, which equals NMI with arithmetic normalization. `beta>1` favours completeness and `beta<1` favours homogeneity.

    ### Normalized Information Distance (NID)

    $$
    \mathrm{NID}=1-\frac{I(U;V)}{\max\{H(U),H(V)\}}
    $$

    (Vinh, Epps & Bailey, 2010) is an information-theoretic *distance* between two partitions, in $[0,1]$: **lower is better** and $0$ for identical partitions.

    ### Purity

    $$
    \mathrm{purity}=\frac1n\sum_j\max_i n_{ij}
    $$

    is the fraction of points that belong to the majority true class of their predicted cluster. It never decreases as the number of clusters grows (a cluster per point has purity 1), so only compare it at the same $K$.

    **Interpretation guidelines**

    - **Higher is better:** RI, ARI, MI, NMI, AMI, FMI, homogeneity, completeness, V-measure and purity. **Lower is better:** NID.
    - **Range $[0,1]$:** RI, NMI, FMI, homogeneity, completeness, V-measure, purity and NID (AMI is nominally in $[0,1]$ and can dip slightly below 0 for near-random labelings).
    - **Other ranges:** ARI can be negative; MI is non-negative but not capped at 1.
    - **Perfect clustering:** RI, ARI, NMI, AMI, FMI, homogeneity, completeness, V-measure and purity all equal 1, and NID equals 0.
    """)
    return


@app.cell
def _(
    ClusteringMetrics,
    GaussianMixture,
    MAX_ITER,
    SEED,
    X_tensor,
    device,
    n_components,
    n_features,
    plot_gmm,
    plt,
    y_tensor,
):
    from sklearn.mixture import GaussianMixture as SklearnGMM
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score, adjusted_mutual_info_score, completeness_score, v_measure_score, rand_score
    _gmm = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=MAX_ITER, init_means='kmeans', device=device)
    _gmm.fit(X_tensor)
    labels_pred_1 = _gmm.predict(X_tensor)
    metrics_to_compare = ['rand_score', 'adjusted_rand_score', 'mutual_info_score', 'normalized_mutual_info_score', 'adjusted_mutual_info_score', 'fowlkes_mallows_score', 'homogeneity_score', 'completeness_score', 'v_measure_score', 'silhouette_score', 'davies_bouldin_index', 'calinski_harabasz_score', 'purity_score', 'dunn_index', 'bic_score', 'aic_score']
    scores_torch = ClusteringMetrics.evaluate_clustering(_gmm, X_tensor, true_labels=y_tensor, metrics=metrics_to_compare)
    scores_torch['bic_score'] = _gmm.bic(X_tensor)
    scores_torch['aic_score'] = _gmm.aic(X_tensor)
    X_np = X_tensor.cpu().numpy()
    y_np = y_tensor.cpu().numpy()
    labels_pred_np = labels_pred_1.cpu().numpy()
    sk_gmm = SklearnGMM(n_components=n_components, covariance_type='full', max_iter=MAX_ITER, init_params='kmeans', random_state=SEED, tol=1e-05, reg_covar=1e-07)
    sk_gmm.fit(X_np)
    scores_sklearn = {'rand_score': rand_score(y_np, labels_pred_np), 'adjusted_rand_score': adjusted_rand_score(y_np, labels_pred_np), 'mutual_info_score': mutual_info_score(y_np, labels_pred_np), 'normalized_mutual_info_score': normalized_mutual_info_score(y_np, labels_pred_np), 'adjusted_mutual_info_score': adjusted_mutual_info_score(y_np, labels_pred_np), 'fowlkes_mallows_score': fowlkes_mallows_score(y_np, labels_pred_np), 'homogeneity_score': homogeneity_score(y_np, labels_pred_np), 'completeness_score': completeness_score(y_np, labels_pred_np), 'v_measure_score': v_measure_score(y_np, labels_pred_np), 'silhouette_score': silhouette_score(X_np, labels_pred_np), 'davies_bouldin_index': davies_bouldin_score(X_np, labels_pred_np), 'calinski_harabasz_score': calinski_harabasz_score(X_np, labels_pred_np)}
    sk_bic = sk_gmm.bic(X_np)
    sk_aic = sk_gmm.aic(X_np)
    rows = []
    for metric in metrics_to_compare:
        torch_val = scores_torch.get(metric, None)
        if metric in scores_sklearn:
            sklearn_val = scores_sklearn.get(metric)
        elif metric == 'bic_score':
            sklearn_val = sk_bic
        elif metric == 'aic_score':
            sklearn_val = sk_aic
        else:
            sklearn_val = None
        if torch_val is not None and sklearn_val is not None:
            abs_diff = abs(torch_val - sklearn_val)
            rel_diff = abs_diff / abs(sklearn_val) * 100 if sklearn_val != 0 else None
        else:
            abs_diff = None
            rel_diff = None
        rows.append({'Metric': metric, 'Torch Score': torch_val, 'Sklearn Score': sklearn_val, 'Absolute Difference': abs_diff, 'Relative Difference (%)': rel_diff})
    _fig, ax = plt.subplots()
    plot_gmm(X=X_np, gmm=_gmm, true_labels=y_tensor, match_labels_to_true=True, ax=ax, title='GMM Predictions', show_ellipses=True, show_incorrect_predictions=True, ellipse_fill=False, ellipse_std_devs=[3], point_size=10)
    plt.gcf()
    return labels_pred_1, rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison with scikit-learn

    The table compares the scores computed by `tgmm`'s implementation with scikit-learn's, to validate correctness and show where they differ. `evaluate_clustering(gmm, X, true_labels=None, metrics=None)` computes a requested subset of the scores for a fitted model in one call (the supervised ones only when `true_labels` is given), which is how the `tgmm` column is produced. Small differences may occur because of:

    - different numerical precision and slightly different algorithms;
    - different handling of edge cases (for example single-point clusters);
    - BIC/AIC, which come from two independently fitted models (`tgmm`'s and scikit-learn's) and so agree only approximately.

    **Key observations**

    - **Information-theoretic scores** (MI, NMI, AMI) agree with scikit-learn to numerical precision.
    - **Pairwise scores** (RI, ARI, FMI) match exactly for identical cluster assignments.
    - **Geometric indices** (Silhouette, Davies-Bouldin, Calinski-Harabasz) may show small differences because of how distances are computed.
    """)
    return


@app.cell
def _(pd, rows):
    df_metrics = pd.DataFrame(rows)
    df_metrics
    return


@app.cell
def _(ClusteringMetrics, X_tensor, labels_pred_1, np, pd, torch, y_tensor):
    from sklearn.metrics.cluster import contingency_matrix as _sk_contingency
    _yt = y_tensor.clone().detach().to(dtype=torch.long)
    _yp = labels_pred_1.clone().detach().to(dtype=torch.long)
    _cont = torch.tensor(_sk_contingency(_yt.cpu().numpy(), _yp.cpu().numpy()), dtype=torch.float64)
    _n = _cont.sum()
    _a, _b = _cont.sum(1), _cont.sum(0)
    _H = lambda c: -(c / _n * torch.log(c / _n)).sum()
    _HU, _HV = _H(_a), _H(_b)
    _mi = ClusteringMetrics.mutual_info_score(_yt, _yp)
    _emi = ClusteringMetrics.expected_mutual_information(_cont)
    _ami_by_hand = (_mi - _emi) / (0.5 * (_HU + _HV) - _emi)
    _h = ClusteringMetrics.homogeneity_score(_yt, _yp)
    _c = ClusteringMetrics.completeness_score(_yt, _yp)
    _rows = [
        ('MI', _mi, 'sum p_ij ln(p_ij / p_i p_j)'),
        ('E[MI] (exact, hypergeometric)', _emi, 'expected_mutual_information(contingency)'),
        ('AMI, by hand from MI, E[MI], H(U), H(V)', float(_ami_by_hand), 'matches adjusted_mutual_info_score'),
        ('AMI, adjusted_mutual_info_score', ClusteringMetrics.adjusted_mutual_info_score(_yt, _yp), ''),
        ('NMI', ClusteringMetrics.normalized_mutual_info_score(_yt, _yp), '2 MI / (H(U) + H(V))'),
        ('V-measure, beta=1', ClusteringMetrics.v_measure_score(_yt, _yp), 'equals NMI (arithmetic)'),
        ('V-measure, beta=0.5 (favours homogeneity)', ClusteringMetrics.v_measure_score(_yt, _yp, beta=0.5), ''),
        ('V-measure, beta=2 (favours completeness)', ClusteringMetrics.v_measure_score(_yt, _yp, beta=2.0), ''),
        ('homogeneity h', _h, ''),
        ('completeness c', _c, ''),
        ('NID (lower is better)', ClusteringMetrics.normalized_information_distance(_yt, _yp), '1 - MI / max(H(U), H(V))'),
        ('purity', ClusteringMetrics.purity_score(_yt, _yp), 'majority-class fraction'),
    ]
    pd.DataFrame(_rows, columns=['score', 'value', 'note'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Purity only goes up with $K$

    Purity never decreases as the number of clusters grows, so on its own it can never tell you to use fewer components. ARI and NMI, in contrast, are penalized for splitting a true class. Using the partitions of the sweep above (predicted labels of the fitted mixture for $K=2,\dots,10$, true $K=4$):
    """)
    return


@app.cell
def _(ClusteringMetrics, labels_by_k, plt, torch, y_tensor):
    _yt = y_tensor.clone().detach().to(dtype=torch.long)
    _ks = sorted(labels_by_k)
    _pur = [ClusteringMetrics.purity_score(_yt, labels_by_k[_k].long()) for _k in _ks]
    _ari = [ClusteringMetrics.adjusted_rand_score(_yt, labels_by_k[_k].long()) for _k in _ks]
    _nmi = [ClusteringMetrics.normalized_mutual_info_score(_yt, labels_by_k[_k].long()) for _k in _ks]
    plt.figure(figsize=(6, 3.6))
    plt.plot(_ks, _pur, 'o-', label='purity')
    plt.plot(_ks, _ari, 's-', label='ARI')
    plt.plot(_ks, _nmi, '^-', label='NMI')
    plt.axvline(4, color='r', linestyle='--', label='true K')
    plt.xlabel('Number of components')
    plt.ylabel('score vs. true labels')
    plt.grid(True)
    plt.legend()
    plt.title('Purity is monotone in K; ARI and NMI are not')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confusion Matrix and Classification Report

    ### Understanding the confusion matrix

    The confusion matrix provides a detailed breakdown of correct and incorrect predictions for each class. Clustering algorithms assign arbitrary label numbers, so before computing it we use **label matching** (`match_predicted_to_true_labels`, the Hungarian algorithm) to align predicted cluster labels with true class labels. Without it the diagonal does not mean "correct".

    `ClusteringMetrics.confusion_matrix(labels_true, labels_pred, labels=None)` indexes rows (true labels) and columns (predicted labels) by the sorted **union** of the labels present in `labels_true` and `labels_pred`, so a predicted id that never occurs in `labels_true` gets its own (all-zero) row and a column instead of being silently dropped. Pass `labels=[...]` to fix the order (and, as in scikit-learn, to ignore samples with other labels).

    The heatmap below shows the confusion matrix as percentages, normalized by the true class sizes.

    ### Classification report

    `classification_report` treats each true class $c$ one-vs-rest, with $TP_c$, $FP_c$ and $FN_c$:

    $$
    \text{precision}=\frac{TP}{TP+FP},\quad
    \text{recall}=\frac{TP}{TP+FN},\quad
    F_1=\frac{2PR}{P+R},\quad
    \text{Jaccard}=\frac{TP}{TP+FP+FN},
    $$

    and also reports the **support** (the number of true instances of the class) and a `roc_auc`. Like the confusion matrix it matches predicted and true labels by *value*, so align cluster ids to class ids first. The classes are the sorted union of the true and predicted labels, and a label that is only ever predicted is listed with support 0.
    """)
    return


@app.cell
def _(
    ClusteringMetrics,
    labels_pred_1,
    match_predicted_to_true_labels,
    plt,
    sns,
    torch,
    y_tensor,
):
    matched_pred = match_predicted_to_true_labels(y_tensor, labels_pred_1)
    cm = ClusteringMetrics.confusion_matrix(y_tensor.clone().detach().to(dtype=torch.long), matched_pred.clone().detach().to(dtype=torch.long))
    cm_np = cm.cpu().numpy().astype(float)
    cm_percent = (cm_np.T / cm_np.sum(axis=1)).T * 100
    plt.figure()
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues')
    plt.title('Confusion Matrix (%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.gcf()
    return cm_np, matched_pred


@app.cell
def _(ClusteringMetrics, matched_pred, pd, torch, y_tensor):
    report = ClusteringMetrics.classification_report(y_tensor.clone().detach().to(dtype=torch.long), matched_pred.clone().detach().to(dtype=torch.long))
    report_df = pd.DataFrame(report).T
    report_df
    return (report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **A note on `roc_auc`.** `ClusteringMetrics.roc_auc_score` is the rank-based (Mann-Whitney) AUC with average ranks for ties,

    $$
    \mathrm{AUC}=\frac{\sum_{i:\,y_i=1}\operatorname{rank}(s_i)-n_+(n_++1)/2}{n_+n_-},
    $$

    so it equals `sklearn.metrics.roc_auc_score` and does not depend on the order of the samples. Inside `classification_report` it is applied to the *binary* one-vs-rest **hard** predictions, where it reduces to $(\mathrm{TPR}+\mathrm{TNR})/2$, the balanced accuracy. For a real AUC pass scores or probabilities to `roc_auc_score` directly. It is `1.0` by convention when only one class is present. The check below recomputes each class's `roc_auc` from the confusion matrix:
    """)
    return


@app.cell
def _(cm_np, report):
    _n = cm_np.sum()
    _errs = []
    for _c in range(cm_np.shape[0]):
        _tp = cm_np[_c, _c]
        _fn = cm_np[_c].sum() - _tp
        _fp = cm_np[:, _c].sum() - _tp
        _tn = _n - _tp - _fn - _fp
        _bal_acc = 0.5 * (_tp / (_tp + _fn) + _tn / (_tn + _fp))
        _errs.append(abs(_bal_acc - report[_c]['roc_auc'] if _c in report else abs(_bal_acc - report[str(_c)]['roc_auc'])))
    print(f'max |balanced accuracy from the confusion matrix - reported roc_auc| over classes: {max(_errs):.2e}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing covariance structures

    The same metrics let you compare several mixtures on one dataset. Here the four-component mixture is fitted with different covariance structures (see [02 -- Covariance Types](02_covariance_types.ipynb)) and scored with a geometric index, the information criteria and, because the truth is known, ARI. The true blobs are one spherical, one spherical and small, one diagonal and one full, so no single restricted type fits all of them and the full type should score best on likelihood and BIC.
    """)
    return


@app.cell
def _(ClusteringMetrics, GaussianMixture, MAX_ITER, SEED, X_tensor, device, n_components, n_features, pd, y_tensor):
    _rows = []
    for _cov_type in ['full', 'diag', 'spherical', 'tied_full']:
        _g = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type=_cov_type, max_iter=MAX_ITER, init_means='kmeans', device=device, random_state=SEED).fit(X_tensor)
        _lab = _g.predict(X_tensor)
        _rows.append({
            'covariance_type': _cov_type,
            'silhouette': ClusteringMetrics.silhouette_score(X_tensor, _lab, n_components=n_components),
            'davies_bouldin': ClusteringMetrics.davies_bouldin_index(X_tensor, _lab, n_components=n_components),
            'log_likelihood': _g.score(X_tensor) * X_tensor.shape[0],
            'bic': _g.bic(X_tensor),
            'aic': _g.aic(X_tensor),
            'ARI': ClusteringMetrics.adjusted_rand_score(y_tensor.long(), _lab.long()),
        })
    pd.DataFrame(_rows).set_index('covariance_type')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KL Divergence Between Mixtures

    The **KL divergence** $D_{KL}(P\parallel Q)$ measures how a distribution $P$ diverges from a reference distribution $Q$:

    $$
    D_{KL}(P\parallel Q)=\int p(x)\log\frac{p(x)}{q(x)}\,dx=\mathbb E_{x\sim p}\big[\log p(x)-\log q(x)\big].
    $$

    It has no closed form for mixtures, so `ClusteringMetrics.kl_divergence_gmm(gmm_p, gmm_q, n_samples=10000)` estimates it by Monte Carlo,

    $$
    D_{KL}(P\parallel Q)\approx\frac1N\sum_{i=1}^N\big[\log p(x_i)-\log q(x_i)\big],\qquad x_i\sim P,
    $$

    averaging over samples drawn from the first mixture. Properties:

    - $D_{KL}(P\parallel Q)=0$ when $P=Q$, and it is always non-negative.
    - It is **asymmetric**: $D_{KL}(P\parallel Q)\ne D_{KL}(Q\parallel P)$ in general.
    - The Monte Carlo estimate has noise proportional to $1/\sqrt{N}$ (`n_samples`).
    - **Lower values** indicate more similar distributions.

    The plot below keeps the reference mixture fixed at the true number of components and varies the number of components of the test mixture.
    """)
    return


@app.cell
def _(
    ClusteringMetrics,
    GaussianMixture,
    MAX_ITER,
    X_tensor,
    device,
    n_components,
    n_features,
    np,
    plt,
    torch,
    tqdm,
):
    RUN_KL_DIVERGENCE = True
    if RUN_KL_DIVERGENCE:
        print('Computing KL(p||q) for different numbers of components...')
        gmm_true = GaussianMixture(n_features=n_features, n_components=n_components, covariance_type='full', max_iter=MAX_ITER, init_means='kmeans', device=device)
        gmm_true.fit(X_tensor)
        test_range = np.arange(1, 15)
        kl_vals = torch.zeros(len(test_range), device=device)
        for _i, _n in tqdm(enumerate(test_range), total=len(test_range)):
            gmm_test = GaussianMixture(n_features=n_features, n_components=_n, covariance_type='full', max_iter=MAX_ITER, init_means='kmeans', device=device)
            gmm_test.fit(X_tensor)
            kl_vals[_i] = ClusteringMetrics.kl_divergence_gmm(gmm_true, gmm_test, n_samples=10000)
        plt.figure()
        plt.plot(test_range, kl_vals.cpu().numpy(), marker='o')
        plt.yscale('log')
        plt.title('KL Divergence: True GMM vs. Various n_components')
        plt.xlabel('Number of Components')
        plt.ylabel('KL Divergence (log scale)')
        plt.grid(True)
    plt.gcf() if RUN_KL_DIVERGENCE else None
    return gmm_test, gmm_true


@app.cell
def _(ClusteringMetrics, gmm_test, gmm_true, torch):
    # gmm_test is the last (14-component) fit of the sweep; compare both directions and the Monte Carlo noise
    torch.manual_seed(0)
    _pq = ClusteringMetrics.kl_divergence_gmm(gmm_true, gmm_test, n_samples=10000)
    _qp = ClusteringMetrics.kl_divergence_gmm(gmm_test, gmm_true, n_samples=10000)
    _small = torch.tensor([ClusteringMetrics.kl_divergence_gmm(gmm_true, gmm_test, n_samples=100) for _ in range(20)])
    _large = torch.tensor([ClusteringMetrics.kl_divergence_gmm(gmm_true, gmm_test, n_samples=10000) for _ in range(20)])
    print(f'KL(true || 14-component) = {_pq:.4f}   KL(14-component || true) = {_qp:.4f}   (asymmetric)')
    print(f'std of the estimate over 20 repeats: n_samples=100 -> {_small.std():.4f}, n_samples=10000 -> {_large.std():.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other helpers

    - `ClusteringMetrics.rand_score` and `ClusteringMetrics.mutual_info_score` are the unadjusted building blocks of ARI and NMI.
    - `ClusteringMetrics.expected_mutual_information(contingency)` is the exact chance correction used by AMI (checked above).
    - `ClusteringMetrics.roc_auc_score` is the rank-based AUC (see the note above).

    ## Summary

    - Unsupervised indices (Silhouette, Davies-Bouldin, Calinski-Harabasz, WB, Wemmert-Gancarski, DBCV, Dunn) and information criteria (BIC, AIC, ICL) can be swept over `n_components`, but they need not agree on the best $K$. Prefer WB, Wemmert-Gancarski and DBCV over Dunn.
    - The unsupervised indices share `metric`, `noise_label` and `chunk_size`. Noise points are dropped and the result is scaled by $f=(n-n_{\text{noise}})/n$, multiplied for higher-is-better and divided for lower-is-better indices.
    - BIC, AIC and ICL are methods of the fitted mixture, not of `ClusteringMetrics`. ICL $\ge$ BIC, with the gap growing as components overlap.
    - Supervised scores compare a clustering with ground truth. They are invariant to relabeling, expect non-negative 0-indexed labels, and `tgmm`'s versions can be checked directly against scikit-learn (AMI uses the exact chance correction). Purity only goes up with $K$.
    - Cluster indices are arbitrary, so predictions are matched to true classes before building a confusion matrix or classification report. The report's `roc_auc` is the balanced accuracy of the hard predictions.
    - `kl_divergence_gmm` estimates $D_{KL}(P\parallel Q)$ between two fitted mixtures by Monte Carlo sampling; it is asymmetric and noisy.

    **Next:** [13 -- Model Selection](13_model_selection.ipynb) for BIC, AIC and ICL sweeps with `select_model`, [12 -- HDBSCAN](12_hdbscan.ipynb) for density-based clusters scored with DBCV, or [10 -- PCA Plotting](10_pca_plotting.ipynb).
    """)
    return


if __name__ == "__main__":
    app.run()
