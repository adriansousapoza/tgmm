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
    return FAST, MAX_ITER


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # HDBSCAN Clustering

    `tgmm.HDBSCAN` is a PyTorch implementation of **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise; Campello, Moulavi & Sander, 2013), following the `scikit-learn-contrib/hdbscan` reference implementation. Unlike a Gaussian mixture it needs no component count and makes no assumption about cluster shape: it finds clusters as dense regions, can recover non-convex shapes, and labels sparse points as **noise** (`-1`).
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
    from matplotlib.patches import Ellipse
    from sklearn.datasets import make_moons
    from scipy.cluster.hierarchy import fcluster
    from scipy.stats import chi2
    from sklearn.metrics import adjusted_rand_score

    from tgmm import GaussianMixture, HDBSCAN, generate_gmm_data, plot_gmm, gmm_proxy
    from tgmm.plotting import CATEGORICAL_PALETTE
    from tutorial_data import make_default_data

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # --- figure style (see notebooks/README.md) ---
    DOCS_RC = {'figure.dpi': 100, 'figure.titlesize': 12, 'axes.titlesize': 10.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
               'legend.fontsize': 8, 'lines.linewidth': 1.4, 'lines.markersize': 5}
    plt.rcParams.update(DOCS_RC)
    N_STD95 = float(np.sqrt(chi2.ppf(0.95, 2)))  # a 2-D Gaussian's 95% region is this many standard deviations out (about 2.45, not 2)
    SEED = 2
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f'Using device: {device}')
    return (
        CATEGORICAL_PALETTE,
        Ellipse,
        GaussianMixture,
        HDBSCAN,
        N_STD95,
        SEED,
        adjusted_rand_score,
        device,
        fcluster,
        generate_gmm_data,
        gmm_proxy,
        make_default_data,
        make_moons,
        np,
        plot_gmm,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How HDBSCAN Works

    The algorithm, as implemented here:

    1. **Core distance.** $\mathrm{core}_k(a)$ is the distance from $a$ to its `min_samples`-th nearest neighbour (counting $a$ itself).
    2. **Mutual reachability distance.** $d_{\mathrm{mreach}}(a,b) = \max\{\mathrm{core}_k(a),\ \mathrm{core}_k(b),\ d(a,b)\}$ (divided by `alpha` if `alpha != 1`). This pushes sparse points away from everything, so noise does not bridge dense regions.
    3. **Minimum spanning tree and hierarchy.** A minimum spanning tree of the complete mutual-reachability graph (Prim's algorithm on the dense $n\times n$ matrix) is converted to a single-linkage hierarchy.
    4. **Condensed tree.** Walking the hierarchy from the root down, a split only counts as a real cluster split if both children have at least `min_cluster_size` points; otherwise the smaller side is treated as points "falling out" of the parent cluster. Cluster density is tracked as $\lambda = 1/d_{\mathrm{mreach}}$.
    5. **Stability-based extraction.** Each cluster $C$ gets a stability $\sigma(C) = \sum_{p\in C}\left(\lambda_p - \lambda_{\mathrm{birth}}(C)\right)$, where $\lambda_p$ is the density level at which point $p$ leaves $C$. Clusters are selected bottom-up by the *excess of mass* rule: keep a cluster if its stability is at least the summed stability of its descendants that would replace it. Unselected points are noise, and `probabilities_` gives each point's membership strength $\lambda_p/\lambda_{\max}$.

    Because step 3 uses the dense pairwise distance matrix, memory grows as $O(n^2)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 1: Non-Convex Clusters (Moons Dataset)

    HDBSCAN excels at finding non-convex clusters that traditional methods like K-Means struggle with.
    """)
    return


@app.cell
def _(HDBSCAN, SEED, device, make_moons, plot_gmm, plt, torch):
    # Generate two interleaving half circles
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=SEED)
    X_moons_tensor = torch.from_numpy(X_moons).float().to(device)
    hdbscan_moons = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', device=device)
    # Fit HDBSCAN
    labels_moons = hdbscan_moons.fit_predict(X_moons_tensor)
    print(f'Number of clusters found: {hdbscan_moons.n_clusters_}')
    print(f'Number of noise points: {(labels_moons == -1).sum().item()}')
    print(f'Cluster sizes: {[(labels_moons == i).sum().item() for i in range(hdbscan_moons.n_clusters_)]}')
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_gmm(X=X_moons, gmm=None, ax=_ax1, true_labels=y_moons, color_by_cluster=True, show_ellipses=False, show_means=False, point_size=8, point_alpha=0.5, title='Original Moons Dataset (Ground Truth)', legend=False)
    _labels_np = labels_moons.cpu().numpy()
    _non_noise = _labels_np != -1
    plot_gmm(X=X_moons[_non_noise], gmm=None, ax=_ax2, true_labels=_labels_np[_non_noise], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=8, point_alpha=0.5, title=f'HDBSCAN Clustering ({hdbscan_moons.n_clusters_} clusters)', legend=False)
    if (~_non_noise).any():
        _ax2.scatter(X_moons[~_non_noise, 0], X_moons[~_non_noise, 1], c='0.4', marker='x', s=14, linewidths=1, label='Noise')
    _ax2.legend()
    # Plot results
    plt.tight_layout()
    # Original data
    # HDBSCAN clustering
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 2: Noise Detection Accuracy

    `min_samples` and `min_cluster_size` control how aggressively HDBSCAN labels
    points as noise. Here we generate data with a known 15% of points injected as
    uniform noise, so we can check the detector against ground truth: which points
    did it correctly flag as noise, and where did it get the noise/cluster call
    wrong (false positives: real points wrongly flagged as noise; false negatives:
    actual noise points that slipped through as a cluster)?

    (A numerical accuracy check against scikit-learn's HDBSCAN implementation,
    confirming the two agree once `min_samples` is defined the same way in both,
    lives in `tests/test_hdbscan.py` rather than here.)
    """)
    return


@app.cell
def _(HDBSCAN, SEED, device, generate_gmm_data, np, plot_gmm, plt):
    # Generate test data with 15% uniform noise
    n_samples_test = [200, 150, 250]
    centers_test = [np.array([0, 0]), np.array([4, 4]), np.array([0, 4])]
    covs_test = [0.5 * np.eye(2), 0.4 * np.eye(2), 0.6 * np.eye(2)]
    X_test, y_test = generate_gmm_data(centers_test, covs_test, n_samples_test, device=device, random_state=SEED, noise_ratio=0.15, noise_scale=1.5)
    print(f'Data generated: {X_test.shape[0]} total points')
    print(f'  - Gaussian points: {sum(n_samples_test)}')
    print(f'  - Noise points (ground truth): {(y_test == -1).sum().item()}\n')
    hdbscan_torch = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', device=device)  # Add 15% uniform noise
    labels_torch = hdbscan_torch.fit_predict(X_test)  # Noise spread 1.5x the data range
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    labels_torch_np = labels_torch.cpu().numpy()
    true_is_noise = y_test_np == -1
    pred_is_noise = labels_torch_np == -1
    correct_mask = true_is_noise == pred_is_noise
    # PyTorch HDBSCAN
    false_positive_mask = pred_is_noise & ~true_is_noise
    false_negative_mask = ~pred_is_noise & true_is_noise
    print(f'Detected clusters: {hdbscan_torch.n_clusters_}')
    print(f'Correct noise/cluster calls: {correct_mask.sum()}/{len(y_test_np)}')
    print(f'False positives (wrongly flagged as noise): {false_positive_mask.sum()}')
    print(f'False negatives (noise missed): {false_negative_mask.sum()}')
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(11, 3.8))
    plot_gmm(X=X_test_np[~true_is_noise], gmm=None, ax=_ax1, true_labels=y_test_np[~true_is_noise], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=8, point_alpha=0.5, legend=False, title='Ground Truth (with uniform noise)', legend_labels=[f'Cluster {i}' for i in range(3)])
    if true_is_noise.any():
        _ax1.scatter(X_test_np[true_is_noise, 0], X_test_np[true_is_noise, 1], c='0.4', marker='x', s=14, linewidths=1, label='Ground Truth Noise')
    _ax1.legend()
    plot_gmm(X=X_test_np[~pred_is_noise], gmm=None, ax=_ax2, true_labels=labels_torch_np[~pred_is_noise], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=8, point_alpha=0.5, legend=False, title=f'PyTorch HDBSCAN ({hdbscan_torch.n_clusters_} clusters)')
    # Classify each point's noise/cluster call against ground truth
    if pred_is_noise.any():
        _ax2.scatter(X_test_np[pred_is_noise, 0], X_test_np[pred_is_noise, 1], c='0.4', marker='x', s=14, linewidths=1, label='Detected Noise')
    _ax2.legend()
    _ax3.scatter(X_test_np[correct_mask, 0], X_test_np[correct_mask, 1], c='0.75', s=8, alpha=0.5, label='Correct')  # flagged as noise, actually a real point
    if false_positive_mask.any():  # actual noise, missed (kept as cluster)
        _ax3.scatter(X_test_np[false_positive_mask, 0], X_test_np[false_positive_mask, 1], c='orange', marker='x', s=14, linewidths=1, label='False positive (wrongly flagged noise)')
    if false_negative_mask.any():
        _ax3.scatter(X_test_np[false_negative_mask, 0], X_test_np[false_negative_mask, 1], c='black', marker='x', s=14, linewidths=1, label='False negative (noise missed)')
    _ax3.set_title('Noise Detection Accuracy')
    _ax3.set_xlabel('Feature 1')
    _ax3.set_ylabel('Feature 2')
    # Visualize
    _ax3.legend()
    plt.tight_layout()
    # Ground truth
    # Noise detection accuracy: correct vs. false positive vs. false negative. This is a
    # binary correctness diagnostic rather than a per-cluster labeling, so it doesn't map
    # onto plot_gmm's cluster-coloring model -- left as a direct scatter.
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 3: Effect of min_cluster_size Parameter

    The `min_cluster_size` parameter controls the minimum size for a cluster. Larger values result in fewer, larger clusters.
    """)
    return


@app.cell
def _(HDBSCAN, SEED, device, generate_gmm_data, np, plot_gmm, plt):
    # Generate complex data with noise
    n_samples_complex = [150, 100, 80, 50]
    centers_complex = [np.array([0, 0]), np.array([3, 3]), np.array([0, 3]), np.array([3, 0])]
    # Variance 0.16 per axis (std 0.4): four well-separated blobs of 150/100/80/50 points, so that raising
    # min_cluster_size past 50, 80 and 100 removes the smallest clusters one by one.
    covs_complex = [0.16 * np.eye(2) for _ in range(4)]
    X_complex, y_complex = generate_gmm_data(centers_complex, covs_complex, n_samples_complex, device=device, random_state=SEED)
    # The smallest true cluster has 50 points, so anything below that won't change what counts as a
    # valid cluster: 10 and 30 would look identical. 70 and 100 cross the size of progressively larger true clusters.
    min_cluster_sizes = [10, 70, 100]
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.8))
    X_np = X_complex.cpu().numpy()
    for _ax, min_size in zip(_axes, min_cluster_sizes):
        hdbscan_test = HDBSCAN(min_cluster_size=min_size, min_samples=5, device=device)
        labels_test = hdbscan_test.fit_predict(X_complex)
        _labels_np = labels_test.cpu().numpy()
        _non_noise = _labels_np != -1
        plot_gmm(X=X_np[_non_noise], gmm=None, ax=_ax, true_labels=_labels_np[_non_noise], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=8, point_alpha=0.5, legend=False, title=f'min_cluster_size={min_size}\n({hdbscan_test.n_clusters_} clusters, {(~_non_noise).sum()} noise)')
        if (~_non_noise).any():
            _ax.scatter(X_np[~_non_noise, 0], X_np[~_non_noise, 1], c='0.4', marker='x', s=14, linewidths=1, label='Noise')
        _ax.legend()
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 4: Different Distance Metrics

    HDBSCAN supports multiple distance metrics, and the "right" one depends on what
    actually distinguishes your clusters. Below, three synthetic datasets are each
    built so that a *different* metric is the one that recovers the true clusters
    correctly, then all three metrics are run on all three datasets (same
    `min_cluster_size`/`min_samples` throughout, so the comparison isolates the
    effect of the metric alone):

    - **Collinear blobs** — clusters differ only in *distance from the origin*, not
      direction. Cosine distance ignores magnitude entirely, so it can't tell these
      apart (everything looks like noise); Euclidean/Manhattan separate them easily.
    - **Rectilinear blob chains** — two "L-shaped" chains of uniform *square*
      blobs (each chain running along a single axis, then turning at a corner)
      plus a third, standalone square blob off on its own. Manhattan's unit ball
      is itself a diamond aligned with the axes, so it bridges the small gaps
      between axis-aligned square blobs cleanly and recovers all 3 true clusters
      perfectly. Euclidean's circular neighborhoods don't extend as efficiently
      across a flat, straight gap, so each chain fragments into one cluster per
      blob (the standalone blob, needing no bridging, is the one thing it still
      gets right).
    - **Angular wedges** — clusters differ only in *direction from the origin*, with
      overlapping radius ranges. Cosine distance is direction-only, so it separates
      these cleanly; Euclidean/Manhattan get confused because points from different
      wedges can be closer in raw distance than points within the same wedge.
    """)
    return


@app.cell
def _(HDBSCAN, adjusted_rand_score, device, np, plot_gmm, plt, torch):
    def make_wedges(angles, r_range, jitter_deg, n_per, seed):
        """Clusters separated by direction from the origin, overlapping in radius."""
        rs = np.random.RandomState(seed)

        def wedge(angle_deg):
            angle = np.deg2rad(angle_deg + rs.uniform(-jitter_deg, jitter_deg, n_per))
            r = rs.uniform(*r_range, n_per)
            return np.stack([r * np.cos(angle), r * np.sin(angle)], axis=1)
        X = np.vstack([wedge(a) for a in angles])
        y = np.repeat(range(len(angles)), n_per)
        return (X, y)

    def make_collinear_blobs(r_centers, r_std, perp_std, angle_deg, n_per, seed):
        """Clusters separated by distance from the origin, along the same direction."""
        rs = np.random.RandomState(seed)
        angle = np.deg2rad(angle_deg)
        direction = np.array([np.cos(angle), np.sin(angle)])
        perp = np.array([-np.sin(angle), np.cos(angle)])

        def blob(r_center):
            r = rs.normal(r_center, r_std, n_per)
            p = rs.normal(0, perp_std, n_per)
            return r[:, None] * direction + p[:, None] * perp
        X = np.vstack([blob(rc) for rc in r_centers])
        y = np.repeat(range(len(r_centers)), n_per)
        return (X, y)
    X_collinear, y_collinear = make_collinear_blobs(r_centers=[3, 9, 15], r_std=0.6, perp_std=0.6, angle_deg=35, n_per=150, seed=11)

    def make_rectilinear_chains(step, side, n_per_blob, n_arm, seed, extra_center):
        """Two L-shaped chains of uniform square blobs (one runs horizontally,
        the other picks up vertically at a diagonal corner from the first) plus
        a third, standalone square blob that belongs to neither chain."""
        rs = np.random.RandomState(seed)

        def square_blob(cx, cy):
            return rs.uniform(-side / 2, side / 2, size=(n_per_blob, 2)) + np.array([cx, cy])
        blobs = [square_blob(i * step, 0) for i in range(n_arm)]
        corner_x, corner_y = (n_arm * step, step)
        blobs += [square_blob(corner_x, corner_y + i * step) for i in range(n_arm)]
        blobs.append(square_blob(*extra_center))
        X = np.vstack(blobs)
        y_labels = np.array([0] * (n_arm * n_per_blob) + [1] * (n_arm * n_per_blob) + [2] * n_per_blob)
        return (X, y_labels)
    X_chains, y_chains = make_rectilinear_chains(step=3, side=2, n_per_blob=60, n_arm=3, seed=1, extra_center=(0, 9))
    X_wedges, y_wedges = make_wedges(angles=[0, 30, 60], r_range=(2, 25), jitter_deg=3, n_per=150, seed=7)
    datasets = [('Collinear blobs\n(same direction,\ndifferent radius)', X_collinear, y_collinear), ('Rectilinear blob chains\n(axis-aligned square blobs,\ndiagonal corner)', X_chains, y_chains), ('Angular wedges\n(same radius range,\ndifferent direction)', X_wedges, y_wedges)]
    metrics = ['euclidean', 'manhattan', 'cosine']
    _fig, _axes = plt.subplots(3, 3, figsize=(11, 11.4))
    for _row, (_name, _X_ds, _y_ds) in enumerate(datasets):
        _X_tensor = torch.from_numpy(_X_ds).float().to(device)
        for _col, _metric in enumerate(metrics):
            _ax = _axes[_row, _col]
            _hdbscan_metric = HDBSCAN(min_cluster_size=30, min_samples=15, metric=_metric, device=device)
            _labels = _hdbscan_metric.fit_predict(_X_tensor).cpu().numpy()
            _ari = adjusted_rand_score(_y_ds, _labels)
            _non_noise = _labels != -1
            plot_gmm(X=_X_ds[_non_noise], gmm=None, ax=_ax, true_labels=_labels[_non_noise], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=6, point_alpha=0.5, legend=False, title=f'{_metric}\nARI={_ari:.2f} ({_hdbscan_metric.n_clusters_} clusters)', xlabel=None, ylabel=None)
            if _non_noise.sum() < len(_labels):
                _ax.scatter(_X_ds[~_non_noise, 0], _X_ds[~_non_noise, 1], c='0.4', marker='x', s=12, linewidths=1)
            if _col == 0:
                _ax.set_ylabel(_name)
            if _row == len(datasets) - 1:
                _ax.set_xlabel('Feature 1')
    _fig.suptitle('Metric vs. dataset shape (min_cluster_size=30, min_samples=15)')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 5: Exactly K Clusters

    HDBSCAN decides the number of clusters itself, but sometimes you need a fixed $K$, for instance to start a $K$-component `GaussianMixture`. There are three ways to get there.

    - **Tune `min_cluster_size`** (and `min_samples`) until $K$ clusters come out. Nothing guarantees such a value exists, and it depends on the data.
    - **Cut the single-linkage tree at $K$ clusters**, as in `fcluster(Z, K, criterion='maxclust')`. HDBSCAN's mutual-reachability tree is a chain: the cheapest cut peels off one outlying point at a time, so the "clusters" are one huge group and $K-1$ singletons. The `min_cluster_size` rule that makes HDBSCAN robust has been skipped.
    - **`HDBSCAN(n_clusters=K)`** (opt-in; the default `n_clusters=None` is unchanged) keeps the condensed tree and the stabilities $\sigma(C)$ of the algorithm above and picks, among all sets of $K$ non-nested clusters, the one with the largest total stability:

    $$\max_{C_1,\dots,C_K\ \text{non-nested}}\ \sum_{k=1}^{K}\sigma(C_k).$$

    This is the excess-of-mass objective with a cardinality constraint. It is solved exactly by dynamic programming over the tree: let $f(v, j)$ be the best total stability of $j$ non-nested clusters inside the subtree of $v$. Then $f(v, 1) = \max\{\sigma(v),\ \max_{c}\, f(c, 1)\}$ (take $v$ itself, or one cluster below it) and, for $j \ge 2$, $f(v, j) = \max_{j_1 + j_2 = j} f(c_1, j_1) + f(c_2, j_2)$ over the two children. Points outside the $K$ chosen clusters stay noise. Extracting a flat clustering from a hierarchy by optimising such an objective is the framework of Campello, Moulavi, Zimek & Sander (2013), and the `hdbscan` package's `hdbscan.flat` reaches a target count by tuning `cluster_selection_epsilon`, which only finds exactly $K$ when the data has $K$ separated density peaks. If the tree has fewer than $K$ non-nested clusters, `n_clusters=K` raises an error that asks for a smaller `min_cluster_size`.

    Below, the three routes on the four tutorial blobs (the same overlapping blobs as in [01 -- Gaussian Mixture Models](01_introduction.ipynb)) with $K=4$.
    """)
    return


@app.cell
def _(HDBSCAN, SEED, device, fcluster, make_default_data, mo, np):
    X_blobs, _ = make_default_data(SEED, device=device)
    K_TARGET = 4
    hdb_free = HDBSCAN(min_cluster_size=30, min_samples=15, device=device).fit(X_blobs)
    hdb_exact = HDBSCAN(min_cluster_size=30, min_samples=15, n_clusters=K_TARGET, device=device).fit(X_blobs)
    # The raw cut: `_single_linkage_tree` is the hierarchy in scipy's linkage format (private; the hdbscan package calls it
    # single_linkage_tree_.to_numpy()), so the recipe `fcluster(Z, K, criterion='maxclust')` applies to it directly.
    cut_labels = fcluster(np.asarray(hdb_free._single_linkage_tree), K_TARGET, criterion='maxclust') - 1

    def _describe(labels):
        _lab = np.asarray(labels)
        _sizes = sorted(np.bincount(_lab[_lab >= 0]).tolist(), reverse=True)
        return f"{len(_sizes)} | {_sizes} | {int((_lab < 0).sum())}"
    _rows = [
        ("`HDBSCAN(min_cluster_size=30, min_samples=15)` (default flat clustering)", _describe(hdb_free.labels_.cpu().numpy())),
        (f"`fcluster(single-linkage tree, {K_TARGET}, 'maxclust')`", _describe(cut_labels)),
        (f"`HDBSCAN(min_cluster_size=30, min_samples=15, n_clusters={K_TARGET})`", _describe(hdb_exact.labels_.cpu().numpy())),
    ]
    mo.md(
        f"**Asking for {K_TARGET} clusters on the four tutorial blobs** ({len(X_blobs)} points)\n\n"
        "| how the clusters are chosen | clusters | sizes | noise points |\n|---|---|---|---|\n"
        + "\n".join(f"| {_name} | {_vals} |" for _name, _vals in _rows)
        + f"\n\nThe default clustering finds {hdb_free.n_clusters_} clusters (not what was asked for), the raw cut gives one cluster with almost every point, and `n_clusters={K_TARGET}` gives exactly {hdb_exact.n_clusters_}, "
        "the set of four non-nested clusters with the largest total stability; the points outside them are noise."
    )
    return K_TARGET, X_blobs, hdb_exact, hdb_free


@app.cell
def _(K_TARGET, X_blobs, hdb_exact, hdb_free, plot_gmm, plt):
    _X_np = X_blobs.cpu().numpy()
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for _ax, _model, _title in zip(_axes, (hdb_free, hdb_exact), ('Default flat clustering', f'n_clusters={K_TARGET}')):
        _labels = _model.labels_.cpu().numpy()
        _kept = _labels != -1
        plot_gmm(X=_X_np[_kept], gmm=None, ax=_ax, true_labels=_labels[_kept], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=8, point_alpha=0.5, legend=False, title=f'{_title} ({_model.n_clusters_} clusters)')
        _ax.scatter(_X_np[~_kept, 0], _X_np[~_kept, 1], c='0.4', marker='x', s=14, linewidths=1, label='Noise')
        _ax.legend()
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `n_clusters=K` does not change which clusters the tree contains, only which of them are kept, so it always returns a valid, density-based clustering with the requested count. It is not a better clustering in general: on strongly overlapping Gaussians the dense cores are what is found, and a plain $k$-means start can be closer to the truth. Using a clustering, with $K$ found or asked for, to start a `GaussianMixture` is the subject of Example 7 below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 6: Using HDBSCAN to Clean Data Before Fitting a GMM

    HDBSCAN's noise label is useful as a preprocessing step for models like GMM
    that assume every point belongs to some component -- outliers can otherwise
    drag component means and inflate covariances. Below: two known Gaussians
    (means/covariances shown as reference ellipses) with uniform noise added on
    top, a GMM fit directly on that noisy data, HDBSCAN's noise flagging on the
    same data, and a GMM re-fit only on the points HDBSCAN kept.
    """)
    return


@app.cell
def _(
    CATEGORICAL_PALETTE,
    GaussianMixture,
    HDBSCAN,
    SEED,
    device,
    generate_gmm_data,
    gmm_proxy,
    np,
    plot_gmm,
    plt,
    torch,
):
    centers_demo = [np.array([0, 0]), np.array([6, 5])]
    covs_demo = [np.array([[1.0, 0.3], [0.3, 0.6]]), np.array([[0.8, -0.2], [-0.2, 1.2]])]
    n_samples_demo = [200, 180]
    X_demo, y_demo = generate_gmm_data(centers_demo, covs_demo, n_samples_demo, device=device, random_state=SEED, noise_ratio=0.3, noise_scale=2.0)
    X_demo_np = X_demo.cpu().numpy()
    y_demo_np = y_demo.cpu().numpy()
    true_proxy = gmm_proxy(means=np.stack(centers_demo), covariances=np.stack(covs_demo), covariance_type='full', predicted_labels=np.zeros(len(X_demo), dtype=int))
    gmm_raw = GaussianMixture(n_components=2, random_state=SEED)
    gmm_raw.fit(X_demo)
    hdbscan_demo = HDBSCAN(min_cluster_size=30, min_samples=15, device=device)
    labels_demo = hdbscan_demo.fit_predict(X_demo)  # 30% of points are uniform noise
    labels_demo_np = labels_demo.cpu().numpy()
    keep_mask = labels_demo_np != -1
    X_clean = X_demo[torch.from_numpy(keep_mask)]
    gmm_clean = GaussianMixture(n_components=2, random_state=SEED)
    gmm_clean.fit(X_clean)
    # Proxy GMM holding the *true* generating parameters, so plot_gmm can draw
    # the true mean/std ellipses even though no GaussianMixture was fit to get them.

    def mean_error(gmm, true_centers):
        means = gmm.means_.cpu().numpy()
        return np.mean([np.linalg.norm(means - c, axis=1).min() for c in true_centers])

    def cov_error(gmm, true_centers, true_covs):
        means = gmm.means_.cpu().numpy()
        covs = gmm.covariances_.cpu().numpy()
        errs = []
        for c, cov in zip(true_centers, true_covs):
            idx = np.argmin(np.linalg.norm(means - c, axis=1))
            errs.append(np.linalg.norm(covs[idx] - cov))
        return np.mean(errs)
    print(f'Points: {X_demo.shape[0]} total, {(y_demo_np == -1).sum()} injected noise')
    print(f'HDBSCAN flagged {(~keep_mask).sum()} points as noise, kept {keep_mask.sum()}\n')
    print(f'Mean error vs. truth        - raw GMM: {mean_error(gmm_raw, centers_demo):.3f}   cleaned GMM: {mean_error(gmm_clean, centers_demo):.3f}')
    print(f'Covariance error vs. truth  - raw GMM: {cov_error(gmm_raw, centers_demo, covs_demo):.3f}   cleaned GMM: {cov_error(gmm_clean, centers_demo, covs_demo):.3f}')
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    _axes[0, 0].scatter(X_demo_np[y_demo_np != -1, 0], X_demo_np[y_demo_np != -1, 1], c=CATEGORICAL_PALETTE[0], s=8, alpha=0.5)
    _axes[0, 0].scatter(X_demo_np[y_demo_np == -1, 0], X_demo_np[y_demo_np == -1, 1], c='0.4', marker='x', s=14, linewidths=1, label='Injected noise')
    plot_gmm(X=X_demo, gmm=true_proxy, ax=_axes[0, 0], title='True Gaussians + injected noise', show_points=False, show_ellipses=True, ellipse_std_devs=[1, 2], show_means=True, legend=True)
    _axes[0, 0].legend(loc='upper right')
    plot_gmm(X=X_demo, gmm=gmm_raw, ax=_axes[0, 1], title='GMM fit directly on raw data', color_by_cluster=True, true_labels=y_demo_np, match_labels_to_true=True, show_ellipses=True, ellipse_std_devs=[1, 2], show_means=True, point_size=8, legend=True)
    _axes[0, 1].legend(loc='upper right')
    _axes[1, 0].scatter(X_demo_np[keep_mask, 0], X_demo_np[keep_mask, 1], c=CATEGORICAL_PALETTE[0], s=8, alpha=0.5, label='Kept')
    _axes[1, 0].scatter(X_demo_np[~keep_mask, 0], X_demo_np[~keep_mask, 1], c='0.4', marker='x', s=14, linewidths=1, label='Flagged as noise')
    _axes[1, 0].set_title(f'HDBSCAN noise flagging ({(~keep_mask).sum()} removed)')
    _axes[1, 0].legend(loc='upper right')
    plot_gmm(X=X_clean, gmm=gmm_clean, ax=_axes[1, 1], title='GMM fit after HDBSCAN noise removal', color_by_cluster=True, true_labels=y_demo_np[keep_mask], match_labels_to_true=True, show_ellipses=True, ellipse_std_devs=[1, 2], show_means=True, point_size=8, legend=True)
    _axes[1, 1].legend(loc='upper right')
    plt.tight_layout()
    # Top-left: raw noisy data with the *true* generating Gaussians as reference
    # Top-right: GMM fit directly on the raw (noisy) data. true_labels +
    # match_labels_to_true so the same physical Gaussian gets the same color
    # here as in every other panel, regardless of which component index the
    # fit happened to assign it internally.
    # Bottom-left: which points HDBSCAN flagged as noise (and would remove)
    # Bottom-right: GMM fit again, this time only on the HDBSCAN-cleaned points.
    # Same true_labels/match_labels_to_true trick, masked down to the kept points.
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 7: Initialising a Gaussian Mixture with HDBSCAN

    A `GaussianMixture` needs a number of components $K$ and a starting point $(\pi_k, \mu_k, \Sigma_k)$ ([03 -- Initialization](03_initialization.ipynb)). HDBSCAN can supply both: run it, take the number of clusters it found as $K$, and estimate the starting parameters from its assignments $z_i \in \{1,\dots,K\}$ ($z_i = -1$ for noise). For each cluster $C_k = \{i : z_i = k\}$

    $$
    \mu_k = \frac{1}{|C_k|}\sum_{i\in C_k} x_i, \qquad \Sigma_k = \widehat{\mathrm{Cov}}(C_k) + \texttt{reg\_covar}\, I, \qquad \pi_k = \frac{|C_k| + n_k^{\text{noise}}}{n},
    $$

    where $n_k^{\text{noise}}$ is the number of noise points whose nearest mean is $\mu_k$. The means and covariances use the cluster's own points only, so outliers cannot distort them; the weights count every noise point for its nearest mean, so they describe all the data and sum to 1. The result goes to the mixture as plain init tensors:

    ```python
    K, means, weights, covariances = hdbscan_gmm_init(X)
    gmm = GaussianMixture(n_components=K, init_means=means, init_weights=weights, init_covariances=covariances).fit(X)
    ```

    `hdbscan_gmm_init` is a notebook helper, not part of the tgmm API. It runs `HDBSCAN` with `min_cluster_size` $= \max\{10,\ \lfloor 0.02\,n\rfloor\}$ (2% of the points) and `min_samples` $= \max\{5,\ \texttt{min\_cluster\_size}/2\}$, a rule of thumb that is not tuned, and raises an error if no cluster is found.
    """)
    return


@app.cell
def _(HDBSCAN, torch):
    def hdbscan_gmm_init(X, min_cluster_size=None, min_samples=None, reg_covar=1e-6, return_labels=False, **kwargs):
        """K and init tensors (means, weights, covariances) for ``covariance_type='full'`` from an HDBSCAN clustering of ``X`` of shape (n, d).

        Extra keyword arguments go to ``HDBSCAN``; with ``return_labels=True`` the HDBSCAN labels (-1 = noise) are returned as a fifth value.
        """
        n, d = X.shape
        mcs = min_cluster_size or max(10, int(0.02 * n))
        ms = min_samples or max(5, mcs // 2)
        labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms, device=X.device.type, **kwargs).fit(X).labels_.to(X.device)
        ids = [k for k in range(int(labels.max()) + 1) if int((labels == k).sum()) >= 2]  # a covariance needs at least 2 points
        if not ids:
            raise ValueError('HDBSCAN found no cluster: lower min_cluster_size')
        means = torch.stack([X[labels == k].mean(0) for k in ids])
        covariances = torch.stack([torch.cov(X[labels == k].T) + reg_covar * torch.eye(d, device=X.device, dtype=X.dtype) for k in ids])
        z = torch.full((n,), -1, device=X.device, dtype=torch.long)
        for j, k in enumerate(ids):
            z[labels == k] = j
        z = torch.where(z >= 0, z, torch.cdist(X, means).argmin(1))  # every noise point counts for its nearest mean
        weights = torch.bincount(z, minlength=len(ids)).to(X.dtype) / n
        return (len(ids), means, weights, covariances) + ((labels,) if return_labels else ())

    return (hdbscan_gmm_init,)


@app.cell
def _(
    FAST,
    GaussianMixture,
    MAX_ITER,
    device,
    generate_gmm_data,
    hdbscan_gmm_init,
    make_default_data,
    np,
    torch,
):
    # (a) the tutorial blobs (draw 0, the one the other tutorials show); (b) four separated N(0, I) blobs plus 1% uniform outliers
    _X_a, _ = make_default_data(0, device=device)
    _corners = [np.array(_c, dtype=float) for _c in [(0, 0), (8, 0), (0, 8), (8, 8)]]
    _X_blobs, _ = generate_gmm_data(_corners, [np.eye(2)] * 4, [300] * 4, device=device, random_state=1)
    _outliers = torch.tensor(np.random.default_rng(0).uniform(-12, 20, (12, 2)), dtype=_X_blobs.dtype, device=device)
    _X_b = torch.cat([_X_blobs, _outliers])
    KMEANS_SEEDS = 5 if FAST else 10
    init_runs = {}
    for _name, _X in {'tutorial blobs': _X_a, 'four separated blobs + 1% outliers': _X_b}.items():
        _K, _means, _weights, _covs, _labels = hdbscan_gmm_init(_X, return_labels=True)
        _from_hdbscan = GaussianMixture(n_components=_K, init_means=_means, init_weights=_weights, init_covariances=_covs, max_iter=MAX_ITER, device=device).fit(_X)
        # the k-means start knows the true K = 4: seed 0, and a few more seeds to see whether seed 0 is typical
        _from_kmeans = [GaussianMixture(n_components=4, init_means='kmeans', random_state=_s, max_iter=MAX_ITER, device=device).fit(_X) for _s in range(KMEANS_SEEDS)]
        init_runs[_name] = dict(X=_X, K=_K, means=_means, weights=_weights, covs=_covs, labels=_labels, hdbscan=_from_hdbscan, kmeans=_from_kmeans)
    return KMEANS_SEEDS, init_runs


@app.cell
def _(KMEANS_SEEDS, init_runs, mo):
    _rows = []
    _notes = []
    for _name, _r in init_runs.items():
        _n, _n_noise = len(_r['X']), int((_r['labels'] == -1).sum())
        _h, _k = _r['hdbscan'], _r['kmeans'][0]
        _rows.append(f"| {_name} | HDBSCAN | {_r['K']} | {_n_noise} of {_n} | {_h.n_iter_} | {float(_h.lower_bound_):.4f} |")
        _rows.append(f"| {_name} | k-means, true $K$ = 4 (seed 0) | 4 | - | {_k.n_iter_} | {float(_k.lower_bound_):.4f} |")
        _hits = sum(float(_g.lower_bound_) > float(_h.lower_bound_) - 1e-3 for _g in _r['kmeans'])
        _notes.append((_name, _hits))
    _a, _b = list(init_runs.values())
    _ll = lambda _g: float(_g.lower_bound_)
    mo.md(
        "| data | start | components $K$ | noise points | EM iterations | final log-likelihood |\n|---|---|---|---|---|---|\n" + "\n".join(_rows)
        + f"\n\nOn the tutorial blobs HDBSCAN finds {_a['K']} clusters but calls {int((_a['labels'] == -1).sum())} of the {len(_a['X'])} points noise: the blobs overlap, so only their dense cores are clusters, "
        f"and the starting covariances are smaller than the blobs' own. EM still ends within {abs(_ll(_a['hdbscan']) - _ll(_a['kmeans'][0])):.1e} of the log-likelihood it reaches from the k-means start, "
        f"in {_a['hdbscan'].n_iter_} iterations instead of {_a['kmeans'][0].n_iter_}. "
        + (f"On the separated blobs the k-means start with seed 0 ends at {_ll(_b['kmeans'][0]):.4f}, below the {_ll(_b['hdbscan']):.4f} from the HDBSCAN start: k-means++ can seed a centre on one of the outliers. "
           if _ll(_b['kmeans'][0]) < _ll(_b['hdbscan']) - 1e-3 else
           f"On the separated blobs both starts end at the same log-likelihood ({_ll(_b['hdbscan']):.4f}). ")
        + f"Over {KMEANS_SEEDS} seeds, {_notes[1][1]} of the k-means starts reach the log-likelihood of the HDBSCAN start on the separated blobs (and {_notes[0][1]} on the tutorial blobs); "
        f"HDBSCAN is deterministic, and it called only {int((_b['labels'] == -1).sum())} points noise there."
    )
    return


@app.cell
def _(CATEGORICAL_PALETTE, Ellipse, N_STD95, init_runs, np, plt):
    _r = init_runs['tutorial blobs']
    _X, _lab = _r['X'].cpu().numpy(), _r['labels'].cpu().numpy()
    _mu, _S, _pi = (_r[_k].cpu().numpy() for _k in ('means', 'covs', 'weights'))
    _fit = _r['hdbscan']
    _fmu, _fS = _fit.means_.detach().cpu().numpy(), _fit.covariances_.detach().cpu().numpy()

    def _ellipse(_ax, _mean, _cov, **_kw):
        _w, _V = np.linalg.eigh(_cov)
        _ax.add_patch(Ellipse(_mean, 2 * N_STD95 * np.sqrt(_w[1]), 2 * N_STD95 * np.sqrt(_w[0]), angle=np.degrees(np.arctan2(_V[1, 1], _V[0, 1])), fill=False, **_kw))

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    _axs[0].scatter(_X[_lab == -1, 0], _X[_lab == -1, 1], s=4, alpha=0.5, color='lightgray', label='noise')
    for _k in range(_r['K']):  # cluster k is component k
        _axs[0].scatter(_X[_lab == _k, 0], _X[_lab == _k, 1], s=4, alpha=0.5, color=CATEGORICAL_PALETTE[_k], label=f'cluster {_k + 1}')
    _axs[0].set_title(f"HDBSCAN: {_r['K']} clusters, {int((_lab == -1).sum())} noise points")
    _axs[0].legend(markerscale=3, loc='upper left')
    _axs[1].scatter(_X[:, 0], _X[:, 1], s=4, alpha=0.5, color='lightgray')
    for _k in range(_r['K']):
        _ellipse(_axs[1], _mu[_k], _S[_k], color=CATEGORICAL_PALETTE[_k])
        _ellipse(_axs[1], _fmu[_k], _fS[_k], color='black', ls='--', lw=0.8)
        _axs[1].annotate(f'$\\pi$={_pi[_k]:.2f}', _mu[_k], textcoords='offset points', xytext=(6, 6), fontsize=8, color=CATEGORICAL_PALETTE[_k])
    _axs[1].scatter(_mu[:, 0], _mu[:, 1], marker='x', s=30, linewidths=1.2, color='red', zorder=3, label='initial mean')
    _axs[1].scatter(_fmu[:, 0], _fmu[:, 1], marker='h', s=25, color='black', zorder=3, label='fitted mean')
    _axs[1].plot([], [], color='gray', label='initial Gaussians (95%)')
    _axs[1].plot([], [], color='black', ls='--', lw=0.8, label='fitted GMM (95%)')
    _axs[1].set_title('Initial Gaussians from the assignments')
    _axs[1].legend(loc='upper left')
    for _ax in _axs:
        _ax.set_aspect('equal')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **When it works.** HDBSCAN counts density peaks, not Gaussians, so the number of clusters it finds is a good $K$ only when the components are separated enough to show as peaks. The next table runs the same rule on several draws of three datasets: the two above and four strongly overlapping Gaussians.
    """)
    return


@app.cell
def _(FAST, device, generate_gmm_data, hdbscan_gmm_init, make_default_data, mo, np, torch):
    _draws = range(4 if FAST else 8)
    _corners = [np.array(_c, dtype=float) for _c in [(0, 0), (8, 0), (0, 8), (8, 8)]]
    _overlap = [np.array(_c, dtype=float) for _c in [(0, 0), (3.2, 0), (0, 3.2), (3.2, 3.2)]]

    def _separated(_s):
        _X, _ = generate_gmm_data(_corners, [np.eye(2)] * 4, [300] * 4, device=device, random_state=_s)
        return torch.cat([_X, torch.tensor(np.random.default_rng(_s).uniform(-12, 20, (12, 2)), dtype=_X.dtype, device=device)])

    _sets = {
        'tutorial blobs': lambda _s: make_default_data(_s, device=device)[0],
        'four separated blobs + 1% outliers': _separated,
        'four strongly overlapping blobs (centres 3.2 apart, 1000 / 500 / 300 / 200 points)': lambda _s: generate_gmm_data(_overlap, [np.eye(2)] * 4, [1000, 500, 300, 200], device=device, random_state=_s)[0],
    }

    def _found(_X):
        try:
            return hdbscan_gmm_init(_X)[0]
        except ValueError:  # no cluster at all
            return 0

    _rows = []
    _found_all = {}
    for _name, _make in _sets.items():
        _ks = [_found(_make(_s)) for _s in _draws]
        _found_all[_name] = _ks
        _rows.append(f"| {_name} | {sum(_k == 4 for _k in _ks)} of {len(_ks)} | {', '.join(str(_k) for _k in _ks)} |")
    _ov = _found_all[list(_sets)[2]]
    mo.md(
        "| data (true $K$ = 4) | draws with $K$ found = 4 | $K$ found, draw by draw (0 = no cluster found) |\n|---|---|---|\n" + "\n".join(_rows)
        + f"\n\nOn the overlapping blobs the Gaussians merge into fewer density peaks: HDBSCAN finds at most {max(_ov)} components there, so $K$ is underestimated" + (f" (in {_ov.count(0)} of the draws no cluster at all)" if 0 in _ov else "") + ". "
        "The tutorial blobs sit in between and their $K$ changes from draw to draw (the draw used above, draw 0, is one where it finds 4). So treat the $K$ found as a candidate: choose $K$ with BIC or ICL ([13 -- Model Selection](13_model_selection.ipynb)) when the clusters overlap, "
        "and if you know $K$, ask HDBSCAN for it with `hdbscan_gmm_init(X, n_clusters=4, min_cluster_size=20)` (the exactly-$K$ selection of Example 5, which needs a `min_cluster_size` small enough for the condensed tree to hold $K$ clusters; they are still dense cores, so check the fit)."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Campello, R. J. G. B., Moulavi, D. & Sander, J. (2013), *Density-based clustering based on hierarchical density estimates*, PAKDD.
    - Campello, R. J. G. B., Moulavi, D., Zimek, A. & Sander, J. (2013), *A framework for semi-supervised and unsupervised optimal extraction of clusters from hierarchies*, Data Mining and Knowledge Discovery 27(3), 344-371 (doi:10.1007/s10618-013-0311-4).
    - McInnes, L., Healy, J. & Astels, S. (2017), *hdbscan: Hierarchical density based clustering*, Journal of Open Source Software 2(11).
    """)
    return


if __name__ == "__main__":
    app.run()
