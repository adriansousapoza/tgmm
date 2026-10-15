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
    # HDBSCAN Clustering

    `tgmm.HDBSCAN` is a PyTorch implementation of **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise; Campello, Moulavi & Sander, 2013), following the `scikit-learn-contrib/hdbscan` reference implementation. Unlike a Gaussian mixture it needs no component count and makes no assumption about cluster shape: it finds clusters as dense regions, can recover non-convex shapes, and labels sparse points as **noise** (`-1`).

    The algorithm, as implemented here:

    1. **Core distance.** $\mathrm{core}_k(a)$ is the distance from $a$ to its `min_samples`-th nearest neighbour (counting $a$ itself).
    2. **Mutual reachability distance.** $d_{\mathrm{mreach}}(a,b) = \max\{\mathrm{core}_k(a),\ \mathrm{core}_k(b),\ d(a,b)\}$ (divided by `alpha` if `alpha != 1`). This pushes sparse points away from everything, so noise does not bridge dense regions.
    3. **Minimum spanning tree and hierarchy.** A minimum spanning tree of the complete mutual-reachability graph (Prim's algorithm on the dense $n\times n$ matrix) is converted to a single-linkage hierarchy.
    4. **Condensed tree.** Walking the hierarchy from the root down, a split only counts as a real cluster split if both children have at least `min_cluster_size` points; otherwise the smaller side is treated as points "falling out" of the parent cluster. Cluster density is tracked as $\lambda = 1/d_{\mathrm{mreach}}$.
    5. **Stability-based extraction.** Each cluster $C$ gets a stability $\sigma(C) = \sum_{p\in C}\left(\lambda_p - \lambda_{\mathrm{birth}}(C)\right)$, where $\lambda_p$ is the density level at which point $p$ leaves $C$. Clusters are selected bottom-up by the *excess of mass* rule: keep a cluster if its stability is at least the summed stability of its descendants that would replace it. Unselected points are noise, and `probabilities_` gives each point's membership strength $\lambda_p/\lambda_{\max}$.

    Because step 3 uses the dense pairwise distance matrix, memory grows as $O(n^2)$.

    **What you'll learn**

    - how HDBSCAN recovers non-convex clusters (two moons) that a Gaussian mixture cannot
    - how `min_samples` and `min_cluster_size` control noise detection and the number of clusters
    - how the choice of `metric` (Euclidean, Manhattan, cosine) changes what counts as a cluster
    - how to use HDBSCAN's noise label to clean data before fitting a `GaussianMixture`

    **Prerequisites:** [01 -- Gaussian Mixture Models](01_gmm.ipynb), [03 -- Visualization](03_visualise.ipynb) (for `plot_gmm`).
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
    from sklearn.datasets import make_moons
    from sklearn.metrics import adjusted_rand_score

    from tgmm import GaussianMixture, HDBSCAN, generate_gmm_data, plot_gmm, gmm_proxy, dynamic_figsize

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 2
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f'Using device: {device}')
    return (
        GaussianMixture,
        HDBSCAN,
        SEED,
        adjusted_rand_score,
        device,
        dynamic_figsize,
        generate_gmm_data,
        gmm_proxy,
        make_moons,
        np,
        plot_gmm,
        plt,
        torch,
    )


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
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_gmm(X=X_moons, gmm=None, ax=_ax1, true_labels=y_moons, color_by_cluster=True, show_ellipses=False, show_means=False, point_size=20, point_alpha=0.6, title='Original Moons Dataset (Ground Truth)', legend=False)
    _labels_np = labels_moons.cpu().numpy()
    _non_noise = _labels_np != -1
    plot_gmm(X=X_moons[_non_noise], gmm=None, ax=_ax2, true_labels=_labels_np[_non_noise], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=20, point_alpha=0.6, title=f'HDBSCAN Clustering ({hdbscan_moons.n_clusters_} clusters)', legend=False)
    if (~_non_noise).any():
        _ax2.scatter(X_moons[~_non_noise, 0], X_moons[~_non_noise, 1], c='red', marker='x', s=50, label='Noise')
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
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(18, 5))
    plot_gmm(X=X_test_np[~true_is_noise], gmm=None, ax=_ax1, true_labels=y_test_np[~true_is_noise], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=20, point_alpha=0.6, legend=False, title='Ground Truth (with uniform noise)', legend_labels=[f'Cluster {i}' for i in range(3)])
    if true_is_noise.any():
        _ax1.scatter(X_test_np[true_is_noise, 0], X_test_np[true_is_noise, 1], c='red', marker='x', s=50, label='Ground Truth Noise', linewidth=2)
    _ax1.legend()
    plot_gmm(X=X_test_np[~pred_is_noise], gmm=None, ax=_ax2, true_labels=labels_torch_np[~pred_is_noise], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=20, point_alpha=0.6, legend=False, title=f'PyTorch HDBSCAN ({hdbscan_torch.n_clusters_} clusters)')
    # Classify each point's noise/cluster call against ground truth
    if pred_is_noise.any():
        _ax2.scatter(X_test_np[pred_is_noise, 0], X_test_np[pred_is_noise, 1], c='red', marker='x', s=50, label='Detected Noise', linewidth=2)
    _ax2.legend()
    _ax3.scatter(X_test_np[correct_mask, 0], X_test_np[correct_mask, 1], c='green', s=20, alpha=0.6, label='Correct')  # flagged as noise, actually a real point
    if false_positive_mask.any():  # actual noise, missed (kept as cluster)
        _ax3.scatter(X_test_np[false_positive_mask, 0], X_test_np[false_positive_mask, 1], c='tomato', marker='x', s=50, label='False positive (wrongly flagged noise)', linewidth=2)
    if false_negative_mask.any():
        _ax3.scatter(X_test_np[false_negative_mask, 0], X_test_np[false_negative_mask, 1], c='darkred', marker='x', s=50, label='False negative (noise missed)', linewidth=2)
    _ax3.set_title('Noise Detection Accuracy', fontsize=14)
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
    covs_complex = [0.4 * np.eye(2) for _ in range(4)]
    X_complex, y_complex = generate_gmm_data(centers_complex, covs_complex, n_samples_complex, device=device, random_state=SEED)
    # The smallest true cluster has 50 points, so anything below that won't change what counts as a
    # valid cluster: 10 and 30 would look identical. 70 and 100 cross the size of progressively larger true clusters.
    min_cluster_sizes = [10, 70, 100]
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
    X_np = X_complex.cpu().numpy()
    for _ax, min_size in zip(_axes, min_cluster_sizes):
        hdbscan_test = HDBSCAN(min_cluster_size=min_size, min_samples=5, device=device)
        labels_test = hdbscan_test.fit_predict(X_complex)
        _labels_np = labels_test.cpu().numpy()
        _non_noise = _labels_np != -1
        plot_gmm(X=X_np[_non_noise], gmm=None, ax=_ax, true_labels=_labels_np[_non_noise], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=20, point_alpha=0.6, legend=False, title=f'min_cluster_size={min_size}\n({hdbscan_test.n_clusters_} clusters, {(~_non_noise).sum()} noise)')
        if (~_non_noise).any():
            _ax.scatter(X_np[~_non_noise, 0], X_np[~_non_noise, 1], c='red', marker='x', s=50, label='Noise')
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
    _fig, _axes = plt.subplots(3, 3, figsize=(16, 15))
    for _row, (_name, _X_ds, _y_ds) in enumerate(datasets):
        _X_tensor = torch.from_numpy(_X_ds).float().to(device)
        for _col, _metric in enumerate(metrics):
            _ax = _axes[_row, _col]
            _hdbscan_metric = HDBSCAN(min_cluster_size=30, min_samples=15, metric=_metric, device=device)
            _labels = _hdbscan_metric.fit_predict(_X_tensor).cpu().numpy()
            _ari = adjusted_rand_score(_y_ds, _labels)
            _non_noise = _labels != -1
            plot_gmm(X=_X_ds[_non_noise], gmm=None, ax=_ax, true_labels=_labels[_non_noise], color_by_cluster=True, show_ellipses=False, show_means=False, point_size=15, point_alpha=0.6, legend=False, title=f'{_metric}\nARI={_ari:.2f} ({_hdbscan_metric.n_clusters_} clusters)', xlabel=None, ylabel=None)
            if _non_noise.sum() < len(_labels):
                _ax.scatter(_X_ds[~_non_noise, 0], _X_ds[~_non_noise, 1], c='red', marker='x', s=30, alpha=1.0, linewidth=1.5)
            _ax.set_title(_ax.get_title(), fontsize=11)
            if _col == 0:
                _ax.set_ylabel(_name, fontsize=10, fontweight='bold')
            if _row == len(datasets) - 1:
                _ax.set_xlabel('Feature 1')
    _fig.suptitle('Metric vs. dataset shape (same min_cluster_size=30, min_samples=15 throughout)', fontsize=13)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 5: Using HDBSCAN to Clean Data Before Fitting a GMM

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
    GaussianMixture,
    HDBSCAN,
    SEED,
    device,
    dynamic_figsize,
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
    _fig, _axes = plt.subplots(2, 2, figsize=dynamic_figsize(2, 2), sharex=True, sharey=True)
    _axes[0, 0].scatter(X_demo_np[y_demo_np != -1, 0], X_demo_np[y_demo_np != -1, 1], c='steelblue', s=10, alpha=0.5)
    _axes[0, 0].scatter(X_demo_np[y_demo_np == -1, 0], X_demo_np[y_demo_np == -1, 1], c='red', marker='x', s=25, alpha=0.7, label='Injected noise')
    plot_gmm(X=X_demo, gmm=true_proxy, ax=_axes[0, 0], title='True Gaussians + injected noise', show_points=False, show_ellipses=True, ellipse_std_devs=[1, 2], show_means=True, legend=True)
    _axes[0, 0].legend(fontsize=8, loc='upper right')
    plot_gmm(X=X_demo, gmm=gmm_raw, ax=_axes[0, 1], title='GMM fit directly on raw data', color_by_cluster=True, true_labels=y_demo_np, match_labels_to_true=True, show_ellipses=True, ellipse_std_devs=[1, 2], show_means=True, point_size=10, legend=True)
    _axes[0, 1].legend(fontsize=8, loc='upper right')
    _axes[1, 0].scatter(X_demo_np[keep_mask, 0], X_demo_np[keep_mask, 1], c='steelblue', s=10, alpha=0.5, label='Kept')
    _axes[1, 0].scatter(X_demo_np[~keep_mask, 0], X_demo_np[~keep_mask, 1], c='red', marker='x', s=25, alpha=0.7, label='Flagged as noise')
    _axes[1, 0].set_title(f'HDBSCAN noise flagging ({(~keep_mask).sum()} removed)')
    _axes[1, 0].legend(fontsize=8, loc='upper right')
    plot_gmm(X=X_clean, gmm=gmm_clean, ax=_axes[1, 1], title='GMM fit after HDBSCAN noise removal', color_by_cluster=True, true_labels=y_demo_np[keep_mask], match_labels_to_true=True, show_ellipses=True, ellipse_std_devs=[1, 2], show_means=True, point_size=10, legend=True)
    _axes[1, 1].legend(fontsize=8, loc='upper right')
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
    ## Summary

    - HDBSCAN finds non-convex, variable-density clusters and flags noise without being told how many clusters to look for.
    - `min_samples` sets how conservative the density estimate (and hence the noise call) is; `min_cluster_size` sets the smallest group that can count as a cluster.
    - The distance metric defines what "dense" means: collinear blobs, axis-aligned chains and angular wedges each need a different metric to be recovered.
    - Used as a preprocessing step, HDBSCAN's noise label removes outliers that would otherwise inflate a `GaussianMixture`'s covariances and shift its means.

    **Next:** [13 -- Model Selection](13_model_selection.ipynb) for choosing the number of mixture components by BIC, AIC and ICL.
    """)
    return


if __name__ == "__main__":
    app.run()
