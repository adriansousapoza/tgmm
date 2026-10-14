"""
DBCV (Moulavi et al. 2014) tests for ClusteringMetrics.dbcv.

The reference implementation below is written independently of
tgmm.metrics.ClusteringMetrics.dbcv (own distance computation via
scipy.spatial.distance.cdist, own degree/internal-node/DSC/DSPC/V/weighting
math) from the same algorithmic description in
docs/superpowers/specs/2026-10-14-phase1-metrics-model-selection-design.md.

It deliberately *shares* the MST primitive (tgmm.hdbscan.prim_mst) with the
production implementation rather than using, say,
scipy.sparse.csgraph.minimum_spanning_tree: mutual-reachability distance is
max(core_dist(x), core_dist(y), d(x, y)), and whenever one point's own core
distance dominates several of its edges, those edges all tie at exactly
that value -- which happens routinely, not as a rare edge case. A graph
with tied edge weights can have multiple equally-minimal spanning trees
with different topologies (different internal-vs-leaf node sets), so two
*independently-but-correctly* implemented MST algorithms can legitimately
produce different valid MSTs and therefore different DBCV values on the
exact same ties -- verified directly (scipy's MST and prim_mst gave the
same total tree weight, 56.150451..., for an identical graph, but
different internal-node counts, 17 vs 11). Sharing the MST step isolates
the test to what's actually novel in DBCV -- the core-distance, DSC/DSPC/V,
and noise-weighting formulas -- rather than failing on an MST tie-breaking
difference that doesn't indicate either implementation is wrong.
"""
import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist

from tgmm.hdbscan import prim_mst
from tgmm.metrics import ClusteringMetrics


def dbcv_reference(X: np.ndarray, labels: np.ndarray, noise_label=-1) -> float:
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n, d = X.shape

    mask = labels != noise_label if noise_label is not None else np.ones(n, dtype=bool)
    cluster_ids = np.unique(labels[mask])

    info = {}
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        Xc = X[idx]
        nc = len(idx)
        if nc < 2:
            info[cid] = dict(nc=nc)
            continue

        D = cdist(Xc, Xc)
        D_safe = D.copy()
        np.fill_diagonal(D_safe, np.inf)
        D_safe = np.clip(D_safe, 1e-12, None)

        mean_inv_pow = (D_safe ** (-d)).sum(axis=1) / (nc - 1)
        core = mean_inv_pow ** (-1.0 / d)

        mreach = np.maximum(np.maximum(core[:, None], core[None, :]), D)
        np.fill_diagonal(mreach, 0.0)

        mst_edges = prim_mst(torch.tensor(mreach, dtype=torch.float64))
        dsc = mst_edges[:, 2].max()

        degree = np.zeros(nc)
        for u, v, _w in mst_edges:
            degree[int(u)] += 1
            degree[int(v)] += 1
        internal = degree >= 2
        if not internal.any():
            internal = np.ones(nc, dtype=bool)

        info[cid] = dict(nc=nc, Xc=Xc, core=core, internal=internal, dsc=dsc)

    total = 0.0
    for cid in cluster_ids:
        ii = info[cid]
        if ii['nc'] < 2:
            continue

        min_dspc = np.inf
        Xi = ii['Xc'][ii['internal']]
        ci = ii['core'][ii['internal']]
        for oid in cluster_ids:
            if oid == cid:
                continue
            jj = info[oid]
            if jj['nc'] < 2:
                continue
            Xj = jj['Xc'][jj['internal']]
            cj = jj['core'][jj['internal']]
            Dc = cdist(Xi, Xj)
            mreach_cross = np.maximum(np.maximum(ci[:, None], cj[None, :]), Dc)
            min_dspc = min(min_dspc, mreach_cross.min())

        dsc_i = ii['dsc']
        if min_dspc == np.inf:
            v = 0.0
        else:
            denom = max(min_dspc, dsc_i)
            v = (min_dspc - dsc_i) / denom if denom > 0 else 0.0

        total += (ii['nc'] / n) * v

    return total


def _make_blobs(seed, n_per=40, d=3, n_clusters=3, spread=1.0, center_scale=6.0):
    rng = np.random.RandomState(seed)
    chunks = []
    labels = []
    for k in range(n_clusters):
        center = rng.uniform(-center_scale, center_scale, size=d)
        chunks.append(rng.randn(n_per, d) * spread + center)
        labels.append(np.full(n_per, k))
    return np.concatenate(chunks, axis=0), np.concatenate(labels)


SEEDS = [0, 1, 2, 3, 4]


@pytest.mark.parametrize("seed", SEEDS)
def test_dbcv_matches_numpy_reference(seed):
    X, labels = _make_blobs(seed)
    ref = dbcv_reference(X, labels, noise_label=None)
    got = ClusteringMetrics.dbcv(
        torch.tensor(X, dtype=torch.float64), torch.tensor(labels), noise_label=None
    )
    assert got == pytest.approx(ref, abs=1e-8)


def test_dbcv_matches_reference_with_noise():
    X, labels = _make_blobs(seed=7, n_per=30)
    rng = np.random.RandomState(7)
    noise_mask = rng.rand(len(labels)) < 0.1
    labels = labels.copy()
    labels[noise_mask] = -1

    ref = dbcv_reference(X, labels, noise_label=-1)
    got = ClusteringMetrics.dbcv(torch.tensor(X, dtype=torch.float64), torch.tensor(labels), noise_label=-1)
    assert got == pytest.approx(ref, abs=1e-8)


def test_dbcv_range():
    X, labels = _make_blobs(seed=2)
    value = ClusteringMetrics.dbcv(
        torch.tensor(X, dtype=torch.float64), torch.tensor(labels), noise_label=None
    )
    assert -1.0 - 1e-8 <= value <= 1.0 + 1e-8


def test_dbcv_prefers_correct_2moons_partition_over_kmeans():
    pytest.importorskip("sklearn")
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_moons

    X_np, y_true = make_moons(n_samples=300, noise=0.07, random_state=0)
    X = torch.tensor(X_np, dtype=torch.float64)

    kmeans_labels = torch.tensor(
        KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X_np)
    )
    true_labels = torch.tensor(y_true)

    dbcv_true = ClusteringMetrics.dbcv(X, true_labels, noise_label=None)
    dbcv_kmeans = ClusteringMetrics.dbcv(X, kmeans_labels, noise_label=None)
    assert dbcv_true > dbcv_kmeans
