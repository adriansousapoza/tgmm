"""Regression tests for HDBSCAN: cluster_selection_epsilon, cluster_selection_method='leaf', and the
single-cluster labelling, checked against scikit-learn-contrib/hdbscan (BSD-3).

Convention note: tgmm's ``min_samples=k`` counts the point itself (as in Campello et al.), while the reference
package counts neighbours *excluding* the point, so ``tgmm min_samples=k`` corresponds to reference
``min_samples=k-1`` (reference outputs below were produced that way, with ``algorithm='generic'``).
"""
import json
import os

import numpy as np
import pytest
import torch

from tgmm.hdbscan import HDBSCAN

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "hdbscan_pre_fix.npz")


def _rle(s):
    """'a3 b2' -> 'aaabb' (the expected strings below are run-length encoded for readability)."""
    return "".join(tok[0] * int(tok[1:]) for tok in s.split())


def _labels_str(labels):
    return "".join("n" if l < 0 else str(int(l)) for l in labels)


def tiny_data():
    """Four tight clusters of 25 points: two close pairs far apart (pair spacing 1.5, between pairs 6.5)."""
    rs = np.random.RandomState(7)
    centers = np.array([[0, 0], [1.5, 0], [8, 0], [9.5, 0]], float)
    return np.vstack([c + 0.18 * rs.randn(25, 2) for c in centers])


def blob_data():
    return np.random.RandomState(5).randn(90, 2)


def fit(X, **kw):
    kw.setdefault("min_cluster_size", 6)
    return HDBSCAN(**kw).fit(torch.tensor(X, dtype=torch.float64))


# --- hard-coded outputs of the reference implementation (hdbscan 0.8.44, algorithm='generic', ms = mcs - 1) -----
TINY_REF = {
    'eom': '025 125 225 325',
    'eom_eps1': '050 150',
    'eom_eps2': '050 150',
    'eom_eps5': '050 150',
    'eom_eps20': '050 150',
    'eom_eps20_asc': '0100',
    'leaf': '025 125 225 325',
    'leaf_eps0.5': '025 125 225 325',
    'leaf_eps2': '050 150',
    'leaf_eps5': '050 150',
    'eom_asc': '025 125 225 325',
    'eom_max45': '025 125 225 325',
    'eom_max45_asc': '025 125 225 325',
    'leaf_asc': '025 125 225 325',
}

BLOB_REF = {
    'blob': 'n90',
    'blob_asc': 'n13 02 n1 01 n3 01 n3 01 n23 01 n16 02 n23',
    'blob_asc_max30': 'n13 02 n1 01 n3 01 n3 01 n23 01 n16 02 n23',
    'blob_asc_eps': '01 n1 05 n2 019 n1 05 n2 01 n1 015 n1 03 n2 03 n2 04 n1 01 n1 01 n1 05 n1 09 n1 01',
    'blob_leaf': 'n90',
    'blob_leaf_eps': 'n90',
}

TINY_KWARGS = {
    'eom': {},
    'eom_eps1': dict(cluster_selection_epsilon=1.0),
    'eom_eps2': dict(cluster_selection_epsilon=2.0),
    'eom_eps5': dict(cluster_selection_epsilon=5.0),
    'eom_eps20': dict(cluster_selection_epsilon=20.0),
    'eom_eps20_asc': dict(cluster_selection_epsilon=20.0, allow_single_cluster=True),
    'leaf': dict(cluster_selection_method='leaf'),
    'leaf_eps0.5': dict(cluster_selection_method='leaf', cluster_selection_epsilon=0.5),
    'leaf_eps2': dict(cluster_selection_method='leaf', cluster_selection_epsilon=2.0),
    'leaf_eps5': dict(cluster_selection_method='leaf', cluster_selection_epsilon=5.0),
    'eom_asc': dict(allow_single_cluster=True),
    'eom_max45': dict(max_cluster_size=45),
    'eom_max45_asc': dict(max_cluster_size=45, allow_single_cluster=True),
    'leaf_asc': dict(cluster_selection_method='leaf', allow_single_cluster=True),
}

BLOB_KWARGS = {
    'blob': {},
    'blob_asc': dict(allow_single_cluster=True),
    'blob_asc_max30': dict(allow_single_cluster=True, max_cluster_size=30),
    'blob_asc_eps': dict(allow_single_cluster=True, cluster_selection_epsilon=0.8),
    'blob_leaf': dict(cluster_selection_method='leaf'),
    'blob_leaf_eps': dict(cluster_selection_method='leaf', cluster_selection_epsilon=0.9),
}


@pytest.mark.parametrize("name", list(TINY_REF))
def test_tiny_matches_reference_labels(name):
    kw = dict(TINY_KWARGS[name])
    m = fit(tiny_data(), **kw)
    assert _labels_str(m.labels_.numpy()) == _rle(TINY_REF[name])


@pytest.mark.parametrize("name", list(BLOB_REF))
def test_blob_matches_reference_labels(name):
    kw = dict(BLOB_KWARGS[name])
    m = fit(blob_data(), min_cluster_size=8, **kw)
    assert _labels_str(m.labels_.numpy()) == _rle(BLOB_REF[name])


# --- defaults are unchanged -------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["moons", "blobs_noise", "overlap"])
@pytest.mark.parametrize("mcs,ms", [(10, None), (25, 10)])
def test_defaults_bit_identical_to_pre_fix_snapshot(name, mcs, ms):
    from sklearn.datasets import make_blobs, make_moons
    if name == "moons":
        X = make_moons(400, noise=0.06, random_state=0)[0]
    elif name == "blobs_noise":
        Xb, _ = make_blobs(450, centers=[[0, 0], [5, 5], [0, 6]], cluster_std=[0.6, 0.9, 0.5], random_state=1)
        X = np.vstack([Xb, np.random.RandomState(0).uniform(-3, 9, (40, 2))])
    else:
        X, _ = make_blobs(500, centers=[[0, 0], [2.2, 0], [8, 8]], cluster_std=1.0, random_state=3)
    snap = np.load(FIXTURE)
    m = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit(torch.tensor(X, dtype=torch.float64))
    assert np.array_equal(m.labels_.numpy().astype(np.int16), snap[f"{name}_{mcs}_{ms}_labels"])
    assert np.array_equal(m.probabilities_.numpy().astype(np.float32), snap[f"{name}_{mcs}_{ms}_probs"])


def test_explicit_defaults_equal_omitted_arguments():
    X = tiny_data()
    a = fit(X)
    b = fit(X, cluster_selection_epsilon=0.0, cluster_selection_method="eom")
    assert torch.equal(a.labels_, b.labels_) and torch.equal(a.probabilities_, b.probabilities_)


# --- invariants -------------------------------------------------------------------------------------------------
def test_epsilon_merges_monotonically():
    X = tiny_data()
    counts = [fit(X, cluster_selection_epsilon=e).n_clusters_ for e in [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 4 and counts[2] == 2


def test_epsilon_merges_leaf_clusters_too():
    X = tiny_data()
    assert fit(X, cluster_selection_method="leaf").n_clusters_ == 4
    assert fit(X, cluster_selection_method="leaf", cluster_selection_epsilon=2.0).n_clusters_ == 2


def test_huge_epsilon_collapses_to_root_only_with_allow_single_cluster():
    X = tiny_data()
    assert fit(X, cluster_selection_epsilon=20.0).n_clusters_ == 2
    assert fit(X, cluster_selection_epsilon=20.0, allow_single_cluster=True).n_clusters_ == 1


def test_leaf_has_at_least_as_many_clusters_as_eom():
    rs = np.random.RandomState(0)
    X = np.vstack([rs.randn(60, 2) * s + c for s, c in [(0.6, (0, 0)), (0.3, (4, 4)), (0.9, (-4, 3))]])
    assert fit(X, min_cluster_size=8, cluster_selection_method="leaf").n_clusters_ >= fit(X, min_cluster_size=8).n_clusters_


def test_invalid_selection_arguments_raise():
    with pytest.raises(ValueError, match="cluster_selection_method"):
        HDBSCAN(cluster_selection_method="bogus")
    with pytest.raises(ValueError, match="cluster_selection_epsilon"):
        HDBSCAN(cluster_selection_epsilon=-0.1)


def test_single_cluster_mode_marks_points_that_left_before_the_split_as_noise():
    m = fit(blob_data(), min_cluster_size=8, allow_single_cluster=True)
    labels = m.labels_.numpy()
    assert m.n_clusters_ == 1 and (labels == -1).any() and (labels == 0).any()
    assert (m.probabilities_.numpy()[labels == -1] == 0).all()


def test_max_cluster_size_applies_to_the_root_in_single_cluster_mode():
    X = tiny_data()
    m = fit(X, max_cluster_size=45, allow_single_cluster=True)
    assert m.n_clusters_ == 4          # the root (size 100) exceeds the limit, so it cannot be the single cluster


def test_probabilities_in_unit_interval_and_zero_for_noise():
    for kw in [dict(), dict(cluster_selection_method="leaf"), dict(cluster_selection_epsilon=1.0),
               dict(cluster_selection_method="leaf", cluster_selection_epsilon=0.5)]:
        m = fit(tiny_data(), **kw)
        p, l = m.probabilities_.numpy(), m.labels_.numpy()
        assert (p >= 0).all() and (p <= 1 + 1e-6).all() and (p[l == -1] == 0).all()


# --- parity with the installed reference package (skipped when it is not importable) ---------------------------
def _reference():
    try:
        import hdbscan as ref
        return ref.HDBSCAN
    except Exception:  # not installed, or shadowed by a source checkout without compiled extensions
        return None


@pytest.mark.skipif(_reference() is None, reason="scikit-learn-contrib/hdbscan is not importable")
@pytest.mark.parametrize("method", ["eom", "leaf"])
@pytest.mark.parametrize("eps", [0.0, 0.3, 0.8, 2.0])
@pytest.mark.parametrize("asc", [False, True])
def test_matches_reference_package_on_random_datasets(method, eps, asc):
    from sklearn.datasets import make_blobs, make_moons
    Ref = _reference()
    datasets = [make_moons(400, noise=0.06, random_state=0)[0],
                np.vstack([make_blobs(450, centers=[[0, 0], [5, 5], [0, 6]], cluster_std=[0.6, 0.9, 0.5], random_state=1)[0],
                           np.random.RandomState(0).uniform(-3, 9, (40, 2))]),
                np.random.RandomState(2).randn(300, 2)]
    for X in datasets:
        for mcs, ms in [(10, None), (25, 10)]:
            r = Ref(min_cluster_size=mcs, min_samples=(ms or mcs) - 1, allow_single_cluster=asc, cluster_selection_epsilon=eps,
                    cluster_selection_method=method, algorithm="generic").fit(X)
            m = HDBSCAN(min_cluster_size=mcs, min_samples=ms, allow_single_cluster=asc, cluster_selection_epsilon=eps,
                        cluster_selection_method=method).fit(torch.tensor(X, dtype=torch.float64))
            ours, theirs = m.labels_.numpy(), r.labels_
            tree = m._condensed_tree
            if method == "leaf" and asc and not (tree["child_size"] > 1).any():
                # Documented deviation: with no child clusters the reference returns all noise (it marks the root and
                # then un-marks it again); tgmm returns the root as the single cluster, as 'eom' does.
                assert (theirs == -1).all() and m.n_clusters_ == 1
                continue
            if method == "eom":
                assert np.array_equal(ours, theirs)
                assert np.allclose(m.probabilities_.numpy(), r.probabilities_, atol=1e-5)
            else:
                # Leaf clusters are tiny, so ties between equal mutual-reachability distances (which the two MST
                # constructions break differently) can move a handful of points.
                assert (ours == theirs).mean() >= 0.99


# --- small inputs ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 6])
def test_fewer_points_than_min_samples_does_not_crash(n):
    m = HDBSCAN(min_cluster_size=10).fit(torch.randn(n, 2, dtype=torch.float64))
    assert m.labels_.shape == (n,) and (m.labels_ == -1).all()
    assert (m.probabilities_ == 0).all() and m.n_clusters_ == 0
