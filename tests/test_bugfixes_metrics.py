"""Oracle tests (scikit-learn) for the metrics bug fixes: ROC-AUC with ties, confusion_matrix with
predicted-only labels, exact AMI, float32 Calinski-Harabasz, plus a randomized sweep of every metric
that has a scikit-learn counterpart."""
import numpy as np
import pytest
import torch
import sklearn.metrics as skm

from tgmm.metrics import ClusteringMetrics as CM


def _t(a, dtype=torch.long):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


# ---------------------------------------------------------------- ROC AUC
@pytest.mark.parametrize("seed", range(5))
def test_roc_auc_continuous_scores_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    assert CM.roc_auc_score(_t(y, torch.float64), _t(s, torch.float64)) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_roc_auc_hard_predictions_ties(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(400) < 0.3).astype(int)
    pred = np.where(rng.random(400) < 0.8, y, 1 - y)  # hard 0/1 predictions, many ties
    expected = skm.roc_auc_score(y, pred)
    assert CM.roc_auc_score(_t(y, torch.float32), _t(pred, torch.float32)) == pytest.approx(expected, abs=1e-12)
    # independent of the order the samples come in (ties must not depend on sort order)
    perm = rng.permutation(400)
    assert CM.roc_auc_score(_t(y[perm], torch.float32), _t(pred[perm], torch.float32)) == pytest.approx(expected, abs=1e-12)


def test_roc_auc_integer_and_heavy_ties_and_imbalance():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.02).astype(int)
    s = rng.integers(0, 4, 1000)
    assert CM.roc_auc_score(_t(y), _t(s)) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-12)


def test_roc_auc_degenerate_single_class_keeps_documented_value():
    assert CM.roc_auc_score(torch.ones(5), torch.rand(5)) == 1.0
    assert CM.roc_auc_score(torch.zeros(5), torch.rand(5)) == 1.0


def test_classification_report_roc_auc_is_hard_label_auc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    pred = np.where(rng.random(300) < 0.7, y, rng.integers(0, 3, 300))
    rep = CM.classification_report(_t(y), _t(pred))
    for c in range(3):
        assert rep[c]["roc_auc"] == pytest.approx(skm.roc_auc_score(y == c, pred == c), abs=1e-12)


# ------------------------------------------------------- confusion matrix
def test_confusion_matrix_matches_sklearn_when_pred_subset_of_true():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    p = rng.integers(0, 3, 200)
    np.testing.assert_array_equal(CM.confusion_matrix(_t(y), _t(p)).numpy(), skm.confusion_matrix(y, p, labels=np.arange(4)))


def test_confusion_matrix_keeps_predicted_only_labels():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0, 2, 1, 2, 1])  # label 2 is never a true label
    cm = CM.confusion_matrix(_t(y), _t(p)).numpy()
    np.testing.assert_array_equal(cm, skm.confusion_matrix(y, p))  # union {0,1,2}, sorted
    assert cm.sum() == 5  # nothing silently dropped


def test_confusion_matrix_labels_argument():
    y = np.array([3, 3, 7, 7]); p = np.array([3, 7, 7, 7])
    cm = CM.confusion_matrix(_t(y), _t(p), labels=[7, 3]).numpy()
    np.testing.assert_array_equal(cm, skm.confusion_matrix(y, p, labels=[7, 3]))


def test_classification_report_lists_predicted_only_label_with_zero_support():
    y = np.array([0, 0, 1, 1]); p = np.array([0, 2, 1, 2])
    rep = CM.classification_report(_t(y), _t(p))
    assert set(rep) == {0, 1, 2}
    assert rep[2]["support"] == 0 and rep[2]["precision"] == 0.0 and rep[2]["recall"] == 0.0


# ------------------------------------------------------------------- AMI
def _labels(seed, n, kt, kp):
    rng = np.random.default_rng(seed)
    return rng.integers(0, kt, n), rng.integers(0, kp, n)


@pytest.mark.parametrize("seed,n,kt,kp", [(0, 50, 3, 3), (1, 200, 5, 8), (2, 1000, 10, 4), (3, 30, 2, 12), (4, 500, 1, 3)])
def test_ami_matches_sklearn_exactly(seed, n, kt, kp):
    y, p = _labels(seed, n, kt, kp)
    assert CM.adjusted_mutual_info_score(_t(y), _t(p)) == pytest.approx(skm.adjusted_mutual_info_score(y, p), abs=1e-8)


def test_ami_correlated_labelings():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 6, 800)
    p = np.where(rng.random(800) < 0.6, y, rng.integers(0, 6, 800))
    assert CM.adjusted_mutual_info_score(_t(y), _t(p)) == pytest.approx(skm.adjusted_mutual_info_score(y, p), abs=1e-8)


@pytest.mark.parametrize("method", ["arithmetic", "min", "max", "geometric"])
def test_ami_average_methods(method):
    y, p = _labels(7, 300, 4, 6)
    assert CM.adjusted_mutual_info_score(_t(y), _t(p), average_method=method) == pytest.approx(
        skm.adjusted_mutual_info_score(y, p, average_method=method), abs=1e-8)


def test_ami_identical_and_degenerate():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert CM.adjusted_mutual_info_score(_t(y), _t(y)) == pytest.approx(1.0, abs=1e-12)
    one = np.zeros(10, dtype=int)
    assert CM.adjusted_mutual_info_score(_t(one), _t(one)) == pytest.approx(skm.adjusted_mutual_info_score(one, one))


def test_ami_non_contiguous_labels_and_large_table():
    rng = np.random.default_rng(8)
    y = rng.integers(0, 100, 20000); p = rng.integers(0, 100, 20000)
    assert CM.adjusted_mutual_info_score(_t(y * 3 + 5), _t(p * 2 + 1)) == pytest.approx(skm.adjusted_mutual_info_score(y, p), abs=1e-8)


# ---------------------------------------------------- Calinski-Harabasz
def test_calinski_harabasz_float32_matches_sklearn_tightly():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5)).astype(np.float32)
    y = rng.integers(0, 4, 500)
    expected = skm.calinski_harabasz_score(X.astype(np.float64), y)
    got = CM.calinski_harabasz_score(torch.from_numpy(X), _t(y))
    assert abs(got - expected) / expected < 1e-6  # float32 *input* quantisation is the only error left
    expected32 = skm.calinski_harabasz_score(X, y)  # sklearn on the same float32 array computes in float32 too
    X64 = torch.from_numpy(X).double()
    assert abs(CM.calinski_harabasz_score(X64, _t(y)) - expected) / expected < 1e-12


# ----------------------------------------- randomized oracle sweep (all)
LABEL_METRICS = [
    ("rand_score", skm.rand_score, 1e-12),
    ("adjusted_rand_score", skm.adjusted_rand_score, 1e-12),
    ("mutual_info_score", skm.mutual_info_score, 1e-6),
    ("normalized_mutual_info_score", skm.normalized_mutual_info_score, 1e-6),
    ("homogeneity_score", skm.homogeneity_score, 1e-10),
    ("completeness_score", skm.completeness_score, 1e-10),
    ("v_measure_score", skm.v_measure_score, 1e-10),
    ("fowlkes_mallows_score", skm.fowlkes_mallows_score, 1e-6),
]


@pytest.mark.parametrize("name,oracle,tol", LABEL_METRICS)
@pytest.mark.parametrize("seed,n,kt,kp", [(0, 100, 3, 3), (1, 400, 6, 4), (2, 60, 2, 9), (3, 1000, 10, 10)])
def test_label_metrics_match_sklearn(name, oracle, tol, seed, n, kt, kp):
    y, p = _labels(seed, n, kt, kp)
    got = getattr(CM, name)(_t(y), _t(p))
    assert got == pytest.approx(oracle(y, p), abs=tol)


@pytest.mark.parametrize("name,oracle,tol", LABEL_METRICS)
def test_label_metrics_edge_cases(name, oracle, tol):
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3]); p_single = np.zeros(8, dtype=int); p_single_each = np.arange(8)
    for p in (p_single, p_single_each, y.copy()):
        assert getattr(CM, name)(_t(y), _t(p)) == pytest.approx(oracle(y, p), abs=max(tol, 1e-7))


def test_purity_matches_definition():
    y, p = _labels(0, 300, 4, 6)
    cont = skm.cluster.contingency_matrix(y, p)
    assert CM.purity_score(_t(y), _t(p)) == pytest.approx(cont.max(axis=0).sum() / 300, abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_internal_indices_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 4)); y = rng.integers(0, 5, 300)
    Xt = torch.from_numpy(X)
    assert CM.silhouette_score(Xt, _t(y)) == pytest.approx(skm.silhouette_score(X, y), abs=1e-10)
    assert CM.davies_bouldin_index(Xt, _t(y)) == pytest.approx(skm.davies_bouldin_score(X, y), abs=1e-10)
    assert CM.calinski_harabasz_score(Xt, _t(y)) == pytest.approx(skm.calinski_harabasz_score(X, y), rel=1e-10)


def test_silhouette_singleton_cluster_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3)); y = rng.integers(0, 3, 60); y[0] = 3  # a singleton cluster
    assert CM.silhouette_score(torch.from_numpy(X), _t(y)) == pytest.approx(skm.silhouette_score(X, y), abs=1e-10)
