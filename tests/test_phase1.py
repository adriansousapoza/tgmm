"""
Phase 1 (v0.4 roadmap): ICL, model selection, and clustering metrics.
See docs/superpowers/specs/2026-10-14-phase1-metrics-model-selection-design.md.

DBCV has its own file, tests/test_metrics_dbcv.py, per the plan.
"""
import math
import os
import warnings

import numpy as np
import pytest
import torch
from sklearn import metrics as skm

from tgmm import GaussianMixture, HDBSCAN
from tgmm.metrics import ClusteringMetrics
from tgmm.model_selection import ModelSelectionResult, select_model

torch.manual_seed(0)


def _blobs(n_per=80, d=2, centers=((0.0, 0.0), (6.0, 0.0), (0.0, 6.0)), seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    chunks = []
    for c in centers:
        chunks.append(torch.randn(n_per, d, generator=gen, dtype=dtype) + torch.tensor(c, dtype=dtype))
    return torch.cat(chunks, dim=0)


X_BLOBS = _blobs()


# ===========================================================================
# icl()
# ===========================================================================

class TestICL:
    def test_icl_ge_bic(self):
        gmm = GaussianMixture(n_components=3, n_features=2, random_state=0, max_iter=200)
        gmm.fit(X_BLOBS)
        assert gmm.icl(X_BLOBS, mode='hard') >= gmm.bic(X_BLOBS) - 1e-8
        assert gmm.icl(X_BLOBS, mode='soft') >= gmm.bic(X_BLOBS) - 1e-8

    def test_icl_equals_bic_when_one_hot(self):
        gmm = GaussianMixture(n_components=3, n_features=2, random_state=0, max_iter=200)
        gmm.fit(X_BLOBS)
        # Force one-hot responsibilities by making components far apart
        # relative to their covariance, so tau is (numerically) one-hot.
        gmm.means_ = torch.tensor([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]], dtype=torch.float64)
        gmm.covariances_ = torch.stack([torch.eye(2, dtype=torch.float64) * 0.01 for _ in range(3)])
        for mode in ('hard', 'soft'):
            assert abs(gmm.icl(X_BLOBS, mode=mode) - gmm.bic(X_BLOBS)) < 1e-6

    def test_icl_invalid_mode_raises(self):
        gmm = GaussianMixture(n_components=3, n_features=2, random_state=0)
        gmm.fit(X_BLOBS)
        with pytest.raises(ValueError):
            gmm.icl(X_BLOBS, mode='nonsense')

    def test_icl_prefers_well_separated_over_overlapping(self):
        # On well-separated blobs, ICL and BIC should be close (near one-hot
        # responsibilities); overlapping data should show a bigger ICL-BIC gap.
        well_separated = _blobs(centers=((0, 0), (20, 0), (0, 20)), seed=1)
        overlapping = _blobs(centers=((0, 0), (1.5, 0), (0, 1.5)), seed=1)

        gmm_sep = GaussianMixture(n_components=3, n_features=2, random_state=0, max_iter=200)
        gmm_sep.fit(well_separated)
        gmm_overlap = GaussianMixture(n_components=3, n_features=2, random_state=0, max_iter=200)
        gmm_overlap.fit(overlapping)

        gap_sep = gmm_sep.icl(well_separated) - gmm_sep.bic(well_separated)
        gap_overlap = gmm_overlap.icl(overlapping) - gmm_overlap.bic(overlapping)
        assert gap_overlap > gap_sep

    def test_icl_gibbs_mode_uses_active_components(self):
        mean_prior, mean_precision_prior, covariance_prior, dof_prior = \
            GaussianMixture.suggest_priors(X_BLOBS, n_components=6, random_state=0)
        gmm = GaussianMixture(
            n_components=None, covariance_type='full', max_components=6,
            mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
            covariance_prior=covariance_prior, degrees_of_freedom_prior=dof_prior,
            max_iter=30, random_state=0,
        )
        gmm.fit(X_BLOBS)
        # Should not raise, and should be finite regardless of how many
        # slots ended up active vs. pruned.
        value = gmm.icl(X_BLOBS)
        assert math.isfinite(value)


# ===========================================================================
# select_model()
# ===========================================================================

class TestSelectModel:
    def test_select_model_basic_sweep(self):
        result = select_model(
            X_BLOBS, n_components=range(1, 6),
            covariance_types=('full', 'diag'),
            criterion='bic', random_state=0, max_iter=100,
        )
        assert len(result.table) == 5 * 2
        assert result.best_ is not None
        assert isinstance(result.models_, dict)
        for row in result.table:
            assert row['bic'] is not None
            assert set(row.keys()) == {
                'n_components', 'covariance_type', 'bic', 'aic', 'icl',
                'log_likelihood', 'converged',
            }

    def test_select_model_best_minimizes_criterion(self):
        result = select_model(
            X_BLOBS, n_components=range(1, 6), covariance_types=('full',),
            criterion='bic', random_state=0, max_iter=100,
        )
        best_bic = min(row['bic'] for row in result.table)
        assert result.best_.bic(X_BLOBS) == pytest.approx(best_bic, abs=1e-6)

    def test_select_model_three_blobs_prefers_three_components(self):
        result = select_model(
            X_BLOBS, n_components=range(1, 7), covariance_types=('full',),
            criterion='bic', random_state=0, n_init=3, max_iter=200,
        )
        assert result.best_.n_components == 3

    def test_select_model_invalid_criterion_raises(self):
        with pytest.raises(ValueError):
            select_model(X_BLOBS, n_components=[2], criterion='not-a-criterion')

    def test_select_model_handles_failing_fits_gracefully(self):
        # n_components larger than n_samples raises inside fit(); the sweep
        # should record a None-filled row instead of propagating.
        tiny_X = X_BLOBS[:3]
        result = select_model(
            tiny_X, n_components=[2, 10], covariance_types=('full',),
            random_state=0,
        )
        rows_by_k = {row['n_components']: row for row in result.table}
        assert rows_by_k[10]['bic'] is None
        assert rows_by_k[10]['converged'] is None
        assert (10, 'full') not in result.models_

    def test_select_model_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        result = select_model(
            X_BLOBS, n_components=range(1, 4), covariance_types=('full',),
            random_state=0, max_iter=50,
        )
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'bic' in df.columns


FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "mclust_icl_fixture.json")


@pytest.mark.skipif(not os.path.exists(FIXTURE_PATH), reason="mclust icl() fixture missing; see tests/fixtures/make_icl_fixture.R")
def test_icl_matches_mclust_fixture():
    import json
    with open(FIXTURE_PATH) as f:
        fixture = json.load(f)
    X = torch.tensor(fixture["X"], dtype=torch.float64)
    gmm = GaussianMixture(n_components=len(fixture["weights"]), n_features=X.shape[1],
                           covariance_type="full")
    gmm.weights_ = torch.tensor(fixture["weights"], dtype=torch.float64)
    gmm.means_ = torch.tensor(fixture["means"], dtype=torch.float64)
    gmm.covariances_ = torch.tensor(fixture["covariances"], dtype=torch.float64)
    gmm.n_components_ = gmm.weights_.shape[0]
    gmm.active_ = torch.ones(gmm.n_components_, dtype=torch.bool)
    assert abs(gmm.icl(X) - fixture["icl"]) < 1e-4


# ===========================================================================
# Internal-index infrastructure: metric, noise_label, n_components-optional,
# chunking, 'precomputed'
# ===========================================================================

def _labeled_blobs(seed=0, n_per=60, d=3, n_clusters=4):
    gen = torch.Generator().manual_seed(seed)
    chunks, labels = [], []
    for k in range(n_clusters):
        center = torch.randn(d, generator=gen) * 6
        chunks.append(torch.randn(n_per, d, generator=gen, dtype=torch.float64) + center)
        labels.append(torch.full((n_per,), k))
    return torch.cat(chunks, dim=0), torch.cat(labels, dim=0)


X4, LABELS4 = _labeled_blobs()


class TestInternalIndexInfra:
    @pytest.mark.parametrize("index_name,sklearn_fn", [
        ("silhouette_score", skm.silhouette_score),
        ("davies_bouldin_index", skm.davies_bouldin_score),
        ("calinski_harabasz_score", skm.calinski_harabasz_score),
    ])
    def test_sklearn_parity_default_metric(self, index_name, sklearn_fn):
        fn = getattr(ClusteringMetrics, index_name)
        got = fn(X4, LABELS4, n_components=None)
        expected = sklearn_fn(X4.numpy(), LABELS4.numpy())
        assert got == pytest.approx(expected, rel=1e-6)

    def test_n_components_optional_handles_noncontiguous_labels(self):
        # Relabel 0,1,2,3 -> 10,20,30,40 (non-contiguous, not 0-indexed).
        remapped = LABELS4 * 10 + 10
        got = ClusteringMetrics.silhouette_score(X4, remapped, n_components=None)
        expected = skm.silhouette_score(X4.numpy(), LABELS4.numpy())
        assert got == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("index_name", [
        "silhouette_score", "dunn_index", "davies_bouldin_index",
        "calinski_harabasz_score", "wb_index", "wemmert_gancarski_index",
    ])
    def test_chunking_matches_unchunked(self, index_name):
        fn = getattr(ClusteringMetrics, index_name)
        kwargs_chunked = {"chunk_size": 17} if index_name in ("silhouette_score", "dunn_index") else {}
        kwargs_full = {"chunk_size": 10**9} if index_name in ("silhouette_score", "dunn_index") else {}
        got_chunked = fn(X4, LABELS4, n_components=None, **kwargs_chunked)
        got_full = fn(X4, LABELS4, n_components=None, **kwargs_full)
        assert got_chunked == pytest.approx(got_full, abs=1e-8)

    @pytest.mark.parametrize("index_name", [
        "davies_bouldin_index", "calinski_harabasz_score", "wb_index", "wemmert_gancarski_index",
    ])
    def test_precomputed_rejected_for_centroid_based_indexes(self, index_name):
        fn = getattr(ClusteringMetrics, index_name)
        with pytest.raises(ValueError):
            fn(X4, LABELS4, n_components=None, metric='precomputed')

    @pytest.mark.parametrize("index_name,maximize", [
        ("silhouette_score", True), ("dunn_index", True),
        ("davies_bouldin_index", False), ("calinski_harabasz_score", True),
        ("wb_index", False), ("wemmert_gancarski_index", True),
    ])
    def test_noise_label_scaling_matches_manual_formula(self, index_name, maximize):
        torch.manual_seed(0)
        labels_with_noise = LABELS4.clone()
        noise_mask = torch.rand(labels_with_noise.shape[0]) < 0.1
        labels_with_noise[noise_mask] = -1

        fn = getattr(ClusteringMetrics, index_name)
        scaled = fn(X4, labels_with_noise, n_components=None, noise_label=-1)

        keep = ~noise_mask
        unscaled = fn(X4[keep], labels_with_noise[keep], n_components=None, noise_label=None)

        n = labels_with_noise.shape[0]
        n_noise = int(noise_mask.sum().item())
        f = (n - n_noise) / n
        expected = unscaled * f if maximize else unscaled / f
        assert scaled == pytest.approx(expected, rel=1e-6)

    def test_mahalanobis_matches_whitened_euclidean(self):
        torch.manual_seed(0)
        d = X4.shape[1]
        A = torch.randn(d, d, dtype=torch.float64)
        VI = A @ A.T + torch.eye(d, dtype=torch.float64) * d  # SPD precision matrix

        got = ClusteringMetrics.silhouette_score(X4, LABELS4, n_components=None, metric='mahalanobis', VI=VI)

        L = torch.linalg.cholesky(VI)
        X_whitened = X4 @ L
        expected = ClusteringMetrics.silhouette_score(X_whitened, LABELS4, n_components=None, metric='euclidean')
        assert got == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# WB index / Wemmert-Gancarski: NumPy brute-force references
# ===========================================================================

def _wb_index_reference(X, labels):
    X = X.numpy()
    labels = labels.numpy()
    K = len(np.unique(labels))
    overall = X.mean(axis=0)
    ssw = 0.0
    ssb = 0.0
    for k in np.unique(labels):
        Xk = X[labels == k]
        ck = Xk.mean(axis=0)
        ssw += ((Xk - ck) ** 2).sum()
        ssb += len(Xk) * ((ck - overall) ** 2).sum()
    return K * ssw / ssb


def _wemmert_gancarski_reference(X, labels):
    X = X.numpy()
    labels = labels.numpy()
    n = len(labels)
    uniq = np.unique(labels)
    centroids = {k: X[labels == k].mean(axis=0) for k in uniq}
    total = 0.0
    for k in uniq:
        Xk = X[labels == k]
        own_dist = np.linalg.norm(Xk - centroids[k], axis=1)
        other_min = np.full(len(Xk), np.inf)
        for j in uniq:
            if j == k:
                continue
            d = np.linalg.norm(Xk - centroids[j], axis=1)
            other_min = np.minimum(other_min, d)
        R = own_dist / other_min
        J = max(0.0, 1.0 - R.mean())
        total += len(Xk) * J
    return total / n


class TestWBAndWemmertGancarski:
    def test_wb_index_matches_reference(self):
        got = ClusteringMetrics.wb_index(X4, LABELS4, n_components=None)
        expected = _wb_index_reference(X4, LABELS4)
        assert got == pytest.approx(expected, rel=1e-6)

    def test_wemmert_gancarski_matches_reference(self):
        got = ClusteringMetrics.wemmert_gancarski_index(X4, LABELS4, n_components=None)
        expected = _wemmert_gancarski_reference(X4, LABELS4)
        assert got == pytest.approx(expected, rel=1e-6)

    def test_wb_lower_for_well_separated_blobs(self):
        well_sep, well_labels = _labeled_blobs(seed=1)
        well_sep = well_sep * 1.0  # base spread 1.0 via randn
        overlap_centers_X = torch.cat([
            torch.randn(60, 3, dtype=torch.float64) + torch.tensor([float(i), 0.0, 0.0], dtype=torch.float64)
            for i in range(4)
        ])
        overlap_labels = torch.cat([torch.full((60,), i) for i in range(4)])
        wb_sep = ClusteringMetrics.wb_index(well_sep, well_labels, n_components=None)
        wb_overlap = ClusteringMetrics.wb_index(overlap_centers_X, overlap_labels, n_components=None)
        assert wb_sep < wb_overlap  # lower is better; well-separated should win


# ===========================================================================
# External metrics: homogeneity / completeness / v_measure / NID
# ===========================================================================

def _nid_reference(labels_true, labels_pred):
    lt = labels_true.numpy()
    lp = labels_pred.numpy()
    n = len(lt)
    classes_t = np.unique(lt)
    classes_p = np.unique(lp)
    contingency = np.zeros((len(classes_t), len(classes_p)))
    for i, c in enumerate(classes_t):
        for j, k in enumerate(classes_p):
            contingency[i, j] = np.sum((lt == c) & (lp == k))

    def entropy(counts):
        p = counts / n
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    h_true = entropy(contingency.sum(axis=1))
    h_pred = entropy(contingency.sum(axis=0))

    p_joint = contingency / n
    p_t = contingency.sum(axis=1, keepdims=True) / n
    p_p = contingency.sum(axis=0, keepdims=True) / n
    mi = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            if p_joint[i, j] > 0:
                mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_t[i, 0] * p_p[0, j]))

    return 1.0 - mi / max(h_true, h_pred)


class TestExternalMetricsFixAndNID:
    def test_homogeneity_matches_sklearn_asymmetric_cardinality(self):
        torch.manual_seed(3)
        n = 400
        labels_true = torch.randint(0, 3, (n,))
        labels_pred = torch.randint(0, 11, (n,))
        labels_pred = torch.where(torch.rand(n) < 0.5, labels_true, labels_pred)
        lt, lp = labels_true.numpy(), labels_pred.numpy()

        assert ClusteringMetrics.homogeneity_score(labels_true, labels_pred) == \
            pytest.approx(skm.homogeneity_score(lt, lp), abs=1e-6)
        assert ClusteringMetrics.completeness_score(labels_true, labels_pred) == \
            pytest.approx(skm.completeness_score(lt, lp), abs=1e-6)
        assert ClusteringMetrics.v_measure_score(labels_true, labels_pred) == \
            pytest.approx(skm.v_measure_score(lt, lp), abs=1e-6)

    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
    def test_v_measure_beta_matches_sklearn(self, beta):
        torch.manual_seed(4)
        n = 300
        labels_true = torch.randint(0, 4, (n,))
        labels_pred = torch.randint(0, 6, (n,))
        labels_pred = torch.where(torch.rand(n) < 0.6, labels_true, labels_pred)
        lt, lp = labels_true.numpy(), labels_pred.numpy()
        got = ClusteringMetrics.v_measure_score(labels_true, labels_pred, beta=beta)
        expected = skm.v_measure_score(lt, lp, beta=beta)
        assert got == pytest.approx(expected, abs=1e-6)

    def test_normalized_information_distance_matches_reference(self):
        torch.manual_seed(5)
        n = 250
        labels_true = torch.randint(0, 4, (n,))
        labels_pred = torch.randint(0, 5, (n,))
        labels_pred = torch.where(torch.rand(n) < 0.5, labels_true, labels_pred)
        got = ClusteringMetrics.normalized_information_distance(labels_true, labels_pred)
        expected = _nid_reference(labels_true, labels_pred)
        assert got == pytest.approx(expected, abs=1e-6)

    def test_normalized_information_distance_zero_for_identical_partitions(self):
        torch.manual_seed(6)
        labels = torch.randint(0, 5, (200,))
        assert abs(ClusteringMetrics.normalized_information_distance(labels, labels)) < 1e-6

    def test_normalized_information_distance_range(self):
        torch.manual_seed(7)
        labels_true = torch.randint(0, 4, (200,))
        labels_pred = torch.randint(0, 4, (200,))
        value = ClusteringMetrics.normalized_information_distance(labels_true, labels_pred)
        assert -1e-8 <= value <= 1.0 + 1e-8


# ===========================================================================
# noise_label on real HDBSCAN output
# ===========================================================================

def test_internal_indexes_accept_real_hdbscan_noise_labels():
    torch.manual_seed(0)
    X = torch.cat([
        torch.randn(80, 2, dtype=torch.float64) + torch.tensor([0.0, 0.0], dtype=torch.float64),
        torch.randn(80, 2, dtype=torch.float64) + torch.tensor([10.0, 0.0], dtype=torch.float64),
        torch.rand(15, 2, dtype=torch.float64) * 40 - 20,  # scattered noise
    ])
    clusterer = HDBSCAN(min_cluster_size=10)
    labels = clusterer.fit_predict(X)
    assert (labels == -1).any(), "expected at least some HDBSCAN noise points for this test to be meaningful"

    for fn_name in ("silhouette_score", "dunn_index"):
        fn = getattr(ClusteringMetrics, fn_name)
        value = fn(X, labels, n_components=None, noise_label=-1)
        assert math.isfinite(value)

    dbcv_value = ClusteringMetrics.dbcv(X, labels, noise_label=-1)
    assert math.isfinite(dbcv_value)
