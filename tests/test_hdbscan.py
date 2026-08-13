"""Pytest suite for tgmm's PyTorch HDBSCAN implementation, including parity with sklearn."""
import numpy as np
import torch
import pytest
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.metrics import adjusted_rand_score

from tgmm.hdbscan import HDBSCAN
from tgmm.synthetic_data import generate_gmm_data


def noisy_blob_data(seed=0):
    n_samples = [200, 150, 250]
    centers = [np.array([0, 0]), np.array([4, 4]), np.array([0, 4])]
    covs = [0.5 * np.eye(2), 0.4 * np.eye(2), 0.6 * np.eye(2)]
    X, y = generate_gmm_data(
        centers, covs, n_samples,
        random_state=seed, noise_ratio=0.15, noise_scale=1.5,
    )
    return X, y


class TestSklearnParity:
    """tgmm's HDBSCAN and sklearn's both define min_samples as including the
    point itself (the k-th nearest neighbor distance where k = min_samples,
    counting the point as its own 1st neighbor). With that convention matched,
    the two implementations should agree closely."""

    def test_matched_min_samples_agrees_with_sklearn(self):
        X, _ = noisy_blob_data(seed=0)
        X_np = X.cpu().numpy()

        torch_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', device='cpu')
        labels_torch = torch_model.fit_predict(X).cpu().numpy()

        sklearn_model = SklearnHDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
        labels_sklearn = sklearn_model.fit_predict(X_np)

        ari = adjusted_rand_score(labels_sklearn, labels_torch)
        assert ari > 0.99

        n_clusters_torch = torch_model.n_clusters_
        n_clusters_sklearn = len(set(labels_sklearn)) - (1 if -1 in labels_sklearn else 0)
        assert n_clusters_torch == n_clusters_sklearn

    def test_mismatched_min_samples_convention_disagrees(self):
        """Regression guard: min_samples=5 (torch) vs min_samples=4 (sklearn)
        looks like a self-inclusion offset but is actually just a parameter
        mismatch (sklearn's min_samples already includes the point itself),
        so it should NOT agree as well as the matched case above."""
        X, _ = noisy_blob_data(seed=0)
        X_np = X.cpu().numpy()

        torch_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', device='cpu')
        labels_torch = torch_model.fit_predict(X).cpu().numpy()

        sklearn_model = SklearnHDBSCAN(min_cluster_size=15, min_samples=4, metric='euclidean')
        labels_sklearn = sklearn_model.fit_predict(X_np)

        ari = adjusted_rand_score(labels_sklearn, labels_torch)
        assert ari < 0.99

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
    def test_matched_min_samples_agrees_across_metrics(self, metric):
        X, _ = noisy_blob_data(seed=1)
        X_np = X.cpu().numpy()

        torch_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric=metric, device='cpu')
        labels_torch = torch_model.fit_predict(X).cpu().numpy()

        sklearn_model = SklearnHDBSCAN(min_cluster_size=15, min_samples=5, metric=metric)
        labels_sklearn = sklearn_model.fit_predict(X_np)

        ari = adjusted_rand_score(labels_sklearn, labels_torch)
        assert ari > 0.95


class TestMinClusterSize:
    def test_larger_min_cluster_size_merges_or_drops_clusters(self):
        centers = [np.array([0, 0]), np.array([3, 3]), np.array([0, 3]), np.array([3, 0])]
        covs = [0.4 * np.eye(2) for _ in range(4)]
        X, _ = generate_gmm_data(centers, covs, [150, 100, 80, 50], random_state=42)

        n_clusters_by_size = {}
        for min_size in [10, 70, 100]:
            model = HDBSCAN(min_cluster_size=min_size, min_samples=5, device='cpu')
            model.fit_predict(X)
            n_clusters_by_size[min_size] = model.n_clusters_

        assert n_clusters_by_size[10] == 4
        assert n_clusters_by_size[70] == 3
        assert n_clusters_by_size[100] == 2
