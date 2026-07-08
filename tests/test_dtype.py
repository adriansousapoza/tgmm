"""Regression tests for float64 support (GitHub issue #2).

The library silently assumed float32 everywhere, so fitting on float64
data crashed inside GMMInitializer (mismatched dtypes hitting torch.cdist).
These tests fit/predict/score/sample end-to-end on float64 data and check
that the model's dtype follows the input data instead of being hardcoded.
"""
import os
import tempfile

import pytest
import torch

from tgmm.gmm import GaussianMixture
from tgmm.gmm_init import GMMInitializer
from tgmm.metrics import ClusteringMetrics


@pytest.mark.parametrize("covariance_type", [
    "full", "diag", "spherical", "tied_full", "tied_diag", "tied_spherical",
])
def test_fit_predict_pipeline_preserves_float64(covariance_type):
    torch.manual_seed(0)
    X = torch.randn(120, 3, dtype=torch.float64)

    gmm = GaussianMixture(n_components=3, covariance_type=covariance_type, max_iter=20)
    gmm.fit(X)

    assert gmm.means_.dtype == torch.float64
    assert gmm.covariances_.dtype == torch.float64
    assert gmm.weights_.dtype == torch.float64

    labels = gmm.predict(X)
    proba = gmm.predict_proba(X)
    score = gmm.score(X)
    scores = gmm.score_samples(X)
    samples, sample_labels = gmm.sample(10)

    assert labels.shape == (120,)
    assert isinstance(score, float)
    assert proba.dtype == torch.float64
    assert scores.dtype == torch.float64
    assert samples.dtype == torch.float64


def test_fit_with_map_priors_preserves_float64():
    torch.manual_seed(0)
    X = torch.randn(120, 3, dtype=torch.float64)

    gmm = GaussianMixture(
        n_components=3,
        n_features=3,
        covariance_type="full",
        max_iter=20,
        weight_concentration_prior=torch.ones(3, dtype=torch.float64) * 2.0,
        mean_prior=torch.zeros(3, dtype=torch.float64),
        mean_precision_prior=0.1,
        covariance_prior=torch.eye(3, dtype=torch.float64),
        degrees_of_freedom_prior=5.0,
    )
    gmm.fit(X)

    assert gmm.means_.dtype == torch.float64
    assert gmm.covariances_.dtype == torch.float64
    assert gmm.weights_.dtype == torch.float64


def test_fit_still_defaults_to_float32():
    torch.manual_seed(0)
    X = torch.randn(60, 2)  # default float32

    gmm = GaussianMixture(n_components=2, max_iter=10)
    gmm.fit(X)

    assert gmm.means_.dtype == torch.float32
    assert gmm.covariances_.dtype == torch.float32


@pytest.mark.parametrize("method", ["kpp", "kmeans", "maxdist", "points", "random"])
def test_gmm_initializer_means_preserve_float64(method):
    torch.manual_seed(0)
    data = torch.randn(50, 3, dtype=torch.float64)

    centroids = getattr(GMMInitializer, method)(data, 3)

    assert centroids.dtype == torch.float64


def test_save_load_roundtrip_preserves_float64():
    torch.manual_seed(0)
    X = torch.randn(120, 3, dtype=torch.float64)

    gmm = GaussianMixture(n_components=3, max_iter=20, random_state=0)
    gmm.fit(X)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
        filepath = f.name
    try:
        gmm.save(filepath)
        loaded = GaussianMixture.load(filepath)
        assert loaded.dtype == torch.float64
        assert loaded.means_.dtype == torch.float64

        state = gmm.to_dict()
        gmm2 = GaussianMixture(n_components=3)
        gmm2.load_state_dict(state)
        assert gmm2.dtype == torch.float64
        assert gmm2.means_.dtype == torch.float64
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_float64_fit_is_more_precise_than_float32():
    """The dtype should not just avoid crashing -- float64 must actually carry
    more precision through internal constants (log(2*pi), reg_covar*eye, ...),
    not silently upcast a float32 constant into an otherwise float64 computation."""
    torch.manual_seed(0)
    X32 = torch.randn(200, 3)
    X64 = X32.double()

    gmm32 = GaussianMixture(n_components=3, covariance_type="full", random_state=0, max_iter=50)
    gmm32.fit(X32)
    gmm64 = GaussianMixture(n_components=3, covariance_type="full", random_state=0, max_iter=50)
    gmm64.fit(X64)

    score32 = gmm32.score_samples(X32).double()
    score64 = gmm64.score_samples(X64)

    # Fitting the same data at both precisions should track closely, but not
    # be bitwise identical -- if it were, float64 computations would be
    # silently collapsing back down to float32 precision somewhere.
    assert not torch.equal(score32, score64)
    assert torch.allclose(score32, score64, atol=1e-4)


def test_silhouette_score_supports_float64():
    torch.manual_seed(0)
    X = torch.randn(60, 3, dtype=torch.float64)
    labels = torch.randint(0, 3, (60,))

    score = ClusteringMetrics.silhouette_score(X, labels, 3)

    assert isinstance(score, float)
