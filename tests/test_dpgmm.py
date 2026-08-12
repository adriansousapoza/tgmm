"""Pytest suite for GaussianMixture's Gibbs-mode (n_components=None) scaffolding, construction, and prediction."""
import os
import tempfile

import pytest
import torch

from tgmm import GaussianMixture


def _dummy_priors(n_features, covariance_type="full"):
    """Trivial NIW prior values satisfying __init__'s required-priors check.

    Used only by tests that bypass fit() and manually inject fitted state
    (make_fitted below) -- the actual values are never consulted, since
    _resolve_priors_gibbs (which reads them) is never called.
    """
    if covariance_type in ("full", "tied_full"):
        covariance_prior = torch.eye(n_features, dtype=torch.float64)
    elif covariance_type in ("diag", "tied_diag"):
        covariance_prior = torch.ones(n_features, dtype=torch.float64)
    else:
        covariance_prior = torch.tensor(1.0, dtype=torch.float64)
    return dict(
        mean_prior=torch.zeros(n_features, dtype=torch.float64),
        mean_precision_prior=1.0,
        covariance_prior=covariance_prior,
        degrees_of_freedom_prior=float(n_features + 2),
    )


def make_fitted(covariance_type="full", n_components=3, n_features=2):
    """Build a Gibbs-mode GaussianMixture with a manually-injected fitted
    state, bypassing fit().

    Mirrors the isolation technique used in test_sampling_memory.py: lets us
    test predict/score/sample scaffolding independently of the (much
    heavier) Gibbs fitting machinery.
    """
    model = GaussianMixture(n_components=None, max_components=n_components, n_features=n_features,
                             covariance_type=covariance_type, **_dummy_priors(n_features, covariance_type))
    model.dtype = torch.float64
    model.means_ = torch.stack([
        torch.full((n_features,), 10.0 * k, dtype=torch.float64) for k in range(n_components)
    ])
    model.weights_ = torch.full((n_components,), 1.0 / n_components, dtype=torch.float64)

    if covariance_type == "full":
        model.covariances_ = torch.eye(n_features, dtype=torch.float64).unsqueeze(0).repeat(n_components, 1, 1)
    elif covariance_type == "diag":
        model.covariances_ = torch.ones(n_components, n_features, dtype=torch.float64)
    elif covariance_type == "spherical":
        model.covariances_ = torch.ones(n_components, dtype=torch.float64)

    model.fitted_ = True
    model.converged_ = True
    model.n_components_ = n_components
    model.active_ = torch.ones(n_components, dtype=torch.bool)
    return model


# ============================================================================
# Construction / validation
# ============================================================================

def test_invalid_covariance_type_raises():
    with pytest.raises(ValueError, match="covariance_type"):
        GaussianMixture(n_components=None, covariance_type="not_a_real_type",
                         **_dummy_priors(2))


def test_defaults():
    model = GaussianMixture(n_components=None, **_dummy_priors(2))
    assert model.max_components == 20
    assert model.alpha == 1.0
    assert model.covariance_type == "full"
    assert model.fitted_ is False


def test_max_components_none_allowed_for_unbounded_gibbs():
    # Per design: None is accepted at construction time and triggers the
    # unbounded (dynamic component count) Gibbs sampler at fit() time.
    GaussianMixture(n_components=None, max_components=None, **_dummy_priors(2))


def test_resolved_seed_k_for_unbounded_gibbs_is_generous():
    # The unbounded Gibbs sampler's initial kmeans seed K must be generous:
    # single-site Gibbs moves split a merged-at-init cluster into two only
    # very slowly, so a too-small seed K can get permanently stuck below
    # the true number of clusters.
    model = GaussianMixture(n_components=None, max_components=None, **_dummy_priors(2))
    assert model._resolve_max_components(1000) == 20


# ============================================================================
# Prediction / scoring on a manually-fitted state (full covariance)
# ============================================================================

def test_predict_assigns_nearest_component():
    model = make_fitted("full", n_components=3, n_features=2)
    X = torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=torch.float64)
    labels = model.predict(X)
    assert labels.tolist() == [0, 1, 2]  # X[k] sits exactly at component k's mean


def test_predict_proba_rows_sum_to_one():
    model = make_fitted("full")
    X = torch.randn(10, 2, dtype=torch.float64)
    proba = model.predict_proba(X)
    assert proba.shape == (10, 3)
    assert torch.allclose(proba.sum(dim=1), torch.ones(10, dtype=torch.float64), atol=1e-6)


def test_score_samples_matches_score_mean():
    model = make_fitted("full")
    X = torch.randn(10, 2, dtype=torch.float64)
    per_sample = model.score_samples(X)
    assert per_sample.shape == (10,)
    assert model.score(X) == pytest.approx(per_sample.mean().item())


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_predict_proba_works_for_every_gibbs_native_covariance_type(covariance_type):
    # Narrowed from the pre-merge version's all-6-types parametrization:
    # n_components=None + tied_* now raises at construction (see
    # test_gibbs_mode_rejects_tied_covariance_at_construction in
    # test_gmm.py), so tied_* is unreachable here. predict_proba's
    # tied_*-covariance coverage lives in test_gmm.py via EM-fitted models
    # instead -- the code path is identical either way (predict_proba is
    # mode-agnostic).
    model = make_fitted(covariance_type)
    X = torch.randn(10, 2, dtype=torch.float64)
    proba = model.predict_proba(X)
    assert proba.shape == (10, 3)
    assert torch.allclose(proba.sum(dim=1), torch.ones(10, dtype=torch.float64), atol=1e-6)


# ============================================================================
# Sampling
# ============================================================================

def test_sample_shape_and_component_membership():
    model = make_fitted("full")
    X, indices = model.sample(100)
    assert X.shape == (100, 2)
    assert indices.shape == (100,)
    assert set(indices.tolist()) <= {0, 1, 2}


# ============================================================================
# Save / load round trip
# ============================================================================

def test_save_load_round_trip_preserves_predictions():
    model = make_fitted("full")
    X = torch.randn(10, 2, dtype=torch.float64)
    expected = model.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "gmm_gibbs.pt")
        model.save(path)
        loaded = GaussianMixture.load(path)

    assert torch.equal(loaded.predict(X), expected)
    assert loaded.covariance_type == model.covariance_type
    assert loaded.n_components_ == model.n_components_
    assert loaded.dtype == model.dtype

    sampled, indices = loaded.sample(10)  # exercises dtype consistency (means_ vs. sampled noise)
    assert sampled.shape == (10, 2)
