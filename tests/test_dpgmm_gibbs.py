"""Pytest suite for GaussianMixture's collapsed Gibbs inference (full covariance)."""
import math

import numpy as np
import torch
import pytest
from sklearn.metrics import adjusted_rand_score

from tgmm import GaussianMixture
from tgmm.synthetic_data import generate_gmm_data


def three_blob_data(n_per_cluster=30, seed=0):
    centers = [[0.0, 0.0], [15.0, 0.0], [7.5, 13.0]]
    covs = [1.5 * torch.eye(2).numpy() for _ in range(3)]
    X, labels = generate_gmm_data(centers, covs, [n_per_cluster] * 3, random_state=seed)
    return X.double(), labels


def _dummy_priors(n_features):
    return dict(
        mean_prior=torch.zeros(n_features, dtype=torch.float64),
        mean_precision_prior=1.0,
        covariance_prior=torch.eye(n_features, dtype=torch.float64),
        degrees_of_freedom_prior=float(n_features + 2),
    )


# ============================================================================
# NIW posterior / multivariate-t marginal likelihood (full covariance)
# ============================================================================

def test_niw_posterior_full_matches_hand_computation_for_two_points():
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="full", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64
    model.mean_prior_ = torch.zeros(d, dtype=torch.float64)
    model.covariance_prior_ = torch.eye(d, dtype=torch.float64)
    model.mean_precision_prior = 1.0
    model.degrees_of_freedom_prior = float(d + 2)
    model.reg_covar = 0.0

    x1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    x2 = torch.tensor([3.0, 0.0], dtype=torch.float64)
    n_k = torch.tensor([2.0], dtype=torch.float64)
    sum_x_k = (x1 + x2).unsqueeze(0)
    sum_xxT_k = (torch.outer(x1, x1) + torch.outer(x2, x2)).unsqueeze(0)

    mu_n, lambda_n, nu_n, psi_n = model._niw_posterior_full(n_k, sum_x_k, sum_xxT_k)

    lambda0, nu0 = 1.0, float(d + 2)
    empirical_mean = (x1 + x2) / 2
    expected_lambda_n = lambda0 + 2
    expected_mu_n = (lambda0 * torch.zeros(d, dtype=torch.float64) + (x1 + x2)) / expected_lambda_n
    S = torch.outer(x1 - empirical_mean, x1 - empirical_mean) + torch.outer(x2 - empirical_mean, x2 - empirical_mean)
    cross = (lambda0 * 2 / expected_lambda_n) * torch.outer(empirical_mean, empirical_mean)
    expected_psi_n = torch.eye(d, dtype=torch.float64) + S + cross

    assert lambda_n.item() == pytest.approx(expected_lambda_n)
    assert nu_n.item() == pytest.approx(nu0 + 2)
    assert torch.allclose(mu_n.squeeze(0), expected_mu_n)
    assert torch.allclose(psi_n.squeeze(0), expected_psi_n)


def test_niw_posterior_full_empty_component_falls_back_to_prior():
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="full", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64
    model.mean_prior_ = torch.tensor([1.0, -1.0], dtype=torch.float64)
    model.covariance_prior_ = 2.0 * torch.eye(d, dtype=torch.float64)
    model.mean_precision_prior = 0.5
    model.degrees_of_freedom_prior = float(d + 2)
    model.reg_covar = 0.0

    n_k = torch.tensor([0.0], dtype=torch.float64)
    sum_x_k = torch.zeros(1, d, dtype=torch.float64)
    sum_xxT_k = torch.zeros(1, d, d, dtype=torch.float64)

    mu_n, lambda_n, nu_n, psi_n = model._niw_posterior_full(n_k, sum_x_k, sum_xxT_k)

    assert torch.allclose(mu_n.squeeze(0), model.mean_prior_)
    assert lambda_n.item() == pytest.approx(0.5)
    assert nu_n.item() == pytest.approx(d + 2)
    assert torch.allclose(psi_n.squeeze(0), model.covariance_prior_)


def test_mvt_log_prob_full_matches_scipy_multivariate_t():
    scipy_stats = pytest.importorskip("scipy.stats")
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="full", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64

    mu_n = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    lambda_n = torch.tensor([2.0], dtype=torch.float64)
    nu_n = torch.tensor([6.0], dtype=torch.float64)  # df = nu_n - d + 1 = 5
    psi_n = torch.eye(d, dtype=torch.float64).unsqueeze(0) * 3.0

    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    log_prob = model._mvt_log_prob_full(x, mu_n, lambda_n, nu_n, psi_n)

    df = 5.0
    scale = (psi_n[0] * (lambda_n[0] + 1) / (lambda_n[0] * df)).numpy()
    expected = scipy_stats.multivariate_t.logpdf(x.numpy(), loc=mu_n[0].numpy(), shape=scale, df=df)

    assert log_prob.shape == (1,)
    assert log_prob.item() == pytest.approx(expected, abs=1e-6)


# ============================================================================
# NIW posterior / marginal likelihood: diag covariance
# ============================================================================

def test_mvt_log_prob_diag_matches_scipy_product_of_t():
    scipy_stats = pytest.importorskip("scipy.stats")
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="diag", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64

    n_k = torch.tensor([3.0], dtype=torch.float64)
    mu_n = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    lambda_n = torch.tensor([4.0], dtype=torch.float64)
    nu_n = torch.tensor([5.0], dtype=torch.float64)  # df = nu_n (no -d+1 for diag)
    psi_n = torch.tensor([[2.0, 1.0]], dtype=torch.float64)

    x = torch.tensor([1.0, -1.0], dtype=torch.float64)
    log_prob = model._mvt_log_prob_diag(x, n_k, mu_n, lambda_n, nu_n, psi_n)

    df = 5.0
    expected = 0.0
    for j in range(d):
        scale_j = (psi_n[0, j] * (lambda_n[0] + 1) / (nu_n[0] * lambda_n[0])).item()
        expected += scipy_stats.t.logpdf(x[j].item(), df=df, loc=mu_n[0, j].item(), scale=math.sqrt(scale_j))

    assert log_prob.item() == pytest.approx(expected, abs=1e-6)


def test_niw_posterior_diag_empty_component_falls_back_to_prior():
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="diag", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64
    model.mean_prior_ = torch.tensor([1.0, -1.0], dtype=torch.float64)
    model.covariance_prior_ = torch.tensor([2.0, 3.0], dtype=torch.float64)
    model.mean_precision_prior = 0.5
    model.degrees_of_freedom_prior = float(d + 2)
    model.reg_covar = 0.0

    n_k = torch.tensor([0.0], dtype=torch.float64)
    sum_x_k = torch.zeros(1, d, dtype=torch.float64)
    sum_x2_k = torch.zeros(1, d, dtype=torch.float64)

    mu_n, lambda_n, nu_n, psi_n = model._niw_posterior_diag(n_k, sum_x_k, sum_x2_k)
    assert torch.allclose(mu_n.squeeze(0), model.mean_prior_)
    assert torch.allclose(psi_n.squeeze(0), model.covariance_prior_)
    assert nu_n.item() == pytest.approx(d + 2)


# ============================================================================
# NIW posterior / marginal likelihood: spherical covariance
# ============================================================================

def test_mvt_log_prob_spherical_matches_scipy_isotropic_multivariate_t():
    scipy_stats = pytest.importorskip("scipy.stats")
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="spherical", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64
    model.degrees_of_freedom_prior = 4.0  # nu0

    n_k = torch.tensor([3.0], dtype=torch.float64)
    mu_n = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    lambda_n = torch.tensor([4.0], dtype=torch.float64)
    nu_n = torch.tensor([7.0], dtype=torch.float64)  # nu0 + n_k, unused by the predictive itself
    psi_n = torch.tensor([6.0], dtype=torch.float64)

    x = torch.tensor([1.0, -1.0], dtype=torch.float64)
    log_prob = model._mvt_log_prob_spherical(x, n_k, mu_n, lambda_n, nu_n, psi_n)

    df = 4.0 + 3.0 * d  # nu0 + n_k * d
    scale = (psi_n[0] * (lambda_n[0] + 1) / (lambda_n[0] * df)).item()
    expected = scipy_stats.multivariate_t.logpdf(
        x.numpy(), loc=mu_n[0].numpy(), shape=scale * torch.eye(d).numpy(), df=df)

    assert log_prob.item() == pytest.approx(expected, abs=1e-6)


def test_niw_posterior_spherical_empty_component_falls_back_to_prior():
    d = 2
    model = GaussianMixture(n_components=None, covariance_type="spherical", **_dummy_priors(d))
    model.n_features = d
    model.dtype = torch.float64
    model.mean_prior_ = torch.tensor([1.0, -1.0], dtype=torch.float64)
    model.covariance_prior_ = torch.tensor(2.0, dtype=torch.float64)
    model.mean_precision_prior = 0.5
    model.degrees_of_freedom_prior = float(d + 2)
    model.reg_covar = 0.0

    n_k = torch.tensor([0.0], dtype=torch.float64)
    sum_x_k = torch.zeros(1, d, dtype=torch.float64)
    sum_sq_k = torch.zeros(1, dtype=torch.float64)

    mu_n, lambda_n, nu_n, psi_n = model._niw_posterior_spherical(n_k, sum_x_k, sum_sq_k)
    assert torch.allclose(mu_n.squeeze(0), model.mean_prior_)
    assert psi_n.item() == pytest.approx(model.covariance_prior_.item())


# ============================================================================
# End-to-end truncated Gibbs fit
# ============================================================================

@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_truncated_gibbs_fit_produces_valid_weights(covariance_type):
    X, true_labels = three_blob_data()
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6, covariance_type=covariance_type)
    model = GaussianMixture(n_components=None, max_components=6, covariance_type=covariance_type,
                             alpha=1.0, max_iter=60, burn_in=20, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    assert model.fitted_
    assert model.weights_.shape == (6,)
    assert model.weights_.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert 2 <= model.n_components_ <= 6

    predicted = model.predict(X)
    assert adjusted_rand_score(true_labels.numpy(), predicted.numpy()) > 0.5


def test_truncated_gibbs_weights_are_self_consistent_with_final_partition():
    # weights_ must reflect the same partition that means_/covariances_ and
    # predict() do -- every sweep re-sorts slots by descending occupancy,
    # so slot index k is not the same semantic component across sweeps (see
    # docs/superpowers/specs/2026-07-28-dpgmm-design.md for the failure
    # mode this guards against: a cross-sweep occupancy average can
    # silently split one real cluster's mass across two reported slots).
    rng = np.random.RandomState(0)
    centers = [rng.randn(10) * 8.0 for _ in range(10)]
    covs = [np.eye(10) for _ in range(10)]
    X, true_labels = generate_gmm_data(centers, covs, [100] * 10, random_state=0)
    X = X.double()

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=20)
    model = GaussianMixture(n_components=None, max_components=20, covariance_type="full",
                             alpha=1.0, max_iter=50, burn_in=15, random_state=0, weight_threshold=10.0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    predicted = model.predict(X)
    predicted_counts = torch.bincount(predicted, minlength=model.weights_.shape[0]).double()
    assert torch.allclose(predicted_counts / X.shape[0], model.weights_, atol=1e-6)
    assert model.n_components_ == 10


def test_truncated_gibbs_rejects_tied_covariance():
    # Moved from fit()-time (pre-merge DPGMM) to construction-time (Task 1's
    # eager __init__ validation) -- no priors needed, the tied_* check
    # fires before the priors-required check.
    with pytest.raises(NotImplementedError, match="tied"):
        GaussianMixture(n_components=None, max_components=5, covariance_type="tied_full")
