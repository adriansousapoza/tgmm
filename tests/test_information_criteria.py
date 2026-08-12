"""BIC/AIC on GaussianMixture: formula correctness and sklearn parity.

BIC/AIC only need a model's own log-likelihood and parameter count, so they
live as `bic`/`aic` methods on the model classes themselves (not as free
functions in `tgmm.metrics.ClusteringMetrics` that take a model apart).
"""
import math

import numpy as np
import pytest
import torch
from sklearn.mixture import GaussianMixture as SklearnGMM

from tgmm.gmm import GaussianMixture


def generate_test_data(n_samples=300, n_features=3, n_clusters=3, seed=0):
    rng = np.random.RandomState(seed)
    X = np.concatenate([
        rng.randn(n_samples // n_clusters, n_features) + rng.uniform(-5, 5, n_features)
        for _ in range(n_clusters)
    ], axis=0)
    return X


X_NP = generate_test_data()
X_TORCH = torch.tensor(X_NP, dtype=torch.float64)


# ============================================================================
# GaussianMixture: parameter counting
#
# Verified against the *fitted* covariances_/means_/weights_ tensors' actual
# shapes (via _expected_covar_shape's contract), not by re-deriving the same
# n*(n+1)/2 arithmetic independently -- for full/diag/spherical/tied_full,
# the sklearn-parity test below is the real independent check (a wrong
# cov_params there would shift bic/aic by n_params_diff * log(n), far past
# its tolerance). tied_diag/tied_spherical have no sklearn equivalent, so
# this fitted-shape check is what catches a wrong count for those two.
# ============================================================================

@pytest.mark.parametrize("covariance_type", [
    "full", "diag", "spherical", "tied_full", "tied_diag", "tied_spherical",
])
def test_gmm_n_parameters_matches_fitted_shapes(covariance_type):
    gmm = GaussianMixture(n_components=3, covariance_type=covariance_type,
                           max_iter=50, random_state=0, device="cpu")
    gmm.fit(X_TORCH)

    if covariance_type in ("full", "tied_full"):
        d = gmm.covariances_.shape[-1]
        n_matrices = gmm.covariances_.numel() // (d * d)
        free_per_matrix = d * (d + 1) // 2  # symmetric matrix: upper triangle + diagonal
        cov_params = n_matrices * free_per_matrix
    else:
        cov_params = gmm.covariances_.numel()

    expected = cov_params + gmm.means_.numel() + gmm.weights_.numel() - 1
    assert gmm._n_parameters() == expected


# ============================================================================
# GaussianMixture: bic/aic match their own defining formula
# ============================================================================

def test_gmm_bic_aic_match_manual_formula():
    gmm = GaussianMixture(n_components=3, covariance_type="full", max_iter=200,
                           random_state=0, device="cpu")
    gmm.fit(X_TORCH)

    n_samples = X_TORCH.shape[0]
    ll = gmm.score(X_TORCH)
    n_params = gmm._n_parameters()

    expected_bic = -2.0 * ll * n_samples + n_params * math.log(n_samples)
    expected_aic = -2.0 * ll * n_samples + 2.0 * n_params

    assert gmm.bic(X_TORCH) == pytest.approx(expected_bic)
    assert gmm.aic(X_TORCH) == pytest.approx(expected_aic)


# ============================================================================
# GaussianMixture: bic/aic match sklearn given identical fitted parameters
#
# Fitting both independently and comparing bic/aic would conflate two things:
# whether the bic/aic *formula* agrees, and whether the two EM implementations
# landed on the same local optimum. To isolate the formula, we fit sklearn's
# GaussianMixture, copy its fitted parameters into a tgmm GaussianMixture
# (bypassing fit() entirely), and compare score/bic/aic computed by each
# implementation over the *same* parameters and the *same* data -- any
# remaining difference is a real bug in the log-density, not sampler noise.
# ============================================================================

@pytest.mark.parametrize("tgmm_cov_type,sklearn_cov_type", [
    ("full", "full"),
    ("diag", "diag"),
    ("spherical", "spherical"),
    ("tied_full", "tied"),
])
def test_gmm_bic_aic_matches_sklearn_given_same_parameters(tgmm_cov_type, sklearn_cov_type):
    n_components = 3
    sk = SklearnGMM(n_components=n_components, covariance_type=sklearn_cov_type,
                     max_iter=200, random_state=0)
    sk.fit(X_NP)

    n_features = X_NP.shape[1]
    gmm = GaussianMixture(n_components=n_components, n_features=n_features,
                           covariance_type=tgmm_cov_type, device="cpu")
    gmm.dtype = torch.float64
    gmm.fitted_ = True
    gmm.converged_ = True
    gmm.means_ = torch.tensor(sk.means_, dtype=torch.float64)
    gmm.weights_ = torch.tensor(sk.weights_, dtype=torch.float64)
    gmm.covariances_ = torch.tensor(sk.covariances_, dtype=torch.float64)

    assert gmm.score(X_TORCH) == pytest.approx(sk.score(X_NP), abs=1e-6)
    assert gmm.bic(X_TORCH) == pytest.approx(sk.bic(X_NP), rel=1e-6)
    assert gmm.aic(X_TORCH) == pytest.approx(sk.aic(X_NP), rel=1e-6)


# ============================================================================
# GaussianMixture with Gibbs mode: parameter counting uses n_components_
# (active slots), not the total number of slots in weights_ -- a truncated-Gibbs
# fit always has exactly max_components slots, many possibly near-zero-weight
# and pruned from active_/n_components_.
# ============================================================================

def make_fitted_gibbs_gmm(n_components=3, n_features=2, n_total_slots=None):
    n_total_slots = n_total_slots or n_components
    # Provide required NIW priors for Gibbs mode
    mean_prior = torch.zeros(n_features, dtype=torch.float64)
    mean_precision_prior = 1e-6
    covariance_prior = torch.eye(n_features, dtype=torch.float64)
    degrees_of_freedom_prior = float(n_features + 1)

    model = GaussianMixture(n_components=None, n_features=n_features,
                            covariance_type="full", max_components=n_total_slots,
                            mean_prior=mean_prior,
                            mean_precision_prior=mean_precision_prior,
                            covariance_prior=covariance_prior,
                            degrees_of_freedom_prior=degrees_of_freedom_prior,
                            device="cpu")
    model.dtype = torch.float64
    model.fitted_ = True
    model.converged_ = True
    model.means_ = torch.stack([
        torch.full((n_features,), 10.0 * k, dtype=torch.float64) for k in range(n_total_slots)
    ])
    model.covariances_ = torch.eye(n_features, dtype=torch.float64).unsqueeze(0).repeat(n_total_slots, 1, 1)

    weights = torch.zeros(n_total_slots, dtype=torch.float64)
    weights[:n_components] = 1.0 / n_components
    # Any remaining slots get a negligible (but nonzero) weight, mirroring a
    # truncated-Gibbs fit that hasn't pruned unused slots down to exactly 0.
    if n_total_slots > n_components:
        leftover = 1e-6
        weights[:n_components] -= leftover / n_components
        weights[n_components:] = leftover / (n_total_slots - n_components)
    model.weights_ = weights

    model.active_ = weights * 1000 > 1.0  # mirrors _finalize_active_components's default threshold
    model.n_components_ = int(model.active_.sum().item())
    return model


def test_gibbs_gmm_n_parameters_uses_active_count_not_total_slots():
    model = make_fitted_gibbs_gmm(n_components=3, n_total_slots=6)
    assert model.n_components_ == 3  # 3 slots pruned as inactive
    d = model.n_features
    expected = 3 * d * (d + 1) / 2 + 3 * d + 2  # full covariance, k=3
    assert model._n_parameters() == int(expected)


def test_gibbs_gmm_bic_aic_match_manual_formula():
    model = make_fitted_gibbs_gmm(n_components=4, n_total_slots=4)
    X = torch.randn(200, model.n_features, dtype=torch.float64) * 5

    n_samples = X.shape[0]
    ll = model.score(X)
    n_params = model._n_parameters()

    expected_bic = -2.0 * ll * n_samples + n_params * math.log(n_samples)
    expected_aic = -2.0 * ll * n_samples + 2.0 * n_params

    assert model.bic(X) == pytest.approx(expected_bic)
    assert model.aic(X) == pytest.approx(expected_aic)
