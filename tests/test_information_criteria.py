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


# ============================================================================
# GaussianMixture, unbounded Gibbs mode: bic()/aic() must charge for the
# same component set they score, not silently mix "likelihood of the full
# mixture" with "parameter count of just the active components."
#
# In truncated Gibbs mode this mismatch is harmless: pruned slots end up
# with essentially zero weight, so including/excluding them from the
# likelihood term barely moves it. Unbounded Gibbs mode is different --
# weight_threshold prunes by *expected count* (weight * n_samples), so a
# component with a real, non-negligible weight can still be pruned purely
# because n_samples is small, and that component then measurably shifts
# score()'s likelihood despite not being charged for in _n_parameters().
# This reproduces that: a genuinely non-negligible-weight second component,
# manually marked inactive via an artificially strict weight_threshold
# (mirroring how make_fitted_gibbs_gmm above injects a controlled, reproducible
# state without running the -- much heavier and non-deterministic -- Gibbs
# sampler itself).
# ============================================================================

def make_fitted_unbounded_gibbs_gmm_with_pruned_component(weight_threshold):
    n_features = 1
    mean_prior = torch.zeros(n_features, dtype=torch.float64)
    mean_precision_prior = 1e-6
    covariance_prior = torch.eye(n_features, dtype=torch.float64)
    degrees_of_freedom_prior = float(n_features + 1)

    # max_components=None: unbounded Gibbs mode (as opposed to the truncated
    # mode used by make_fitted_gibbs_gmm above).
    model = GaussianMixture(n_components=None, n_features=n_features,
                             covariance_type="full", max_components=None,
                             mean_prior=mean_prior,
                             mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior,
                             degrees_of_freedom_prior=degrees_of_freedom_prior,
                             weight_threshold=weight_threshold, device="cpu")
    model.dtype = torch.float64
    model.fitted_ = True
    model.converged_ = True
    # Two components close enough together that the second one's density
    # measurably overlaps the first's over the tested range, so pruning it
    # from the likelihood term actually moves the log-likelihood -- not just
    # in principle.
    model.means_ = torch.tensor([[0.0], [1.5]], dtype=torch.float64)
    model.covariances_ = torch.eye(n_features, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)
    model.weights_ = torch.tensor([0.95, 0.05], dtype=torch.float64)
    # weight_threshold is set high enough (relative to n_samples used in the
    # test below) that the second component's expected count falls at/under
    # it despite its weight (0.05) being far from negligible.
    n_samples = 200
    model.active_ = (model.weights_ * n_samples) > weight_threshold
    model.n_components_ = int(model.active_.sum().item())
    return model


def test_gibbs_unbounded_bic_aic_use_active_only_likelihood_not_full_score():
    # weight_threshold=15: component 1's expected count is 0.05*200=10,
    # which is <= 15, so it's pruned; component 0's is 0.95*200=190 > 15,
    # so it stays active. n_components_ == 1 while weights_.shape[0] == 2.
    model = make_fitted_unbounded_gibbs_gmm_with_pruned_component(weight_threshold=15.0)
    assert model.n_components_ == 1
    assert model.weights_.shape[0] == 2

    X = torch.linspace(-3.0, 4.5, 200, dtype=torch.float64).unsqueeze(1)
    n_samples = X.shape[0]

    full_ll = model.score(X)          # score() still reflects the full 2-component mixture
    active_ll = model._active_score(X)  # log-likelihood under component 0 alone

    # Independent closed-form check: with only component 0 active (mean=0,
    # covariance=I in 1-D) and correctly renormalized to weight 1.0, the
    # active-only mixture is exactly a standard N(0, 1) -- computed here
    # from the Gaussian log-density formula directly, not by calling
    # anything _active_score/bic/aic themselves use, so this actually
    # exercises the renormalization and masking rather than just checking
    # self-consistency between bic/aic and _active_score.
    expected_active_ll = (-0.5 * math.log(2 * math.pi) - 0.5 * X.squeeze(1) ** 2).mean().item()
    assert active_ll == pytest.approx(expected_active_ll)

    # Sanity check this is actually a reproducing case: the pruned
    # component's overlap with the tested range must move the likelihood
    # by a non-trivial amount, or this test wouldn't catch a regression
    # back to the old (buggy) behavior.
    assert active_ll != pytest.approx(full_ll)
    assert abs(active_ll - full_ll) > 0.05  # nats/sample

    n_params = model._n_parameters()  # charges for n_components_ == 1
    expected_bic = -2.0 * active_ll * n_samples + n_params * math.log(n_samples)
    expected_aic = -2.0 * active_ll * n_samples + 2.0 * n_params

    assert model.bic(X) == pytest.approx(expected_bic)
    assert model.aic(X) == pytest.approx(expected_aic)

    # And explicitly confirm this differs from the pre-fix behavior, which
    # would have paired the 1-component parameter count with the full
    # 2-component score() likelihood instead.
    old_buggy_bic = -2.0 * full_ll * n_samples + n_params * math.log(n_samples)
    old_buggy_aic = -2.0 * full_ll * n_samples + 2.0 * n_params
    assert model.bic(X) != pytest.approx(old_buggy_bic)
    assert model.aic(X) != pytest.approx(old_buggy_aic)

    # score()/score_samples() themselves must stay unaffected -- bic/aic's
    # fix must not change what score() returns.
    assert model.score(X) == pytest.approx(full_ll)


def test_gibbs_truncated_bic_aic_unaffected_when_no_pruning():
    # No pruning (n_components_ == weights_.shape[0]) must be a no-op:
    # bic()/aic() should behave exactly as before this fix, i.e. use plain
    # score(). Reuses make_fitted_gibbs_gmm (truncated mode) from above with
    # n_total_slots == n_components.
    model = make_fitted_gibbs_gmm(n_components=4, n_total_slots=4)
    assert model.n_components_ == model.weights_.shape[0]
    X = torch.randn(150, model.n_features, dtype=torch.float64) * 5

    assert model._score_for_information_criterion(X) == pytest.approx(model.score(X))
