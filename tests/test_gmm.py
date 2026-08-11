"""Pytest suite for GaussianMixture covering settings, combinations, and edge cases."""
import os
import tempfile

import pytest
import torch

from tgmm.gmm import GaussianMixture


def generate_test_data(n_samples=200, n_features=3, n_clusters=3, seed=42):
    """Generate synthetic clustered data."""
    generator = torch.Generator().manual_seed(seed)
    X_list = []
    for _ in range(n_clusters):
        mean = torch.randn(n_features, generator=generator) * 5
        cov = torch.eye(n_features) + torch.randn(n_features, n_features, generator=generator) * 0.1
        cov = cov @ cov.T  # Make positive definite
        X_cluster = torch.randn(n_samples // n_clusters, n_features, generator=generator) @ cov.T + mean
        X_list.append(X_cluster)
    return torch.cat(X_list, dim=0)


X_2D = generate_test_data(n_samples=200, n_features=2, n_clusters=3)
X_3D = generate_test_data(n_samples=200, n_features=3, n_clusters=4)
X_5D = generate_test_data(n_samples=300, n_features=5, n_clusters=5)
X_10D = generate_test_data(n_samples=400, n_features=10, n_clusters=3)


# ============================================================================
# 1. Covariance types
# ============================================================================

@pytest.mark.parametrize("covariance_type,expected_shape", [
    ("full", (3, 3, 3)),
    ("diag", (3, 3)),
    ("spherical", (3,)),
    ("tied_full", (3, 3)),
    ("tied_diag", (3,)),
])
def test_covariance_type_shapes(covariance_type, expected_shape):
    gmm = GaussianMixture(n_components=3, covariance_type=covariance_type)
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert gmm.covariances_.shape == expected_shape


def test_tied_spherical_covariance_is_scalar():
    gmm = GaussianMixture(n_components=3, covariance_type="tied_spherical")
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert gmm.covariances_.dim() == 0


def test_tied_alias_resolves_to_tied_full():
    gmm = GaussianMixture(n_components=3, covariance_type="tied")
    gmm.fit(X_3D)
    assert gmm.covariance_type == "tied_full"
    assert gmm.fitted_


def test_isotropic_alias_resolves_to_spherical():
    gmm = GaussianMixture(n_components=3, covariance_type="isotropic")
    gmm.fit(X_3D)
    assert gmm.covariance_type == "spherical"
    assert gmm.fitted_


# ============================================================================
# 2. Mean initialization methods
# ============================================================================

@pytest.mark.parametrize("method", ["kmeans", "kpp", "random", "points", "maxdist"])
def test_init_means_methods(method):
    gmm = GaussianMixture(n_components=3, init_means=method)
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert gmm.means_.shape == (3, 3)


def test_init_means_tensor():
    means = torch.randn(3, 3)
    gmm = GaussianMixture(n_components=3, init_means=means)
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert torch.allclose(gmm.initial_means_.cpu(), means.cpu(), atol=1e-5)


# ============================================================================
# 3. Weight initialization methods
# ============================================================================

def test_init_weights_uniform():
    gmm = GaussianMixture(n_components=3, init_weights="uniform")
    gmm.fit(X_3D)
    assert gmm.fitted_
    expected = torch.ones(3, device=gmm.device) / 3
    assert torch.allclose(gmm.initial_weights_, expected, atol=1e-5)


@pytest.mark.parametrize("method", ["random", "kmeans"])
def test_init_weights_sum_to_one(method):
    gmm = GaussianMixture(n_components=3, init_weights=method)
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert torch.allclose(gmm.initial_weights_.sum(), torch.tensor(1.0), atol=1e-5)


def test_init_weights_tensor():
    weights = torch.tensor([0.5, 0.3, 0.2])
    gmm = GaussianMixture(n_components=3, init_weights=weights)
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert torch.allclose(gmm.initial_weights_.cpu(), weights.cpu(), atol=1e-5)


# ============================================================================
# 4. Covariance initialization methods
# ============================================================================

@pytest.mark.parametrize("method", ["empirical", "eye", "random", "global"])
def test_init_covariances_methods(method):
    gmm = GaussianMixture(n_components=3, init_covariances=method)
    gmm.fit(X_3D)
    assert gmm.fitted_


def test_init_covariances_tensor_full():
    covs = torch.stack([torch.eye(3) for _ in range(3)])
    gmm = GaussianMixture(n_components=3, covariance_type="full", init_covariances=covs)
    gmm.fit(X_3D)
    assert gmm.fitted_


def test_init_covariances_tensor_diag():
    covs = torch.ones(3, 3)
    gmm = GaussianMixture(n_components=3, covariance_type="diag", init_covariances=covs)
    gmm.fit(X_3D)
    assert gmm.fitted_


# ============================================================================
# 5. Initialization combinations
# ============================================================================

def test_all_empirical_initialization():
    gmm = GaussianMixture(
        n_components=3,
        init_means="kmeans",
        init_weights="kmeans",
        init_covariances="empirical",
    )
    gmm.fit(X_3D)
    assert gmm.fitted_


def test_all_random_initialization():
    gmm = GaussianMixture(
        n_components=3,
        init_means="random",
        init_weights="random",
        init_covariances="random",
    )
    gmm.fit(X_3D)
    assert gmm.fitted_


def test_mixed_string_and_tensor_initialization():
    means = torch.randn(3, 3)
    gmm = GaussianMixture(
        n_components=3,
        init_means=means,
        init_weights="uniform",
        init_covariances="global",
    )
    gmm.fit(X_3D)
    assert gmm.fitted_


def test_all_explicit_tensor_initialization():
    means = torch.randn(3, 3)
    weights = torch.tensor([0.4, 0.4, 0.2])
    covs = torch.stack([torch.eye(3) * 1.5 for _ in range(3)])
    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        init_means=means,
        init_weights=weights,
        init_covariances=covs,
    )
    gmm.fit(X_3D)
    assert gmm.fitted_


# ============================================================================
# 6. Training parameters
# ============================================================================

def test_max_iter_is_respected():
    gmm = GaussianMixture(n_components=3, max_iter=5)
    gmm.fit(X_3D)
    assert gmm.n_iter_ <= 5


def test_tol_parameter():
    gmm = GaussianMixture(n_components=3, tol=1e-2)
    gmm.fit(X_3D)
    assert gmm.fitted_


def test_reg_covar_parameter():
    gmm = GaussianMixture(n_components=3, reg_covar=1e-4)
    gmm.fit(X_3D)
    assert gmm.fitted_


def test_random_state_reproducibility():
    gmm1 = GaussianMixture(n_components=3, random_state=42)
    gmm1.fit(X_3D)
    gmm2 = GaussianMixture(n_components=3, random_state=42)
    gmm2.fit(X_3D)
    assert torch.allclose(gmm1.means_, gmm2.means_, atol=1e-4)


def test_verbose_parameter():
    gmm = GaussianMixture(n_components=3, verbose=True, verbose_interval=5)
    gmm.fit(X_3D)
    assert gmm.fitted_


def test_warm_start_parameter():
    gmm = GaussianMixture(n_components=3, warm_start=True, max_iter=5)
    gmm.fit(X_3D)
    gmm.fit(X_3D)  # Should start from previous solution
    assert gmm.fitted_


def test_n_init_parameter():
    gmm = GaussianMixture(n_components=3, n_init=3, random_state=42)
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert gmm.best_random_state_ is not None


# ============================================================================
# 7. Classification EM (CEM)
# ============================================================================

def test_cem_algorithm():
    gmm = GaussianMixture(n_components=3, cem=True)
    gmm.fit(X_3D)
    assert gmm.fitted_


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical", "tied_full"])
def test_cem_with_covariance_types(covariance_type):
    gmm = GaussianMixture(n_components=3, cem=True, covariance_type=covariance_type)
    gmm.fit(X_3D)
    assert gmm.fitted_


# ============================================================================
# 8. Priors (MAP estimation)
# ============================================================================

def test_weight_prior():
    prior = torch.ones(3) * 2.0
    gmm = GaussianMixture(n_components=3, weight_concentration_prior=prior)
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert gmm.use_weight_prior


def test_mean_prior():
    mean_prior = torch.zeros(3)  # (n_features,) - will be broadcast
    gmm = GaussianMixture(
        n_components=3,
        n_features=3,
        mean_prior=mean_prior,
        mean_precision_prior=0.1,
    )
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert gmm.use_mean_prior


def test_covariance_prior():
    cov_prior = torch.eye(3)
    gmm = GaussianMixture(
        n_components=3,
        n_features=3,
        covariance_type="full",
        covariance_prior=cov_prior,
        degrees_of_freedom_prior=5.0,
    )
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert gmm.use_covariance_prior


def test_all_priors_combined():
    gmm = GaussianMixture(
        n_components=3,
        n_features=3,
        covariance_type="full",
        weight_concentration_prior=torch.ones(3) * 2.0,
        mean_prior=torch.zeros(3),
        mean_precision_prior=0.1,
        covariance_prior=torch.eye(3),
        degrees_of_freedom_prior=5.0,
    )
    gmm.fit(X_3D)
    assert gmm.fitted_


# ============================================================================
# 9. Model methods (predict / score / sample)
# ============================================================================

@pytest.fixture(scope="module")
def fitted_gmm():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_3D)
    return gmm


def test_predict(fitted_gmm):
    labels = fitted_gmm.predict(X_3D)
    assert labels.shape == (X_3D.shape[0],)
    assert labels.min() >= 0
    assert labels.max() < 3


def test_predict_proba(fitted_gmm):
    X_test = X_3D.to(fitted_gmm.device)
    probs = fitted_gmm.predict_proba(X_test)
    assert probs.shape == (X_test.shape[0], 3)
    expected = torch.ones(X_test.shape[0], device=fitted_gmm.device)
    assert torch.allclose(probs.sum(dim=1), expected, atol=1e-5)


def test_score(fitted_gmm):
    score = fitted_gmm.score(X_3D)
    assert isinstance(score, float)


def test_score_samples(fitted_gmm):
    scores = fitted_gmm.score_samples(X_3D)
    assert scores.shape == (X_3D.shape[0],)


def test_sample(fitted_gmm):
    samples, labels = fitted_gmm.sample(100)
    assert samples.shape == (100, 3)
    assert labels.shape == (100,)


# ============================================================================
# 10. Save / load
# ============================================================================

def test_save_and_load_model():
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_3D)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
        filepath = f.name

    try:
        gmm.save(filepath)
        assert os.path.exists(filepath)

        gmm_loaded = GaussianMixture.load(filepath)
        assert gmm_loaded.fitted_
        assert torch.allclose(gmm_loaded.means_, gmm.means_, atol=1e-5)
        assert torch.allclose(gmm_loaded.weights_, gmm.weights_, atol=1e-5)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_to_dict_and_load_state_dict():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_3D)

    state = gmm.to_dict()
    assert isinstance(state, dict)
    assert "means_" in state
    assert "weights_" in state

    gmm2 = GaussianMixture(n_components=3)
    gmm2.load_state_dict(state)
    assert torch.allclose(gmm2.means_, gmm.means_, atol=1e-5)


# ============================================================================
# 11. Different data dimensions
# ============================================================================

@pytest.mark.parametrize("X,n_components,expected_shape", [
    (X_2D, 3, (3, 2)),
    (X_3D, 4, (4, 3)),
    (X_5D, 5, (5, 5)),
    (X_10D, 3, (3, 10)),
])
def test_different_data_dimensions(X, n_components, expected_shape):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X)
    assert gmm.fitted_
    assert gmm.means_.shape == expected_shape


def test_1d_data():
    X_1d = torch.randn(100, 1)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X_1d)
    assert gmm.fitted_
    assert gmm.means_.shape == (2, 1)


# ============================================================================
# 12. Edge cases
# ============================================================================

def test_single_component():
    gmm = GaussianMixture(n_components=1)
    gmm.fit(X_3D)
    assert gmm.fitted_


def test_many_components():
    gmm = GaussianMixture(n_components=10)
    gmm.fit(X_10D)
    assert gmm.fitted_


def test_small_dataset():
    X_small = torch.randn(20, 2)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X_small)
    assert gmm.fitted_


def test_explicit_n_features():
    gmm = GaussianMixture(n_components=3, n_features=3)
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert gmm.n_features == 3


def test_device_cpu():
    gmm = GaussianMixture(n_components=3, device="cpu")
    gmm.fit(X_3D)
    assert gmm.fitted_
    assert gmm.device.type == "cpu"


@pytest.mark.parametrize("reg_covar", [1e-10, 1.0])
def test_reg_covar_extremes(reg_covar):
    gmm = GaussianMixture(n_components=3, reg_covar=reg_covar)
    gmm.fit(X_3D)
    assert gmm.fitted_


# ============================================================================
# 13. Error handling
# ============================================================================

def test_deprecated_init_params_raises():
    with pytest.raises(TypeError, match="init_params.*init_means"):
        GaussianMixture(n_components=3, init_params="kmeans")


def test_deprecated_cov_init_method_raises():
    with pytest.raises(TypeError, match="cov_init_method.*init_covariances"):
        GaussianMixture(n_components=3, cov_init_method="empirical")


def test_deprecated_weights_init_raises():
    with pytest.raises(TypeError, match="weights_init.*init_weights"):
        GaussianMixture(n_components=3, weights_init=torch.ones(3))


def test_invalid_init_means_method_raises():
    with pytest.raises(ValueError, match="invalid_method"):
        gmm = GaussianMixture(n_components=3, init_means="invalid_method")
        gmm.fit(X_3D)


def test_wrong_means_tensor_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        gmm = GaussianMixture(n_components=3, init_means=torch.randn(2, 3))
        gmm.fit(X_3D)


def test_wrong_weights_tensor_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        gmm = GaussianMixture(n_components=3, init_weights=torch.randn(2))
        gmm.fit(X_3D)


def test_invalid_covariance_type_raises():
    with pytest.raises(Exception):
        gmm = GaussianMixture(n_components=3, covariance_type="invalid")
        gmm.fit(X_3D)


# ============================================================================
# Gibbs mode (n_components=None) construction
# ============================================================================

def test_gibbs_mode_requires_all_four_priors():
    with pytest.raises(ValueError, match="mean_prior"):
        GaussianMixture(n_components=None, mean_precision_prior=0.1,
                         covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)


def test_gibbs_mode_requires_mean_precision_prior():
    with pytest.raises(ValueError, match="mean_precision_prior"):
        GaussianMixture(n_components=None, mean_prior=torch.zeros(2),
                         covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)


def test_gibbs_mode_requires_covariance_prior():
    with pytest.raises(ValueError, match="covariance_prior"):
        GaussianMixture(n_components=None, mean_prior=torch.zeros(2),
                         mean_precision_prior=0.1, degrees_of_freedom_prior=4.0)


def test_gibbs_mode_requires_degrees_of_freedom_prior():
    with pytest.raises(ValueError, match="degrees_of_freedom_prior"):
        GaussianMixture(n_components=None, mean_prior=torch.zeros(2),
                         mean_precision_prior=0.1, covariance_prior=torch.eye(2))


def test_gibbs_mode_error_message_lists_all_missing_priors():
    with pytest.raises(ValueError, match="mean_prior.*mean_precision_prior.*covariance_prior.*degrees_of_freedom_prior"):
        GaussianMixture(n_components=None)


def test_gibbs_mode_with_all_priors_constructs_successfully():
    model = GaussianMixture(n_components=None, mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                             covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
    assert model.n_components is None
    assert model.max_components == 20  # default


@pytest.mark.parametrize("covariance_type", ["tied_full", "tied_diag", "tied_spherical"])
def test_gibbs_mode_rejects_tied_covariance_at_construction(covariance_type):
    # Must raise at __init__, before any prior is even checked -- so no
    # prior args are needed to trigger this.
    with pytest.raises(NotImplementedError, match="tied"):
        GaussianMixture(n_components=None, covariance_type=covariance_type)


def test_em_mode_unaffected_by_gibbs_validation():
    # n_components=K (EM mode) must still work with zero priors -- plain MLE,
    # exactly as today. This is the regression guard for "no behavior change."
    model = GaussianMixture(n_components=3)
    assert model.n_components == 3
    assert model.use_mean_prior is False
    assert model.use_covariance_prior is False


def test_em_mode_silently_accepts_gibbs_only_params():
    # Gibbs-only params (max_components, alpha, burn_in, weight_threshold,
    # init_k) must not raise when passed alongside a fixed n_components --
    # they're simply unused by the EM path, not an error condition.
    model = GaussianMixture(n_components=3, max_components=99, alpha=2.0, burn_in=5,
                             weight_threshold=0.5, init_k=10)
    model.fit(X_3D)
    assert model.fitted_
    assert model.n_components_ == 3


def test_gibbs_mode_silently_accepts_em_only_params():
    # EM-only params (n_init, warm_start) must not raise when passed
    # alongside n_components=None -- simply unused by the Gibbs path.
    model = GaussianMixture(n_components=None, max_components=4, n_init=5, warm_start=True,
                             mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                             covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
    assert model.n_components is None


def test_gibbs_mode_new_params_stored():
    model = GaussianMixture(n_components=None, max_components=15, alpha=2.0, burn_in=5,
                             weight_threshold=0.5, init_k=10,
                             mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                             covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
    assert model.max_components == 15
    assert model.alpha == 2.0
    assert model.burn_in == 5
    assert model.weight_threshold == 0.5
    assert model.init_k == 10


def test_gibbs_mode_sets_weight_concentration_prior_to_none():
    # Gibbs mode has no weight-concentration-prior concept (no
    # Dirichlet/stick-breaking weight parameter the way EM's variational
    # path does), but the attribute must still exist post-construction --
    # every other mode sets it (to a tensor or None) via _init_priors, and
    # code like save() reads it unconditionally via __dict__.
    model = GaussianMixture(n_components=None, mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                             covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
    assert model.weight_concentration_prior is None


def test_gibbs_mode_rejects_non_positive_mean_precision_prior():
    # Gibbs-mode eager validation checks prior *presence* (see
    # test_gibbs_mode_requires_mean_precision_prior) but must also check
    # *validity*, matching EM mode's _init_priors path which raises for
    # mean_precision_prior <= 0.
    with pytest.raises(ValueError, match="mean_precision_prior"):
        GaussianMixture(n_components=None, mean_prior=torch.zeros(2), mean_precision_prior=-1.0,
                         covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)


def test_gibbs_mode_use_weight_prior_flag_matches_none_value():
    # use_weight_prior is a flag other code trusts to mean
    # "weight_concentration_prior is a real, usable tensor." Gibbs mode
    # has no weight-concentration-prior concept at all (weights there are
    # controlled via alpha, not a Dirichlet concentration prior), so even
    # when the caller passes a real tensor, it must be silently ignored
    # -- same treatment as other EM-only args (n_init, warm_start) -- and
    # the flag must agree with the (always-None) stored value rather than
    # staying True from the raw constructor argument.
    model = GaussianMixture(n_components=None, weight_concentration_prior=torch.ones(3),
                             mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                             covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
    assert model.weight_concentration_prior is None
    assert model.use_weight_prior is False


# ============================================================================
# suggest_priors
# ============================================================================

def test_suggest_priors_returns_four_element_tuple():
    X = torch.randn(100, 3, dtype=torch.float64)
    result = GaussianMixture.suggest_priors(X, n_components=4)
    assert len(result) == 4
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = result
    assert mean_prior.shape == (3,)
    assert isinstance(mean_precision_prior, float)
    assert covariance_prior.shape == (3, 3)  # covariance_type='full' default
    assert degrees_of_freedom_prior == pytest.approx(5.0)  # n_features + 2


def test_suggest_priors_covariance_type_diag_shape():
    X = torch.randn(100, 3, dtype=torch.float64)
    _, _, covariance_prior, _ = GaussianMixture.suggest_priors(X, n_components=4, covariance_type="diag")
    assert covariance_prior.shape == (3,)


def test_suggest_priors_covariance_type_spherical_shape():
    X = torch.randn(100, 3, dtype=torch.float64)
    _, _, covariance_prior, _ = GaussianMixture.suggest_priors(X, n_components=4, covariance_type="spherical")
    assert covariance_prior.shape == ()


def test_suggest_priors_result_is_directly_usable():
    # The whole point: pass suggest_priors' output straight into the
    # constructor without any reshaping.
    torch.manual_seed(0)
    X = torch.cat([torch.randn(50, 2, dtype=torch.float64) - 5.0,
                    torch.randn(50, 2, dtype=torch.float64) + 5.0])
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=2)
    model = GaussianMixture(n_components=None, max_components=5, mean_prior=mean_prior,
                             mean_precision_prior=mean_precision_prior, covariance_prior=covariance_prior,
                             degrees_of_freedom_prior=degrees_of_freedom_prior, random_state=0)
    assert model.fitted_ is False  # constructed but not yet fit -- this test only checks construction


# ============================================================================
# 14. Stress tests
# ============================================================================

@pytest.mark.parametrize("covariance_type", [
    "full", "diag", "spherical", "tied_full", "tied_diag", "tied_spherical",
])
@pytest.mark.parametrize("mean_init", ["kmeans", "kpp", "random"])
@pytest.mark.parametrize("weight_init", ["uniform", "random", "kmeans"])
@pytest.mark.parametrize("cov_init", ["empirical", "eye", "global"])
def test_all_covariance_and_init_combinations(covariance_type, mean_init, weight_init, cov_init):
    gmm = GaussianMixture(
        n_components=3,
        covariance_type=covariance_type,
        init_means=mean_init,
        init_weights=weight_init,
        init_covariances=cov_init,
        max_iter=10,
    )
    gmm.fit(X_3D)
    assert gmm.fitted_


@pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
def test_multiple_random_seeds(seed):
    gmm = GaussianMixture(n_components=3, random_state=seed, max_iter=10)
    gmm.fit(X_3D)
    assert gmm.fitted_


@pytest.mark.parametrize("n_init", [1, 2, 5, 10])
def test_different_n_init_values(n_init):
    gmm = GaussianMixture(n_components=3, n_init=n_init, random_state=42, max_iter=10)
    gmm.fit(X_3D)
    assert gmm.fitted_


# ============================================================================
# 15. Supervised fitting (labels)
# ============================================================================

def generate_labeled_test_data(n_per_class=60, n_features=3, n_classes=3, seed=7):
    """Generate synthetic per-class clustered data with known labels."""
    generator = torch.Generator().manual_seed(seed)
    X_list, y_list = [], []
    for c in range(n_classes):
        mean = torch.randn(n_features, generator=generator) * 5
        cov = torch.eye(n_features) + torch.randn(n_features, n_features, generator=generator) * 0.1
        cov = cov @ cov.T  # positive definite
        X_c = torch.randn(n_per_class, n_features, generator=generator) @ cov.T + mean
        X_list.append(X_c)
        y_list.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)


X_SUP, Y_SUP = generate_labeled_test_data()


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical", "tied_full"])
def test_supervised_fit_recovers_per_class_mean(covariance_type):
    gmm = GaussianMixture(n_components=3, covariance_type=covariance_type)
    gmm.fit(X_SUP, labels=Y_SUP)
    assert gmm.fitted_
    for c in range(3):
        expected_mean = X_SUP[Y_SUP == c].mean(dim=0)
        assert torch.allclose(gmm.means_[c].cpu(), expected_mean, atol=1e-4)


def test_supervised_fit_recovers_per_class_full_covariance():
    gmm = GaussianMixture(n_components=3, covariance_type="full", reg_covar=0.0)
    gmm.fit(X_SUP, labels=Y_SUP)
    for c in range(3):
        X_c = X_SUP[Y_SUP == c]
        expected_cov = torch.cov(X_c.T, correction=0)
        assert torch.allclose(gmm.covariances_[c].cpu(), expected_cov, atol=1e-4)


def test_supervised_fit_recovers_class_weights():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_SUP, labels=Y_SUP)
    expected_weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    assert torch.allclose(gmm.weights_.cpu(), expected_weights, atol=1e-4)


def test_supervised_fit_converges_in_one_iteration():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_SUP, labels=Y_SUP)
    assert gmm.converged_
    assert gmm.n_iter_ == 1


def test_supervised_fit_predict_proba_sums_to_one():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_SUP, labels=Y_SUP)
    probs = gmm.predict_proba(X_SUP)
    assert torch.allclose(probs.sum(dim=1).cpu(), torch.ones(X_SUP.size(0)), atol=1e-5)


def test_supervised_fit_noncontiguous_labels_round_trip():
    y_noncontig = Y_SUP * 3 + 2  # {0, 1, 2} -> {2, 5, 8}
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_SUP, labels=y_noncontig)
    assert torch.equal(gmm.classes_.cpu(), torch.tensor([2, 5, 8]))
    preds = gmm.predict(X_SUP)
    mapped = gmm.classes_.cpu()[preds.cpu()]
    # Well-separated synthetic classes: predictions should match original labels almost always.
    assert (mapped == y_noncontig).float().mean() > 0.95


def test_supervised_fit_wrong_n_components_raises():
    gmm = GaussianMixture(n_components=2)  # only 2 slots for 3 distinct labels
    with pytest.raises(ValueError, match="n_components"):
        gmm.fit(X_SUP, labels=Y_SUP)


def test_supervised_fit_wrong_labels_length_raises():
    gmm = GaussianMixture(n_components=3)
    with pytest.raises(ValueError, match="n_samples"):
        gmm.fit(X_SUP, labels=Y_SUP[:-1])


def test_supervised_fit_with_mean_prior_shrinks_estimate():
    """A mean prior pulled toward zero should measurably shrink the supervised
    per-class mean estimate compared to the unregularized MLE, confirming
    _m_step's prior handling is actually exercised by the supervised path."""
    gmm_mle = GaussianMixture(n_components=3)
    gmm_mle.fit(X_SUP, labels=Y_SUP)

    gmm_map = GaussianMixture(
        n_components=3,
        n_features=3,
        mean_prior=torch.zeros(3),
        mean_precision_prior=5.0,
    )
    gmm_map.fit(X_SUP, labels=Y_SUP)

    mle_norm = gmm_mle.means_.norm(dim=1)
    map_norm = gmm_map.means_.norm(dim=1)
    assert torch.all(map_norm < mle_norm)


def test_unsupervised_fit_unaffected_by_labels_param_default():
    """labels=None (the default) must be unchanged from before this feature existed."""
    gmm_a = GaussianMixture(n_components=3, random_state=0)
    gmm_a.fit(X_3D)
    gmm_b = GaussianMixture(n_components=3, random_state=0)
    gmm_b.fit(X_3D, labels=None)
    assert torch.allclose(gmm_a.means_.cpu(), gmm_b.means_.cpu())
    assert torch.allclose(gmm_a.weights_.cpu(), gmm_b.weights_.cpu())


def test_supervised_fit_then_unsupervised_fit_clears_classes_():
    """`classes_` is documented as set only after a supervised fit ("absent
    otherwise"). A later unsupervised fit on the same instance must not
    leave a stale label mapping from a previous supervised fit lying
    around -- predict()/classes_ lookups would silently mis-map otherwise."""
    gmm = GaussianMixture(n_components=3, random_state=0, max_iter=5)
    gmm.fit(X_SUP, labels=Y_SUP)
    assert hasattr(gmm, 'classes_')

    gmm.fit(X_SUP)  # unsupervised refit (labels=None) on the same instance
    assert not hasattr(gmm, 'classes_')


# ============================================================================
# 16. Supervised fitting: covariance-degeneracy warning (Fix 2)
# ============================================================================

def _make_sparse_class_data(counts, n_features=3, seed=11):
    """Synthetic labeled data where class `c` gets `counts[c]` samples."""
    generator = torch.Generator().manual_seed(seed)
    X_list, y_list = [], []
    for c, n in enumerate(counts):
        mean = torch.randn(n_features, generator=generator) * 5
        X_c = torch.randn(n, n_features, generator=generator) + mean
        X_list.append(X_c)
        y_list.append(torch.full((n,), c, dtype=torch.long))
    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)


def test_supervised_fit_singleton_class_warns_full_covariance():
    """A class with a single sample can't support a 'full' covariance
    estimate from its own data alone -- this should now warn (previously
    the weight-based check could never fire for a realistic dataset)."""
    X, y = _make_sparse_class_data([1, 49, 50], n_features=3)
    gmm = GaussianMixture(n_components=3, covariance_type='full')
    with pytest.warns(UserWarning, match="too few samples"):
        gmm.fit(X, labels=y)


def test_supervised_fit_two_sample_class_warns_full_covariance_3d():
    """2 samples in 3D give a rank-deficient scatter matrix for 'full'
    covariance -- the reviewer's repro case (eigenvalues like
    [1e-6, 1e-6, 0.62] and a spuriously high score_samples)."""
    X, y = _make_sparse_class_data([2, 50, 50], n_features=3)
    gmm = GaussianMixture(n_components=3, covariance_type='full')
    with pytest.warns(UserWarning, match="too few samples"):
        gmm.fit(X, labels=y)


def test_supervised_fit_well_populated_no_degeneracy_warning(recwarn):
    """The existing 60-samples-per-class fixture is well populated relative
    to n_features=3 for 'full' covariance and must not trigger the
    degeneracy warning."""
    gmm = GaussianMixture(n_components=3, covariance_type='full')
    gmm.fit(X_SUP, labels=Y_SUP)
    degeneracy_warnings = [
        w for w in recwarn.list
        if issubclass(w.category, UserWarning) and "too few samples" in str(w.message)
    ]
    assert not degeneracy_warnings


def test_gibbs_mode_truncated_fit_produces_valid_partition():
    torch.manual_seed(0)
    centers = [[0.0, 0.0], [15.0, 0.0], [7.5, 13.0]]
    covs = [1.5 * torch.eye(2) for _ in range(3)]
    from tgmm.synthetic_data import generate_gmm_data
    X, true_labels = generate_gmm_data(centers, [c.numpy() for c in covs], [30, 30, 30], random_state=0)
    X = X.double()

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6)
    model = GaussianMixture(n_components=None, max_components=6, covariance_type="full",
                             alpha=1.0, max_iter=60, burn_in=20, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    assert model.fitted_
    assert model.weights_.shape == (6,)
    assert model.weights_.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert 2 <= model.n_components_ <= 6

    from sklearn.metrics import adjusted_rand_score
    predicted = model.predict(X)
    assert adjusted_rand_score(true_labels.numpy(), predicted.numpy()) > 0.5


def test_gibbs_mode_unbounded_fit_produces_valid_partition():
    torch.manual_seed(0)
    centers = [[0.0, 0.0], [15.0, 0.0], [7.5, 13.0]]
    covs = [1.5 * torch.eye(2) for _ in range(3)]
    from tgmm.synthetic_data import generate_gmm_data
    X, true_labels = generate_gmm_data(centers, [c.numpy() for c in covs], [25, 25, 25], random_state=0)
    X = X.double()

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6)
    model = GaussianMixture(n_components=None, max_components=None, covariance_type="full",
                             alpha=1.0, max_iter=50, burn_in=15, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    assert model.fitted_
    assert model.max_components is None  # unbounded config untouched by fitting
    assert model.n_components_ == int(model.active_.sum().item())
    assert model.n_components_ >= 1

    from sklearn.metrics import adjusted_rand_score
    predicted = model.predict(X)
    assert adjusted_rand_score(true_labels.numpy(), predicted.numpy()) > 0.3


def test_gibbs_mode_supervised_labels_raises():
    with pytest.raises(ValueError, match="labels"):
        model = GaussianMixture(n_components=None, mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                                 covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
        model.fit(torch.randn(20, 2, dtype=torch.float64), labels=torch.zeros(20, dtype=torch.long))


def test_gibbs_mode_fit_honors_max_iter_override():
    torch.manual_seed(0)
    model = GaussianMixture(n_components=None, max_components=5, max_iter=5,
                             mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                             covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
    model.fit(torch.randn(30, 2, dtype=torch.float64), max_iter=50)
    assert model.n_iter_ == 50


@pytest.mark.parametrize("bad_max_iter", [0, -3])
def test_gibbs_mode_fit_rejects_non_positive_max_iter_override(bad_max_iter):
    torch.manual_seed(0)
    model = GaussianMixture(n_components=None, max_components=5, max_iter=5,
                             mean_prior=torch.zeros(2), mean_precision_prior=0.1,
                             covariance_prior=torch.eye(2), degrees_of_freedom_prior=4.0)
    with pytest.raises(ValueError, match="Invalid max_iter"):
        model.fit(torch.randn(30, 2, dtype=torch.float64), max_iter=bad_max_iter)
    # the rejected override must not leak into instance state
    assert model.max_iter == 5
