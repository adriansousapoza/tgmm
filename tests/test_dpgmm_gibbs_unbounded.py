"""Pytest suite for GaussianMixture's unbounded collapsed Gibbs sampler (Neal's Algorithm 3, full covariance)."""
import torch
import pytest
from sklearn.metrics import adjusted_rand_score

from tgmm import GaussianMixture
from tgmm.synthetic_data import generate_gmm_data


def three_blob_data(n_per_cluster=25, seed=0):
    centers = [[0.0, 0.0], [15.0, 0.0], [7.5, 13.0]]
    covs = [1.5 * torch.eye(2).numpy() for _ in range(3)]
    X, labels = generate_gmm_data(centers, covs, [n_per_cluster] * 3, random_state=seed)
    return X.double(), labels


def _dummy_priors(n_features=2):
    return dict(
        mean_prior=torch.zeros(n_features, dtype=torch.float64),
        mean_precision_prior=1.0,
        covariance_prior=torch.eye(n_features, dtype=torch.float64),
        degrees_of_freedom_prior=float(n_features + 2),
    )


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_unbounded_gibbs_fit_produces_valid_partition(covariance_type):
    X, true_labels = three_blob_data()
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6, covariance_type=covariance_type, random_state=0)
    model = GaussianMixture(n_components=None, max_components=None, covariance_type=covariance_type,
                             alpha=1.0, max_iter=50, burn_in=15, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    assert model.fitted_
    assert model.max_components is None  # unbounded config is untouched by fitting
    assert model.n_components_ == int(model.active_.sum().item())
    assert model.n_components_ >= 1
    assert model.weights_.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (model.weights_ > 0).all()  # every surviving component is non-empty by construction

    predicted = model.predict(X)
    assert adjusted_rand_score(true_labels.numpy(), predicted.numpy()) > 0.3


def test_unbounded_gibbs_tracks_component_count_history():
    X, _ = three_blob_data(n_per_cluster=15)
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6, random_state=0)
    model = GaussianMixture(n_components=None, max_components=None, covariance_type="full",
                             alpha=1.0, max_iter=30, burn_in=10, random_state=0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    assert hasattr(model, "n_components_history_")
    assert len(model.n_components_history_) == 30 - 10
    assert all(k >= 1 for k in model.n_components_history_)


def test_unbounded_gibbs_n_components_respects_weight_threshold():
    # A far-flung outlier likely ends up in its own near-singleton component.
    # n_components_ must apply the same weight_threshold rule as the other
    # inference path (_finalize_active_components), not just count every
    # surviving component regardless of how little data supports it.
    X, _ = three_blob_data()
    outlier = torch.tensor([[200.0, 200.0]], dtype=torch.float64)
    X = torch.cat([X, outlier], dim=0)

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=6, random_state=0)
    model = GaussianMixture(n_components=None, max_components=None, covariance_type="full",
                             alpha=1.0, max_iter=50, burn_in=15, random_state=0, weight_threshold=1.0,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    expected_active = model.weights_ * X.shape[0] > model.weight_threshold
    assert torch.equal(model.active_, expected_active)
    assert model.n_components_ == int(expected_active.sum().item())


@pytest.mark.parametrize("covariance_type", ["tied_full", "tied_diag", "tied_spherical"])
def test_unbounded_gibbs_rejects_tied_covariance(covariance_type):
    with pytest.raises(NotImplementedError, match="tied"):
        GaussianMixture(n_components=None, max_components=None, covariance_type=covariance_type)


def test_init_k_overrides_auto_heuristic():
    # _resolve_max_components's automatic seed (max(2, min(20, n // 2)))
    # must be overridden whenever init_k is set, regardless of n_samples --
    # that's the whole point of init_k (giving unbounded a caller-chosen
    # starting K instead of one implicitly coupled to n_samples).
    model = GaussianMixture(n_components=None, max_components=None, init_k=7, **_dummy_priors())
    assert model._resolve_max_components(n_samples=10_000) == 7
    assert model._resolve_max_components(n_samples=3) == 7


def test_init_k_unset_falls_back_to_auto_heuristic():
    model = GaussianMixture(n_components=None, max_components=None, **_dummy_priors())
    assert model._resolve_max_components(n_samples=1000) == 20  # min(20, 1000 // 2)
    assert model._resolve_max_components(n_samples=6) == 3      # min(20, 6 // 2)


def test_unbounded_matches_truncated_given_same_init_k():
    # On well-separated data, truncated (max_components=K) and unbounded
    # (max_components=None, init_k=K) should start from the identical
    # k-means partition and settle on the identical final one for the same
    # random_state -- the scenario notebooks/dpgmm_gibbs_sampling.ipynb's
    # "Truncated and unbounded Gibbs agree" section demonstrates. This is
    # not a universal guarantee on harder (overlapping/non-Gaussian) data,
    # where the two samplers' slightly different per-sweep rules can still
    # drift apart from the same start -- see the design doc.
    X, true_labels = three_blob_data(n_per_cluster=40)
    K_SEED = 10
    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=K_SEED, random_state=0)

    truncated = GaussianMixture(n_components=None, max_components=K_SEED, covariance_type="full",
                                 alpha=1.0, max_iter=50, random_state=0, weight_threshold=1.0,
                                 mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                                 covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    truncated.fit(X)

    unbounded = GaussianMixture(n_components=None, max_components=None, init_k=K_SEED, covariance_type="full",
                                 alpha=1.0, max_iter=50, burn_in=15, random_state=0, weight_threshold=1.0,
                                 mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                                 covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    unbounded.fit(X)

    assert truncated.n_components_ == unbounded.n_components_ == 3
    agreement = adjusted_rand_score(truncated.predict(X).numpy(), unbounded.predict(X).numpy())
    assert agreement == pytest.approx(1.0)
