"""Regression tests for GaussianMixture.sample() memory blowup (GitHub issue #1).

MultivariateNormal(loc, covariance_matrix=cov) materializes a full
(n_samples, n_features, n_features) tensor internally when `cov` has a batch
dimension of 1 that needs broadcasting to n_samples -- exactly what happened
for every non-'full' covariance type, since their shared/diagonal/spherical
covariance was densified into a full matrix before being handed to
MultivariateNormal. For the reported case (n_components=100, n_features=768,
covariance_type='tied_spherical', sampling 40_000 points) that's ~94GB,
causing GPU OOM.
"""
import pytest
import torch

from tgmm.gmm import GaussianMixture


def test_sample_never_broadcasts_covariance_via_multivariate_normal(monkeypatch):
    """Unconstrained bulk sample() must never go through MultivariateNormal
    with a covariance that needs broadcasting across samples -- that's
    exactly what densifies to (n_samples, d, d) and caused the reported OOM.
    This holds for all covariance types, including 'full', since 'full' is
    now sampled by grouping per component (only n_components distinct
    matrices) rather than gathering into a per-sample batch."""
    import tgmm.gmm as gmm_module

    calls = []
    original_mvn = gmm_module.MultivariateNormal

    class SpyMVN(original_mvn):
        def __init__(self, *args, **kwargs):
            calls.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(gmm_module, "MultivariateNormal", SpyMVN)

    for covariance_type in [
        "full", "diag", "spherical", "tied_full", "tied_diag", "tied_spherical",
    ]:
        calls.clear()
        torch.manual_seed(0)
        gmm = GaussianMixture(n_components=4, covariance_type=covariance_type, max_iter=5)
        X = torch.randn(200, 6)
        gmm.fit(X)
        gmm.sample(500)
        assert not calls, (
            f"{covariance_type} unexpectedly used MultivariateNormal -- "
            f"would densify the covariance to (n_samples, n_features, n_features)"
        )


def test_sample_full_covariance_still_works():
    """'full' genuinely needs a per-sample covariance matrix -- just check it
    still runs correctly. That memory cost is inherent to 'full', not a bug."""
    torch.manual_seed(0)
    gmm = GaussianMixture(n_components=5, covariance_type="full", max_iter=5)
    X = torch.randn(200, 20)
    gmm.fit(X)

    samples, indices = gmm.sample(500)
    assert samples.shape == (500, 20)
    assert indices.shape == (500,)


@pytest.mark.parametrize("covariance_type", [
    "full", "diag", "spherical", "tied_full", "tied_diag", "tied_spherical",
])
def test_sample_statistics_match_component_parameters(covariance_type):
    """The fast-path sampling formulas must be statistically equivalent to
    proper MVN sampling: both the mean AND the covariance of samples drawn
    from a single component must match that component's fitted parameters.
    Uses gmm._build_covariances_for_sampling (still exact/unchanged, since it
    only ever builds a batch-of-1 matrix) as the ground-truth reference
    covariance for the comparison."""
    torch.manual_seed(0)
    gmm = GaussianMixture(n_components=3, covariance_type=covariance_type, max_iter=5)
    X = torch.randn(300, 4)
    gmm.fit(X)

    samples, _ = gmm.sample(50000, component=0)
    empirical_mean = samples.mean(dim=0)
    centered = samples - empirical_mean
    empirical_cov = (centered.T @ centered) / (samples.shape[0] - 1)

    true_cov = gmm._build_covariances_for_sampling(torch.tensor([0]), 1)[0]

    assert torch.allclose(empirical_mean, gmm.means_[0], atol=0.1)
    assert torch.allclose(empirical_cov, true_cov, atol=0.15)


def test_sample_matches_reported_issue_scale():
    """Directly reproduces the exact scenario from GitHub issue #1:
    n_components=100, n_features=768, covariance_type='tied_spherical',
    sampling 40_000 points at once. Must complete quickly on CPU without the
    ~94GB densified-covariance allocation the bug caused."""
    torch.manual_seed(0)
    n_features = 768
    gmm = GaussianMixture(n_components=100, covariance_type="tied_spherical", max_iter=3)
    X = torch.randn(500, n_features)
    gmm.fit(X)

    samples, indices = gmm.sample(40_000)

    assert samples.shape == (40_000, n_features)
    assert indices.shape == (40_000,)
