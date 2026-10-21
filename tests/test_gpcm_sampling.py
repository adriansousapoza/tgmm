"""sample() for the eight GPCM-only covariance codes (VEI, EVI, VEE, EVE, EEV, VVE, VEV, EVV).

They store dense (K, d, d) covariances like 'full', but three sampling helpers only recognised 'full' and raised
``ValueError: Unsupported covariance type`` (plain sampling, per-component sampling, constrained sampling).
"""
import pytest
import torch

from tgmm import GaussianMixture

NEW_CODES = ['VEI', 'EVI', 'VEE', 'EVE', 'EEV', 'VVE', 'VEV', 'EVV']


def _fit(code):
    g = torch.Generator().manual_seed(0)
    mus = torch.tensor([[0.0, 0.0, 0.0], [6.0, 0.0, 1.0], [0.0, 6.0, -1.0]])
    X = torch.cat([mu + torch.randn(150, 3, generator=g) * torch.tensor([1.0, 0.5, 0.3]) for mu in mus]).double()
    return GaussianMixture(n_components=3, covariance_type=code, random_state=0, max_iter=100).fit(X)


@pytest.mark.parametrize("code", NEW_CODES)
def test_plain_and_component_sampling(code):
    model = _fit(code)
    samples, comps = model.sample(300)
    assert samples.shape == (300, 3) and comps.shape == (300,)
    assert torch.isfinite(samples).all()
    only, _ = model.sample(40, component=1)
    assert only.shape == (40, 3)


@pytest.mark.parametrize("code", NEW_CODES)
def test_constrained_sampling_confidence_and_std_range(code):
    model = _fit(code)
    inside, labels = model.sample(60, confidence=0.9)
    assert inside.shape == (60, 3)
    outliers, _ = model.sample(20, std_range=(3.0, float('inf')))
    assert outliers.shape == (20, 3)


@pytest.mark.parametrize("code", NEW_CODES)
def test_sample_moments_match_the_fitted_covariance(code):
    model = _fit(code)
    k = 0
    draws, _ = model.sample(20000, component=k)
    cov = torch.cov(draws.T.double())
    assert torch.allclose(cov, model.covariances_[k].double(), atol=0.05, rtol=0.1)
    assert torch.allclose(draws.mean(0).double(), model.means_[k].double(), atol=0.05)
