"""tied_spherical MLE keeps the tgmm <= 0.3.0 summation order.

The pooled scatter is one reduction ``einsum('nk,nkd->', resp, (x - mu)^2)`` over samples, components and features.
Summing per-component partial sums instead (0.4.0) differs in the last digit, which iterated fits amplify
(docs/superpowers/specs/2026-10-19-tied-spherical-summation-order.md).
"""
import pytest
import torch

from tgmm import GaussianMixture


def _data(dtype, n=1200, d=3, K=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(K, d, generator=g) * 3
    X = torch.cat([means[k] + torch.randn(n // K, d, generator=g) * (0.5 + 0.3 * k) for k in range(K)])
    return X.to(dtype)


def _one_m_step(model, X):
    model.dtype = X.dtype  # fit() sets this from the data; we bypass fit to isolate one M-step
    model._allocate_parameters(X)
    resp, _ = model._e_step(X)
    stats = model._compute_stats(X, resp)
    model._m_step_from_stats(stats)
    return resp


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_m_step_uses_single_reduction_order(dtype):
    X = _data(dtype)
    model = GaussianMixture(n_components=4, covariance_type='tied_spherical', n_features=3, init_means=X[::300][:4].clone(),
                            random_state=0)
    resp = _one_m_step(model, X)
    nk = resp.sum(dim=0) + 1e-20
    diff = X.unsqueeze(1) - model.means_.unsqueeze(0)
    expected = torch.einsum('nk,nkd->', resp, diff.pow(2)) / (nk.sum() * X.shape[1]) + model.reg_covar
    assert torch.equal(model.covariances_, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_agrees_with_per_component_sum_to_rounding(dtype):
    X = _data(dtype)
    model = GaussianMixture(n_components=4, covariance_type='tied_spherical', n_features=3, init_means=X[::300][:4].clone(),
                            random_state=0)
    resp = _one_m_step(model, X)
    stats = model._compute_stats(X, resp)
    per_component = stats.s2.sum() / ((stats.nk + 1e-20).sum() * X.shape[1]) + model.reg_covar
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(model.covariances_, per_component, rtol=tol, atol=0)


def test_chunked_fit_still_matches_unchunked():
    X = _data(torch.float64)
    kwargs = dict(n_components=4, covariance_type='tied_spherical', init_means=X[::300][:4].clone(), random_state=0,
                  max_iter=30, tol=1e-12)
    a = GaussianMixture(**kwargs).fit(X)
    b = GaussianMixture(batch_size=256, **kwargs).fit(X)
    assert torch.allclose(a.covariances_, b.covariances_, rtol=1e-8)
    assert torch.allclose(a.means_, b.means_, rtol=1e-8, atol=1e-10)


def test_mean_prior_falls_back_to_parallel_axis_path():
    """With a mean prior means_ != ybar, so the covariance must use the shifted scatter."""
    X = _data(torch.float64)
    m = GaussianMixture(n_components=4, covariance_type='tied_spherical', n_features=3, init_means=X[::300][:4].clone(),
                        mean_prior=torch.zeros(4, 3, dtype=torch.float64), mean_precision_prior=5.0,
                        random_state=0, max_iter=5, tol=1e-30)
    m.fit(X)
    assert torch.isfinite(m.covariances_).all() and float(m.covariances_) > 0


def test_stats_aux_is_scaled_and_merged_for_online_updates():
    X = _data(torch.float64)
    m = GaussianMixture(n_components=4, covariance_type='tied_spherical', n_features=3, init_means=X[::300][:4].clone(),
                        batch_size=200, step_size=0.5, max_iter=3, random_state=0)
    m.fit(X)
    assert torch.isfinite(m.covariances_).all()
