"""v0.5 item 1.4: ``frozen_components`` and ``fit(X, sample_weight=...)``.

See docs/superpowers/specs/2026-10-15-v05-frozen-sample-weight-design.md.
"""
import os
import tempfile
import warnings

import numpy as np
import pytest
import torch

from tgmm import GaussianMixture, PositiveUnlabeledMixture, StudentTMixture, VonMisesFisherMixture

DT = torch.float64
CLASSIC = ['full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical']
ALGORITHMS = ['em', 'cem', 'sem', 'saem']
PARAMS = ('weights_', 'means_', 'covariances_', 'dof_', 'kappa_', 'noise_weight_')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def blobs(n_per=40, d=2, seed=0, K=3):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[0.0, 4.0, -4.0], [4.0, -4.0, 0.0], [-4.0, 0.0, 4.0]], dtype=DT)[:, :d] if d <= 3 else None
    X, y = [], []
    for k in range(K):
        X.append(torch.randn(n_per, d, generator=g, dtype=DT) * (0.8 + 0.2 * k) + centers[k])
        y.append(torch.full((n_per,), k))
    return torch.cat(X), torch.cat(y)


def sphere_blobs(n_per=40, d=3, seed=0, K=3):
    X, y = blobs(n_per, d, seed, K)
    return X / X.norm(dim=1, keepdim=True), y


def init_means_for(X, K=3):
    idx = torch.linspace(0, X.shape[0] - 1, K).long() if K > 1 else torch.tensor([0])
    return X[idx].clone()


def fit_quiet(model, X, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.fit(X, *args, **kw)


def params(model):
    out = {}
    for name in PARAMS:
        v = getattr(model, name, None)
        if torch.is_tensor(v):
            out[name] = v.detach().clone()
    return out


def assert_same_params(a, b, atol=1e-8, rtol=1e-6):
    pa, pb = params(a), params(b)
    assert pa.keys() == pb.keys()
    for name in pa:
        assert torch.allclose(pa[name], pb[name], atol=atol, rtol=rtol), (
            f"{name} differs: max |diff| = {(pa[name] - pb[name]).abs().max().item():.3e}"
        )


def make(family, X, K=3, cov='full', **kw):
    """A model with fully explicit initialisation, so two fits start from identical parameters."""
    base = dict(n_components=K, n_features=X.shape[1], init_means=init_means_for(X, K), init_weights='uniform',
                max_iter=25, tol=1e-30, random_state=0)
    if family == 'gmm':
        return GaussianMixture(covariance_type=cov, init_covariances='eye', **{**base, **kw})
    if family == 't':
        return StudentTMixture(covariance_type=cov, init_covariances='eye', **{**base, **kw})
    return VonMisesFisherMixture(**{**base, **kw})


def data_for(family, **kw):
    return sphere_blobs(**kw) if family == 'vmf' else blobs(**kw)


def int_weights(n, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 4, (n,), generator=g).to(DT)


# ===========================================================================
# 1. row sample_weight
# ===========================================================================
@pytest.mark.parametrize('cov', CLASSIC)
def test_integer_weights_equal_row_duplication_gaussian(cov):
    X, _ = blobs()
    w = int_weights(X.shape[0])
    X_dup = X.repeat_interleave(w.long(), dim=0)
    a = fit_quiet(make('gmm', X, cov=cov), X, sample_weight=w)
    b = fit_quiet(make('gmm', X, cov=cov), X_dup)
    assert_same_params(a, b)


@pytest.mark.parametrize('cov', ['full', 'diag', 'spherical', 'tied_full'])
def test_integer_weights_equal_row_duplication_student_t(cov):
    X, _ = blobs()
    w = int_weights(X.shape[0])
    X_dup = X.repeat_interleave(w.long(), dim=0)
    a = fit_quiet(make('t', X, cov=cov), X, sample_weight=w)
    b = fit_quiet(make('t', X, cov=cov), X_dup)
    assert_same_params(a, b)


def test_integer_weights_equal_row_duplication_vmf():
    X, _ = sphere_blobs()
    w = int_weights(X.shape[0])
    X_dup = X.repeat_interleave(w.long(), dim=0)
    a = fit_quiet(make('vmf', X), X, sample_weight=w)
    b = fit_quiet(make('vmf', X), X_dup)
    assert_same_params(a, b)


@pytest.mark.parametrize('family', ['gmm', 't', 'vmf'])
def test_zero_weight_rows_are_ignored(family):
    X, _ = data_for(family)
    g = torch.Generator().manual_seed(5)
    w = (torch.rand(X.shape[0], generator=g) > 0.3).to(DT)
    keep = w > 0
    a = fit_quiet(make(family, X), X, sample_weight=w)
    b = fit_quiet(make(family, X), X[keep])
    assert_same_params(a, b)


@pytest.mark.parametrize('family', ['gmm', 't', 'vmf'])
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_constant_weight_is_a_no_op_for_every_algorithm(family, algorithm):
    X, _ = data_for(family)
    kw = dict(algorithm=algorithm, max_iter=12)
    a = fit_quiet(make(family, X, **kw), X)
    b = fit_quiet(make(family, X, **kw), X, sample_weight=torch.full((X.shape[0],), 3.0, dtype=DT))
    assert_same_params(a, b, atol=1e-7)


@pytest.mark.parametrize('family', ['gmm', 't', 'vmf'])
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_weighted_fit_runs_and_is_finite(family, algorithm):
    X, _ = data_for(family, n_per=60)
    w = int_weights(X.shape[0])
    m = fit_quiet(make(family, X, algorithm=algorithm, max_iter=15), X, sample_weight=w)
    for name, v in params(m).items():
        assert torch.isfinite(v).all(), name
    assert abs(m.weights_.sum().item() - 1.0) < 1e-6
    assert np.isfinite(m.lower_bound_)


def test_all_ones_weights_are_bit_identical_to_none():
    X, _ = blobs()
    a = fit_quiet(make('gmm', X), X)
    b = fit_quiet(make('gmm', X), X, sample_weight=torch.ones(X.shape[0], dtype=DT))
    for name, v in params(a).items():
        assert torch.equal(v, getattr(b, name))
    assert a.lower_bound_ == b.lower_bound_


def test_weights_with_map_priors_equal_duplication():
    X, _ = blobs()
    w = int_weights(X.shape[0])
    X_dup = X.repeat_interleave(w.long(), dim=0)
    prior = dict(
        weight_concentration_prior=torch.full((3,), 2.0, dtype=DT),
        mean_prior=torch.zeros(3, 2, dtype=DT), mean_precision_prior=0.5,
        covariance_prior=torch.eye(2, dtype=DT).repeat(3, 1, 1), degrees_of_freedom_prior=4.0,
    )
    a = fit_quiet(make('gmm', X, **prior), X, sample_weight=w)
    b = fit_quiet(make('gmm', X, **prior), X_dup)
    assert_same_params(a, b)


def test_weights_with_noise_equal_duplication():
    X, _ = blobs()
    noise = (torch.rand(15, 2, generator=torch.Generator().manual_seed(9), dtype=DT) - 0.5) * 24
    X = torch.cat([X, noise])
    w = int_weights(X.shape[0])
    X_dup = X.repeat_interleave(w.long(), dim=0)
    kw = dict(noise=True, noise_volume=24.0 * 24.0)
    a = fit_quiet(make('gmm', X, **kw), X, sample_weight=w)
    b = fit_quiet(make('gmm', X, **kw), X_dup)
    assert_same_params(a, b)
    assert abs(a.weights_.sum().item() + a.noise_weight_.item() - 1.0) < 1e-6


def test_weights_with_missing_values_equal_duplication():
    X, _ = blobs()
    g = torch.Generator().manual_seed(2)
    mask = torch.rand(X.shape, generator=g) < 0.15
    mask[mask.all(dim=1)] = False
    Xn = X.clone()
    Xn[mask] = float('nan')
    w = int_weights(X.shape[0])
    Xn_dup = Xn.repeat_interleave(w.long(), dim=0)
    a = fit_quiet(make('gmm', X, missing='em'), Xn, sample_weight=w)
    b = fit_quiet(make('gmm', X, missing='em'), Xn_dup)
    assert_same_params(a, b)


@pytest.mark.parametrize('family,cov', [('gmm', 'full'), ('gmm', 'diag'), ('t', 'full'), ('vmf', None)])
def test_batch_size_chunked_equals_unchunked_with_weights(family, cov):
    X, _ = data_for(family, n_per=50)
    w = int_weights(X.shape[0])
    kw = {} if cov is None else {'cov': cov}
    a = fit_quiet(make(family, X, **kw), X, sample_weight=w)
    b = fit_quiet(make(family, X, batch_size=37, **kw), X, sample_weight=w)
    assert_same_params(a, b, atol=1e-8)


def test_n_init_and_warm_start_with_weights():
    X, _ = blobs()
    w = int_weights(X.shape[0])
    m = GaussianMixture(n_components=3, n_init=2, random_state=0, max_iter=30)
    fit_quiet(m, X, sample_weight=w)
    assert m.fitted_ and torch.isfinite(m.means_).all()
    lb = m.lower_bound_
    m2 = GaussianMixture(n_components=3, warm_start=True, random_state=0, max_iter=5)
    fit_quiet(m2, X, sample_weight=w)
    fit_quiet(m2, X, sample_weight=w)
    assert np.isfinite(m2.lower_bound_) and m2.lower_bound_ >= lb - 5.0


def test_weighted_lower_bound_is_the_weighted_mean_log_likelihood():
    X, _ = blobs()
    w = int_weights(X.shape[0])
    m = fit_quiet(make('gmm', X, max_iter=60, tol=1e-9), X, sample_weight=w)
    expected = (w * m.score_samples(X)).sum() / w.sum()
    assert m.lower_bound_ == pytest.approx(expected.item(), rel=1e-6)


@pytest.mark.parametrize('bad', [
    torch.tensor([-1.0] + [1.0] * 119), torch.tensor([float('nan')] + [1.0] * 119), torch.zeros(120),
    torch.ones(100), torch.ones(120, 2, 2),
])
def test_sample_weight_validation(bad):
    X, _ = blobs()
    with pytest.raises(ValueError):
        GaussianMixture(n_components=3).fit(X, sample_weight=bad.to(DT))


def test_sample_weight_unsupported_combinations():
    X, y = blobs()
    w = torch.ones(X.shape[0], dtype=DT) * 2
    priors = dict(zip(('mean_prior', 'mean_precision_prior', 'covariance_prior', 'degrees_of_freedom_prior'),
                      GaussianMixture.suggest_priors(X, n_components=3)))
    with pytest.raises(NotImplementedError, match='Gibbs'):
        GaussianMixture(n_components=None, **priors).fit(X, sample_weight=w)
    with pytest.raises(NotImplementedError):
        GaussianMixture(n_components=3).fit(X, labels=y, sample_weight=w)
    with pytest.raises(NotImplementedError):
        GaussianMixture(n_components=3, step_size='auto', batch_size=32).fit(X, sample_weight=w)


# ===========================================================================
# 2. per-feature weights
# ===========================================================================
def ref_gaussian(X, W, means, var, pi, iters, spherical, reg=1e-6):
    X, W = X.numpy(), W.numpy()
    means, var, pi = means.numpy().copy(), var.numpy().copy(), pi.numpy().copy()
    n, d = X.shape
    K = len(pi)
    for _ in range(iters):
        logp = np.empty((n, K))
        for k in range(K):
            diff2 = (X - means[k]) ** 2
            if spherical:
                logp[:, k] = -0.5 * (W.sum(1) * (np.log(2 * np.pi) + np.log(var[k])) + (W * diff2).sum(1) / var[k])
            else:
                logp[:, k] = -0.5 * (W * (np.log(2 * np.pi) + np.log(var[k]) + diff2 / var[k])).sum(1)
        logp += np.log(pi)
        r = np.exp(logp - np.logaddexp.reduce(logp, axis=1, keepdims=True))
        pi = r.sum(0) / n
        N = r.T @ W
        means = (r.T @ (W * X)) / N
        for k in range(K):
            s = (r[:, k:k + 1] * W * (X - means[k]) ** 2).sum(0)
            var[k] = (s.sum() / N[k].sum() + reg) if spherical else (s / N[k] + reg)
    return pi, means, var


def ref_student_diag(X, W, means, var, pi, nu, iters, reg=1e-6):
    X, W = X.numpy(), W.numpy()
    means, var, pi = means.numpy().copy(), var.numpy().copy(), pi.numpy().copy()
    n, d = X.shape
    K = len(pi)
    p = W.sum(1)
    from scipy.special import gammaln
    for _ in range(iters):
        logp = np.empty((n, K))
        u = np.empty((n, K))
        for k in range(K):
            delta = (W * (X - means[k]) ** 2 / var[k]).sum(1)
            logp[:, k] = (gammaln((nu + p) / 2) - gammaln(nu / 2) - 0.5 * p * np.log(nu * np.pi)
                          - 0.5 * (W * np.log(var[k])).sum(1) - 0.5 * (nu + p) * np.log1p(delta / nu))
            u[:, k] = (nu + p) / (nu + delta)
        logp += np.log(pi)
        r = np.exp(logp - np.logaddexp.reduce(logp, axis=1, keepdims=True))
        pi = r.sum(0) / n
        for k in range(K):
            ruw = r[:, k:k + 1] * u[:, k:k + 1] * W
            means[k] = (ruw * X).sum(0) / ruw.sum(0)
            var[k] = (ruw * (X - means[k]) ** 2).sum(0) / (r[:, k:k + 1] * W).sum(0) + reg
    return pi, means, var


def feature_weights(n, d, seed=4, zero_frac=0.25):
    g = torch.Generator().manual_seed(seed)
    W = torch.rand(n, d, generator=g, dtype=DT) * 1.5 + 0.25
    zero = torch.rand(n, d, generator=g) < zero_frac
    zero[zero.all(dim=1)] = False
    W[zero] = 0.0
    return W


@pytest.mark.parametrize('cov', ['diag', 'spherical'])
def test_per_feature_weights_match_numpy_reference_gaussian(cov):
    X, _ = blobs(d=3)
    W = feature_weights(*X.shape)
    m = make('gmm', X, cov=cov, max_iter=15)
    mu0 = m.init_means.clone()
    K, d = 3, X.shape[1]
    var0 = torch.full((K, d), 1.0 + 1e-6, dtype=DT) if cov == 'diag' else torch.full((K,), 1.0 + 1e-6, dtype=DT)
    fit_quiet(m, X, sample_weight=W)
    pi, mu, var = ref_gaussian(X, W, mu0, var0, torch.full((K,), 1 / K, dtype=DT), 15, cov == 'spherical')
    assert np.allclose(m.weights_.numpy(), pi, atol=1e-8)
    assert np.allclose(m.means_.numpy(), mu, atol=1e-8)
    assert np.allclose(m.covariances_.numpy(), var, atol=1e-8)


def test_per_feature_weights_match_numpy_reference_student_t_diag():
    X, _ = blobs(d=3)
    W = feature_weights(*X.shape)
    m = make('t', X, cov='diag', dof=4.0, max_iter=15)
    mu0 = m.init_means.clone()
    var0 = torch.full((3, 3), 1.0 + 1e-6, dtype=DT)
    fit_quiet(m, X, sample_weight=W)
    pi, mu, var = ref_student_diag(X, W, mu0, var0, torch.full((3,), 1 / 3, dtype=DT), 4.0, 15)
    assert np.allclose(m.weights_.numpy(), pi, atol=1e-8)
    assert np.allclose(m.means_.numpy(), mu, atol=1e-8)
    assert np.allclose(m.covariances_.numpy(), var, atol=1e-8)


@pytest.mark.parametrize('family,cov', [('gmm', 'diag'), ('gmm', 'spherical'), ('t', 'diag')])
def test_all_ones_per_feature_weights_are_bit_identical_to_none(family, cov):
    X, _ = blobs(d=3)
    a = fit_quiet(make(family, X, cov=cov), X)
    b = fit_quiet(make(family, X, cov=cov), X, sample_weight=torch.ones(X.shape, dtype=DT))
    for name, v in params(a).items():
        assert torch.equal(v, getattr(b, name)), name


@pytest.mark.parametrize('family,cov', [('gmm', 'diag'), ('gmm', 'spherical'), ('t', 'diag')])
def test_per_feature_weights_chunked_equals_unchunked(family, cov):
    X, _ = blobs(d=3, n_per=50)
    W = feature_weights(*X.shape)
    a = fit_quiet(make(family, X, cov=cov), X, sample_weight=W)
    b = fit_quiet(make(family, X, cov=cov, batch_size=41), X, sample_weight=W)
    assert_same_params(a, b)


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_per_feature_weights_all_algorithms(algorithm):
    X, _ = blobs(d=3, n_per=50)
    W = feature_weights(*X.shape)
    m = fit_quiet(make('gmm', X, cov='diag', algorithm=algorithm, max_iter=15), X, sample_weight=W)
    assert all(torch.isfinite(v).all() for v in params(m).values())


def test_zero_feature_weight_equals_a_missing_entry_for_diag_gaussians():
    """With independent features, weight 0 on an entry is exactly marginalisation: the same as missing='em'."""
    X, _ = blobs(d=3)
    g = torch.Generator().manual_seed(8)
    hole = torch.rand(X.shape, generator=g) < 0.2
    hole[hole.all(dim=1)] = False
    W = (~hole).to(DT)
    Xn = X.clone()
    Xn[hole] = float('nan')
    a = fit_quiet(make('gmm', X, cov='diag'), X, sample_weight=W)
    b = fit_quiet(make('gmm', X, cov='diag', missing='em'), Xn)
    assert_same_params(a, b, atol=1e-6)


def test_per_feature_weights_validation():
    X, _ = blobs(d=3)
    W = torch.ones(X.shape, dtype=DT)
    for cov in ['full', 'tied_full', 'tied_diag', 'tied_spherical', 'VEI', 'EVE']:
        with pytest.raises(ValueError, match='per-feature'):
            GaussianMixture(n_components=3, covariance_type=cov).fit(X, sample_weight=W)
    with pytest.raises(ValueError, match='per-feature'):
        StudentTMixture(n_components=3, covariance_type='full').fit(X, sample_weight=W)
    with pytest.raises(ValueError, match='per-feature'):
        VonMisesFisherMixture(n_components=3).fit(sphere_blobs(d=3)[0], sample_weight=W)


def test_per_feature_weights_unsupported_combinations():
    X, _ = blobs(d=3)
    W = torch.ones(X.shape, dtype=DT)
    prior = dict(mean_prior=torch.zeros(3, 3, dtype=DT), mean_precision_prior=0.5)
    with pytest.raises(NotImplementedError):
        GaussianMixture(n_components=3, n_features=3, covariance_type='diag', **prior).fit(X, sample_weight=W)
    with pytest.raises(NotImplementedError):
        GaussianMixture(n_components=3, covariance_type='diag', noise=True).fit(X, sample_weight=W)
    with pytest.raises(NotImplementedError):
        GaussianMixture(n_components=3, covariance_type='diag', missing='em').fit(X, sample_weight=W)
    with pytest.raises(NotImplementedError):
        GaussianMixture(n_components=3, covariance_type='diag', step_size='auto', batch_size=32).fit(X, sample_weight=W)


# ===========================================================================
# 3. frozen_components
# ===========================================================================
def frozen_model(family, X, frozen=(1,), cov='full', **kw):
    m = make(family, X, cov=cov, frozen_components=frozen, **kw) if family != 'vmf' else \
        make(family, X, frozen_components=frozen, **kw)
    return m


@pytest.mark.parametrize('cov', ['full', 'diag', 'spherical'])
def test_frozen_components_keep_initial_parameters_gaussian(cov):
    X, _ = blobs()
    probe = make('gmm', X, cov=cov)
    probe.dtype = DT
    probe._allocate_parameters(X, set_random_state=False)
    mu0, cov0 = probe.means_.clone(), probe.covariances_.clone()
    m = frozen_model('gmm', X, frozen=(1,), cov=cov)
    fit_quiet(m, X)
    assert torch.equal(m.means_[1], mu0[1])
    assert torch.equal(m.covariances_[1], cov0[1])
    assert not torch.allclose(m.means_[0], mu0[0], atol=1e-3)               # unfrozen components do move
    assert not torch.allclose(m.covariances_[0], cov0[0], atol=1e-3)
    assert abs(m.weights_.sum().item() - 1.0) < 1e-6


def test_frozen_covariance_equals_its_initial_value():
    X, _ = blobs()
    cov0 = torch.stack([torch.eye(2, dtype=DT) * s for s in (1.0, 2.5, 0.7)])
    m = GaussianMixture(n_components=3, n_features=2, init_means=init_means_for(X), init_covariances=cov0,
                        frozen_components=(0, 2), max_iter=20, tol=1e-30)
    fit_quiet(m, X)
    assert torch.equal(m.covariances_[0], cov0[0]) and torch.equal(m.covariances_[2], cov0[2])
    assert not torch.equal(m.covariances_[1], cov0[1])


def test_frozen_weights_still_update():
    X, _ = blobs()
    m = frozen_model('gmm', X, frozen=(0,))
    fit_quiet(m, X)
    assert abs(m.weights_.sum().item() - 1.0) < 1e-6
    assert (m.weights_ > 0).all()


def test_frozen_components_student_t_and_vmf():
    X, _ = blobs()
    m = frozen_model('t', X, frozen=(2,), cov='full')
    mu0 = m.init_means.clone()
    fit_quiet(m, X)
    assert torch.equal(m.means_[2], mu0[2])
    assert m.dof_[2].item() == 5.0                      # dof_init: frozen components keep their dof too
    assert m.dof_[0].item() != 5.0

    Xs, _ = sphere_blobs()
    v = frozen_model('vmf', Xs, frozen=(1,))
    mu0, k0 = v.init_means.clone(), None
    fit_quiet(v, Xs)
    assert torch.allclose(v.means_[1], mu0[1] / mu0[1].norm())
    assert v.kappa_[1].item() == 10.0                   # kappa_init
    assert v.kappa_[0].item() != 10.0


@pytest.mark.parametrize('family', ['gmm', 't', 'vmf'])
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_frozen_components_all_algorithms(family, algorithm):
    X, _ = data_for(family, n_per=60)
    m = frozen_model(family, X, frozen=(0, 2), algorithm=algorithm, max_iter=15)
    mu0 = m.init_means.clone()
    fit_quiet(m, X)
    ref = mu0 if family != 'vmf' else mu0 / mu0.norm(dim=1, keepdim=True)
    assert torch.allclose(m.means_[0], ref[0]) and torch.allclose(m.means_[2], ref[2])
    assert abs(m.weights_.sum().item() - 1.0) < 1e-6
    assert all(torch.isfinite(v).all() for v in params(m).values())


def test_frozen_component_is_not_reseeded_by_sem():
    X, _ = blobs()
    far = torch.tensor([[100.0, 100.0], [0.0, 4.0], [4.0, -4.0]], dtype=DT)       # component 0 gets no points
    m = GaussianMixture(n_components=3, n_features=2, init_means=far, init_covariances='eye', algorithm='sem',
                        frozen_components=(0,), max_iter=12, random_state=0)
    fit_quiet(m, X)
    assert torch.equal(m.means_[0], far[0])
    unfrozen = GaussianMixture(n_components=3, n_features=2, init_means=far, init_covariances='eye',
                               algorithm='sem', max_iter=12, random_state=0)
    fit_quiet(unfrozen, X)
    assert not torch.equal(unfrozen.means_[0], far[0])          # without freezing, SEM re-seeds it


def test_frozen_components_are_not_shrunk_by_priors():
    X, _ = blobs()
    m = GaussianMixture(n_components=3, n_features=2, init_means=init_means_for(X), init_covariances='eye',
                        mean_prior=torch.zeros(3, 2, dtype=DT), mean_precision_prior=50.0,
                        covariance_prior=torch.eye(2, dtype=DT).repeat(3, 1, 1), degrees_of_freedom_prior=20.0,
                        frozen_components=(1,), max_iter=20, tol=1e-30)
    mu0 = m.init_means.clone()
    fit_quiet(m, X)
    assert torch.equal(m.means_[1], mu0[1])
    assert not torch.allclose(m.means_[0], mu0[0], atol=1e-3)


def test_frozen_with_noise_noise_weight_updates():
    X, _ = blobs()
    noise = (torch.rand(15, 2, generator=torch.Generator().manual_seed(9), dtype=DT) - 0.5) * 24
    X = torch.cat([X, noise])
    m = frozen_model('gmm', X, frozen=(0,), noise=True)
    mu0 = m.init_means.clone()
    fit_quiet(m, X)
    assert torch.equal(m.means_[0], mu0[0])
    assert abs(m.weights_.sum().item() + m.noise_weight_.item() - 1.0) < 1e-6


def test_frozen_components_use_previous_fit_under_warm_start():
    X, _ = blobs()
    m = GaussianMixture(n_components=3, n_features=2, init_means=init_means_for(X), init_covariances='eye',
                        max_iter=10, tol=1e-30, random_state=0)
    fit_quiet(m, X)
    snapshot = m.means_.clone(), m.covariances_.clone()
    m.frozen_components = (0,)
    m.warm_start = True
    fit_quiet(m, X)
    assert torch.equal(m.means_[0], snapshot[0][0]) and torch.equal(m.covariances_[0], snapshot[1][0])


def test_frozen_batch_size_equality():
    X, _ = blobs(n_per=50)
    a = fit_quiet(frozen_model('gmm', X, frozen=(1,)), X)
    b = fit_quiet(frozen_model('gmm', X, frozen=(1,), batch_size=37), X)
    assert_same_params(a, b)


@pytest.mark.parametrize('kwargs,match', [
    (dict(covariance_type='tied_full'), 'tied'),
    (dict(covariance_type='tied_diag'), 'tied'),
    (dict(covariance_type='tied_spherical'), 'tied'),
    (dict(covariance_type='VEI'), 'GPCM'),
    (dict(covariance_type='EEE'), 'tied'),
    (dict(covariance_type='VEV'), 'GPCM'),
])
def test_frozen_unsupported_covariance_types(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GaussianMixture(n_components=3, frozen_components=(0,), **kwargs)


def test_frozen_unsupported_shared_parameters_and_invalid_indices():
    with pytest.raises(ValueError, match='shared'):
        StudentTMixture(n_components=3, dof='shared', frozen_components=(0,))
    with pytest.raises(ValueError, match='shared'):
        VonMisesFisherMixture(n_components=3, concentration='shared', frozen_components=(0,))
    with pytest.raises(ValueError, match='Gibbs'):
        GaussianMixture(n_components=None, frozen_components=(0,))
    for bad in [(3,), (-1,), (0, 0), (0.5,), ('a',)]:
        with pytest.raises((ValueError, TypeError)):
            GaussianMixture(n_components=3, frozen_components=bad)
    assert GaussianMixture(n_components=3, frozen_components=[2, 0]).frozen_components == (0, 2)
    assert GaussianMixture(n_components=3).frozen_components == ()


def test_frozen_components_without_any_freeze_is_bit_identical():
    X, _ = blobs()
    a = fit_quiet(make('gmm', X), X)
    b = fit_quiet(make('gmm', X, frozen_components=()), X)
    for name, v in params(a).items():
        assert torch.equal(v, getattr(b, name))


def test_frozen_components_persistence_round_trips():
    X, _ = blobs()
    Xs, _ = sphere_blobs()
    models = [
        (frozen_model('gmm', X, frozen=(1,)), X, GaussianMixture),
        (frozen_model('t', X, frozen=(0, 2)), X, StudentTMixture),
        (frozen_model('vmf', Xs, frozen=(1,)), Xs, VonMisesFisherMixture),
    ]
    for m, data, cls in models:
        fit_quiet(m, data)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'm.pth')
            m.save(path)
            assert cls.load(path).frozen_components == m.frozen_components
        state = m.save_state_dict()
        fresh = make('gmm', X) if cls is GaussianMixture else (make('t', X) if cls is StudentTMixture else make('vmf', Xs))
        fresh.load_state_dict(state)
        assert fresh.frozen_components == m.frozen_components
        assert cls.__name__ and m.to_dict()['frozen_components'] == list(m.frozen_components)


# ===========================================================================
# 4. PositiveUnlabeledMixture(anchor='fixed'): normalisation bug fix
# ===========================================================================
def pu_data(seed=0):
    g = torch.Generator().manual_seed(seed)
    n_pos, n_neg = 600, 1400
    pos = torch.randn(n_pos, 2, generator=g, dtype=DT) * 0.7 + torch.tensor([3.0, 3.0], dtype=DT)
    neg = torch.randn(n_neg, 2, generator=g, dtype=DT) * 1.0 + torch.tensor([-1.0, -1.0], dtype=DT)
    X = torch.cat([pos, neg])
    is_pos = torch.cat([torch.ones(n_pos, dtype=torch.bool), torch.zeros(n_neg, dtype=torch.bool)])
    labeled = torch.zeros_like(is_pos)
    labeled[:int(0.3 * n_pos)] = True
    return X, is_pos, labeled


def test_pu_fixed_anchor_weights_are_normalised_and_class_prior_is_unbiased():
    X, is_pos, labeled = pu_data()
    pu = PositiveUnlabeledMixture(n_components_positive=1, n_components_negative=1, family='gaussian',
                                  anchor='fixed', covariance_type='full', random_state=0, max_iter=200)
    fit_quiet(pu, X, labeled)
    assert abs(pu.model_.weights_.sum().item() - 1.0) < 1e-6
    unlabeled = ~labeled
    true_prior = (is_pos & unlabeled).sum().item() / unlabeled.sum().item()
    assert abs(pu.class_prior_ - true_prior) < 0.05
    # the positive block is frozen at its labeled-only fit and never moves
    assert pu.model_.frozen_components == (1,)
    ref = GaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(X[labeled])
    assert torch.allclose(pu.model_.means_[1], ref.means_[0], atol=1e-6)
    assert torch.allclose(pu.model_.covariances_[1], ref.covariances_[0], atol=1e-6)


@pytest.mark.parametrize('family', ['student_t'])
def test_pu_fixed_anchor_other_families_normalised(family):
    X, is_pos, labeled = pu_data()
    pu = PositiveUnlabeledMixture(n_components_positive=1, n_components_negative=1, family=family,
                                  anchor='fixed', random_state=0, max_iter=100)
    fit_quiet(pu, X, labeled)
    assert abs(pu.model_.weights_.sum().item() - 1.0) < 1e-6


def test_pu_rejects_frozen_components_kwarg():
    with pytest.raises(ValueError, match='frozen_components'):
        PositiveUnlabeledMixture(frozen_components=(0,))
