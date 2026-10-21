"""Fixed values and bounds for GaussianMixture (weight_bounds, fixed_weights, mean_bounds, covariance_bounds,
frozen_means, frozen_covariances). See docs/superpowers/specs/2026-10-21-gmm-constraints-design.md."""
import os
import tempfile
import warnings

import numpy as np
import pytest
import torch
from scipy import optimize

from tgmm import GaussianMixture, select_model
from tgmm.constraints import box_qp_coordinate_ascent, clip_eigenvalues, water_fill_weights

DT = torch.float64
SIX = ['full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical']
TIED = ('tied_full', 'tied_diag', 'tied_spherical')


def blobs(n_per=120, d=2, seed=0, spread=1.0, centres=((0.2, 0.2), (0.8, 0.3), (0.5, 0.9))):
    g = torch.Generator().manual_seed(seed)
    parts = []
    for c in centres:
        c = torch.tensor(list(c) + [0.0] * (d - len(c)), dtype=DT)
        parts.append(c + 0.12 * spread * torch.randn(n_per, d, generator=g, dtype=DT))
    return torch.cat(parts)


def fit_steps(model, X, n_steps=12, check=None):
    """Run EM one iteration at a time (warm start) and return the log-likelihood trace; `check(model)` after each."""
    trace = []
    for _ in range(n_steps):
        model.fit(X)
        trace.append(float(model.score(X)) * X.shape[0])
        if check is not None:
            check(model)
    return trace


def stepper(**kw):
    kw.setdefault('n_components', 3)
    return GaussianMixture(max_iter=1, tol=1e-30, warm_start=True, random_state=0, **kw)


@pytest.fixture(autouse=True)
def _silence():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# ---------------------------------------------------------------------------
# (a) defaults are bit-identical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('cov_type', SIX + ['EVE'])
def test_defaults_are_bit_identical(cov_type):
    X = blobs()
    kw = dict(n_components=3, covariance_type=cov_type, random_state=0, max_iter=40)
    a = GaussianMixture(**kw).fit(X)
    b = GaussianMixture(weight_bounds=None, fixed_weights=None, mean_bounds=None, covariance_bounds=None,
                        frozen_means=(), frozen_covariances=(), **kw).fit(X)
    for name in ('weights_', 'means_', 'covariances_'):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert a.n_iter_ == b.n_iter_


@pytest.mark.parametrize('cov_type', SIX)
def test_inactive_bounds_reproduce_plain_em(cov_type):
    X = blobs()
    kw = dict(n_components=3, covariance_type=cov_type, random_state=0, max_iter=60, tol=1e-9)
    a = GaussianMixture(**kw).fit(X)
    b = GaussianMixture(weight_bounds=(0.0, 1.0), mean_bounds=(-100.0, 100.0), covariance_bounds=(0.0, 1e6), **kw).fit(X)
    assert torch.allclose(a.weights_, b.weights_, atol=1e-10)
    assert torch.allclose(a.means_, b.means_, atol=1e-9)
    assert torch.allclose(a.covariances_, b.covariances_, atol=1e-9)


# ---------------------------------------------------------------------------
# (b) each block against an independent oracle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('seed', range(6))
def test_weight_water_filling_matches_a_numerical_optimiser(seed):
    rng = np.random.default_rng(seed)
    K = 5
    n = rng.uniform(0.5, 50.0, K)
    lo = rng.uniform(0.0, 0.12, K)
    hi = rng.uniform(0.3, 0.7, K)
    got = water_fill_weights(torch.tensor(n, dtype=DT), torch.tensor(lo, dtype=DT), torch.tensor(hi, dtype=DT)).numpy()
    res = optimize.minimize(lambda p: -(n * np.log(np.clip(p, 1e-300, None))).sum(), x0=np.full(K, 1.0 / K),
                            bounds=list(zip(lo, hi)), constraints=[{'type': 'eq', 'fun': lambda p: p.sum() - 1.0}],
                            method='SLSQP', options={'ftol': 1e-14, 'maxiter': 500})
    assert abs(got.sum() - 1.0) < 1e-12
    assert np.all(got >= lo - 1e-12) and np.all(got <= hi + 1e-12)
    assert np.allclose(got, res.x, atol=1e-5)


def test_weight_water_filling_unconstrained_is_proportional_and_fixed_when_degenerate():
    n = torch.tensor([3.0, 1.0, 6.0], dtype=DT)
    got = water_fill_weights(n, torch.zeros(3, dtype=DT), torch.ones(3, dtype=DT))
    assert torch.allclose(got, n / n.sum(), atol=1e-14)
    pinned = water_fill_weights(n, torch.tensor([0.2, 0.3, 0.5], dtype=DT), torch.tensor([0.2, 0.3, 0.5], dtype=DT))
    assert torch.allclose(pinned, torch.tensor([0.2, 0.3, 0.5], dtype=DT), atol=1e-12)
    with pytest.raises(ValueError):
        water_fill_weights(n, torch.full((3,), 0.5, dtype=DT), torch.ones(3, dtype=DT))     # sum(lo) > 1


@pytest.mark.parametrize('seed', range(5))
def test_box_qp_coordinate_ascent_matches_lbfgsb(seed):
    rng = np.random.default_rng(seed)
    d = 4
    A = rng.normal(size=(d, d))
    Sigma = A @ A.T + 0.5 * np.eye(d)
    P = np.linalg.inv(Sigma)
    ybar = rng.normal(size=d) * 2.0
    lo, hi = np.full(d, -0.5), np.full(d, 0.7)
    x0 = np.clip(rng.normal(size=d), lo, hi)
    got = box_qp_coordinate_ascent(torch.tensor(ybar, dtype=DT)[None], torch.tensor(P, dtype=DT)[None],
                                   torch.tensor(lo, dtype=DT)[None], torch.tensor(hi, dtype=DT)[None],
                                   torch.tensor(x0, dtype=DT)[None], max_sweeps=5000, tol=1e-14)[0].numpy()
    res = optimize.minimize(lambda x: 0.5 * (x - ybar) @ P @ (x - ybar), x0=x0, jac=lambda x: P @ (x - ybar),
                            bounds=list(zip(lo, hi)), method='L-BFGS-B', options={'ftol': 1e-15, 'gtol': 1e-12})
    assert np.all(got >= lo - 1e-12) and np.all(got <= hi + 1e-12)
    assert np.allclose(got, res.x, atol=1e-5)


def test_box_qp_one_sweep_never_decreases_the_objective():
    rng = np.random.default_rng(0)
    d = 5
    A = rng.normal(size=(d, d))
    P = np.linalg.inv(A @ A.T + 0.3 * np.eye(d))
    ybar = rng.normal(size=d) * 3
    lo, hi = np.full(d, -1.0), np.full(d, 1.0)
    x0 = np.clip(rng.normal(size=d), lo, hi)
    obj = lambda x: -0.5 * (x - ybar) @ P @ (x - ybar)
    out = box_qp_coordinate_ascent(*(torch.tensor(a, dtype=DT)[None] for a in (ybar, P, lo, hi, x0)), max_sweeps=1)[0].numpy()
    assert obj(out) >= obj(x0) - 1e-12


@pytest.mark.parametrize('seed', range(4))
def test_clip_eigenvalues_matches_numerical_minimisation(seed):
    """argmin n log|S| + tr(S^-1 T) over lo I <= S <= hi I equals the eigenvalue-clipped T / n."""
    rng = np.random.default_rng(seed)
    d, n = 3, 40.0
    B = rng.normal(size=(d, d))
    T = B @ B.T * 3 + 0.01 * np.eye(d)
    lo, hi = 0.4, 2.5
    got = clip_eigenvalues(torch.tensor(T / n, dtype=DT)[None], lo, hi)[0].numpy()
    w = np.linalg.eigvalsh(got)
    assert w.min() >= lo - 1e-10 and w.max() <= hi + 1e-10

    def build(theta):
        L = np.zeros((d, d))
        L[np.tril_indices(d)] = theta
        return L @ L.T

    def objective(theta):
        S = build(theta)
        ev = np.linalg.eigvalsh(S)
        if ev.min() < lo or ev.max() > hi:
            return 1e6 + 1e3 * (max(lo - ev.min(), 0) + max(ev.max() - hi, 0))
        return n * np.log(ev).sum() + np.trace(np.linalg.solve(S, T))

    L0 = np.linalg.cholesky(got + 1e-3 * np.eye(d))
    res = optimize.minimize(objective, L0[np.tril_indices(d)], method='Nelder-Mead',
                            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 20000, 'maxfev': 20000})
    ours = n * np.log(w).sum() + np.trace(np.linalg.solve(got, T))
    assert ours <= res.fun + 1e-6            # nothing the optimiser finds beats the clipped solution


def test_clip_eigenvalues_leaves_feasible_matrices_untouched():
    S = torch.tensor([[[2.0, 0.3], [0.3, 1.0]]], dtype=DT)
    assert torch.equal(clip_eigenvalues(S, 0.1, 5.0), S)


# ---------------------------------------------------------------------------
# (c) + (d) monotone likelihood; bounds hold after every iteration
# ---------------------------------------------------------------------------
def _assert_monotone(trace):
    assert all(b >= a - 1e-8 * max(1.0, abs(a)) for a, b in zip(trace, trace[1:])), trace


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('cov_type', SIX)
def test_weight_bounds_monotone_and_respected(cov_type, seed):
    X = blobs(seed=seed)
    lo, hi = torch.tensor([0.25, 0.2, 0.15], dtype=DT), torch.tensor([0.45, 0.5, 0.5], dtype=DT)

    def check(m):
        assert torch.all(m.weights_ >= lo - 1e-12) and torch.all(m.weights_ <= hi + 1e-12)
        assert abs(float(m.weights_.sum()) - 1.0) < 1e-12

    _assert_monotone(fit_steps(stepper(covariance_type=cov_type, weight_bounds=(lo, hi)), X, 12, check))


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('cov_type', SIX)
def test_mean_bounds_monotone_and_respected(cov_type, seed):
    X = blobs(seed=seed, spread=1.5)
    lo, hi = torch.tensor([0.3, 0.3], dtype=DT), torch.tensor([0.7, 0.8], dtype=DT)

    def check(m):
        assert torch.all(m.means_ >= lo - 1e-12) and torch.all(m.means_ <= hi + 1e-12)

    _assert_monotone(fit_steps(stepper(covariance_type=cov_type, mean_bounds=(lo, hi)), X, 12, check))


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('cov_type', SIX)
def test_covariance_bounds_monotone_and_respected(cov_type, seed):
    X = blobs(seed=seed, spread=1.4)
    lo, hi = 0.02, 0.2

    def check(m):
        c = m.covariances_
        if cov_type in ('full', 'tied_full'):
            ev = torch.linalg.eigvalsh(c if c.dim() == 3 else c[None])
            assert ev.min() >= lo - 1e-10 and ev.max() <= hi + 1e-10
        else:
            assert c.min() >= lo - 1e-12 and c.max() <= hi + 1e-12

    _assert_monotone(fit_steps(stepper(covariance_type=cov_type, covariance_bounds=(lo, hi)), X, 12, check))


@pytest.mark.parametrize('cov_type', ['full', 'diag', 'spherical'])
def test_all_constraints_together_monotone(cov_type):
    X = blobs(spread=1.3)
    m = stepper(covariance_type=cov_type, weight_bounds=(0.2, 0.5), mean_bounds=(0.25, 0.8), covariance_bounds=(0.01, 0.1))
    _assert_monotone(fit_steps(m, X, 15))
    assert torch.all(m.means_ >= 0.25 - 1e-12) and torch.all(m.means_ <= 0.8 + 1e-12)


def test_per_component_covariance_bounds_and_one_sided_mean_bounds():
    X = blobs(spread=1.3)
    m = stepper(covariance_type='diag', covariance_bounds=(torch.tensor([0.0, 0.05, 0.0], dtype=DT), torch.tensor([0.5, 0.5, 0.01], dtype=DT)),
                mean_bounds=(torch.tensor([-float('inf'), 0.0], dtype=DT), torch.tensor([float('inf'), 0.6], dtype=DT)))
    fit_steps(m, X, 10)
    assert torch.all(m.covariances_[1] >= 0.05 - 1e-12) and torch.all(m.covariances_[2] <= 0.01 + 1e-12)
    assert torch.all(m.means_[:, 1] <= 0.6 + 1e-12) and torch.all(m.means_[:, 1] >= -1e-12)


def test_fixed_weights_stay_exactly_the_given_tensor():
    X = blobs()
    w = torch.tensor([0.5, 0.3, 0.2], dtype=DT)

    def check(m):
        assert torch.equal(m.weights_, w), 'weights moved'

    m = stepper(fixed_weights=w, init_weights=w.clone())
    fit_steps(m, X, 6, check)
    assert torch.equal(m.weights_, w)


@pytest.mark.parametrize('cov_type', ['full', 'diag', 'spherical'])
def test_frozen_means_and_covariances_stay_bit_identical(cov_type):
    X = blobs()
    g = torch.Generator().manual_seed(1)
    means0 = torch.tensor([[0.2, 0.2], [0.8, 0.3], [0.5, 0.9]], dtype=DT)
    if cov_type == 'full':
        cov0 = (0.02 * torch.eye(2, dtype=DT)).repeat(3, 1, 1)
    elif cov_type == 'diag':
        cov0 = torch.full((3, 2), 0.02, dtype=DT)
    else:
        cov0 = torch.full((3,), 0.02, dtype=DT)
    m = stepper(covariance_type=cov_type, init_means=means0.clone(), init_covariances=cov0.clone(),
                frozen_means=[0], frozen_covariances=[1])
    fit_steps(m, X, 8)
    assert torch.equal(m.means_[0], means0[0])                  # frozen mean never moves
    assert torch.equal(m.covariances_[1], cov0[1])              # frozen covariance never moves
    assert not torch.equal(m.means_[1], means0[1]) and not torch.equal(m.covariances_[0], cov0[0])


def test_frozen_mean_uses_the_scatter_about_the_fixed_mean():
    """Covariance MLE with the means pinned: T/n with T the scatter about the pinned means (not about ybar)."""
    X = blobs()
    means0 = torch.tensor([[0.3, 0.3], [0.8, 0.3], [0.5, 0.9]], dtype=DT)
    m = GaussianMixture(n_components=3, n_features=2, covariance_type='diag', init_means=means0.clone(),
                        frozen_means=[0, 1, 2], reg_covar=0.0, random_state=0)
    m.dtype = DT
    m._allocate_parameters(X)
    resp, _ = m._e_step(X)
    m._m_step(X, resp)
    nk = resp.sum(0) + 1e-20
    expected = torch.einsum('nk,nkd->kd', resp, (X[:, None, :] - means0[None]) ** 2) / nk[:, None]
    assert torch.equal(m.means_, means0)
    assert torch.allclose(m.covariances_, expected, rtol=1e-10, atol=1e-12)


def test_frozen_components_combines_with_frozen_means_and_bounds():
    X = blobs()
    means0 = torch.tensor([[0.2, 0.2], [0.8, 0.3], [0.5, 0.9]], dtype=DT)
    m = stepper(covariance_type='diag', init_means=means0.clone(), frozen_components=[2], frozen_means=[0],
                mean_bounds=(0.0, 1.0), covariance_bounds=(0.01, 0.2))
    fit_steps(m, X, 6)
    assert torch.equal(m.means_[0], means0[0]) and torch.equal(m.means_[2], means0[2])


# ---------------------------------------------------------------------------
# (e) recovery examples
# ---------------------------------------------------------------------------
def test_mean_bounds_keep_the_fit_inside_the_box_when_free_em_leaves_it():
    g = torch.Generator().manual_seed(3)
    clean = torch.cat([0.2 + 0.05 * torch.randn(150, 1, generator=g, dtype=DT), 0.8 + 0.05 * torch.randn(150, 1, generator=g, dtype=DT)])
    outliers = 3.0 + 0.1 * torch.randn(30, 1, generator=g, dtype=DT)
    X = torch.cat([clean, outliers])
    free = GaussianMixture(n_components=2, random_state=0, max_iter=200, n_init=3).fit(X)
    boxed = GaussianMixture(n_components=2, random_state=0, max_iter=200, n_init=3, mean_bounds=(0.0, 1.0)).fit(X)
    assert bool((free.means_ > 1.0).any())                       # setup: the unconstrained fit chases the outliers
    assert torch.all(boxed.means_ >= -1e-12) and torch.all(boxed.means_ <= 1.0 + 1e-12)


def test_fixing_a_known_component_helps_recover_the_overlapping_rest():
    errs_free, errs_fixed = [], []
    truth = torch.tensor([[0.0, 0.0], [1.6, 0.0], [0.8, 1.4]], dtype=DT)
    for seed in range(8):
        g = torch.Generator().manual_seed(seed)
        X = torch.cat([truth[k] + torch.randn(60, 2, generator=g, dtype=DT) * 0.9 for k in range(3)])
        free = GaussianMixture(n_components=3, covariance_type='spherical', random_state=seed, max_iter=300, n_init=4).fit(X)
        fixed = GaussianMixture(n_components=3, covariance_type='spherical', random_state=seed, max_iter=300,
                                init_means=truth.clone() + 0.3, frozen_means=[0], frozen_covariances=[0],
                                init_covariances=torch.tensor([0.81, 0.8, 0.8], dtype=DT)).fit(X)
        fixed.means_[0] = truth[0]
        def err(m):
            d = torch.cdist(m.means_, truth)
            return float(d.min(dim=0).values.mean())
        errs_free.append(err(free)); errs_fixed.append(err(fixed))
    assert np.mean(errs_fixed) < np.mean(errs_free)


# ---------------------------------------------------------------------------
# (f) algorithms, chunking, persistence, select_model
# ---------------------------------------------------------------------------
CONSTRAINTS = dict(weight_bounds=(0.2, 0.5), mean_bounds=(0.25, 0.8), covariance_bounds=(0.01, 0.1))


def _within(m):
    assert torch.all(m.means_ >= 0.25 - 1e-9) and torch.all(m.means_ <= 0.8 + 1e-9)
    assert torch.all(m.weights_ >= 0.2 - 1e-9) and torch.all(m.weights_ <= 0.5 + 1e-9)
    assert torch.isfinite(m.covariances_).all()


@pytest.mark.parametrize('algorithm', ['em', 'cem', 'sem', 'saem'])
def test_algorithms_respect_the_constraints(algorithm):
    X = blobs(spread=1.3)
    m = GaussianMixture(n_components=3, covariance_type='diag', algorithm=algorithm, random_state=0, max_iter=40,
                        **CONSTRAINTS).fit(X)
    _within(m)


def test_online_em_respects_the_constraints():
    X = blobs(spread=1.3)
    m = GaussianMixture(n_components=3, covariance_type='diag', batch_size=90, step_size='auto', random_state=0,
                        max_iter=5, **CONSTRAINTS).fit(X)
    _within(m)
    m.partial_fit(X[:100])
    _within(m)


@pytest.mark.parametrize('cov_type', ['full', 'diag', 'spherical'])
def test_chunked_fit_equals_unchunked_with_constraints(cov_type):
    X = blobs(spread=1.3)
    kw = dict(n_components=3, covariance_type=cov_type, random_state=0, max_iter=30, tol=1e-12, **CONSTRAINTS)
    a = GaussianMixture(**kw).fit(X)
    b = GaussianMixture(batch_size=97, **kw).fit(X)
    for name in ('weights_', 'means_', 'covariances_'):
        assert torch.allclose(getattr(a, name), getattr(b, name), atol=1e-8), name


def test_row_sample_weight_integer_weights_equal_duplication():
    X = blobs(n_per=40, spread=1.3)
    w = torch.randint(1, 4, (X.shape[0],)).to(DT)
    Xdup = torch.repeat_interleave(X, w.long(), dim=0)
    kw = dict(n_components=3, covariance_type='diag', init_means=X[::40][:3].clone(), max_iter=25, tol=1e-12, **CONSTRAINTS)
    a = GaussianMixture(**kw).fit(X, sample_weight=w)
    b = GaussianMixture(**kw).fit(Xdup)
    assert torch.allclose(a.means_, b.means_, atol=1e-8) and torch.allclose(a.weights_, b.weights_, atol=1e-8)


def test_missing_values_work_with_mean_and_covariance_bounds():
    X = blobs(spread=1.3)
    g = torch.Generator().manual_seed(0)
    Xm = X.clone()
    mask = torch.rand(Xm.shape, generator=g) < 0.1
    mask[:, 0] &= ~mask[:, 1]
    Xm[mask] = float('nan')
    m = GaussianMixture(n_components=3, covariance_type='full', missing='em', random_state=0, max_iter=40,
                        mean_bounds=(0.25, 0.8), covariance_bounds=(0.01, 0.1)).fit(Xm)
    assert torch.all(m.means_ >= 0.25 - 1e-9) and torch.all(m.means_ <= 0.8 + 1e-9)


def test_noise_component_works_with_mean_and_covariance_bounds():
    X = blobs(spread=1.3)
    m = GaussianMixture(n_components=3, covariance_type='diag', noise=True, random_state=0, max_iter=40,
                        mean_bounds=(0.25, 0.8), covariance_bounds=(0.01, 0.1)).fit(X)
    assert torch.all(m.means_ >= 0.25 - 1e-9) and torch.all(m.means_ <= 0.8 + 1e-9)


def test_weight_bounds_with_a_dirichlet_prior():
    X = blobs()
    m = GaussianMixture(n_components=3, covariance_type='diag', random_state=0, max_iter=40,
                        weight_concentration_prior=torch.full((3,), 5.0, dtype=DT), weight_bounds=(0.3, 0.4)).fit(X)
    assert torch.all(m.weights_ >= 0.3 - 1e-9) and torch.all(m.weights_ <= 0.4 + 1e-9)


def test_save_load_and_state_dict_keep_the_constraints():
    X = blobs(spread=1.3)
    means0 = torch.tensor([[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]], dtype=DT)
    m = GaussianMixture(n_components=3, covariance_type='diag', random_state=0, max_iter=15, init_means=means0,
                        frozen_means=[1], **CONSTRAINTS).fit(X)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'm.pth')
        m.save(path)
        loaded = GaussianMixture.load(path)
    clone = GaussianMixture(n_components=3, n_features=2)
    clone.load_state_dict(m.save_state_dict())
    for other in (loaded, clone):
        assert other.frozen_means == (1,)
        assert other.weight_bounds is not None and other.mean_bounds is not None and other.covariance_bounds is not None
        assert float(other.score(X)) == pytest.approx(float(m.score(X)))
    assert 'mean_bounds' in m.to_dict()


def test_select_model_passes_constraints_through():
    X = blobs(spread=1.3)
    result = select_model(X, n_components=[2, 3], covariance_types=['diag'], random_state=0, max_iter=30,
                          mean_bounds=(0.25, 0.8), covariance_bounds=(0.01, 0.1))
    best = result.best_
    assert torch.all(best.means_ >= 0.25 - 1e-9) and torch.all(best.means_ <= 0.8 + 1e-9)
    assert best.covariance_bounds is not None


def test_active_constraints_reports_binding_bounds():
    X = blobs(spread=1.3)
    m = GaussianMixture(n_components=3, covariance_type='diag', random_state=0, max_iter=60,
                        mean_bounds=(0.45, 0.55), covariance_bounds=(0.05, 0.5),
                        weight_bounds=(torch.tensor([0.5, 0.1, 0.1], dtype=DT), 0.8)).fit(X)
    active = m.active_constraints_
    assert active['means_at_lower'].shape == m.means_.shape
    assert bool(active['means_at_lower'].any() or active['means_at_upper'].any())
    assert bool(active['weights_at_lower'].any() or active['weights_at_upper'].any())
    assert GaussianMixture(n_components=2).active_constraints_ is None


# ---------------------------------------------------------------------------
# bic / aic count only the removed (fixed / frozen) parameters
# ---------------------------------------------------------------------------
def test_information_criteria_discount_fixed_parameters_only():
    X = blobs()
    means0 = torch.tensor([[0.2, 0.2], [0.8, 0.3], [0.5, 0.9]], dtype=DT)
    cov0 = torch.full((3, 2), 0.02, dtype=DT)
    w = torch.tensor([0.4, 0.3, 0.3], dtype=DT)
    free = GaussianMixture(n_components=3, covariance_type='diag', random_state=0, max_iter=5).fit(X)
    p_free = free._n_parameters()
    assert p_free == 3 * 2 + 3 * 2 + 2
    bounded = GaussianMixture(n_components=3, covariance_type='diag', random_state=0, max_iter=5, **CONSTRAINTS).fit(X)
    assert bounded._n_parameters() == p_free                                       # bounds do not change the count
    fixed_w = GaussianMixture(n_components=3, covariance_type='diag', random_state=0, max_iter=5, fixed_weights=w).fit(X)
    assert fixed_w._n_parameters() == p_free - 2                                   # K - 1 weights removed
    fm = GaussianMixture(n_components=3, covariance_type='diag', random_state=0, max_iter=5, init_means=means0,
                         frozen_means=[0, 2]).fit(X)
    assert fm._n_parameters() == p_free - 2 * 2                                    # d per frozen mean
    fc = GaussianMixture(n_components=3, covariance_type='diag', random_state=0, max_iter=5, init_covariances=cov0,
                         frozen_covariances=[1]).fit(X)
    assert fc._n_parameters() == p_free - 2                                        # d per frozen diag covariance
    full = GaussianMixture(n_components=3, covariance_type='full', random_state=0, max_iter=5,
                           init_covariances=(0.02 * torch.eye(2, dtype=DT)).repeat(3, 1, 1), frozen_covariances=[0]).fit(X)
    assert full._n_parameters() == 3 * 2 + 3 * 3 + 2 - 3                           # d(d+1)/2 per frozen full covariance


# ---------------------------------------------------------------------------
# (g) documented errors
# ---------------------------------------------------------------------------
def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match='fixed_weights'):
        GaussianMixture(n_components=3, fixed_weights=torch.tensor([0.5, 0.5, 0.5]))                  # does not sum to 1
    with pytest.raises(ValueError, match='fixed_weights'):
        GaussianMixture(n_components=3, fixed_weights=torch.tensor([0.5, 0.5]))                       # wrong length
    with pytest.raises(ValueError, match='both'):
        GaussianMixture(n_components=3, fixed_weights=torch.tensor([0.4, 0.3, 0.3]), weight_bounds=(0.1, 0.9))
    with pytest.raises(ValueError, match='weight_bounds'):
        GaussianMixture(n_components=3, weight_bounds=(0.5, 0.9))                                     # sum(lo) > 1
    with pytest.raises(ValueError, match='weight_bounds'):
        GaussianMixture(n_components=3, weight_bounds=(0.0, 0.2))                                     # sum(hi) < 1
    with pytest.raises(ValueError, match='mean_bounds'):
        GaussianMixture(n_components=3, mean_bounds=(1.0, 0.0))                                       # lo > hi
    with pytest.raises(ValueError, match='covariance_bounds'):
        GaussianMixture(n_components=3, covariance_bounds=(-0.1, 1.0))                                # negative lower bound
    with pytest.raises(ValueError, match='covariance_bounds'):
        GaussianMixture(n_components=3, covariance_bounds=(0.5, 0.1))                                 # lo > hi
    with pytest.raises(ValueError, match='frozen_means'):
        GaussianMixture(n_components=3, frozen_means=[5], init_means=torch.zeros(3, 2))               # out of range
    with pytest.raises(ValueError, match='init_means'):
        GaussianMixture(n_components=3, frozen_means=[0])                                             # needs init_means tensor
    with pytest.raises(ValueError, match='init_covariances'):
        GaussianMixture(n_components=3, frozen_covariances=[0])                                       # needs init_covariances tensor
    with pytest.raises(ValueError, match='frozen_covariances'):
        GaussianMixture(n_components=3, covariance_type='tied_full', frozen_covariances=[0],
                        init_covariances=torch.eye(2))                                                # shared covariance
    with pytest.raises(ValueError, match='frozen_covariances'):
        GaussianMixture(n_components=3, covariance_type='EVE', frozen_covariances=[0],
                        init_covariances=torch.eye(2).repeat(3, 1, 1))


def test_unsupported_combinations_raise():
    kw = dict(mean_bounds=(0.0, 1.0))
    with pytest.raises(NotImplementedError, match='Gibbs'):
        GaussianMixture(n_components=None, **kw)
    with pytest.raises(NotImplementedError, match='prior'):
        GaussianMixture(n_components=3, n_features=2, mean_prior=torch.zeros(3, 2), mean_precision_prior=1.0, **kw)
    with pytest.raises(NotImplementedError, match='prior'):
        GaussianMixture(n_components=3, n_features=2, covariance_prior=torch.eye(2).repeat(3, 1, 1), degrees_of_freedom_prior=4.0,
                        covariance_bounds=(0.0, 1.0))
    with pytest.raises(NotImplementedError, match='GPCM'):
        GaussianMixture(n_components=3, covariance_type='EVE', covariance_bounds=(0.0, 1.0))
    with pytest.raises(NotImplementedError, match='noise'):
        GaussianMixture(n_components=3, noise=True, weight_bounds=(0.1, 0.6))
    with pytest.raises(NotImplementedError, match='noise'):
        GaussianMixture(n_components=3, noise=True, fixed_weights=torch.tensor([0.4, 0.3, 0.3]))


def test_labels_and_feature_weights_are_rejected_at_fit_time():
    X = blobs()
    labels = torch.randint(0, 3, (X.shape[0],))
    with pytest.raises(NotImplementedError, match='labels'):
        GaussianMixture(n_components=3, weight_bounds=(0.1, 0.6)).fit(X, labels=labels)
    with pytest.raises(NotImplementedError, match='per-feature'):
        GaussianMixture(n_components=3, covariance_type='diag', mean_bounds=(0.0, 1.0)).fit(X, sample_weight=torch.ones_like(X))
    # mean / covariance bounds are fine with labels
    GaussianMixture(n_components=3, covariance_type='diag', mean_bounds=(0.0, 1.0), max_iter=5).fit(X, labels=labels)


# ---------------------------------------------------------------------------
# (h) initial parameters are projected, with a warning
# ---------------------------------------------------------------------------
def test_initial_values_are_projected_with_a_warning():
    X = blobs()
    means0 = torch.tensor([[-1.0, 0.2], [0.8, 3.0], [0.5, 0.9]], dtype=DT)
    cov0 = torch.tensor([[2.0, 2.0], [0.02, 0.02], [0.02, 0.02]], dtype=DT)
    m = GaussianMixture(n_components=3, covariance_type='diag', init_means=means0, init_covariances=cov0,
                        mean_bounds=(0.0, 1.0), covariance_bounds=(0.01, 0.5), random_state=0, max_iter=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        m.fit(X)
    assert any('projected' in str(w.message) for w in caught)
    assert torch.all(m.initial_means_ >= 0.0) and torch.all(m.initial_means_ <= 1.0)
    assert float(m.initial_covariances_.max()) <= 0.5 + 1e-12


def test_no_warning_when_initial_values_are_feasible():
    X = blobs()
    means0 = torch.tensor([[0.2, 0.2], [0.8, 0.3], [0.5, 0.9]], dtype=DT)
    m = GaussianMixture(n_components=3, covariance_type='diag', init_means=means0, mean_bounds=(0.0, 1.0),
                        covariance_bounds=(0.001, 0.5), random_state=0, max_iter=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        m.fit(X)
    assert not any('projected' in str(w.message) for w in caught)
