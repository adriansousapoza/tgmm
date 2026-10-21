"""Regression tests for known bugs fixed in the mixture code (branch ``fix-known-bugs``).

1. ``StudentTMixture(covariance_type='spherical')`` divided by ``n_k`` instead of ``d * n_k``.
2. ``GaussianMixture`` NIW update for ``tied_spherical`` added ``reg_covar * n * d`` instead of ``reg_covar * d``.
3. ``unlabeled_weight < 1`` left ``weights_`` summing to less than 1.
4. k-means / empirical-covariance initialisation produced ``NaN`` for empty or singleton clusters.
5. Stale class docstrings (Student-t / vMF supervised fitting, ECME fixed point).
6. ``save`` / ``load`` / ``to_dict`` / state dicts dropped semi-supervised and init-only constructor arguments.
7. The E-step weight guard ``log(w + 1e-20)`` (left unchanged; documented below).
8. MAP covariance update docstrings (no behaviour change).
"""
import inspect
import os
import tempfile
import warnings

import numpy as np
import pytest
import torch

from tgmm import GaussianMixture, GMMInitializer, StudentTMixture, VonMisesFisherMixture

DT = torch.float64


def blobs(n_per=60, d=2, seed=0, K=3, spread=0.7):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[0.0, 5.0, -5.0, 8.0], [5.0, -5.0, 0.0, 8.0], [-5.0, 0.0, 5.0, -8.0]], dtype=DT)[:, :d]
    X, y = [], []
    for k in range(K):
        X.append(torch.randn(n_per, d, generator=g, dtype=DT) * spread + centers[k])
        y.append(torch.full((n_per,), k))
    return torch.cat(X), torch.cat(y)


def one_step_fit(model, X):
    """One EM iteration from the model's explicit initial state (no convergence warning noise)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X, max_iter=1, tol=1e-300)
    return model


# ---------------------------------------------------------------------------
# 1. Student-t spherical
# ---------------------------------------------------------------------------
def test_student_t_spherical_recovers_unit_variance_in_3d():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(4000, 3, generator=g, dtype=DT)
    m = StudentTMixture(n_components=1, covariance_type='spherical', dof=1e4, random_state=0, tol=1e-10, max_iter=200)
    m.fit(X)
    assert abs(m.covariances_.item() - 1.0) < 0.05, m.covariances_
    gm = GaussianMixture(n_components=1, covariance_type='spherical', random_state=0).fit(X)
    assert abs(m.covariances_.item() - gm.covariances_.item()) < 1e-3


@pytest.mark.parametrize('scale_update', ['ecm', 'ecme'])
@pytest.mark.parametrize('d', [2, 3, 5])
def test_student_t_spherical_one_ecm_step_matches_numpy(scale_update, d):
    X, _ = blobs(n_per=50, d=min(d, 4), seed=1)
    if d == 5:
        X = torch.cat([X, torch.randn(X.shape[0], 1, dtype=DT)], dim=1)
    n, d = X.shape
    K = 3
    mu0 = X[torch.tensor([0, 60, 120])].clone()
    sig0 = torch.tensor([1.5, 2.0, 2.5], dtype=DT)
    w0 = torch.tensor([0.3, 0.3, 0.4], dtype=DT)
    m = StudentTMixture(
        n_components=K, n_features=d, covariance_type='spherical', dof=5.0, scale_update=scale_update, reg_covar=0.0,
        init_means=mu0, init_covariances=sig0, init_weights=w0, random_state=0,
    )
    one_step_fit(m, X)

    # NumPy reference ECM step at the initial parameters
    x = X.numpy()
    mu, s2, w = mu0.numpy(), sig0.numpy(), w0.numpy()
    nuv = 5.0
    delta = ((x[:, None, :] - mu[None]) ** 2).sum(-1) / s2[None]
    logt = (
        np.vectorize(__import__('math').lgamma)((nuv + d) / 2) - __import__('math').lgamma(nuv / 2)
        - d / 2 * np.log(nuv * np.pi) - 0.5 * d * np.log(s2)[None] - (nuv + d) / 2 * np.log1p(delta / nuv)
    )
    logj = logt + np.log(w)[None]
    tau = np.exp(logj - logj.max(1, keepdims=True))
    tau /= tau.sum(1, keepdims=True)
    u = (nuv + d) / (nuv + delta)
    nk = tau.sum(0)
    mk = (tau * u).sum(0)
    ybar = (tau * u).T @ x / mk[:, None]
    scat = np.array([((tau[:, k] * u[:, k])[:, None] * (x - ybar[k]) ** 2).sum() for k in range(K)])
    denom = nk if scale_update == 'ecm' else mk
    expected = scat / (d * denom)
    assert np.allclose(m.covariances_.numpy(), expected, rtol=1e-9), (m.covariances_.numpy(), expected)
    assert np.allclose(m.means_.numpy(), ybar, rtol=1e-9)


@pytest.mark.parametrize('scale_update', ['ecm', 'ecme'])
def test_student_t_spherical_log_likelihood_is_monotone(scale_update):
    X, _ = blobs(n_per=80, d=3, seed=2)
    m = StudentTMixture(n_components=3, covariance_type='spherical', scale_update=scale_update, random_state=0,
                        tol=1e-300, max_iter=60)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X)
    hist = m.log_likelihood_history_.numpy()
    assert (np.diff(hist) > -1e-8).all(), np.diff(hist).min()


def test_student_t_spherical_matches_gaussian_spherical_at_large_dof():
    X, _ = blobs(n_per=100, d=3, seed=3)
    kw = dict(n_components=3, covariance_type='spherical', random_state=0, tol=1e-10, max_iter=300)
    t = StudentTMixture(dof=1e6, **kw).fit(X)
    g = GaussianMixture(**kw).fit(X)
    order_t = t.means_[:, 0].argsort()
    order_g = g.means_[:, 0].argsort()
    assert torch.allclose(t.covariances_[order_t], g.covariances_[order_g], rtol=1e-3)
    assert torch.allclose(t.means_[order_t], g.means_[order_g], atol=1e-3)


def test_student_t_spherical_missing_em_matches_gaussian_at_large_dof():
    X, _ = blobs(n_per=120, d=3, seed=4)
    g = torch.Generator().manual_seed(5)
    Xn = X.clone()
    holes = torch.rand(Xn.shape, generator=g) < 0.15
    holes[holes.all(dim=1)] = False
    Xn[holes] = float('nan')
    kw = dict(n_components=3, covariance_type='spherical', missing='em', random_state=0, tol=1e-10, max_iter=300)
    t = StudentTMixture(dof=1e6, **kw).fit(Xn)
    gm = GaussianMixture(**kw).fit(Xn)
    a, b = t.means_[:, 0].argsort(), gm.means_[:, 0].argsort()
    assert torch.allclose(t.covariances_[a], gm.covariances_[b], rtol=1e-3)


def test_student_t_spherical_integer_sample_weights_equal_duplication():
    X, _ = blobs(n_per=30, d=2, seed=6)
    reps = torch.randint(1, 4, (X.shape[0],), generator=torch.Generator().manual_seed(1))
    Xd = X.repeat_interleave(reps, dim=0)
    kw = dict(n_components=3, covariance_type='spherical', dof=7.0, tol=1e-300, max_iter=15, random_state=0,
              init_means=X[[0, 30, 60]].clone(), init_covariances=torch.ones(3, dtype=DT))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = StudentTMixture(**kw).fit(X, sample_weight=reps.to(DT))
        b = StudentTMixture(**kw).fit(Xd)
    assert torch.allclose(a.covariances_, b.covariances_, rtol=1e-8)
    assert torch.allclose(a.means_, b.means_, rtol=1e-8)


# ---------------------------------------------------------------------------
# 2. NIW tied spherical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('reg', [0.0, 1e-3, 0.5])
def test_niw_tied_spherical_matches_closed_form(reg):
    X, _ = blobs(n_per=40, d=2, seed=7)
    n, d = X.shape
    K = 3
    mu0 = torch.tensor([[0.5, 4.0], [4.0, -3.0], [-4.0, 1.0]], dtype=DT)
    lam0, nu0, psi0 = 0.7, 4.0, 2.5
    init_means = X[[0, 40, 80]].clone()
    m = GaussianMixture(
        n_components=K, n_features=d, covariance_type='tied_spherical', reg_covar=reg,
        init_means=init_means, init_covariances=torch.tensor(1.3, dtype=DT), init_weights=torch.full((K,), 1 / 3, dtype=DT),
        mean_prior=mu0, mean_precision_prior=lam0, covariance_prior=torch.tensor(psi0, dtype=DT),
        degrees_of_freedom_prior=nu0, random_state=0,
    )
    one_step_fit(m, X)

    x = X.numpy()
    mu, var = init_means.numpy(), 1.3
    logp = -0.5 * ((x[:, None, :] - mu[None]) ** 2).sum(-1) / var - 0.5 * d * np.log(var)
    tau = np.exp(logp - logp.max(1, keepdims=True))
    tau /= tau.sum(1, keepdims=True)
    nk = tau.sum(0)
    ybar = tau.T @ x / nk[:, None]
    scat = sum((tau[:, k][:, None] * (x - ybar[k]) ** 2).sum() for k in range(K))
    cross = sum(lam0 * nk[k] / (lam0 + nk[k]) * ((ybar[k] - mu0.numpy()[k]) ** 2).sum() for k in range(K))
    psi_n = psi0 + scat + cross + reg * d
    expected = psi_n / ((nu0 + nk.sum()) * d)
    assert abs(m.covariances_.item() - expected) < 1e-9 * max(1.0, abs(expected)), (m.covariances_.item(), expected)


def test_niw_tied_spherical_regularisation_is_consistent_with_other_types():
    """reg_covar must perturb the NIW variance by O(reg / nu_n), not O(reg * n / nu_n)."""
    X, _ = blobs(n_per=200, d=2, seed=8)
    kw = dict(n_components=3, n_features=2, covariance_type='tied_spherical', random_state=0, tol=1e-300,
              mean_prior=torch.zeros(3, 2, dtype=DT), mean_precision_prior=0.1,
              covariance_prior=torch.tensor(1.0, dtype=DT), degrees_of_freedom_prior=3.0,
              init_means=X[[0, 200, 400]].clone(), init_covariances=torch.tensor(1.0, dtype=DT))
    lo = one_step_fit(GaussianMixture(reg_covar=0.0, **kw), X).covariances_.item()
    hi = one_step_fit(GaussianMixture(reg_covar=0.1, **kw), X).covariances_.item()
    # psi_n gains 0.1 * d = 0.2; divided by (nu0 + n) * d = 603 * 2 -> 1.7e-4
    assert hi - lo == pytest.approx(0.1 * 2 / (603 * 2), rel=1e-6)


# ---------------------------------------------------------------------------
# 3. unlabeled_weight normalisation
# ---------------------------------------------------------------------------
def semi_labels(y, frac=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    lab = y.clone()
    lab[torch.rand(y.shape[0], generator=g) > frac] = -1
    return lab


@pytest.mark.parametrize('cls', [GaussianMixture, StudentTMixture])
@pytest.mark.parametrize('w', [0.0, 0.3, 1.0, 2.0])
def test_weights_sum_to_one_for_any_unlabeled_weight(cls, w):
    X, y = blobs(n_per=60, d=2, seed=9)
    lab = semi_labels(y)
    m = cls(n_components=3, unlabeled_weight=w, random_state=0, max_iter=60)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X, labels=lab)
    assert m.weights_.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_vmf_weights_sum_to_one_with_unlabeled_weight():
    X, y = blobs(n_per=60, d=3, seed=10)
    X = X / X.norm(dim=1, keepdim=True)
    m = VonMisesFisherMixture(n_components=3, unlabeled_weight=0.3, random_state=0, max_iter=60)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X, labels=semi_labels(y))
    assert m.weights_.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_unlabeled_weight_zero_gives_labeled_only_weights():
    X, y = blobs(n_per=60, d=2, seed=11)
    lab = semi_labels(y, frac=0.5)
    keep = lab != -1
    a = GaussianMixture(n_components=3, unlabeled_weight=0.0, random_state=0, max_iter=80, tol=1e-9)
    b = GaussianMixture(n_components=3, random_state=0, max_iter=80, tol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a.fit(X, labels=lab)
        b.fit(X[keep], labels=lab[keep])
    assert torch.allclose(a.weights_, b.weights_, atol=1e-6)
    assert torch.allclose(a.means_, b.means_, atol=1e-6)


def test_unlabeled_weight_one_is_unchanged():
    """The default path is bit-identical: weights are the plain n_k / n."""
    X, y = blobs(n_per=60, d=2, seed=12)
    lab = semi_labels(y)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = GaussianMixture(n_components=3, random_state=0, max_iter=40).fit(X, labels=lab)
        b = GaussianMixture(n_components=3, unlabeled_weight=1.0, random_state=0, max_iter=40).fit(X, labels=lab)
    assert torch.equal(a.weights_, b.weights_)
    assert a.weights_.sum().item() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. initialisation robustness
# ---------------------------------------------------------------------------
def _old_kmeans(data, k, max_iter=1000, atol=1e-4, random_state=None):
    """The pre-fix implementation, for bit-identity on well-behaved inputs."""
    centroids = GMMInitializer.kpp(data, k, random_state=random_state)
    for _ in range(max_iter):
        distances = torch.cdist(data, centroids)
        labels = torch.argmin(distances, dim=1)
        new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(k)])
        if torch.allclose(centroids, new_centroids, atol=atol):
            break
        centroids = new_centroids
    return centroids


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_kmeans_is_bit_identical_when_no_cluster_is_empty(seed):
    X, _ = blobs(n_per=80, d=2, seed=seed)
    a = GMMInitializer.kmeans(X, 3, random_state=seed)
    b = _old_kmeans(X, 3, random_state=seed)
    assert torch.equal(a, b)


def test_kmeans_survives_empty_clusters():
    base = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=DT)
    X = base.repeat_interleave(20, dim=0)  # only 3 distinct points but 6 requested centres
    for seed in range(5):
        c = GMMInitializer.kmeans(X, 6, random_state=seed)
        assert c.shape == (6, 2)
        assert torch.isfinite(c).all()


@pytest.mark.parametrize('cov_type', ['full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical'])
def test_empirical_covariance_init_finite_with_singleton_clusters(cov_type):
    X, _ = blobs(n_per=10, d=2, seed=13)
    means = X.clone()  # every point is its own cluster: all singletons
    cov = GMMInitializer.init_covariances_empirical(X, means, cov_type, 1e-6)
    assert torch.isfinite(cov).all()
    if cov_type == 'full':
        assert (torch.linalg.eigvalsh(cov) > 0).all()


def test_empirical_covariance_init_is_unchanged_without_singletons():
    X, _ = blobs(n_per=60, d=2, seed=14)
    means = torch.tensor([[0.0, 5.0], [5.0, -5.0], [-5.0, 0.0]], dtype=DT)
    got = GMMInitializer.init_covariances_empirical(X, means, 'full', 1e-6)
    labels = torch.cdist(X, means).argmin(1)
    ref = torch.stack([torch.cov(X[labels == k].T) + 1e-6 * torch.eye(2, dtype=DT) for k in range(3)])
    assert torch.equal(got, ref)


@pytest.mark.parametrize('algorithm', ['em', 'sem'])
def test_many_components_on_few_points_gives_finite_parameters(algorithm):
    g = torch.Generator().manual_seed(0)
    centers = [(0.0, 0.0), (5.0, 5.0), (-5.0, 5.0)]
    X = torch.cat([torch.randn(60, 2, generator=g, dtype=DT) * 0.5 + torch.tensor(c, dtype=DT) for c in centers])
    m = GaussianMixture(n_components=25, algorithm=algorithm, random_state=0, max_iter=30)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X)
    assert torch.isfinite(m.covariances_).all()
    assert torch.isfinite(m.means_).all()
    assert torch.isfinite(m.weights_).all()


@pytest.mark.parametrize('cov_type', ['full', 'diag', 'spherical'])
def test_allocate_parameters_finite_with_singleton_clusters(cov_type):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(40, 2, generator=g, dtype=DT)
    m = GaussianMixture(n_components=30, covariance_type=cov_type, random_state=0)
    m._allocate_parameters(X)
    assert torch.isfinite(m.covariances_).all()


# ---------------------------------------------------------------------------
# 5. stale docstrings
# ---------------------------------------------------------------------------
def test_class_docstrings_are_not_stale():
    import re
    for cls in (StudentTMixture, VonMisesFisherMixture):
        doc = ' '.join(inspect.getdoc(cls).split())
        assert not re.search(r"labels=\.\.\.\)`\) are (intentionally )?Gaussian-only", doc), cls.__name__
        assert 'different fixed point' not in doc, cls.__name__
        assert 'components_per_class' in doc, cls.__name__
    assert 'same fit' in ' '.join(inspect.getdoc(StudentTMixture).split())


# ---------------------------------------------------------------------------
# 6. persistence round trips for every constructor argument
# ---------------------------------------------------------------------------
COMMON_NON_DEFAULT = dict(
    max_iter=77, tol=3e-5, reg_covar=2e-6, n_init=2, init_weights='random', random_state=5, warm_start=True,
    batch_size=64, convergence='aitken', sem_burn_in=3, sem_estimate='mean', components_per_class={0: 1, 1: 2},
    unlabeled_weight=0.4, label_confidence=0.8, noise_volume=12.5, noise_prior=0.1,
    verbose=True, verbose_interval=3, step_size=0.7,
)
# Two configurations per family so that every constructor argument takes a non-default value in at least one
# (``frozen_components`` is incompatible with shared dof / concentration, hence the split).
FAMILIES = {
    'gaussian': (GaussianMixture, [
        dict(n_components=3, n_features=2, covariance_type='diag', gpcm_tol=1e-4, gpcm_max_iter=7, init_means='kpp',
             init_covariances='global', algorithm='sem', cem=False, temperature=(0.5, 5), noise=True, missing='error',
             frozen_components=(1,)),
        dict(n_components=3, n_features=2, covariance_type='diag', weight_bounds=(0.1, 0.6),
             mean_bounds=(torch.tensor([0.0, -1.0]), torch.tensor([1.0, 2.0])), covariance_bounds=(0.01, 0.5),
             frozen_means=(0,), frozen_covariances=(2,),
             init_means=torch.tensor([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]]),
             init_covariances=torch.full((3, 2), 0.1)),
        dict(n_components=3, n_features=2, fixed_weights=torch.tensor([0.5, 0.3, 0.2])),
    ]),
    'student_t': (StudentTMixture, [
        dict(n_components=3, n_features=2, covariance_type='diag', dof=3.5, dof_init=3.0, dof_bounds=(0.5, 500.0),
             scale_update='ecme', init_means='kpp', init_covariances='global', algorithm='cem', temperature=(0.5, 5),
             noise=True, missing='error', frozen_components=(1,)),
        dict(n_components=3, n_features=2, covariance_type='tied_full', dof='shared', init_means='kpp'),
    ]),
    'vmf': (VonMisesFisherMixture, [
        dict(n_components=3, n_features=3, concentration='per_component', kappa_init=4.0, init_means='points',
             algorithm='cem', temperature=(0.5, 5), noise=True, missing='error', frozen_components=(1,)),
        dict(n_components=3, n_features=3, concentration='shared'),
    ]),
}
INIT_ONLY_EXEMPT = {'noise_init', 'weight_concentration_prior', 'mean_prior', 'mean_precision_prior',
                    'covariance_prior', 'degrees_of_freedom_prior', 'max_components', 'alpha', 'burn_in',
                    'weight_threshold', 'init_k'}  # Gibbs / prior options: covered by GaussianMixture's own tests


def _norm(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, (list, tuple)):
        return [_norm(a) for a in v]
    if isinstance(v, dict):
        return {k: _norm(a) for k, a in v.items()}
    return v


def _constructor_params(cls):
    return [p for p in inspect.signature(cls.__init__).parameters if p not in ('self', 'device', 'kwargs')]


def _configs(name):
    cls, specifics = FAMILIES[name]
    return cls, [{**COMMON_NON_DEFAULT, **spec} for spec in specifics]


@pytest.mark.parametrize('name', list(FAMILIES))
def test_every_constructor_argument_is_covered_by_the_roundtrip_test(name):
    cls, configs = _configs(name)
    covered = set().union(*[set(c) for c in configs])
    missing = set(_constructor_params(cls)) - covered - INIT_ONLY_EXEMPT
    assert not missing, f"add a non-default value for {sorted(missing)} to this test"


def _fitted_like(cls, kw):
    """A model with every constructor argument non-default and allocated (fitted-looking) state."""
    model = cls(**kw)
    model.noise_init = torch.tensor([True, False, False, True])
    d = model.n_features
    X, _ = blobs(n_per=15, d=min(d, 3), seed=15)
    if cls is VonMisesFisherMixture:
        X = X / X.norm(dim=1, keepdim=True)
    model._allocate_parameters(X.float())
    model.fitted_ = True
    model.n_components_ = model.n_components
    return model


@pytest.mark.parametrize('route', ['save_load', 'to_dict_load_state_dict'])
@pytest.mark.parametrize('name', list(FAMILIES))
def test_constructor_arguments_roundtrip(name, route):
    cls, configs = _configs(name)
    for kw in configs:
        model = _fitted_like(cls, kw)
        if route == 'save_load':
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'm.pt')
                model.save(path)
                loaded = cls.load(path)
        else:
            loaded = cls(n_components=3, n_features=model.n_features)
            loaded.load_state_dict(model.save_state_dict())
        for p in _constructor_params(cls):
            if p in ('init_means', 'init_weights') and not isinstance(getattr(model, p), (str, type(None))):
                continue
            assert _norm(getattr(loaded, p)) == _norm(getattr(model, p)), f"{cls.__name__}.{p} lost in {route}"


@pytest.mark.parametrize('name', ['gaussian', 'student_t', 'vmf'])
def test_fitted_semi_supervised_state_roundtrips(name):
    cls = FAMILIES[name][0]
    X, y = blobs(n_per=40, d=3, seed=16)
    if name == 'vmf':
        X = X / X.norm(dim=1, keepdim=True)
    lab = semi_labels(y, frac=0.5)
    kw = dict(n_components=3, random_state=0, max_iter=40, unlabeled_weight=0.5, label_confidence=0.9)
    if name == 'gaussian':
        kw['covariance_type'] = 'diag'
    model = cls(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X, labels=lab)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'm.pt')
        model.save(path)
        loaded = cls.load(path)
    assert loaded.unlabeled_weight == 0.5 and loaded.label_confidence == 0.9
    assert torch.equal(loaded.classes_, model.classes_)
    assert torch.equal(loaded.class_of_component_, model.class_of_component_)
    assert torch.equal(loaded.predict_class(X), model.predict_class(X))


# ---------------------------------------------------------------------------
# 7. E-step weight guard
# ---------------------------------------------------------------------------
def test_estep_weight_guard_is_a_tiny_perturbation():
    """``log(w + 1e-20)`` is kept: replacing it by ``log(clamp(w, 1e-20))`` is not bit-identical in float64 for small
    w (the guard shifts w by ~1e-20 relative to w), so it would change existing numerical results in the last
    digits. The perturbation is far below any statistical tolerance for realistic weights."""
    w = torch.tensor([1e-3, 1e-6], dtype=DT)
    assert torch.allclose(torch.log(w + 1e-20), torch.log(w.clamp(min=1e-20)), rtol=1e-12)
    tiny = torch.tensor([1e-17], dtype=DT)
    assert not torch.equal(torch.log(tiny + 1e-20), torch.log(tiny))
