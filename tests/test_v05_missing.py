"""v0.5 item 1.2: missing values (``missing='em'``) for GaussianMixture and StudentTMixture.

See docs/superpowers/specs/2026-10-15-v05-missing-values-design.md.
"""
import itertools
import warnings

import numpy as np
import pytest
import torch
from scipy import stats as sps

from tgmm import GaussianMixture, StudentTMixture, VonMisesFisherMixture
from tgmm.metrics import ClusteringMetrics

DT = torch.float64


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------
def make_blobs(n_per=2000, seed=0, shared_cov=False):
    g = torch.Generator().manual_seed(seed)
    # every coordinate separates the three clusters, so any single observed entry identifies the cluster
    centers = torch.tensor([[0.0, 6.0, -6.0], [6.0, -6.0, 0.0], [-6.0, 0.0, 6.0]], dtype=DT)
    base = torch.tensor([[1.0, 0.4, 0.2], [0.4, 0.8, 0.1], [0.2, 0.1, 0.6]], dtype=DT)
    covs = [base] * 3 if shared_cov else [base, 0.7 * base + 0.2 * torch.eye(3, dtype=DT), 1.3 * base]
    X, y = [], []
    for k in range(3):
        L = torch.linalg.cholesky(covs[k])
        X.append(torch.randn(n_per, 3, generator=g, dtype=DT) @ L.T + centers[k])
        y.append(torch.full((n_per,), k))
    return torch.cat(X), torch.cat(y)


def mcar(X, frac=0.2, seed=1):
    """Set entries to NaN completely at random, keeping at least one observed entry per row."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(X.shape, generator=g) < frac
    all_missing = mask.all(dim=1)
    for i in torch.nonzero(all_missing).flatten().tolist():
        mask[i, int(torch.randint(X.shape[1], (1,), generator=g))] = False
    Xn = X.clone()
    Xn[mask] = float('nan')
    return Xn, mask


def ari(y, labels):
    return ClusteringMetrics.adjusted_rand_score(y, labels)


def match_means(A, B):
    """Max abs difference between two (K, d) mean sets under the best permutation."""
    best = None
    for perm in itertools.permutations(range(A.shape[0])):
        err = (A[list(perm)] - B).abs().max().item()
        best = err if best is None else min(best, err)
    return best


def quiet_fit(model, X, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.fit(X, **kw)


def set_fixed(model, weights, means, covs, dof=None):
    model.weights_ = torch.as_tensor(weights, dtype=DT)
    model.means_ = torch.as_tensor(means, dtype=DT)
    model.covariances_ = torch.as_tensor(covs, dtype=DT)
    if dof is not None:
        model.dof_ = torch.as_tensor(dof, dtype=DT)
    model.n_features = model.means_.shape[1]
    model.dtype = DT
    model.fitted_ = True
    model.converged_ = True
    model.n_components_ = model.means_.shape[0]


# ---------------------------------------------------------------------------
# numpy reference implementations (K = 1)
# ---------------------------------------------------------------------------
def ref_conditional(row, mu, S):
    o = ~np.isnan(row)
    m = ~o
    xt = row.copy()
    C = np.zeros((len(row), len(row)))
    delta = (row[o] - mu[o]) @ np.linalg.solve(S[np.ix_(o, o)], row[o] - mu[o])
    if m.any():
        A = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
        xt[m] = mu[m] + A @ (row[o] - mu[o])
        C[np.ix_(m, m)] = S[np.ix_(m, m)] - A @ S[np.ix_(o, m)]
    return xt, C, delta, int(o.sum())


def ref_gaussian_em_step(X, mu, S):
    n, d = X.shape
    sx, sxx = np.zeros(d), np.zeros((d, d))
    for row in X:
        xt, C, _, _ = ref_conditional(row, mu, S)
        sx += xt
        sxx += np.outer(xt, xt) + C
    mu_new = sx / n
    return mu_new, sxx / n - np.outer(mu_new, mu_new)


def ref_t_ecm_step(X, mu, S, nu):
    n, d = X.shape
    m_sum, s1, rows = 0.0, np.zeros(d), []
    for row in X:
        xt, C, delta, p = ref_conditional(row, mu, S)
        u = (nu + p) / (nu + delta)
        rows.append((xt, C, u))
        m_sum += u
        s1 += u * xt
    mu_new = s1 / m_sum
    S_new = sum(u * np.outer(xt - mu_new, xt - mu_new) + C for xt, C, u in rows) / n
    return mu_new, S_new


@pytest.fixture(scope='module')
def small_data():
    X, y = make_blobs(n_per=60, seed=3)
    Xn, mask = mcar(X, 0.25, seed=4)
    return X, y, Xn, mask


# ---------------------------------------------------------------------------
# 'error' mode and the no-NaN identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('cls', [GaussianMixture, StudentTMixture])
def test_error_mode_rejects_nan_everywhere(cls, small_data):
    X, y, Xn, mask = small_data
    model = cls(n_components=3, random_state=0)
    with pytest.raises(ValueError, match="missing='em'"):
        model.fit(Xn)
    quiet_fit(model, X)
    for fn in (model.predict, model.predict_proba, model.score_samples, model.score):
        with pytest.raises(ValueError, match="missing='em'"):
            fn(Xn)
    with pytest.raises(ValueError, match="missing='em'"):
        model.bic(Xn)


def test_vmf_error_mode_validates_nan_and_rejects_em():
    X = torch.randn(60, 4, dtype=DT)
    X = X / X.norm(dim=1, keepdim=True)
    Xn = X.clone()
    Xn[3, 1] = float('nan')
    with pytest.raises(ValueError, match="NaN"):
        VonMisesFisherMixture(n_components=2, random_state=0).fit(Xn)
    with pytest.raises(ValueError, match="missing"):
        VonMisesFisherMixture(n_components=2, missing='em')


def test_invalid_missing_value():
    with pytest.raises(ValueError, match="missing"):
        GaussianMixture(n_components=2, missing='ignore')
    with pytest.raises(ValueError, match="missing"):
        StudentTMixture(n_components=2, missing='drop')


@pytest.mark.parametrize('covariance_type', [
    'full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical', 'EEE', 'EVE', 'VVE', 'VEV', 'EEV', 'EVV',
])
@pytest.mark.parametrize('algorithm', ['em', 'cem'])
def test_gaussian_no_nan_identical_to_error_mode(covariance_type, algorithm):
    X, _ = make_blobs(n_per=80, seed=5)
    kw = dict(n_components=3, covariance_type=covariance_type, algorithm=algorithm, random_state=0, max_iter=40)
    a = quiet_fit(GaussianMixture(missing='error', **kw), X)
    b = quiet_fit(GaussianMixture(missing='em', **kw), X)
    assert torch.equal(a.weights_, b.weights_)
    assert torch.equal(a.means_, b.means_)
    assert torch.equal(a.covariances_, b.covariances_)
    assert a.lower_bound_ == b.lower_bound_


@pytest.mark.parametrize('covariance_type', ['full', 'diag', 'spherical', 'tied_full'])
def test_student_t_no_nan_identical_to_error_mode(covariance_type):
    X, _ = make_blobs(n_per=80, seed=6)
    kw = dict(n_components=3, covariance_type=covariance_type, random_state=0, max_iter=40)
    a = quiet_fit(StudentTMixture(missing='error', **kw), X)
    b = quiet_fit(StudentTMixture(missing='em', **kw), X)
    assert torch.equal(a.means_, b.means_)
    assert torch.equal(a.covariances_, b.covariances_)
    assert torch.equal(a.dof_, b.dof_)
    assert a.lower_bound_ == b.lower_bound_


# ---------------------------------------------------------------------------
# closed-form checks, K = 1
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def fixed_params():
    rng = np.random.RandomState(0)
    A = rng.randn(4, 4)
    S = A @ A.T + 4 * np.eye(4)
    mu = rng.randn(4)
    return mu, S


def _incomplete_rows(mu, S, n=30, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.multivariate_normal(mu, S, size=n)
    mask = rng.rand(n, 4) < 0.35
    mask[mask.all(axis=1), 0] = False
    X[mask] = np.nan
    return X


def test_gaussian_score_samples_and_impute_match_closed_forms(fixed_params):
    mu, S = fixed_params
    X = _incomplete_rows(mu, S)
    model = GaussianMixture(n_components=1, covariance_type='full', missing='em')
    set_fixed(model, [1.0], mu[None], S[None])
    Xt = torch.tensor(X, dtype=DT)

    ll = model.score_samples(Xt).numpy()
    expected = [sps.multivariate_normal(mu[~np.isnan(r)], S[np.ix_(~np.isnan(r), ~np.isnan(r))]).logpdf(r[~np.isnan(r)])
                for r in X]
    np.testing.assert_allclose(ll, expected, rtol=1e-10, atol=1e-10)

    imputed = model.impute(Xt).numpy()
    for r, got in zip(X, imputed):
        xt, _, _, _ = ref_conditional(r, mu, S)
        np.testing.assert_allclose(got, xt, rtol=1e-10, atol=1e-10)
    assert not np.isnan(imputed).any()
    np.testing.assert_array_equal(imputed[~np.isnan(X)], X[~np.isnan(X)])


@pytest.mark.parametrize('covariance_type', ['full', 'tied_full', 'EEE'])
def test_gaussian_k1_em_matches_numpy_reference(covariance_type, fixed_params):
    mu, S = fixed_params
    X = _incomplete_rows(mu, S, n=80, seed=2)
    mu0, S0 = np.zeros(4), 3 * np.eye(4)
    model = GaussianMixture(
        n_components=1, covariance_type=covariance_type, missing='em', reg_covar=0.0, max_iter=3, tol=1e-300,
        init_means=torch.tensor(mu0[None], dtype=DT),
        init_covariances=torch.tensor(S0 if covariance_type in ('tied_full', 'EEE') else S0[None], dtype=DT),
    )
    quiet_fit(model, torch.tensor(X, dtype=DT))
    m, C = mu0, S0
    for _ in range(3):
        m, C = ref_gaussian_em_step(X, m, C)
    np.testing.assert_allclose(model.means_[0].numpy(), m, rtol=1e-8, atol=1e-10)
    got_cov = model.covariances_.numpy()
    got_cov = got_cov if got_cov.ndim == 2 else got_cov[0]
    np.testing.assert_allclose(got_cov, C, rtol=1e-8, atol=1e-10)


def test_gaussian_diag_k1_matches_marginal_formula(fixed_params):
    mu, _ = fixed_params
    var = np.array([1.0, 2.0, 0.5, 3.0])
    X = _incomplete_rows(mu, np.diag(var), n=40, seed=3)
    model = GaussianMixture(n_components=1, covariance_type='diag', missing='em')
    set_fixed(model, [1.0], mu[None], var[None])
    ll = model.score_samples(torch.tensor(X, dtype=DT)).numpy()
    expected = [sps.multivariate_normal(mu[~np.isnan(r)], np.diag(var[~np.isnan(r)])).logpdf(r[~np.isnan(r)]) for r in X]
    np.testing.assert_allclose(ll, expected, rtol=1e-10)
    imputed = model.impute(torch.tensor(X, dtype=DT)).numpy()
    miss = np.isnan(X)
    np.testing.assert_allclose(imputed[miss], np.broadcast_to(mu, X.shape)[miss], rtol=1e-12)


def test_student_t_score_samples_and_impute_match_closed_forms(fixed_params):
    mu, S = fixed_params
    X = _incomplete_rows(mu, S)
    nu = 4.0
    model = StudentTMixture(n_components=1, covariance_type='full', dof=nu, missing='em')
    set_fixed(model, [1.0], mu[None], S[None], dof=[nu])
    Xt = torch.tensor(X, dtype=DT)
    ll = model.score_samples(Xt).numpy()
    expected = [sps.multivariate_t(mu[~np.isnan(r)], S[np.ix_(~np.isnan(r), ~np.isnan(r))], df=nu).logpdf(r[~np.isnan(r)])
                for r in X]
    np.testing.assert_allclose(ll, expected, rtol=1e-9, atol=1e-10)
    imputed = model.impute(Xt).numpy()
    for r, got in zip(X, imputed):
        xt, _, _, _ = ref_conditional(r, mu, S)
        np.testing.assert_allclose(got, xt, rtol=1e-10, atol=1e-10)
    # observed-dimension weights: u = (nu + p) / (nu + delta_o)
    w = model.sample_weights(Xt).numpy()
    for r, got in zip(X, w):
        _, _, delta, p = ref_conditional(r, mu, S)
        np.testing.assert_allclose(got, (nu + p) / (nu + delta), rtol=1e-10)


def test_student_t_k1_ecm_matches_numpy_reference(fixed_params):
    mu, S = fixed_params
    X = _incomplete_rows(mu, S, n=80, seed=4)
    nu = 5.0
    mu0, S0 = np.zeros(4), 3 * np.eye(4)
    model = StudentTMixture(
        n_components=1, covariance_type='full', dof=nu, missing='em', reg_covar=0.0, max_iter=3, tol=1e-300,
        init_means=torch.tensor(mu0[None], dtype=DT), init_covariances=torch.tensor(S0[None], dtype=DT),
    )
    quiet_fit(model, torch.tensor(X, dtype=DT))
    m, C = mu0, S0
    for _ in range(3):
        m, C = ref_t_ecm_step(X, m, C, nu)
    np.testing.assert_allclose(model.means_[0].numpy(), m, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(model.covariances_[0].numpy(), C, rtol=1e-8, atol=1e-10)


def test_predict_proba_uses_observed_marginals(small_data):
    X, y, Xn, mask = small_data
    model = quiet_fit(GaussianMixture(n_components=3, missing='em', random_state=0, max_iter=100), Xn)
    proba = model.predict_proba(Xn).detach().numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    for i in range(0, Xn.shape[0], 17):
        o = ~mask[i].numpy()
        logp = np.array([
            np.log(model.weights_[k].item()) + sps.multivariate_normal(
                model.means_[k].numpy()[o], model.covariances_[k].numpy()[np.ix_(o, o)]).logpdf(Xn[i].numpy()[o])
            for k in range(3)
        ])
        expected = np.exp(logp - logp.max())
        np.testing.assert_allclose(proba[i], expected / expected.sum(), rtol=1e-8, atol=1e-10)
    assert model.predict(Xn).shape == (Xn.shape[0],)


# ---------------------------------------------------------------------------
# recovery under 20% MCAR
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def mcar_data():
    X, y = make_blobs(n_per=2000, seed=0)
    Xn, mask = mcar(X, 0.2, seed=1)
    return X, y, Xn, mask


def _monotone(model, tol=1e-7):
    hist = model.log_likelihood_history_
    return (hist[1:] - hist[:-1]).min().item() > -tol


def test_gaussian_mcar_recovery_full(mcar_data):
    X, y, Xn, mask = mcar_data
    complete = quiet_fit(GaussianMixture(n_components=3, random_state=0, max_iter=300, tol=1e-8), X)
    incomplete = quiet_fit(GaussianMixture(n_components=3, missing='em', random_state=0, max_iter=300, tol=1e-8), Xn)
    assert match_means(incomplete.means_, complete.means_) < 0.05
    assert ari(y, incomplete.predict(Xn)) > 0.95
    assert _monotone(incomplete)
    # imputation error is far below the data scale
    imputed = incomplete.impute(Xn)
    assert torch.sqrt(((imputed - X)[mask]).pow(2).mean()).item() < 1.2


@pytest.mark.parametrize('covariance_type', ['diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical', 'EEE', 'EVE', 'VEV'])
def test_gaussian_mcar_each_covariance_type(covariance_type):
    X, y = make_blobs(n_per=300, seed=2, shared_cov=True)
    Xn, _ = mcar(X, 0.2, seed=3)
    model = quiet_fit(GaussianMixture(
        n_components=3, covariance_type=covariance_type, missing='em', random_state=0, max_iter=200, tol=1e-8), Xn)
    assert torch.isfinite(model.means_).all() and torch.isfinite(model.covariances_).all()
    assert ari(y, model.predict(Xn)) > 0.95
    assert _monotone(model)


def test_student_t_mcar_recovery_and_monotone():
    X, y = make_blobs(n_per=1000, seed=0)
    Xn, mask = mcar(X, 0.2, seed=1)
    complete = quiet_fit(StudentTMixture(n_components=3, random_state=0, max_iter=150, tol=1e-6), X)
    for dof in ('per_component', 8.0):
        incomplete = quiet_fit(StudentTMixture(
            n_components=3, dof=dof, missing='em', random_state=0, max_iter=150, tol=1e-6), Xn)
        assert match_means(incomplete.means_, complete.means_) < 0.06
        assert ari(y, incomplete.predict(Xn)) > 0.95
        assert _monotone(incomplete)
    # the t recovers genuinely heavy tails from incomplete data
    g = torch.Generator().manual_seed(9)
    nu_true = 3.0
    z = torch.randn(3000, 3, generator=g, dtype=DT)
    w = torch._standard_gamma(torch.full((3000, 1), nu_true / 2, dtype=DT), generator=g) * (2.0 / nu_true)
    heavy_n, _ = mcar(z / w.sqrt(), 0.2, seed=10)
    fit = quiet_fit(StudentTMixture(n_components=1, missing='em', random_state=0, max_iter=200, tol=1e-7), heavy_n)
    assert 2.4 < fit.dof_.item() < 3.8


@pytest.mark.parametrize('covariance_type', ['diag', 'spherical', 'tied_full'])
def test_student_t_mcar_each_covariance_type(covariance_type):
    X, y = make_blobs(n_per=300, seed=2, shared_cov=True)
    Xn, _ = mcar(X, 0.2, seed=3)
    model = quiet_fit(StudentTMixture(
        n_components=3, covariance_type=covariance_type, missing='em', random_state=0, max_iter=200, tol=1e-8), Xn)
    assert ari(y, model.predict(Xn)) > 0.95
    assert _monotone(model)


# ---------------------------------------------------------------------------
# algorithms, batch_size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('algorithm', ['em', 'cem', 'sem', 'saem'])
@pytest.mark.parametrize('cls', [GaussianMixture, StudentTMixture])
def test_algorithms_with_missing(algorithm, cls):
    X, y = make_blobs(n_per=250, seed=7)
    Xn, _ = mcar(X, 0.2, seed=8)
    model = quiet_fit(cls(n_components=3, algorithm=algorithm, missing='em', random_state=0, max_iter=60), Xn)
    assert torch.isfinite(model.means_).all()
    assert ari(y, model.predict(Xn)) > 0.9
    assert torch.isfinite(torch.as_tensor(model.score(Xn)))


@pytest.mark.parametrize('covariance_type', ['full', 'diag', 'spherical', 'tied_full', 'EVE'])
def test_gaussian_batch_size_equals_unchunked(covariance_type):
    X, _ = make_blobs(n_per=200, seed=11)
    Xn, _ = mcar(X, 0.2, seed=12)
    kw = dict(n_components=3, covariance_type=covariance_type, missing='em', random_state=0, max_iter=25, tol=1e-12)
    a = quiet_fit(GaussianMixture(**kw), Xn)
    b = quiet_fit(GaussianMixture(batch_size=97, **kw), Xn)
    torch.testing.assert_close(a.means_, b.means_, rtol=0, atol=1e-8)
    torch.testing.assert_close(a.covariances_, b.covariances_, rtol=0, atol=1e-8)
    torch.testing.assert_close(a.weights_, b.weights_, rtol=0, atol=1e-8)


def test_student_t_batch_size_equals_unchunked():
    X, _ = make_blobs(n_per=200, seed=11)
    Xn, _ = mcar(X, 0.2, seed=12)
    kw = dict(n_components=3, missing='em', random_state=0, max_iter=25, tol=1e-12)
    a = quiet_fit(StudentTMixture(**kw), Xn)
    b = quiet_fit(StudentTMixture(batch_size=97, **kw), Xn)
    torch.testing.assert_close(a.means_, b.means_, rtol=0, atol=1e-8)
    torch.testing.assert_close(a.covariances_, b.covariances_, rtol=0, atol=1e-8)
    torch.testing.assert_close(a.dof_, b.dof_, rtol=0, atol=1e-8)


def test_information_criteria_and_warm_start_with_missing(small_data):
    X, y, Xn, mask = small_data
    model = quiet_fit(GaussianMixture(n_components=3, missing='em', random_state=0, n_init=2, max_iter=50), Xn)
    assert np.isfinite(model.bic(Xn)) and np.isfinite(model.aic(Xn)) and np.isfinite(model.icl(Xn))
    model.warm_start = True
    quiet_fit(model, Xn, max_iter=5)
    assert torch.isfinite(model.means_).all()


# ---------------------------------------------------------------------------
# unsupported combinations and input validation
# ---------------------------------------------------------------------------
def test_noise_with_missing_not_supported():
    with pytest.raises(NotImplementedError, match='noise'):
        GaussianMixture(n_components=2, missing='em', noise=True)
    with pytest.raises(NotImplementedError, match='noise'):
        StudentTMixture(n_components=2, missing='em', noise=True)


def test_online_em_and_partial_fit_not_supported(small_data):
    X, y, Xn, mask = small_data
    with pytest.raises(NotImplementedError, match='step_size'):
        GaussianMixture(n_components=3, missing='em', batch_size=32, step_size='auto')
    model = quiet_fit(GaussianMixture(n_components=3, missing='em', random_state=0, max_iter=10), Xn)
    with pytest.raises(NotImplementedError, match='partial_fit'):
        model.partial_fit(Xn)


def test_gibbs_and_labels_not_supported(small_data):
    X, y, Xn, mask = small_data
    with pytest.raises(NotImplementedError, match='Gibbs'):
        GaussianMixture(n_components=None, missing='em')
    with pytest.raises(NotImplementedError, match='labels'):
        GaussianMixture(n_components=3, missing='em').fit(Xn, labels=y)
    with pytest.raises(NotImplementedError, match='labels'):
        StudentTMixture(n_components=3, missing='em').fit(Xn, labels=y)


def test_student_t_ecme_with_missing_not_supported():
    with pytest.raises(NotImplementedError, match='ecme'):
        StudentTMixture(n_components=2, missing='em', scale_update='ecme')


def test_all_missing_row_and_column_rejected():
    X, _ = make_blobs(n_per=30, seed=1)
    row = X.clone()
    row[5] = float('nan')
    with pytest.raises(ValueError, match='all missing'):
        GaussianMixture(n_components=3, missing='em').fit(row)
    col = X.clone()
    col[:, 1] = float('nan')
    with pytest.raises(ValueError, match='column'):
        GaussianMixture(n_components=3, missing='em').fit(col)


def test_many_patterns_warns():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(3000, 12, generator=g, dtype=DT)
    mask = torch.rand(X.shape, generator=g) < 0.5
    mask[mask.all(dim=1), 0] = False
    X[mask] = float('nan')
    with pytest.warns(UserWarning, match='patterns'):
        GaussianMixture(n_components=2, covariance_type='diag', missing='em', max_iter=1, random_state=0).fit(X)


def test_impute_requires_em_mode_and_handles_complete_data(small_data):
    X, y, Xn, mask = small_data
    model = quiet_fit(GaussianMixture(n_components=3, missing='em', random_state=0, max_iter=30), Xn)
    out = model.impute(X)
    torch.testing.assert_close(out, X)
    assert model.impute(Xn).shape == Xn.shape
    with pytest.raises(RuntimeError, match="missing='em'"):
        quiet_fit(GaussianMixture(n_components=3, random_state=0), X).impute(X)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('cls', [GaussianMixture, StudentTMixture])
def test_save_load_and_state_dict_keep_missing(cls, small_data, tmp_path):
    X, y, Xn, mask = small_data
    model = quiet_fit(cls(n_components=3, missing='em', random_state=0, max_iter=30), Xn)
    path = str(tmp_path / 'm.pt')
    model.save(path)
    loaded = cls.load(path)
    assert loaded.missing == 'em'
    torch.testing.assert_close(loaded.score_samples(Xn), model.score_samples(Xn))
    fresh = cls(n_components=3, n_features=3)
    fresh.load_state_dict(model.save_state_dict())
    assert fresh.missing == 'em'


# ---------------------------------------------------------------------------
# interactions with other features
# ---------------------------------------------------------------------------
def test_map_priors_work_with_missing(small_data):
    X, y, Xn, mask = small_data
    model = quiet_fit(GaussianMixture(
        n_components=3, n_features=3, missing='em', random_state=0, max_iter=60,
        mean_prior=torch.zeros(3, 3, dtype=DT), mean_precision_prior=0.1,
        covariance_prior=torch.eye(3, dtype=DT).unsqueeze(0).repeat(3, 1, 1), degrees_of_freedom_prior=5.0,
    ), Xn)
    assert torch.isfinite(model.means_).all() and torch.isfinite(model.covariances_).all()
    assert ari(y, model.predict(Xn)) > 0.9


def test_annealing_with_missing(small_data):
    X, y, Xn, mask = small_data
    model = quiet_fit(GaussianMixture(
        n_components=3, missing='em', random_state=0, max_iter=60, temperature=(0.3, 10)), Xn)
    assert ari(y, model.predict(Xn)) > 0.9


def test_sem_reseeding_never_sees_nan(small_data):
    X, y, Xn, mask = small_data
    # K=5 > 3 true clusters: mean-imputed rows form small spurious k-means clusters that SEM empties and re-seeds
    model = quiet_fit(GaussianMixture(
        n_components=5, algorithm='sem', missing='em', random_state=0, max_iter=25), Xn)
    assert torch.isfinite(model.means_).all() and torch.isfinite(model.covariances_).all()
    assert model.n_empty_components_ > 0  # the re-seeding path (which reads X) actually ran


def test_n_init_and_predict_on_new_incomplete_rows(small_data):
    X, y, Xn, mask = small_data
    model = quiet_fit(GaussianMixture(n_components=3, missing='em', random_state=0, n_init=3, max_iter=60), Xn)
    new, _ = mcar(make_blobs(n_per=10, seed=21)[0], 0.3, seed=22)
    labels = model.predict(new)
    assert labels.shape == (30,)
    assert model.predict_proba(new).shape == (30, 3)
