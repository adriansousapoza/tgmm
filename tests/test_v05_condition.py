"""v0.5 item 1.3: GaussianMixture.marginal / condition, ConditionalGaussianMixture, StudentTMixture.marginal."""
import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t

from tgmm import ConditionalGaussianMixture, GaussianMixture, StudentTMixture

CLASSIC = ['full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical']
GPCM_NEW = ['VEI', 'EVI', 'VEE', 'EVE', 'EEV', 'VVE', 'VEV', 'EVV']
ALL_TYPES = CLASSIC + GPCM_NEW


def _data(n=450, d=4, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(K, d, d, generator=g, dtype=torch.float64) * 0.5
    centers = torch.randn(K, d, generator=g, dtype=torch.float64) * 3.0
    parts = [centers[k] + torch.randn(n // K, d, generator=g, dtype=torch.float64) @ A[k] for k in range(K)]
    return torch.cat(parts)


def _fit(cov_type, d=4, K=3, **kw):
    X = _data(d=d, K=K)
    return GaussianMixture(n_components=K, covariance_type=cov_type, random_state=0, max_iter=60,
                           device='cpu', **kw).fit(X), X


def _dense(g):
    """(K, d, d) covariance of a fitted model, built independently of the library code under test."""
    K, d = g.weights_.shape[0], g.means_.shape[1]
    c, t = g.covariances_, g.covariance_type
    eye = torch.eye(d, dtype=c.dtype)
    if t == 'diag':
        return torch.diag_embed(c)
    if t == 'spherical':
        return c.view(K, 1, 1) * eye
    if t == 'tied_full':
        return c.unsqueeze(0).expand(K, d, d)
    if t == 'tied_diag':
        return torch.diag_embed(c).unsqueeze(0).expand(K, d, d)
    if t == 'tied_spherical':
        return (c * eye).unsqueeze(0).expand(K, d, d)
    return c


def _ref_marginal_loglik(g, Xm, dims):
    S, mu, w = _dense(g).numpy(), g.means_.numpy(), g.weights_.numpy()
    cols = [np.log(w[k]) + multivariate_normal(mu[k][dims], S[k][np.ix_(dims, dims)]).logpdf(Xm.numpy())
            for k in range(len(w))]
    return logsumexp(np.stack(cols), axis=0)


# ---------------------------------------------------------------------------
# marginal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('cov_type', ALL_TYPES)
def test_marginal_loglik_matches_scipy_for_every_covariance_type(cov_type):
    g, X = _fit(cov_type)
    dims = [2, 0]
    m = g.marginal(dims)
    assert isinstance(m, GaussianMixture)
    assert m.n_features == 2 and m.n_components == g.n_components
    assert m.fitted_ and m.converged_
    assert torch.equal(m.weights_, g.weights_)
    assert torch.equal(m.means_, g.means_[:, dims])
    ref = _ref_marginal_loglik(g, X[:, dims], dims)
    got = m.score_samples(X[:, dims]).detach().numpy()
    np.testing.assert_allclose(got, ref, atol=1e-8)


@pytest.mark.parametrize('cov_type', ALL_TYPES)
def test_marginal_covariance_type_rules(cov_type):
    g, _ = _fit(cov_type)
    m = g.marginal([1, 3])
    if cov_type in CLASSIC:
        assert m.covariance_type == cov_type          # diag / spherical / tied_* keep their type
    else:
        assert m.covariance_type == 'full'            # GPCM decomposition does not survive restriction
        assert m.volume_ is None and m.shape_ is None and m.orientation_ is None
    assert m.dtype == g.dtype


def test_marginal_is_usable_downstream():
    g, X = _fit('full')
    m = g.marginal([0, 1])
    samples, labels = m.sample(50)
    assert samples.shape == (50, 2)
    assert m.predict(X[:, [0, 1]]).shape == (X.shape[0],)
    assert m.bic(X[:, [0, 1]]) == pytest.approx(m.bic(X[:, [0, 1]]))


def test_marginal_single_dim_and_all_dims():
    g, X = _fit('full')
    one = g.marginal([3])
    assert one.n_features == 1 and one.covariances_.shape == (3, 1, 1)
    full = g.marginal([0, 1, 2, 3])
    np.testing.assert_allclose(full.score_samples(X).detach().numpy(),
                               g.score_samples(X).detach().numpy(), atol=1e-10)


@pytest.mark.parametrize('bad', [[], [0, 0], [4], [-1], [0, 9]])
def test_marginal_validation(bad):
    g, _ = _fit('full')
    with pytest.raises(ValueError):
        g.marginal(bad)


def test_marginal_requires_fitted_model():
    with pytest.raises(RuntimeError):
        GaussianMixture(n_components=2, n_features=3).marginal([0])


def test_marginal_and_condition_reject_noise_and_gibbs():
    noisy = GaussianMixture(n_components=2, n_features=3, noise=True)
    with pytest.raises(NotImplementedError, match='noise'):
        noisy.marginal([0])
    with pytest.raises(NotImplementedError, match='noise'):
        noisy.condition(torch.zeros(1, 1), [0])
    X = _data()
    mp, mpp, cp, dfp = GaussianMixture.suggest_priors(X, n_components=3)
    gibbs = GaussianMixture(n_components=None, mean_prior=mp, mean_precision_prior=mpp, covariance_prior=cp,
                            degrees_of_freedom_prior=dfp, max_components=5, max_iter=3, device='cpu')
    with pytest.raises(NotImplementedError, match='Gibbs'):
        gibbs.marginal([0])
    with pytest.raises(NotImplementedError, match='Gibbs'):
        gibbs.condition(torch.zeros(1, 1), [0])


# ---------------------------------------------------------------------------
# condition
# ---------------------------------------------------------------------------
def _ref_condition(g, x_obs, obs):
    """Independent NumPy loop reference (per sample, per component)."""
    d = g.means_.shape[1]
    mis = [j for j in range(d) if j not in obs]
    S, mu, pi = _dense(g).numpy(), g.means_.numpy(), g.weights_.numpy()
    K, n = len(pi), x_obs.shape[0]
    W, M = np.zeros((n, K)), np.zeros((n, K, len(mis)))
    C = np.zeros((K, len(mis), len(mis)))
    for k in range(K):
        Soo, Smo = S[k][np.ix_(obs, obs)], S[k][np.ix_(mis, obs)]
        A = Smo @ np.linalg.inv(Soo)
        C[k] = S[k][np.ix_(mis, mis)] - A @ Smo.T
        for i in range(n):
            M[i, k] = mu[k][mis] + A @ (x_obs[i] - mu[k][obs])
            W[i, k] = np.log(pi[k]) + multivariate_normal(mu[k][obs], Soo).logpdf(x_obs[i])
    W = np.exp(W - logsumexp(W, axis=1, keepdims=True))
    return W, M, C, mis


@pytest.mark.parametrize('cov_type', ALL_TYPES)
def test_condition_matches_numpy_reference(cov_type):
    g, X = _fit(cov_type)
    obs = [3, 1]
    cond = g.condition(X[:7][:, obs], obs)
    assert isinstance(cond, ConditionalGaussianMixture)
    W, M, C, mis = _ref_condition(g, X[:7][:, obs].numpy(), obs)
    assert cond.obs_dims == obs and cond.missing_dims == mis == [0, 2]
    np.testing.assert_allclose(cond.weights_.numpy(), W, atol=1e-9)
    np.testing.assert_allclose(cond.means_.numpy(), M, atol=1e-9)
    np.testing.assert_allclose(cond.covariances_.numpy(), C, atol=1e-9)
    assert cond.weights_.shape == (7, 3) and cond.means_.shape == (7, 3, 2) and cond.covariances_.shape == (3, 2, 2)
    np.testing.assert_allclose(cond.weights_.sum(1).numpy(), 1.0, atol=1e-12)


def test_k1_matches_bivariate_normal_conditional():
    rng = torch.Generator().manual_seed(3)
    Z = torch.randn(4000, 2, generator=rng, dtype=torch.float64)
    X = Z @ torch.tensor([[1.0, 0.8], [0.0, 0.6]], dtype=torch.float64) + torch.tensor([1.0, -2.0], dtype=torch.float64)
    g = GaussianMixture(n_components=1, covariance_type='full', device='cpu', reg_covar=0.0).fit(X)
    mu, S = g.means_[0], g.covariances_[0]
    x0 = torch.tensor([[0.3], [1.5], [-1.0]], dtype=torch.float64)
    cond = g.condition(x0, [0])
    expected_mean = mu[1] + S[0, 1] / S[0, 0] * (x0[:, 0] - mu[0])
    expected_var = S[1, 1] - S[0, 1] ** 2 / S[0, 0]
    torch.testing.assert_close(cond.mean()[:, 0], expected_mean, atol=1e-12, rtol=0)
    torch.testing.assert_close(cond.covariances_[0, 0, 0], expected_var, atol=1e-12, rtol=0)
    assert torch.equal(cond.weights_, torch.ones(3, 1, dtype=torch.float64))


def test_mean_is_weighted_component_mean():
    g, X = _fit('full')
    cond = g.condition(X[:20, :2], [0, 1])
    expected = (cond.weights_.unsqueeze(-1) * cond.means_).sum(1)
    torch.testing.assert_close(cond.mean(), expected)


def test_condition_commutes_with_marginalising_unobserved_dims():
    """Conditioning on dim 0 then integrating out dims 2, 3 equals marginalising to (0, 1) then conditioning on 0."""
    g, X = _fit('full')
    full = g.condition(X[:6, [0]], [0])                    # over dims (1, 2, 3)
    via_marginal = g.marginal([0, 1]).condition(X[:6, [0]], [0])   # over dim (1,)
    torch.testing.assert_close(full.weights_, via_marginal.weights_, atol=1e-12, rtol=0)
    torch.testing.assert_close(full.means_[:, :, :1], via_marginal.means_, atol=1e-12, rtol=0)
    torch.testing.assert_close(full.covariances_[:, :1, :1], via_marginal.covariances_, atol=1e-12, rtol=0)
    # density of dim 1 given dim 0: integrating dims 2, 3 out of the conditional mixture, row by row
    ys = torch.linspace(-4, 4, 9, dtype=torch.float64).unsqueeze(1)
    for i in range(6):
        a = full.as_gmm(i).marginal([0]).score_samples(ys)
        b = via_marginal.as_gmm(i).score_samples(ys)
        torch.testing.assert_close(a, b, atol=1e-10, rtol=0)


def test_gmr_recovers_noisy_sine():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2000, 1, generator=g, dtype=torch.float64) * 2 * np.pi
    y = torch.sin(x) + 0.1 * torch.randn(2000, 1, generator=g, dtype=torch.float64)
    gmm = GaussianMixture(n_components=8, covariance_type='full', random_state=0, n_init=3, max_iter=200,
                          device='cpu').fit(torch.cat([x, y], dim=1))
    grid = torch.linspace(0.3, 2 * np.pi - 0.3, 200, dtype=torch.float64).unsqueeze(1)
    pred = gmm.condition(grid, [0]).mean()
    rmse = torch.sqrt(((pred - torch.sin(grid)) ** 2).mean()).item()
    assert rmse < 0.1, rmse


def test_log_prob_integrates_to_one_and_matches_as_gmm():
    g, X = _fit('full')
    cond = g.condition(X[:5, [0, 1, 2]], [0, 1, 2])            # 1-D conditional over dim 3
    ys = torch.linspace(-60, 60, 24001, dtype=torch.float64)
    lp = torch.stack([cond.as_gmm(i).score_samples(ys.unsqueeze(1)) for i in range(5)])
    integral = torch.trapezoid(lp.exp(), ys, dim=1)
    torch.testing.assert_close(integral, torch.ones(5, dtype=torch.float64), atol=1e-6, rtol=0)
    # a single-row conditional broadcasts against many Y, so log_prob itself can be integrated too
    single = g.condition(X[:1, [0, 1, 2]], [0, 1, 2])
    lp_single = single.log_prob(ys.unsqueeze(1))
    assert lp_single.shape == (24001,)
    assert torch.trapezoid(lp_single.exp(), ys).item() == pytest.approx(1.0, abs=1e-6)
    # log_prob(Y) for one Y per row agrees with the per-row GaussianMixture
    torch.manual_seed(0)
    Y = cond.sample(1)[:, 0]          # plausible points: no component is numerically negligible
    lp_rows = cond.log_prob(Y)
    for i in range(5):
        assert lp_rows[i].item() == pytest.approx(cond.as_gmm(i).score_samples(Y[i:i + 1]).item(), abs=1e-10)


def test_log_prob_matches_scipy_mixture():
    g, X = _fit('full')
    obs = [0, 2]
    cond = g.condition(X[:4, obs], obs)
    Y = torch.randn(4, 2, dtype=torch.float64)
    W, M, C, mis = _ref_condition(g, X[:4, obs].numpy(), obs)
    ref = [logsumexp([np.log(W[i, k]) + multivariate_normal(M[i, k], C[k]).logpdf(Y[i].numpy())
                      for k in range(3)]) for i in range(4)]
    np.testing.assert_allclose(cond.log_prob(Y).numpy(), ref, atol=1e-9)


def test_sample_moments_match_mixture_moments():
    g, X = _fit('full')
    obs = [0, 1]
    cond = g.condition(X[:3, obs], obs)
    torch.manual_seed(0)
    S = cond.sample(40000)
    assert S.shape == (3, 40000, 2) and S.dtype == torch.float64
    mean = cond.mean()
    torch.testing.assert_close(S.mean(1), mean, atol=0.06, rtol=0.03)
    # analytic mixture covariance: sum_k w (C_k + (m_k - mean)(m_k - mean)^T)
    dm = cond.means_ - mean.unsqueeze(1)
    cov = (cond.weights_[:, :, None, None] * (cond.covariances_.unsqueeze(0) + dm.unsqueeze(-1) * dm.unsqueeze(-2))).sum(1)
    emp = torch.stack([torch.cov(S[i].T) for i in range(3)])
    torch.testing.assert_close(emp, cov, atol=0.1, rtol=0.06)


def test_sample_default_single_draw_shape():
    g, X = _fit('diag')
    cond = g.condition(X[:4, :2], [0, 1])
    assert cond.sample().shape == (4, 1, 2)


@pytest.mark.parametrize('obs, ok', [([0], True), ([0, 1, 2], True), ([], False), ([0, 0], False), ([4], False),
                                     ([-1], False), ([0, 1, 2, 3], False)])
def test_condition_validation(obs, ok):
    g, X = _fit('full')
    x = X[:3][:, obs] if obs and all(0 <= j < 4 for j in obs) else torch.zeros(3, max(len(obs), 1), dtype=torch.float64)
    if ok:
        g.condition(x, obs)
    else:
        with pytest.raises(ValueError):
            g.condition(x, obs)


def test_condition_shape_mismatch_and_1d_input():
    g, X = _fit('full')
    with pytest.raises(ValueError):
        g.condition(X[:3, :3], [0, 1])
    single = g.condition(X[0, [0, 1]], [0, 1])             # 1-D vector = one row
    assert single.weights_.shape == (1, 3)
    with pytest.raises(ValueError):
        single.log_prob(torch.zeros(1, 5, dtype=torch.float64))


def test_condition_float32_and_dtype_device_honoured():
    X = _data().float()
    g = GaussianMixture(n_components=3, covariance_type='full', random_state=0, max_iter=30, device='cpu').fit(X)
    cond = g.condition(X[:5, :2], [0, 1])
    assert cond.means_.dtype == torch.float32 and cond.weights_.dtype == torch.float32
    assert cond.mean().device == X.device
    assert g.marginal([0, 1]).means_.dtype == torch.float32


def test_condition_after_missing_em_fit():
    X = _data()
    Xn = X.clone()
    mask = torch.rand(Xn.shape, generator=torch.Generator().manual_seed(1)) < 0.1
    mask[:, 0] = False
    Xn[mask] = float('nan')
    g = GaussianMixture(n_components=3, covariance_type='full', missing='em', random_state=0, max_iter=60,
                        device='cpu').fit(Xn)
    cond = g.condition(X[:5, :2], [0, 1])
    assert cond.mean().shape == (5, 2)
    assert g.marginal([0, 1]).missing == 'em'


# ---------------------------------------------------------------------------
# Student-t
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('cov_type', ['full', 'diag', 'spherical', 'tied_full'])
def test_student_t_marginal_is_exact(cov_type):
    X = _data(d=3)
    t = StudentTMixture(n_components=3, covariance_type=cov_type, random_state=0, max_iter=40, device='cpu').fit(X)
    dims = [2, 0]
    m = t.marginal(dims)
    assert isinstance(m, StudentTMixture) and m.n_features == 2
    torch.testing.assert_close(m.dof_, t.dof_)
    d = X.shape[1]
    c = t.covariances_
    if cov_type == 'full':
        S = c
    elif cov_type == 'diag':
        S = torch.diag_embed(c)
    elif cov_type == 'spherical':
        S = c.view(-1, 1, 1) * torch.eye(d, dtype=c.dtype)
    else:
        S = c.unsqueeze(0).expand(3, d, d)
    Xm = X[:, dims]
    cols = [np.log(t.weights_[k].item())
            + multivariate_t(t.means_[k][dims].numpy(), S[k][np.ix_(dims, dims)].numpy(), df=t.dof_[k].item()).logpdf(Xm.numpy())
            for k in range(3)]
    ref = logsumexp(np.stack(cols), axis=0)
    np.testing.assert_allclose(m.score_samples(Xm).detach().numpy(), ref, atol=1e-8)


def test_student_t_condition_not_implemented_and_marginal_validation():
    X = _data(d=3)
    t = StudentTMixture(n_components=2, random_state=0, max_iter=20, device='cpu').fit(X)
    with pytest.raises(NotImplementedError, match='degrees of freedom'):
        t.condition(X[:3, :1], [0])
    with pytest.raises(ValueError):
        t.marginal([0, 0])
    with pytest.raises(NotImplementedError, match='noise'):
        StudentTMixture(n_components=2, n_features=3, noise=True).marginal([0])
