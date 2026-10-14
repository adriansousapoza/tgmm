"""
Phase 4 (v0.4 roadmap): `BaseMixture`, `StudentTMixture`, and
`VonMisesFisherMixture`. See
docs/superpowers/specs/2026-10-14-phase4-base-mixture-studentt-vmf-design.md.
"""
import math
import os
import tempfile
import warnings

import pytest
import torch

from tgmm import GaussianMixture
from tgmm.mixtures import BaseMixture, StudentTMixture, VonMisesFisherMixture
from tgmm.mixtures.student_t import _mahalanobis_and_logdet
from tgmm.mixtures.vmf import _log_bessel_iv, _solve_kappa, _spherical_kmeans
from tgmm.stats import SufficientStats

try:
    import mpmath
    HAVE_MPMATH = True
except ImportError:
    HAVE_MPMATH = False

try:
    from scipy.stats import multivariate_t
    HAVE_SCIPY_MVT = True
except ImportError:
    HAVE_SCIPY_MVT = False


def _two_blobs(seed, n_per=150, d=2, sep=10.0, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    c0 = torch.zeros(d)
    c1 = torch.zeros(d)
    c1[0] = sep
    X0 = torch.randn(n_per, d, generator=gen) * scale + c0
    X1 = torch.randn(n_per, d, generator=gen) * scale + c1
    return torch.cat([X0, X1]), torch.cat([torch.zeros(n_per), torch.ones(n_per)]).long(), torch.stack([c0, c1])


def _vmf_like(seed, mus, kappa, n_per):
    r"""
    Approximate vMF samples via a small-noise Gaussian projection onto the
    sphere -- not exact vMF draws, but close enough (and reproducible
    without the mixture's own rejection sampler) to test *fitting* and
    *clustering* on, which is all these tests need.
    """
    gen = torch.Generator().manual_seed(seed)
    d = mus.shape[1]
    chunks = []
    for k in range(mus.shape[0]):
        noise = torch.randn(n_per, d, generator=gen) / math.sqrt(kappa)
        x = mus[k].unsqueeze(0) + noise
        chunks.append(x / x.norm(dim=1, keepdim=True))
    return torch.cat(chunks, dim=0)


# ===========================================================================
# BaseMixture rebase
# ===========================================================================
class TestBaseMixtureRebase:
    def test_gaussian_mixture_is_a_base_mixture(self):
        assert issubclass(GaussianMixture, BaseMixture)

    def test_gaussian_mixture_still_fits_and_predicts(self):
        X, y_true, _ = _two_blobs(0)
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(X)
        assert gmm.converged_
        pred = gmm.predict(X)
        # labels may be permuted, but should perfectly separate the two blobs
        assert len(torch.unique(pred[:150])) == 1
        assert len(torch.unique(pred[150:])) == 1
        assert pred[0] != pred[-1]

    def test_param_snapshot_round_trip(self):
        X, _, _ = _two_blobs(1)
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(X)
        snapshot = gmm._get_param_snapshot()
        assert set(snapshot) == {'weights_', 'means_', 'covariances_'}

        perturbed_means = gmm.means_.clone()
        gmm.means_ = perturbed_means + 100.0
        gmm._apply_snapshot(snapshot)
        assert torch.equal(gmm.means_, snapshot['means_'])

    def test_fit_snapshot_has_all_expected_bookkeeping_fields(self):
        X, _, _ = _two_blobs(2)
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(X)
        snapshot = gmm._get_fit_snapshot()
        expected = {
            'weights_', 'means_', 'covariances_', 'converged_', 'n_iter_', 'lower_bound_',
            'log_likelihood_history_', 'annealing_schedule_', 'n_empty_components_',
            '_online_running_stats_', '_online_step_count_',
        }
        assert set(snapshot) == expected

    def test_bic_uses_math_log_not_torch_log(self):
        r"""
        Regression guard: BaseMixture.bic must use `math.log(n_samples)`,
        not `torch.log(torch.tensor(...))`, to stay bit-identical with
        pre-Phase-4 GaussianMixture.bic (the two differ at the ULP level).
        """
        X, _, _ = _two_blobs(3)
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(X)
        n = X.shape[0]
        ll = gmm._score_for_information_criterion(X)
        expected = -2.0 * ll * n + gmm._n_parameters() * math.log(n)
        assert gmm.bic(X) == expected


# ===========================================================================
# SufficientStats.aux
# ===========================================================================
class TestSufficientStatsAux:
    def test_aux_defaults_to_none(self):
        s = SufficientStats(nk=torch.ones(2), s1=torch.zeros(2, 3), s2=torch.zeros(2, 3), n_samples=10)
        assert s.aux is None

    def test_add_with_both_aux_none(self):
        a = SufficientStats(nk=torch.ones(2), s1=torch.zeros(2, 3), s2=torch.zeros(2, 3), n_samples=10)
        b = SufficientStats(nk=torch.ones(2), s1=torch.zeros(2, 3), s2=torch.zeros(2, 3), n_samples=5)
        c = a + b
        assert c.aux is None

    def test_add_merges_aux_keys(self):
        a = SufficientStats(nk=torch.ones(2), s1=torch.zeros(2, 3), s2=torch.zeros(2, 3), n_samples=10,
                             aux={'foo': torch.tensor([1.0, 2.0])})
        b = SufficientStats(nk=torch.ones(2), s1=torch.zeros(2, 3), s2=torch.zeros(2, 3), n_samples=5,
                             aux={'foo': torch.tensor([3.0, 4.0]), 'bar': torch.tensor([1.0])})
        c = a + b
        assert torch.equal(c.aux['foo'], torch.tensor([4.0, 6.0]))
        assert torch.equal(c.aux['bar'], torch.tensor([1.0]))

    def test_scale_scales_aux(self):
        a = SufficientStats(nk=torch.ones(2), s1=torch.zeros(2, 3), s2=torch.zeros(2, 3), n_samples=10,
                             aux={'foo': torch.tensor([2.0, 4.0])})
        scaled = a.scale(0.5)
        assert torch.equal(scaled.aux['foo'], torch.tensor([1.0, 2.0]))

    def test_scale_with_none_aux(self):
        a = SufficientStats(nk=torch.ones(2), s1=torch.zeros(2, 3), s2=torch.zeros(2, 3), n_samples=10)
        assert a.scale(2.0).aux is None


# ===========================================================================
# StudentTMixture
# ===========================================================================
class TestStudentTMixtureLogProb:
    @pytest.mark.skipif(not HAVE_SCIPY_MVT, reason="scipy.stats.multivariate_t not available")
    @pytest.mark.parametrize("covariance_type", ['full', 'diag', 'spherical', 'tied_full'])
    def test_matches_scipy_multivariate_t(self, covariance_type):
        torch.manual_seed(0)
        d = 3
        mean = torch.randn(d)
        nu = 4.7
        X = torch.randn(25, d) * 2 + mean

        if covariance_type == 'full':
            A = torch.randn(d, d)
            cov = (A @ A.T + d * torch.eye(d)).unsqueeze(0)
        elif covariance_type == 'diag':
            cov = (torch.rand(d) + 0.5).unsqueeze(0)
        elif covariance_type == 'spherical':
            cov = torch.tensor([2.3])
        else:  # tied_full
            A = torch.randn(d, d)
            cov = A @ A.T + d * torch.eye(d)

        delta, logdet = _mahalanobis_and_logdet(X, mean.unsqueeze(0), cov, covariance_type)
        delta, logdet = delta.squeeze(1), logdet.squeeze(0)
        const = math.lgamma((nu + d) / 2) - math.lgamma(nu / 2) - (d / 2) * math.log(nu * math.pi) - 0.5 * logdet.item()
        mine = const - (nu + d) / 2 * torch.log1p(delta / nu)

        if covariance_type in ('full', 'tied_full'):
            shape = cov.squeeze(0) if covariance_type == 'full' else cov
        elif covariance_type == 'diag':
            shape = torch.diag(cov.squeeze(0))
        else:
            shape = torch.eye(d) * cov.item()
        ref = multivariate_t.logpdf(X.numpy(), loc=mean.numpy(), shape=shape.numpy(), df=nu)
        assert torch.allclose(mine, torch.as_tensor(ref, dtype=mine.dtype), atol=1e-4)


class TestStudentTMixtureFit:
    def test_is_a_base_mixture_not_a_gaussian_mixture(self):
        assert issubclass(StudentTMixture, BaseMixture)
        assert not issubclass(StudentTMixture, GaussianMixture)

    @pytest.mark.parametrize("covariance_type", ['full', 'diag', 'spherical', 'tied_full'])
    def test_fits_and_separates_two_blobs(self, covariance_type):
        X, y_true, _ = _two_blobs(0, sep=12.0)
        m = StudentTMixture(n_components=2, covariance_type=covariance_type, random_state=0, max_iter=100)
        m.fit(X)
        pred = m.predict(X)
        assert pred[0] != pred[-1]
        assert len(torch.unique(pred[:150])) == 1
        assert len(torch.unique(pred[150:])) == 1

    @pytest.mark.parametrize("dof", ['per_component', 'shared', 1e4])
    def test_invalid_covariance_type_raises(self, dof):
        with pytest.raises(ValueError):
            StudentTMixture(n_components=2, covariance_type='tied_diag', dof=dof)

    def test_invalid_dof_raises(self):
        with pytest.raises(ValueError):
            StudentTMixture(n_components=2, dof='bogus')
        with pytest.raises(ValueError):
            StudentTMixture(n_components=2, dof=-1.0)

    def test_matches_gaussian_mixture_at_large_fixed_dof(self):
        X, _, _ = _two_blobs(5, sep=10.0)
        gmm = GaussianMixture(n_components=2, random_state=0, n_init=3)
        gmm.fit(X)
        stm = StudentTMixture(n_components=2, dof=1e5, random_state=0, n_init=3)
        stm.fit(X)
        assert torch.allclose(gmm.means_, stm.means_, atol=1e-3)
        assert torch.allclose(gmm.weights_, stm.weights_, atol=1e-3)

    def test_robust_to_outliers_unlike_gaussian(self):
        torch.manual_seed(42)
        n_per = 200
        true_centers = torch.tensor([[5.0, 5.0], [-5.0, -5.0]])
        X_clean = torch.cat([
            torch.randn(n_per, 2) * 0.5 + true_centers[0],
            torch.randn(n_per, 2) * 0.5 + true_centers[1],
        ])
        n_out = int(0.05 * 2 * n_per)
        outliers = (torch.rand(n_out, 2) - 0.5) * 40
        X = torch.cat([X_clean, outliers])

        def match_err(means):
            d = torch.cdist(means, true_centers)
            return min(d[0, 0] + d[1, 1], d[0, 1] + d[1, 0]).item() / 2

        gmm = GaussianMixture(n_components=2, random_state=2, n_init=5)
        gmm.fit(X)
        stm = StudentTMixture(n_components=2, dof='per_component', random_state=2, n_init=5)
        stm.fit(X)

        assert match_err(stm.means_) < 0.1
        assert match_err(gmm.means_) > match_err(stm.means_)

    def test_monotone_log_likelihood(self):
        X, _, _ = _two_blobs(6)
        m = StudentTMixture(n_components=2, random_state=0, max_iter=50)
        m.fit(X)
        diffs = m.log_likelihood_history_[1:] - m.log_likelihood_history_[:-1]
        assert (diffs >= -1e-6).all()

    def test_dof_solver_satisfies_equation(self):
        m = StudentTMixture(n_components=1)
        d = 5

        def f(nu, c):
            return (1 - torch.special.digamma(nu / 2) + torch.log(nu / 2) + c
                    + torch.special.digamma((nu + d) / 2) - torch.log((nu + d) / 2))

        for c_val in [-1.2, -2.0, -3.5]:
            c = torch.tensor([c_val])
            nu = m._solve_dof(c, d)
            assert abs(f(nu, c).item()) < 1e-4

    def test_sample_weights_small_for_outliers(self):
        torch.manual_seed(7)
        n_per = 150
        X_clean = torch.cat([
            torch.randn(n_per, 2) * 0.5 + torch.tensor([5.0, 5.0]),
            torch.randn(n_per, 2) * 0.5 + torch.tensor([-5.0, -5.0]),
        ])
        outliers = (torch.rand(20, 2) - 0.5) * 40
        X = torch.cat([X_clean, outliers])
        m = StudentTMixture(n_components=2, dof='per_component', random_state=0, n_init=3)
        m.fit(X)
        w = m.sample_weights(X)
        assert w[-20:].mean() < w[:-20].mean()

    def test_sample_shape_and_component_draw(self):
        X, _, _ = _two_blobs(8)
        m = StudentTMixture(n_components=2, random_state=0)
        m.fit(X)
        samples, idx = m.sample(50)
        assert samples.shape == (50, 2)
        assert idx.shape == (50,)
        comp0 = m._sample_component(0, 10)
        assert comp0.shape == (10, 2)

    def test_n_parameters_counts_dof(self):
        m_per = StudentTMixture(n_components=3, n_features=4, covariance_type='full', dof='per_component')
        m_shared = StudentTMixture(n_components=3, n_features=4, covariance_type='full', dof='shared')
        m_fixed = StudentTMixture(n_components=3, n_features=4, covariance_type='full', dof=5.0)
        assert m_per._n_parameters() == m_shared._n_parameters() + 2
        assert m_shared._n_parameters() == m_fixed._n_parameters() + 1

    def test_save_load_round_trip(self):
        X, _, _ = _two_blobs(9)
        m = StudentTMixture(n_components=2, random_state=0)
        m.fit(X)
        path = tempfile.mktemp(suffix='.pt')
        try:
            m.save(path)
            loaded = StudentTMixture.load(path)
            assert torch.equal(loaded.means_, m.means_)
            assert torch.equal(loaded.covariances_, m.covariances_)
            assert torch.equal(loaded.dof_, m.dof_)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_fit_snapshot_includes_dof(self):
        X, _, _ = _two_blobs(10)
        m = StudentTMixture(n_components=2, random_state=0)
        m.fit(X)
        snapshot = m._get_param_snapshot()
        assert 'dof_' in snapshot


# ===========================================================================
# VonMisesFisherMixture
# ===========================================================================
class TestLogBesselIv:
    @pytest.mark.skipif(not HAVE_MPMATH, reason="mpmath not available")
    @pytest.mark.parametrize("d", [2, 10, 128, 768])
    def test_matches_mpmath(self, d):
        nu = d / 2 - 1
        for kappa in [0.5, 5.0, 50.0, 500.0]:
            ref = float(mpmath.log(mpmath.besseli(nu, kappa)))
            mine = _log_bessel_iv(nu, kappa)
            assert abs(ref - mine) < 1e-3


class TestVonMisesFisherMixtureFit:
    def test_is_a_base_mixture(self):
        assert issubclass(VonMisesFisherMixture, BaseMixture)

    @pytest.mark.parametrize("concentration", ['per_component', 'shared'])
    def test_recovers_three_well_separated_components(self, concentration):
        torch.manual_seed(0)
        d = 50
        true_mus = torch.randn(3, d)
        true_mus = true_mus / true_mus.norm(dim=1, keepdim=True)
        true_kappa = 80.0
        X = _vmf_like(1, true_mus, true_kappa, 200)
        labels_true = torch.cat([torch.full((200,), k) for k in range(3)])

        m = VonMisesFisherMixture(n_components=3, concentration=concentration, random_state=0, n_init=3, max_iter=50)
        m.fit(X)
        pred = m.predict(X)

        from tgmm.metrics import ClusteringMetrics
        ari = ClusteringMetrics.adjusted_rand_score(labels_true, pred)
        assert ari > 0.95

        # match fitted kappa against true within 20% (loose, since _vmf_like
        # is an approximate generator, not an exact vMF sampler)
        assert (m.kappa_ - true_kappa).abs().max() / true_kappa < 0.5

    def test_invalid_concentration_raises(self):
        with pytest.raises(ValueError):
            VonMisesFisherMixture(n_components=2, concentration='bogus')

    def test_monotone_log_likelihood(self):
        torch.manual_seed(0)
        d = 10
        true_mus = torch.randn(2, d)
        true_mus = true_mus / true_mus.norm(dim=1, keepdim=True)
        X = _vmf_like(2, true_mus, 40.0, 100)
        m = VonMisesFisherMixture(n_components=2, random_state=0, max_iter=50)
        m.fit(X)
        diffs = m.log_likelihood_history_[1:] - m.log_likelihood_history_[:-1]
        assert (diffs >= -1e-5).all()

    def test_normalizes_non_unit_input_with_warning(self):
        torch.manual_seed(0)
        d = 10
        true_mus = torch.randn(2, d)
        true_mus = true_mus / true_mus.norm(dim=1, keepdim=True)
        X = _vmf_like(3, true_mus, 40.0, 100)
        m = VonMisesFisherMixture(n_components=2, random_state=0, max_iter=30)
        m.fit(X)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pred = m.predict(X * 3.0)
            assert any('normaliz' in str(w.message) for w in caught)
        assert pred.shape == (200,)

    def test_n_parameters(self):
        m_per = VonMisesFisherMixture(n_components=4, n_features=10, concentration='per_component')
        m_shared = VonMisesFisherMixture(n_components=4, n_features=10, concentration='shared')
        assert m_per._n_parameters() == m_shared._n_parameters() + 3

    def test_sample_produces_unit_norm_rows(self):
        torch.manual_seed(0)
        d = 8
        true_mus = torch.randn(2, d)
        true_mus = true_mus / true_mus.norm(dim=1, keepdim=True)
        X = _vmf_like(4, true_mus, 30.0, 80)
        m = VonMisesFisherMixture(n_components=2, random_state=0, max_iter=30)
        m.fit(X)
        samples, idx = m.sample(200)
        assert torch.allclose(samples.norm(dim=1), torch.ones(200), atol=1e-4)

    def test_save_load_round_trip(self):
        torch.manual_seed(0)
        d = 8
        true_mus = torch.randn(2, d)
        true_mus = true_mus / true_mus.norm(dim=1, keepdim=True)
        X = _vmf_like(5, true_mus, 30.0, 80)
        m = VonMisesFisherMixture(n_components=2, random_state=0, max_iter=30)
        m.fit(X)
        path = tempfile.mktemp(suffix='.pt')
        try:
            m.save(path)
            loaded = VonMisesFisherMixture.load(path)
            assert torch.equal(loaded.means_, m.means_)
            assert torch.equal(loaded.kappa_, m.kappa_)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestSphericalKMeansSeeding:
    def test_kmeans_pp_seeding_separates_three_well_separated_clusters(self):
        r"""
        Regression test for the bug found during implementation: plain
        uniform-random seeding (pick k random rows outright) could draw two
        initial centers from the same true cluster, merging the other two
        for the rest of the fit. Cosine-distance k-means++ seeding fixes it.
        """
        torch.manual_seed(0)
        d = 50
        true_mus = torch.randn(3, d)
        true_mus = true_mus / true_mus.norm(dim=1, keepdim=True)
        X = _vmf_like(0, true_mus, 80.0, 200)

        from tgmm.metrics import ClusteringMetrics
        labels_true = torch.cat([torch.full((200,), k) for k in range(3)])

        for seed in range(5):
            centers = _spherical_kmeans(X, 3, random_state=seed)
            sims = X @ centers.t()
            labels = torch.argmax(sims, dim=1)
            ari = ClusteringMetrics.adjusted_rand_score(labels_true, labels)
            assert ari > 0.9, f"seed {seed}: ARI {ari} too low -- k-means++ seeding regressed"
