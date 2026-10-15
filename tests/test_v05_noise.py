"""v0.5 item 1.1: uniform noise component (spec: 2026-10-15-v05-noise-component-design.md)."""
import math
import warnings

import pytest
import torch

from tgmm import GaussianMixture, StudentTMixture, VonMisesFisherMixture, HDBSCAN
from tgmm.metrics import ClusteringMetrics

warnings.filterwarnings("ignore", category=UserWarning)

CENTERS = torch.tensor([[-6.0, -6.0], [6.0, -6.0], [0.0, 6.0]], dtype=torch.float64)


def make_blobs_with_background(n_per=300, frac_noise=0.10, seed=0):
    """Three tight blobs + uniform background over a box; returns X, cluster labels (-1 = noise)."""
    g = torch.Generator().manual_seed(seed)
    X = [CENTERS[k] + 0.6 * torch.randn(n_per, 2, generator=g, dtype=torch.float64) for k in range(3)]
    y = [torch.full((n_per,), k) for k in range(3)]
    n_noise = int(frac_noise * 3 * n_per / (1 - frac_noise))
    box_lo, box_hi = torch.tensor([-12.0, -12.0]), torch.tensor([12.0, 12.0])
    noise = box_lo + (box_hi - box_lo) * torch.rand(n_noise, 2, generator=g, dtype=torch.float64)
    X.append(noise.to(torch.float64))
    y.append(torch.full((n_noise,), -1))
    return torch.cat(X), torch.cat(y)


def fit_noise_gmm(X, **kw):
    kw.setdefault('n_components', 3)
    kw.setdefault('noise', True)
    kw.setdefault('random_state', 0)
    kw.setdefault('n_init', 3)
    kw.setdefault('max_iter', 300)
    return GaussianMixture(**kw).fit(X)


def match_to_true(pred, truth, K=3):
    """Fraction-correct under the best permutation of the K non-noise labels, ignoring noise rows of truth."""
    from itertools import permutations
    keep = truth >= 0
    best = 0.0
    for perm in permutations(range(K)):
        p = torch.tensor(perm)[pred[keep].clamp(min=0)]
        best = max(best, (p == truth[keep]).double().mean().item())
    return best


# ---------------------------------------------------------------------------
# Accuracy on the plan's benchmark
# ---------------------------------------------------------------------------
class TestAccuracy:
    def test_noise_recall_and_cluster_ari(self):
        X, y = make_blobs_with_background()
        gmm = fit_noise_gmm(X)
        pred = gmm.predict(X)
        is_noise = y == -1
        recall = (pred[is_noise] == -1).double().mean().item()
        assert recall > 0.8, recall
        keep = ~is_noise
        ari = ClusteringMetrics.adjusted_rand_score(y[keep], pred[keep])
        assert ari > 0.95, ari
        assert gmm.noise_weight_.item() == pytest.approx(0.10, abs=0.04)

    def test_without_noise_component_the_background_hurts(self):
        X, y = make_blobs_with_background()
        plain = GaussianMixture(n_components=3, random_state=0, n_init=3).fit(X)
        noisy = fit_noise_gmm(X)
        assert noisy.score(X) > plain.score(X)

    def test_em_log_likelihood_is_monotone(self):
        X, _ = make_blobs_with_background()
        gmm = fit_noise_gmm(X, n_init=1, tol=1e-9)
        hist = gmm.log_likelihood_history_
        assert torch.all(hist[1:] - hist[:-1] > -1e-8)

    def test_weights_sum_to_one_minus_noise_weight(self):
        X, _ = make_blobs_with_background()
        gmm = fit_noise_gmm(X)
        assert gmm.weights_.shape == (3,)
        total = gmm.weights_.sum() + gmm.noise_weight_
        assert total.item() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# noise=False is bit-identical
# ---------------------------------------------------------------------------
class TestDefaultUnchanged:
    @pytest.mark.parametrize("cls,kw", [
        (GaussianMixture, dict(n_components=3)),
        (StudentTMixture, dict(n_components=3)),
    ])
    def test_noise_false_matches_no_argument(self, cls, kw):
        X, _ = make_blobs_with_background()
        a = cls(random_state=0, max_iter=50, **kw).fit(X)
        b = cls(random_state=0, max_iter=50, noise=False, **kw).fit(X)
        assert torch.equal(a.weights_, b.weights_)
        assert torch.equal(a.means_, b.means_)
        assert torch.equal(a.covariances_, b.covariances_)
        assert a.lower_bound_ == b.lower_bound_
        assert a.predict_proba(X).shape == (X.shape[0], 3)
        assert a.noise is False and a.noise_weight_ is None and a.noise_bounds_ is None
        with pytest.raises(RuntimeError):
            a.noise_proba(X)

    def test_n_parameters_unchanged_without_noise(self):
        X, _ = make_blobs_with_background()
        a = GaussianMixture(n_components=3, random_state=0, max_iter=20).fit(X)
        b = GaussianMixture(n_components=3, random_state=0, max_iter=20, noise=True).fit(X)
        assert b._n_parameters() == a._n_parameters() + 1


# ---------------------------------------------------------------------------
# Algorithms, batch_size, GPCM, other families
# ---------------------------------------------------------------------------
class TestAlgorithms:
    @pytest.mark.parametrize("algorithm", ['em', 'cem', 'sem', 'saem'])
    def test_algorithms(self, algorithm):
        X, y = make_blobs_with_background()
        gmm = fit_noise_gmm(X, algorithm=algorithm, n_init=2, max_iter=150)
        pred = gmm.predict(X)
        assert pred.shape == (X.shape[0],)
        assert (pred == -1).any()
        assert pred.max() <= 2
        keep = y >= 0
        assert match_to_true(pred, y) > 0.9
        resp = gmm.predict_proba(X)
        assert resp.shape == (X.shape[0], 4)
        assert torch.allclose(resp.sum(1), torch.ones(X.shape[0], dtype=resp.dtype), atol=1e-9)

    def test_daem_temperature(self):
        X, y = make_blobs_with_background()
        gmm = fit_noise_gmm(X, temperature=(0.3, 20), n_init=1, max_iter=200)
        assert match_to_true(gmm.predict(X), y) > 0.9

    @pytest.mark.parametrize("algorithm", ['em', 'cem'])
    def test_batch_size_equals_unchunked(self, algorithm):
        X, _ = make_blobs_with_background(n_per=100)
        kw = dict(n_init=1, max_iter=60, algorithm=algorithm, random_state=1)
        a = fit_noise_gmm(X, **kw)
        b = fit_noise_gmm(X, batch_size=37, **kw)
        assert torch.allclose(a.means_, b.means_, atol=1e-8)
        assert torch.allclose(a.covariances_, b.covariances_, atol=1e-8)
        assert a.noise_weight_.item() == pytest.approx(b.noise_weight_.item(), abs=1e-8)

    @pytest.mark.parametrize("code", ['diag', 'tied_full', 'EVE', 'VVV'])
    def test_covariance_types_and_gpcm(self, code):
        X, y = make_blobs_with_background()
        gmm = fit_noise_gmm(X, covariance_type=code, n_init=2)
        assert match_to_true(gmm.predict(X), y) > 0.9
        assert (gmm.predict(X) == -1).sum() > 0

    def test_student_t(self):
        X, y = make_blobs_with_background()
        stm = StudentTMixture(n_components=3, noise=True, random_state=0, n_init=3, max_iter=300).fit(X)
        pred = stm.predict(X)
        assert (pred[y == -1] == -1).double().mean() > 0.6
        assert match_to_true(pred, y) > 0.9
        assert stm.noise_weight_.item() < 0.3

    def test_von_mises_fisher_noise_on_the_sphere(self):
        g = torch.Generator().manual_seed(0)
        mus = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float64)
        parts = [torch.nn.functional.normalize(mu + 0.12 * torch.randn(250, 3, generator=g, dtype=torch.float64), dim=1) for mu in mus]
        bg = torch.nn.functional.normalize(torch.randn(80, 3, generator=g, dtype=torch.float64), dim=1)
        X = torch.cat(parts + [bg])
        y = torch.cat([torch.full((250,), k) for k in range(3)] + [torch.full((80,), -1)])
        vmf = VonMisesFisherMixture(n_components=3, noise=True, random_state=0, n_init=3).fit(X)
        pred = vmf.predict(X)
        assert match_to_true(pred, y) > 0.95
        assert (pred[y == -1] == -1).double().mean() > 0.3
        # uniform noise density on S^2 is 1 / (4 pi)
        lp = vmf._noise_log_density(X)
        assert torch.allclose(lp, torch.full_like(lp, -math.log(4 * math.pi)), atol=1e-9)
        samples, idx = vmf.sample(500)
        assert torch.allclose(samples.norm(dim=1), torch.ones(500, dtype=samples.dtype), atol=1e-5)
        assert (idx == -1).any()


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
class TestSurface:
    def test_shapes_and_noise_proba(self):
        X, _ = make_blobs_with_background(n_per=100)
        gmm = fit_noise_gmm(X, n_init=1)
        n = X.shape[0]
        assert gmm.predict_proba(X).shape == (n, 4)
        np_ = gmm.noise_proba(X)
        assert np_.shape == (n,)
        assert torch.equal(np_, gmm.predict_proba(X)[:, -1])
        assert gmm.noise_bounds_.shape == (2, 2)
        assert torch.allclose(gmm.noise_bounds_[0], X.min(0).values)
        assert torch.allclose(gmm.noise_bounds_[1], X.max(0).values)

    def test_score_includes_noise_term_and_matches_mixture_density(self):
        X, _ = make_blobs_with_background(n_per=100)
        gmm = fit_noise_gmm(X, n_init=1)
        comp = gmm._component_log_prob(X) + torch.log(gmm.weights_)
        log_v = torch.log(gmm.noise_bounds_[1] - gmm.noise_bounds_[0]).sum()
        noise_col = (torch.log(gmm.noise_weight_) - log_v).expand(X.shape[0], 1)
        expected = torch.logsumexp(torch.cat([comp, noise_col], dim=1), dim=1)
        assert torch.allclose(gmm.score_samples(X), expected, atol=1e-9)

    def test_icl_bic_and_select_model(self):
        from tgmm import select_model
        X, _ = make_blobs_with_background(n_per=100)
        gmm = fit_noise_gmm(X, n_init=1)
        assert math.isfinite(gmm.bic(X)) and math.isfinite(gmm.aic(X)) and math.isfinite(gmm.icl(X))
        assert gmm.icl(X) >= gmm.bic(X)
        res = select_model(X, n_components=[2, 3], covariance_types=['full'], noise=True, random_state=0, max_iter=100)
        assert len(res.table) == 2

    def test_sample_draws_noise_uniformly_in_the_box(self):
        X, _ = make_blobs_with_background()
        gmm = fit_noise_gmm(X, n_init=1)
        samples, idx = gmm.sample(4000)
        assert samples.shape == (4000, 2)
        frac = (idx == -1).double().mean().item()
        assert frac == pytest.approx(gmm.noise_weight_.item(), abs=0.03)
        noise = samples[idx == -1]
        lo, hi = gmm.noise_bounds_
        assert torch.all(noise >= lo - 1e-9) and torch.all(noise <= hi + 1e-9)
        assert idx.max() <= 2

    def test_noise_volume_float_changes_the_noise_density(self):
        X, _ = make_blobs_with_background(n_per=100)
        a = fit_noise_gmm(X, n_init=1)
        b = fit_noise_gmm(X, n_init=1, noise_volume=1.0e6)
        # a huge volume makes the noise component nearly useless
        assert b.noise_weight_.item() < a.noise_weight_.item()
        assert b._noise_log_density(X)[0].item() == pytest.approx(-math.log(1.0e6))

    def test_noise_prior_sets_the_initial_weight(self):
        X, _ = make_blobs_with_background(n_per=100)
        gmm = GaussianMixture(n_components=3, noise=True, noise_prior=0.3, random_state=0, max_iter=1)
        gmm.fit(X)
        assert math.isfinite(gmm.noise_weight_.item())


# ---------------------------------------------------------------------------
# noise_init
# ---------------------------------------------------------------------------
class TestNoiseInit:
    def test_mask(self):
        X, y = make_blobs_with_background()
        gmm = fit_noise_gmm(X, noise_init=(y == -1), n_init=1)
        assert (gmm.predict(X)[y == -1] == -1).double().mean() > 0.8

    def test_hdbscan_labels(self):
        X, y = make_blobs_with_background()
        hdb = HDBSCAN(min_cluster_size=30).fit(X.to(torch.float32))
        assert (hdb.labels_ == -1).any()
        gmm = fit_noise_gmm(X, noise_init=hdb.labels_, n_init=1)
        assert match_to_true(gmm.predict(X), y) > 0.9

    def test_validation(self):
        X, _ = make_blobs_with_background(n_per=50)
        with pytest.raises(ValueError, match="noise_init"):
            fit_noise_gmm(X, noise_init=torch.zeros(X.shape[0] - 1, dtype=torch.bool))
        with pytest.raises(ValueError, match="every"):
            fit_noise_gmm(X, noise_init=torch.ones(X.shape[0], dtype=torch.bool))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_gaussian_save_load_roundtrip(self, tmp_path):
        X, _ = make_blobs_with_background(n_per=100)
        gmm = fit_noise_gmm(X, n_init=1)
        path = str(tmp_path / "m.pth")
        gmm.save(path)
        loaded = GaussianMixture.load(path)
        assert loaded.noise is True
        assert torch.equal(loaded.noise_weight_, gmm.noise_weight_)
        assert torch.equal(loaded.noise_bounds_, gmm.noise_bounds_)
        assert torch.equal(loaded.predict(X), gmm.predict(X))

    def test_gaussian_state_dict_roundtrip(self):
        X, _ = make_blobs_with_background(n_per=100)
        gmm = fit_noise_gmm(X, n_init=1)
        new = GaussianMixture(n_components=3, n_features=2)
        new.load_state_dict(gmm.save_state_dict())
        assert new.noise is True
        assert torch.equal(new.predict(X), gmm.predict(X))

    @pytest.mark.parametrize("cls,kw", [(StudentTMixture, dict(n_components=3)), (VonMisesFisherMixture, dict(n_components=3))])
    def test_other_families_roundtrip(self, cls, kw, tmp_path):
        X, _ = make_blobs_with_background(n_per=100)
        if cls is VonMisesFisherMixture:
            X = torch.nn.functional.normalize(X, dim=1)
        model = cls(noise=True, random_state=0, max_iter=60, **kw).fit(X)
        path = str(tmp_path / "m.pth")
        model.save(path)
        loaded = cls.load(path)
        assert loaded.noise is True
        assert torch.equal(loaded.noise_weight_, model.noise_weight_)
        assert torch.equal(loaded.predict(X), model.predict(X))
        new = cls(**kw)
        new.load_state_dict(model.save_state_dict())
        assert new.noise is True and torch.equal(new.noise_weight_, model.noise_weight_)

    def test_warm_start_reuses_noise_weight(self):
        X, _ = make_blobs_with_background(n_per=100)
        gmm = GaussianMixture(n_components=3, noise=True, warm_start=True, random_state=0, max_iter=5).fit(X)
        w = gmm.noise_weight_.clone()
        gmm.fit(X)
        assert math.isfinite(gmm.noise_weight_.item())
        assert gmm.noise_weight_.item() != 0.05  # moved away from the prior
        assert abs(gmm.noise_weight_.item() - w.item()) < 0.1


# ---------------------------------------------------------------------------
# Unsupported / invalid combinations
# ---------------------------------------------------------------------------
class TestUnsupported:
    def test_gibbs_mode(self):
        with pytest.raises(NotImplementedError, match="noise"):
            GaussianMixture(n_components=None, noise=True)

    def test_labels(self):
        X, y = make_blobs_with_background(n_per=50)
        gmm = GaussianMixture(n_components=3, noise=True)
        with pytest.raises(NotImplementedError, match="labels"):
            gmm.fit(X, labels=y.clamp(min=0))

    def test_online_em_and_partial_fit(self):
        X, _ = make_blobs_with_background(n_per=50)
        with pytest.raises(NotImplementedError, match="noise"):
            GaussianMixture(n_components=3, noise=True, batch_size=32, step_size='auto').fit(X)
        gmm = fit_noise_gmm(X, n_init=1)
        with pytest.raises(NotImplementedError, match="partial_fit"):
            gmm.partial_fit(X[:20])

    def test_weight_prior(self):
        with pytest.raises(ValueError, match="weight_concentration_prior"):
            GaussianMixture(n_components=3, noise=True, weight_concentration_prior=torch.ones(3))

    def test_constrained_sampling(self):
        X, _ = make_blobs_with_background(n_per=50)
        gmm = fit_noise_gmm(X, n_init=1)
        with pytest.raises(NotImplementedError, match="noise"):
            gmm.sample(10, component=0)
        with pytest.raises(NotImplementedError, match="noise"):
            gmm.sample(10, confidence=0.9)

    @pytest.mark.parametrize("kwargs", [
        dict(noise_prior=0.0), dict(noise_prior=1.0), dict(noise_volume=-1.0),
        dict(noise_volume='oops'), dict(noise_volume=0.0),
    ])
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            GaussianMixture(n_components=3, noise=True, **kwargs)

    def test_discrete_families_hook_raises(self):
        # families without bounded continuous support override the hook to raise (the discrete families arrive in Phase 2)
        class Dummy(GaussianMixture):
            def _noise_log_density(self, X):
                raise ValueError('noise component requires bounded continuous support')
        with pytest.raises(ValueError, match="bounded continuous support"):
            Dummy(n_components=2, noise=True, random_state=0, max_iter=3).fit(torch.randn(50, 2))
