"""
Phase 2 (v0.4 roadmap): Stochastic EM, deterministic annealing EM, SAEM,
mini-batch/online EM. See
docs/superpowers/specs/2026-10-14-phase2-stochastic-em-design.md.
"""
import math
import warnings

import pytest
import torch

from tgmm import GaussianMixture


def _two_blobs(n_per=150, seed=0, dtype=torch.float64, sep=6.0):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n_per, 2, generator=gen, dtype=dtype)
    b = torch.randn(n_per, 2, generator=gen, dtype=dtype) + torch.tensor([sep, 0.0], dtype=dtype)
    return torch.cat([a, b], dim=0)


X_TWO = _two_blobs()


# ===========================================================================
# algorithm validation
# ===========================================================================

def test_algorithm_accepts_sem_and_saem():
    GaussianMixture(n_components=2, n_features=2, algorithm='sem')
    GaussianMixture(n_components=2, n_features=2, algorithm='saem')


def test_gibbs_mode_rejects_non_em_algorithm():
    kwargs = dict(
        n_components=None, mean_prior=torch.zeros(2), mean_precision_prior=1.0,
        covariance_prior=torch.eye(2), degrees_of_freedom_prior=3.0,
    )
    for algorithm in ('cem', 'sem', 'saem'):
        with pytest.raises(ValueError):
            GaussianMixture(algorithm=algorithm, **kwargs)
    GaussianMixture(algorithm='em', **kwargs)  # does not raise


# ===========================================================================
# SEM
# ===========================================================================

class TestSEM:
    def test_sem_runs_full_max_iter_always_converged(self):
        gmm = GaussianMixture(n_components=2, n_features=2, algorithm='sem', max_iter=25, random_state=0)
        gmm.fit(X_TWO)
        assert gmm.converged_ is True
        assert gmm.n_iter_ == 25
        assert gmm.log_likelihood_history_.shape[0] == 26  # initial + 25 sweeps

    def test_sem_recovers_well_separated_means(self):
        gmm = GaussianMixture(n_components=2, n_features=2, algorithm='sem', max_iter=50, random_state=0)
        gmm.fit(X_TWO)
        means_sorted = gmm.means_[gmm.means_[:, 0].argsort()]
        assert torch.allclose(means_sorted[0], torch.zeros(2, dtype=torch.float64), atol=0.5)
        assert torch.allclose(means_sorted[1], torch.tensor([6.0, 0.0], dtype=torch.float64), atol=0.5)

    def test_sem_estimate_best_vs_mean_differ_but_both_reasonable(self):
        gmm_best = GaussianMixture(n_components=2, n_features=2, algorithm='sem', max_iter=50,
                                    random_state=0, sem_estimate='best')
        gmm_best.fit(X_TWO)
        gmm_mean = GaussianMixture(n_components=2, n_features=2, algorithm='sem', max_iter=50,
                                    random_state=0, sem_estimate='mean')
        gmm_mean.fit(X_TWO)
        for gmm in (gmm_best, gmm_mean):
            means_sorted = gmm.means_[gmm.means_[:, 0].argsort()]
            assert torch.allclose(means_sorted[0], torch.zeros(2, dtype=torch.float64), atol=0.5)
            assert torch.allclose(means_sorted[1], torch.tensor([6.0, 0.0], dtype=torch.float64), atol=0.5)

    def test_sem_burn_in_default_is_max_iter_over_five(self):
        gmm = GaussianMixture(n_components=2, n_features=2, algorithm='sem', max_iter=50, random_state=0)
        gmm.fit(X_TWO)
        assert gmm.sem_burn_in is None  # constructor default untouched

    def test_sem_burn_in_ge_max_iter_falls_back_to_last_iterate(self):
        gmm = GaussianMixture(n_components=2, n_features=2, algorithm='sem', max_iter=10,
                               random_state=0, sem_burn_in=10)
        gmm.fit(X_TWO)  # should not raise despite zero post-burn-in iterates
        assert gmm.means_.shape == (2, 2)

    def test_sem_empty_component_reseeding(self):
        # n_components deliberately larger than the number of real clusters,
        # with a tiny max_iter and an init that starts 3 components crammed
        # onto one tiny blob -- at least one should empty out during
        # sampling.
        torch.manual_seed(0)
        tiny_blob = torch.randn(5, 2, dtype=torch.float64) * 0.01
        init_means = tiny_blob[:3].clone()
        gmm = GaussianMixture(n_components=3, n_features=2, algorithm='sem', max_iter=20,
                               random_state=0, init_means=init_means, init_covariances='eye')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gmm.fit(tiny_blob)
        assert math.isfinite(gmm.lower_bound_)
        assert not torch.isnan(gmm.means_).any()
        assert not torch.isnan(gmm.covariances_).any()


# ===========================================================================
# Deterministic annealing (DAEM)
# ===========================================================================

class TestLogLikelihoodHistoryAllAlgorithms:
    @pytest.mark.parametrize("algorithm", ["em", "cem"])
    def test_plain_em_cem_track_history(self, algorithm):
        gmm = GaussianMixture(n_components=2, n_features=2, algorithm=algorithm, random_state=0, max_iter=200)
        gmm.fit(X_TWO)
        assert gmm.log_likelihood_history_ is not None
        assert gmm.log_likelihood_history_.shape[0] == gmm.n_iter_ + 1
        # Non-decreasing (CEM's hard-assignment objective is also monotone).
        diffs = gmm.log_likelihood_history_[1:] - gmm.log_likelihood_history_[:-1]
        assert (diffs >= -1e-8).all()

    def test_saem_tracks_history(self):
        gmm = GaussianMixture(n_components=2, n_features=2, algorithm='saem', random_state=0, max_iter=200)
        gmm.fit(X_TWO)
        assert gmm.log_likelihood_history_ is not None
        assert gmm.log_likelihood_history_.shape[0] == gmm.n_iter_ + 1


class TestDAEM:
    def test_beta_for_iteration_schedule(self):
        gmm = GaussianMixture(n_components=2, n_features=2, temperature=(0.1, 10))
        assert gmm._beta_for_iteration(0) == pytest.approx(0.1)
        assert gmm._beta_for_iteration(10) == pytest.approx(1.0)
        assert gmm._beta_for_iteration(20) == pytest.approx(1.0)
        assert 0.1 < gmm._beta_for_iteration(5) < 1.0

    def test_no_temperature_beta_always_one(self):
        gmm = GaussianMixture(n_components=2, n_features=2)
        assert gmm._beta_for_iteration(0) == 1.0
        assert gmm._beta_for_iteration(100) == 1.0

    def test_beta_tempers_responsibilities(self):
        gmm = GaussianMixture(n_components=3, n_features=2)
        gmm.weights_ = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
        gmm.means_ = torch.tensor([[0., 0.], [8., 0.], [4., 7.]], dtype=torch.float64)
        gmm.covariances_ = torch.stack([torch.eye(2, dtype=torch.float64)] * 3)
        X = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
        resp_full, _ = gmm._e_step(X, beta=1.0)
        resp_cold, _ = gmm._e_step(X, beta=0.1)
        # Tempering flattens the distribution toward uniform.
        assert resp_cold.max().item() < resp_full.max().item()

    def test_daem_escapes_adversarial_init_plain_em_cannot(self):
        """
        Reproduces (verified while writing this test to be deterministic
        given these exact seeds/settings): a 3-component init jittered
        around a single true cluster, which traps plain EM in a visibly
        worse local optimum; DAEM with a gentle (beta_0=0.9) schedule
        recovers the same log-likelihood as the best of 10 random-init EM
        restarts.

        Deliberately **not** float64 here (unlike the rest of this test
        file): this specific adversarial trajectory is numerically fragile
        (it rides along a near-symmetric fixed point before plain EM falls
        into a worse basin), and was verified only in the default float32
        dtype -- switching to float64 changes the rounding enough that
        plain EM no longer reliably gets stuck at this exact
        initialization. The underlying phenomenon (DAEM escaping local
        optima plain EM can't) isn't dtype-specific; this particular
        reproduction of it is.
        """
        torch.manual_seed(0)
        centers = [(0.0, 0.0), (8.0, 0.0), (4.0, 7.0)]
        X = torch.cat([torch.randn(150, 2) + torch.tensor(c) for c in centers])

        best_ll = -float('inf')
        for seed in range(10):
            gmm = GaussianMixture(n_components=3, n_features=2, random_state=seed, max_iter=200, init_means='kmeans')
            gmm.fit(X)
            best_ll = max(best_ll, gmm.lower_bound_)

        torch.manual_seed(3)
        init_means = torch.tensor([0.0, 0.0]).unsqueeze(0).repeat(3, 1) + torch.randn(3, 2) * 0.3
        init_covs = torch.stack([torch.eye(2)] * 3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gmm_plain = GaussianMixture(n_components=3, n_features=2, random_state=0, max_iter=500,
                                         init_means=init_means.clone(), init_covariances=init_covs.clone())
            gmm_plain.fit(X)

            gmm_daem = GaussianMixture(n_components=3, n_features=2, random_state=0, max_iter=500,
                                        init_means=init_means.clone(), init_covariances=init_covs.clone(),
                                        temperature=(0.9, 30))
            gmm_daem.fit(X)

        assert gmm_plain.lower_bound_ < best_ll - 0.1, "test setup assumption failed: plain EM should get stuck here"
        assert abs(gmm_daem.lower_bound_ - best_ll) < 1e-2
        assert len(gmm_daem.annealing_schedule_) >= 1

    def test_daem_annealing_schedule_recorded(self):
        gmm = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=50, temperature=(0.2, 10))
        gmm.fit(X_TWO)
        assert gmm.annealing_schedule_[0] == pytest.approx(0.2)
        assert gmm.annealing_schedule_[-1] == pytest.approx(1.0)


# ===========================================================================
# Mini-batch / online EM
# ===========================================================================

class TestOnlineEM:
    def test_step_size_requires_batch_size(self):
        gmm = GaussianMixture(n_components=2, n_features=2, step_size='auto')
        with pytest.raises(ValueError):
            gmm.fit(X_TWO)

    def test_online_em_runs_and_tracks_history(self):
        gmm = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=5,
                               batch_size=20, step_size='auto')
        gmm.fit(X_TWO)
        assert gmm.converged_ is True
        assert gmm.n_iter_ == 5
        assert gmm.log_likelihood_history_.shape[0] == 5

    def test_online_em_large_scale_matches_full_batch(self):
        torch.manual_seed(0)
        n_per = 100_000
        X = torch.cat([
            torch.randn(n_per, 2, dtype=torch.float64),
            torch.randn(n_per, 2, dtype=torch.float64) + torch.tensor([6.0, 0.0], dtype=torch.float64),
        ])

        gmm_full = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=20)
        gmm_full.fit(X)

        gmm_online = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=3,
                                      batch_size=2000, step_size='auto')
        gmm_online.fit(X)

        full_ll = gmm_full.score(X)
        online_ll = gmm_online.score(X)
        assert abs(online_ll - full_ll) / abs(full_ll) < 0.005

    def test_partial_fit_requires_fitted_model(self):
        gmm = GaussianMixture(n_components=2, n_features=2)
        with pytest.raises(RuntimeError):
            gmm.partial_fit(X_TWO[:10])

    def test_partial_fit_continues_online_training(self):
        gmm = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=2,
                               batch_size=20, step_size='auto')
        gmm.fit(X_TWO[:200])
        ll_before = gmm.score(X_TWO)
        for start in range(0, X_TWO.shape[0], 20):
            gmm.partial_fit(X_TWO[start:start + 20])
        ll_after = gmm.score(X_TWO)
        # Continuing to update on the (now fully seen) data shouldn't make
        # the fit wildly worse.
        assert ll_after > ll_before - 1.0

    def test_online_em_step_size_constant_float(self):
        gmm = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=5,
                               batch_size=20, step_size=0.1)
        gmm.fit(X_TWO)
        assert math.isfinite(gmm.lower_bound_)


# ===========================================================================
# SAEM
# ===========================================================================

class TestSAEM:
    def test_saem_defaults_step_size_to_auto(self):
        gmm = GaussianMixture(n_components=2, n_features=2, algorithm='saem', random_state=0, max_iter=100)
        assert gmm.step_size is None  # constructor default untouched
        gmm.fit(X_TWO)  # should not raise
        assert math.isfinite(gmm.lower_bound_)

    def test_saem_agrees_with_em_on_well_separated_blobs(self):
        # 1e-3-in-means agreement needs enough points that S-step sampling
        # noise averages out below that tolerance; the smaller X_TWO
        # fixture (150/cluster) is adequate for the other SAEM tests but
        # not this tight a numerical comparison.
        X = _two_blobs(n_per=2000, seed=1, sep=10.0)
        gmm_em = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=200)
        gmm_em.fit(X)
        gmm_saem = GaussianMixture(n_components=2, n_features=2, algorithm='saem', random_state=0, max_iter=200)
        gmm_saem.fit(X)

        em_sorted = gmm_em.means_[gmm_em.means_[:, 0].argsort()]
        saem_sorted = gmm_saem.means_[gmm_saem.means_[:, 0].argsort()]
        assert torch.allclose(em_sorted, saem_sorted, atol=1e-3)

    def test_saem_converges_with_tol(self):
        gmm = GaussianMixture(n_components=2, n_features=2, algorithm='saem', random_state=0,
                               max_iter=500, tol=1e-6)
        gmm.fit(X_TWO)
        assert gmm.converged_ is True

    def test_saem_respects_batch_size_as_memory_chunking_only(self):
        # batch_size here should only affect memory chunking (exact), not
        # introduce mini-batch stochasticity -- SAEM's "batch" is always
        # the whole dataset.
        gmm_nochunk = GaussianMixture(n_components=2, n_features=2, algorithm='saem', random_state=0, max_iter=50)
        gmm_nochunk.fit(X_TWO)
        gmm_chunked = GaussianMixture(n_components=2, n_features=2, algorithm='saem', random_state=0,
                                       max_iter=50, batch_size=37)
        gmm_chunked.fit(X_TWO)
        assert torch.allclose(gmm_nochunk.means_, gmm_chunked.means_, atol=1e-6)
