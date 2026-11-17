"""
Phase 4 (v0.4 roadmap): `BaseMixture` as the shared machinery of `GaussianMixture`, and `SufficientStats.aux`.
(The Student-t and von Mises-Fisher mixtures this phase also introduced have been removed: the repo keeps only
Gaussian mixtures.) See docs/superpowers/specs/2026-10-14-phase4-base-mixture-studentt-vmf-design.md for the history.
"""
import math

import torch

from tgmm import GaussianMixture
from tgmm.mixtures import BaseMixture
from tgmm.stats import SufficientStats


def _two_blobs(seed, n_per=150, d=2, sep=10.0, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    c0 = torch.zeros(d)
    c1 = torch.zeros(d)
    c1[0] = sep
    X0 = torch.randn(n_per, d, generator=gen) * scale + c0
    X1 = torch.randn(n_per, d, generator=gen) * scale + c1
    return torch.cat([X0, X1]), torch.cat([torch.zeros(n_per), torch.ones(n_per)]).long(), torch.stack([c0, c1])


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
