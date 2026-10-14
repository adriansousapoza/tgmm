"""
Phase 5 (v0.4 roadmap): semi-supervised and positive-unlabeled fitting. See
docs/superpowers/specs/2026-10-14-phase5-semi-supervised-pu-design.md.
"""
import math

import pytest
import torch

from tgmm import GaussianMixture, PositiveUnlabeledMixture, StudentTMixture, VonMisesFisherMixture
from tgmm.metrics import ClusteringMetrics


def _two_blobs(seed, n_per=150, d=2, sep=10.0, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    c0 = torch.zeros(d)
    c1 = torch.zeros(d)
    c1[0] = sep
    X0 = torch.randn(n_per, d, generator=gen) * scale + c0
    X1 = torch.randn(n_per, d, generator=gen) * scale + c1
    X = torch.cat([X0, X1])
    labels = torch.cat([torch.zeros(n_per), torch.ones(n_per)]).long()
    return X, labels


def _partial_labels(labels, frac, seed):
    gen = torch.Generator().manual_seed(seed)
    mask = torch.rand(labels.shape[0], generator=gen) < frac
    return torch.where(mask, labels, torch.full_like(labels, -1)), mask


# ===========================================================================
# 5.1: supervised fit (components_per_class, no -1)
# ===========================================================================
class TestSupervisedFit:
    def test_fully_labeled_matches_closed_form_mle(self):
        X, labels = _two_blobs(0, sep=10.0)
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
        gmm.fit(X, labels=labels)
        for c in [0, 1]:
            Xc = X[labels == c]
            k = (gmm.classes_ == c).nonzero(as_tuple=True)[0].item()
            assert torch.allclose(gmm.means_[k], Xc.mean(dim=0), atol=1e-5)
            expected_cov = torch.cov(Xc.T) * (Xc.shape[0] - 1) / Xc.shape[0]
            assert torch.allclose(gmm.covariances_[k], expected_cov, atol=1e-4)
        assert gmm.converged_ and gmm.n_iter_ == 1

    def test_classes_and_class_of_component_default_none(self):
        gmm = GaussianMixture(n_components=2, random_state=0)
        assert gmm.classes_ is None
        assert gmm.class_of_component_ is None

    def test_unsupervised_refit_clears_classes(self):
        X, labels = _two_blobs(1)
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(X, labels=labels)
        assert gmm.classes_ is not None
        gmm.fit(X)
        assert gmm.classes_ is None
        assert gmm.class_of_component_ is None

    def test_predict_class_recovers_labels(self):
        X, labels = _two_blobs(2, sep=12.0)
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(X, labels=labels)
        pred_class = gmm.predict_class(X)
        assert torch.equal(pred_class, labels)

    def test_predict_class_without_fit_labels_raises(self):
        X, _ = _two_blobs(3)
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(X)
        with pytest.raises(RuntimeError):
            gmm.predict_class(X)

    def test_components_per_class_dict_captures_bimodal_class(self):
        gen = torch.Generator().manual_seed(4)
        c0a = torch.randn(80, 2, generator=gen) * 0.4 + torch.tensor([0.0, 0.0])
        c0b = torch.randn(80, 2, generator=gen) * 0.4 + torch.tensor([6.0, 6.0])
        c1 = torch.randn(160, 2, generator=gen) * 0.4 + torch.tensor([12.0, 0.0])
        X = torch.cat([c0a, c0b, c1])
        labels = torch.cat([torch.zeros(160), torch.ones(160)]).long()

        gmm = GaussianMixture(n_components=3, components_per_class={0: 2, 1: 1}, random_state=0)
        gmm.fit(X, labels=labels)
        assert torch.equal(gmm.class_of_component_, torch.tensor([0, 0, 1]))
        pred_class = gmm.predict_class(X)
        assert (pred_class == labels).float().mean() > 0.95

    def test_components_per_class_total_mismatch_raises(self):
        X, labels = _two_blobs(5)
        with pytest.raises(ValueError, match="components_per_class implies"):
            GaussianMixture(n_components=5, random_state=0).fit(X, labels=labels)

    def test_components_per_class_dict_missing_label_raises(self):
        X, labels = _two_blobs(6)
        with pytest.raises(ValueError, match="missing an entry"):
            GaussianMixture(n_components=2, components_per_class={0: 1}, random_state=0).fit(X, labels=labels)

    @pytest.mark.parametrize("covariance_type", ['tied_full', 'tied_diag', 'tied_spherical'])
    def test_tied_covariance_with_block_gt_1_raises(self, covariance_type):
        X, labels = _two_blobs(7)
        with pytest.raises(ValueError, match="tied covariance"):
            GaussianMixture(
                n_components=3, covariance_type=covariance_type,
                components_per_class={0: 2, 1: 1}, random_state=0,
            ).fit(X, labels=labels)

    def test_sparse_component_warning(self):
        X, labels = _two_blobs(8, n_per=100)
        # Shrink class 1 down to 2 points -- too few for a 'full' 2-D covariance.
        class1_idx = (labels == 1).nonzero(as_tuple=True)[0]
        keep = torch.zeros(X.shape[0], dtype=torch.bool)
        keep[labels == 0] = True
        keep[class1_idx[:2]] = True
        X_sparse, labels_sparse = X[keep], labels[keep]
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
        with pytest.warns(UserWarning, match="too few samples"):
            gmm.fit(X_sparse, labels=labels_sparse)

    def test_other_families_support_supervised_fit(self):
        X, labels = _two_blobs(9, sep=12.0)
        stm = StudentTMixture(n_components=2, random_state=0)
        stm.fit(X, labels=labels)
        assert torch.equal(stm.predict_class(X), labels)

        Xn = X / X.norm(dim=1, keepdim=True)
        vmf = VonMisesFisherMixture(n_components=2, random_state=0)
        vmf.fit(Xn, labels=labels)
        assert vmf.classes_ is not None


# ===========================================================================
# 5.1: semi-supervised EM (some -1)
# ===========================================================================
class TestSemiSupervisedFit:
    def test_all_labeled_matches_supervised_path(self):
        X, labels = _two_blobs(10, sep=10.0)
        gmm_sup = GaussianMixture(n_components=2, random_state=0)
        gmm_sup.fit(X, labels=labels)

        gmm_semi = GaussianMixture(n_components=2, random_state=0, max_iter=50)
        gmm_semi.fit(X, labels=labels)  # no -1 present -- same path, sanity check
        assert torch.allclose(gmm_sup.means_, gmm_semi.means_, atol=1e-6)

    def test_semi_supervised_beats_unsupervised_on_overlapping_blobs(self):
        # Overlapping enough that unsupervised EM's cluster <-> class
        # correspondence is unreliable across seeds; a few labels per class
        # fix that. Individual seeds can go either way on a single finite
        # sample (an overlapping-blob dataset's realized overlap is itself
        # seed-sensitive), so the claim is about the *average* over many
        # independent datasets, not a per-seed guarantee.
        semi_aris, unsup_aris = [], []
        for seed in range(15):
            X, true_labels = _two_blobs(seed, n_per=200, sep=3.0, scale=1.8)
            labels, mask = _partial_labels(true_labels, 0.20, seed=100 + seed)

            gmm_semi = GaussianMixture(n_components=2, random_state=seed, max_iter=100)
            gmm_semi.fit(X, labels=labels)
            semi_aris.append(ClusteringMetrics.adjusted_rand_score(true_labels, gmm_semi.predict(X)))

            gmm_unsup = GaussianMixture(n_components=2, random_state=seed, max_iter=100)
            gmm_unsup.fit(X)
            unsup_aris.append(ClusteringMetrics.adjusted_rand_score(true_labels, gmm_unsup.predict(X)))

        assert sum(semi_aris) / len(semi_aris) > sum(unsup_aris) / len(unsup_aris)

    def test_unlabeled_weight_zero_equals_supervised_on_labeled_subset(self):
        X, true_labels = _two_blobs(12, n_per=200, sep=3.0, scale=1.8)
        labels, mask = _partial_labels(true_labels, 0.15, seed=13)

        gmm_uw0 = GaussianMixture(n_components=2, random_state=0, unlabeled_weight=0.0, max_iter=100)
        gmm_uw0.fit(X, labels=labels)

        gmm_sup_only = GaussianMixture(n_components=2, random_state=0)
        gmm_sup_only.fit(X[mask], labels=true_labels[mask])

        assert torch.allclose(gmm_uw0.means_, gmm_sup_only.means_, atol=1e-4)

    def test_monotone_log_likelihood(self):
        X, true_labels = _two_blobs(14, n_per=150, sep=4.0, scale=1.5)
        labels, _ = _partial_labels(true_labels, 0.2, seed=15)
        gmm = GaussianMixture(n_components=2, random_state=0, max_iter=50)
        gmm.fit(X, labels=labels)
        diffs = gmm.log_likelihood_history_[1:] - gmm.log_likelihood_history_[:-1]
        assert (diffs >= -1e-6).all()

    def test_cem_semi_supervised_runs(self):
        X, true_labels = _two_blobs(16, n_per=100, sep=4.0, scale=1.3)
        labels, _ = _partial_labels(true_labels, 0.2, seed=17)
        gmm = GaussianMixture(n_components=2, algorithm='cem', random_state=0, max_iter=50)
        gmm.fit(X, labels=labels)
        assert gmm.fitted_

    def test_label_confidence_tolerates_some_mislabeling(self):
        X, true_labels = _two_blobs(18, n_per=200, sep=10.0, scale=1.0)
        labels, mask = _partial_labels(true_labels, 0.3, seed=19)
        # Corrupt a few labeled rows (flip their label) to simulate label noise.
        labeled_idx = mask.nonzero(as_tuple=True)[0]
        corrupt_idx = labeled_idx[:5]
        labels[corrupt_idx] = 1 - labels[corrupt_idx]

        gmm_hard = GaussianMixture(n_components=2, label_confidence=1.0, random_state=0, max_iter=100)
        gmm_hard.fit(X, labels=labels)
        gmm_soft = GaussianMixture(n_components=2, label_confidence=0.8, random_state=0, max_iter=100)
        gmm_soft.fit(X, labels=labels)
        # Both should still basically recover the two well-separated blobs.
        ari_hard = ClusteringMetrics.adjusted_rand_score(true_labels, gmm_hard.predict(X))
        ari_soft = ClusteringMetrics.adjusted_rand_score(true_labels, gmm_soft.predict(X))
        assert ari_hard > 0.8 and ari_soft > 0.8

    @pytest.mark.parametrize("kwargs", [{'algorithm': 'sem'}, {'algorithm': 'saem'}, {'step_size': 0.5, 'batch_size': 20}])
    def test_unsupported_algorithm_raises_not_implemented(self, kwargs):
        X, true_labels = _two_blobs(20, n_per=50)
        labels, _ = _partial_labels(true_labels, 0.2, seed=21)
        with pytest.raises(NotImplementedError):
            GaussianMixture(n_components=2, random_state=0, **kwargs).fit(X, labels=labels)

    def test_other_families_support_semisupervised_fit(self):
        X, true_labels = _two_blobs(22, n_per=150, sep=8.0, scale=1.2)
        labels, _ = _partial_labels(true_labels, 0.2, seed=23)

        stm = StudentTMixture(n_components=2, random_state=0, max_iter=50)
        stm.fit(X, labels=labels)
        assert ClusteringMetrics.adjusted_rand_score(true_labels, stm.predict(X)) > 0.8

        Xn = X / X.norm(dim=1, keepdim=True)
        vmf = VonMisesFisherMixture(n_components=2, random_state=0, max_iter=50)
        vmf.fit(Xn, labels=labels)
        assert ClusteringMetrics.adjusted_rand_score(true_labels, vmf.predict(Xn)) > 0.7


# ===========================================================================
# 5.2: PositiveUnlabeledMixture
# ===========================================================================
def _pu_data(seed, d=10, n_pos=300, n_neg=700, sep=4.0, label_frac=0.2):
    gen = torch.Generator().manual_seed(seed)
    pos = torch.randn(n_pos, d, generator=gen) * 1.0 + sep
    neg = torch.randn(n_neg, d, generator=gen) * 1.0 - sep
    X = torch.cat([pos, neg])
    true_positive = torch.cat([torch.ones(n_pos, dtype=torch.bool), torch.zeros(n_neg, dtype=torch.bool)])

    gen2 = torch.Generator().manual_seed(seed + 1000)
    pos_idx = true_positive.nonzero(as_tuple=True)[0]
    n_label = int(label_frac * pos_idx.numel())
    labeled = pos_idx[torch.randperm(pos_idx.numel(), generator=gen2)[:n_label]]
    positive_mask = torch.zeros(X.shape[0], dtype=torch.bool)
    positive_mask[labeled] = True
    return X, true_positive, positive_mask, n_pos / (n_pos + n_neg)


def _f1_positive(pred_positive, true_positive):
    tp = (pred_positive & true_positive).sum().item()
    fp = (pred_positive & ~true_positive).sum().item()
    fn = (~pred_positive & true_positive).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


class TestPositiveUnlabeledMixture:
    def test_constructor_validation(self):
        with pytest.raises(NotImplementedError):
            PositiveUnlabeledMixture(n_components_negative=None)
        with pytest.raises(ValueError, match="anchor='prior'"):
            PositiveUnlabeledMixture(family='student_t', anchor='prior')
        with pytest.raises(ValueError, match="family"):
            PositiveUnlabeledMixture(family='bogus')
        with pytest.raises(ValueError, match="anchor"):
            PositiveUnlabeledMixture(anchor='bogus')
        with pytest.raises(ValueError, match="set internally"):
            PositiveUnlabeledMixture(n_components=3)
        with pytest.raises(ValueError, match="set internally"):
            PositiveUnlabeledMixture(components_per_class=2)

    @pytest.mark.parametrize("anchor", ['prior', 'fixed'])
    def test_class_prior_and_f1_gaussian(self, anchor):
        X, true_positive, positive_mask, pi_true = _pu_data(0)
        pu = PositiveUnlabeledMixture(
            n_components_positive=1, n_components_negative=1, family='gaussian',
            anchor=anchor, random_state=0, max_iter=100,
        )
        pu.fit(X, positive_mask)
        assert abs(pu.class_prior_ - pi_true) < 0.07
        f1 = _f1_positive(pu.predict_positive(X), true_positive)
        assert f1 > 0.9

    def test_free_anchor_runs_and_documents_no_guarantee(self):
        X, true_positive, positive_mask, pi_true = _pu_data(1)
        pu = PositiveUnlabeledMixture(
            n_components_positive=1, n_components_negative=1, family='gaussian',
            anchor='free', random_state=0, max_iter=100,
        )
        pu.fit(X, positive_mask)
        # Only check it runs and produces a probability in [0, 1] -- 'free'
        # has no identifiability guarantee on overlapping data (see design doc).
        assert 0.0 <= pu.class_prior_ <= 1.0

    @pytest.mark.parametrize("family,anchor", [('student_t', 'fixed'), ('student_t', 'free'), ('vmf', 'fixed'), ('vmf', 'free')])
    def test_other_families(self, family, anchor):
        X, true_positive, positive_mask, pi_true = _pu_data(2, d=8)
        if family == 'vmf':
            X = X / X.norm(dim=1, keepdim=True)
        pu = PositiveUnlabeledMixture(
            n_components_positive=1, n_components_negative=1, family=family,
            anchor=anchor, random_state=0, max_iter=100,
        )
        pu.fit(X, positive_mask)
        f1 = _f1_positive(pu.predict_positive(X), true_positive)
        assert f1 > 0.85

    def test_predict_proba_positive_in_unit_interval(self):
        X, true_positive, positive_mask, _ = _pu_data(3)
        pu = PositiveUnlabeledMixture(random_state=0, max_iter=100)
        pu.fit(X, positive_mask)
        proba = pu.predict_proba_positive(X)
        assert (proba >= 0).all() and (proba <= 1 + 1e-6).all()

    def test_getattr_delegates_to_underlying_model(self):
        X, true_positive, positive_mask, _ = _pu_data(4)
        pu = PositiveUnlabeledMixture(random_state=0, max_iter=100)
        pu.fit(X, positive_mask)
        # bic/score/etc. aren't defined on PositiveUnlabeledMixture itself --
        # should transparently forward to the underlying fitted mixture.
        assert isinstance(pu.bic(X), float)
        assert isinstance(pu.score(X), float)

    def test_fit_requires_at_least_one_labeled_positive(self):
        X, true_positive, positive_mask, _ = _pu_data(5)
        pu = PositiveUnlabeledMixture(random_state=0)
        with pytest.raises(ValueError, match="at least one labeled positive"):
            pu.fit(X, torch.zeros(X.shape[0], dtype=torch.bool))

    def test_positive_block_is_last_in_component_order(self):
        # classes_ sorts to [0 (negative), 1 (positive)]; the positive block
        # is therefore the LAST n_components_positive components, not the first.
        X, true_positive, positive_mask, _ = _pu_data(6, n_pos=150, n_neg=350)
        pu = PositiveUnlabeledMixture(
            n_components_positive=1, n_components_negative=3, random_state=0, max_iter=50,
        )
        pu.fit(X, positive_mask)
        # The positive block's mean should be close to the labeled positives' mean.
        pos_mean = X[positive_mask].mean(dim=0)
        assert torch.allclose(pu.model_.means_[-1], pos_mean, atol=2.0)
