"""v0.5 Phase 1 acceptance: a compact cross-feature matrix over the five Phase 1 features.

Each feature has its own detailed test file (``test_v05_noise.py``, ``test_v05_missing.py``,
``test_v05_condition.py``, ``test_v05_frozen.py``, ``test_v05_generic.py``). This file only checks that

* every feature finishes with finite parameters for each EM variant;
* the features that are meant to combine do combine;
* the new arguments at their defaults leave every fit bit-identical to a model constructed without them.

The pre-existing bit-identity fixtures (``tests/fixtures/phase0_pre_refactor_fits.pt``, used by ``test_phase0.py``) keep
guarding the default code path against the pre-v0.4 numbers.
"""
import pytest
import torch

from tgmm import GaussianMixture, StudentTMixture, VonMisesFisherMixture

ALGORITHMS = ['em', 'cem', 'sem']


def _blobs(n_per=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X = torch.cat([c + 0.7 * torch.randn(n_per, 2, generator=g) for c in centers])
    return X


@pytest.fixture(scope='module')
def X():
    return _blobs()


@pytest.fixture(scope='module')
def X_nan(X):
    g = torch.Generator().manual_seed(1)
    holes = torch.rand(X.shape, generator=g) < 0.15
    holes[holes.all(1), 0] = False          # keep one observed entry in every row
    out = X.clone()
    out[holes] = float('nan')
    return out


def _finite(model):
    assert torch.isfinite(model.means_).all()
    assert torch.isfinite(model.covariances_).all()
    assert torch.isfinite(model.weights_).all()
    total = float(model.weights_.sum()) + float(getattr(model, 'noise_weight_', 0.0) or 0.0)
    assert total == pytest.approx(1.0, abs=1e-4)


def _make(algorithm, **kwargs):
    return GaussianMixture(n_components=3, algorithm=algorithm, max_iter=40, random_state=0, **kwargs)


@pytest.mark.parametrize('algorithm', ALGORITHMS)
class TestEachFeatureAcrossAlgorithms:
    def test_noise(self, X, algorithm):
        gmm = _make(algorithm, noise=True).fit(X)
        _finite(gmm)
        assert gmm.predict_proba(X).shape == (len(X), 4)
        assert set(gmm.predict(X).unique().tolist()) <= {-1, 0, 1, 2}

    def test_missing(self, X_nan, algorithm):
        gmm = _make(algorithm, missing='em').fit(X_nan)
        _finite(gmm)
        assert torch.isfinite(gmm.impute(X_nan)).all()

    def test_frozen_components(self, X, algorithm):
        init = torch.tensor([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
        gmm = _make(algorithm, init_means=init, frozen_components=[0]).fit(X)
        _finite(gmm)
        assert torch.equal(gmm.means_[0], init[0])

    def test_sample_weight(self, X, algorithm):
        w = torch.rand(len(X), generator=torch.Generator().manual_seed(2)) + 0.5
        gmm = _make(algorithm).fit(X, sample_weight=w)
        _finite(gmm)

    def test_condition_and_marginal(self, X, algorithm):
        gmm = _make(algorithm).fit(X)
        cond = gmm.condition(X[:5, :1], [0])
        assert cond.mean().shape == (5, 1)
        assert gmm.marginal([1]).n_features == 1


class TestStudentTCombinations:
    def test_noise(self, X):
        stm = StudentTMixture(n_components=3, noise=True, max_iter=40, random_state=0).fit(X)
        assert torch.isfinite(stm.means_).all() and torch.isfinite(stm.dof_).all()
        assert stm.predict_proba(X).shape == (len(X), 4)

    def test_missing(self, X_nan):
        stm = StudentTMixture(n_components=3, missing='em', max_iter=40, random_state=0).fit(X_nan)
        assert torch.isfinite(stm.means_).all() and torch.isfinite(stm.covariances_).all()
        assert torch.isfinite(stm.impute(X_nan)).all()

    def test_frozen_and_weights(self, X):
        init = torch.tensor([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
        stm = StudentTMixture(n_components=3, init_means=init, frozen_components=[1], max_iter=40, random_state=0)
        stm.fit(X, sample_weight=torch.ones(len(X)))
        assert torch.equal(stm.means_[1], init[1])


class TestCombiningFeatures:
    def test_noise_with_frozen_components_and_weights(self, X):
        init = torch.tensor([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
        gmm = GaussianMixture(n_components=3, noise=True, init_means=init, frozen_components=[2], max_iter=40, random_state=0)
        gmm.fit(X, sample_weight=torch.ones(len(X)))
        _finite(gmm)
        assert torch.equal(gmm.means_[2], init[2])

    def test_missing_with_sample_weight(self, X_nan):
        gmm = GaussianMixture(n_components=3, missing='em', max_iter=40, random_state=0)
        gmm.fit(X_nan, sample_weight=torch.ones(len(X_nan)))
        _finite(gmm)

    def test_noise_and_missing_do_not_combine(self):
        with pytest.raises(NotImplementedError):
            GaussianMixture(n_components=2, noise=True, missing='em')


def _same_fit(a, b):
    for name in ('weights_', 'means_', 'covariances_'):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert a.lower_bound_ == b.lower_bound_
    assert a.n_iter_ == b.n_iter_


class TestDefaultsAreBitIdentical:
    """The new arguments at their defaults change nothing."""

    def test_gaussian(self, X):
        base = GaussianMixture(n_components=3, max_iter=60, random_state=0).fit(X)
        explicit = GaussianMixture(n_components=3, max_iter=60, random_state=0, noise=False, missing='error',
                                   frozen_components=()).fit(X, sample_weight=None)
        _same_fit(base, explicit)

    def test_gaussian_gpcm_code(self, X):
        base = GaussianMixture(n_components=3, covariance_type='EVE', max_iter=60, random_state=0).fit(X)
        explicit = GaussianMixture(n_components=3, covariance_type='EVE', max_iter=60, random_state=0, noise=False,
                                   missing='error', frozen_components=()).fit(X, sample_weight=None)
        _same_fit(base, explicit)

    def test_student_t(self, X):
        base = StudentTMixture(n_components=3, max_iter=60, random_state=0).fit(X)
        explicit = StudentTMixture(n_components=3, max_iter=60, random_state=0, noise=False, missing='error',
                                   frozen_components=()).fit(X, sample_weight=None)
        _same_fit(base, explicit)
        assert torch.equal(base.dof_, explicit.dof_)

    def test_von_mises_fisher(self, X):
        U = torch.nn.functional.normalize(X - X.mean(0), dim=1)
        base = VonMisesFisherMixture(n_components=3, max_iter=60, random_state=0).fit(U)
        explicit = VonMisesFisherMixture(n_components=3, max_iter=60, random_state=0, noise=False,
                                         frozen_components=()).fit(U, sample_weight=None)
        assert torch.equal(base.means_, explicit.means_)
        assert torch.equal(base.kappa_, explicit.kappa_)
        assert base.lower_bound_ == explicit.lower_bound_

    def test_missing_em_without_nans_equals_error(self, X):
        base = GaussianMixture(n_components=3, max_iter=60, random_state=0, missing='error').fit(X)
        em = GaussianMixture(n_components=3, max_iter=60, random_state=0, missing='em').fit(X)
        _same_fit(base, em)

    def test_all_ones_weights_equal_no_weights(self, X):
        base = GaussianMixture(n_components=3, max_iter=60, random_state=0).fit(X)
        ones = GaussianMixture(n_components=3, max_iter=60, random_state=0).fit(X, sample_weight=torch.ones(len(X)))
        _same_fit(base, ones)
