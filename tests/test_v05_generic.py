"""v0.5 item 1.5: the `Component` protocol and `GenericMixture` (protocol and autograd components).

All components used here are defined in this file on purpose: the library ships none.
Spec: docs/superpowers/specs/2026-10-15-v05-component-protocol-design.md
"""
import copy
import math
import warnings

import pytest
import torch
import torch.nn as nn

from tgmm import GaussianMixture, GenericMixture
from tgmm.mixtures import Component
from tgmm.mixtures.protocol import component_n_parameters

DT = torch.float64


# ---------------------------------------------------------------------------
# Test components
# ---------------------------------------------------------------------------
class DiagNormal:
    """Protocol component: diagonal Gaussian, exact weighted-moment M-step (same as GaussianMixture 'diag')."""

    def __init__(self, mean, var, reg=1e-6, frozen=False):
        self.mean = torch.as_tensor(mean, dtype=DT).clone()
        self.var = torch.as_tensor(var, dtype=DT).clone()
        self.reg = reg
        if frozen:
            self.frozen = True
        self._reset()

    def _reset(self):
        d = self.mean.shape[0]
        self._sw = torch.zeros((), dtype=DT)
        self._s1 = torch.zeros(d, dtype=DT)
        self._s2 = torch.zeros(d, dtype=DT)

    def log_probability(self, X):
        return (-0.5 * ((X - self.mean) ** 2 / self.var + torch.log(2 * math.pi * self.var))).sum(1)

    def summarize(self, X, sample_weight):
        w = sample_weight.to(DT)
        self._sw = self._sw + w.sum()
        self._s1 = self._s1 + (w[:, None] * X).sum(0)
        self._s2 = self._s2 + (w[:, None] * X * X).sum(0)

    def from_summaries(self):
        if float(self._sw) > 1e-12:
            self.mean = self._s1 / self._sw
            self.var = (self._s2 / self._sw - self.mean ** 2).clamp(min=0) + self.reg
        self._reset()

    def sample(self, n):
        return self.mean + self.var.sqrt() * torch.randn(n, self.mean.shape[0], dtype=DT)

    def n_parameters(self):
        return 2 * self.mean.shape[0]


class Exponential:
    """Protocol component: 1-D exponential on x >= 0, closed-form M-step."""

    def __init__(self, rate):
        self.rate = torch.as_tensor(float(rate), dtype=DT)
        self._reset()

    def _reset(self):
        self._sw = torch.zeros((), dtype=DT)
        self._sx = torch.zeros((), dtype=DT)

    def log_probability(self, X):
        x = X[:, 0]
        return torch.where(x >= 0, torch.log(self.rate) - self.rate * x, torch.full_like(x, -1e30))

    def summarize(self, X, sample_weight):
        w = sample_weight.to(DT)
        self._sw = self._sw + w.sum()
        self._sx = self._sx + (w * X[:, 0]).sum()

    def from_summaries(self):
        if float(self._sw) > 1e-12:
            self.rate = self._sw / self._sx.clamp(min=1e-12)
        self._reset()

    def sample(self, n):
        return torch.distributions.Exponential(self.rate).sample((n, 1))

    def n_parameters(self):
        return 1


class NoCount:
    """Protocol component without `n_parameters`: exercises the tensor-counting fallback."""

    def __init__(self, d):
        self.mean = torch.zeros(d, dtype=DT)
        self.scale = torch.ones(d, dtype=DT)
        self._acc = torch.zeros(d, dtype=DT)          # private accumulators are not parameters

    def log_probability(self, X):
        return torch.zeros(X.shape[0], dtype=DT)

    def summarize(self, X, sample_weight):
        pass

    def from_summaries(self):
        pass


class LaplaceModule(nn.Module):
    """Autograd component: independent Laplace per feature with learnable location and log-scale."""

    def __init__(self, loc, scale):
        super().__init__()
        self.loc = nn.Parameter(torch.as_tensor(loc, dtype=DT).clone())
        self.log_scale = nn.Parameter(torch.log(torch.as_tensor(scale, dtype=DT)).clone())

    def log_prob(self, X):
        b = self.log_scale.exp()
        return (-(X - self.loc).abs() / b - math.log(2.0) - self.log_scale).sum(1)

    def sample(self, n):
        d = torch.distributions.Laplace(self.loc.detach(), self.log_scale.exp().detach())
        return d.sample((n,))


class StudentModule(nn.Module):
    """Autograd component: Student-t (fixed dof) with learnable location and log-scale."""

    def __init__(self, loc, scale, df=4.0):
        super().__init__()
        self.loc = nn.Parameter(torch.as_tensor(loc, dtype=DT).clone())
        self.log_scale = nn.Parameter(torch.log(torch.as_tensor(scale, dtype=DT)).clone())
        self.df = df

    def log_prob(self, X):
        return torch.distributions.StudentT(self.df, self.loc, self.log_scale.exp()).log_prob(X).sum(1)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def blobs(n_per=150, seed=0, centers=((0.0, 0.0), (5.0, 5.0)), scale=0.8):
    g = torch.Generator().manual_seed(seed)
    parts = [torch.randn(n_per, 2, generator=g, dtype=DT) * scale + torch.tensor(c, dtype=DT) for c in centers]
    X = torch.cat(parts)
    y = torch.arange(len(centers)).repeat_interleave(n_per)
    return X, y


def diag_components(init_means, init_var=1.0, **kw):
    return [DiagNormal(m, torch.full((len(m),), init_var), **kw) for m in init_means]


INIT = [[0.5, 0.5], [4.0, 4.5]]


def fit_generic(X, comps=None, **kw):
    comps = comps if comps is not None else diag_components(INIT)
    kw.setdefault('max_iter', 100)
    kw.setdefault('tol', 1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return GenericMixture(comps, **kw).fit(X)


# ---------------------------------------------------------------------------
# (9) protocol conformance and parameter counting
# ---------------------------------------------------------------------------
def test_protocol_conformance():
    assert isinstance(DiagNormal([0.0], [1.0]), Component)
    assert isinstance(Exponential(1.0), Component)
    assert not isinstance(object(), Component)
    assert not isinstance(LaplaceModule([0.0], [1.0]), Component)  # autograd modules are adapted, not protocol


def test_rejects_non_components_and_bad_counts():
    with pytest.raises(TypeError, match="log_probability"):
        GenericMixture([object(), object()])
    with pytest.raises(ValueError):
        GenericMixture([])
    with pytest.raises(ValueError, match="n_init"):
        GenericMixture(diag_components(INIT), n_init=3)
    with pytest.raises(ValueError, match="sem_estimate"):
        GenericMixture(diag_components(INIT), algorithm='sem', sem_estimate='mean')


def test_n_parameters_explicit_fallback_and_autograd():
    assert component_n_parameters(DiagNormal([0.0, 0.0], [1.0, 1.0])) == 4
    assert component_n_parameters(NoCount(3)) == 6          # public tensors only: mean + scale
    X, _ = blobs()
    gm = fit_generic(X, [LaplaceModule([0.5, 0.5], [1.0, 1.0]), LaplaceModule([4.0, 4.5], [1.0, 1.0])])
    assert gm._n_parameters() == 4 + 4 + 1                  # two Laplaces (2d params each) + (K - 1)
    assert math.isfinite(gm.bic(X)) and math.isfinite(gm.aic(X))
    gm2 = fit_generic(X)
    assert gm2._n_parameters() == 4 + 4 + 1
    assert gm2.bic(X) == pytest.approx(-2 * gm2.score(X) * X.shape[0] + 9 * math.log(X.shape[0]))


# ---------------------------------------------------------------------------
# (1) parity with GaussianMixture(covariance_type='diag')
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ["em", "cem"])
def test_parity_with_gaussian_mixture_diag(algorithm):
    X, _ = blobs()
    gm = fit_generic(X, algorithm=algorithm)
    ref = GaussianMixture(
        n_components=2, covariance_type='diag', algorithm=algorithm, max_iter=100, tol=1e-8, reg_covar=1e-6,
        init_means=torch.tensor(INIT, dtype=DT), init_covariances=torch.ones(2, 2, dtype=DT),
        init_weights=torch.full((2,), 0.5, dtype=DT),
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref.fit(X)
    assert gm.lower_bound_ == pytest.approx(ref.lower_bound_, abs=1e-6)
    assert gm.n_iter_ == ref.n_iter_
    assert torch.allclose(gm.weights_, ref.weights_, atol=1e-8)
    for k in range(2):
        assert torch.allclose(gm.components_[k].mean, ref.means_[k], atol=1e-6)
        assert torch.allclose(gm.components_[k].var, ref.covariances_[k], atol=1e-6)
    assert torch.equal(gm.predict(X), ref.predict(X))
    assert torch.allclose(gm.score_samples(X), ref.score_samples(X), atol=1e-6)


# ---------------------------------------------------------------------------
# (2) optional pomegranate parity (adapter lives here, the library never imports pomegranate)
# ---------------------------------------------------------------------------
def test_pomegranate_normal_parity():
    # NB: a cloned `pomegranate/` reference repo at the repository root can shadow the real package as an
    # (empty) namespace package, so importorskip("pomegranate") alone would not skip: import the class itself.
    try:
        from pomegranate.distributions import Normal
    except ImportError:
        pytest.skip("pomegranate 1.x is not installed")

    class PGNormal:
        def __init__(self, means, covs):
            self.dist = Normal(means=torch.tensor(means, dtype=DT), covs=torch.tensor(covs, dtype=DT),
                               covariance_type='diag')

        def log_probability(self, X):
            return self.dist.log_probability(X)

        def summarize(self, X, sample_weight):
            self.dist.summarize(X, sample_weight=sample_weight[:, None])

        def from_summaries(self):
            self.dist.from_summaries()

        def n_parameters(self):
            return 4

    X, _ = blobs()
    comps = [PGNormal(m, [1.0, 1.0]) for m in INIT]
    gm = fit_generic(X, comps, max_iter=60)
    ref = GaussianMixture(
        n_components=2, covariance_type='diag', max_iter=60, tol=1e-8, reg_covar=0.0,
        init_means=torch.tensor(INIT, dtype=DT), init_covariances=torch.ones(2, 2, dtype=DT),
        init_weights=torch.full((2,), 0.5, dtype=DT),
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref.fit(X)
    assert gm.lower_bound_ == pytest.approx(ref.lower_bound_, abs=1e-4)


# ---------------------------------------------------------------------------
# (3) mixed Exponential + Normal
# ---------------------------------------------------------------------------
def test_mixed_exponential_normal_fits_and_samples():
    g = torch.Generator().manual_seed(1)
    expo = torch.distributions.Exponential(torch.tensor(2.0, dtype=DT)).sample((400, 1))
    norm = torch.randn(300, 1, generator=g, dtype=DT) * 0.5 + 8.0
    X = torch.cat([expo, norm])
    comps = [Exponential(1.0), DiagNormal([7.0], [4.0])]
    gm = fit_generic(X, comps, max_iter=200)
    hist = gm.log_likelihood_history_
    assert bool((hist[1:] - hist[:-1] >= -1e-9).all()), "EM log-likelihood must be monotone"
    assert gm.components_[0].rate.item() == pytest.approx(2.0, rel=0.15)
    assert gm.components_[1].mean.item() == pytest.approx(8.0, abs=0.2)
    assert gm.weights_[0].item() == pytest.approx(400 / 700, abs=0.05)
    S, idx = gm.sample(500)
    assert S.shape == (500, 1) and idx.shape == (500,)
    assert bool((S[idx == 0] >= 0).all())
    assert S[idx == 1].mean().item() == pytest.approx(8.0, abs=0.3)


# ---------------------------------------------------------------------------
# (4) autograd components via _gem_step
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module_cls", [LaplaceModule, StudentModule])
def test_autograd_components_monotone_and_accurate(module_cls):
    g = torch.Generator().manual_seed(2)
    lap = torch.distributions.Laplace(torch.zeros(2, dtype=DT), torch.full((2,), 0.7, dtype=DT))
    torch.manual_seed(2)
    X = torch.cat([lap.sample((250,)), lap.sample((250,)) + 6.0])
    comps = [module_cls([0.5, 0.5], [1.0, 1.0]), module_cls([5.0, 5.5], [1.0, 1.0])]
    gm = fit_generic(X, comps, max_iter=60, tol=1e-7)
    hist = gm.log_likelihood_history_
    assert bool((hist[1:] - hist[:-1] >= -1e-8).all()), "GEM log-likelihood must not decrease"
    assert hist[-1] > hist[0]
    locs = sorted(float(c.module.loc[0].detach()) for c in gm.components_)
    assert locs[0] == pytest.approx(0.0, abs=0.2) and locs[1] == pytest.approx(6.0, abs=0.2)
    assert gm.weights_.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_autograd_sample_requires_component_sample():
    X, _ = blobs()
    gm = fit_generic(X, [StudentModule([0.5, 0.5], [1.0, 1.0]), StudentModule([4.0, 4.5], [1.0, 1.0])], max_iter=5)
    with pytest.raises(NotImplementedError, match="sample"):
        gm.sample(10)
    gm2 = fit_generic(X, [LaplaceModule([0.5, 0.5], [1.0, 1.0]), LaplaceModule([4.0, 4.5], [1.0, 1.0])], max_iter=5)
    assert gm2.sample(20)[0].shape == (20, 2)


# ---------------------------------------------------------------------------
# (5) batch_size chunking is exact
# ---------------------------------------------------------------------------
def test_batch_size_equals_unchunked_protocol():
    X, _ = blobs()
    a = fit_generic(X, max_iter=30)
    b = fit_generic(X, max_iter=30, batch_size=37)
    assert torch.allclose(a.weights_, b.weights_, atol=1e-8)
    for k in range(2):
        assert torch.allclose(a.components_[k].mean, b.components_[k].mean, atol=1e-8)
        assert torch.allclose(a.components_[k].var, b.components_[k].var, atol=1e-8)
    assert a.lower_bound_ == pytest.approx(b.lower_bound_, abs=1e-8)


def test_batch_size_equals_unchunked_autograd():
    X, _ = blobs()

    def mk():
        return [LaplaceModule([0.5, 0.5], [1.0, 1.0]), LaplaceModule([4.0, 4.5], [1.0, 1.0])]

    a = fit_generic(X, mk(), max_iter=10)
    b = fit_generic(X, mk(), max_iter=10, batch_size=64)
    assert a.lower_bound_ == pytest.approx(b.lower_bound_, abs=1e-6)
    for k in range(2):
        assert torch.allclose(a.components_[k].module.loc, b.components_[k].module.loc, atol=1e-5)


# ---------------------------------------------------------------------------
# (6) algorithms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ["em", "cem", "sem"])
def test_algorithms_fit_and_recover(algorithm):
    X, y = blobs()
    torch.manual_seed(0)
    gm = fit_generic(X, algorithm=algorithm, max_iter=40)
    pred = gm.predict(X)
    agree = max((pred == y).double().mean().item(), (pred != y).double().mean().item())
    assert agree > 0.97
    means = sorted(float(c.mean[0]) for c in gm.components_)
    assert means[0] == pytest.approx(0.0, abs=0.3) and means[1] == pytest.approx(5.0, abs=0.3)


def test_unsupported_algorithms_raise():
    X, _ = blobs()
    with pytest.raises(NotImplementedError, match="saem"):
        fit_generic(X, algorithm='saem')
    with pytest.raises(NotImplementedError, match="step_size|online"):
        fit_generic(X, step_size='auto', batch_size=50)
    gm = fit_generic(X, max_iter=3)
    with pytest.raises(NotImplementedError):
        gm.partial_fit(X[:10])


def test_temperature_annealing_runs():
    X, _ = blobs()
    gm = fit_generic(X, temperature=(0.5, 5), max_iter=40)
    assert gm.annealing_schedule_ is not None
    assert math.isfinite(gm.lower_bound_)


def test_sem_empty_component_reseed_hook():
    X, _ = blobs()
    calls = []

    class Reseedable(DiagNormal):
        def reseed(self, x):
            calls.append(x.clone())
            self.mean = x.clone()

    far = [Reseedable([0.5, 0.5], [1.0, 1.0]), Reseedable([4.0, 4.5], [1.0, 1.0]),
           DiagNormal([500.0, 500.0], [0.01, 0.01])]
    with pytest.warns(UserWarning, match="re-seeded|emptied"):
        gm = GenericMixture(far, algorithm='sem', max_iter=6, tol=1e-8).fit(X)
    assert math.isfinite(gm.lower_bound_)
    # the third component has no `reseed`: left unchanged, no crash; reseedable ones are only called when empty
    assert gm.n_empty_components_ >= 1


# ---------------------------------------------------------------------------
# (7) frozen components
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["argument", "attribute"])
def test_frozen_components_keep_their_state(how):
    X, _ = blobs()
    if how == 'argument':
        comps, kw = diag_components(INIT), {'frozen_components': (0,)}
    else:
        comps, kw = [DiagNormal(INIT[0], [1.0, 1.0], frozen=True), DiagNormal(INIT[1], [1.0, 1.0])], {}
    gm = fit_generic(X, comps, **kw)
    assert torch.equal(gm.components_[0].mean, torch.tensor(INIT[0], dtype=DT))
    assert torch.equal(gm.components_[0].var, torch.ones(2, dtype=DT))
    assert not torch.allclose(gm.components_[1].mean, torch.tensor(INIT[1], dtype=DT))
    assert gm.weights_.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_frozen_autograd_component():
    X, _ = blobs()
    frozen = LaplaceModule([0.0, 0.0], [1.0, 1.0])
    frozen.frozen = True
    gm = fit_generic(X, [frozen, LaplaceModule([4.0, 4.5], [1.0, 1.0])], max_iter=10)
    assert torch.equal(gm.components_[0].module.loc.detach(), torch.zeros(2, dtype=DT))


# ---------------------------------------------------------------------------
# (8) sample_weight
# ---------------------------------------------------------------------------
def test_integer_sample_weight_equals_duplicated_rows():
    X, _ = blobs(n_per=60)
    w = torch.randint(1, 4, (X.shape[0],))
    X_dup = X.repeat_interleave(w, dim=0)
    a = fit_generic(X_dup, max_iter=40)
    b = fit_generic(X, max_iter=40)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c = GenericMixture(diag_components(INIT), max_iter=40, tol=1e-8).fit(X, sample_weight=w.to(DT))
    del b
    assert torch.allclose(a.weights_, c.weights_, atol=1e-8)
    for k in range(2):
        assert torch.allclose(a.components_[k].mean, c.components_[k].mean, atol=1e-8)
        assert torch.allclose(a.components_[k].var, c.components_[k].var, atol=1e-8)


def test_per_feature_sample_weight_rejected():
    X, _ = blobs(n_per=20)
    with pytest.raises(ValueError, match="per-feature"):
        GenericMixture(diag_components(INIT)).fit(X, sample_weight=torch.ones_like(X))


# ---------------------------------------------------------------------------
# noise / missing / labels / n_init
# ---------------------------------------------------------------------------
def test_noise_requires_explicit_volume_and_works():
    X, _ = blobs()
    with pytest.raises(NotImplementedError, match="noise_volume"):
        GenericMixture(diag_components(INIT), noise=True)
    g = torch.Generator().manual_seed(3)
    bg = torch.rand(40, 2, generator=g, dtype=DT) * 12 - 3
    Xn = torch.cat([X, bg])
    gm = fit_generic(Xn, noise=True, noise_volume=144.0, max_iter=80)
    assert (gm.weights_.sum() + gm.noise_weight_).item() == pytest.approx(1.0, abs=1e-10)
    assert gm.predict_proba(Xn).shape == (Xn.shape[0], 3)
    assert bool((gm.predict(Xn) == -1).any())
    assert gm._n_parameters() == 4 + 4 + 1 + 1


def test_missing_values_not_supported():
    X, _ = blobs(n_per=20)
    with pytest.raises(ValueError, match="missing"):
        GenericMixture(diag_components(INIT), missing='em')
    Xn = X.clone()
    Xn[0, 0] = float('nan')
    with pytest.raises(ValueError):
        GenericMixture(diag_components(INIT)).fit(Xn)


def test_semi_supervised_and_supervised_labels():
    X, y = blobs()
    labels = y.clone()
    labels[torch.rand(len(y), generator=torch.Generator().manual_seed(4)) > 0.2] = -1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gm = GenericMixture(diag_components(INIT), max_iter=60, tol=1e-8).fit(X, labels=labels)
    assert gm.classes_.tolist() == [0, 1]
    pred = gm.predict_class(X)
    assert (pred == y).double().mean().item() > 0.97
    full = GenericMixture(diag_components(INIT)).fit(X, labels=y)
    assert full.converged_
    assert (full.predict_class(X) == y).double().mean().item() > 0.97


def test_components_per_class_block_refit_unsupported():
    X, y = blobs()
    comps = diag_components(INIT + [[2.0, 2.0], [6.0, 6.0]])
    gm = GenericMixture(comps, components_per_class=2)
    with pytest.raises(NotImplementedError, match="components_per_class"):
        gm.fit(X, labels=y)


def test_warm_start_continues_from_fitted_components():
    X, _ = blobs()
    slow = lambda: [DiagNormal([2.0, 2.0], [4.0, 4.0]), DiagNormal([3.0, 3.0], [4.0, 4.0])]  # slow to converge
    gm = GenericMixture(slow(), max_iter=2, tol=1e-12, warm_start=True)
    cold = GenericMixture(slow(), max_iter=2, tol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gm.fit(X)
        ll1 = gm.lower_bound_
        gm.fit(X)                      # continues from the fitted components
        cold.fit(X)
        cold.fit(X)                    # restarts from the initial components each time
    assert gm.lower_bound_ > ll1 + 1e-6
    assert cold.lower_bound_ == pytest.approx(ll1, abs=1e-12)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def test_save_load_round_trip(tmp_path):
    X, _ = blobs()
    gm = fit_generic(X, max_iter=20)
    path = str(tmp_path / "generic.pt")
    gm.save(path)
    loaded = GenericMixture.load(path)
    assert torch.allclose(loaded.weights_, gm.weights_)
    assert torch.allclose(loaded.score_samples(X), gm.score_samples(X))
    assert loaded.n_components == 2 and loaded.fitted_


def test_save_load_round_trip_autograd(tmp_path):
    X, _ = blobs()
    gm = fit_generic(X, [LaplaceModule([0.5, 0.5], [1.0, 1.0]), LaplaceModule([4.0, 4.5], [1.0, 1.0])], max_iter=5)
    path = str(tmp_path / "generic_ag.pt")
    gm.save(path)
    loaded = GenericMixture.load(path)
    assert torch.allclose(loaded.score_samples(X), gm.score_samples(X))


def test_defaults_untouched_for_existing_families():
    # item 1.5 must not alter anything for the existing families: same fit twice is bit-identical
    X, _ = blobs()
    a = GaussianMixture(n_components=2, random_state=0, max_iter=30).fit(X)
    b = GaussianMixture(n_components=2, random_state=0, max_iter=30).fit(X)
    assert torch.equal(a.means_, b.means_) and torch.equal(a.covariances_, b.covariances_)
