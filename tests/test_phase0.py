"""
Phase 0 (v0.4 roadmap): sufficient-statistics M-step, algorithm switch,
E-step/stats chunking, Aitken convergence, and the full-covariance memory
fix. See docs/superpowers/specs/2026-10-14-phase0-sufficient-stats-design.md.
"""
import os
import subprocess
import sys
import textwrap
import warnings

import pytest
import torch

from tgmm import GaussianMixture
from tgmm.stats import SufficientStats

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "phase0_pre_refactor_fits.pt")

COVARIANCE_TYPES = ["full", "diag", "spherical", "tied_full", "tied_diag", "tied_spherical"]
MODES = ["mle", "map", "niw"]


def make_priors(mode, covariance_type, d):
    if mode == "mle":
        return {}
    if covariance_type in ("full", "tied_full"):
        cov_prior = torch.eye(d, dtype=torch.float64) * 2.0
    elif covariance_type in ("diag", "tied_diag"):
        cov_prior = torch.full((d,), 2.0, dtype=torch.float64)
    else:
        cov_prior = torch.tensor(2.0, dtype=torch.float64)
    kwargs = dict(covariance_prior=cov_prior, degrees_of_freedom_prior=float(d + 2))
    if mode == "niw":
        kwargs["mean_prior"] = torch.zeros(d, dtype=torch.float64)
        kwargs["mean_precision_prior"] = 1.5
    return kwargs


# ===========================================================================
# SufficientStats dataclass
# ===========================================================================

class TestSufficientStats:
    def test_add_sums_all_fields(self):
        a = SufficientStats(
            nk=torch.tensor([1.0, 2.0]),
            s1=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            s2=torch.tensor([[[1.0]], [[2.0]]]),
            n_samples=3,
        )
        b = SufficientStats(
            nk=torch.tensor([3.0, 4.0]),
            s1=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            s2=torch.tensor([[[1.0]], [[2.0]]]),
            n_samples=7,
        )
        c = a + b
        assert torch.equal(c.nk, torch.tensor([4.0, 6.0]))
        assert torch.equal(c.s1, torch.tensor([[2.0, 2.0], [4.0, 4.0]]))
        assert torch.equal(c.s2, torch.tensor([[[2.0]], [[4.0]]]))
        assert c.n_samples == 10

    def test_scale_leaves_n_samples_untouched(self):
        a = SufficientStats(
            nk=torch.tensor([2.0, 4.0]),
            s1=torch.tensor([[2.0], [4.0]]),
            s2=torch.tensor([2.0, 4.0]),
            n_samples=5,
        )
        scaled = a.scale(0.5)
        assert torch.equal(scaled.nk, torch.tensor([1.0, 2.0]))
        assert torch.equal(scaled.s1, torch.tensor([[1.0], [2.0]]))
        assert torch.equal(scaled.s2, torch.tensor([1.0, 2.0]))
        assert scaled.n_samples == 5


# ===========================================================================
# _compute_stats correctness
# ===========================================================================

def _brute_force_stats(X, resp, base_type):
    nk = resp.sum(dim=0)
    s1 = resp.t() @ X
    ybar = s1 / nk.unsqueeze(1)
    diff = X.unsqueeze(1) - ybar.unsqueeze(0)
    if base_type == "full":
        s2 = torch.einsum("nk,nkd,nke->kde", resp, diff, diff)
    elif base_type == "diag":
        s2 = torch.einsum("nk,nkd->kd", resp, diff.pow(2))
    else:
        s2 = torch.einsum("nk,nk->k", resp, diff.pow(2).sum(dim=2))
    return nk, s1, s2


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_compute_stats_matches_brute_force(covariance_type):
    torch.manual_seed(0)
    n, d, k = 97, 6, 4
    X = torch.randn(n, d, dtype=torch.float64)
    logits = torch.randn(n, k, dtype=torch.float64)
    resp = torch.softmax(logits, dim=1)

    gmm = GaussianMixture(n_components=k, n_features=d, covariance_type=covariance_type)
    stats = gmm._compute_stats(X, resp)

    base_type = covariance_type.replace("tied_", "")
    nk, s1, s2 = _brute_force_stats(X, resp, base_type)

    assert torch.allclose(stats.nk, nk, atol=1e-10)
    assert torch.allclose(stats.s1, s1, atol=1e-10)
    assert torch.allclose(stats.s2, s2, atol=1e-8)
    assert stats.n_samples == n


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
@pytest.mark.parametrize("batch_size", [10, 33, 97, 500])
def test_compute_stats_chunked_matches_unchunked(covariance_type, batch_size):
    torch.manual_seed(1)
    n, d, k = 97, 5, 3
    X = torch.randn(n, d, dtype=torch.float64)
    resp = torch.softmax(torch.randn(n, k, dtype=torch.float64), dim=1)

    gmm = GaussianMixture(n_components=k, n_features=d, covariance_type=covariance_type)
    unchunked = gmm._compute_stats(X, resp, batch_size=None)
    chunked = gmm._compute_stats(X, resp, batch_size=batch_size)

    assert torch.allclose(unchunked.nk, chunked.nk, atol=1e-10)
    assert torch.allclose(unchunked.s1, chunked.s1, atol=1e-10)
    assert torch.allclose(unchunked.s2, chunked.s2, atol=1e-10)
    assert unchunked.n_samples == chunked.n_samples


# ===========================================================================
# Bit-for-bit pre/post refactor parity
# ===========================================================================

@pytest.mark.skipif(not os.path.exists(FIXTURE_PATH), reason="phase0 pre-refactor fixture missing")
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
@pytest.mark.parametrize("mode", MODES)
def test_matches_pre_refactor_fixture(covariance_type, mode):
    fixtures = torch.load(FIXTURE_PATH, weights_only=False)
    key = f"{covariance_type}__{mode}"
    fixture = fixtures[key]
    X = fixture["X"]
    d = X.shape[1]

    priors = make_priors(mode, covariance_type, d)
    gmm = GaussianMixture(
        n_components=4,
        n_features=d,
        covariance_type=covariance_type,
        max_iter=15,
        tol=1e-10,
        n_init=1,
        random_state=0,
        init_means="kmeans",
        init_weights="uniform",
        init_covariances="empirical",
        **priors,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gmm.fit(X)

    assert torch.allclose(gmm.weights_, fixture["weights_"], atol=1e-10)
    assert torch.allclose(gmm.means_, fixture["means_"], atol=1e-10)
    assert torch.allclose(gmm.covariances_, fixture["covariances_"], atol=1e-10)
    assert abs(gmm.lower_bound_ - fixture["lower_bound_"].item()) < 1e-10
    assert gmm.n_iter_ == fixture["n_iter_"].item()
    assert gmm.converged_ == bool(fixture["converged_"].item())


# ===========================================================================
# Memory regression (full covariance MLE/MAP/NIW)
# ===========================================================================

_MEMORY_CEILING_BYTES = 2 * 1024**3

_MEMORY_TEST_SCRIPT = textwrap.dedent(
    """
    import resource
    import torch
    from tgmm import GaussianMixture

    torch.manual_seed(0)
    n, d, k = 20000, 64, 10
    X = torch.randn(n, d, dtype=torch.float32)

    kwargs = dict(n_components=k, n_features=d, covariance_type="full",
                  max_iter=2, n_init=1, random_state=0)
    mode = {mode!r}
    if mode in ("map", "niw"):
        kwargs["covariance_prior"] = torch.eye(d, dtype=torch.float32) * 2.0
        kwargs["degrees_of_freedom_prior"] = float(d + 2)
    if mode == "niw":
        kwargs["mean_prior"] = torch.zeros(d, dtype=torch.float32)
        kwargs["mean_precision_prior"] = 1.5

    gmm = GaussianMixture(**kwargs)
    gmm.fit(X)
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """
)


@pytest.mark.parametrize("mode", ["mle", "map", "niw"])
def test_full_covariance_memory_ceiling(mode):
    """
    n=20000, K=10, d=64, float32 full covariance: the pre-refactor
    `diff.unsqueeze(3) * diff.unsqueeze(2)` elementwise product materializes
    an (n, K, d, d) tensor here -- ~3.3 GB, comfortably over the 2 GB
    ceiling (verified by hand against a bare torch.zeros(n, k, d, d) of the
    same size while writing this test). The einsum-based
    `_compute_stats`/`_scatter_about_means` path should need only
    O(n * K * d) ~ tens of MB.

    Runs in a subprocess because `ru_maxrss` is a process-lifetime
    high-water mark -- an earlier, unrelated peak in this test process
    could otherwise mask a real regression. tracemalloc is not used (see
    the design doc): it does not see PyTorch's native CPU allocations at
    all.
    """
    if torch.cuda.is_available():
        pytest.skip("GPU present; this test only exercises the CPU (ru_maxrss) path")

    script = _MEMORY_TEST_SCRIPT.format(mode=mode)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    peak_kb = int(result.stdout.strip().splitlines()[-1])
    peak_bytes = peak_kb * 1024
    assert peak_bytes < _MEMORY_CEILING_BYTES, (
        f"Peak RSS {peak_bytes / 1e9:.2f} GB exceeded the 2 GB ceiling for "
        f"n=20000, K=10, d=64 float32 full-covariance EM ({mode})."
    )


# ===========================================================================
# `algorithm` argument and `cem` backward compatibility
# ===========================================================================

def test_algorithm_default_is_em():
    gmm = GaussianMixture(n_components=2, n_features=2)
    assert gmm.algorithm == "em"
    assert gmm.cem is False


def test_algorithm_rejects_unknown_value():
    with pytest.raises(ValueError):
        GaussianMixture(n_components=2, n_features=2, algorithm="not-a-real-algorithm")


def test_algorithm_accepts_sem():
    # 'sem' was accepted-but-NotImplementedError at construction time in
    # Phase 0 (this test's own history); Phase 2 implements it for real --
    # see tests/test_phase2.py for SEM's actual behavior.
    gmm = GaussianMixture(n_components=2, n_features=2, algorithm="sem")
    assert gmm.algorithm == "sem"


def test_cem_true_warns_deprecation_and_sets_algorithm():
    with pytest.warns(DeprecationWarning):
        gmm = GaussianMixture(n_components=2, n_features=2, cem=True)
    assert gmm.algorithm == "cem"
    assert gmm.cem is True


def test_cem_true_conflicting_algorithm_raises():
    with pytest.raises(ValueError):
        GaussianMixture(n_components=2, n_features=2, cem=True, algorithm="sem")


def test_cem_true_matches_algorithm_cem_result():
    torch.manual_seed(0)
    X = torch.randn(200, 3)
    X[:100] += 5.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        gmm_legacy = GaussianMixture(n_components=2, n_features=3, random_state=0, cem=True)
        gmm_legacy.fit(X)

    gmm_new = GaussianMixture(n_components=2, n_features=3, random_state=0, algorithm="cem")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gmm_new.fit(X)

    assert torch.allclose(gmm_legacy.weights_, gmm_new.weights_, atol=1e-10)
    assert torch.allclose(gmm_legacy.means_, gmm_new.means_, atol=1e-10)
    assert torch.allclose(gmm_legacy.covariances_, gmm_new.covariances_, atol=1e-10)


# ===========================================================================
# batch_size (E-step + stats chunking through a real fit)
# ===========================================================================

@pytest.mark.parametrize("batch_size", [7, 31])
def test_batch_size_matches_unchunked_fit(batch_size):
    torch.manual_seed(0)
    X = torch.randn(150, 3, dtype=torch.float64)
    X[:75] += 4.0

    gmm_full = GaussianMixture(
        n_components=2, n_features=3, random_state=0, max_iter=25, tol=1e-10
    )
    gmm_full.fit(X)

    gmm_chunked = GaussianMixture(
        n_components=2, n_features=3, random_state=0, max_iter=25, tol=1e-10,
        batch_size=batch_size,
    )
    gmm_chunked.fit(X)

    assert torch.allclose(gmm_full.weights_, gmm_chunked.weights_, atol=1e-8)
    assert torch.allclose(gmm_full.means_, gmm_chunked.means_, atol=1e-8)
    assert torch.allclose(gmm_full.covariances_, gmm_chunked.covariances_, atol=1e-8)


# ===========================================================================
# convergence='aitken'
# ===========================================================================

def test_convergence_relative_is_unchanged_default():
    gmm = GaussianMixture(n_components=2, n_features=2)
    assert gmm.convergence == "relative"


def test_convergence_rejects_unknown_value():
    with pytest.raises(ValueError):
        GaussianMixture(n_components=2, n_features=2, convergence="not-a-rule")


def test_convergence_aitken_converges_to_a_similar_solution():
    torch.manual_seed(0)
    X = torch.randn(300, 2, dtype=torch.float64)
    X[:150] += torch.tensor([5.0, 5.0], dtype=torch.float64)

    gmm_relative = GaussianMixture(
        n_components=2, n_features=2, random_state=0, max_iter=200, tol=1e-8
    )
    gmm_relative.fit(X)

    gmm_aitken = GaussianMixture(
        n_components=2, n_features=2, random_state=0, max_iter=200, tol=1e-8,
        convergence="aitken",
    )
    gmm_aitken.fit(X)

    assert gmm_aitken.converged_
    assert abs(gmm_aitken.lower_bound_ - gmm_relative.lower_bound_) < 1e-4


# ===========================================================================
# save/load round-trip for the new fields
# ===========================================================================

def test_save_load_round_trips_new_fields(tmp_path):
    gmm = GaussianMixture(
        n_components=2, n_features=2, algorithm="cem", batch_size=16,
        convergence="aitken",
    )
    X = torch.randn(50, 2)
    gmm.fit(X)

    path = str(tmp_path / "model.pt")
    gmm.save(path)
    loaded = GaussianMixture.load(path)

    assert loaded.algorithm == "cem"
    assert loaded.batch_size == 16
    assert loaded.convergence == "aitken"
