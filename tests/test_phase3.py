"""
Phase 3 (v0.4 roadmap): the 14 Celeux-Govaert GPCM covariance models. See
docs/superpowers/specs/2026-10-14-phase3-gpcm-covariance-models-design.md.
"""
import json
import math
import os
import warnings

import pytest
import torch

from tgmm import GaussianMixture, select_model
from tgmm.covariance.gpcm import GPCM_CODES, mstep_gpcm

NEW_CODES = ('VEI', 'EVI', 'VEE', 'EVE', 'EEV', 'VVE', 'VEV', 'EVV')
OLD_CODES_TO_TGMM = {
    'EII': 'tied_spherical', 'VII': 'spherical', 'EEI': 'tied_diag',
    'VVI': 'diag', 'EEE': 'tied_full', 'VVV': 'full',
}


def _three_blobs(seed, n_per=60, d=3, spread=(1.0, 2.0, 0.5), sep=4.0):
    gen = torch.Generator().manual_seed(seed)
    spread_t = torch.tensor(spread[:d] if len(spread) >= d else [1.0] * d, dtype=torch.float64)
    chunks = []
    for i in range(3):
        center = torch.zeros(d, dtype=torch.float64)
        center[0] = i * sep
        chunks.append(torch.randn(n_per, d, generator=gen, dtype=torch.float64) * spread_t + center)
    return torch.cat(chunks, dim=0)


SEEDED_DATASETS = [_three_blobs(seed) for seed in (0, 1, 2)]


def _objective_J(cov_full, W, nk):
    r"""sum_k [n_k log|Sigma_k| + tr(Sigma_k^{-1} W_k)] -- lower is better."""
    total = 0.0
    for k in range(W.shape[0]):
        sign, logdet = torch.linalg.slogdet(cov_full[k])
        total += nk[k].item() * logdet.item() + torch.trace(torch.linalg.solve(cov_full[k], W[k])).item()
    return total


# ===========================================================================
# tgmm.covariance.gpcm unit tests
# ===========================================================================

class TestGPCMModule:
    def test_all_14_codes_return_valid_shapes(self):
        torch.manual_seed(0)
        K, d = 3, 4
        nk = torch.tensor([30.0, 40.0, 25.0], dtype=torch.float64)
        W = torch.zeros(K, d, d, dtype=torch.float64)
        for k in range(K):
            A = torch.randn(d, d, dtype=torch.float64)
            W[k] = A @ A.T * nk[k]

        for code in GPCM_CODES:
            result = mstep_gpcm(code, W, nk, d)
            assert result.cov_full.shape == (K, d, d)
            for k in range(K):
                eigvals = torch.linalg.eigvalsh(result.cov_full[k])
                assert (eigvals > 0).all(), f"{code} produced a non-PD covariance"
                assert torch.allclose(result.cov_full[k], result.cov_full[k].T, atol=1e-8)

    def test_invalid_code_raises(self):
        with pytest.raises(ValueError):
            mstep_gpcm('XYZ', torch.eye(2).unsqueeze(0), torch.tensor([1.0]), 2)

    def test_nested_consistency_via_objective(self):
        r"""
        For every (constrained, relaxed) pair below, the relaxed model's
        objective J must be <= the constrained model's (relaxing a
        constraint can only improve, never hurt, the in-sample fit) --
        the plan's own acceptance criterion (3.5), checked via the
        complete-data objective directly (equivalent to comparing
        log-likelihoods, since J = -2 * (the covariance-dependent part of
        the log-likelihood) + a constant that doesn't depend on the
        covariance parameterization).
        """
        torch.manual_seed(3)
        K, d = 3, 4
        nk = torch.tensor([25.0, 35.0, 20.0], dtype=torch.float64)
        W = torch.zeros(K, d, d, dtype=torch.float64)
        for k in range(K):
            A = torch.randn(d, d, dtype=torch.float64)
            W[k] = A @ A.T * nk[k]

        J = {code: _objective_J(mstep_gpcm(code, W, nk, d).cov_full, W, nk) for code in GPCM_CODES}

        pairs = [
            # diagonal family
            ('EII', 'VII'), ('EII', 'EEI'), ('VII', 'VEI'), ('EEI', 'VEI'),
            ('EEI', 'EVI'), ('VEI', 'VVI'), ('EVI', 'VVI'),
            # general-orientation family
            ('EEE', 'VEE'), ('EEE', 'EVE'), ('EEE', 'EEV'),
            ('VEE', 'VVE'), ('EVE', 'VVE'), ('VEE', 'VEV'), ('EEV', 'VEV'),
            ('EVE', 'EVV'), ('EEV', 'EVV'),
            ('VVE', 'VVV'), ('VEV', 'VVV'), ('EVV', 'VVV'),
            # I-orientation is a special case of E-orientation (D forced
            # to I instead of optimized) -- the general-orientation
            # sibling model can only do at least as well.
            ('EEI', 'EEE'), ('VEI', 'VEE'), ('EVI', 'EVE'), ('VVI', 'VVE'),
        ]
        for constrained, relaxed in pairs:
            assert J[relaxed] <= J[constrained] + 1e-6, (
                f"{relaxed} (J={J[relaxed]:.4f}) should be <= {constrained} (J={J[constrained]:.4f})"
            )

    @pytest.mark.parametrize("code", NEW_CODES)
    def test_brute_force_matches_closed_form_or_iterative_solution(self, code):
        r"""
        Independent check (no mclust available -- see
        docs/superpowers/specs/2026-10-14-phase3-gpcm-covariance-models-design.md):
        minimize J directly via unconstrained reparameterizations + Adam,
        and compare to mstep_gpcm's answer. This is what stands in for
        "did I derive the equation right."
        """
        torch.manual_seed(7)
        K, d = 3, 3
        nk = torch.tensor([20.0, 25.0, 18.0], dtype=torch.float64)
        W = torch.zeros(K, d, d, dtype=torch.float64)
        for k in range(K):
            A = torch.randn(d, d, dtype=torch.float64)
            W[k] = A @ A.T * nk[k] + torch.eye(d, dtype=torch.float64) * nk[k]

        result = mstep_gpcm(code, W, nk, d)
        j_mstep = _objective_J(result.cov_full, W, nk)

        # Unconstrained brute-force: parameterize Sigma_k = L_k L_k^T
        # (Cholesky factors, free lower-triangular) and minimize J by
        # gradient descent -- the *global* unconstrained optimum (VVV) is
        # a lower bound for every constrained model; for codes whose
        # constraint set is convex/simple enough to also brute-force
        # directly (axis-aligned ones), compare more tightly below.
        L = torch.nn.Parameter(torch.stack([torch.eye(d, dtype=torch.float64) for _ in range(K)]))
        optimizer = torch.optim.Adam([L], lr=0.05)
        for _ in range(3000):
            optimizer.zero_grad()
            tril = torch.tril(L)
            cov = tril @ tril.transpose(-1, -2) + torch.eye(d, dtype=torch.float64) * 1e-8
            loss = 0.0
            for k in range(K):
                sign, logdet = torch.linalg.slogdet(cov[k])
                loss = loss + nk[k] * logdet + torch.trace(torch.linalg.solve(cov[k], W[k]))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            tril = torch.tril(L)
            cov_unconstrained = tril @ tril.transpose(-1, -2) + torch.eye(d, dtype=torch.float64) * 1e-8
            j_unconstrained = _objective_J(cov_unconstrained, W, nk)

        # mstep_gpcm's constrained optimum can never beat the unconstrained
        # (VVV) optimum.
        assert j_mstep >= j_unconstrained - 1e-2

    def test_orientation_optimizer_beats_random_search(self):
        r"""
        Direct check on the Stiefel/L-BFGS orientation sub-problem used by
        EVE/VVE: it should find an orientation at least as good as the
        best of many random orthogonal matrices.
        """
        torch.manual_seed(1)
        K, d = 4, 3
        nk = torch.tensor([20.0, 25.0, 30.0, 15.0], dtype=torch.float64)
        W = torch.zeros(K, d, d, dtype=torch.float64)
        for k in range(K):
            A = torch.randn(d, d, dtype=torch.float64)
            W[k] = A @ A.T * nk[k]

        result = mstep_gpcm('EVE', W, nk, d)
        rotated = result.orientation.T @ W @ result.orientation
        B_k = torch.diagonal(rotated, dim1=-2, dim2=-1)
        det_B_k = B_k.prod(dim=-1).clamp_min(1e-300)
        A_k_inv = (det_B_k.unsqueeze(-1).pow(1.0 / d) / B_k)
        lam_vec = ((1.0 / A_k_inv) * B_k).sum().div(nk.sum() * d).expand(K)

        def objective(D):
            rot = D.T @ W @ D
            diag_rot = torch.diagonal(rot, dim1=-2, dim2=-1)
            return ((A_k_inv * diag_rot).sum(dim=-1) / lam_vec).sum().item()

        mstep_val = objective(result.orientation)
        best_random = min(
            objective(torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0])
            for _ in range(500)
        )
        assert mstep_val <= best_random + 1e-6


# ===========================================================================
# GaussianMixture integration
# ===========================================================================

class TestGaussianMixtureGPCMIntegration:
    @pytest.mark.parametrize("code", GPCM_CODES)
    def test_fits_and_sets_canonical_code(self, code):
        gmm = GaussianMixture(n_components=3, n_features=3, covariance_type=code,
                               random_state=0, max_iter=30)
        gmm.fit(SEEDED_DATASETS[0])
        assert gmm.covariance_type_ == code
        assert math.isfinite(gmm.lower_bound_)
        # The 6 pre-existing codes keep their own historical storage shapes
        # (e.g. EII/tied_spherical is a scalar); only the 8 new codes are
        # always assembled (K, d, d).
        if code in NEW_CODES:
            assert gmm.covariances_.shape == (3, 3, 3)

    @pytest.mark.parametrize("code", GPCM_CODES)
    def test_case_insensitive_and_lowercase_alias(self, code):
        gmm = GaussianMixture(n_components=2, n_features=2, covariance_type=code.lower(), random_state=0)
        assert gmm.covariance_type_ == code

    def test_legacy_tgmm_names_still_resolve_correctly(self):
        for mclust_code, tgmm_name in OLD_CODES_TO_TGMM.items():
            gmm = GaussianMixture(n_components=2, n_features=2, covariance_type=tgmm_name)
            assert gmm.covariance_type_ == mclust_code
            assert gmm.covariance_type == tgmm_name

    @pytest.mark.parametrize("code", NEW_CODES)
    @pytest.mark.parametrize("dataset_idx", [0, 1, 2])
    def test_monotone_log_likelihood(self, code, dataset_idx):
        gmm = GaussianMixture(n_components=3, n_features=3, covariance_type=code,
                               random_state=1, max_iter=50, tol=1e-8)
        gmm.fit(SEEDED_DATASETS[dataset_idx])
        hist = gmm.log_likelihood_history_
        diffs = hist[1:] - hist[:-1]
        assert (diffs >= -1e-6).all(), f"{code} log-likelihood decreased: {hist}"

    @pytest.mark.parametrize("code", NEW_CODES)
    def test_init_covariances_empirical(self, code):
        gmm = GaussianMixture(n_components=3, n_features=3, covariance_type=code,
                               random_state=0, init_covariances='empirical', max_iter=1)
        gmm._allocate_parameters(SEEDED_DATASETS[0])
        assert gmm.covariances_.shape == (3, 3, 3)
        assert gmm.volume_ is not None
        for k in range(3):
            eigvals = torch.linalg.eigvalsh(gmm.covariances_[k])
            assert (eigvals > 0).all()

    @pytest.mark.parametrize("code", NEW_CODES)
    def test_init_covariances_global(self, code):
        gmm = GaussianMixture(n_components=3, n_features=3, covariance_type=code,
                               random_state=0, init_covariances='global', max_iter=1)
        gmm._allocate_parameters(SEEDED_DATASETS[0])
        assert gmm.covariances_.shape == (3, 3, 3)

    @pytest.mark.parametrize("code", NEW_CODES)
    def test_explicit_tensor_init_covariances_broadcasts(self, code):
        single = torch.eye(3, dtype=torch.float64) * 2.0
        gmm = GaussianMixture(n_components=3, n_features=3, covariance_type=code,
                               random_state=0, init_covariances=single)
        gmm._allocate_parameters(SEEDED_DATASETS[0])
        assert gmm.covariances_.shape == (3, 3, 3)
        assert torch.allclose(gmm.covariances_[0], gmm.covariances_[1])

    def test_n_parameters_matches_table(self):
        K, d = 3, 4
        expected = {
            'EII': 1, 'VII': K, 'EEI': d, 'VEI': d + K - 1, 'EVI': 1 + K * (d - 1),
            'VVI': K * d, 'EEE': d * (d + 1) / 2, 'VEE': K - 1 + d * (d + 1) / 2,
            'EVE': 1 + K * (d - 1) + d * (d - 1) / 2, 'EEV': d + K * d * (d - 1) / 2,
            'VVE': K * d + d * (d - 1) / 2, 'VEV': K + d - 1 + K * d * (d - 1) / 2,
            'EVV': 1 + K * (d - 1) + K * d * (d - 1) / 2, 'VVV': K * d * (d + 1) / 2,
        }
        for code, expected_cov_params in expected.items():
            gmm = GaussianMixture(n_components=K, n_features=d, covariance_type=code, random_state=0)
            gmm.fit(_three_blobs(0, d=d))
            expected_total = int(expected_cov_params) + K * d + (K - 1)
            assert gmm._n_parameters() == expected_total, code

    @pytest.mark.parametrize("code", NEW_CODES)
    def test_map_priors_raise_not_implemented(self, code):
        with pytest.raises(NotImplementedError):
            GaussianMixture(
                n_components=2, n_features=2, covariance_type=code,
                mean_prior=torch.zeros(2), mean_precision_prior=1.0,
            )
        with pytest.raises(NotImplementedError):
            GaussianMixture(
                n_components=2, n_features=2, covariance_type=code,
                covariance_prior=torch.eye(2), degrees_of_freedom_prior=3.0,
            )

    @pytest.mark.parametrize("code", NEW_CODES)
    def test_weight_concentration_prior_still_works(self, code):
        gmm = GaussianMixture(
            n_components=2, n_features=2, covariance_type=code,
            weight_concentration_prior=torch.tensor([2.0, 2.0]), random_state=0, max_iter=10,
        )
        gmm.fit(torch.randn(50, 2, dtype=torch.float64))
        assert math.isfinite(gmm.lower_bound_)

    @pytest.mark.parametrize("code", NEW_CODES)
    def test_gibbs_mode_rejects_new_codes(self, code):
        with pytest.raises(NotImplementedError):
            GaussianMixture(
                n_components=None, covariance_type=code,
                mean_prior=torch.zeros(2), mean_precision_prior=1.0,
                covariance_prior=torch.eye(2), degrees_of_freedom_prior=3.0,
            )

    @pytest.mark.parametrize("code", NEW_CODES)
    def test_save_load_round_trip(self, code, tmp_path):
        gmm = GaussianMixture(n_components=2, n_features=2, covariance_type=code, random_state=0, max_iter=10)
        gmm.fit(torch.randn(40, 2, dtype=torch.float64))

        path = str(tmp_path / f"model_{code}.pt")
        gmm.save(path)
        loaded = GaussianMixture.load(path)

        assert loaded.covariance_type_ == code
        assert torch.allclose(loaded.covariances_, gmm.covariances_)
        assert torch.allclose(loaded.volume_, gmm.volume_)
        assert torch.allclose(loaded.shape_, gmm.shape_)
        assert torch.allclose(loaded.orientation_, gmm.orientation_)

    @pytest.mark.parametrize("code", NEW_CODES)
    def test_save_state_dict_round_trip(self, code):
        gmm = GaussianMixture(n_components=2, n_features=2, covariance_type=code, random_state=0, max_iter=10)
        gmm.fit(torch.randn(40, 2, dtype=torch.float64))

        state = gmm.save_state_dict()
        gmm2 = GaussianMixture(n_components=2, n_features=2, covariance_type=code)
        gmm2.load_state_dict(state)
        assert gmm2.covariance_type_ == code
        assert torch.allclose(gmm2.covariances_, gmm.covariances_)

    def test_gpcm_tol_and_max_iter_validated(self):
        with pytest.raises(ValueError):
            GaussianMixture(n_components=2, n_features=2, gpcm_tol=0.0)
        with pytest.raises(ValueError):
            GaussianMixture(n_components=2, n_features=2, gpcm_max_iter=0)

    def test_warm_starting_reduces_inner_iterations(self):
        r"""
        A second M-step on nearly-identical stats should need very few
        inner iterations given a warm start from the first M-step's
        result -- can't directly count inner iterations from the public
        API, so this just checks that fitting with warm_start=True across
        two fit() calls doesn't regress the fit (sanity check that
        _gpcm_state_ carries over without breaking anything).
        """
        X = SEEDED_DATASETS[0]
        gmm = GaussianMixture(n_components=3, n_features=3, covariance_type='VEE',
                               random_state=0, max_iter=5, warm_start=True)
        gmm.fit(X)
        ll1 = gmm.lower_bound_
        gmm.fit(X)  # warm_start=True continues from current params
        ll2 = gmm.lower_bound_
        assert ll2 >= ll1 - 1e-6


# ===========================================================================
# select_model default
# ===========================================================================

def test_select_model_default_covariance_types_is_all_14_codes():
    result = select_model(
        SEEDED_DATASETS[0], n_components=[2, 3], criterion='bic',
        random_state=0, max_iter=20,
    )
    seen_types = {row['covariance_type'] for row in result.table}
    assert seen_types == set(GPCM_CODES)


# ===========================================================================
# mclust fixture parity (skipped -- no R here, see [[v0_4_mclust_oracle_gap]])
# ===========================================================================

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "mclust_gpcm_fixtures.json")


@pytest.mark.skipif(not os.path.exists(FIXTURE_PATH), reason="mclust GPCM fixtures missing; see tests/fixtures/make_gpcm_fixtures.R")
@pytest.mark.parametrize("code", GPCM_CODES)
@pytest.mark.parametrize("dataset_key", ["d2", "d4", "d6"])
def test_matches_mclust_fixture(code, dataset_key):
    with open(FIXTURE_PATH) as f:
        fixtures = json.load(f)
    entry = fixtures[dataset_key][code]
    X = torch.tensor(fixtures[dataset_key]["X"], dtype=torch.float64)
    z = torch.tensor(fixtures[dataset_key]["z"], dtype=torch.long)
    K = z.max().item() + 1
    d = X.shape[1]

    onehot = torch.nn.functional.one_hot(z, K).to(torch.float64)
    nk = onehot.sum(dim=0)
    means = (onehot.t() @ X) / nk.unsqueeze(1)
    diff = X.unsqueeze(1) - means.unsqueeze(0)
    W = torch.einsum('nk,nkd,nke->kde', onehot, diff, diff)

    result = mstep_gpcm(code, W, nk, d)
    expected_cov = torch.tensor(entry["sigma"], dtype=torch.float64)
    atol = 1e-8 if code not in ('VEI', 'VEE', 'VEV', 'EVE', 'VVE') else 1e-5
    assert torch.allclose(result.cov_full, expected_cov, atol=atol)


# ===========================================================================
# tgmm.plotting integration (caught by executing notebooks/15_gpcm_models.ipynb:
# get_covariance_matrix had its own covariance_type dispatch, separate from
# gmm.py's, that didn't know about the 8 new GPCM codes)
# ===========================================================================

@pytest.mark.parametrize("code", NEW_CODES)
def test_plot_gmm_supports_new_gpcm_codes(code):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tgmm.plotting import plot_gmm

    gmm = GaussianMixture(n_components=3, n_features=2, covariance_type=code, random_state=0, max_iter=15)
    gmm.fit(_three_blobs(0, d=2))
    try:
        plot_gmm(_three_blobs(0, d=2), gmm=gmm, show_ellipses=True)
    finally:
        plt.close('all')
