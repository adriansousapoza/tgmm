r"""
Celeux-Govaert (1995) Gaussian Parsimonious Clustering Models (GPCM):
M-step updates for all 14 eigenvalue-decomposed covariance
parameterizations :math:`\Sigma_k = \lambda_k D_k A_k D_k^T`.

Every update here is derived from the complete-data objective

.. math::
    J = \sum_k \left[ n_k \log|\Sigma_k| + \mathrm{tr}(\Sigma_k^{-1} W_k) \right]
      = \sum_k \left[ n_k d \log(\lambda_k) +
          \frac{1}{\lambda_k} \mathrm{tr}(A_k^{-1} D_k^T W_k D_k) \right]

(to be minimized), not from `mclust`'s source -- see
docs/superpowers/specs/2026-10-14-phase3-gpcm-covariance-models-design.md
for the full derivation of each formula below.

`mstep_gpcm` implements all 14 codes for a uniform, testable interface,
but `GaussianMixture` only calls it for the 8 codes without an existing
dedicated (and already-tested) implementation in `tgmm/gmm.py`
(`full`/`diag`/`spherical`/`tied_full`/`tied_diag`/`tied_spherical`, i.e.
VVV/VVI/VII/EEE/EEI/EII, keep using their existing code paths unchanged).
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal

GPCM_CODES = (
    'EII', 'VII', 'EEI', 'VEI', 'EVI', 'VVI',
    'EEE', 'VEE', 'EVE', 'EEV', 'VVE', 'VEV', 'EVV', 'VVV',
)

_ITERATIVE_CODES = ('VEI', 'VEE', 'VEV', 'EVE', 'VVE')


@dataclass
class GPCMResult:
    r"""
    The decomposition :math:`\Sigma_k = \lambda_k D_k A_k D_k^T` an
    `mstep_gpcm` call produced.

    Attributes
    ----------
    volume : torch.Tensor
        Scalar (E-volume models) or shape `(K,)` (V-volume models).
    shape : torch.Tensor
        Shape `(d,)` (E-shape models) or `(K, d)` (V-shape models).
        Diagonal of `A_k`, determinant 1 for the 8 codes `GaussianMixture`
        actually uses this for (`VEI`, `EVI`, `VEE`, `EVE`, `EEV`, `VVE`,
        `VEV`, `EVV`). The 6 classic codes (`EII`, `VII`, `EEI`, `VVI`,
        `EEE`, `VVV`) are also available through `mstep_gpcm`, but there
        `volume` is 1 and `shape` holds the unnormalized variances /
        eigenvalues (the decomposition is trivially absorbed into `shape`).
    orientation : torch.Tensor
        Shape `(d, d)` (E- or I-orientation models) or `(K, d, d)`
        (V-orientation models).
    cov_full : torch.Tensor
        Assembled `(K, d, d)` covariance matrices, `reg_covar * I` already
        added.
    """

    volume: torch.Tensor
    shape: torch.Tensor
    orientation: torch.Tensor
    cov_full: torch.Tensor


def _eigh_descending(M: torch.Tensor):
    r"""`torch.linalg.eigh` with eigenvalues/eigenvectors reordered descending."""
    eigvals, eigvecs = torch.linalg.eigh(M)
    idx = torch.flip(torch.argsort(eigvals, dim=-1), dims=[-1])
    eigvals = torch.gather(eigvals, -1, idx)
    eigvecs = torch.gather(eigvecs, -1, idx.unsqueeze(-2).expand_as(eigvecs))
    return eigvals, eigvecs


def _assemble_full(volume, shape, orientation, K: int, d: int, reg_covar: float) -> torch.Tensor:
    r"""
    `Sigma_k = volume_k * orientation_k @ diag(shape_k) @ orientation_k^T`
    (or `orientation`/`shape`/`volume` broadcast across k for E-models),
    plus `reg_covar * I`.
    """
    dtype, device = shape.dtype, shape.device
    vol = volume if volume.dim() > 0 else volume.expand(K)
    shp = shape if shape.dim() > 1 else shape.unsqueeze(0).expand(K, d)
    orient = orientation if orientation.dim() > 2 else orientation.unsqueeze(0).expand(K, d, d)

    diag_shape = torch.diag_embed(shp)  # (K, d, d)
    cov = vol.view(K, 1, 1) * (orient @ diag_shape @ orient.transpose(-1, -2))
    cov = cov + reg_covar * torch.eye(d, dtype=dtype, device=device).unsqueeze(0)
    return cov


def mstep_gpcm(
    code: str,
    W: torch.Tensor,
    nk: torch.Tensor,
    d: int,
    state: Optional[GPCMResult] = None,
    reg_covar: float = 0.0,
    gpcm_tol: float = 1e-6,
    gpcm_max_iter: int = 20,
) -> GPCMResult:
    r"""
    One M-step update for GPCM covariance code `code`.

    Parameters
    ----------
    code : str
        One of the 14 mclust codes (case-insensitive), see `GPCM_CODES`.
    W : torch.Tensor
        Per-component scatter, shape `(K, d, d)` -- `W_k = sum_i r_ik
        (x_i - ybar_k)(x_i - ybar_k)^T` (Phase 0's `stats.s2` for `full`
        covariance).
    nk : torch.Tensor
        Effective per-component sample counts, shape `(K,)`.
    d : int
        Number of features.
    state : GPCMResult, optional
        Previous M-step's result, used to warm-start the 5 iterative
        codes (`VEI`, `VEE`, `VEV`, `EVE`, `VVE`). Ignored by the other 9.
    reg_covar : float, optional
        Added to the assembled covariance's diagonal (not the
        decomposition). (default: 0.0)
    gpcm_tol : float, optional
        Relative-change stopping tolerance for the iterative codes'
        complete-data objective. (default: 1e-6)
    gpcm_max_iter : int, optional
        Maximum inner iterations for the iterative codes. (default: 20)

    Returns
    -------
    GPCMResult
    """
    code = code.upper()
    if code not in GPCM_CODES:
        raise ValueError(f"code must be one of {GPCM_CODES}, got {code!r}")

    K = W.shape[0]
    n = nk.sum()
    dtype, device = W.dtype, W.device
    I_d = torch.eye(d, dtype=dtype, device=device)

    diagW = torch.diagonal(W, dim1=-2, dim2=-1)  # (K, d) -- diag(W_k)

    # ===================================================================
    # Diagonal (axis-aligned, orientation = I) models: EII, VII, EEI, VVI
    # (the last 4 already have dedicated, tested code in tgmm/gmm.py --
    # implemented here too so tests can exercise all 14 codes uniformly).
    # ===================================================================
    if code == 'EII':
        lam = (diagW.sum()) / (n * d)
        volume, shape, orientation = lam, torch.ones(d, dtype=dtype, device=device), I_d
    elif code == 'VII':
        lam_k = diagW.sum(dim=-1) / (nk * d)
        volume, shape, orientation = lam_k, torch.ones(d, dtype=dtype, device=device), I_d
    elif code == 'EEI':
        b = diagW.sum(dim=0) / n  # (d,) -- this IS lambda * A already (free diagonal)
        volume, shape, orientation = torch.tensor(1.0, dtype=dtype, device=device), b, I_d
    elif code == 'VVI':
        b_k = diagW / nk.unsqueeze(-1)  # (K, d)
        volume, shape, orientation = torch.ones(K, dtype=dtype, device=device), b_k, I_d

    # ===================================================================
    # VEI: volume varies, shape shared, axis-aligned. Iterative.
    # ===================================================================
    elif code == 'VEI':
        A = state.shape.clone() if (state is not None and state.shape.dim() == 1) else torch.ones(d, dtype=dtype, device=device)
        lam_k = None
        prev_J = None
        for _ in range(gpcm_max_iter):
            A_inv = 1.0 / A
            lam_k = (diagW * A_inv).sum(dim=-1) / (nk * d)
            B = (diagW / lam_k.unsqueeze(-1)).sum(dim=0)
            A = B / B.prod().clamp_min(1e-300).pow(1.0 / d)

            J = (nk * d * torch.log(lam_k)).sum() + (diagW * (1.0 / A) / lam_k.unsqueeze(-1)).sum()
            if prev_J is not None and abs(prev_J - J) < gpcm_tol * abs(prev_J).clamp_min(1e-300):
                break
            prev_J = J
        volume, shape, orientation = lam_k, A, I_d

    # ===================================================================
    # EVI: volume shared, shape varies, axis-aligned. Closed form.
    # ===================================================================
    elif code == 'EVI':
        B_k = diagW  # (K, d)
        det_B_k = B_k.prod(dim=-1).clamp_min(1e-300)  # (K,)
        A_k = B_k / det_B_k.unsqueeze(-1).pow(1.0 / d)
        lam = det_B_k.pow(1.0 / d).sum() / n
        volume, shape, orientation = lam, A_k, I_d

    # ===================================================================
    # EEV: volume & shape shared, orientation varies. Closed form.
    # ===================================================================
    elif code == 'EEV':
        eigvals, eigvecs = _eigh_descending(W)  # (K, d), (K, d, d)
        Omega = eigvals.sum(dim=0)  # (d,)
        det_Omega = Omega.prod().clamp_min(1e-300)
        A = Omega / det_Omega.pow(1.0 / d)
        lam = det_Omega.pow(1.0 / d) / n
        volume, shape, orientation = lam, A, eigvecs

    # ===================================================================
    # VEV: shape shared, volume & orientation vary. Iterative (D fixed
    # per-k from the data; lambda_k/A alternate).
    # ===================================================================
    elif code == 'VEV':
        eigvals, eigvecs = _eigh_descending(W)  # (K, d), (K, d, d) -- Omega_k, D_k
        A = state.shape.clone() if (state is not None and state.shape.dim() == 1) else torch.ones(d, dtype=dtype, device=device)
        lam_k = None
        prev_J = None
        for _ in range(gpcm_max_iter):
            A_inv = 1.0 / A
            lam_k = (eigvals * A_inv).sum(dim=-1) / (nk * d)
            Omega_sum = (eigvals / lam_k.unsqueeze(-1)).sum(dim=0)
            A = Omega_sum / Omega_sum.prod().clamp_min(1e-300).pow(1.0 / d)

            J = (nk * d * torch.log(lam_k)).sum() + (eigvals * (1.0 / A) / lam_k.unsqueeze(-1)).sum()
            if prev_J is not None and abs(prev_J - J) < gpcm_tol * abs(prev_J).clamp_min(1e-300):
                break
            prev_J = J
        volume, shape, orientation = lam_k, A, eigvecs

    # ===================================================================
    # EVV: volume shared, shape & orientation vary. Closed form.
    # ===================================================================
    elif code == 'EVV':
        det_W_k = torch.linalg.det(W).clamp_min(1e-300)  # (K,)
        C_k = W / det_W_k.view(K, 1, 1).pow(1.0 / d)
        lam = det_W_k.pow(1.0 / d).sum() / n
        eigvals, eigvecs = _eigh_descending(C_k)  # exposed shape_/orientation_ only
        volume, shape, orientation = lam, eigvals, eigvecs
        cov_full = lam * C_k + reg_covar * I_d.unsqueeze(0)
        return GPCMResult(volume=volume, shape=shape, orientation=orientation, cov_full=cov_full)

    # ===================================================================
    # VEE: shape & orientation shared (C = D A D^T, det 1), volume
    # varies. Iterative.
    # ===================================================================
    elif code == 'VEE':
        C = state.orientation.clone() @ torch.diag_embed(state.shape.clone()) @ state.orientation.clone().transpose(-1, -2) \
            if (state is not None and state.orientation.dim() == 2) else I_d.clone()
        lam_k = None
        prev_J = None
        for _ in range(gpcm_max_iter):
            C_inv = torch.linalg.inv(C)
            lam_k = torch.einsum('kij,ji->k', W, C_inv) / (nk * d)
            B = (W / lam_k.view(K, 1, 1)).sum(dim=0)
            det_B = torch.linalg.det(B).clamp_min(1e-300)
            C = B / det_B.pow(1.0 / d)

            C_inv = torch.linalg.inv(C)  # J is evaluated at the *updated* (lam_k, C)
            J = (nk * d * torch.log(lam_k)).sum() + torch.einsum('kij,ji->k', W, C_inv).div(lam_k).sum()
            if prev_J is not None and abs(prev_J - J) < gpcm_tol * abs(prev_J).clamp_min(1e-300):
                break
            prev_J = J
        eigvals, eigvecs = _eigh_descending(C)
        volume, shape, orientation = lam_k, eigvals, eigvecs
        cov_full = lam_k.view(K, 1, 1) * C.unsqueeze(0) + reg_covar * I_d.unsqueeze(0)
        return GPCMResult(volume=volume, shape=shape, orientation=orientation, cov_full=cov_full)

    # ===================================================================
    # EVE / VVE: orientation shared, shape varies, volume shared (EVE) or
    # varies (VVE). Iterative; the orientation sub-step is a Stiefel-
    # manifold optimization (no closed form).
    # ===================================================================
    elif code in ('EVE', 'VVE'):
        D = state.orientation.clone() if (state is not None and state.orientation.dim() == 2) else I_d.clone()
        A_k = None
        lam = None
        lam_k = None
        prev_J = None
        for _ in range(gpcm_max_iter):
            rotated = D.T @ W @ D  # (K, d, d)
            B_k = torch.diagonal(rotated, dim1=-2, dim2=-1)  # (K, d)
            det_B_k = B_k.prod(dim=-1).clamp_min(1e-300)
            A_k = B_k / det_B_k.unsqueeze(-1).pow(1.0 / d)
            A_k_inv = 1.0 / A_k

            if code == 'EVE':
                lam = (A_k_inv * B_k).sum() / (n * d)
                lam_vec = lam.expand(K)
            else:  # VVE
                lam_k = (A_k_inv * B_k).sum(dim=-1) / (nk * d)
                lam_vec = lam_k

            D = _optimize_orientation(W, A_k_inv, lam_vec, D)

            rotated = D.T @ W @ D
            B_k_check = torch.diagonal(rotated, dim1=-2, dim2=-1)
            J = (nk * d * torch.log(lam_vec)).sum() + (A_k_inv * B_k_check / lam_vec.unsqueeze(-1)).sum()
            if prev_J is not None and abs(prev_J - J) < gpcm_tol * abs(prev_J).clamp_min(1e-300):
                break
            prev_J = J

        volume = lam if code == 'EVE' else lam_k
        shape, orientation = A_k, D

    # ===================================================================
    # EEE, VVV: existing tgmm types ('tied_full', 'full'), implemented
    # here too only so every code is available through one interface for
    # testing -- tgmm/gmm.py keeps using its own dedicated code.
    # ===================================================================
    elif code == 'EEE':
        C = W.sum(dim=0) / n
        eigvals, eigvecs = _eigh_descending(C)
        volume, shape, orientation = torch.tensor(1.0, dtype=dtype, device=device), eigvals, eigvecs
        cov_full = C.unsqueeze(0).expand(K, d, d).clone() + reg_covar * I_d.unsqueeze(0)
        return GPCMResult(volume=volume, shape=shape, orientation=orientation, cov_full=cov_full)
    elif code == 'VVV':
        C_k = W / nk.view(K, 1, 1)
        eigvals, eigvecs = _eigh_descending(C_k)
        volume, shape, orientation = torch.ones(K, dtype=dtype, device=device), eigvals, eigvecs
        cov_full = C_k + reg_covar * I_d.unsqueeze(0)
        return GPCMResult(volume=volume, shape=shape, orientation=orientation, cov_full=cov_full)
    else:  # pragma: no cover -- exhaustive above
        raise AssertionError(code)

    cov_full = _assemble_full(volume, shape, orientation, K, d, reg_covar)
    return GPCMResult(volume=volume, shape=shape, orientation=orientation, cov_full=cov_full)


def _optimize_orientation(
    W: torch.Tensor, A_inv: torch.Tensor, lam: torch.Tensor, D_init: torch.Tensor,
    n_steps: int = 20,
) -> torch.Tensor:
    r"""
    `argmin_{D orthogonal} sum_k tr(diag(A_inv[k]) @ D^T @ W[k] @ D) / lam[k]`,
    via `torch.nn.utils.parametrizations.orthogonal` + up to `n_steps`
    `torch.optim.LBFGS` steps, initialized at `D_init`. A GEM step: `D_init`
    is kept unchanged unless the objective strictly decreases. Reference
    implementation per the plan (Browne & McNicholas 2014's Stiefel-manifold
    accelerated line search, or the classical Flury-Gautschi pairwise
    rotations, may replace this later if profiling shows it's the
    bottleneck).
    """
    d = D_init.shape[0]

    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.D = nn.Parameter(torch.eye(d, dtype=D_init.dtype, device=D_init.device))

    holder = _Holder()
    orthogonal(holder, 'D')
    with torch.no_grad():
        holder.D = D_init

    def objective() -> torch.Tensor:
        D = holder.D
        rotated = D.T @ W @ D  # (K, d, d)
        diag_rot = torch.diagonal(rotated, dim1=-2, dim2=-1)  # (K, d)
        return ((A_inv * diag_rot).sum(dim=-1) / lam).sum()

    with torch.no_grad():
        initial_obj = objective().item()

    optimizer = torch.optim.LBFGS(
        holder.parameters(), max_iter=n_steps, line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        final_obj = objective().item()

    if final_obj < initial_obj:
        return holder.D.detach().clone()
    return D_init
