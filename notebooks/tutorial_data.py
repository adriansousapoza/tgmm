"""The default dataset shared by the tutorials: four 2-D Gaussian blobs.

Tutorial 01 (Introduction) builds and explains this dataset; every later tutorial that needs "the four blobs" imports it from here, so the
settings live in one place and the later notebooks need no data section of their own::

    from tutorial_data import make_default_data
    X_tensor, y_tensor = make_default_data(SEED, device=device)

This is a plain module, not a notebook (it is not listed in the tutorial nav and is not exported to the docs).
"""
import numpy as np

from tgmm import generate_gmm_data

N_SAMPLES = [800, 200, 1000, 1000]
CENTERS = [np.array([0, 2]), np.array([2, -2]), np.array([0, 0]), np.array([2, 2])]
COVS = [
    np.eye(2),                              # spherical covariance
    0.25 * np.eye(2),                       # spherical covariance, small and with few points
    np.array([[4.0, 0.0], [0.0, 0.25]]),    # diagonal covariance, elongated along x
    np.array([[0.29, 1.1], [1.1, 4.25]]),   # full covariance: very elongated and tilted (correlation 0.99)
]
# These are the blobs the tutorials have always shown (an earlier version of generate_gmm_data sampled with Z @ cov, so the matrices
# requested then, 1.0*I, 0.5*I, [[2, 0], [0, 0.5]] and [[0.2, 0.5], [0.5, 2]], really produced cov.T @ cov: the matrices above).


def default_blobs():
    """(centers, covs, n_samples) of the default four blobs, as fresh copies."""
    return [c.copy() for c in CENTERS], [S.copy() for S in COVS], list(N_SAMPLES)


def make_default_data(seed, device='cpu'):
    """Draw the default dataset with ``generate_gmm_data``: ``(X_tensor, y_tensor)``, seeded so the same ``seed`` gives the same points."""
    centers, covs, n_samples = default_blobs()
    return generate_gmm_data(centers, covs, n_samples, device=device, random_state=seed)
