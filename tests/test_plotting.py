"""Regression tests for plot_gmm / get_covariance_matrix."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch

from tgmm.gmm import GaussianMixture
from tgmm.plotting import plot_gmm


@pytest.mark.parametrize("covariance_type", [
    "full", "diag", "spherical", "tied_full", "tied_diag", "tied_spherical",
])
def test_plot_gmm_renders_ellipses_for_all_covariance_types(covariance_type):
    """spherical/tied_spherical covariances have shape (n_components,) / scalar,
    not (n_features, n_features) -- get_covariance_matrix must still build a
    2x2 identity for ellipse rendering instead of deriving the size from the
    covariance tensor's own (unrelated) shape."""
    torch.manual_seed(0)
    X = torch.randn(80, 2)

    gmm = GaussianMixture(n_components=4, covariance_type=covariance_type, max_iter=20)
    gmm.fit(X)

    fig, ax = plt.subplots()
    try:
        plot_gmm(X.numpy(), gmm=gmm, ax=ax, show_ellipses=True)
    finally:
        plt.close(fig)
