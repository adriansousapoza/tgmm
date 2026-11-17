"""Regression tests for plot_gmm / get_covariance_matrix."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch

from tgmm.gmm import GaussianMixture
from tgmm.plotting import plot_gmm, create_colormap, CATEGORICAL_PALETTE


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


class TestCreateColormap:
    """create_colormap must always return exactly n_colors entries, even when
    a fixed-length list (like the default CATEGORICAL_PALETTE) is shorter
    than n_colors -- previously it silently returned the list as-is, which
    would IndexError downstream in plot_gmm for >8-component GMMs."""

    def test_list_shorter_than_n_colors_cycles_instead_of_running_out(self):
        colors = create_colormap(['red', 'blue', 'green'], n_colors=7)
        assert len(colors) == 7
        assert colors == ['red', 'blue', 'green', 'red', 'blue', 'green', 'red']

    def test_list_longer_than_n_colors_truncates(self):
        colors = create_colormap(['red', 'blue', 'green', 'yellow'], n_colors=2)
        assert colors == ['red', 'blue']

    def test_default_categorical_palette_cycles_for_many_components(self):
        colors = create_colormap(CATEGORICAL_PALETTE, n_colors=10)
        assert len(colors) == 10
        assert colors[:8] == CATEGORICAL_PALETTE
        assert colors[8] == CATEGORICAL_PALETTE[0]
        assert colors[9] == CATEGORICAL_PALETTE[1]


def test_plot_gmm_renders_with_more_components_than_palette_colors():
    """9 components exceeds len(CATEGORICAL_PALETTE) == 8; plot_gmm must not
    crash when coloring by cluster with the default palette."""
    torch.manual_seed(0)
    X = torch.randn(200, 2)

    gmm = GaussianMixture(n_components=9, covariance_type="diag", max_iter=10)
    gmm.fit(X)

    fig, ax = plt.subplots()
    try:
        plot_gmm(X.numpy(), gmm=gmm, ax=ax, color_by_cluster=True, show_ellipses=True)
    finally:
        plt.close(fig)
