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


class TestPlotVmf:
    """plot_vmf / plot_vmf_cosine / vmf_cap_angle for VonMisesFisherMixture."""

    @staticmethod
    def _fitted(d, n=120):
        from tgmm import VonMisesFisherMixture
        torch.manual_seed(0)
        X = torch.randn(n, d)
        X[: n // 2, 0] += 4.0
        X = X / X.norm(dim=1, keepdim=True)
        return X, VonMisesFisherMixture(n_components=2, random_state=0, max_iter=30).fit(X)

    @pytest.mark.parametrize("d", [2, 3])
    def test_plot_vmf_circle_and_sphere(self, d):
        from tgmm import plot_vmf
        X, vmf = self._fitted(d)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d') if d == 3 else fig.add_subplot()
        try:
            assert plot_vmf(X, vmf, ax=ax) is ax
            plot_vmf(X, None, ax=ax, true_labels=(X[:, 0] > 0).long())  # no model: points only
        finally:
            plt.close(fig)

    def test_plot_vmf_rejects_other_dimensions_and_non_3d_axes(self):
        from tgmm import plot_vmf
        X, vmf = self._fitted(5)
        with pytest.raises(ValueError, match="plot_vmf_cosine"):
            plot_vmf(X, vmf)
        X3, vmf3 = self._fitted(3)
        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError, match="3-D axes"):
                plot_vmf(X3, vmf3, ax=ax)
        finally:
            plt.close(fig)

    def test_plot_vmf_cosine_any_dimension(self):
        from tgmm import plot_vmf_cosine
        X, vmf = self._fitted(20)
        fig, ax = plt.subplots()
        try:
            assert plot_vmf_cosine(X, vmf, ax=ax) is ax
        finally:
            plt.close(fig)

    def test_angle_density_normalised_and_cap_mass(self):
        import numpy as np
        from tgmm import vmf_angle_density, vmf_cap_angle
        for kappa, d in [(0.0, 3), (5.0, 3), (50.0, 2), (300.0, 768)]:
            grid = np.linspace(0, np.pi, 200001)
            f = vmf_angle_density(kappa, d, grid)
            trapezoid = getattr(np, 'trapezoid', None) or np.trapz
            assert abs(trapezoid(f, grid) - 1.0) < 1e-3
            theta = vmf_cap_angle(kappa, d, 0.9)
            mask = grid <= theta
            mass = np.sum(0.5 * (f[mask][1:] + f[mask][:-1]) * np.diff(grid[mask]))
            assert abs(mass - 0.9) < 5e-3
        # kappa=0, d=3 is uniform on the sphere: P(theta <= t) = (1 - cos t)/2
        assert abs(vmf_cap_angle(0.0, 3, 0.5) - np.pi / 2) < 1e-3
        with pytest.raises(ValueError):
            vmf_cap_angle(1.0, 3, 1.0)
