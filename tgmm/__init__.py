__version__ = "0.3.0"

from .gmm import GaussianMixture
from .gmm_init import GMMInitializer
from .hdbscan import HDBSCAN
from .metrics import ClusteringMetrics
from .plotting import (
    plot_gmm, dynamic_figsize, match_predicted_to_true_labels,
    confidence_ellipse_std_devs, gmm_proxy,
    CATEGORICAL_PALETTE, SEQUENTIAL_CMAP, STATUS_CRITICAL,
)
from .synthetic_data import generate_gmm_data