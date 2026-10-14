__version__ = "0.4.0"

from .gmm import GaussianMixture
from .gmm_init import GMMInitializer
from .hdbscan import HDBSCAN
from .metrics import ClusteringMetrics
from .mixtures import BaseMixture, StudentTMixture, VonMisesFisherMixture
from .model_selection import ModelSelectionResult, select_model
from .semi_supervised import PositiveUnlabeledMixture
from .stats import SufficientStats
from .plotting import (
    plot_gmm, dynamic_figsize, match_predicted_to_true_labels,
    confidence_ellipse_std_devs, gmm_proxy,
    CATEGORICAL_PALETTE, SEQUENTIAL_CMAP, STATUS_CRITICAL,
)
from .synthetic_data import generate_gmm_data