__version__ = "0.4.2"

from .gmm import GaussianMixture
from .gmm_init import GMMInitializer
from .hdbscan import HDBSCAN
from .metrics import ClusteringMetrics
from .mixtures import (
    BaseMixture, Component, ConditionalGaussianMixture, GenericMixture, StudentTMixture, VonMisesFisherMixture,
)
from .model_selection import ModelSelectionResult, select_model
from .semi_supervised import PositiveUnlabeledMixture
from .stats import SufficientStats
from .plotting import (
    plot_gmm, plot_vmf, plot_vmf_cosine, vmf_angle_density, vmf_cap_angle, dynamic_figsize, match_predicted_to_true_labels,
    confidence_ellipse_std_devs, gmm_proxy,
    CATEGORICAL_PALETTE, SEQUENTIAL_CMAP, STATUS_CRITICAL,
)
from .synthetic_data import generate_gmm_data