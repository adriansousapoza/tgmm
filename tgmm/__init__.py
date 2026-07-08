__version__ = "0.2.1"

from .gmm import GaussianMixture
from .gmm_init import GMMInitializer
from .metrics import ClusteringMetrics
from .plotting import plot_gmm, dynamic_figsize, match_predicted_to_true_labels
from .synthetic_data import generate_gmm_data