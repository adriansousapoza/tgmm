"""
Improved plotting module for Gaussian Mixture Models.

This module provides clean, intuitive plotting functions for GMM visualization
with clear parameter control instead of confusing 'modes'.

Importing this module (or `tgmm`, which imports it) applies a seaborn-based
theme process-wide via `sns.set_theme()`, so every matplotlib/seaborn plot in
the process -- not just `plot_gmm()` calls -- picks up the same look and the
same categorical color order.
"""

import torch
import math
from types import SimpleNamespace
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
import numpy as np

##############################################################################
# Palette
##############################################################################
# Fixed-order categorical palette (colorblind-safe adjacent contrast) and a
# single-hue sequential ramp for continuous coloring (e.g. log-probabilities).
# Hues are assigned by fixed slot, never cycled based on data -- see
# `create_colormap` for the >8-cluster fallback (cycles with a repeat, since a
# GMM component count isn't bounded the way a curated series list is).

CATEGORICAL_PALETTE = [
    '#2a78d6',  # 1: blue
    '#eb6834',  # 2: orange
    '#1baf7a',  # 3: aqua
    '#eda100',  # 4: yellow
    '#e87ba4',  # 5: magenta
    '#008300',  # 6: green
    '#4a3aa7',  # 7: violet
    '#e34948',  # 8: red
]

_SEQUENTIAL_BLUE_STEPS = [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
    '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b',
]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list('tgmm_blue', _SEQUENTIAL_BLUE_STEPS)
try:
    plt.colormaps.register(SEQUENTIAL_CMAP)
except ValueError:
    pass  # already registered -- e.g. this module was reloaded in a live kernel

# Status color for "this is wrong", kept visually distinct from the
# categorical red (slot 8) so a misclassified point never reads as "cluster 8".
STATUS_CRITICAL = '#d03b3b'

##############################################################################
# Theme
##############################################################################

sns.set_theme(
    style='whitegrid',
    palette=CATEGORICAL_PALETTE,
    rc={
        "figure.figsize": (7, 4.5),
        "figure.dpi": 300,
        "figure.titlesize": 20,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelsize": 10,
        "axes.labelcolor": "#52514e",
        "axes.titlesize": 16,
        "axes.titlecolor": "#0b0b0b",
        "grid.color": "#e1e0d9",
        "grid.alpha": 0.7,
        "text.color": "#0b0b0b",
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 400,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "legend.frameon": True,
        "legend.framealpha": 1,
        "legend.edgecolor": "#c3c2b7",
        "legend.facecolor": "#fcfcfb",
    },
)


##############################################################################
# Main plotting function
##############################################################################

def plot_gmm(
    X,
    gmm=None,

    # Data point styling
    show_points=True,
    point_size=5,
    point_alpha=0.5,
    point_color='auto',  # 'auto', 'black', array-like, or colormap name
    
    # Cluster visualization
    color_by_cluster=False,
    true_labels=None,
    match_labels_to_true=False,
    cluster_colors=CATEGORICAL_PALETTE,  # Can be colormap name, single color, or list of colors
    show_incorrect_predictions=False,  # replaces 'outliers' mode
    incorrect_color=STATUS_CRITICAL,  # color for misclassified points when show_incorrect_predictions=True

    # Continuous coloring (replaces 'continuous' mode)
    log_probs=None,
    colormap=SEQUENTIAL_CMAP,
    colorbar_label='Log Probability',
    
    # Component ellipses
    show_ellipses=True,
    ellipse_std_devs=[1, 2, 3],  # List of standard deviations to show
    ellipse_alpha=0.5,
    ellipse_colors='auto',  # 'auto' uses same as clusters
    ellipse_fill=True,
    ellipse_line_style='dotted',
    ellipse_line_width=2,
    ellipse_line_color='#0b0b0b',
    ellipse_line_alpha=0.5,
    
    # Component centers/means
    show_means=True,
    mean_marker='h',
    mean_size=25,
    mean_color='black',
    
    # Initial means (if provided)
    show_initial_means=False,
    initial_mean_marker='H',
    initial_mean_size=25,
    initial_mean_color='red',
    
    # Weight visualization
    scale_alpha_by_weight=False,
    scale_size_by_weight=False,
    
    # Plot styling
    ax=None,
    title='GMM Visualization',
    xlabel='Feature 1',
    ylabel='Feature 2',
    legend=True,
    legend_labels=None,
):
    """
    Plot Gaussian Mixture Model results with fine-grained control.
    
    Supports 2D data visualization with comprehensive styling options.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        The input data points to plot. Must be 2D data.
    gmm : fitted GMM object, optional
        A fitted Gaussian Mixture Model with predict(), means\\_, covariances\\_, etc.
    
    Data Points
    -----------
    show_points : bool, default=True
        Whether to show the data points.
    point_size : float, default=10
        Size of data points.
    point_alpha : float, default=0.6
        Transparency of data points.
    point_color : str or array-like, default='auto'
        Color specification for points. Options:
        - 'auto': Use cluster colors if color_by_cluster=True or true_labels is provided, else black
        - 'black': All points black
        - colormap name: Use specified colormap to color points (e.g., 'viridis', 'plasma')
        - single color: Use single color for all points (e.g., 'red', '#FF0000')
        - array-like: Custom colors/values for each point
    
    Clustering
    ----------
    color_by_cluster : bool, default=True
        Whether to color points by their cluster assignment.
    true_labels : array-like, optional
        Ground truth cluster labels for the data points X. These should be the 
        correct/actual cluster assignments for each data point, not predictions.
        Used for visualization comparison and accuracy assessment.
    match_labels_to_true : bool, default=False
        Whether to remap predicted labels to match true labels using Hungarian algorithm.
        This helps align predicted cluster IDs with true cluster IDs for better visualization.
    cluster_colors : str, list, or single color, default=CATEGORICAL_PALETTE
        Color specification for clusters. Options:
        - The default: `CATEGORICAL_PALETTE`, this module's fixed-order,
          colorblind-safe 8-color palette (cycles if more than 8 clusters)
        - Matplotlib colormap name (e.g., 'turbo', 'viridis')
        - Single color for all clusters (e.g., 'red', '#FF0000', (1, 0, 0))
        - List of specific colors (e.g., ['green', 'blue', 'red', 'yellow'])
    show_incorrect_predictions : bool, default=False
        Highlight incorrectly classified points in `incorrect_color` and correctly
        classified points in their matched cluster's color (from `cluster_colors`).
        Requires both true_labels and a fitted GMM model to compare predictions.
    incorrect_color : str, default=STATUS_CRITICAL ('#d03b3b')
        Color for misclassified points when show_incorrect_predictions=True.
        Deliberately distinct from `CATEGORICAL_PALETTE`'s red (slot 8) so a
        misclassified point never reads as "cluster 8".

    Continuous Coloring
    -------------------
    log_probs : array-like, optional
        Log-probability values (or other continuous values) to use for continuous
        coloring of data points. Commonly used for log-likelihoods from GMM.
    colormap : str or Colormap, default=SEQUENTIAL_CMAP
        Colormap for continuous values -- this module's single-hue (blue),
        light-to-dark sequential ramp by default. Accepts any matplotlib
        colormap name or Colormap object.
    colorbar_label : str, default='Log Probability'
        Label for the colorbar.
    
    Component Ellipses
    ------------------
    show_ellipses : bool, default=True
        Whether to show confidence ellipses for components.
    ellipse_std_devs : list, default=[1, 2]
        Standard deviations for ellipse boundaries.
    ellipse_alpha : float, default=0.3
        Transparency of ellipses.
    ellipse_colors : str, list, single color, or 'auto', default='auto'
        Colors for ellipses. Options:
        - 'auto': Use same colors as clusters
        - Matplotlib colormap name (e.g., 'turbo', 'viridis')  
        - Single color for all ellipses (e.g., 'red', '#FF0000')
        - List of specific colors (e.g., ['green', 'blue', 'red'])
    ellipse_fill : bool, default=True
        Whether ellipses should be filled.
    ellipse_line_style : str, default='dotted'
        Line style for ellipse boundaries.
    ellipse_line_width : float, default=2
        Line width for ellipse boundaries.
    ellipse_line_color : str, default='#0b0b0b'
        Color for ellipse boundary lines.
    ellipse_line_alpha : float, default=0.5
        Transparency for ellipse boundary lines (0=transparent, 1=opaque).
    
    Component Centers
    -----------------
    show_means : bool, default=True
        Whether to show component means.
    mean_marker : str, default='h'
        Marker style for component means.
    mean_size : float, default=25
        Size of mean markers.
    mean_color : str, default='black'
        Color of mean markers.
    
    Initial Means
    -------------
    show_initial_means : bool, default=False
        Whether to show initial means. If True, will automatically use gmm.initial\\_means\\_ 
        if available from the fitted GMM model.
    initial_mean_marker : str, default='H'
        Marker style for initial means.
    initial_mean_size : float, default=25
        Size of initial mean markers.
    initial_mean_color : str, default='red'
        Color of initial mean markers.
    
    Weight Visualization
    --------------------
    scale_alpha_by_weight : bool, default=False
        Scale ellipse transparency by component weight.
    scale_size_by_weight : bool, default=False
        Scale point/marker size by component weight.
    
    Styling
    -------
    ax : matplotlib.Axes, optional
        Axes to plot on. If None, uses current axes.
    title : str, default='GMM Visualization'
        Plot title.
    xlabel : str, default='Feature 1'
        X-axis label.
    ylabel : str, default='Feature 2'
        Y-axis label.
    legend : bool, default=True
        Whether to show legend.
    legend_labels : list, optional
        Custom labels for legend entries.
    
    Returns
    -------
    ax : matplotlib.Axes
        The axes object with the plot.
    """
    if ax is None:
        ax = plt.gca()

    # Convert inputs to tensors
    X = ensure_tensor_on_cpu(X, dtype=torch.float32)
    n_samples, n_features = X.shape
    
    # Ensure data is 2D for plotting
    if n_features != 2:
        raise ValueError(f"Data has {n_features} features but plotting requires 2D data. "
                        "Please reduce dimensionality to 2D before plotting.")
    
    # Get GMM predictions if available
    predicted_labels = None
    n_components = 1
    gmm_means_2d = None
    gmm_weights = None
    
    if gmm is not None:
        # Use 2D data for predictions
        predicted_labels = gmm.predict(X).cpu()
        n_components = gmm.n_components
        gmm_weights = ensure_tensor_on_cpu(gmm.weights_, dtype=torch.float32)
        
        # Get GMM means (should already be 2D)
        gmm_means_2d = ensure_tensor_on_cpu(gmm.means_, dtype=torch.float32)
    
    # Determine final labels for coloring
    final_labels = None
    # Maps a raw component index (as used by gmm.predict()/get_covariance_matrix)
    # to the true-label index it was Hungarian-matched to, so ellipses can use
    # the same color as the points matched to them. None means "no matching
    # requested" -- component index and label index are used interchangeably.
    component_label_map = None

    # Auto-enable cluster coloring if true_labels are provided
    should_color_by_cluster = color_by_cluster or (true_labels is not None)

    if should_color_by_cluster and predicted_labels is not None:
        if true_labels is not None and match_labels_to_true:
            true_labels_tensor = ensure_tensor_on_cpu(true_labels, dtype=torch.int64)
            component_label_map = compute_label_mapping(true_labels_tensor, predicted_labels)
            final_labels = predicted_labels.clone()
            for old_label, new_label in component_label_map.items():
                final_labels[predicted_labels == old_label] = new_label
        else:
            final_labels = predicted_labels
    elif should_color_by_cluster and true_labels is not None:
        true_labels_tensor = ensure_tensor_on_cpu(true_labels, dtype=torch.int64)
        final_labels = true_labels_tensor
        # Get the actual number of unique clusters instead of assuming consecutive labels
        unique_labels = torch.unique(final_labels)
        n_components = len(unique_labels)
    
    # Set up colors
    cluster_color_list = None
    if final_labels is not None:
        cluster_color_list = create_colormap(cluster_colors, n_components)
    
    # Plot data points
    if show_points:
        if log_probs is not None:
            # Continuous coloring
            log_probs_tensor = ensure_tensor_on_cpu(log_probs, dtype=torch.float32)
            sc = ax.scatter(X[:, 0], X[:, 1], c=log_probs_tensor, cmap=colormap, 
                          s=point_size, alpha=point_alpha)
            if legend:
                cb = plt.colorbar(sc, ax=ax)
                cb.set_label(colorbar_label)
                
        elif show_incorrect_predictions and true_labels is not None and predicted_labels is not None:
            # Highlight correct vs incorrect predictions. `final_labels` is
            # already the (possibly Hungarian-matched) predicted label per
            # point -- computed above alongside cluster_color_list -- so
            # correct points are colored by that same cluster color instead
            # of a flat green, matching what the ellipses/legend show.
            true_labels_tensor = ensure_tensor_on_cpu(true_labels, dtype=torch.int64)
            correct_mask = (final_labels == true_labels_tensor)
            incorrect_mask = ~correct_mask

            unique_labels = torch.unique(true_labels_tensor).cpu()
            for i, label_val in enumerate(unique_labels):
                mask = correct_mask & (true_labels_tensor == label_val)
                if mask.any():
                    label_text = legend_labels[i] if legend_labels and i < len(legend_labels) else f"Cluster {label_val.item()+1}"
                    ax.scatter(X[mask, 0], X[mask, 1],
                              color=cluster_color_list[i], s=point_size, alpha=point_alpha,
                              label=label_text, marker='.')

            ax.scatter(X[incorrect_mask, 0], X[incorrect_mask, 1],
                      c=incorrect_color, s=point_size, alpha=point_alpha,
                      label='Incorrect', marker='.')
                      
        elif isinstance(point_color, str) and point_color == 'auto':
            if should_color_by_cluster and final_labels is not None:
                # Auto color by cluster
                unique_labels = torch.unique(final_labels).cpu()
                for i, label_val in enumerate(unique_labels):
                    mask = (final_labels == label_val)
                    if mask.any():
                        label_text = legend_labels[i] if legend_labels and i < len(legend_labels) else f"Cluster {label_val.item()+1}"
                        # Use 'color' keyword for single colors to avoid matplotlib warnings
                        ax.scatter(X[mask, 0], X[mask, 1], 
                                 color=cluster_color_list[i], s=point_size, 
                                 alpha=point_alpha, label=label_text)
            else:
                # Auto color = black
                ax.scatter(X[:, 0], X[:, 1], c='black', s=point_size, alpha=point_alpha)
                
        elif isinstance(point_color, str):
            # Check if it's a colormap name
            try:
                cmap = plt.get_cmap(point_color)
                
                # If no cluster labels available, use continuous coloring, otherwise discrete
                if final_labels is None:
                    point_indices = torch.arange(n_samples, dtype=torch.float32) / max(1, n_samples - 1)
                    ax.scatter(X[:, 0], X[:, 1], c=point_indices, cmap=cmap, 
                             s=point_size, alpha=point_alpha)
                else:
                    # Use discrete colors from the colormap for each cluster
                    unique_labels = torch.unique(final_labels).cpu()
                    n_unique = len(unique_labels)
                    
                    # Create colormap indices that are well-distributed
                    if n_unique == 1:
                        color_indices = [0.5]  # Use middle of colormap for single cluster
                    else:
                        color_indices = [i / (n_unique - 1) for i in range(n_unique)]
                    
                    colors = [cmap(idx) for idx in color_indices]
                    
                    for i, label_val in enumerate(unique_labels):
                        mask = (final_labels == label_val)
                        if mask.any():
                            label_text = legend_labels[i] if legend_labels and i < len(legend_labels) else f"Cluster {label_val.item()+1}"
                            ax.scatter(X[mask, 0], X[mask, 1], 
                                     color=colors[i], s=point_size, 
                                     alpha=point_alpha, label=label_text)
            except ValueError:
                # Not a colormap name, treat as a single color
                ax.scatter(X[:, 0], X[:, 1], c=point_color, s=point_size, alpha=point_alpha)
            except ValueError:
                # Not a colormap name, treat as a single color
                ax.scatter(X[:, 0], X[:, 1], c=point_color, s=point_size, alpha=point_alpha)
        else:
            # Handle lists/arrays and other color specifications
            if isinstance(point_color, (list, tuple)) and len(point_color) != n_samples:
                # If it's a list but doesn't match sample size, treat as single color or colormap
                if len(point_color) == 1:
                    # Single color in a list
                    ax.scatter(X[:, 0], X[:, 1], c=point_color[0], s=point_size, alpha=point_alpha)
                elif final_labels is not None:
                    # Use as cluster colors if we have clusters
                    unique_labels = torch.unique(final_labels).cpu()
                    for i, label_val in enumerate(unique_labels):
                        if i < len(point_color):  # Only use available colors
                            mask = (final_labels == label_val)
                            if mask.any():
                                label_text = legend_labels[i] if legend_labels and i < len(legend_labels) else f"Cluster {label_val.item()+1}"
                                ax.scatter(X[mask, 0], X[mask, 1], 
                                         color=point_color[i], s=point_size, 
                                         alpha=point_alpha, label=label_text)
                else:
                    # No cluster info, use first color from the list
                    ax.scatter(X[:, 0], X[:, 1], c=point_color[0], s=point_size, alpha=point_alpha)
            else:
                # Explicit color array that matches sample size, or other valid matplotlib color spec
                ax.scatter(X[:, 0], X[:, 1], c=point_color, s=point_size, alpha=point_alpha)
    
    # Plot component ellipses and determine ellipse colors
    ellipse_color_list = None
    if show_ellipses and gmm is not None:
        # Determine ellipse colors
        if ellipse_colors == 'auto' and cluster_color_list is not None:
            ellipse_color_list = cluster_color_list
        elif ellipse_colors == 'auto':
            # Fallback to default colormap when auto is requested but no cluster colors available
            ellipse_color_list = create_colormap(cluster_colors, n_components)
        else:
            ellipse_color_list = create_colormap(ellipse_colors, n_components)
        
        for i in range(n_components):
            mean_i = gmm_means_2d[i]
            cov_i = get_covariance_matrix(gmm, i)
            # When components were Hungarian-matched to true labels for point
            # coloring, use that same mapped index for the ellipse color so a
            # component's ellipse always matches the color of the points
            # matched to it, not just its raw (arbitrary) component index.
            color_idx = component_label_map.get(i, i) if component_label_map else i

            # Calculate ellipse parameters
            vals, vecs = torch.linalg.eigh(cov_i)
            idx = torch.argsort(vals, descending=True)
            vals, vecs = vals[idx], vecs[:, idx]

            angle = (180.0 / math.pi) * torch.atan2(vecs[1, 0], vecs[0, 0])

            # Create ellipses for each standard deviation
            for j, std_dev in enumerate(ellipse_std_devs):
                width = 2.0 * std_dev * torch.sqrt(vals[0])
                height = 2.0 * std_dev * torch.sqrt(vals[1])

                # Adjust alpha for multiple ellipses (fade inner ellipses)
                current_alpha = ellipse_alpha * (1 - j * 0.3 / len(ellipse_std_devs))
                if scale_alpha_by_weight:
                    current_alpha *= (gmm_weights[i] / gmm_weights.max()).item()

                # Create face color with proper alpha
                if ellipse_fill:
                    face_color_with_alpha = (*to_rgba(ellipse_color_list[color_idx])[:3], current_alpha)
                else:
                    face_color_with_alpha = 'none'
                
                # Create edge color with proper alpha
                edge_color_with_alpha = (*to_rgba(ellipse_line_color)[:3], ellipse_line_alpha)
                
                # Label by color_idx (the display/true-label slot this ellipse's
                # color was drawn from), not the raw component index i -- so the
                # legend text always matches the swatch color, and matches the
                # "Cluster N" point labels (which are also indexed in that same
                # display space) when both are shown together.
                if j == 0 and legend:
                    ellipse_label = (legend_labels[color_idx]
                                      if legend_labels and color_idx < len(legend_labels)
                                      else f'Component {color_idx + 1}')
                else:
                    ellipse_label = None

                ellipse = Ellipse(
                    (mean_i[0].item(), mean_i[1].item()),
                    width.item(), height.item(),
                    angle=angle.item(),
                    facecolor=face_color_with_alpha,
                    edgecolor=edge_color_with_alpha,
                    linewidth=ellipse_line_width,
                    linestyle=ellipse_line_style,
                    label=ellipse_label
                )
                ax.add_patch(ellipse)
        
        # Add legend entry for ellipse boundaries (only once)
        if legend:
            # Create sigma labels (1σ, 2σ, 3σ, etc.). Rounded to 3 significant
            # figures since chi2-derived multipliers (see
            # confidence_ellipse_std_devs) are irrational, not clean integers.
            sigma_labels = [f"{std:.3g}σ" for std in ellipse_std_devs]
            sigma_text = "[" + ", ".join(sigma_labels) + "]"
            
            # Create a dummy line for legend only
            ax.plot([], [], c=ellipse_line_color, linestyle=ellipse_line_style, 
                      linewidth=ellipse_line_width, alpha=ellipse_line_alpha,
                      label=f'{sigma_text}')
    
    # Plot component means
    if show_means and gmm is not None:
        for i in range(n_components):
            mean_i = gmm_means_2d[i]
            size = mean_size
            if scale_size_by_weight:
                size *= (gmm_weights[i] / gmm_weights.max()).item()
                
            ax.scatter(mean_i[0].item(), mean_i[1].item(), 
                      c=mean_color, marker=mean_marker, s=size,
                      label='Component Mean' if i == 0 and legend else None)
    
    # Plot initial means
    if show_initial_means:
        # Use gmm.initial_means_ if available
        if gmm is not None and hasattr(gmm, 'initial_means_') and gmm.initial_means_ is not None:
            means_to_plot = ensure_tensor_on_cpu(gmm.initial_means_, dtype=torch.float32)
            for i in range(means_to_plot.shape[0]):
                ax.scatter(means_to_plot[i, 0].item(), means_to_plot[i, 1].item(),
                          c=initial_mean_color, marker=initial_mean_marker, 
                          s=initial_mean_size,
                          label='Initial Mean' if i == 0 and legend else None)
    
    # Finalize plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best')
    
    return ax




##############################################################################
# Helper functions
##############################################################################

def dynamic_figsize(rows, cols, base_width=7, base_height=4.5):
    """
    Scale a `plt.subplots` figure size to the grid shape, so each subplot
    keeps roughly the same size regardless of how many rows/cols it has.

    Parameters
    ----------
    rows, cols : int
        Subplot grid shape, as passed to `plt.subplots(rows, cols)`.
    base_width, base_height : float, default=7, 4.5
        Target size of a single subplot, in inches.

    Returns
    -------
    tuple of float
        `(width, height)` for the whole figure, for `figsize=`.
    """
    return (cols * base_width, rows * base_height)


def confidence_ellipse_std_devs(confidence_levels, df=2):
    """
    Convert confidence levels (e.g. 0.95) to the standard-deviation
    multipliers `plot_gmm`'s `ellipse_std_devs` expects, via the
    Mahalanobis-distance/chi-squared relationship, so a "0.95" ellipse
    actually encloses 95% of probability mass in `df` dimensions.

    This is *not* the same as passing the confidence level directly as a
    literal number of standard deviations: e.g. `sqrt(chi2.ppf(0.95, df=2))`
    is ~2.4477, whereas literally 2 standard deviations in 2D covers only
    ~86.5% of the mass (the two coincide only in 1D).

    Parameters
    ----------
    confidence_levels : float or list of float
        Confidence level(s) in (0, 1), e.g. 0.95 for a 95% confidence region.
    df : int, default=2
        Dimensionality of the ellipse (2 for the 2D ellipses `plot_gmm` draws).

    Returns
    -------
    list of float
        Standard-deviation multipliers, one per input confidence level --
        pass directly as `ellipse_std_devs`.
    """
    if np.isscalar(confidence_levels):
        confidence_levels = [confidence_levels]
    return [float(np.sqrt(chi2.ppf(level, df))) for level in confidence_levels]


def gmm_proxy(means, covariances, weights=None, covariance_type='full', predicted_labels=None):
    """
    Build a minimal object satisfying `plot_gmm`'s `gmm=` interface from raw
    Gaussian parameters, for visualizing distributions that were never fit
    with `GaussianMixture` -- e.g. the true generating components behind
    synthetic data, or parameters already projected to 2D for plotting.

    Parameters
    ----------
    means : array-like, shape (n_components, 2)
        Component means.
    covariances : array-like
        Component covariances, in the shape `get_covariance_matrix` expects
        for `covariance_type` (e.g. (n_components, 2, 2) for 'full').
    weights : array-like, shape (n_components,), optional
        Component weights. Defaults to uniform. Only affects plotting when
        `scale_alpha_by_weight`/`scale_size_by_weight` are used.
    covariance_type : str, default='full'
        Passed through to `get_covariance_matrix` -- one of 'full', 'diag',
        'spherical', 'tied_full', 'tied_diag', 'tied_spherical'.
    predicted_labels : array-like, optional
        Per-point component/label assignment to return from `predict()`, for
        `color_by_cluster`/`show_incorrect_predictions`. Required whenever
        `plot_gmm` is called with `show_points=True` and cluster-coloring
        enabled, since `predict()` is always invoked once `gmm` is set.

    Returns
    -------
    SimpleNamespace
        Exposes `n_components`, `means_`, `covariances_`, `weights_`,
        `covariance_type`, and `predict()` -- everything `plot_gmm` needs.
    """
    means = ensure_tensor_on_cpu(means, dtype=torch.float32)
    n_components = means.shape[0]
    if weights is None:
        weights = torch.full((n_components,), 1.0 / n_components)
    if predicted_labels is not None:
        predicted_labels = ensure_tensor_on_cpu(predicted_labels, dtype=torch.int64)
    return SimpleNamespace(
        n_components=n_components,
        means_=means,
        covariances_=ensure_tensor_on_cpu(covariances, dtype=torch.float32),
        weights_=ensure_tensor_on_cpu(weights, dtype=torch.float32),
        covariance_type=covariance_type,
        predict=lambda X: predicted_labels,
    )


def ensure_tensor_on_cpu(tensor_or_array, dtype=None):
    """
    Convert a tensor, numpy array, or array-like to a CPU `torch.Tensor`.

    Every plotting function in this module routes its inputs through this,
    so callers can pass numpy arrays, Python lists, or GPU tensors
    interchangeably.

    Parameters
    ----------
    tensor_or_array : torch.Tensor or array-like
        Input to convert. A `torch.Tensor` is moved to CPU as-is (no copy of
        the underlying data beyond what `.cpu()` requires); anything else is
        passed through `torch.tensor(...)`.
    dtype : torch.dtype, optional
        If given, cast the result to this dtype.

    Returns
    -------
    torch.Tensor
        CPU tensor, cast to `dtype` if given.
    """
    if isinstance(tensor_or_array, torch.Tensor):
        out = tensor_or_array.cpu()
    else:
        out = torch.tensor(tensor_or_array, device='cpu')
    if dtype is not None:
        out = out.to(dtype)
    return out


def create_colormap(colors, n_colors=8):
    """
    Create a list of color tuples from various color specifications.
    
    Parameters
    ----------
    colors : str, list, or single color
        Color specification. Can be:
        - A matplotlib colormap name (e.g., 'turbo', 'viridis')
        - A single color (e.g., 'red', '#FF0000', (1, 0, 0))
        - A list of colors (e.g., ['red', 'blue', 'green']) -- cycles if
          `n_colors` exceeds the list length, rather than running out.
    n_colors : int, default=8
        Number of colors to generate.

    Returns
    -------
    list
        List of color tuples/values, length `n_colors`.
    """
    # If colors is a list, return it directly (assuming it contains valid colors)
    if isinstance(colors, (list, tuple)) and not isinstance(colors, str):
        # Check if it's an RGB/RGBA tuple (single color) vs a list of colors
        if len(colors) in [3, 4] and all(isinstance(x, (int, float, np.floating)) for x in colors):
            # It's a single RGB/RGBA color, replicate it
            return [colors] * n_colors
        else:
            # It's a list of colors -- cycle if more are requested than given
            colors = list(colors)
            return [colors[i % len(colors)] for i in range(n_colors)]
    
    # If colors is a string, check if it's a colormap name
    if isinstance(colors, str):
        # Try to get it as a colormap first
        try:
            cmap = plt.get_cmap(colors)
            # If successful, generate colors from the colormap
            if n_colors == 1:
                return [cmap(0.5)]
            step = 1.0 / (n_colors - 1)
            return [cmap(i * step) for i in range(n_colors)]
        except ValueError:
            # Not a colormap name, treat as a single color name
            return [colors] * n_colors
    
    # Fallback: return as-is
    return [colors] * n_colors


def compute_label_mapping(true_labels, pred_labels):
    """
    Hungarian-match predicted cluster indices to true label indices.

    Parameters
    ----------
    true_labels : torch.Tensor
        Ground-truth labels as a 1D tensor.
    pred_labels : torch.Tensor
        Predicted cluster labels as a 1D tensor.

    Returns
    -------
    dict
        {predicted_label: true_label} for the best-matching assignment.
    """
    true_labels = true_labels.view(-1)
    pred_labels = pred_labels.view(-1)

    unique_true = torch.unique(true_labels)
    unique_pred = torch.unique(pred_labels)

    # Build contingency matrix
    contingency = torch.zeros((len(unique_true), len(unique_pred)), dtype=torch.int64)
    for i, t in enumerate(unique_true):
        for j, p in enumerate(unique_pred):
            contingency[i, j] = torch.sum((true_labels == t) & (pred_labels == p))

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(-contingency.numpy())

    return {int(unique_pred[j].item()): int(unique_true[i].item())
            for i, j in zip(row_ind, col_ind)}


def match_predicted_to_true_labels(true_labels, pred_labels):
    """
    Remap predicted labels to match true labels using Hungarian algorithm.

    Parameters
    ----------
    true_labels : torch.Tensor
        Ground-truth labels as a 1D tensor.
    pred_labels : torch.Tensor
        Predicted cluster labels as a 1D tensor.

    Returns
    -------
    torch.Tensor
        Remapped predicted labels that best match the true labels.
    """
    pred_labels = pred_labels.view(-1)
    mapping = compute_label_mapping(true_labels, pred_labels)

    # Apply mapping
    remapped = pred_labels.clone()
    for idx in range(remapped.size(0)):
        old_label = int(remapped[idx])
        remapped[idx] = mapping.get(old_label, old_label)

    return remapped


def get_covariance_matrix(gmm, component_idx):
    """
    Extract the full 2x2 covariance matrix for one GMM component, expanding
    whatever `gmm.covariance_type` shorthand it's stored in ('diag',
    'spherical', 'tied_*', etc.) into the dense form `plot_gmm`'s ellipse
    math needs.

    Parameters
    ----------
    gmm : fitted GMM object
        Must expose `covariances_` and `covariance_type` (e.g. a
        `GaussianMixture` or a `gmm_proxy(...)` stand-in).
    component_idx : int
        Which component's covariance to extract.

    Returns
    -------
    torch.Tensor, shape (2, 2)
        Dense covariance matrix for the requested component.
    """
    if gmm is None:
        raise ValueError("GMM object is required")
        
    covariances = ensure_tensor_on_cpu(gmm.covariances_, dtype=torch.float32)
    cov_type = gmm.covariance_type
        
    # Get the covariance matrix
    if cov_type == 'full':
        cov_matrix = covariances[component_idx]
    elif cov_type == 'diag':
        cov_matrix = torch.diag(covariances[component_idx])
    elif cov_type == 'spherical':
        # plot_gmm only ever renders 2D ellipses, so the identity is always 2x2
        # regardless of n_components (covariances here has shape (n_components,))
        var_val = covariances[component_idx]
        cov_matrix = torch.eye(2) * var_val
    elif cov_type == 'tied_full':
        cov_matrix = covariances
    elif cov_type == 'tied_diag':
        cov_matrix = torch.diag(covariances)
    elif cov_type == 'tied_spherical':
        # covariances is a 0-dim scalar tensor here, so there is no shape to derive a size from
        var_val = covariances
        cov_matrix = torch.eye(2) * var_val
    else:
        raise ValueError(f"Unsupported covariance_type: {cov_type}")
    
    return cov_matrix