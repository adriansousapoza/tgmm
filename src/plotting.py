import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse

# Set global style parameters.
plt.rcParams.update({
    "figure.figsize": (8, 6),       # Default figure size
    "figure.dpi": 400,              # Default figure dpi
    "figure.titlesize": 20,         # Figure title size
    "axes.grid": True,              # Add grid by default
    "grid.alpha": 0.3,              # Grid transparency
    "axes.facecolor": "#f5f5f5",     # Light background for axes
    "axes.edgecolor": "#333333",     # Change axes edge color
    "axes.labelsize": 14,           # Font size for axis labels
    "axes.titlesize": 16,           # Font size for titles
    "xtick.labelsize": 12,          # Font size for x-ticks
    "ytick.labelsize": 12,          # Font size for y-ticks
    "legend.fontsize": 12,          # Font size for legends
    "lines.linewidth": 2,           # Default line width
    "lines.markersize": 8,          # Default marker size
    "font.family": "DejaVu Sans",   # Font type
    "savefig.dpi": 400,             # DPI for saved figures
    "savefig.format": "pdf",        # Default save format
    "savefig.bbox": "tight",        # Adjust layout when saving
})

def dynamic_figsize(rows, cols, base_width=8, base_height=6):
    """
    Adjust figure size dynamically based on subplot rows and cols.
    
    Args:
        rows (int): Number of rows of subplots.
        cols (int): Number of columns of subplots.
        base_width (int): Width per subplot.
        base_height (int): Height per subplot.
    
    Returns:
        tuple: Adjusted figure size.
    """
    return (cols * base_width, rows * base_height)

def plot_gmm(
    X,
    gmm=None,
    labels=None,
    ax=None,
    title='GMM Results',
    init_means=None,
    legend_labels=None,
    xlabel='Feature 1',
    ylabel='Feature 2',
    mode='cluster',         # 'cluster' or 'continuous'
    color_values=None,        # required in continuous mode
    cmap_cont='viridis',      # colormap for continuous mode
    cbar_label='Color'        # label for colorbar in continuous mode
):
    """
    Plot function that supports two modes:
    
    1. 'cluster': Plot data points colored by cluster, with optional GMM ellipses and means.
       If gmm is provided, ellipses are drawn; if labels are not provided, they are computed.
       
    2. 'continuous': Plot data points with colors representing continuous values (e.g., log-likelihood or probabilities),
       and add a colorbar.
    
    Parameters
    ----------
    X : np.ndarray
        Original 2D data (shape: (N, 2)).
    gmm : GaussianMixture or None
        A fitted GaussianMixture instance. If provided in 'cluster' mode, ellipses and means are plotted.
    labels : np.ndarray or torch.Tensor or None
        Predicted cluster labels for each point in X. In 'cluster' mode, if None and gmm is provided, labels are computed.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, uses current Axes.
    title : str
        Title for the plot.
    init_means : torch.Tensor or None
        Initial means (k, 2) to display them in red '+' markers (only in 'cluster' mode).
    legend_labels : list of str or None
        List of legend labels for each cluster (only in 'cluster' mode).
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    mode : str
        Plotting mode: 'cluster' for discrete cluster plots, 'continuous' for continuous color mapping.
    color_values : np.ndarray or torch.Tensor or None
        Continuous values for each data point used for coloring (required in 'continuous' mode).
    cmap_cont : str or Colormap
        Colormap to use for continuous plotting (default 'viridis').
    cbar_label : str
        Label for the colorbar (only in 'continuous' mode).
    """
    if ax is None:
        ax = plt.gca()
    
    # Set common axis labels and title.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if mode == 'cluster':
        # If labels are not provided, try to infer them if a GMM is provided.
        if labels is None:
            if gmm is not None:
                # Convert X to a torch tensor if necessary.
                X_tensor = torch.from_numpy(X).float() if not isinstance(X, torch.Tensor) else X
                labels = gmm.predict(X_tensor).detach().cpu().numpy()
            else:
                labels = np.zeros(X.shape[0], dtype=int)
        else:
            # Ensure labels are a numpy array.
            if not isinstance(labels, np.ndarray):
                labels = labels.detach().cpu().numpy() if hasattr(labels, 'detach') else np.array(labels)
        
        # Determine the number of clusters for color mapping.
        if gmm is not None:
            n_components = gmm.n_components
        else:
            n_components = int(np.max(labels)) + 1
        
        cmap = ListedColormap(plt.cm.Dark2(np.linspace(0, 1, n_components)))
    
        if legend_labels is None:
            legend_labels = [f'Cluster {i}' for i in range(n_components)]
    
        # Plot data points by predicted labels.
        for i, color, ll in zip(range(n_components), cmap.colors, legend_labels):
            mask = (labels == i)
            ax.scatter(X[mask, 0], X[mask, 1], c=[color], s=10, label=ll, alpha=0.5, marker='o')
    
        # If a GMM is provided, plot the final means and covariance ellipses.
        if gmm is not None:
            for n, color in zip(range(gmm.n_components), cmap.colors):
                mean = gmm.means_[n].detach().cpu().numpy()
                # Determine covariance matrix based on covariance type.
                if gmm.covariance_type in ['full', 'diag', 'spherical']:
                    # Use per-component covariance.
                    if gmm.covariance_type == 'full':
                        cov = gmm.covariances_[n].detach().cpu().numpy()
                    elif gmm.covariance_type == 'diag':
                        diag_vals = gmm.covariances_[n].detach().cpu().numpy()
                        cov = np.diag(diag_vals)
                    elif gmm.covariance_type == 'spherical':
                        var = gmm.covariances_[n].detach().cpu().item()
                        cov = np.eye(gmm.n_features) * var
                elif gmm.covariance_type in ['tied_full', 'tied_diag', 'tied_spherical']:
                    # Use the common covariance (do not index by n).
                    if gmm.covariance_type == 'tied_full':
                        cov = gmm.covariances_.detach().cpu().numpy()
                    elif gmm.covariance_type == 'tied_diag':
                        diag_vals = gmm.covariances_.detach().cpu().numpy()
                        cov = np.diag(diag_vals)
                    elif gmm.covariance_type == 'tied_spherical':
                        var = gmm.covariances_.detach().cpu().item()
                        cov = np.eye(gmm.n_features) * var
                else:
                    raise ValueError(f"Unsupported covariance_type: {gmm.covariance_type}")
                
                # Eigen-decomposition to obtain ellipse parameters.
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                
                # Plot ellipses for 1, 2, and 3 standard deviations.
                for std_dev, alpha_ellipse in zip([1, 2, 3], [0.3, 0.2, 0.1]):
                    width, height = 2 * np.sqrt(vals) * std_dev
                    ellipse = Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor=color,
                        alpha=alpha_ellipse,
                        edgecolor=None
                    )
                    ax.add_patch(ellipse)
                
                # Mark final mean as a black circle.
                if n == 0:
                    ax.plot(mean[0], mean[1], 'k.', markersize=10, label = 'Final Means')
                else:
                    ax.plot(mean[0], mean[1], 'k.', markersize=10)
    
        # Plot initial means if provided.
        if init_means is not None:
            init_means_cpu = init_means.detach().cpu().numpy() if hasattr(init_means, 'detach') else np.array(init_means)
            for i in range(init_means_cpu.shape[0]):
                if i == 0:
                    ax.plot(init_means_cpu[i, 0], init_means_cpu[i, 1], 'r+', markersize=10, markeredgewidth=2, label = 'Initial Means')
                else:
                    ax.plot(init_means_cpu[i, 0], init_means_cpu[i, 1], 'r+', markersize=10, markeredgewidth=2)
    
        ax.legend(loc='best', markerscale=1.5)
    
    elif mode == 'continuous':
        # In continuous mode, a continuous array of color values is required.
        if color_values is None:
            raise ValueError("In continuous mode, the parameter 'color_values' must be provided.")
        # Ensure color_values are a numpy array.
        if not isinstance(color_values, np.ndarray):
            color_values = color_values.detach().cpu().numpy() if hasattr(color_values, 'detach') else np.array(color_values)
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=color_values, cmap=cmap_cont, s=2)
        cbar = plt.gcf().colorbar(scatter, ax=ax)
        cbar.set_label(cbar_label)
    
    else:
        raise ValueError("Mode must be either 'cluster' or 'continuous'.")
