import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from scipy.optimize import linear_sum_assignment

# Set global style parameters.
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 400,
    "figure.titlesize": 20,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.facecolor": "#f5f5f5",
    "axes.edgecolor": "#333333",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "font.family": "DejaVu Sans",
    "savefig.dpi": 400,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "legend.frameon": True,
    "legend.framealpha": 1,
    "legend.edgecolor": "black",
    "legend.facecolor": "lightgrey",
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


# Helper: Hungarian algorithm to match predicted and true labels.
def match_labels(y_true_tensor, y_pred_tensor):
    if isinstance(y_true_tensor, np.ndarray):
        y_true_tensor = torch.tensor(y_true_tensor)
    if isinstance(y_pred_tensor, np.ndarray):
        y_pred_tensor = torch.tensor(y_pred_tensor)
        
    y_true_cpu = y_true_tensor.cpu().long()
    y_pred_cpu = y_pred_tensor.cpu().long()
    max_true = y_true_cpu.max().item() + 1
    max_pred = y_pred_cpu.max().item() + 1

    # Build contingency matrix.
    cont = np.zeros((max_true, max_pred), dtype=int)
    for i in range(y_true_cpu.size(0)):
        cont[y_true_cpu[i], y_pred_cpu[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-cont)  # maximize total assignment
    mapping = {col_ind[j]: row_ind[j] for j in range(len(row_ind))}
    matched_labels = np.array([mapping.get(p, p) for p in y_pred_cpu.numpy()], dtype=int)
    return matched_labels


# Updated plot_gmm function with a new mode 'outliers'
def plot_gmm(
    X=None,
    gmm=None,
    labels=None,        # Predicted labels (tensor or numpy array)
    true_labels=None,   # True labels (tensor or numpy array)
    ax=None,
    title='GMM Results',
    init_means=None,
    legend_labels=None,
    xlabel='Feature 1',
    ylabel='Feature 2',
    mode='cluster',     # 'cluster', 'continuous', 'ellipses', 'dots', 'weights', 'means', 'covariances', or 'outliers'
    color_values=None,
    cmap_cont='viridis',
    cmap_seq='Greens',
    cbar_label='Color',
    std_devs=[1, 2, 3],
    base_alpha=0.8,
    alpha_from_weight=False,
    dashed_outer=False
):
    if ax is None:
        ax = plt.gca()
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if mode == 'outliers':
        if true_labels is None:
            raise ValueError("For mode 'outliers', true_labels must be provided.")
        # Assume X is 2D; no PCA needed.
        # Ensure X is a NumPy array.
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X
        
        # Ensure labels is a tensor.
        if not isinstance(labels, torch.Tensor):
            labels_tensor = torch.tensor(labels)
        else:
            labels_tensor = labels

        # Similarly, ensure true_labels is a tensor.
        if not isinstance(true_labels, torch.Tensor):
            true_labels_tensor = torch.tensor(true_labels)
        else:
            true_labels_tensor = true_labels
        
        # Match predicted labels with true labels.
        matched_pred = match_labels(true_labels_tensor, labels_tensor)
        correct = (matched_pred == true_labels_tensor.cpu().numpy())
        incorrect = ~correct
        
        ax.scatter(X_np[correct, 0], X_np[correct, 1], c='green', s=25, 
                   label='Correctly predicted', marker='.', alpha=0.3)
        ax.scatter(X_np[incorrect, 0], X_np[incorrect, 1], c='red', s=25, 
                   label='Incorrectly predicted', marker='.', alpha=1)
        
        # Plot the 2D means and ellipses for each component.
        n_comps = gmm.means_.shape[0]
        for i in range(n_comps):
            mean_2d = gmm.means_[i].detach().cpu().numpy()
            cov_2d = gmm.covariances_[i].detach().cpu().numpy()  # Expected shape: (2, 2)
            ax.scatter(mean_2d[0], mean_2d[1], c='black', marker='x')
            
            # Compute eigenvalues and eigenvectors for the 2D covariance.
            eigvals, eigvecs = np.linalg.eigh(cov_2d)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            
            # 95% confidence ellipse: scaling factor ~5.991 for 2D.
            width, height = 2 * np.sqrt(5.991 * eigvals)
            ell = Ellipse(mean_2d, width, height, angle=angle,
                          edgecolor='black', facecolor='none', linestyle='--')
            ax.add_patch(ell)
        
        ax.set_title(title)
        ax.legend(markerscale=2)
        return ax
    
    # --- Plot data points ---
    if X is not None:
        if mode in ['dots', 'weights', 'means', 'covariances']:
            ax.scatter(X[:, 0], X[:, 1], c='k', s=10, marker='.')
        elif mode in ['cluster', 'continuous']:
            if mode == 'cluster':
                if labels is None:
                    if gmm is not None:
                        X_tensor = torch.from_numpy(X).float() if not isinstance(X, torch.Tensor) else X
                        labels = gmm.predict(X_tensor).detach().cpu().numpy()
                    else:
                        labels = np.zeros(X.shape[0], dtype=int)
                else:
                    if not isinstance(labels, np.ndarray):
                        labels = labels.detach().cpu().numpy() if hasattr(labels, 'detach') else np.array(labels)
                
                if gmm is not None:
                    n_components = gmm.n_components
                else:
                    n_components = int(np.max(labels)) + 1
                
                cmap = ListedColormap(plt.get_cmap('Dark2')(np.linspace(0, 1, n_components)))
                if legend_labels is None:
                    legend_labels = [f'Cluster {i}' for i in range(n_components)]
    
                for i, color, ll in zip(range(n_components), cmap.colors, legend_labels):
                    mask_pts = (labels == i)
                    ax.scatter(X[mask_pts, 0], X[mask_pts, 1], c=[color], s=10, label=ll, alpha=0.5, marker='o')
            elif mode == 'continuous':
                if color_values is None:
                    raise ValueError("In continuous mode, the parameter 'color_values' must be provided.")
                if not isinstance(color_values, np.ndarray):
                    color_values = color_values.detach().cpu().numpy() if hasattr(color_values, 'detach') else np.array(color_values)
                scatter = ax.scatter(X[:, 0], X[:, 1], c=color_values, cmap=cmap_cont, s=2)
                cbar = plt.gcf().colorbar(scatter, ax=ax)
                cbar.set_label(cbar_label)
    
    # --- Plot ellipses and means ---
    if gmm is not None:
        if mode in ['cluster', 'ellipses']:
            if alpha_from_weight:
                weights_array = np.array([float(w.detach().cpu().item()) for w in gmm.weights_])
                max_weight = weights_array.max()
                std_to_plot = 2
            if 'cmap' not in locals():
                cmap = ListedColormap(plt.get_cmap('Dark2')(np.linspace(0, 1, gmm.n_components)))
            for n, color in zip(range(gmm.n_components), cmap.colors):
                mean = gmm.means_[n].detach().cpu().numpy()
                if gmm.covariance_type in ['full', 'diag', 'spherical']:
                    if gmm.covariance_type == 'full':
                        cov = gmm.covariances_[n].detach().cpu().numpy()
                    elif gmm.covariance_type == 'diag':
                        diag_vals = gmm.covariances_[n].detach().cpu().numpy()
                        cov = np.diag(diag_vals)
                    elif gmm.covariance_type == 'spherical':
                        var = gmm.covariances_[n].detach().cpu().item()
                        cov = np.eye(gmm.n_features) * var
                elif gmm.covariance_type in ['tied_full', 'tied_diag', 'tied_spherical']:
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
                
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                
                if alpha_from_weight:
                    w = float(gmm.weights_[n].detach().cpu().item())
                    alpha_val = (w / max_weight) * base_alpha
                    width, height = 2 * std_to_plot * np.sqrt(vals)
                    kwargs = {}
                    if dashed_outer:
                        kwargs['linestyle'] = '-'
                    ell = Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor=color,
                        alpha=alpha_val,
                        edgecolor=color,
                        label=f"w={w:.2f}",
                        **kwargs
                    )
                    ax.add_patch(ell)
                else:
                    if not isinstance(std_devs, (list, tuple)):
                        std_devs = [std_devs]
                    if len(std_devs) == 1:
                        alphas = [base_alpha]
                    elif len(std_devs) == 2:
                        alphas = [base_alpha, base_alpha * 0.66]
                    elif len(std_devs) == 3:
                        alphas = [base_alpha, base_alpha * 0.66, base_alpha * 0.33]
                    else:
                        alphas = [base_alpha * (1 - i / len(std_devs)) for i in range(len(std_devs))]
                    for std_dev, alpha_val in zip(std_devs, alphas):
                        width, height = 2 * np.sqrt(vals) * std_dev
                        ell = Ellipse(
                            xy=mean,
                            width=width,
                            height=height,
                            angle=angle,
                            facecolor=color,
                            alpha=alpha_val,
                            edgecolor=None
                        )
                        ax.add_patch(ell)
                ax.scatter(mean[0], mean[1], c='k', s=10, marker='.')
        
        elif mode == 'weights':
            weights_array = np.array([float(w.detach().cpu().item()) for w in gmm.weights_])
            max_weight = weights_array.max()
            std_to_plot = 2
            n_components = gmm.n_components
            fill_color = "orange"
            line_colors = plt.cm.OrRd(np.linspace(0.4, 0.9, n_components))
            proxy_handles = []
            proxy_labels = []
            for n in range(n_components):
                mean = gmm.means_[n].detach().cpu().numpy()
                if gmm.covariance_type in ['full', 'diag', 'spherical']:
                    if gmm.covariance_type == 'full':
                        cov = gmm.covariances_[n].detach().cpu().numpy()
                    elif gmm.covariance_type == 'diag':
                        diag_vals = gmm.covariances_[n].detach().cpu().numpy()
                        cov = np.diag(diag_vals)
                    elif gmm.covariance_type == 'spherical':
                        var = gmm.covariances_[n].detach().cpu().item()
                        cov = np.eye(gmm.n_features) * var
                elif gmm.covariance_type in ['tied_full', 'tied_diag', 'tied_spherical']:
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
                
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * std_to_plot * np.sqrt(vals)
                w = float(gmm.weights_[n].detach().cpu().item())
                alpha_val = (w / max_weight) * base_alpha

                ell_filled = Ellipse(
                    xy=mean,
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor=fill_color,
                    alpha=alpha_val,
                    edgecolor='none'
                )
                ax.add_patch(ell_filled)
                outline_color = line_colors[n]
                ell_outline = Ellipse(
                    xy=mean,
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor='none',
                    edgecolor=outline_color,
                    linewidth=2.0,
                )
                ax.add_patch(ell_outline)
                ax.scatter(mean[0], mean[1], c=outline_color, s=10, marker='.')
                proxy = Ellipse((0, 0), 1, 1, alpha=1, facecolor='none', edgecolor=outline_color, linewidth=2.0)
                proxy_handles.append(proxy)
                proxy_labels.append(f"w={w:.2f}")
            # Only call legend if there are handles.
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(proxy_handles, proxy_labels, loc='best', markerscale=1.5)
        
        elif mode == 'means':
            std_to_plot = 2
            for n in range(gmm.n_components):
                mean = gmm.means_[n].detach().cpu().numpy()
                if gmm.covariance_type in ['full', 'diag', 'spherical']:
                    if gmm.covariance_type == 'full':
                        cov = gmm.covariances_[n].detach().cpu().numpy()
                    elif gmm.covariance_type == 'diag':
                        diag_vals = gmm.covariances_[n].detach().cpu().numpy()
                        cov = np.diag(diag_vals)
                    elif gmm.covariance_type == 'spherical':
                        var = gmm.covariances_[n].detach().cpu().item()
                        cov = np.eye(gmm.n_features) * var
                elif gmm.covariance_type in ['tied_full', 'tied_diag', 'tied_spherical']:
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
                
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * std_to_plot * np.sqrt(vals)
                ell = Ellipse(
                    xy=mean,
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor='blue',
                    alpha=base_alpha,
                    edgecolor='blue'
                )
                ax.add_patch(ell)
                if n == 0:
                    ax.scatter(mean[0], mean[1], c='yellow', s=50, marker='.', label='Final Mean')
                else:
                    ax.scatter(mean[0], mean[1], c='yellow', s=50, marker='.')
        
        elif mode == 'covariances':
            # Use a single base color from the sequential colormap for all components.
            base_color = plt.get_cmap(cmap_seq)(0.7)
            if not isinstance(std_devs, (list, tuple)):
                std_devs = [std_devs]
            if len(std_devs) == 1:
                alphas = [base_alpha]
            elif len(std_devs) == 2:
                alphas = [base_alpha, base_alpha * 2/3]
            elif len(std_devs) == 3:
                alphas = [base_alpha, base_alpha * 2/3, base_alpha * 1/3]
            else:
                alphas = [base_alpha * (1 - i / len(std_devs)) for i in range(len(std_devs))]
                
            for n in range(gmm.n_components):
                mean = gmm.means_[n].detach().cpu().numpy()
                if gmm.covariance_type in ['full', 'diag', 'spherical']:
                    if gmm.covariance_type == 'full':
                        cov = gmm.covariances_[n].detach().cpu().numpy()
                    elif gmm.covariance_type == 'diag':
                        diag_vals = gmm.covariances_[n].detach().cpu().numpy()
                        cov = np.diag(diag_vals)
                    elif gmm.covariance_type == 'spherical':
                        var = gmm.covariances_[n].detach().cpu().item()
                        cov = np.eye(gmm.n_features) * var
                elif gmm.covariance_type in ['tied_full', 'tied_diag', 'tied_spherical']:
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
                
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                
                # Draw ellipses from outer (largest std) first to inner.
                for std_dev, alpha_val in zip(std_devs, alphas):
                    width, height = 2 * std_dev * np.sqrt(vals)
                    ell = Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor=base_color,
                        alpha=alpha_val,
                        edgecolor=base_color
                    )
                    ax.add_patch(ell)
                ax.scatter(mean[0], mean[1], c='k', s=10, marker='.')
    
    # --- Plot initial means if provided ---
    if init_means is not None:
        init_means_cpu = init_means.detach().cpu().numpy() if hasattr(init_means, 'detach') else np.array(init_means)
        marker = 'x' if mode == 'means' else '+'
        for i in range(init_means_cpu.shape[0]):
            if i == 0:
                ax.scatter(init_means_cpu[i, 0], init_means_cpu[i, 1],
                           c='r', marker=marker, s=50, label='Initial Means')
            else:
                ax.scatter(init_means_cpu[i, 0], init_means_cpu[i, 1],
                           c='r', marker=marker, s=50)
    
    if mode in ['cluster']:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best', markerscale=1.5)
    elif mode not in ['cluster', 'continuous', 'ellipses', 'dots', 'weights', 'means', 'covariances']:
        raise ValueError("Mode must be one of 'cluster', 'continuous', 'ellipses', 'dots', 'weights', 'means', or 'covariances'.")
    
    return ax
