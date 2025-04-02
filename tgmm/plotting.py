import torch
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse

# --- Matplotlib defaults ---
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


##############################################################################
# Helper functions
##############################################################################

def dynamic_figsize(rows, cols, base_width=8, base_height=6):
    return (cols * base_width, rows * base_height)

def ensure_tensor_on_cpu(tensor_or_array, dtype=None):
    """
    Convert the input to a CPU torch.Tensor of a specified dtype (if given).
    - If already a torch.Tensor, just .cpu() it.
    - If it's a NumPy array (or list/scalar), convert via torch.tensor(...).
    """
    if isinstance(tensor_or_array, torch.Tensor):
        out = tensor_or_array.cpu()
    else:
        out = torch.tensor(tensor_or_array, device='cpu')
    if dtype is not None:
        out = out.to(dtype)
    return out

def make_colormap(name, n_colors=8):
    """
    Create a list of RGBA color tuples from a matplotlib colormap.
    We'll do a simple linear stepping.
    """
    cmap = plt.get_cmap(name)
    if n_colors == 1:
        # Just pick the middle color if there's only 1 cluster
        return [cmap(0.5)]
    step = 1.0 / (n_colors - 1)
    return [cmap(i * step) for i in range(n_colors)]

def torch_unique_with_counts(x):
    """
    PyTorch equivalent of np.unique(x, return_counts=True).
    Returns (unique_vals, counts).
    """
    uniques, counts = x.unique(return_counts=True)
    return uniques, counts


##############################################################################
# Single Label-Matching Function (by size)
##############################################################################

from scipy.optimize import linear_sum_assignment

def match_true_labels(labels_ref, labels_pred):
    """
    Remap `labels_pred` onto `labels_ref` using linear assignment (Hungarian algorithm)
    on the contingency matrix between true and predicted labels.
    
    Parameters
    ----------
    labels_ref : torch.Tensor
        Ground-truth labels as a 1D tensor.
    labels_pred : torch.Tensor
        Predicted cluster labels as a 1D tensor.
    
    Returns
    -------
    torch.Tensor
        A new tensor where each predicted label is remapped to the corresponding true label.
    """
    # Flatten the tensors
    labels_ref = labels_ref.view(-1)
    labels_pred = labels_pred.view(-1)

    # Get unique labels from true and predicted labels
    unique_true = torch.unique(labels_ref)
    unique_pred = torch.unique(labels_pred)

    # Build the contingency matrix (rows: true labels, cols: predicted labels)
    contingency = torch.zeros((len(unique_true), len(unique_pred)), dtype=torch.int64)
    for i, t in enumerate(unique_true):
        for j, p in enumerate(unique_pred):
            contingency[i, j] = torch.sum((labels_ref == t) & (labels_pred == p))
    
    # Convert contingency matrix to numpy array for the Hungarian algorithm
    contingency_np = contingency.numpy()

    # Solve the linear assignment problem on the negative contingency (to maximize matching)
    row_ind, col_ind = linear_sum_assignment(-contingency_np)

    # Create mapping: predicted label (from unique_pred) -> true label (from unique_true)
    mapping = { int(unique_pred[j].item()): int(unique_true[i].item()) 
                for i, j in zip(row_ind, col_ind) }

    # Remap each entry in labels_pred using the mapping; if a label is not mapped, leave it unchanged
    remapped = labels_pred.clone()
    for idx in range(remapped.size(0)):
        old_label = int(remapped[idx])
        remapped[idx] = mapping.get(old_label, old_label)
        
    return remapped



##############################################################################
# Main Plot Function
##############################################################################

def plot_gmm(
    X,                # torch.Tensor or NumPy array (N x 2)
    gmm=None,
    labels=None,      # "True" labels, or user-provided
    match_labels=False,
    ax=None,
    mode='cluster',   # can be 'cluster', 'outliers', etc.
    title='GMM Results',
    weights=None,     # user-provided mixture weights (if gmm=None)
    means=None,       # user-provided means
    covariances=None, # user-provided covariances
    covariance_type='full',
    init_means=None,  
    cmap_cont='viridis',
    cmap_seq='Greens',
    std_devs=(1, 2, 3),
    base_alpha=0.8,
    alpha_from_weight=False,
    dashed_outer=False,
    xlabel='Feature 1',
    ylabel='Feature 2',
    legend_labels=None,
    color_values=None,
    cbar_label='Color',
):
    """
    Plots data in 2D using either:
      - 'cluster' or 'outliers' (colored by cluster correctness),
      - 'continuous' (color_values),
      - 'dots' (plain),
      - 'ellipses', 'weights', 'means', 'covariances' (show component ellipses).

    If mode='outliers', we must have both `gmm` and `labels` (the "true" labels).
    Then we do GMM.predict(X), optionally match them to the true labels if
    match_labels=True, and highlight correct vs incorrect.

    GMM is expected to have:
      - gmm.predict(X) => torch.LongTensor (cluster IDs)
      - gmm.weights_, gmm.means_, gmm.covariances_, gmm.covariance_type
      - gmm.n_components
    """
    if ax is None:
        ax = plt.gca()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 1) Convert data & labels to Torch on CPU
    X = ensure_tensor_on_cpu(X, dtype=torch.float32)
    if labels is not None:
        labels = ensure_tensor_on_cpu(labels, dtype=torch.int64)
    N = X.shape[0]

    # 2) Possibly get predicted labels from GMM
    pred_labels = None
    if gmm is not None:
        pred_labels = gmm.predict(X).cpu()
        n_components = gmm.n_components
    else:
        n_components = 1

    # 3) Handle "outliers" mode first
    if mode == 'outliers':
        if (gmm is None) or (labels is None):
            raise ValueError("mode='outliers' requires both gmm=... and labels=...")

        # If match_labels=True, align predicted to ground-truth
        if match_labels:
            final_labels = match_true_labels(labels, pred_labels)
        else:
            final_labels = pred_labels

        # Plot correct vs incorrect
        correct_mask = (final_labels == labels)
        incorrect_mask = ~correct_mask

        ax.scatter(
            X[correct_mask, 0], 
            X[correct_mask, 1],
            c='green', s=20, marker='.', alpha=0.3,
            label='Correct'
        )
        ax.scatter(
            X[incorrect_mask, 0],
            X[incorrect_mask, 1],
            c='red', s=20, marker='.',
            alpha=0.8, label='Incorrect'
        )

        # Also plot a ~95% ellipse for each GMM component
        # => chi-square for 2 DOF at 95% is ~5.991
        chi2_95 = 5.991
        means_ = ensure_tensor_on_cpu(gmm.means_, dtype=torch.float32)
        covs_  = ensure_tensor_on_cpu(gmm.covariances_, dtype=torch.float32)

        for i in range(n_components):
            mean_i = means_[i]
            cov_i  = covs_[i]

            vals, vecs = torch.linalg.eigh(cov_i)
            idx = torch.argsort(vals, descending=True)
            vals, vecs = vals[idx], vecs[:, idx]

            angle = (180.0 / math.pi) * torch.atan2(vecs[1, 0], vecs[0, 0])
            # width, height
            w_ = 2.0 * (chi2_95 * vals).sqrt()
            ellipse = Ellipse(
                (mean_i[0].item(), mean_i[1].item()),
                w_[0].item(), w_[1].item(),
                angle=angle.item(),
                facecolor='none',
                edgecolor='black',
                linewidth=1.5,
                linestyle='--'
            )
            ax.add_patch(ellipse)
            # Mark the component mean
            ax.scatter(
                mean_i[0].item(),
                mean_i[1].item(),
                c='black', marker='x', s=50
            )

        ax.set_title(title)
        ax.legend(markerscale=1.5)
        return ax

    # 4) If not in outliers mode, we do the usual logic

    # If user didn't provide a GMM, maybe they gave direct labels or nothing
    if gmm is None:
        # no gmm => use user-provided labels to figure out n_components
        if labels is not None:
            n_components = int(labels.max().item()) + 1
        else:
            n_components = 1

    # Determine final_labels in normal cluster modes
    if mode in ['cluster', 'ellipses','weights','means','covariances','dots','continuous']:
        if pred_labels is not None:  
            if labels is not None and match_labels:
                final_labels = match_true_labels(labels, pred_labels)
            else:
                # default => use pred_labels as-is
                final_labels = pred_labels
        else:
            final_labels = labels  # might be None if user just wants 'dots'

    # 5) Plot data for the simpler modes
    if mode == 'dots':
        ax.scatter(X[:, 0], X[:, 1], c='k', s=10, marker='.')

    elif mode == 'continuous':
        if color_values is None:
            raise ValueError("In 'continuous' mode, color_values must be provided.")
        color_values = ensure_tensor_on_cpu(color_values, dtype=torch.float32)
        sc = ax.scatter(X[:, 0], X[:, 1], c=color_values, cmap=cmap_cont, s=10)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(cbar_label)

    elif mode == 'cluster':
        if final_labels is None:
            # fallback => black points
            ax.scatter(X[:, 0], X[:, 1], c='k', s=10, marker='.')
        else:
            if legend_labels is None:
                legend_labels = [f"Cluster {i}" for i in range(n_components)]
            color_list = make_colormap('Dark2', n_components)

            for i in range(n_components):
                mask = (final_labels == i)
                ax.scatter(
                    X[mask, 0], X[mask, 1],
                    c=[color_list[i]],
                    s=10, alpha=0.5,
                    label=legend_labels[i]
                )
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='best', markerscale=1.5)

    else:
        # 'ellipses', 'weights', 'means', 'covariances' => show black points
        ax.scatter(X[:, 0], X[:, 1], c='k', s=10, marker='.')

    # 6) Pull out GMM or user-supplied params for ellipse plotting
    if gmm is not None:
        m_weights     = ensure_tensor_on_cpu(gmm.weights_,     dtype=torch.float32)
        m_means       = ensure_tensor_on_cpu(gmm.means_,       dtype=torch.float32)
        m_covariances = ensure_tensor_on_cpu(gmm.covariances_, dtype=torch.float32)
        cov_type      = gmm.covariance_type
        comp_count    = gmm.n_components
    elif (weights is not None) and (means is not None) and (covariances is not None):
        m_weights     = ensure_tensor_on_cpu(weights,     dtype=torch.float32)
        m_means       = ensure_tensor_on_cpu(means,       dtype=torch.float32)
        m_covariances = ensure_tensor_on_cpu(covariances, dtype=torch.float32)
        cov_type      = covariance_type
        comp_count    = m_means.shape[0]
    else:
        m_weights = m_means = m_covariances = None
        cov_type = None
        comp_count = 0

    def get_full_cov(i):
        """Return the full 2D covariance for the i-th component."""
        if cov_type == 'full':
            return m_covariances[i]
        elif cov_type == 'diag':
            return torch.diag(m_covariances[i])
        elif cov_type == 'spherical':
            var_val = m_covariances[i]
            d = m_means.shape[1]
            return torch.eye(d) * var_val
        elif cov_type == 'tied_full':
            return m_covariances
        elif cov_type == 'tied_diag':
            return torch.diag(m_covariances)
        elif cov_type == 'tied_spherical':
            var_val = m_covariances
            d = m_means.shape[1]
            return torch.eye(d) * var_val
        else:
            raise ValueError(f"Unsupported covariance_type: {cov_type}")

    # 7) If we have means/covs and user wants ellipse modes
    if (m_means is not None) and (mode in ['ellipses','cluster','weights','means','covariances']):
        
        # --------------------------------------------------------------------
        # 'weights' => single ellipse each, alpha scaled by weight
        # --------------------------------------------------------------------
        if mode == 'weights':
            wmax = m_weights.max()
            line_colors = make_colormap('OrRd', comp_count)

            for i in range(comp_count):
                mean_i = m_means[i]
                cov_i  = get_full_cov(i)
                vals, vecs = torch.linalg.eigh(cov_i)
                idx = torch.argsort(vals, descending=True)
                vals, vecs = vals[idx], vecs[:, idx]

                angle = (180.0 / math.pi) * torch.atan2(vecs[1, 0], vecs[0, 0])
                width, height = 2.0 * 2.0 * vals.sqrt()
                alpha_val = (m_weights[i] / wmax) * base_alpha

                # Filled ellipse
                efill = Ellipse(
                    (mean_i[0].item(), mean_i[1].item()),
                    width=width.item(), height=height.item(),
                    angle=angle.item(),
                    facecolor='orange',
                    alpha=alpha_val.item(),
                    edgecolor='none'
                )
                ax.add_patch(efill)

                # Outline
                edge_col = line_colors[i]
                eoutline = Ellipse(
                    (mean_i[0].item(), mean_i[1].item()),
                    width=width.item(), height=height.item(),
                    angle=angle.item(),
                    facecolor='none',
                    edgecolor=edge_col,
                    linewidth=2.0
                )
                ax.add_patch(eoutline)

                ax.scatter(mean_i[0].item(), mean_i[1].item(), c=[edge_col], s=20, marker='x')

        # --------------------------------------------------------------------
        # 'means' => highlight each mean with a ~2 stdev ellipse
        # --------------------------------------------------------------------
        elif mode == 'means':
            for i in range(comp_count):
                mean_i = m_means[i]
                cov_i  = get_full_cov(i)
                vals, vecs = torch.linalg.eigh(cov_i)
                idx = torch.argsort(vals, descending=True)
                vals, vecs = vals[idx], vecs[:, idx]

                angle = (180.0 / math.pi) * torch.atan2(vecs[1, 0], vecs[0, 0])
                w_ = 2.0 * 2.0 * vals.sqrt()
                width, height = w_[0], w_[1] if w_.size(0) > 1 else w_[0]

                e = Ellipse(
                    (mean_i[0].item(), mean_i[1].item()),
                    width=width.item(), height=height.item(),
                    angle=angle.item(),
                    facecolor='blue',
                    alpha=base_alpha,
                    edgecolor='blue'
                )
                ax.add_patch(e)

                ax.scatter(
                    mean_i[0].item(), mean_i[1].item(),
                    c='yellow', s=50,
                    marker='o' if i == 0 else '.',
                    label='Mean' if i == 0 else None
                )

        # --------------------------------------------------------------------
        # 'covariances' => multiple ellipses per component based on std_devs
        # --------------------------------------------------------------------
        elif mode == 'covariances':
            if isinstance(std_devs, (int, float)):
                std_devs = [std_devs]
            base_col = plt.get_cmap(cmap_seq)(0.7)
            # build alpha ladder
            if len(std_devs) == 1:
                alpha_list = [base_alpha]
            elif len(std_devs) == 2:
                alpha_list = [base_alpha, base_alpha * 0.66]
            elif len(std_devs) == 3:
                alpha_list = [base_alpha, base_alpha * 0.66, base_alpha * 0.33]
            else:
                alpha_list = [base_alpha*(1 - j/len(std_devs)) for j in range(len(std_devs))]

            for i in range(comp_count):
                mean_i = m_means[i]
                cov_i  = get_full_cov(i)
                vals, vecs = torch.linalg.eigh(cov_i)
                idx = torch.argsort(vals, descending=True)
                vals, vecs = vals[idx], vecs[:, idx]

                angle = (180.0 / math.pi) * torch.atan2(vecs[1, 0], vecs[0, 0])
                for s, a_val in zip(std_devs, alpha_list):
                    w_ = 2.0 * s * vals.sqrt()
                    e = Ellipse(
                        (mean_i[0].item(), mean_i[1].item()),
                        w_[0].item(), w_[1].item(),
                        angle=angle.item(),
                        facecolor=base_col,
                        alpha=a_val,
                        edgecolor=base_col
                    )
                    ax.add_patch(e)
                ax.scatter(mean_i[0].item(), mean_i[1].item(), c='k', s=10, marker='.')

        # --------------------------------------------------------------------
        # 'ellipses' => single or multiple ellipses in cluster style
        # --------------------------------------------------------------------
        elif mode in ['ellipses','cluster']:
            color_list = make_colormap('Dark2', comp_count)
            if alpha_from_weight and (m_weights is not None):
                wmax = m_weights.max()
                for i in range(comp_count):
                    mean_i = m_means[i]
                    cov_i  = get_full_cov(i)
                    vals, vecs = torch.linalg.eigh(cov_i)
                    idx = torch.argsort(vals, descending=True)
                    vals, vecs = vals[idx], vecs[:, idx]

                    angle = (180.0 / math.pi) * torch.atan2(vecs[1, 0], vecs[0, 0])
                    w_ = 2.0 * 2.0 * vals.sqrt()
                    alpha_val = (m_weights[i] / wmax) * base_alpha

                    e = Ellipse(
                        (mean_i[0].item(), mean_i[1].item()),
                        w_[0].item(), w_[1].item(),
                        angle=angle.item(),
                        facecolor=color_list[i],
                        alpha=alpha_val.item(),
                        edgecolor=color_list[i]
                    )
                    if dashed_outer:
                        e.set_linestyle('--')
                    ax.add_patch(e)
                    ax.scatter(mean_i[0].item(), mean_i[1].item(), c='k', s=20, marker='.')
            else:
                # multiple ellipses per comp for given std_devs
                if isinstance(std_devs, (int, float)):
                    std_devs = [std_devs]

                for i in range(comp_count):
                    mean_i = m_means[i]
                    cov_i  = get_full_cov(i)
                    vals, vecs = torch.linalg.eigh(cov_i)
                    idx = torch.argsort(vals, descending=True)
                    vals, vecs = vals[idx], vecs[:, idx]

                    angle = (180.0 / math.pi) * torch.atan2(vecs[1, 0], vecs[0, 0])

                    # generate alpha ladder
                    if len(std_devs) == 1:
                        alpha_list = [base_alpha]
                    elif len(std_devs) == 2:
                        alpha_list = [base_alpha, base_alpha * 0.66]
                    elif len(std_devs) == 3:
                        alpha_list = [base_alpha, base_alpha * 0.66, base_alpha * 0.33]
                    else:
                        alpha_list = [base_alpha*(1 - j/len(std_devs)) for j in range(len(std_devs))]

                    for s, a_val in zip(std_devs, alpha_list):
                        w_ = 2.0 * s * vals.sqrt()
                        e = Ellipse(
                            (mean_i[0].item(), mean_i[1].item()),
                            w_[0].item(), w_[1].item(),
                            angle=angle.item(),
                            facecolor=color_list[i],
                            alpha=a_val,
                            edgecolor=None
                        )
                        ax.add_patch(e)
                    ax.scatter(mean_i[0].item(), mean_i[1].item(), c='k', s=20, marker='.')

    # 8) Optionally plot initial means
    if init_means is not None:
        init_means = ensure_tensor_on_cpu(init_means, dtype=torch.float32)
        mark = 'x' if mode == 'means' else '+'
        for i in range(init_means.shape[0]):
            lbl = 'Initial Means' if i == 0 else None
            ax.scatter(
                init_means[i,0].item(),
                init_means[i,1].item(),
                c='r', marker=mark, s=50, label=lbl
            )

    ax.set_title(title)
    return ax
