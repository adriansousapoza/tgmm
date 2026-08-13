# Plotting and Visualization

The `tgmm.plotting` module provides one flexible function, `plot_gmm`, for visualizing 2D GMM
clustering results, plus two small helpers: `dynamic_figsize` and
`match_predicted_to_true_labels`.

## Overview

`plot_gmm` supports:

- **2D scatter plots** colored by predicted cluster, true labels, or a continuous value (e.g. log-likelihood)
- **Confidence ellipses** at one or more standard-deviation levels per component
- **Component means**, and optionally the pre-EM **initial means**
- **Weight-scaled** marker size/alpha to visualize mixing weights
- Highlighting **incorrect predictions** against ground truth

`plot_gmm` only accepts 2D data (`X.shape[1] == 2`) -- for higher-dimensional data, project to 2D
yourself first (e.g. with PCA) and pass the projected coordinates. See the
[PCA Plotting tutorial](../notebooks/08_pca_plotting.ipynb) for a worked example.

## Basic 2D Plotting

```python
from tgmm import GaussianMixture, plot_gmm
import matplotlib.pyplot as plt

# Fit GMM
gmm = GaussianMixture(n_components=3, n_features=2)
gmm.fit(X_tensor)

# Plot results (X can be a numpy array or torch tensor)
plt.figure(figsize=(8, 6))
plot_gmm(X, gmm, title='GMM Clustering Results', show_ellipses=True, show_means=True)
plt.show()
```

## Coloring by Cluster and Comparing to Ground Truth

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Color by predicted cluster
plot_gmm(X, gmm, ax=axes[0], color_by_cluster=True, title='Predicted Clusters')

# Compare against ground truth, highlighting misclassifications
plot_gmm(
    X, gmm,
    ax=axes[1],
    true_labels=true_labels,
    color_by_cluster=True,
    match_labels_to_true=True,
    show_incorrect_predictions=True,
    title='Predictions vs Truth'
)
plt.tight_layout()
plt.show()
```

## Continuous Coloring (Log-Likelihood)

```python
log_probs = gmm.score_samples(X_tensor)

plot_gmm(
    X, gmm,
    log_probs=log_probs.cpu().numpy(),
    colormap='viridis',
    colorbar_label='Log Probability',
    show_ellipses=False,
    title='Colored by Log-Likelihood'
)
plt.show()
```

## Confidence Ellipses

Show one or more standard-deviation levels per component:

```python
plot_gmm(
    X, gmm,
    show_ellipses=True,
    ellipse_std_devs=[1, 2, 3],  # 1, 2, and 3 std-dev ellipses
    ellipse_alpha=0.3,
    ellipse_fill=True
)
```

## Initial vs Final Means

Show where EM started (`gmm.initial_means_`) alongside the fitted means:

```python
plot_gmm(
    X, gmm,
    show_initial_means=True,
    show_means=True,
    ellipse_std_devs=[1],
    title='Initial vs Final Means'
)
```

## Weight Visualization

Scale marker size and/or ellipse transparency by each component's mixing weight:

```python
plot_gmm(
    X, gmm,
    scale_alpha_by_weight=True,
    scale_size_by_weight=True,
    title='Scaled by Component Weight'
)
```

## Custom Styling

```python
plot_gmm(
    X, gmm,
    # Point styling
    point_size=8,
    point_alpha=0.7,

    # Ellipse styling
    ellipse_std_devs=[1, 2, 3],
    ellipse_alpha=0.3,
    ellipse_fill=True,
    ellipse_line_style='--',

    # Mean markers
    mean_marker='*',
    mean_size=100,
    mean_color='red',

    title='Custom Styled GMM'
)
```

## Side-by-Side Comparison

```python
from tgmm import dynamic_figsize

cov_types = ['full', 'diag', 'spherical']
fig, axes = plt.subplots(1, len(cov_types), figsize=dynamic_figsize(1, len(cov_types)))

for ax, cov_type in zip(axes, cov_types):
    gmm = GaussianMixture(n_components=3, n_features=2, covariance_type=cov_type, random_state=42)
    gmm.fit(X_tensor)
    plot_gmm(X, gmm, ax=ax, show_ellipses=True, title=f'{cov_type.capitalize()} Covariance')

plt.tight_layout()
plt.show()
```

## High-Dimensional Data

`plot_gmm` requires exactly 2 features. Project higher-dimensional data down first, for example
with PCA:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_hd.numpy())

gmm = GaussianMixture(n_components=5, n_features=2)
gmm.fit(torch.tensor(X_2d, dtype=torch.float32))

plot_gmm(X_2d, gmm, show_ellipses=True, title='PCA Projection with Clusters')
```

See the [PCA Plotting tutorial](../notebooks/08_pca_plotting.ipynb) for fitting directly in the
original high-dimensional space and only projecting for visualization.

## Matching Predicted Labels to Ground Truth

Cluster indices from `predict()` are arbitrary -- use `match_predicted_to_true_labels` to remap
them (via the Hungarian algorithm) before computing supervised metrics or coloring plots
consistently:

```python
from tgmm import match_predicted_to_true_labels

predicted_labels = gmm.predict(X_tensor)
aligned_labels = match_predicted_to_true_labels(true_labels, predicted_labels)
```

`plot_gmm`'s `match_labels_to_true=True` option does this internally when both `true_labels`
and a fitted `gmm` are supplied.

## Tips for Effective Visualization

1. **Project first**: Use PCA (or another method) for data with more than 2 features
2. **Adjust transparency**: Use `point_alpha` / `ellipse_alpha` to see overlapping clusters
3. **Use multiple std-dev levels**: `ellipse_std_devs=[1, 2, 3]` shows spread at a glance
4. **Compare models side-by-side**: Loop over `covariance_type` or initialization method with one `ax` per panel
5. **Check initial vs. final means**: `show_initial_means=True` helps debug poor convergence

## Complete API Reference

For full details on `plot_gmm` and its helpers, see the [API Reference](../api/plotting.md).
