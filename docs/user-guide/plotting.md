# Plotting and Visualization

The `plotting` module provides tools for visualizing GMM clustering results, including PCA-based projections for high-dimensional data.

## Overview

Visualization functions include:

- **2D/3D scatter plots** with cluster assignments
- **PCA projections** for high-dimensional data
- **Confidence ellipses** showing cluster distributions
- **Component overlays** visualizing individual Gaussians
- **Interactive plots** using Plotly

## Basic 2D Plotting

```python
from tgmm import GaussianMixture
from tgmm.plotting import plot_gmm_2d
import torch

# Fit GMM
gmm = GaussianMixture(n_components=3, n_features=2)
gmm.fit(X)

# Plot results
plot_gmm_2d(
    X, 
    gmm, 
    show_ellipses=True,
    show_centers=True,
    title='GMM Clustering Results'
)
```

## PCA Projection for High-Dimensional Data

When data has more than 2-3 dimensions, use PCA to project to 2D:

```python
from tgmm.plotting import plot_gmm_pca

# X has shape (n_samples, n_features) where n_features > 3
gmm = GaussianMixture(n_components=5, n_features=X.shape[1])
gmm.fit(X)

# Project to 2D using PCA and plot
plot_gmm_pca(
    X, 
    gmm,
    n_components=2,  # Project to 2D
    show_variance=True,  # Show explained variance
    show_ellipses=True
)
```

## 3D Visualization

```python
from tgmm.plotting import plot_gmm_3d

# For 3D data
gmm = GaussianMixture(n_components=4, n_features=3)
gmm.fit(X_3d)

plot_gmm_3d(
    X_3d,
    gmm,
    show_ellipses=True,
    interactive=True  # Use Plotly for rotation
)
```

## Confidence Ellipses

Visualize cluster distributions with confidence ellipses:

```python
plot_gmm_2d(
    X, 
    gmm,
    show_ellipses=True,
    confidence=0.95,  # 95% confidence ellipse
    alpha=0.3  # Transparency
)
```

## Component Visualization

Show individual Gaussian components:

```python
from tgmm.plotting import plot_components

plot_components(
    gmm,
    component_ids=[0, 1, 2],  # Which components to show
    show_means=True,
    show_covariances=True
)
```

## Convergence Monitoring

Track convergence during fitting:

```python
from tgmm.plotting import plot_convergence

gmm = GaussianMixture(n_components=3, n_features=2, verbose=True)
gmm.fit(X)

# Plot log-likelihood over iterations
plot_convergence(gmm.lower_bound_history_)
```

## Customization

### Color Maps

```python
plot_gmm_2d(
    X, 
    gmm,
    cmap='viridis',  # Color map for clusters
    show_ellipses=True
)
```

### Figure Size and DPI

```python
plot_gmm_2d(
    X, 
    gmm,
    figsize=(12, 8),
    dpi=150
)
```

### Custom Styling

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
plot_gmm_2d(
    X, 
    gmm,
    ax=ax,  # Use custom axes
    show_ellipses=True,
    ellipse_color='red',
    center_marker='X',
    center_size=200
)
ax.set_xlabel('Feature 1', fontsize=14)
ax.set_ylabel('Feature 2', fontsize=14)
plt.tight_layout()
plt.show()
```

## Advanced Examples

### Side-by-Side Comparison

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, cov_type in enumerate(['full', 'diag', 'spherical']):
    gmm = GaussianMixture(
        n_components=3, 
        n_features=2,
        covariance_type=cov_type
    )
    gmm.fit(X)
    
    plot_gmm_2d(
        X, 
        gmm,
        ax=axes[i],
        show_ellipses=True,
        title=f'{cov_type.capitalize()} Covariance'
    )

plt.tight_layout()
plt.show()
```

### Animation of EM Steps

```python
from tgmm.plotting import animate_em_steps

gmm = GaussianMixture(n_components=3, n_features=2, max_iter=20)
gmm.fit(X, track_steps=True)  # Enable step tracking

# Create animation showing EM iterations
animate_em_steps(
    X, 
    gmm,
    save_path='em_animation.gif',
    fps=2
)
```

### Probability Contours

```python
from tgmm.plotting import plot_probability_contours

plot_probability_contours(
    X, 
    gmm,
    n_levels=10,
    show_data=True,
    log_scale=True  # Use log probability
)
```

## PCA Visualization Details

When working with high-dimensional data:

```python
from tgmm.plotting import plot_gmm_pca
import numpy as np

# High-dimensional data
X_hd = torch.randn(1000, 50)  # 50 dimensions
gmm = GaussianMixture(n_components=5, n_features=50)
gmm.fit(X_hd)

# Project to 2D and visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot with cluster coloring
plot_gmm_pca(
    X_hd, 
    gmm,
    n_components=2,
    ax=ax1,
    show_variance=True,
    title='PCA Projection with Clusters'
)

# Plot showing component densities
plot_gmm_pca(
    X_hd, 
    gmm,
    n_components=2,
    ax=ax2,
    show_density=True,
    title='PCA Projection with Density'
)

plt.tight_layout()
plt.show()
```

### Explained Variance

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(X_hd.numpy())

# Plot variance explained
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(pca.explained_variance_ratio_, 'o-')
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.xlabel('Component')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance')
plt.tight_layout()
plt.show()
```

## Interactive Plots with Plotly

```python
from tgmm.plotting import plot_gmm_interactive

# Create interactive 3D plot
plot_gmm_interactive(
    X_3d, 
    gmm,
    title='Interactive GMM Visualization',
    show_ellipses=True,
    save_html='gmm_plot.html'
)
```

## Tips for Effective Visualization

1. **Choose appropriate projections**: Use PCA for high-dimensional data
2. **Adjust transparency**: Use `alpha` parameter to see overlapping clusters
3. **Scale data**: Standardize features for better ellipse visualization
4. **Use confidence levels**: Show different probability contours (e.g., 0.68, 0.95, 0.99)
5. **Compare models**: Plot multiple covariance types side-by-side
6. **Monitor convergence**: Plot log-likelihood to ensure proper fitting

## Complete API Reference

For full details on all plotting functions, see the [API Reference](../api/plotting.md).
