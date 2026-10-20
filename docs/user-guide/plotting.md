# Plotting and Visualization

The `tgmm.plotting` module provides `plot_gmm` for visualizing 2D mixture-model clustering results,
[`plot_vmf` and `plot_vmf_cosine`](#von-mises-fisher-mixtures) for directional (unit-norm) data, plus a few small helpers: `dynamic_figsize`, `match_predicted_to_true_labels`,
`confidence_ellipse_std_devs` and `gmm_proxy`. `plot_gmm` accepts any fitted `tgmm` mixture (it only needs
`means_`, `covariances_`, `weights_`, `covariance_type` and `predict`), including every GPCM covariance
code, and the module exports the colour constants `CATEGORICAL_PALETTE`, `SEQUENTIAL_CMAP` and
`STATUS_CRITICAL` used as its defaults.

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

### Ellipses that enclose a given probability

`ellipse_std_devs` is in units of standard deviations (Mahalanobis radius). In $d=2$ a radius of $r$ encloses
$1-e^{-r^2/2}$ of a Gaussian's mass -- 39% for $r=1$, 86.5% for $r=2$, 98.9% for $r=3$ -- not the 68/95/99.7% of the 1-D rule.
To draw ellipses with a chosen coverage $q$, convert it through the $\chi^2_2$ quantile, $r=\sqrt{\chi^2_{2,q}}$,
with `confidence_ellipse_std_devs`:

```python
from tgmm import confidence_ellipse_std_devs

plot_gmm(X, gmm, ellipse_std_devs=confidence_ellipse_std_devs([0.5, 0.95]))  # 50% and 95% regions
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

## von Mises-Fisher Mixtures

`plot_gmm` draws Euclidean ellipses, so it cannot represent a
[`VonMisesFisherMixture`](other-mixture-families.md#vonmisesfishermixture-directional-data), whose data live on
the unit sphere and whose components are described by a mean *direction* $\mu_k$ and a concentration $\kappa_k$.
Two functions cover that case.

### `plot_vmf`: the circle ($d=2$) and the sphere ($d=3$)

```python
from tgmm import VonMisesFisherMixture, plot_vmf

vmf = VonMisesFisherMixture(n_components=3).fit(X_unit)   # X_unit: (n, 3), rows of unit norm

fig = plt.figure()
ax = fig.add_subplot(projection='3d')                      # a 3-D axes for d=3; omit `ax` to create one
plot_vmf(X_unit, vmf, ax=ax, cap_masses=(0.5, 0.9))
```

Points are coloured by predicted component (or by `true_labels=`). Each mean direction is an arrow from the
origin, and each concentration is drawn as nested **spherical caps** -- the vMF analogue of confidence
ellipses. A cap of angular radius $\theta_m$ around $\mu_k$ contains probability $m$ of the component, where
$\theta_m$ solves $P(\theta\le\theta_m)=m$ under the angle density

$$
f_\kappa(\theta)\;\propto\;e^{\kappa\cos\theta}\sin^{d-2}\theta,\qquad\theta=\arccos(\mu^{\top}x)\in[0,\pi].
$$

(The $\sin^{d-2}\theta$ factor is the surface area of the ring at angle $\theta$; the density is normalised
numerically.) On the circle the caps are arcs just outside the unit circle, on the sphere they are rings of
constant angular distance from $\mu_k$. `show_means`, `show_caps`, `cap_masses`, `legend_labels`,
`view_elev` and `view_azim` control the rest.

### `plot_vmf_cosine`: any dimension

```python
from tgmm import plot_vmf_cosine

plot_vmf_cosine(X_embeddings, vmf)   # works for d = 768 just as for d = 5
```

For every component it histograms the angle between each assigned point and the component's mean direction
and overlays the fitted $f_{\kappa_k}(\theta)$ (dashed). If the vMF assumption holds, histogram and curve
agree. In high dimension the typical angle is far from $0^\circ$ even for large $\kappa$: the mode solves
$\kappa\sin^2\theta=(d-2)\cos\theta$.

The helpers `vmf_angle_density(kappa, n_features, theta)` and `vmf_cap_angle(kappa, n_features, mass)` expose the
angle density and cap radius directly. See the
[von Mises-Fisher tutorial](../notebooks/17_von_mises_fisher.ipynb) for a gallery of vMF distributions on the
circle and the sphere and worked examples of all three.
