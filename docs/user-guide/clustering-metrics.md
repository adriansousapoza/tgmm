# Clustering Metrics

The `ClusteringMetrics` class provides comprehensive evaluation metrics for clustering results, including both supervised and unsupervised measures.

`ClusteringMetrics` is a collection of `@staticmethod`s -- there is no instance to construct,
call the methods directly on the class.

## Overview

Metrics are divided into two categories:

**Unsupervised** (no ground truth labels needed, but require `n_components`):
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Dunn Index
- BIC / AIC

**Supervised** (requires ground truth labels):
- Rand Index / Adjusted Rand Index (ARI)
- Mutual Information / Normalized / Adjusted Mutual Information (NMI)
- Purity
- Fowlkes-Mallows, Homogeneity, Completeness, V-Measure
- Confusion Matrix
- Classification Report (precision, recall, F1, Jaccard, ROC-AUC per class)

## Usage

```python
from tgmm import ClusteringMetrics, GaussianMixture
import torch

# Fit GMM
gmm = GaussianMixture(n_components=3, n_features=2)
gmm.fit(X)
predicted_labels = gmm.predict(X)

# Compute unsupervised metrics (call directly on the class)
silhouette = ClusteringMetrics.silhouette_score(X, predicted_labels, n_components=3)
davies_bouldin = ClusteringMetrics.davies_bouldin_index(X, predicted_labels, n_components=3)
calinski_harabasz = ClusteringMetrics.calinski_harabasz_score(X, predicted_labels, n_components=3)

print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.3f}")

# If you have ground truth labels
ari = ClusteringMetrics.adjusted_rand_score(true_labels, predicted_labels)
nmi = ClusteringMetrics.normalized_mutual_info_score(true_labels, predicted_labels)
purity = ClusteringMetrics.purity_score(true_labels, predicted_labels)

print(f"ARI: {ari:.3f}")
print(f"NMI: {nmi:.3f}")
print(f"Purity: {purity:.3f}")
```

## Unsupervised Metrics

These all take the number of clusters (`n_components`) in addition to the data and labels.

### Silhouette Score

Measures how similar objects are to their own cluster compared to other clusters.

**Range**: [-1, 1]
**Interpretation**: Higher is better
- 1: Perfect clustering
- 0: Overlapping clusters
- -1: Wrong clustering

```python
silhouette = ClusteringMetrics.silhouette_score(X, labels, n_components=3)
```

### Davies-Bouldin Index

Measures the average similarity between each cluster and its most similar cluster.

**Range**: [0, ∞)
**Interpretation**: Lower is better

```python
db_index = ClusteringMetrics.davies_bouldin_index(X, labels, n_components=3)
```

### Calinski-Harabasz Index

Ratio of between-cluster dispersion to within-cluster dispersion.

**Range**: [0, ∞)
**Interpretation**: Higher is better

```python
ch_index = ClusteringMetrics.calinski_harabasz_score(X, labels, n_components=3)
```

### Dunn Index

Ratio of the smallest inter-cluster distance to the largest intra-cluster distance.

**Range**: [0, ∞)
**Interpretation**: Higher is better

```python
dunn = ClusteringMetrics.dunn_index(X, labels, n_components=3)
```

## Supervised Metrics

### Adjusted Rand Index (ARI)

Measures similarity between two clusterings, adjusted for chance.

**Range**: [-1, 1]
**Interpretation**: Higher is better
- 1: Perfect agreement
- 0: Random labeling
- Negative: Worse than random

```python
ari = ClusteringMetrics.adjusted_rand_score(true_labels, pred_labels)
```

### Normalized Mutual Information (NMI)

Measures mutual information normalized to [0, 1].

**Range**: [0, 1]
**Interpretation**: Higher is better
- 1: Perfect agreement
- 0: Independent labelings

```python
nmi = ClusteringMetrics.normalized_mutual_info_score(true_labels, pred_labels)
```

### Purity

Fraction of correctly clustered samples in the majority class of each cluster.

**Range**: [0, 1]
**Interpretation**: Higher is better

```python
purity = ClusteringMetrics.purity_score(true_labels, pred_labels)
```

### Confusion Matrix

```python
conf_matrix = ClusteringMetrics.confusion_matrix(true_labels, pred_labels)
print(conf_matrix)
```

### Classification Report (Precision / Recall / F1)

Returns a dict keyed by class label, each with `precision`, `recall`, `f1-score`,
`support`, `jaccard`, and `roc_auc`:

```python
report = ClusteringMetrics.classification_report(true_labels, pred_labels)
for label, scores in report.items():
    print(f"Class {label}: F1={scores['f1-score']:.3f}, precision={scores['precision']:.3f}, recall={scores['recall']:.3f}")
```

## Model Comparison Example

```python
from tgmm import GaussianMixture, ClusteringMetrics

results = []

for cov_type in ['full', 'diag', 'spherical']:
    gmm = GaussianMixture(
        n_components=3,
        n_features=2,
        covariance_type=cov_type,
        random_state=42
    )
    gmm.fit(X)
    labels = gmm.predict(X)

    results.append({
        'type': cov_type,
        'silhouette': ClusteringMetrics.silhouette_score(X, labels, n_components=3),
        'davies_bouldin': ClusteringMetrics.davies_bouldin_index(X, labels, n_components=3),
        'log_likelihood': gmm.lower_bound_
    })

# Print comparison
for result in results:
    print(f"{result['type']:12s} - "
          f"Silhouette: {result['silhouette']:.3f}, "
          f"DB: {result['davies_bouldin']:.3f}, "
          f"LogLik: {result['log_likelihood']:.2f}")
```

## Complete API Reference

For full details on all metrics, see the [API Reference](../api/clustering-metrics.md).
