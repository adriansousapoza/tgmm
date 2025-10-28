# Clustering Metrics

The `ClusteringMetrics` class provides comprehensive evaluation metrics for clustering results, including both supervised and unsupervised measures.

## Overview

Metrics are divided into two categories:

**Unsupervised** (no ground truth labels needed):
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

**Supervised** (requires ground truth labels):
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Purity
- Accuracy
- Confusion Matrix
- F1 Score

## Usage

```python
from tgmm import ClusteringMetrics, GaussianMixture
import torch

# Fit GMM
gmm = GaussianMixture(n_components=3, n_features=2)
gmm.fit(X)
predicted_labels = gmm.predict(X)

# Create metrics calculator
metrics = ClusteringMetrics()

# Compute unsupervised metrics
silhouette = metrics.silhouette_score(X, predicted_labels)
davies_bouldin = metrics.davies_bouldin_score(X, predicted_labels)
calinski_harabasz = metrics.calinski_harabasz_score(X, predicted_labels)

print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.3f}")

# If you have ground truth labels
ari = metrics.adjusted_rand_index(true_labels, predicted_labels)
nmi = metrics.normalized_mutual_info(true_labels, predicted_labels)
purity = metrics.purity(true_labels, predicted_labels)

print(f"ARI: {ari:.3f}")
print(f"NMI: {nmi:.3f}")
print(f"Purity: {purity:.3f}")
```

## Unsupervised Metrics

### Silhouette Score

Measures how similar objects are to their own cluster compared to other clusters.

**Range**: [-1, 1]  
**Interpretation**: Higher is better  
- 1: Perfect clustering
- 0: Overlapping clusters
- -1: Wrong clustering

```python
silhouette = metrics.silhouette_score(X, labels)
```

### Davies-Bouldin Index

Measures the average similarity between each cluster and its most similar cluster.

**Range**: [0, ∞)  
**Interpretation**: Lower is better

```python
db_index = metrics.davies_bouldin_score(X, labels)
```

### Calinski-Harabasz Index

Ratio of between-cluster dispersion to within-cluster dispersion.

**Range**: [0, ∞)  
**Interpretation**: Higher is better

```python
ch_index = metrics.calinski_harabasz_score(X, labels)
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
ari = metrics.adjusted_rand_index(true_labels, pred_labels)
```

### Normalized Mutual Information (NMI)

Measures mutual information normalized to [0, 1].

**Range**: [0, 1]  
**Interpretation**: Higher is better
- 1: Perfect agreement
- 0: Independent labelings

```python
nmi = metrics.normalized_mutual_info(true_labels, pred_labels)
```

### Purity

Fraction of correctly clustered samples in the majority class of each cluster.

**Range**: [0, 1]  
**Interpretation**: Higher is better

```python
purity = metrics.purity(true_labels, pred_labels)
```

### Confusion Matrix

```python
conf_matrix = metrics.confusion_matrix(true_labels, pred_labels)
print(conf_matrix)
```

### F1 Score

Harmonic mean of precision and recall.

```python
f1 = metrics.f1_score(true_labels, pred_labels)
```

## Model Comparison Example

```python
from tgmm import GaussianMixture, ClusteringMetrics

metrics = ClusteringMetrics()
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
        'silhouette': metrics.silhouette_score(X, labels),
        'davies_bouldin': metrics.davies_bouldin_score(X, labels),
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
