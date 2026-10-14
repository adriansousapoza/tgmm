# Clustering Metrics

The `ClusteringMetrics` class provides comprehensive evaluation metrics for clustering results, including both supervised and unsupervised measures.

`ClusteringMetrics` is a collection of `@staticmethod`s -- there is no instance to construct,
call the methods directly on the class.

## Overview

Metrics are divided into two categories:

**Unsupervised** (no ground truth labels needed; `n_components` is optional
-- see [Shared options](#shared-options-for-unsupervised-indexes) below):
- Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Dunn Index
- WB-Index, Wemmert-Gancarski Index, DBCV -- three indexes that ranked at the
  top of the Simpson, Campello & Stojanovski (2025) clustering-validity
  benchmark; prefer these over Dunn Index (ranked 25th of 26 in that same
  benchmark, kept only for completeness)

BIC/AIC/ICL aren't here -- they only need a model's own log-likelihood and
parameter count, so they're `bic`/`aic`/`icl` methods on `GaussianMixture`
directly (`gmm.bic(X)`, `gmm.aic(X)`, `gmm.icl(X)`) -- including when
constructed with `n_components=None` for Gibbs-sampling mode -- not
`ClusteringMetrics` calls. See [Model Selection](#model-selection) below.

**Supervised** (requires ground truth labels):
- Rand Index / Adjusted Rand Index (ARI)
- Mutual Information / Normalized / Adjusted Mutual Information (NMI)
- Purity
- Fowlkes-Mallows, Homogeneity, Completeness, V-Measure (weighted via `beta`)
- Normalized Information Distance (NID)
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

### Shared options for unsupervised indexes

`silhouette_score`, `davies_bouldin_index`, `calinski_harabasz_score`,
`dunn_index`, `wb_index`, and `wemmert_gancarski_index` all share the same
extra keyword arguments:

- `n_components`: optional now -- if omitted, it's inferred from the
  distinct values in `labels`, which no longer need to be contiguous or
  0-indexed. Passing it explicitly (the historical required argument)
  keeps working exactly as before.
- `metric`: `'euclidean'` (default), `'manhattan'`, `'cosine'`,
  `'mahalanobis'` (pass the precision matrix as `VI`), or -- for
  `silhouette_score` and `dunn_index` only -- `'precomputed'` (pass an
  `(n, n)` distance matrix as `X`). The other four indexes need to average
  points into a cluster centroid, which a precomputed distance matrix
  doesn't support, and raise a clear `ValueError` if asked to.
  Passing the pooled precision of a fitted `GaussianMixture`
  (`torch.linalg.inv` of the weighted average of `covariances_`) as `VI`
  with `metric='mahalanobis'` gives the GMM-matched version of the index.
- `noise_label`: a label value (e.g. `-1` for `HDBSCAN` output) marking
  noise points. Noise rows are dropped before computing the index, and the
  result is scaled by `(n - n_noise) / n`.
- `chunk_size` (silhouette/Dunn only): row-chunk size for the pairwise
  distance computation, bounding peak memory regardless of `n_samples`
  (default `4096`). Chunking is exact, not an approximation.

`dbcv` has its own, simpler signature (`metric`, `noise_label=-1`) -- its
noise handling is built into the index's own weighting formula rather than
the generic rescaling above.

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

> Ranked 25th of 26 indexes on partition ranking in the Simpson, Campello &
> Stojanovski (2025) benchmark -- kept for completeness/familiarity, not as
> a recommended choice. Prefer WB-Index, Wemmert-Gancarski, or DBCV below.

### WB-Index

`K * SSW / SSB` (Zhao & Fränti, 2014) -- same SSW/SSB as Calinski-Harabasz.

**Range**: [0, ∞)
**Interpretation**: Lower is better

```python
wb = ClusteringMetrics.wb_index(X, labels, n_components=3)
```

### Wemmert-Gancarski Index

Compares each point's distance to its own centroid against its distance to
the *nearest other* centroid; penalises clusters where points sit closer to
another cluster's centroid than their own.

**Range**: [0, ∞)
**Interpretation**: Higher is better

```python
wg = ClusteringMetrics.wemmert_gancarski_index(X, labels, n_components=3)
```

### DBCV (Density-Based Clustering Validation)

Moulavi, Jaskowiak, Campello, Zimek & Sander (2014). Built on an "all-points
core distance" and per-cluster minimum spanning trees over the
mutual-reachability graph -- the natural validity index for density-based
partitions (e.g. `HDBSCAN` output, including noise points), where
centroid-based indexes like Davies-Bouldin don't make sense for
non-convex clusters.

**Range**: [-1, 1]
**Interpretation**: Higher is better

```python
score = ClusteringMetrics.dbcv(X, labels, noise_label=-1)  # -1 is HDBSCAN's noise label
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

### Homogeneity, Completeness, V-Measure

Homogeneity asks "does each cluster contain only one true class?";
completeness asks "is each true class kept in one cluster?" (Rosenberg &
Hirschberg, 2007). V-measure is their weighted harmonic mean, via `beta`
(`beta=1.0`, the default, weighs them equally; `beta > 1` favours
completeness, `beta < 1` favours homogeneity).

**Range**: [0, 1] each. **Interpretation**: Higher is better.

```python
h = ClusteringMetrics.homogeneity_score(true_labels, pred_labels)
c = ClusteringMetrics.completeness_score(true_labels, pred_labels)
v = ClusteringMetrics.v_measure_score(true_labels, pred_labels, beta=1.0)
```

### Normalized Information Distance (NID)

`1 - I(U; V) / max(H(U), H(V))` (Vinh, Epps & Bailey, 2010) -- an
information-theoretic *distance* between two partitions.

**Range**: [0, 1]
**Interpretation**: Lower is better; 0 for identical partitions.

```python
nid = ClusteringMetrics.normalized_information_distance(true_labels, pred_labels)
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
        'log_likelihood': gmm.lower_bound_,
        'bic': gmm.bic(X),
        'aic': gmm.aic(X),
    })

# Print comparison
for result in results:
    print(f"{result['type']:12s} - "
          f"Silhouette: {result['silhouette']:.3f}, "
          f"DB: {result['davies_bouldin']:.3f}, "
          f"LogLik: {result['log_likelihood']:.2f}, "
          f"BIC: {result['bic']:.2f}, "
          f"AIC: {result['aic']:.2f}")
```

## Model Selection

### ICL (Integrated Completed Likelihood)

`icl(X, mode='hard')` on a fitted `GaussianMixture` adds an entropy penalty
to `bic` for ambiguous, overlapping-cluster assignments (Biernacki, Celeux &
Govaert, 2000). Lower is better, same convention as `bic`; `ICL >= BIC`
always, and the two are identical when responsibilities are one-hot
(perfectly separated clusters).

```python
gmm = GaussianMixture(n_components=3, n_features=2, random_state=0)
gmm.fit(X)
print(gmm.bic(X), gmm.icl(X))       # icl(X) >= bic(X)
print(gmm.icl(X, mode='soft'))      # entropy-form penalty instead of hard-argmax
```

ICL tends to prefer *fewer, better-separated* components than BIC on data
with overlapping clusters -- see the
[Model Selection notebook](../notebooks/13_model_selection.ipynb) for a
worked BIC-vs-ICL comparison where they pick different `K`.

### `select_model`: sweeping `n_components` and `covariance_type`

```python
from tgmm import select_model

result = select_model(
    X,
    n_components=range(1, 11),
    covariance_types=('full', 'diag', 'spherical', 'tied_full'),
    criterion='bic',   # or 'aic', 'icl'
    n_init=3,
    random_state=0,
)

print(result.best_.n_components, result.best_.covariance_type)
df = result.to_dataframe()  # lazy pandas import
```

`result.table` has one row per `(n_components, covariance_type)` combination
(`bic`, `aic`, `icl`, `log_likelihood`, `converged`); a combination whose fit
raises (e.g. a singular covariance at a large `K`) gets `None` scores
instead of aborting the whole sweep. `result.models_[(K, covariance_type)]`
gives the fitted model for any successful combination. See the
[API reference](../api/model-selection.md) for the full signature.

## Complete API Reference

For full details on all metrics, see the [API Reference](../api/clustering-metrics.md).
