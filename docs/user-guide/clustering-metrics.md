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
parameter count, so they're `bic`/`aic`/`icl` methods on every fitted mixture
directly (`gmm.bic(X)`, `gmm.aic(X)`, `gmm.icl(X)`) -- including when
`GaussianMixture` is constructed with `n_components=None` for Gibbs-sampling
mode -- not `ClusteringMetrics` calls. See [Model Selection](#model-selection) below.

**Supervised** (requires ground truth labels):
- Rand Index / Adjusted Rand Index (ARI)
- Mutual Information / Normalized (NMI) / Adjusted (AMI, approximate) Mutual Information
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
  result is then penalized for having discarded points: with
  $f = (n - n_{\text{noise}})/n$, indexes where higher is better
  (silhouette, CH, Dunn, WG) are multiplied by $f$, and indexes where lower
  is better (Davies-Bouldin, WB) are divided by $f$.
- `chunk_size` (silhouette/Dunn only): row-chunk size for the pairwise
  distance computation, bounding peak memory regardless of `n_samples`
  (default `4096`). Chunking is exact, not an approximation.

`dbcv` has its own, simpler signature (`metric`, `noise_label=-1`) -- its
noise handling is built into the index's own weighting formula rather than
the generic rescaling above.

### Notation

Throughout, a clustering assigns each of $n$ points $x_i$ to one of $K$
clusters $C_1,\dots,C_K$ with sizes $n_k = |C_k|$ and centroids
$c_k = \frac{1}{n_k}\sum_{x\in C_k} x$ (the plain coordinate-wise mean --
note this is the *hard-assignment* centroid, not the GMM's `means_`), and
$d(\cdot,\cdot)$ is the chosen `metric` (Euclidean by default). When
`noise_label` is set, noise rows are removed first and $n$ counts only the
remaining points.

### Silhouette Score

For a point $i$ in cluster $C_{k(i)}$, let

$$
a(i) = \frac{1}{n_{k(i)} - 1}\sum_{j \in C_{k(i)},\, j\neq i} d(i,j), \qquad
b(i) = \min_{l \neq k(i)} \frac{1}{n_l}\sum_{j \in C_l} d(i,j),
$$

the mean distance to its own cluster and to the *nearest other* cluster.
Then

$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}, \qquad
\text{Silhouette} = \frac{1}{n}\sum_i s(i),
$$

with $s(i) = 0$ for singleton clusters. Requires at least two clusters.

**Range**: [-1, 1]
**Interpretation**: Higher is better
- 1: Perfect clustering
- 0: Overlapping clusters
- -1: Wrong clustering

```python
silhouette = ClusteringMetrics.silhouette_score(X, labels, n_components=3)
```

### Davies-Bouldin Index

With the within-cluster scatter $S_k = \frac{1}{n_k}\sum_{x\in C_k} d(x, c_k)$
(mean distance to the centroid) and centroid separation
$M_{kl} = d(c_k, c_l)$,

$$
\mathrm{DB} = \frac{1}{K}\sum_{k=1}^{K}\max_{l \neq k}\frac{S_k + S_l}{M_{kl}},
$$

the average, over clusters, of the similarity to the *most similar* other
cluster (large when two clusters are wide and close).

**Range**: [0, ∞)
**Interpretation**: Lower is better

```python
db_index = ClusteringMetrics.davies_bouldin_index(X, labels, n_components=3)
```

### Calinski-Harabasz Index

Let $\bar x$ be the global mean,

$$
\mathrm{SSW} = \sum_{k}\sum_{x\in C_k} d(x, c_k)^2, \qquad
\mathrm{SSB} = \sum_{k} n_k\, d(c_k, \bar x)^2,
$$

the within- and between-cluster sums of squared distances. Then

$$
\mathrm{CH} = \frac{\mathrm{SSB}/(K-1)}{\mathrm{SSW}/(n-K)},
$$

a variance-ratio (pseudo-$F$) statistic. For the Euclidean metric
$\mathrm{SSW}+\mathrm{SSB}$ is the total sum of squares.

**Range**: [0, ∞)
**Interpretation**: Higher is better

```python
ch_index = ClusteringMetrics.calinski_harabasz_score(X, labels, n_components=3)
```

### Dunn Index

$$
D = \frac{\min_{k \neq l}\; d(C_k, C_l)}{\max_{k}\;\operatorname{diam}(C_k)}, \qquad
d(C_k, C_l) = \min_{x\in C_k,\, y\in C_l} d(x,y), \quad
\operatorname{diam}(C_k) = \max_{x,y\in C_k} d(x,y),
$$

the smallest between-cluster gap over the largest within-cluster diameter.
It depends on only two pairs of points, so it is very sensitive to
outliers.

**Range**: [0, ∞)
**Interpretation**: Higher is better

```python
dunn = ClusteringMetrics.dunn_index(X, labels, n_components=3)
```

> Ranked 25th of 26 indexes on partition ranking in the Simpson, Campello &
> Stojanovski (2025) benchmark -- kept for completeness/familiarity, not as
> a recommended choice. Prefer WB-Index, Wemmert-Gancarski, or DBCV below.

### WB-Index

$$
\mathrm{WB} = K\,\frac{\mathrm{SSW}}{\mathrm{SSB}}
$$

(Zhao & Fränti, 2014) -- same SSW/SSB as Calinski-Harabasz, with the
number of clusters as a built-in penalty.

**Range**: [0, ∞)
**Interpretation**: Lower is better

```python
wb = ClusteringMetrics.wb_index(X, labels, n_components=3)
```

### Wemmert-Gancarski Index

Compares each point's distance to its own centroid against its distance to
the *nearest other* centroid:

$$
R(x) = \frac{d(x, c_{k(x)})}{\min_{l \neq k(x)} d(x, c_l)}, \qquad
J_k = \max\Big(0,\; 1 - \frac{1}{n_k}\sum_{x\in C_k} R(x)\Big), \qquad
\mathrm{WG} = \frac{1}{n}\sum_k n_k J_k .
$$

A cluster contributes $0$ when, on average, its points are no closer to
their own centroid than to another's.

**Range**: [0, ∞) (in practice [0, 1])
**Interpretation**: Higher is better

```python
wg = ClusteringMetrics.wemmert_gancarski_index(X, labels, n_components=3)
```

### DBCV (Density-Based Clustering Validation)

Moulavi, Jaskowiak, Campello, Zimek & Sander (2014). The natural validity
index for density-based partitions (e.g. `HDBSCAN` output, including noise
points), where centroid-based indexes like Davies-Bouldin don't make sense
for non-convex clusters. For every cluster $C$ (computed within the
cluster, using the $d_f$ = `n_features`-th power of distances):

1. **All-points core distance** of $x\in C$:
   $$a_{\text{core}}(x) = \Big(\frac{1}{|C|-1}\sum_{y\in C,\,y\neq x} d(x,y)^{-d_f}\Big)^{-1/d_f}.$$
2. **Mutual reachability**: $d_{\text{mreach}}(x,y) = \max\{a_{\text{core}}(x),\, a_{\text{core}}(y),\, d(x,y)\}$.
3. A **minimum spanning tree** (MST) over $C$'s mutual-reachability graph.
   **Density sparseness** $\mathrm{DSC}(C)$ = the largest MST edge weight.
4. **Density separation** $\mathrm{DSPC}(C_i, C_j)$ = the smallest
   mutual-reachability distance between an *internal* node (MST degree
   $\ge 2$) of $C_i$ and an internal node of $C_j$.

The per-cluster validity and the index are

$$
V(C_i) = \frac{\min_{j\neq i}\mathrm{DSPC}(C_i,C_j) - \mathrm{DSC}(C_i)}
              {\max\{\min_{j\neq i}\mathrm{DSPC}(C_i,C_j),\; \mathrm{DSC}(C_i)\}},
\qquad
\mathrm{DBCV} = \sum_i \frac{n_i}{n}\, V(C_i),
$$

where $n$ **includes the noise points** (they add to the denominator only),
which is DBCV's own noise handling -- so it does *not* use the generic
`noise_label` rescaling of the other indexes. Clusters with fewer than 2
points contribute $0$. Note this is the all-points core distance of the DBCV
paper, not the $k$-NN core distance that `HDBSCAN` itself uses internally.

**Range**: [-1, 1]
**Interpretation**: Higher is better

```python
score = ClusteringMetrics.dbcv(X, labels, noise_label=-1)  # -1 is HDBSCAN's noise label
```

## Supervised Metrics

These compare a predicted partition $V$ (`labels_pred`) with a reference
partition $U$ (`labels_true`). Most are built from the **contingency table**
$n_{ij} = |\{x : u(x) = i,\ v(x) = j\}|$ with row sums $a_i = \sum_j n_{ij}$,
column sums $b_j = \sum_i n_{ij}$ and total $n$. Cluster labels are
arbitrary, so every score here is invariant to relabeling the clusters.

!!! note "Label format"
    The supervised scores build the contingency table by indexing, so they
    expect **non-negative, 0-indexed integer labels** (e.g. from
    `gmm.predict`). To score density-based output containing a `-1` noise
    label, drop or remap the noise rows first.

### Rand Index and Adjusted Rand Index (ARI)

Count pairs of points: $\mathrm{TP}$ = pairs together in both partitions,
$\mathrm{TN}$ = apart in both, $\mathrm{FP}$/$\mathrm{FN}$ the
disagreements. Then

$$
\mathrm{RI} = \frac{\mathrm{TP} + \mathrm{TN}}{\binom{n}{2}}, \qquad
\mathrm{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \Big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\Big]\Big/\binom{n}{2}}
{\tfrac12\Big[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\Big] - \Big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\Big]\Big/\binom{n}{2}}.
$$

ARI subtracts the Rand index's expected value under random labeling with
the same cluster sizes (the generalized hypergeometric model), so a random
partition scores $\approx 0$.

**Range**: ARI in [-1, 1]; RI in [0, 1]
**Interpretation**: Higher is better
- 1: Perfect agreement
- 0: Random labeling
- Negative: Worse than random

```python
ari = ClusteringMetrics.adjusted_rand_score(true_labels, pred_labels)
```

### Mutual Information, NMI and AMI

With $p_{ij} = n_{ij}/n$, $p_i = a_i/n$, $p_j = b_j/n$, entropies
$H(U) = -\sum_i p_i\ln p_i$, $H(V) = -\sum_j p_j \ln p_j$:

$$
\mathrm{MI}(U,V) = \sum_{ij} p_{ij}\ln\frac{p_{ij}}{p_i\,p_j}, \qquad
\mathrm{NMI} = \frac{2\,\mathrm{MI}}{H(U) + H(V)} .
$$

NMI uses the arithmetic-mean normalization and lies in $[0, 1]$ (all in
nats). The **AMI** (`adjusted_mutual_info_score`) is meant to correct MI for
chance, $\mathrm{AMI} = \dfrac{\mathrm{MI} - \mathbb{E}[\mathrm{MI}]}{\tfrac12(H(U)+H(V)) - \mathbb{E}[\mathrm{MI}]}$.

!!! warning "AMI is an approximation"
    The exact $\mathbb{E}[\mathrm{MI}]$ under the hypergeometric model
    (Vinh, Epps & Bailey, 2010) is an expensive triple sum. tgmm replaces it
    with the rough approximation $\mathbb{E}[\mathrm{MI}] \approx H(U)H(V)/n$,
    so `adjusted_mutual_info_score` is **not** numerically equal to
    scikit-learn's. It still behaves like a chance-corrected score (near $0$
    for random labelings, $1$ for identical ones), but use scikit-learn's
    implementation if you need exact AMI values. NMI and ARI, in contrast,
    match scikit-learn.

**Range**: NMI in [0, 1]; AMI $\lesssim 1$, may be slightly negative
**Interpretation**: Higher is better
- 1: Perfect agreement
- 0: Independent labelings

```python
nmi = ClusteringMetrics.normalized_mutual_info_score(true_labels, pred_labels)
```

### Fowlkes-Mallows

The geometric mean of pairwise precision and recall,

$$
\mathrm{FM} = \sqrt{\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}\cdot\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}},
\qquad \mathrm{TP} = \sum_{ij}\binom{n_{ij}}{2}.
$$

**Range**: [0, 1] -- Higher is better.

### Homogeneity, Completeness, V-Measure

Homogeneity asks "does each cluster contain only one true class?";
completeness asks "is each true class kept in one cluster?" (Rosenberg &
Hirschberg, 2007):

$$
h = 1 - \frac{H(U \mid V)}{H(U)}, \qquad
c = 1 - \frac{H(V \mid U)}{H(V)}, \qquad
V_\beta = \frac{(1+\beta)\,h\,c}{\beta\,h + c},
$$

with the conventions $h = 1$ when $H(U) = 0$ and $c = 1$ when $H(V) = 0$.
V-measure is their weighted harmonic mean, via `beta` (`beta=1.0`, the
default, weighs them equally and gives $2hc/(h+c)$, the same as NMI with
arithmetic normalization; `beta > 1` favours completeness, `beta < 1`
favours homogeneity).

**Range**: [0, 1] each. **Interpretation**: Higher is better.

```python
h = ClusteringMetrics.homogeneity_score(true_labels, pred_labels)
c = ClusteringMetrics.completeness_score(true_labels, pred_labels)
v = ClusteringMetrics.v_measure_score(true_labels, pred_labels, beta=1.0)
```

### Normalized Information Distance (NID)

$$
\mathrm{NID} = 1 - \frac{I(U;V)}{\max\{H(U), H(V)\}}
$$

(Vinh, Epps & Bailey, 2010) -- an information-theoretic *distance* between
two partitions.

**Range**: [0, 1]
**Interpretation**: Lower is better; 0 for identical partitions.

```python
nid = ClusteringMetrics.normalized_information_distance(true_labels, pred_labels)
```

### Purity

$$
\mathrm{purity} = \frac{1}{n}\sum_{j}\max_i n_{ij},
$$

the fraction of points that belong to the majority true class of their
predicted cluster. It never decreases as the number of clusters grows (a
cluster per point has purity 1), so only compare it at the same $K$.

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

Rows and columns are both indexed by the distinct values of
`true_labels`, so it assumes predicted cluster ids have already been aligned
to the true class ids (cluster ids are arbitrary in clustering: align them
first, e.g. by the Hungarian algorithm). Predicted ids that don't occur in
`true_labels` are not counted in any column.

### Classification Report (Precision / Recall / F1)

Treats each true class $c$ one-vs-rest, with $\mathrm{TP}_c$, $\mathrm{FP}_c$,
$\mathrm{FN}_c$:

$$
\text{precision} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \quad
\text{recall} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \quad
F_1 = \frac{2PR}{P+R}, \quad
\text{Jaccard} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}} .
$$

Returns a dict keyed by class label, each with `precision`, `recall`,
`f1-score`, `support`, `jaccard`, and `roc_auc`. Like the confusion matrix,
it matches predicted and true labels by *value*, so align cluster ids to
class ids first.

```python
report = ClusteringMetrics.classification_report(true_labels, pred_labels)
for label, scores in report.items():
    print(f"Class {label}: F1={scores['f1-score']:.3f}, precision={scores['precision']:.3f}, recall={scores['recall']:.3f}")
```

!!! warning "`roc_auc` is a naive hard-label AUC"
    `ClusteringMetrics.roc_auc_score` is computed from the *binary*
    one-vs-rest predictions (not from probabilities), where a ROC curve has
    a single operating point; the value is the area of a step path that
    depends on how tied predictions happen to be ordered, and does not equal
    the textbook value (for binary predictions, $(\mathrm{TPR} +
    \mathrm{TNR})/2$). Treat the `roc_auc` entry as indicative only; for
    a real AUC pass `predict_proba` outputs to `sklearn.metrics.roc_auc_score`.

### Other helpers

- `ClusteringMetrics.kl_divergence_gmm(gmm_p, gmm_q, n_samples=10000)`: a
  Monte-Carlo estimate of $D_{\mathrm{KL}}(p\,\Vert\,q) = \mathbb{E}_{x\sim p}[\log p(x) - \log q(x)]$
  between two fitted mixtures, averaging $\log p - \log q$ over samples drawn
  from `gmm_p`. It is not symmetric and has Monte-Carlo noise
  $\propto 1/\sqrt{n_{\text{samples}}}$.
- `ClusteringMetrics.evaluate_clustering(gmm_model, X, true_labels=None, metrics=None)`:
  computes a requested subset of the above for a fitted model in one call
  (supervised scores only when `true_labels` is given).
- `rand_score`, `mutual_info_score`: the unadjusted building blocks of ARI
  and NMI.

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

Likelihood-based criteria need a model's own log-likelihood and parameter
count, so they are `bic`/`aic`/`icl` **methods on every fitted mixture**
(`GaussianMixture`, `StudentTMixture`, `VonMisesFisherMixture` -- including
`GaussianMixture(n_components=None, ...)` in Gibbs mode), not
`ClusteringMetrics` calls. With $\ell(\hat\theta) = \sum_i \log p(x_i \mid \hat\theta)$
the maximized log-likelihood (`gmm.score(X) * n`) and $p$ the number of free
parameters:

$$
\mathrm{BIC} = -2\,\ell(\hat\theta) + p\,\ln n, \qquad
\mathrm{AIC} = -2\,\ell(\hat\theta) + 2p
$$

(both "lower is better"). BIC's $\ln n$ penalty is heavier than AIC's $2$
once $n > 7$, so BIC prefers smaller models and is consistent for the true
$K$ when the model family is correct; AIC targets predictive accuracy and
tends to pick more components.

### Free parameter counts

For `GaussianMixture`, $p = (K-1) + Kd + p_\Sigma$: $K-1$ free weights, $Kd$
means, and covariance parameters

| `covariance_type` | $p_\Sigma$ | | `covariance_type` | $p_\Sigma$ |
| --- | --- | --- | --- | --- |
| `full` / `VVV` | $K\,d(d+1)/2$ | | `VEI` | $K + (d-1)$ |
| `diag` / `VVI` | $Kd$ | | `EVI` | $1 + K(d-1)$ |
| `spherical` / `VII` | $K$ | | `VEE` | $K + (d-1) + d(d-1)/2$ |
| `tied_full` / `EEE` | $d(d+1)/2$ | | `EVE` | $1 + K(d-1) + d(d-1)/2$ |
| `tied_diag` / `EEI` | $d$ | | `EEV` | $d + K\,d(d-1)/2$ |
| `tied_spherical` / `EII` | $1$ | | `VVE` | $Kd + d(d-1)/2$ |
| | | | `VEV` | $K + (d-1) + K\,d(d-1)/2$ |
| | | | `EVV` | $1 + K(d-1) + K\,d(d-1)/2$ |

(the GPCM codes are explained in [GPCM Covariance
Models](gpcm-covariance-models.md)). Student-t adds $K$ (or $1$, or $0$)
degrees-of-freedom parameters, and vMF has $(K-1) + K(d-1) + K$ (or $+1$ for
a shared $\kappa$) -- see [Other Mixture Families](other-mixture-families.md).
In Gibbs mode, $K$ is the number of *active* components.

### ICL (Integrated Completed Likelihood)

`icl(X, mode='hard')` adds an entropy penalty to `bic` for ambiguous,
overlapping-cluster assignments (Biernacki, Celeux & Govaert, 2000). With
responsibilities $r_{ik}$ and $\hat z_i = \arg\max_k r_{ik}$,

$$
\mathrm{ICL}_{\text{hard}} = \mathrm{BIC} - 2\sum_i \ln r_{i\hat z_i}, \qquad
\mathrm{ICL}_{\text{soft}} = \mathrm{BIC} - 2\sum_{i,k} r_{ik}\ln r_{ik}.
$$

Both penalties are $\ge 0$, so lower is better, same convention as `bic`;
$\mathrm{ICL} \ge \mathrm{BIC}$ always, and the two are identical when
responsibilities are one-hot (perfectly separated clusters). $-\sum
r\ln r$ is the entropy of the soft assignment: ICL is BIC applied to the
*completed* (data + cluster labels) likelihood, so it penalizes solutions
whose components overlap.

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

`covariance_types` defaults to **all 14 GPCM codes** (the mclust-style
workflow); pass an explicit subset, as above, to narrow the search.
`result.table` has one row per `(n_components, covariance_type)` combination
(`bic`, `aic`, `icl`, `log_likelihood`, `converged`); a combination whose fit
raises (e.g. a singular covariance at a large `K`) gets `None` scores
instead of aborting the whole sweep. `result.models_[(K, covariance_type)]`
gives the fitted model for any successful combination, and `best_` is the
one minimizing `criterion`. Any other keyword (`max_iter`, `reg_covar`, ...)
is forwarded to every `GaussianMixture`. See the
[API reference](../api/model-selection.md) for the full signature.

## Complete API Reference

For full details on all metrics, see the [API Reference](../api/clustering-metrics.md).
