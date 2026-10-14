# Phase 1: clustering metrics and model selection — Design

Part of the [tgmm v0.4 roadmap](../../../tgmm%20v0.4%20roadmap%20coding-agent%20plan.md).
Builds on [[Phase 0]](2026-10-14-phase0-sufficient-stats-design.md) (`icl()`
reuses the active-components machinery `bic`/`aic` already have).

## Scope recap

1. `GaussianMixture.icl(X, mode='hard'|'soft')`.
2. `tgmm.model_selection.select_model(...)` + `ModelSelectionResult`.
3. Three new internal (no-ground-truth) indexes: `wb_index`,
   `wemmert_gancarski_index`, `dbcv`.
4. Shared `metric`/`noise_label`/optional-`n_components`/chunking
   infrastructure for the internal indexes.
5. External (ground-truth) indexes: fix two pre-existing precision bugs,
   add `beta` to `v_measure_score`, add `normalized_information_distance`.

## Pre-existing state that changes the plan slightly

`tgmm/metrics.py` already has `homogeneity_score`, `completeness_score`,
`v_measure_score`, and `fowlkes_mallows_score` — the plan's 1.5 reads as if
these are new, but they predate this phase. There is also **no existing
test file for `ClusteringMetrics` at all** (one stray call in
`test_dtype.py` is the only coverage), so none of this was previously
checked against sklearn.

Checked empirically against sklearn before writing anything:

| Function | sklearn agreement | Verdict |
| --- | --- | --- |
| `fowlkes_mallows_score` | ~1e-7 (float32 rounding only) | keep as-is |
| `normalized_mutual_info_score` | ~1e-7 | keep as-is (not in Phase 1's list) |
| `adjusted_rand_score` | ~1e-7 | keep as-is |
| `homogeneity_score` | ~7e-4 relative | **real bug, must fix** |
| `completeness_score` | ~2e-3 relative | **real bug, must fix** |
| `v_measure_score` | ~1e-5 (errors partly cancel in the harmonic mean) | fix via fixing the two above |
| `adjusted_mutual_info_score` | ~2.6% relative | known-approximate (own docstring says so); **not in Phase 1's list, left alone** |

The `homogeneity_score`/`completeness_score` bug is `+1e-10` smoothing
added asymmetrically inside the entropy logs (to dodge `log(0)`), which
doesn't cancel between the numerator and denominator entropy terms. Fixed
by switching to `torch.special.xlogy(p, p)` (defined as `0` at `p=0`,
exactly, no epsilon), which is also what makes `normalized_information_distance`
exact.

## `GaussianMixture.icl(X, mode='hard')`

```
ICL_hard = BIC - 2 * sum_i log(tau_hat[i, z_hat[i]]),   z_hat[i] = argmax_k tau_hat[i, k]
ICL_soft = BIC - 2 * sum_i sum_k tau_hat[i, k] * log(tau_hat[i, k])
```

Both reduce to exactly `BIC` when responsibilities are one-hot (`log(1) =
0` in `hard`; `1*log(1) + 0*log(0) = 0` in `soft`, using the `x log x -> 0`
convention) — matches the acceptance criterion directly, and confirms
"soft...replaces the sum" means literal substitution with the same `-2`
prefactor, not a separately-signed correction term.

Implementation: refactored `_active_score`/`_score_for_information_criterion`
(the Gibbs-mode-pruned-components machinery `bic`/`aic` already use) to
share a new `_active_e_step(X)` helper (same active-components
weight/mean/covariance swap, now returning `(resp, log_prob_norm)` instead
of just the log-likelihood), plus `_is_pruned_gibbs_fit()` for the shared
"should I use the active-only mixture" condition. `icl()` calls `self.bic(X)`
for the BIC term and a new `_resp_for_information_criterion(X)` (mirrors
`_score_for_information_criterion` but returns `resp`) for the penalty
term, so Gibbs-mode pruned fits are scored consistently throughout.
`tau_hat` (the max responsibility per row) is always `>= 1/K > 0` by
construction, so `hard` mode needs no zero-guard; `soft` mode uses
`torch.special.xlogy` for the same reason as the external-metric fix above.

No mclust fixture test yet — R isn't installed here (see
[[v0_4_mclust_oracle_gap]] in session memory). `tests/fixtures/make_icl_fixture.R`
is committed as reproducibility documentation (unrun); the fixture-parity
test is `pytest.mark.skipif(not fixture_exists)`.

## `tgmm/model_selection.py`

```python
@dataclass
class ModelSelectionResult:
    table: list[dict]          # n_components, covariance_type, bic, aic, icl, log_likelihood, converged
    best_: GaussianMixture
    models_: dict[tuple[int, str], GaussianMixture]

    def to_dataframe(self): ...  # lazy pandas import
```

`select_model(X, n_components=range(1, 11), covariance_types=(...),
criterion='bic', n_init=1, random_state=None, **gmm_kwargs)`: grid over
`(K, type)`, fit a `GaussianMixture(n_components=K, covariance_type=type,
n_init=n_init, random_state=random_state, **gmm_kwargs)` for each, record
`bic(X)`/`aic(X)`/`icl(X)`/`lower_bound_`/`converged_`. A fit that raises
(singular covariance, etc.) gets a table row with every score `None` and
is excluded from the `criterion`-minimising search, not propagated.
`criterion` must be one of `'bic' | 'aic' | 'icl'`. Phase 1's default
`covariance_types` is the 6 existing names; the docstring says Phase 3
will change the default to the 14 GPCM codes (not implemented yet).

## Internal-index infrastructure (`tgmm/metrics.py`)

### Distance computation

Two static helpers used by every internal index:

- `_pairwise_distances(X, Y, metric='euclidean', VI=None)`: `(n_x, n_y)`
  distances. `'euclidean'`/`'manhattan'` via `torch.cdist(..., p=2|1)`.
  `'mahalanobis'` via a whitening transform: `L = cholesky(VI)`,
  `cdist(X @ L, Y @ L, p=2)` — exact, since `(x-y)^T VI (x-y) = ||L^T(x-y)||^2
  = ||x@L - y@L||^2` for `VI = L L^T`. `'cosine'` via `1 - cosine_similarity`
  directly (*not* routed through the mahalanobis-style trick: Euclidean
  distance between L2-normalised vectors is `sqrt(2 * cosine_distance)`, a
  monotonic but non-linear rescaling that would silently corrupt any index
  that takes *ratios* of distances, e.g. Davies-Bouldin/WB). `'precomputed'`
  is not handled here (see below).
- `_pairwise_chunked(X, metric, chunk_size=4096, VI=None)`: generator
  yielding `(chunk, n)` blocks via `_pairwise_distances(X[chunk], X, ...)`.
  `metric='precomputed'` means `X` already *is* the `(n, n)` matrix; chunks
  are row-slices of it directly, no computation.

### `'precomputed'` is only supported by the purely-pairwise indexes

`silhouette_score`, `dunn_index`, and `dbcv` never need a point *in feature
space* — only pairwise distances — so `'precomputed'` works for all three.
`davies_bouldin_index`, `calinski_harabasz_score`, `wb_index`, and
`wemmert_gancarski_index` are all centroid-based (need to average points in
coordinate space to get a cluster centroid, and compute point-to-centroid
distances in that same space); `'precomputed'` has no coordinate space to
average over. These four raise `ValueError` up front naming the specific
index if `metric='precomputed'` is passed, rather than silently doing
something meaningless. This is a judgment call the plan's own text doesn't
resolve explicitly (it describes `metric`/`precomputed` as shared by "all
internal indexes" without flagging this tension) but is necessary for
correctness.

Centroid-based indexes get their per-metric centroid handling for free from
`_pairwise_distances(points, centroids, metric, VI)`: `'euclidean'`/`'manhattan'`
centroid = coordinate-wise mean; `'mahalanobis'` centroid = mean in the
whitened space (equivalent to whitening the mean, since the transform is
linear); `'cosine'` centroid = the (unnormalised) mean vector, with distance
computed as `1 - cosine_similarity(point, mean)` -- the conventional
"distance to the cluster's mean direction."

### Label resolution (`noise_label`, optional `n_components`)

`_resolve_labels(labels, n_components=None, noise_label=None) ->
(labels_for_grouping, K, keep_mask)`:

- `noise_label` given: `keep_mask = labels != noise_label`; everything
  downstream operates on `labels[keep_mask]`/`X[keep_mask]`.
- `n_components` given (the historical required argument): **used as-is,
  for full backward compatibility** — existing callers that already pass
  contiguous `0..n_components-1` labels get identical behaviour, no
  remapping. This is why every call site in existing code (and in
  `evaluate_clustering`) keeps working unchanged.
- `n_components=None` (new): infer `K` from `torch.unique`, and
  dense-remap labels via `torch.searchsorted(uniq, labels)` so input labels
  need not be contiguous or start at 0.

### Noise-weighting

`_apply_noise_scaling(value, labels_before_filtering, noise_label,
maximize)`: multiplies (maximise indexes) or divides (minimise indexes) by
`f = (n - n_noise) / n`, per the plan. Applied by every internal index
*except* `dbcv`, whose own `(n_i / n) V(C_i)` weighting (`n` counting noise
points in the denominator) already accounts for noise directly.

### Rewritten indexes

- **`silhouette_score`**: `D_chunk @ one_hot(labels, K)` accumulated into
  an `(n, K)` per-cluster-distance-sum matrix; `a_i` from the own-cluster
  column (`/ (n_own - 1)`), `b_i` from the min over other columns (`/
  n_other`). Singleton clusters (`n_own <= 1`) get silhouette `0` by
  convention (**sklearn's own rule** — the pre-refactor code left these as
  `0/inf -> NaN`, silently poisoning the mean; fixed as part of this
  rewrite since the function is being rebuilt for chunking anyway and a
  NaN obviously fails any sklearn-parity assertion).
- **`dunn_index`**: running `max_intra`/`min_inter` reduced per chunk
  (same-cluster vs different-cluster boolean masks against the chunk's
  row labels).
- **`davies_bouldin_index`**: per-cluster mean distance-to-own-centroid
  vector `S` computed once; `(S[:, None] + S[None, :]) / M` (M = centroid
  pairwise distances) with the diagonal masked to `-inf` before the
  per-row max, replacing the double Python loop.
- **`calinski_harabasz_score`**: unchanged math, vectorised to use the
  shared centroid/distance helpers and the new label resolution.

### New internal indexes

- **`wb_index(X, labels, ...)`**: `K * SSW / SSB`, reusing the exact
  SSW/SSB computation `calinski_harabasz_score` already has (same helper,
  shared rather than re-derived) — min is better.
- **`wemmert_gancarski_index`**: `R(x) = d(x, c_own) / min_{j != own}
  d(x, c_j)`; `J_k = max(0, 1 - mean_{x in C_k} R(x))`; `WG = sum_k n_k J_k
  / n` — max is better, computed via the centroid-distance helper (an
  `(n, K)` distance-to-every-centroid matrix, own-column vs.
  min-of-other-columns, exactly as in `davies_bouldin_index`'s pattern).
- **`dbcv(X, labels, metric='euclidean', noise_label=-1)`** (Moulavi et
  al. 2014): for each cluster, per the "all-points core distance" (not the
  k-NN core distance HDBSCAN itself uses):

  ```
  a_coredist(x) = ( (1 / (|C| - 1)) * sum_{y in C, y != x} d(x, y)^(-d_features) )^(-1 / d_features)
  ```

  mutual-reachability within the cluster `d_mreach(x, y) = max(a_coredist(x),
  a_coredist(y), d(x, y))`, then a **per-cluster** MST over that
  mutual-reachability graph (reusing `HDBSCAN._build_linkage_tree`'s Prim
  loop, factored out as a small standalone `_prim_mst(distance_matrix) ->
  list[(u, v, weight)]` so `tgmm/hdbscan.py` doesn't need to change: the
  existing method is now one line that calls it and keeps its linkage-tree
  conversion). Density Sparseness of a Cluster (`DSC_i`) = max MST edge
  weight incident to an *internal* node (MST degree >= 2); Density
  Separation between Clusters (`DSPC_ij`) = min mutual-reachability
  distance between an internal node of cluster `i`'s MST and an internal
  node of cluster `j`'s MST (computed directly, not from either MST, since
  it's a cross-cluster quantity). If a cluster's MST has no internal nodes
  (every node is a leaf — happens for 2-3-point clusters), falls back to
  using all nodes, the standard practical fallback (mirrors the `hdbscan`
  package's own `validity_index`). `V(C_i) = (min_j DSPC_ij - DSC_i) /
  max(min_j DSPC_ij, DSC_i)`; `DBCV = sum_i (n_i / n) V(C_i)`, `n` counting
  noise points in the denominator (not the numerator) — this *is* the
  noise weighting, which is why `dbcv` is exempt from
  `_apply_noise_scaling`.

  Tested against an independent NumPy reference implementation in
  `tests/test_metrics_dbcv.py` (the plan's own instruction for anything
  sklearn doesn't have) — written separately from the production
  PyTorch version, from the same algorithmic description above. One
  deliberate exception: the reference **shares** `prim_mst` with the
  production code rather than using, say,
  `scipy.sparse.csgraph.minimum_spanning_tree`. Verified directly while
  writing this test: mutual-reachability distance is `max(core_dist(x),
  core_dist(y), d(x, y))`, so whenever one point's core distance
  dominates several of its edges, those edges tie exactly — routine, not
  rare — and a tied-weight graph can have multiple equally-minimal
  spanning trees with *different* topologies (different internal/leaf
  node sets). scipy's MST and `prim_mst` gave the same total tree weight
  on an identical graph (56.150451...) but different internal-node counts
  (17 vs. 11), which propagated into a real DBCV value mismatch between
  two independently-correct implementations — not a bug in either.
  Sharing the MST step keeps the test focused on what's actually novel in
  DBCV (core distance, DSC/DSPC/V, noise weighting) instead of failing on
  this tie-breaking non-issue.

`dunn_index`'s docstring gets the one-paragraph note the plan asks for
(ranked 25/26 in the Simpson/Campello/Stojanovski 2025 benchmark, kept for
completeness).

## External indexes (`tgmm/metrics.py`)

- `homogeneity_score`/`completeness_score`: rewritten using
  `torch.special.xlogy(p, p)` for every entropy term (`H(C)`, `H(K)`,
  `H(C|K)`, `H(K|C)`), `float64` internally regardless of input dtype (the
  existing contingency-matrix pattern hardcodes `torch.float`/float32),
  and sklearn's own zero-entropy convention (`H(C) == 0 -> homogeneity =
  1.0`, symmetric for completeness) instead of dividing by a
  near-zero-but-not-zero smoothed denominator.
- `v_measure_score(labels_true, labels_pred, beta=1.0)`: now
  `(1 + beta) * h * c / (beta * h + c)` (`beta=1.0` recovers the existing
  harmonic-mean formula exactly).
- `normalized_information_distance(labels_true, labels_pred)`: `1 -
  I(U;V) / max(H(U), H(V))` (Vinh, Epps & Bailey 2010), `I(U;V)` reusing
  the existing `mutual_info_score`, entropies via the same `xlogy` helper.
  Tested against a NumPy reference in the test file (no sklearn
  equivalent).
- `fowlkes_mallows_score`, `adjusted_rand_score`,
  `normalized_mutual_info_score`: left untouched (already within 1e-6 of
  sklearn once compared at matching dtype in the test; see table above).
- `adjusted_mutual_info_score`: left untouched. It is not in Phase 1's
  index list, its own docstring already discloses it is a rough
  approximation, and a real fix needs the expected-MI-under-the-
  generalised-hypergeometric-model combinatorial sum -- out of scope here;
  left as a flagged pre-existing gap, not silently papered over.

## Non-goals for this phase

- No fix to `adjusted_mutual_info_score`'s approximation (see above).
- No change to how `rand_score`/`adjusted_rand_score`/`mutual_info_score`/
  `purity_score`/`confusion_matrix`/`classification_report` build their
  contingency matrices (still a Python loop over samples, still assume
  contiguous 0-indexed labels) — out of this phase's named scope; flagged,
  not silently fixed, to avoid uncontrolled blast radius in files this
  phase isn't chartered to touch.
- `select_model`'s default `covariance_types` stays the 6 existing names;
  Phase 3 changes the default to the 14 GPCM codes.
- No mclust fixture for `icl()` (see [[v0_4_mclust_oracle_gap]]).
