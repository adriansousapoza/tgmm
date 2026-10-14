import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union

from .hdbscan import prim_mst


class ClusteringMetrics:
    r"""
    A collection of clustering and classification metrics, both unsupervised and supervised.

    This class provides methods such as:
    
    - **KL divergence** for comparing two GMMs (via Monte Carlo).
    - **Unsupervised metrics** (silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn index).
    - **Supervised metrics** (Rand index, ARI, mutual info variants, purity, classification report).

    All methods are static for convenience, and most accept data in PyTorch Tensors
    (potentially on GPU). For supervised metrics, the user must provide `labels_true`
    and `labels_pred` as integer-encoded 1D tensors of the same shape.

    BIC/AIC are not here: they only need a model's own log-likelihood and
    parameter count, so they live as `bic`/`aic` methods on `GaussianMixture`
    directly (e.g. ``gmm.bic(X)``), covering both EM and Gibbs-sampling modes
    (selected via `n_components` as an integer or ``None``), rather than as
    free functions that take a model apart.

    Example
    -------
    .. code-block:: python

        from myproject.clustering_metrics import ClusteringMetrics

        # Suppose gmm is a fitted GaussianMixture, gmm2 is another GMM
        # Compare them via KL divergence:
        kl_pq = ClusteringMetrics.kl_divergence_gmm(gmm, gmm2)
        print("KL(p||q) =", kl_pq)

        # Evaluate clustering performance w.r.t. true labels:
        pred_labels = gmm.predict(X_tensor)
        ari = ClusteringMetrics.adjusted_rand_score(labels_true, pred_labels)
        print("ARI =", ari)
    """

    # ========================================================================
    # Shared internal-index infrastructure
    #
    # See docs/superpowers/specs/2026-10-14-phase1-metrics-model-selection-design.md
    # for the design rationale, in particular why 'precomputed' is only
    # supported by the purely-pairwise indexes (silhouette_score, dunn_index,
    # dbcv) and not the centroid-based ones (davies_bouldin_index,
    # calinski_harabasz_score, wb_index, wemmert_gancarski_index).
    # ========================================================================

    @staticmethod
    def _pairwise_distances(
        X: torch.Tensor, Y: torch.Tensor, metric: str = 'euclidean',
        VI: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        `(n_x, n_y)` distances between rows of `X` and rows of `Y`.

        `metric` is one of `'euclidean'`, `'manhattan'`, `'mahalanobis'`
        (requires `VI`, a `(d, d)` precision matrix), or `'cosine'`.
        `'precomputed'` is not handled here -- callers that accept it treat
        the input as already *being* the distance matrix rather than
        points to measure between.

        `'mahalanobis'` is computed by whitening both `X` and `Y`
        (`L = cholesky(VI)`, `X @ L`, `Y @ L`) and taking the Euclidean
        distance in that space: for `VI = L L^T`,
        `(x-y)^T VI (x-y) = ||L^T(x-y)||^2 = ||x@L - y@L||^2`, so this is
        exact, not an approximation.

        `'cosine'` is computed directly as `1 - cosine_similarity`, *not*
        via the same whitening trick -- Euclidean distance between
        L2-normalised vectors is `sqrt(2 * cosine_distance)`, a monotonic
        but non-linear rescaling that would corrupt any index taking a
        *ratio* of distances (e.g. Davies-Bouldin, WB).
        """
        if metric == 'euclidean':
            return torch.cdist(X, Y, p=2)
        elif metric == 'manhattan':
            return torch.cdist(X, Y, p=1)
        elif metric == 'mahalanobis':
            if VI is None:
                raise ValueError("metric='mahalanobis' requires VI (a (d, d) precision matrix).")
            L = torch.linalg.cholesky(VI.to(dtype=X.dtype, device=X.device))
            return torch.cdist(X @ L, Y @ L, p=2)
        elif metric == 'cosine':
            Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
            Yn = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-12)
            return 1.0 - Xn @ Yn.t()
        else:
            raise ValueError(
                f"Unsupported metric: {metric!r}. Expected one of 'euclidean', "
                "'manhattan', 'mahalanobis', 'cosine' (or 'precomputed', where "
                "applicable -- see the calling method's docstring)."
            )

    @classmethod
    def _require_coordinate_metric(cls, metric: str, index_name: str):
        r"""
        Raise a clear error if `metric='precomputed'` is passed to a
        centroid-based index -- there is no coordinate space to average
        points in when all that's given is a distance matrix.
        """
        if metric == 'precomputed':
            raise ValueError(
                f"metric='precomputed' is not supported by {index_name} (it needs "
                "to compute a cluster centroid in coordinate space, which a "
                "precomputed distance matrix doesn't provide). Use "
                "silhouette_score, dunn_index, or dbcv for a precomputed-distance "
                "workflow instead."
            )

    @staticmethod
    def _pairwise_chunked(
        X: torch.Tensor, metric: str = 'euclidean', chunk_size: int = 4096,
        VI: Optional[torch.Tensor] = None,
    ):
        r"""
        Yield `(chunk, n)` distance blocks, row-chunked over `X`.

        When `metric='precomputed'`, `X` is itself the `(n, n)` distance
        matrix, and chunks are row-slices of it directly (no distance
        computation). Otherwise each chunk is
        `_pairwise_distances(X[start:end], X, metric, VI)`.
        """
        n = X.shape[0]
        if metric == 'precomputed':
            if X.shape[0] != X.shape[1]:
                raise ValueError("metric='precomputed' requires a square (n, n) distance matrix.")
            for start in range(0, n, chunk_size):
                yield X[start:start + chunk_size]
            return

        for start in range(0, n, chunk_size):
            yield ClusteringMetrics._pairwise_distances(X[start:start + chunk_size], X, metric, VI)

    @staticmethod
    def _resolve_labels(
        labels: torch.Tensor, n_components: Optional[int] = None,
        noise_label: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        r"""
        Resolve `labels` into `(dense_labels, n_clusters, keep_mask)`.

        `noise_label`, if given, marks rows to exclude (`keep_mask =
        labels != noise_label`); `dense_labels`/`n_clusters` only describe
        the kept rows.

        `n_components`, if given, is trusted as-is for full backward
        compatibility: every pre-Phase-1 call site already passes
        contiguous `0..n_components-1` labels, so no remapping happens in
        that case. `n_components=None` (new) infers the cluster count from
        `torch.unique` and dense-remaps via `torch.searchsorted`, so labels
        need not be contiguous or start at 0.
        """
        labels = labels.to(torch.long)
        if noise_label is not None:
            keep_mask = labels != noise_label
        else:
            keep_mask = torch.ones_like(labels, dtype=torch.bool)

        kept = labels[keep_mask]
        if n_components is not None:
            return kept, int(n_components), keep_mask

        uniq = torch.unique(kept, sorted=True)
        dense = torch.searchsorted(uniq, kept)
        return dense, int(uniq.numel()), keep_mask

    @staticmethod
    def _apply_noise_scaling(
        value: float, original_labels: torch.Tensor, noise_label: Optional[int],
        maximize: bool,
    ) -> float:
        r"""
        Scale `value` by `f = (n - n_noise) / n` when `noise_label` is set
        -- multiply for maximise-is-better indexes, divide for
        minimise-is-better ones. A no-op when `noise_label` is `None`.
        """
        if noise_label is None:
            return value
        n = original_labels.shape[0]
        n_noise = int((original_labels == noise_label).sum().item())
        if n_noise == 0:
            return value
        f = (n - n_noise) / n
        return value * f if maximize else value / f

    @staticmethod
    def _cluster_centroids(
        X: torch.Tensor, labels: torch.Tensor, n_clusters: int,
    ) -> torch.Tensor:
        r"""Per-cluster coordinate-wise mean, shape `(n_clusters, n_features)`."""
        return torch.stack([X[labels == k].mean(dim=0) for k in range(n_clusters)])

    @staticmethod
    def _ssw_ssb(
        X: torch.Tensor, labels: torch.Tensor, n_clusters: int,
        metric: str = 'euclidean', VI: Optional[torch.Tensor] = None,
    ):
        r"""
        Within-cluster (SSW) and between-cluster (SSB) sum of squared
        distances-to-centroid, shared by `calinski_harabasz_score` and
        `wb_index`. Also returns `centroids` and the full `(n, K)`
        distance-to-every-centroid matrix, since `wemmert_gancarski_index`
        needs the latter too and it's already computed here.

        Returns
        -------
        ssw, ssb : torch.Tensor (scalar)
        centroids : torch.Tensor, shape (n_clusters, n_features)
        D : torch.Tensor, shape (n_samples, n_clusters)
            Distance from each point to every centroid (own-cluster column
            included).
        """
        centroids = ClusteringMetrics._cluster_centroids(X, labels, n_clusters)
        overall = X.mean(dim=0, keepdim=True)

        D = ClusteringMetrics._pairwise_distances(X, centroids, metric, VI)  # (n, K)
        idx = torch.arange(X.shape[0], device=X.device)
        own_dist = D[idx, labels]
        ssw = own_dist.pow(2).sum()

        D_overall = ClusteringMetrics._pairwise_distances(centroids, overall, metric, VI).squeeze(1)  # (K,)
        counts = torch.bincount(labels, minlength=n_clusters).to(X.dtype)
        ssb = (counts * D_overall.pow(2)).sum()

        return ssw, ssb, centroids, D

    @staticmethod
    def silhouette_score(
        X: torch.Tensor,
        labels: torch.Tensor,
        n_components: Optional[int] = None,
        metric: str = 'euclidean',
        noise_label: Optional[int] = None,
        chunk_size: int = 4096,
        VI: Optional[torch.Tensor] = None,
    ) -> float:
        r"""
        Compute the silhouette score for a partition of the data.

        $$ \mathrm{silhouette}(i) = \frac{b_i - a_i}{\max(a_i, b_i)} $$

        Where:
        - $a_i$ is the mean distance to points in the same cluster.
        - $b_i$ is the minimum mean distance to points in a different cluster.

        Points in a singleton cluster (the only member of their cluster)
        get silhouette `0` by convention (sklearn's own rule -- $a_i$ is
        undefined with no other same-cluster point to average over).

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features), or an (n, n) precomputed
            distance matrix when `metric='precomputed'`.
        labels : torch.Tensor
            Cluster labels, shape (n_samples,). Need not be contiguous or
            0-indexed when `n_components` is left `None`.
        n_components : int, optional
            Number of clusters. If `None` (default), inferred from the
            distinct values in `labels`.
        metric : str, optional
            One of 'euclidean', 'manhattan', 'cosine', 'mahalanobis'
            (requires `VI`), or 'precomputed'. (default: 'euclidean')
        noise_label : int, optional
            Label value marking noise points (e.g. -1 for HDBSCAN output).
            Noise rows are dropped before computing the index, and the
            result is scaled by `(n - n_noise) / n`. (default: None)
        chunk_size : int, optional
            Row-chunk size for the pairwise-distance computation, bounding
            peak memory to `O(chunk_size * n_samples)` regardless of
            `n_samples`. (default: 4096)
        VI : torch.Tensor, optional
            `(d, d)` precision matrix for `metric='mahalanobis'`. Passing
            the pooled precision of a fitted `GaussianMixture`
            (`torch.linalg.inv` of the weighted average of `covariances_`)
            gives the GMM-matched silhouette.

        Returns
        -------
        float
            The mean silhouette score over all samples (range is typically [-1, 1],
            though it’s seldom negative in practice if distances are Euclidean).
        """
        original_labels = labels
        dense_labels, n_clusters, keep_mask = ClusteringMetrics._resolve_labels(
            labels, n_components, noise_label
        )
        assert n_clusters > 1, "Silhouette score is only defined when there are at least 2 clusters."

        X_eff = X[keep_mask][:, keep_mask] if metric == 'precomputed' else X[keep_mask]
        dense_labels = dense_labels.to(X_eff.device)
        n = X_eff.shape[0]

        one_hot = F.one_hot(dense_labels, n_clusters).to(X_eff.dtype)  # (n, K)
        counts = one_hot.sum(dim=0)  # (K,) = n_k

        sums = torch.zeros(n, n_clusters, dtype=X_eff.dtype, device=X_eff.device)
        start = 0
        for D_chunk in ClusteringMetrics._pairwise_chunked(X_eff, metric, chunk_size, VI):
            end = start + D_chunk.shape[0]
            sums[start:end] = D_chunk @ one_hot
            start = end

        idx = torch.arange(n, device=X_eff.device)
        own_count = counts[dense_labels]
        own_sum = sums[idx, dense_labels]
        a = own_sum / (own_count - 1).clamp_min(1)

        mean_dists = sums / counts.unsqueeze(0)
        mean_dists[idx, dense_labels] = float('inf')
        b = mean_dists.min(dim=1).values

        silhouette = (b - a) / torch.max(a, b)
        silhouette = torch.where(own_count <= 1, torch.zeros_like(silhouette), silhouette)

        value = silhouette.mean().item()
        return ClusteringMetrics._apply_noise_scaling(value, original_labels, noise_label, maximize=True)

    @staticmethod
    def davies_bouldin_index(
        X: torch.Tensor, labels: torch.Tensor, n_components: Optional[int] = None,
        metric: str = 'euclidean', noise_label: Optional[int] = None,
        VI: Optional[torch.Tensor] = None,
    ) -> float:
        r"""
        Compute the Davies-Bouldin index (lower is better).

        $$ \text{DB} = \frac{1}{k} \sum_i \max_{j \neq i} \frac{S_i + S_j}{M_{ij}} $$

        Where:
        - $S_i$ is the average distance of points in cluster i to its centroid.
        - $M_{ij}$ is the distance between cluster centroids i and j.

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features). `metric='precomputed'` is
            not supported (no coordinate space to average a centroid in).
        labels : torch.Tensor
            Cluster labels, shape (n_samples,). Need not be contiguous or
            0-indexed when `n_components` is left `None`.
        n_components : int, optional
            Number of clusters (must be >= 2). If `None` (default),
            inferred from the distinct values in `labels`.
        metric : str, optional
            One of 'euclidean', 'manhattan', 'cosine', 'mahalanobis'
            (requires `VI`). (default: 'euclidean')
        noise_label : int, optional
            Label value marking noise points; dropped before computing the
            index, result scaled by `(n - n_noise) / n`. (default: None)
        VI : torch.Tensor, optional
            `(d, d)` precision matrix for `metric='mahalanobis'`.

        Returns
        -------
        float
            Davies-Bouldin index.
        """
        ClusteringMetrics._require_coordinate_metric(metric, 'davies_bouldin_index')
        original_labels = labels
        dense_labels, n_clusters, keep_mask = ClusteringMetrics._resolve_labels(
            labels, n_components, noise_label
        )
        assert n_clusters > 1, "Davies-Bouldin index is only defined when >= 2 clusters."

        X_eff = X[keep_mask]
        dense_labels = dense_labels.to(X_eff.device)
        counts = torch.bincount(dense_labels, minlength=n_clusters)

        centroids = ClusteringMetrics._cluster_centroids(X_eff, dense_labels, n_clusters)
        D_own = ClusteringMetrics._pairwise_distances(X_eff, centroids, metric, VI)  # (n, K)
        idx = torch.arange(X_eff.shape[0], device=X_eff.device)
        own_dist = D_own[idx, dense_labels]

        S = torch.zeros(n_clusters, dtype=X_eff.dtype, device=X_eff.device)
        S.scatter_add_(0, dense_labels, own_dist)
        S = S / counts.clamp_min(1).to(S.dtype)

        M = ClusteringMetrics._pairwise_distances(centroids, centroids, metric, VI)  # (K, K)
        ratio = (S.unsqueeze(1) + S.unsqueeze(0)) / M
        ratio.fill_diagonal_(float('-inf'))

        value = ratio.max(dim=1).values.mean().item()
        return ClusteringMetrics._apply_noise_scaling(value, original_labels, noise_label, maximize=False)

    @staticmethod
    def calinski_harabasz_score(
        X: torch.Tensor, labels: torch.Tensor, n_components: Optional[int] = None,
        metric: str = 'euclidean', noise_label: Optional[int] = None,
        VI: Optional[torch.Tensor] = None,
    ) -> float:
        r"""
        Compute the Calinski-Harabasz index (a ratio of between-cluster dispersion
        to within-cluster dispersion).

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features). `metric='precomputed'` is
            not supported (no coordinate space to average a centroid in).
        labels : torch.Tensor
            Cluster labels. Need not be contiguous or 0-indexed when
            `n_components` is left `None`.
        n_components : int, optional
            Number of clusters. If `None` (default), inferred from the
            distinct values in `labels`.
        metric : str, optional
            One of 'euclidean', 'manhattan', 'cosine', 'mahalanobis'
            (requires `VI`). (default: 'euclidean')
        noise_label : int, optional
            Label value marking noise points; dropped before computing the
            index, result scaled by `(n - n_noise) / n`. (default: None)
        VI : torch.Tensor, optional
            `(d, d)` precision matrix for `metric='mahalanobis'`.

        Returns
        -------
        float
            Calinski-Harabasz index (higher is better).
        """
        ClusteringMetrics._require_coordinate_metric(metric, 'calinski_harabasz_score')
        original_labels = labels
        dense_labels, n_clusters, keep_mask = ClusteringMetrics._resolve_labels(
            labels, n_components, noise_label
        )
        X_eff = X[keep_mask]
        dense_labels = dense_labels.to(X_eff.device)
        n_samples = X_eff.shape[0]

        ssw, ssb, _, _ = ClusteringMetrics._ssw_ssb(X_eff, dense_labels, n_clusters, metric, VI)
        CH = (ssb / (n_clusters - 1)) / (ssw / (n_samples - n_clusters))

        value = CH.item()
        return ClusteringMetrics._apply_noise_scaling(value, original_labels, noise_label, maximize=True)

    @staticmethod
    def wb_index(
        X: torch.Tensor, labels: torch.Tensor, n_components: Optional[int] = None,
        metric: str = 'euclidean', noise_label: Optional[int] = None,
        VI: Optional[torch.Tensor] = None,
    ) -> float:
        r"""
        WB-index (Zhao & Fr\"anti, 2014): lower is better.

        $$ \text{WB} = K \cdot \frac{\text{SSW}}{\text{SSB}} $$

        with SSW/SSB defined exactly as in `calinski_harabasz_score` (same
        shared helper -- ranked among the best-performing internal indexes
        in the Simpson, Campello & Stojanovski (2025) benchmark).

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features). `metric='precomputed'` is
            not supported (no coordinate space to average a centroid in).
        labels : torch.Tensor
            Cluster labels. Need not be contiguous or 0-indexed when
            `n_components` is left `None`.
        n_components : int, optional
            Number of clusters. If `None` (default), inferred from the
            distinct values in `labels`.
        metric : str, optional
            One of 'euclidean', 'manhattan', 'cosine', 'mahalanobis'
            (requires `VI`). (default: 'euclidean')
        noise_label : int, optional
            Label value marking noise points; dropped before computing the
            index, result scaled by `(n - n_noise) / n`. (default: None)
        VI : torch.Tensor, optional
            `(d, d)` precision matrix for `metric='mahalanobis'`.

        Returns
        -------
        float
            WB-index (lower is better).
        """
        ClusteringMetrics._require_coordinate_metric(metric, 'wb_index')
        original_labels = labels
        dense_labels, n_clusters, keep_mask = ClusteringMetrics._resolve_labels(
            labels, n_components, noise_label
        )
        X_eff = X[keep_mask]
        dense_labels = dense_labels.to(X_eff.device)

        ssw, ssb, _, _ = ClusteringMetrics._ssw_ssb(X_eff, dense_labels, n_clusters, metric, VI)
        value = (n_clusters * ssw / ssb).item()
        return ClusteringMetrics._apply_noise_scaling(value, original_labels, noise_label, maximize=False)

    @staticmethod
    def wemmert_gancarski_index(
        X: torch.Tensor, labels: torch.Tensor, n_components: Optional[int] = None,
        metric: str = 'euclidean', noise_label: Optional[int] = None,
        VI: Optional[torch.Tensor] = None,
    ) -> float:
        r"""
        Wemmert-Gancarski index: higher is better.

        $$ R(x) = \frac{d(x, c_{\text{own}})}{\min_{j \neq \text{own}} d(x, c_j)},
           \qquad J_k = \max\left(0, 1 - \overline{R(x)}_{x \in C_k}\right),
           \qquad \text{WG} = \frac{1}{n}\sum_k n_k J_k $$

        Ranked among the best-performing internal indexes in the Simpson,
        Campello & Stojanovski (2025) benchmark.

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features). `metric='precomputed'` is
            not supported (no coordinate space to average a centroid in).
        labels : torch.Tensor
            Cluster labels. Need not be contiguous or 0-indexed when
            `n_components` is left `None`.
        n_components : int, optional
            Number of clusters. If `None` (default), inferred from the
            distinct values in `labels`.
        metric : str, optional
            One of 'euclidean', 'manhattan', 'cosine', 'mahalanobis'
            (requires `VI`). (default: 'euclidean')
        noise_label : int, optional
            Label value marking noise points; dropped before computing the
            index, result scaled by `(n - n_noise) / n`. (default: None)
        VI : torch.Tensor, optional
            `(d, d)` precision matrix for `metric='mahalanobis'`.

        Returns
        -------
        float
            Wemmert-Gancarski index (higher is better).
        """
        ClusteringMetrics._require_coordinate_metric(metric, 'wemmert_gancarski_index')
        original_labels = labels
        dense_labels, n_clusters, keep_mask = ClusteringMetrics._resolve_labels(
            labels, n_components, noise_label
        )
        X_eff = X[keep_mask]
        dense_labels = dense_labels.to(X_eff.device)
        n = X_eff.shape[0]

        centroids = ClusteringMetrics._cluster_centroids(X_eff, dense_labels, n_clusters)
        D = ClusteringMetrics._pairwise_distances(X_eff, centroids, metric, VI)  # (n, K)
        idx = torch.arange(n, device=X_eff.device)
        own_dist = D[idx, dense_labels]
        D_others = D.clone()
        D_others[idx, dense_labels] = float('inf')
        min_other = D_others.min(dim=1).values

        R = own_dist / min_other.clamp_min(1e-300)
        counts = torch.bincount(dense_labels, minlength=n_clusters).to(X_eff.dtype)
        sum_R = torch.zeros(n_clusters, dtype=X_eff.dtype, device=X_eff.device)
        sum_R.scatter_add_(0, dense_labels, R)
        mean_R = sum_R / counts.clamp_min(1)
        J = torch.clamp(1.0 - mean_R, min=0.0)

        value = ((counts * J).sum() / n).item()
        return ClusteringMetrics._apply_noise_scaling(value, original_labels, noise_label, maximize=True)

    @staticmethod
    def dunn_index(
        X: torch.Tensor, labels: torch.Tensor, n_components: Optional[int] = None,
        metric: str = 'euclidean', noise_label: Optional[int] = None,
        chunk_size: int = 4096, VI: Optional[torch.Tensor] = None,
    ) -> float:
        r"""
        Compute the Dunn index:

        $$ D = \frac{\min_{i \neq j} d(C_i, C_j)}{\max_k \mathrm{diam}(C_k)} $$

        Where:
        - $d(C_i, C_j)$ is the minimum distance between any points in clusters i, j.
        - $\mathrm{diam}(C_k)$ is the maximum distance between any points in cluster k.

        Higher Dunn index indicates better cluster separation.

        .. note::
           Ranked 25th of 26 indexes on partition ranking in the Simpson,
           Campello & Stojanovski (2025) clustering-validity benchmark --
           kept here for completeness/familiarity, not as a recommended
           choice. Prefer `wb_index`, `wemmert_gancarski_index`, or `dbcv`,
           which ranked at the top of that benchmark.

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features), or an (n, n) precomputed
            distance matrix when `metric='precomputed'`.
        labels : torch.Tensor
            Cluster labels. Need not be contiguous or 0-indexed when
            `n_components` is left `None`.
        n_components : int, optional
            Number of clusters. If `None` (default), inferred from the
            distinct values in `labels`.
        metric : str, optional
            One of 'euclidean', 'manhattan', 'cosine', 'mahalanobis'
            (requires `VI`), or 'precomputed'. (default: 'euclidean')
        noise_label : int, optional
            Label value marking noise points; dropped before computing the
            index, result scaled by `(n - n_noise) / n`. (default: None)
        chunk_size : int, optional
            Row-chunk size for the pairwise-distance computation. (default: 4096)
        VI : torch.Tensor, optional
            `(d, d)` precision matrix for `metric='mahalanobis'`.

        Returns
        -------
        float
            Dunn index (higher is better).
        """
        original_labels = labels
        dense_labels, n_clusters, keep_mask = ClusteringMetrics._resolve_labels(
            labels, n_components, noise_label
        )
        X_eff = X[keep_mask][:, keep_mask] if metric == 'precomputed' else X[keep_mask]
        dense_labels = dense_labels.to(X_eff.device)

        max_intra = 0.0
        min_inter = float('inf')
        start = 0
        for D_chunk in ClusteringMetrics._pairwise_chunked(X_eff, metric, chunk_size, VI):
            end = start + D_chunk.shape[0]
            row_labels = dense_labels[start:end]
            same = row_labels.unsqueeze(1) == dense_labels.unsqueeze(0)
            diff = ~same
            if same.any():
                max_intra = max(max_intra, D_chunk[same].max().item())
            if diff.any():
                min_inter = min(min_inter, D_chunk[diff].min().item())
            start = end

        value = (min_inter / max_intra) if max_intra > 0 else 0.0
        return ClusteringMetrics._apply_noise_scaling(value, original_labels, noise_label, maximize=True)

    @staticmethod
    def dbcv(
        X: torch.Tensor, labels: torch.Tensor, metric: str = 'euclidean',
        noise_label: Optional[int] = -1, VI: Optional[torch.Tensor] = None,
    ) -> float:
        r"""
        Density-Based Clustering Validation (Moulavi, Jaskowiak, Campello,
        Zimek & Sander, 2014). Higher is better, range `[-1, 1]`.

        For each cluster :math:`C`, the **all-points core distance** of a
        point :math:`x \in C` (not the k-NN core distance HDBSCAN itself
        uses) is

        .. math::
            a_{\text{coredist}}(x) = \left( \frac{1}{|C| - 1}
                \sum_{y \in C, y \neq x} d(x, y)^{-d_{\text{features}}}
            \right)^{-1 / d_{\text{features}}}

        and the mutual-reachability distance within :math:`C` is
        :math:`d_{\text{mreach}}(x, y) = \max(a_{\text{coredist}}(x),
        a_{\text{coredist}}(y), d(x, y))`. A minimum spanning tree is built
        over each cluster's own mutual-reachability graph (reusing
        `tgmm.hdbscan.prim_mst`, the same Prim's-algorithm implementation
        `HDBSCAN` itself uses). **Density Sparseness of a Cluster** (DSC)
        is the maximum edge weight in that MST; **Density Separation
        between a Pair of Clusters** (DSPC) is the minimum
        mutual-reachability distance between an *internal* node (MST
        degree >= 2 -- excludes boundary/leaf points) of one cluster's MST
        and an internal node of the other's.

        .. note::
           For a tree with >= 3 nodes, every edge has at least one
           endpoint of degree >= 2 (an edge between two degree-1 nodes
           would disconnect the rest of the tree) -- so restricting DSC to
           "edges incident to an internal node" is equivalent to no
           restriction at all for any cluster with >= 3 points, which is
           why DSC below is simply the MST's maximum edge weight. The
           `internal`/leaf distinction is only load-bearing for DSPC,
           which compares nodes *across* two different clusters.

        .. math::
            V(C_i) = \frac{\min_j \text{DSPC}_{ij} - \text{DSC}_i}
                           {\max(\min_j \text{DSPC}_{ij}, \text{DSC}_i)},
            \qquad \text{DBCV} = \sum_i \frac{n_i}{n} V(C_i)

        with `n` counting noise points in the denominator (not the
        numerator) -- this *is* `dbcv`'s noise handling, which is why it
        does not go through `_apply_noise_scaling` the way the other
        internal indexes do.

        Clusters with fewer than 2 points contribute 0 (the all-points
        core distance needs at least one other point in the cluster to
        average over); a cluster whose MST has no internal nodes at all
        (every node is a leaf -- only possible for 2-point clusters) falls
        back to treating every node as internal for DSPC purposes, the
        standard practical fallback.

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
        labels : torch.Tensor
            Cluster labels, shape (n_samples,). Need not be contiguous.
        metric : str, optional
            One of 'euclidean', 'manhattan', 'cosine', 'mahalanobis'
            (requires `VI`). (default: 'euclidean')
        noise_label : int, optional
            Label value marking noise points (HDBSCAN's convention is
            -1). Pass `None` if there is no noise label in `labels`.
            (default: -1)
        VI : torch.Tensor, optional
            `(d, d)` precision matrix for `metric='mahalanobis'`.

        Returns
        -------
        float
            DBCV score in `[-1, 1]` (higher is better).
        """
        labels = labels.to(torch.long)
        n = labels.shape[0]
        d_features = X.shape[1]

        if noise_label is not None:
            keep_mask = labels != noise_label
        else:
            keep_mask = torch.ones_like(labels, dtype=torch.bool)
        cluster_ids = torch.unique(labels[keep_mask], sorted=True).tolist()
        if len(cluster_ids) < 1:
            return 0.0

        per_cluster = {}
        for cid in cluster_ids:
            Xc = X[labels == cid]
            nc = Xc.shape[0]
            if nc < 2:
                per_cluster[cid] = dict(Xc=Xc, nc=nc, core_dist=None, internal_mask=None, dsc=0.0)
                continue

            D_full = ClusteringMetrics._pairwise_distances(Xc, Xc, metric, VI)
            D_safe = D_full.clone()
            D_safe.fill_diagonal_(float('inf'))
            D_safe = D_safe.clamp_min(1e-12)

            mean_inv_pow = D_safe.pow(-float(d_features)).sum(dim=1) / (nc - 1)
            core_dist = mean_inv_pow.pow(-1.0 / d_features)

            mreach = torch.maximum(torch.maximum(core_dist.unsqueeze(1), core_dist.unsqueeze(0)), D_full)
            mreach.fill_diagonal_(0.0)

            mst_edges = prim_mst(mreach)  # (nc - 1, 3): [u, v, weight], unsorted
            degree = torch.zeros(nc, dtype=torch.long)
            for u, v, _w in mst_edges:
                degree[int(u)] += 1
                degree[int(v)] += 1
            internal_mask = degree >= 2
            if not bool(internal_mask.any()):
                internal_mask = torch.ones(nc, dtype=torch.bool)

            dsc = float(mst_edges[:, 2].max())

            per_cluster[cid] = dict(Xc=Xc, nc=nc, core_dist=core_dist, internal_mask=internal_mask, dsc=dsc)

        total = 0.0
        for cid in cluster_ids:
            info_i = per_cluster[cid]
            if info_i['nc'] < 2:
                continue

            min_dspc = float('inf')
            Xi_int = info_i['Xc'][info_i['internal_mask']]
            core_i_int = info_i['core_dist'][info_i['internal_mask']]
            for other_id in cluster_ids:
                if other_id == cid:
                    continue
                info_j = per_cluster[other_id]
                if info_j['nc'] < 2:
                    continue
                Xj_int = info_j['Xc'][info_j['internal_mask']]
                core_j_int = info_j['core_dist'][info_j['internal_mask']]

                D_cross = ClusteringMetrics._pairwise_distances(Xi_int, Xj_int, metric, VI)
                mreach_cross = torch.maximum(
                    torch.maximum(core_i_int.unsqueeze(1), core_j_int.unsqueeze(0)), D_cross
                )
                min_dspc = min(min_dspc, float(mreach_cross.min()))

            dsc_i = info_i['dsc']
            if min_dspc == float('inf'):
                v_i = 0.0  # no other non-trivial cluster to compare against
            else:
                denom = max(min_dspc, dsc_i)
                v_i = (min_dspc - dsc_i) / denom if denom > 0 else 0.0

            total += (info_i['nc'] / n) * v_i

        return total

    # --------------------------
    # Supervised Metrics
    # --------------------------
    @staticmethod
    def rand_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Rand Index (RI) measures the similarity between two clusterings.
        It counts the agreement of pairwise assignments.

        $ \text{RI} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} $

        Parameters
        ----------
        labels_true : torch.Tensor
            Ground-truth labels, shape (n_samples,).
        labels_pred : torch.Tensor
            Predicted labels, shape (n_samples,).

        Returns
        -------
        float
            Rand Index in [0, 1].
        """
        device = labels_true.device
        n_samples = labels_true.size(0)

        # Build contingency matrix
        contingency = torch.zeros(
            (labels_true.max() + 1, labels_pred.max() + 1),
            dtype=torch.float, device=device
        )
        for i in range(n_samples):
            contingency[labels_true[i], labels_pred[i]] += 1

        sum_comb_c = torch.sum(contingency.pow(2) - contingency) / 2
        sum_comb = torch.sum(contingency.sum(dim=1).pow(2) - contingency.sum(dim=1)) / 2
        sum_comb_pred = torch.sum(contingency.sum(dim=0).pow(2) - contingency.sum(dim=0)) / 2

        tp = sum_comb_c
        fp = sum_comb_pred - tp
        fn = sum_comb - tp
        tn = n_samples * (n_samples - 1) / 2 - tp - fp - fn

        ri = (tp + tn) / (tp + fp + fn + tn)
        return ri.item()

    @staticmethod
    def adjusted_rand_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Adjusted Rand Index (ARI), which adjusts RI for chance.

        $ \mathrm{ARI} = \frac{ \mathrm{RI} - \mathrm{E}[\mathrm{RI}] }{ \max(\mathrm{RI}) - \mathrm{E}[\mathrm{RI}] } $

        Parameters
        ----------
        labels_true : torch.Tensor
            Ground-truth labels.
        labels_pred : torch.Tensor
            Predicted labels.

        Returns
        -------
        float
            ARI in [-1, 1], though it’s typically in [0, 1].
        """
        device = labels_true.device
        n_samples = labels_true.size(0)

        # Build contingency matrix
        contingency = torch.zeros(
            (labels_true.max() + 1, labels_pred.max() + 1),
            dtype=torch.float, device=device
        )
        for i in range(n_samples):
            contingency[labels_true[i], labels_pred[i]] += 1

        sum_comb_c = torch.sum(contingency.pow(2) - contingency) / 2
        sum_comb = torch.sum(contingency.sum(dim=1).pow(2) - contingency.sum(dim=1)) / 2
        sum_comb_pred = torch.sum(contingency.sum(dim=0).pow(2) - contingency.sum(dim=0)) / 2

        expected_index = sum_comb * sum_comb_pred / (n_samples * (n_samples - 1) / 2)
        max_index = (sum_comb + sum_comb_pred) / 2
        rand_index = sum_comb_c

        ari = (rand_index - expected_index) / (max_index - expected_index)
        return ari.item()

    @staticmethod
    def mutual_info_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Mutual Information (MI) between two clusterings.

        $ \mathrm{MI}(U, V) = \sum_{u \in U}\sum_{v \in V} p(u, v) \log\frac{p(u,v)}{p(u)p(v)} $

        Parameters
        ----------
        labels_true : torch.Tensor
            Ground-truth labels.
        labels_pred : torch.Tensor
            Predicted labels.

        Returns
        -------
        float
            Mutual information (>= 0).
        """
        device = labels_true.device
        contingency = torch.zeros(
            (labels_true.max() + 1, labels_pred.max() + 1),
            dtype=torch.float, device=device
        )
        for i in range(labels_true.size(0)):
            contingency[labels_true[i], labels_pred[i]] += 1

        contingency /= contingency.sum()
        outer = contingency.sum(dim=1, keepdim=True) * contingency.sum(dim=0, keepdim=True)
        nonzero = contingency > 0
        mi = (contingency[nonzero] *
              (torch.log(contingency[nonzero]) - torch.log(outer[nonzero]))).sum()
        return mi.item()

    @staticmethod
    def adjusted_mutual_info_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Adjusted Mutual Information (AMI).

        Parameters
        ----------
        labels_true : torch.Tensor
            Ground-truth labels.
        labels_pred : torch.Tensor
            Predicted labels.

        Returns
        -------
        float
            AMI in [0, 1].
        """
        mi = ClusteringMetrics.mutual_info_score(labels_true, labels_pred)
        n_samples = labels_true.size(0)
        true_counts = torch.bincount(labels_true)
        pred_counts = torch.bincount(labels_pred)

        h_true = -torch.sum((true_counts / n_samples) *
                            torch.log(true_counts / n_samples + 1e-10))
        h_pred = -torch.sum((pred_counts / n_samples) *
                            torch.log(pred_counts / n_samples + 1e-10))

        # This "expected_mi" here is a rough approximation
        expected_mi = (h_true * h_pred) / n_samples
        ami = (mi - expected_mi) / (0.5 * (h_true + h_pred) - expected_mi)
        return ami.item()

    @staticmethod
    def normalized_mutual_info_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Normalized Mutual Information (NMI) in [0, 1].

        $ \text{NMI} = \frac{2 \times \mathrm{MI}}{H(\text{true}) + H(\text{pred})} $

        Parameters
        ----------
        labels_true : torch.Tensor
            Ground-truth labels.
        labels_pred : torch.Tensor
            Predicted labels.

        Returns
        -------
        float
            NMI.
        """
        mi = ClusteringMetrics.mutual_info_score(labels_true, labels_pred)
        n_samples = labels_true.size(0)
        true_counts = torch.bincount(labels_true).float()
        pred_counts = torch.bincount(labels_pred).float()

        h_true = -torch.sum((true_counts / n_samples) *
                            torch.log(true_counts / n_samples + 1e-10))
        h_pred = -torch.sum((pred_counts / n_samples) *
                            torch.log(pred_counts / n_samples + 1e-10))

        nmi = 2.0 * mi / (h_true + h_pred)
        return nmi.item()

    @staticmethod
    def fowlkes_mallows_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Fowlkes-Mallows index (FM) = $ \sqrt{ \text{precision} \times \text{recall} } $.

        Parameters
        ----------
        labels_true : torch.Tensor
            Ground-truth labels.
        labels_pred : torch.Tensor
            Predicted labels.

        Returns
        -------
        float
            FM in [0, 1].
        """
        device = labels_true.device
        n_samples = labels_true.size(0)
        contingency = torch.zeros(
            (labels_true.max() + 1, labels_pred.max() + 1),
            dtype=torch.float, device=device
        )
        for i in range(n_samples):
            contingency[labels_true[i], labels_pred[i]] += 1

        tp = torch.sum(contingency.pow(2)) - n_samples
        tp_fp = torch.sum(contingency.sum(dim=0).pow(2)) - n_samples
        tp_fn = torch.sum(contingency.sum(dim=1).pow(2)) - n_samples

        fm = torch.sqrt(tp / tp_fp * tp / tp_fn)
        return fm.item()

    @staticmethod
    def _contingency_matrix(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> torch.Tensor:
        r"""
        `(n_true_classes, n_pred_clusters)` contingency counts, in
        `float64` regardless of input dtype, built via `scatter_add_`
        rather than a Python loop over samples. Assumes contiguous
        0-indexed labels (matching every other contingency-based method in
        this class); shared by `homogeneity_score`, `completeness_score`,
        and `normalized_information_distance`.
        """
        device = labels_true.device
        lt = labels_true.to(torch.long)
        lp = labels_pred.to(torch.long)
        n_true = int(lt.max().item()) + 1
        n_pred = int(lp.max().item()) + 1
        contingency = torch.zeros((n_true, n_pred), dtype=torch.float64, device=device)
        flat_idx = lt * n_pred + lp
        contingency.view(-1).scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float64))
        return contingency

    @staticmethod
    def _entropy_from_counts(counts: torch.Tensor, n_samples: int) -> torch.Tensor:
        r"""
        Shannon entropy (nats) of the distribution given by `counts / n_samples`.
        Uses `torch.special.xlogy` (defined as exactly `0` at `x=0`,
        regardless of `y`) instead of an `epsilon`-smoothed `log`, which is
        what made the pre-Phase-1 `homogeneity_score`/`completeness_score`
        fail sklearn parity -- see the design doc.
        """
        p = counts.to(torch.float64) / n_samples
        return -torch.special.xlogy(p, p).sum()

    @staticmethod
    def homogeneity_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Homogeneity: each cluster contains only members of a single class.

        $$ h = 1 - \frac{H(C \mid K)}{H(C)} $$

        (Rosenberg & Hirschberg, 2007), `H(C)` the entropy of the true
        class distribution, `H(C \mid K)` the entropy of the true class
        conditional on the predicted cluster. `h = 1.0` by convention when
        `H(C) = 0` (every point has the same true class, trivially
        homogeneous regardless of partition).

        Parameters
        ----------
        labels_true : torch.Tensor
            Ground-truth labels, contiguous 0-indexed integers.
        labels_pred : torch.Tensor
            Predicted labels, contiguous 0-indexed integers.

        Returns
        -------
        float
            Homogeneity in [0, 1].
        """
        n_samples = labels_true.size(0)
        contingency = ClusteringMetrics._contingency_matrix(labels_true, labels_pred)
        class_counts = contingency.sum(dim=1)    # n_c
        cluster_counts = contingency.sum(dim=0)  # n_k

        h_c = ClusteringMetrics._entropy_from_counts(class_counts, n_samples)
        if h_c == 0:
            return 1.0

        p_joint = contingency / n_samples
        p_c_given_k = contingency / cluster_counts.clamp_min(1).unsqueeze(0)
        h_c_given_k = -torch.special.xlogy(p_joint, p_c_given_k).sum()

        return (1.0 - h_c_given_k / h_c).item()

    @staticmethod
    def completeness_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Completeness: all members of a given class are assigned to the
        same cluster.

        $$ c = 1 - \frac{H(K \mid C)}{H(K)} $$

        (Rosenberg & Hirschberg, 2007), `H(K)` the entropy of the
        predicted cluster distribution, `H(K \mid C)` the entropy of the
        predicted cluster conditional on the true class. `c = 1.0` by
        convention when `H(K) = 0` (everything in one cluster, trivially
        complete regardless of the true classes).

        Parameters
        ----------
        labels_true : torch.Tensor
            Ground-truth labels, contiguous 0-indexed integers.
        labels_pred : torch.Tensor
            Predicted labels, contiguous 0-indexed integers.

        Returns
        -------
        float
            Completeness in [0, 1].
        """
        n_samples = labels_true.size(0)
        contingency = ClusteringMetrics._contingency_matrix(labels_true, labels_pred)
        class_counts = contingency.sum(dim=1)    # n_c
        cluster_counts = contingency.sum(dim=0)  # n_k

        h_k = ClusteringMetrics._entropy_from_counts(cluster_counts, n_samples)
        if h_k == 0:
            return 1.0

        p_joint = contingency / n_samples
        p_k_given_c = contingency / class_counts.clamp_min(1).unsqueeze(1)
        h_k_given_c = -torch.special.xlogy(p_joint, p_k_given_c).sum()

        return (1.0 - h_k_given_c / h_k).item()

    @staticmethod
    def v_measure_score(labels_true: torch.Tensor, labels_pred: torch.Tensor, beta: float = 1.0) -> float:
        r"""
        V-measure: the weighted harmonic mean of homogeneity and completeness.

        $$ V_\beta = \frac{(1 + \beta) \cdot h \cdot c}{\beta \cdot h + c} $$

        `beta=1.0` (default) weighs homogeneity and completeness equally
        and recovers the familiar `2hc / (h + c)` form; `beta > 1` weighs
        completeness more, `beta < 1` weighs homogeneity more.

        Parameters
        ----------
        labels_true : torch.Tensor
        labels_pred : torch.Tensor
        beta : float, optional
            Completeness/homogeneity weighting. (default: 1.0)

        Returns
        -------
        float
            V-measure in [0, 1].
        """
        h = ClusteringMetrics.homogeneity_score(labels_true, labels_pred)
        c = ClusteringMetrics.completeness_score(labels_true, labels_pred)
        denom = beta * h + c
        if denom == 0:
            return 0.0
        return (1.0 + beta) * h * c / denom

    @staticmethod
    def normalized_information_distance(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Normalized Information Distance (Vinh, Epps & Bailey, 2010):

        $$ \text{NID} = 1 - \frac{I(U; V)}{\max(H(U), H(V))} $$

        a metric (satisfies the triangle inequality, unlike `1 - \text{NMI}`
        under some normalizations) in `[0, 1]`, `0` for identical
        partitions. Reuses `mutual_info_score` for `I(U; V)`.

        Parameters
        ----------
        labels_true : torch.Tensor
            Contiguous 0-indexed integers.
        labels_pred : torch.Tensor
            Contiguous 0-indexed integers.

        Returns
        -------
        float
            NID in [0, 1] (0 = identical partitions).
        """
        n_samples = labels_true.size(0)
        contingency = ClusteringMetrics._contingency_matrix(labels_true, labels_pred)
        h_true = ClusteringMetrics._entropy_from_counts(contingency.sum(dim=1), n_samples)
        h_pred = ClusteringMetrics._entropy_from_counts(contingency.sum(dim=0), n_samples)
        max_h = torch.max(h_true, h_pred)

        if max_h == 0:
            return 0.0

        mi = ClusteringMetrics.mutual_info_score(labels_true, labels_pred)
        return (1.0 - mi / max_h.item())

    @staticmethod
    def purity_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Purity score measures how many samples belong to the correct cluster.

        $ \mathrm{purity} = \frac{1}{N} \sum_k \max_j \lvert C_k \cap U_j \rvert $

        Parameters
        ----------
        labels_true : torch.Tensor
        labels_pred : torch.Tensor

        Returns
        -------
        float
            Purity in [0, 1].
        """
        device = labels_true.device
        n_samples = labels_true.size(0)
        contingency = torch.zeros(
            (labels_true.max() + 1, labels_pred.max() + 1),
            dtype=torch.float, device=device
        )
        for i in range(n_samples):
            contingency[labels_true[i], labels_pred[i]] += 1

        purity = torch.sum(torch.max(contingency, dim=0).values) / n_samples
        return purity.item()

    @staticmethod
    def confusion_matrix(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> torch.Tensor:
        r"""
        Compute a confusion matrix.

        Parameters
        ----------
        labels_true : torch.Tensor
        labels_pred : torch.Tensor

        Returns
        -------
        torch.Tensor
            A 2D (C x C) matrix, where C is the number of unique labels.
        """
        unique_labels = torch.unique(labels_true)
        num_labels = unique_labels.size(0)
        cm = torch.zeros((num_labels, num_labels), dtype=torch.int32)

        for i, label_true in enumerate(unique_labels):
            for j, label_pred in enumerate(unique_labels):
                cm[i, j] = ((labels_true == label_true) & (labels_pred == label_pred)).sum().item()

        return cm

    @staticmethod
    def classification_report(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> dict:
        r"""
        Compute a simple classification report for each class: precision, recall, F1,
        Jaccard index, ROC-AUC, and support.

        Parameters
        ----------
        labels_true : torch.Tensor
        labels_pred : torch.Tensor

        Returns
        -------
        dict
            A dictionary keyed by class label, each containing:
            "precision", "recall", "f1-score", "support", "jaccard", "roc_auc".
        """
        device = labels_true.device
        unique_labels = torch.unique(labels_true)
        report = {}
        
        for label in unique_labels:
            true_positives = ((labels_true == label) & (labels_pred == label)).sum().item()
            false_positives = ((labels_true != label) & (labels_pred == label)).sum().item()
            false_negatives = ((labels_true == label) & (labels_pred != label)).sum().item()

            precision = (true_positives / (true_positives + false_positives)
                         if (true_positives + false_positives) > 0 else 0.0)
            recall = (true_positives / (true_positives + false_negatives)
                      if (true_positives + false_negatives) > 0 else 0.0)
            f1_score = (2.0 * (precision * recall) / (precision + recall)
                        if (precision + recall) > 0 else 0.0)
            support = (labels_true == label).sum().item()
            jaccard_index = (true_positives / (true_positives + false_positives + false_negatives)
                             if (true_positives + false_positives + false_negatives) > 0 else 0.0)

            binary_true = (labels_true == label).float().to(device)
            binary_pred = (labels_pred == label).float().to(device)
            roc_auc = ClusteringMetrics.roc_auc_score(binary_true, binary_pred)

            report[int(label)] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "support": support,
                "jaccard": jaccard_index,
                "roc_auc": roc_auc
            }
    
        return report

    @staticmethod
    def roc_auc_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Compute a naive ROC-AUC for binary predictions (0 or 1).
        If all labels are the same class, returns 1.0 by definition.

        Parameters
        ----------
        labels_true : torch.Tensor
            Binary ground-truth (0 or 1), shape (n_samples,).
        labels_pred : torch.Tensor
            Binary predictions or real-valued probabilities, shape (n_samples,).

        Returns
        -------
        float
            Area Under the ROC Curve (AUC).
        """
        if labels_true.sum() == 0 or labels_true.sum() == labels_true.size(0):
            # Degenerate case: all positives or all negatives => AUC = 1.0 or undefined
            return 1.0

        sorted_indices = torch.argsort(labels_pred, descending=True)
        labels_true = labels_true[sorted_indices]

        tpr = torch.cumsum(labels_true, dim=0) / labels_true.sum()
        fpr = torch.cumsum(1 - labels_true, dim=0) / (labels_true.size(0) - labels_true.sum())

        auc = torch.trapz(tpr, fpr)
        return auc.item()
    
    @staticmethod
    def kl_divergence_gmm(gmm_p, gmm_q, n_samples: int = 10000) -> float:
        r"""
        Approximate the KL divergence $D_{KL}(p \Vert q)$ between two
        Gaussian Mixture Models using Monte Carlo sampling from ``gmm_p``.

        Parameters
        ----------
        gmm_p : GaussianMixture
            The first GMM (interpreted as distribution p).
        gmm_q : GaussianMixture
            The second GMM (interpreted as distribution q).
        n_samples : int, optional
            Number of samples to draw from gmm_p for the Monte Carlo approximation
            (default: 10000).

        Returns
        -------
        float
            Approximated KL divergence $ \mathrm{E}_{x \sim p}[\log p(x) - \log q(x)]$.
        """
        device = gmm_p.device
        samples, _ = gmm_p.sample(n_samples)  # (n_samples, n_features)
        samples = samples.to(device)

        # Log-likelihood of samples under both GMMs
        log_p = gmm_p.score_samples(samples)
        log_q = gmm_q.score_samples(samples)

        kl_divergence = (log_p - log_q).mean().item()
        return kl_divergence

    @staticmethod
    def evaluate_clustering(
        gmm_model,
        X: torch.Tensor,
        true_labels: Optional[torch.Tensor] = None,
        metrics: Optional[list] = None
    ) -> dict:
        r"""
        Evaluate a fitted GMM using a set of requested metrics, possibly with ground-truth labels.

        Parameters
        ----------
        gmm_model : GaussianMixture
            A fitted GMM model (must have ``fitted_ == True``).
        X : torch.Tensor
            Data to evaluate, shape (n_samples, n_features).
        true_labels : torch.Tensor or None
            Ground-truth labels for supervised metrics (optional).
        metrics : list of str or None
            Which metrics to compute. If None, uses a default set.

        Returns
        -------
        results : dict
            A dictionary of metric_name -> metric_value pairs.

        Notes
        -----
        BIC/AIC are not computed here -- call ``gmm_model.bic(X)`` /
        ``gmm_model.aic(X)`` directly.
        """
        if not gmm_model.fitted_:
            raise ValueError("The GMM model must be fitted before evaluation.")

        if metrics is None:
            metrics = [
                # Supervised
                "rand_score", "adjusted_rand_score", "mutual_info_score",
                "normalized_mutual_info_score", "adjusted_mutual_info_score",
                "fowlkes_mallows_score", "homogeneity_score", "completeness_score",
                "v_measure_score", "purity_score",
                # Classification-based
                "classification_report", "confusion_matrix",
                # Unsupervised
                "silhouette_score", "davies_bouldin_index", "calinski_harabasz_score",
                "dunn_index",
            ]

        # Predict cluster labels
        pred_labels = gmm_model.predict(X).cpu()
        results = {}

        # If ground-truth labels provided, compute supervised metrics
        if true_labels is not None:
            true_labels = true_labels.cpu()

            if "rand_score" in metrics:
                results["rand_score"] = ClusteringMetrics.rand_score(true_labels, pred_labels)
            if "adjusted_rand_score" in metrics:
                results["adjusted_rand_score"] = ClusteringMetrics.adjusted_rand_score(true_labels, pred_labels)
            if "mutual_info_score" in metrics:
                results["mutual_info_score"] = ClusteringMetrics.mutual_info_score(true_labels, pred_labels)
            if "adjusted_mutual_info_score" in metrics:
                results["adjusted_mutual_info_score"] = ClusteringMetrics.adjusted_mutual_info_score(true_labels, pred_labels)
            if "normalized_mutual_info_score" in metrics:
                results["normalized_mutual_info_score"] = ClusteringMetrics.normalized_mutual_info_score(true_labels, pred_labels)
            if "fowlkes_mallows_score" in metrics:
                results["fowlkes_mallows_score"] = ClusteringMetrics.fowlkes_mallows_score(true_labels, pred_labels)
            if "homogeneity_score" in metrics:
                results["homogeneity_score"] = ClusteringMetrics.homogeneity_score(true_labels, pred_labels)
            if "completeness_score" in metrics:
                results["completeness_score"] = ClusteringMetrics.completeness_score(true_labels, pred_labels)
            if "v_measure_score" in metrics:
                results["v_measure_score"] = ClusteringMetrics.v_measure_score(true_labels, pred_labels)
            if "purity_score" in metrics:
                results["purity_score"] = ClusteringMetrics.purity_score(true_labels, pred_labels)
            if "classification_report" in metrics:
                results["classification_report"] = ClusteringMetrics.classification_report(true_labels, pred_labels)
            if "confusion_matrix" in metrics:
                results["confusion_matrix"] = ClusteringMetrics.confusion_matrix(true_labels, pred_labels)

        # Unsupervised metrics
        unique_pred_labels = torch.unique(pred_labels)
        if len(unique_pred_labels) > 1:
            if "silhouette_score" in metrics:
                results["silhouette_score"] = ClusteringMetrics.silhouette_score(
                    X.cpu(), pred_labels, gmm_model.n_components
                )
            if "davies_bouldin_index" in metrics:
                results["davies_bouldin_index"] = ClusteringMetrics.davies_bouldin_index(
                    X.cpu(), pred_labels, gmm_model.n_components
                )
            if "calinski_harabasz_score" in metrics:
                results["calinski_harabasz_score"] = ClusteringMetrics.calinski_harabasz_score(
                    X.cpu(), pred_labels, gmm_model.n_components
                )
            if "dunn_index" in metrics:
                results["dunn_index"] = ClusteringMetrics.dunn_index(
                    X.cpu(), pred_labels, gmm_model.n_components
                )

        return results
