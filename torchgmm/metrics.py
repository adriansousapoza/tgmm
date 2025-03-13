import torch
from typing import Optional, List, Dict, Union

class ClusteringMetrics:
    r"""
    A collection of clustering and classification metrics, both unsupervised and supervised.

    This class provides methods such as:
    
    - **KL divergence** for comparing two GMMs (via Monte Carlo).
    - **Information criteria** (AIC, BIC) for model selection.
    - **Unsupervised metrics** (silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn index).
    - **Supervised metrics** (Rand index, ARI, mutual info variants, purity, classification report).

    All methods are static for convenience, and most accept data in PyTorch Tensors
    (potentially on GPU). For supervised metrics, the user must provide `labels_true`
    and `labels_pred` as integer-encoded 1D tensors of the same shape.

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

    @staticmethod
    def kl_divergence_gmm(gmm_p, gmm_q, n_samples: int = 10000) -> float:
        r"""
        Approximate the KL divergence :math:`D_{KL}(p \Vert q)` between two
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
            Approximated KL divergence :math:`\mathrm{E}_{x \sim p} [\log p(x) - \log q(x)]`.
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
    def bic_score(
        lower_bound_: float,
        X: torch.Tensor,
        n_components: int,
        covariance_type: str
    ) -> float:
        r"""
        Compute the Bayesian Information Criterion (BIC) for a GMM given its
        average log-likelihood (lower bound).

        BIC = n_params * ln(n_samples) - 2 * log_likelihood

        Parameters
        ----------
        lower_bound_ : float
            Average (per-sample) log-likelihood or lower bound from the GMM.
        X : torch.Tensor
            Data used in fitting, shape (n_samples, n_features).
        n_components : int
            Number of mixture components in the GMM.
        covariance_type : str
            Covariance type, one of {'full', 'tied', 'diag', 'spherical'}.

        Returns
        -------
        float
            The BIC score (lower is better).
        """
        n_samples, n_features = X.shape

        # Determine number of free parameters in covariance
        if covariance_type == 'full':
            cov_params = n_components * n_features * (n_features + 1) / 2.0
        elif covariance_type == 'diag':
            cov_params = n_components * n_features
        elif covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.0
        elif covariance_type == 'spherical':
            cov_params = n_components
        else:
            raise ValueError(f"Unsupported covariance type: {covariance_type}")

        # Means + weights
        mean_params = n_features * n_components
        weight_params = n_components - 1

        n_parameters = cov_params + mean_params + weight_params
        log_likelihood = lower_bound_ * n_samples  # total log-likelihood

        bic = n_parameters * torch.log(torch.tensor(n_samples, dtype=torch.float)) - 2.0 * log_likelihood
        return bic.item()

    @staticmethod
    def aic_score(
        lower_bound_: float,
        X: torch.Tensor,
        n_components: int,
        covariance_type: str
    ) -> float:
        r"""
        Compute the Akaike Information Criterion (AIC) for a GMM.

        AIC = 2 * n_params - 2 * log_likelihood

        Parameters
        ----------
        lower_bound_ : float
            Average (per-sample) log-likelihood from the GMM.
        X : torch.Tensor
            Data used in fitting, shape (n_samples, n_features).
        n_components : int
            Number of mixture components.
        covariance_type : str
            Covariance type, one of {'full', 'tied', 'diag', 'spherical'}.

        Returns
        -------
        float
            The AIC score (lower is better).
        """
        n_samples, n_features = X.shape
        if covariance_type == 'full':
            cov_params = n_components * n_features * (n_features + 1) / 2.0
        elif covariance_type == 'diag':
            cov_params = n_components * n_features
        elif covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.0
        elif covariance_type == 'spherical':
            cov_params = n_components
        else:
            raise ValueError(f"Unsupported covariance type: {covariance_type}")

        mean_params = n_features * n_components
        weight_params = n_components - 1

        n_parameters = cov_params + mean_params + weight_params
        log_likelihood = lower_bound_ * n_samples
        aic = 2.0 * n_parameters - 2.0 * log_likelihood

        return aic

    @staticmethod
    def silhouette_score(
        X: torch.Tensor,
        labels: torch.Tensor,
        n_components: int
    ) -> float:
        r"""
        Compute the silhouette score for a partition of the data.

        .. math::
            \mathrm{silhouette}(i) = \frac{b_i - a_i}{\max(a_i, b_i)}

        Where:
        - \(a_i\) is the mean distance to points in the same cluster.
        - \(b_i\) is the minimum mean distance to points in a different cluster.

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
        labels : torch.Tensor
            Cluster labels, shape (n_samples,).
        n_components : int
            Number of clusters (must be >= 2).

        Returns
        -------
        float
            The mean silhouette score over all samples (range is typically [-1, 1],
            though it’s seldom negative in practice if distances are Euclidean).
        """
        assert n_components > 1, "Silhouette score is only defined when there are at least 2 clusters."
        
        labels = labels.to(X.device)
        distances = torch.cdist(X, X)  # (n_samples, n_samples)
        
        A = torch.zeros(labels.size(0), dtype=torch.float, device=X.device)
        B = torch.full((labels.size(0),), float('inf'), dtype=torch.float, device=X.device)

        # For each cluster i, compute intra-cluster distances and inter-cluster distances
        for i in range(n_components):
            mask_i = (labels == i)
            if mask_i.sum() <= 1:
                continue

            intra_cluster_distances = distances[mask_i][:, mask_i]
            # a_i: average distance within the same cluster
            A[mask_i] = intra_cluster_distances.sum(dim=1) / (mask_i.sum() - 1)

            # b_i: minimum distance to any other cluster
            for j in range(n_components):
                if i == j:
                    continue
                mask_j = (labels == j)
                if mask_j.sum() == 0:
                    continue
                inter_cluster_distances = distances[mask_i][:, mask_j]
                B[mask_i] = torch.min(B[mask_i], inter_cluster_distances.mean(dim=1))

        silhouette_scores = (B - A) / torch.max(A, B)
        return silhouette_scores.mean().item()

    @staticmethod
    def davies_bouldin_index(X: torch.Tensor, labels: torch.Tensor, n_components: int) -> float:
        r"""
        Compute the Davies-Bouldin index (lower is better).

        .. math::
            DB = \frac{1}{k} \sum_i \max_{j \neq i} \frac{S_i + S_j}{M_{ij}}

        Where:
        - \(S_i\) is the average distance of points in cluster i to its centroid.
        - \(M_{ij}\) is the distance between cluster centroids i and j.

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
        labels : torch.Tensor
            Cluster labels, shape (n_samples,).
        n_components : int
            Number of clusters (must be >= 2).

        Returns
        -------
        float
            Davies-Bouldin index.
        """
        assert n_components > 1, "Davies-Bouldin index is only defined when >= 2 clusters."
        labels = labels.to(X.device)

        # Compute cluster centroids
        centroids = [X[labels == i].mean(dim=0) for i in range(n_components)]
        centroids = torch.stack(centroids, dim=0)

        # Distance matrix among centroids
        cluster_distances = torch.cdist(centroids, centroids)

        similarities = torch.zeros((n_components, n_components), device=X.device)

        for i in range(n_components):
            mask_i = (labels == i)
            dist_i = torch.norm(X[mask_i] - centroids[i], dim=1).mean()

            for j in range(n_components):
                if i == j:
                    continue
                mask_j = (labels == j)
                dist_j = torch.norm(X[mask_j] - centroids[j], dim=1).mean()
                similarities[i, j] = (dist_i + dist_j) / cluster_distances[i, j]

        db_index = torch.max(similarities, dim=1).values.mean()
        return db_index.item()

    @staticmethod
    def calinski_harabasz_score(X: torch.Tensor, labels: torch.Tensor, n_components: int) -> float:
        r"""
        Compute the Calinski-Harabasz index (a ratio of between-cluster dispersion
        to within-cluster dispersion).

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
        labels : torch.Tensor
            Cluster labels.
        n_components : int
            Number of clusters.

        Returns
        -------
        float
            Calinski-Harabasz index (higher is better).
        """
        labels = labels.to(X.device)
        centroid_overall = X.mean(dim=0)
        
        # Cluster centroids
        centroids = [X[labels == i].mean(dim=0) for i in range(n_components)]
        centroids = torch.stack(centroids)

        # Between-cluster dispersion (SSB) & within-cluster (SSW)
        SSB = sum((labels == i).sum() * torch.norm(centroids[i] - centroid_overall).pow(2)
                  for i in range(n_components))
        SSW = sum(torch.norm(X[labels == i] - centroids[i], dim=1).pow(2).sum()
                  for i in range(n_components))

        n_samples = X.shape[0]
        CH = (SSB / (n_components - 1)) / (SSW / (n_samples - n_components))
        return CH.item()

    @staticmethod
    def dunn_index(X: torch.Tensor, labels: torch.Tensor, n_components: int) -> float:
        r"""
        Compute the Dunn index:

        .. math::
            D = \frac{\min_{i \neq j} d(C_i, C_j)}{\max_k \mathrm{diam}(C_k)}

        Where:
        - \(d(C_i, C_j)\) is the minimum distance between any points in clusters i, j.
        - \(\mathrm{diam}(C_k)\) is the maximum distance between any points in cluster k.

        Higher Dunn index indicates better cluster separation.

        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
        labels : torch.Tensor
            Cluster labels.
        n_components : int
            Number of clusters.

        Returns
        -------
        float
            Dunn index (higher is better).
        """
        labels = labels.to(X.device)
        distances = torch.cdist(X, X)

        min_intercluster_dist = float('inf')
        max_intracluster_dist = 0.0

        for i in range(n_components):
            mask_i = (labels == i)
            if mask_i.sum() <= 1:
                continue

            intra_distances = distances[mask_i][:, mask_i]
            max_intracluster_dist = max(max_intracluster_dist, intra_distances.max().item())

            for j in range(i + 1, n_components):
                mask_j = (labels == j)
                if mask_j.sum() == 0:
                    continue
                inter_distances = distances[mask_i][:, mask_j]
                current_min = inter_distances.min().item()
                if current_min < min_intercluster_dist:
                    min_intercluster_dist = current_min

        dunn_index = (min_intercluster_dist / max_intracluster_dist) if max_intracluster_dist > 0 else 0.0
        return dunn_index

    # --------------------------
    # Supervised Metrics
    # --------------------------
    @staticmethod
    def rand_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Rand Index (RI) measures the similarity between two clusterings.
        It counts the agreement of pairwise assignments.

        .. math::
            \text{RI} = \frac{TP + TN}{TP + TN + FP + FN}

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

        .. math::
            \mathrm{ARI} = \frac{ \mathrm{RI} - \mathrm{E}[\mathrm{RI}] }{ \max(\mathrm{RI}) - \mathrm{E}[\mathrm{RI}] }

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

        .. math::
            \mathrm{MI}(U, V) = \sum_{u \in U}\sum_{v \in V} p(u, v) \log\frac{p(u,v)}{p(u)p(v)}

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

        expected_mi = (h_true * h_pred) / n_samples  # rough approximation
        ami = (mi - expected_mi) / (0.5 * (h_true + h_pred) - expected_mi)
        return ami.item()

    @staticmethod
    def normalized_mutual_info_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Normalized Mutual Information (NMI) in [0, 1].

        Parameters
        ----------
        labels_true : torch.Tensor
            Ground-truth labels.
        labels_pred : torch.Tensor
            Predicted labels.

        Returns
        -------
        float
            NMI = 2 * MI / (H(true) + H(pred)).
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
        Fowlkes-Mallows index (FM) = sqrt(precision * recall).

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
    def completeness_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Completeness score measures how much each cluster contains only samples
        of a single class.

        Parameters
        ----------
        labels_true : torch.Tensor
            Ground-truth labels.
        labels_pred : torch.Tensor
            Predicted labels.

        Returns
        -------
        float
            Completeness in [0, 1].
        """
        device = labels_true.device
        n_samples = labels_true.size(0)
        contingency = torch.zeros(
            (labels_true.max() + 1, labels_pred.max() + 1),
            dtype=torch.float, device=device
        )
        for i in range(n_samples):
            contingency[labels_true[i], labels_pred[i]] += 1

        entropy_true = -torch.sum(labels_true.bincount().float() / n_samples *
                                  torch.log(labels_true.bincount().float() / n_samples + 1e-10))

        # H(C | K) (conditional entropy)
        entropy_cond = -torch.sum(contingency / n_samples *
                                  torch.log((contingency + 1e-10) /
                                            contingency.sum(dim=1, keepdim=True)))

        comp_score = 1 - entropy_cond / entropy_true
        return comp_score.item()

    @staticmethod
    def homogeneity_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Homogeneity score measures if each cluster contains samples of only one class.

        Parameters
        ----------
        labels_true : torch.Tensor
        labels_pred : torch.Tensor

        Returns
        -------
        float
            Homogeneity in [0, 1].
        """
        device = labels_true.device
        n_samples = labels_true.size(0)
        contingency = torch.zeros(
            (labels_true.max() + 1, labels_pred.max() + 1),
            dtype=torch.float, device=device
        )
        for i in range(n_samples):
            contingency[labels_true[i], labels_pred[i]] += 1

        entropy_pred = -torch.sum(labels_pred.bincount().float() / n_samples *
                                  torch.log(labels_pred.bincount().float() / n_samples + 1e-10))

        # H(K | C)
        entropy_cond = -torch.sum(contingency / n_samples *
                                  torch.log((contingency + 1e-10) /
                                            contingency.sum(dim=0, keepdim=True)))

        hom_score = 1 - entropy_cond / entropy_pred
        return hom_score.item()

    @staticmethod
    def v_measure_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        V-measure = 2 * (homogeneity * completeness) / (homogeneity + completeness).

        Parameters
        ----------
        labels_true : torch.Tensor
        labels_pred : torch.Tensor

        Returns
        -------
        float
            V-measure in [0, 1].
        """
        homogeneity = ClusteringMetrics.homogeneity_score(labels_true, labels_pred)
        completeness = ClusteringMetrics.completeness_score(labels_true, labels_pred)
        if (homogeneity + completeness) == 0:
            return 0.0
        return 2.0 * (homogeneity * completeness) / (homogeneity + completeness)

    @staticmethod
    def purity_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
        r"""
        Purity score measures how many samples belong to the correct cluster.

        .. math::
            \mathrm{purity} = \frac{1}{N} \sum_k \max_j |C_k \cap U_j|

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
        # labels_pred = labels_pred[sorted_indices]  # not strictly needed for AUC calc

        tpr = torch.cumsum(labels_true, dim=0) / labels_true.sum()
        fpr = torch.cumsum(1 - labels_true, dim=0) / (labels_true.size(0) - labels_true.sum())

        auc = torch.trapz(tpr, fpr)
        return auc.item()

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
                "dunn_index", "bic_score", "aic_score",
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

        # Info criteria
        if "bic_score" in metrics:
            results["bic_score"] = ClusteringMetrics.bic_score(
                gmm_model.lower_bound_,
                X.cpu(),
                gmm_model.n_components,
                gmm_model.covariance_type
            )
        if "aic_score" in metrics:
            results["aic_score"] = ClusteringMetrics.aic_score(
                gmm_model.lower_bound_,
                X.cpu(),
                gmm_model.n_components,
                gmm_model.covariance_type
            )

        return results
