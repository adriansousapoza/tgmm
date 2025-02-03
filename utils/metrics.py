import torch
from typing import Optional


class ClusteringMetrics:
    """
    Clustering metrics for evaluating clustering quality, both unsupervised and supervised.
    """

    @staticmethod
    def kl_divergence_gmm(gmm_p, gmm_q, n_samples=10000):
        """
        Approximate the KL divergence between two GMMs using Monte Carlo sampling.

        Parameters
        ----------
        gmm_p : GaussianMixture
            The first GMM (p).
        gmm_q : GaussianMixture
            The second GMM (q).
        n_samples : int, default=10000
            Number of samples to draw from gmm_p.

        Returns
        -------
        kl_divergence : float
            Approximated KL divergence D(p || q).
        """
        device = gmm_p.device

        # Sample from gmm_p
        samples, _ = gmm_p.sample(n_samples)  # sample() returns (samples, component_indices)
        samples = samples.to(device)

        # Compute log probabilities under both GMMs
        log_p = gmm_p.score_samples(samples)
        log_q = gmm_q.score_samples(samples)

        # Compute the Monte Carlo estimate of KL divergence
        kl_divergence = (log_p - log_q).mean().item()

        return kl_divergence

    @staticmethod
    def bic_score(lower_bound_, X, n_components, covariance_type):
        """
        Compute Bayesian Information Criterion (BIC) for a GMM given the average lower bound.

        Parameters
        ----------
        lower_bound_ : float
            Average log-likelihood (lower bound).
        X : torch.Tensor
            Input data of shape (n_samples, n_features).
        n_components : int
            Number of components in the GMM.
        covariance_type : str
            Covariance type (one of 'full', 'tied', 'diag', 'spherical').

        Returns
        -------
        float
            BIC score (the lower, the better).
        """
        n_samples, n_features = X.shape
        # Number of free covariance parameters depends on covariance_type
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
        bic = n_parameters * torch.log(torch.tensor(n_samples, dtype=torch.float)) - 2 * log_likelihood

        return bic.item()

    @staticmethod
    def aic_score(lower_bound_, X, n_components, covariance_type):
        """
        Compute Akaike Information Criterion (AIC) for a GMM given the average lower bound.
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
        weight_params = n_components - 1  # Because weights sum to 1

        n_parameters = cov_params + mean_params + weight_params
        log_likelihood = lower_bound_ * n_samples
        aic = 2 * n_parameters - 2 * log_likelihood

        return aic

    @staticmethod
    def silhouette_score(X, labels, n_components):
        """
        Compute the silhouette score. Assumes at least 2 clusters.
        """
        assert n_components > 1, "Silhouette score is only defined when there are at least 2 clusters."
        
        labels = labels.to(X.device)
        distances = torch.cdist(X, X)
        A = torch.zeros(labels.size(0), dtype=torch.float, device=X.device)
        B = torch.full((labels.size(0),), float('inf'), dtype=torch.float, device=X.device)

        for i in range(n_components):
            mask = (labels == i)
            if mask.sum() <= 1:
                continue

            intra_cluster_distances = distances[mask][:, mask]
            A[mask] = intra_cluster_distances.sum(dim=1) / (mask.sum() - 1)

            for j in range(n_components):
                if i != j:
                    inter_cluster_mask = (labels == j)
                    inter_cluster_distances = distances[mask][:, inter_cluster_mask]
                    B[mask] = torch.min(B[mask], inter_cluster_distances.mean(dim=1))

        silhouette_scores = (B - A) / torch.max(A, B)
        overall_score = silhouette_scores.mean()

        return overall_score.item()

    @staticmethod
    def davies_bouldin_index(X, labels, n_components):
        """
        Compute the Davies-Bouldin index. Assumes at least 2 clusters.
        """
        assert n_components > 1, "Davies-Bouldin index is only defined when there are at least 2 clusters."
        labels = labels.to(X.device)

        # Compute cluster centroids
        centroids = [X[labels == i].mean(dim=0) for i in range(n_components)]
        centroids = torch.stack(centroids)
        
        cluster_distances = torch.cdist(centroids, centroids)
        similarities = torch.zeros((n_components, n_components), device=X.device)

        for i in range(n_components):
            mask_i = (labels == i)
            dist_i = torch.norm(X[mask_i] - centroids[i], dim=1).mean()
            
            for j in range(n_components):
                if i != j:
                    mask_j = (labels == j)
                    dist_j = torch.norm(X[mask_j] - centroids[j], dim=1).mean()
                    similarities[i, j] = (dist_i + dist_j) / cluster_distances[i, j]
                    
        max_similarity = torch.max(similarities, dim=1).values
        return max_similarity.mean().item()

    @staticmethod
    def calinski_harabasz_score(X, labels, n_components):
        """
        Compute the Calinski-Harabasz index.
        """
        labels = labels.to(X.device)
        centroid_overall = X.mean(dim=0)
        
        # Cluster centroids
        centroids = [X[labels == i].mean(dim=0) for i in range(n_components)]
        centroids = torch.stack(centroids)
        
        SSB = sum((labels == i).sum() * torch.norm(centroids[i] - centroid_overall).pow(2) 
                  for i in range(n_components))
        SSW = sum(torch.norm(X[labels == i] - centroids[i], dim=1).pow(2).sum() 
                  for i in range(n_components))
        
        n_samples = X.shape[0]
        CH = (SSB / (n_components - 1)) / (SSW / (n_samples - n_components))
        return CH.item()

    @staticmethod
    def dunn_index(X, labels, n_components):
        """
        Compute the Dunn Index.
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
                if mask_j.sum() <= 0:
                    continue

                inter_distances = distances[mask_i][:, mask_j]
                current_min = inter_distances.min().item()
                if current_min < min_intercluster_dist:
                    min_intercluster_dist = current_min

        dunn_index = min_intercluster_dist / max_intracluster_dist if max_intracluster_dist > 0 else 0
        return dunn_index

    # --------------------------
    # Supervised Metrics
    # --------------------------
    @staticmethod
    def rand_score(labels_true, labels_pred):
        """
        Rand index (RI): a simple measure of cluster similarity.
        """
        n_samples = labels_true.size(0)
        contingency = torch.zeros((labels_true.max() + 1, labels_pred.max() + 1), 
                                  dtype=torch.float, device=labels_true.device)
        for i in range(n_samples):
            contingency[labels_true[i], labels_pred[i]] += 1

        sum_comb_c = torch.sum(contingency.pow(2) - contingency) / 2
        sum_comb = torch.sum(contingency.sum(dim=1).pow(2) - contingency.sum(dim=1)) / 2
        sum_comb_pred = torch.sum(contingency.sum(dim=0).pow(2) - contingency.sum(dim=0)) / 2

        tp = sum_comb_c
        fp = sum_comb_pred - tp
        fn = sum_comb - tp
        tn = n_samples * (n_samples - 1) / 2 - tp - fp - fn

        rand_index = (tp + tn) / (tp + fp + fn + tn)
        return rand_index.item()

    @staticmethod
    def adjusted_rand_score(labels_true, labels_pred):
        """
        Adjusted Rand Index (ARI).
        """
        n_samples = labels_true.size(0)
        contingency = torch.zeros((labels_true.max() + 1, labels_pred.max() + 1), dtype=torch.float)
        for i in range(n_samples):
            contingency[labels_true[i], labels_pred[i]] += 1

        sum_comb_c = torch.sum(contingency.pow(2) - contingency) / 2
        sum_comb = torch.sum(contingency.sum(dim=1).pow(2) - contingency.sum(dim=1)) / 2
        sum_comb_pred = torch.sum(contingency.sum(dim=0).pow(2) - contingency.sum(dim=0)) / 2

        expected_index = sum_comb * sum_comb_pred / (n_samples * (n_samples - 1) / 2)
        max_index = (sum_comb + sum_comb_pred) / 2
        rand_index = sum_comb_c

        adjusted_rand_index = (rand_index - expected_index) / (max_index - expected_index)
        return adjusted_rand_index.item()

    @staticmethod
    def mutual_info_score(labels_true, labels_pred):
        """
        Compute the Mutual Information (MI) between two clusterings.
        """
        contingency = torch.zeros((labels_true.max() + 1, labels_pred.max() + 1), dtype=torch.float)
        for i in range(labels_true.size(0)):
            contingency[labels_true[i], labels_pred[i]] += 1

        contingency /= contingency.sum()
        outer = contingency.sum(dim=1).unsqueeze(1) * contingency.sum(dim=0).unsqueeze(0)
        nonzero = contingency > 0
        mi = (contingency[nonzero] * (torch.log(contingency[nonzero]) - torch.log(outer[nonzero]))).sum()
        return mi.item()

    @staticmethod
    def adjusted_mutual_info_score(labels_true, labels_pred):
        """
        Adjusted Mutual Information (AMI).
        """
        mi = ClusteringMetrics.mutual_info_score(labels_true, labels_pred)
        n_samples = labels_true.size(0)
        true_counts = torch.bincount(labels_true)
        pred_counts = torch.bincount(labels_pred)

        h_true = -torch.sum((true_counts / n_samples) * torch.log(true_counts / n_samples + 1e-10))
        h_pred = -torch.sum((pred_counts / n_samples) * torch.log(pred_counts / n_samples + 1e-10))

        expected_mi = (h_true * h_pred) / n_samples
        ami = (mi - expected_mi) / (0.5 * (h_true + h_pred) - expected_mi)
        return ami.item()

    @staticmethod
    def normalized_mutual_info_score(labels_true, labels_pred):
        """
        Normalized Mutual Information (NMI).
        """
        mi = ClusteringMetrics.mutual_info_score(labels_true, labels_pred)
        h_true = -torch.sum(labels_true.bincount().float() / labels_true.size(0) *
                            torch.log(labels_true.bincount().float() / labels_true.size(0) + 1e-10))
        h_pred = -torch.sum(labels_pred.bincount().float() / labels_pred.size(0) *
                            torch.log(labels_pred.bincount().float() / labels_pred.size(0) + 1e-10))

        nmi = 2 * mi / (h_true + h_pred)
        return nmi.item()

    @staticmethod
    def fowlkes_mallows_score(labels_true, labels_pred):
        """
        Fowlkes-Mallows score.
        """
        n_samples = labels_true.size(0)
        contingency = torch.zeros((labels_true.max() + 1, labels_pred.max() + 1), dtype=torch.float)
        for i in range(n_samples):
            contingency[labels_true[i], labels_pred[i]] += 1

        tp = torch.sum(contingency.pow(2)) - n_samples
        tp_fp = torch.sum(contingency.sum(dim=0).pow(2)) - n_samples
        tp_fn = torch.sum(contingency.sum(dim=1).pow(2)) - n_samples

        return torch.sqrt(tp / tp_fp * tp / tp_fn).item()

    @staticmethod
    def completeness_score(labels_true, labels_pred):
        """
        Completeness score.
        """
        contingency = torch.zeros((labels_true.max() + 1, labels_pred.max() + 1), dtype=torch.float)
        for i in range(labels_true.size(0)):
            contingency[labels_true[i], labels_pred[i]] += 1

        n_samples = labels_true.size(0)
        entropy_true = -torch.sum(labels_true.bincount().float() / n_samples *
                                  torch.log(labels_true.bincount().float() / n_samples + 1e-10))
        # Conditional entropy H(C|K)
        entropy_cond = -torch.sum(contingency / n_samples *
                                  torch.log((contingency + 1e-10) /
                                            contingency.sum(dim=1, keepdim=True)))

        comp_score = 1 - entropy_cond / entropy_true
        return comp_score.item()

    @staticmethod
    def homogeneity_score(labels_true, labels_pred):
        """
        Homogeneity score.
        """
        contingency = torch.zeros((labels_true.max() + 1, labels_pred.max() + 1), dtype=torch.float)
        for i in range(labels_true.size(0)):
            contingency[labels_true[i], labels_pred[i]] += 1

        n_samples = labels_true.size(0)
        entropy_pred = -torch.sum(labels_pred.bincount().float() / n_samples *
                                  torch.log(labels_pred.bincount().float() / n_samples + 1e-10))
        # Conditional entropy H(K|C)
        entropy_cond = -torch.sum(contingency / n_samples *
                                  torch.log((contingency + 1e-10) /
                                            contingency.sum(dim=0, keepdim=True)))

        hom_score = 1 - entropy_cond / entropy_pred
        return hom_score.item()

    @staticmethod
    def v_measure_score(labels_true, labels_pred):
        """
        V-measure score (harmonic mean of homogeneity and completeness).
        """
        homogeneity = ClusteringMetrics.homogeneity_score(labels_true, labels_pred)
        completeness = ClusteringMetrics.completeness_score(labels_true, labels_pred)
        if homogeneity + completeness == 0:
            return 0.0
        v_measure = 2 * (homogeneity * completeness) / (homogeneity + completeness)
        return v_measure

    @staticmethod
    def purity_score(labels_true, labels_pred):
        """
        Purity score (supervised).
        """
        n_samples = labels_true.size(0)
        device = labels_true.device

        contingency = torch.zeros(
            (labels_true.max() + 1, labels_pred.max() + 1),
            dtype=torch.float,
            device=device
        )
        for i in range(n_samples):
            contingency[labels_true[i], labels_pred[i]] += 1

        purity = torch.sum(torch.max(contingency, dim=0).values) / n_samples
        return purity.item()

    @staticmethod
    def confusion_matrix(labels_true, labels_pred):
        """
        Compute confusion matrix for classification-based metrics.
        """
        unique_labels = torch.unique(labels_true)
        num_labels = unique_labels.size(0)
        cm = torch.zeros((num_labels, num_labels), dtype=torch.int32)

        for i, label_true in enumerate(unique_labels):
            for j, label_pred in enumerate(unique_labels):
                cm[i, j] = ((labels_true == label_true) & (labels_pred == label_pred)).sum().item()
        return cm

    @staticmethod
    def classification_report(labels_true, labels_pred):
        """
        Compute precision, recall, F1-score, etc. for each class, including a simple ROC-AUC.
        """
        unique_labels = torch.unique(labels_true)
        report = {}
        
        for label in unique_labels:
            true_positives = ((labels_true == label) & (labels_pred == label)).sum().item()
            false_positives = ((labels_true != label) & (labels_pred == label)).sum().item()
            false_negatives = ((labels_true == label) & (labels_pred != label)).sum().item()

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0.0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0.0
            )
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = (labels_true == label).sum().item()
            jaccard_index = (
                true_positives / (true_positives + false_positives + false_negatives)
                if (true_positives + false_positives + false_negatives) > 0
                else 0.0
            )

            binary_true = (labels_true == label).float()
            binary_pred = (labels_pred == label).float()
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
    def roc_auc_score(labels_true, labels_pred):
        """
        Compute a naive ROC-AUC for binary predictions (1=label, 0=not label).
        Expects labels_pred to be 0/1 or real-valued probabilities (in [0,1]).
        """
        if labels_true.sum() == 0 or labels_true.sum() == labels_true.size(0):
            # Degenerate case: all positives or all negatives => AUC = 1.0 or not well-defined
            return 1.0

        # Sort by predicted values in descending order
        sorted_indices = torch.argsort(labels_pred, descending=True)
        labels_true = labels_true[sorted_indices]
        labels_pred = labels_pred[sorted_indices]

        tpr = torch.cumsum(labels_true, dim=0) / labels_true.sum()  # cumulative true positives
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
        """
        Evaluate clustering metrics for a fitted GMM against given data (and true labels if provided).

        Parameters
        ----------
        gmm_model : GaussianMixture
            The fitted GMM model to evaluate. Must have .fitted_ == True.
        X : torch.Tensor
            Data to evaluate, shape (n_samples, n_features).
        true_labels : torch.Tensor or None, default=None
            Ground-truth labels for supervised metrics. If None, only unsupervised metrics are computed.
        metrics : list of str or None, default=None
            Metrics to compute. If None, a default set is used.

        Returns
        -------
        results : dict
            A dictionary with metric names as keys and computed values as floats.
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

        # Predict cluster labels from GMM
        pred_labels = gmm_model.predict(X).cpu()
        results = {}

        # If true labels are provided, compute supervised metrics
        if true_labels is not None:
            if isinstance(true_labels, torch.Tensor):
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
