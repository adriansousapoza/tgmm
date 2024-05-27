import torch
from torch import nn
from torch.distributions import MultivariateNormal
from utils.metrics import ClusteringMetrics

class GaussianMixture(nn.Module):
    """
    Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm.

    Parameters
    ----------
    n_features : int
        The number of features.
    n_components : int, default=1
        The number of mixture components.
    covariance_type : {'full', 'tied', 'diagonal', 'spherical'}, default='full'
        The type of covariance to use. Must be one of:
        - 'full': each component has its own general covariance matrix
        - 'tied': all components share the same general covariance matrix
        - 'diagonal': each component has its own diagonal covariance matrix
        - 'spherical': each component has its own single variance
    tol : float, default=1e-5
        The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
    reg_covar : float, default=1e-10
        Non-negative regularization added to the diagonal of covariance. Allows to assure that the covariance matrices are all positive.
    max_iter : int, default=1000
        The maximum number of EM iterations to perform.
    n_init : int, default=1
        The number of initializations to perform. The best results are kept.
    init_params : {'random', 'kpp', 'points'}, default='random'
        The method used to initialize the weights, the means and the precisions. Must be one of:
        - 'random': parameters are initialized randomly
        - 'kpp': parameters are initialized using the K-means++ method
        - 'points': parameters are initialized using a subset of the data points
    weights_init : array-like, shape (n_components,), default=None
        The initial weights. If None, weights are initialized using the init_params method.
    means_init : array-like, shape (n_components, n_features), default=None
        The initial means. If None, means are initialized using the init_params method.
    covariances_init : array-like, default=None
        The initial variances. If None, variances are initialized using the init_params method. The shape depends on the covariance_type:
        - 'full': (n_components, n_features, n_features)
        - 'tied': (n_features, n_features)
        - 'diagonal': (n_components, n_features)
        - 'spherical': (n_components,)
    precisions_init : array-like, default=None
        The initial precisions. If None, precisions are initialized using the init_params method. The shape depends on the covariance_type:
        - 'full': (n_components, n_features, n_features)
        - 'tied': (n_features, n_features)
        - 'diagonal': (n_components, n_features)
        - 'spherical': (n_components,)
    random_state : int, default=None
        Controls the random seed given at the beginning of the algorithm.
    warm_start : bool, default=False
        If True, the solution of the last fitting is used as initialization for the next call.
    verbose : int, default=0
        Controls the verbosity of the algorithm.
    verbose_interval : int, default=10
        The number of iterations between each verbose output.


    Attributes
    ----------
    weights_ : torch.Tensor, shape (n_components,)
        The weights of each mixture component.
    means_ : torch.Tensor, shape (n_components, n_features)
        The mean of each mixture component.
    covariances_ : torch.Tensor
        The covariance of each mixture component. The shape depends on the covariance_type:
        - 'full': (n_components, n_features, n_features)
        - 'tied': (n_features, n_features)
        - 'diagonal': (n_components, n_features)
        - 'spherical': (n_components,)
    precisions_ : torch.Tensor
        The precision of each mixture component. The precision is the inverse of the covariance. The shape depends on the covariance_type:
        - 'full': (n_components, n_features, n_features)
        - 'tied': (n_features, n_features)
        - 'diagonal': (n_components, n_features)
        - 'spherical': (n_components,)
    precisions_cholesky_ : torch.Tensor
        The Cholesky decomposition of the precision of each mixture component. The shape depends on the covariance_type:
        - 'full': (n_components, n_features, n_features)
        - 'tied': (n_features, n_features)
        - 'diagonal': (n_components, n_features)
        - 'spherical': (n_components,)
    converged_ : bool
        True when the EM algorithm has converged.
    n_iter_ : int
        The number of iterations performed.
    lower_bound_ : float
        The lower bound value of the log-likelihood at the end of the last EM iteration.

    Methods
    -------
    TODO
    """

    def __init__(
            self,
            n_features,
            n_components=1, 
            covariance_type='diagonal', 
            tol=1e-5, 
            reg_covar=1e-10, 
            max_iter=1000,
            n_init=1,
            init_params='random', 
            weights_init=None,
            means_init=None,
            covariances_init=None,
            precisions_init=None,
            random_state=None,
            warm_start=False,
            verbose=0,
            verbose_interval=10,
            device=None
        ):
        
        super(GaussianMixture, self).__init__()

        self.n_features = n_features
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.covariances_init = covariances_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.precisions_ = None

        self._init_params()

    def _init_params(self):
        """
        Initialize the model parameters.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        # Initialize weights
        if self.weights_init is not None:
            self.weights_ = self.weights_init
        else:
            self.weights_ = torch.full((self.n_components,), 1.0 / self.n_components, dtype=torch.float, device=self.device)
        
        # Initialize means
        if self.means_init is not None:
            self.means_ = self.means_init
        else:
            self.means_ = torch.randn(self.n_components, self.n_features, device=self.device)
        
        # Initialize variances and precisions
        if self.covariance_type == 'full':
            if self.covariances_init is not None:
                self.covariances_ = self.covariances_init
                self.precisions_ = torch.inverse(self.covariances_)
            elif self.precisions_init is not None:
                self.precisions_ = self.precisions_init
                self.covariances_ = torch.inverse(self.precisions_)
            else:
                self.covariances_ = torch.stack([torch.eye(self.n_features) * (1.0 + self.reg_covar) for _ in range(self.n_components)])
                self.precisions_ = torch.inverse(self.covariances_)
        elif self.covariance_type == 'diagonal':
            if self.covariances_init is not None:
                self.covariances_ = self.covariances_init
                self.precisions_ = 1.0 / self.covariances_
            elif self.precisions_init is not None:
                self.precisions_ = self.precisions_init
                self.covariances_ = 1.0 / self.precisions_
            else:
                self.covariances_ = torch.ones(self.n_components, self.n_features) + self.reg_covar
                self.precisions_ = 1.0 / self.covariances_
        elif self.covariance_type == 'spherical':
            if self.covariances_init is not None:
                self.covariances_ = self.covariances_init
                self.precisions_ = 1.0 / self.covariances_
            elif self.precisions_init is not None:
                self.precisions_ = self.precisions_init
                self.covariances_ = 1.0 / self.precisions_
            else:
                self.covariances_ = torch.ones(self.n_components) + self.reg_covar
                self.precisions_ = 1.0 / self.covariances_
        elif self.covariance_type == 'tied':
            if self.covariances_init is not None:
                self.covariances_ = self.covariances_init
                self.precisions_ = torch.inverse(self.covariances_)
            elif self.precisions_init is not None:
                self.precisions_ = self.precisions_init
                self.covariances_ = torch.inverse(self.precisions_)
            else:
                self.covariances_ = torch.eye(self.n_features) * (1.0 + self.reg_covar)
                self.precisions_ = torch.inverse(self.covariances_)
        else:
            print("Covariance type not supported. Please use 'full', 'tied', 'diagonal' or 'spherical' instead.")

        # Move parameters to the device
        self.weights_ = self.weights_.to(self.device)
        self.means_ = self.means_.to(self.device)
        self.covariances_ = self.covariances_.to(self.device)
        self.precisions_ = self.precisions_.to(self.device)

    def _update_weights(self, resp):
        self.weights_ = resp.sum(dim=0) / resp.sum()
    
    def _update_means(self, X, resp):
        weighted_sum = torch.mm(resp.t(), X)
        weights_sum = resp.sum(dim=0).unsqueeze(-1)
        self.means_ = weighted_sum / weights_sum

    def _update_covariances(self, X, resp):
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means_[k].unsqueeze(0)
                weighted_diff = diff.unsqueeze(2) * resp[:, k].unsqueeze(-1).unsqueeze(-1)
                cov_k = torch.matmul(weighted_diff, diff.unsqueeze(1)).sum(dim=0) / resp[:, k].sum()
                self.covariances_[k] = cov_k + torch.eye(self.n_features, device=self.device) * self.reg_covar
        elif self.covariance_type == 'diagonal':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = (resp[:, k].unsqueeze(-1) * diff.pow(2)).sum(dim=0) / resp[:, k].sum()
                self.covariances_[k] = self.covariances_[k] + self.reg_covar
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = (resp[:, k] * diff.pow(2).sum(dim=1)).sum() / (self.n_features * resp[:, k].sum())
                self.covariances_[k] = self.covariances_[k] + self.reg_covar
        elif self.covariance_type == 'tied':
            total_resp = resp.sum(dim=0)
            average_covariance = torch.zeros((self.n_features, self.n_features), device=self.device)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                average_covariance = average_covariance + torch.mm((resp[:, k].unsqueeze(1) * diff).t(), diff)
            self.covariances_ = average_covariance / total_resp.sum()
            self.covariances_ = self.covariances_ + torch.eye(self.n_features, device=self.device) * self.reg_covar
        else:
            print("Covariance type not supported (yet)")

    def _e_step(self, X):
        X = X.to(self.device)
        n_samples, n_features = X.shape
        log_resp = torch.zeros((n_samples, self.n_components), dtype=torch.float, device=self.device)

        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))

        for k in range(self.n_components):
            mean = self.means_[k]
            if self.covariance_type == 'full':
                cov_matrix = self.covariances_[k] + torch.eye(n_features, device=self.device) * self.reg_covar
                precision = torch.inverse(cov_matrix)
                log_det_cov = torch.logdet(cov_matrix)
            elif self.covariance_type == 'diagonal':
                cov_matrix = self.covariances_[k] + self.reg_covar
                precision = 1.0 / cov_matrix
                log_det_cov = torch.sum(torch.log(cov_matrix))
            elif self.covariance_type == 'tied':
                cov_matrix = self.covariances_ + torch.eye(n_features, device=self.device) * self.reg_covar
                precision = torch.inverse(cov_matrix)
                log_det_cov = torch.logdet(cov_matrix)
            elif self.covariance_type == 'spherical':
                cov_matrix = self.covariances_[k] + self.reg_covar
                precision = 1.0 / cov_matrix
                log_det_cov = n_features * torch.log(cov_matrix)
            else:
                raise ValueError("Unsupported covariance type")
            
            diff = X - mean
            if self.covariance_type in ['full', 'tied']:
                mahalanobis = torch.sum(diff @ precision * diff, dim=1)
            else:
                mahalanobis = torch.sum(diff * precision * diff, dim=1)
            
            log_prob = -0.5 * (n_features * log_2pi + log_det_cov + mahalanobis)
            log_resp[:, k] = torch.log(self.weights_[k]) + log_prob

        log_prob_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)
        log_resp = log_resp - log_prob_norm
        return torch.exp(log_resp), log_prob_norm.sum()
    
    def _m_step(self, X, resp):
        self._update_weights(resp)
        self._update_means(X, resp)
        self._update_covariances(X, resp)

    def _init_krandom(self, data, k):
        mu = torch.mean(data, dim=0)
        if data.dim() == 1:
            cov = torch.var(data)
            samples = torch.randn(k, device=data.device) * torch.sqrt(cov)
        else:
            cov = torch.cov(data.t())
            samples = torch.randn(k, data.size(1), device=data.device) @ torch.linalg.cholesky(cov).t()
        samples += mu
        return samples
    
    def _init_kpoints(self, data, k):
        indices = torch.randperm(data.size(0), device=data.device)[:k]
        return data[indices]
    
    def _init_maxdist(self, data, k):
        n_samples, _ = data.shape
        centroids = self._init_kpp(data, k)
        initial_idx = torch.randint(0, n_samples, (1,), device=data.device)
        centroids[0] = data[initial_idx]

        for i in range(1, k):
            dist_sq = torch.cdist(data, centroids[:i]).pow(2).min(dim=1)[0]
            selected_idx = torch.argmax(dist_sq)
            centroids[i] = data[selected_idx]

        return centroids

    def _init_kpp(self, data, k):
        n_samples, _ = data.shape
        centroids = torch.empty((k, data.size(1)), device=data.device)
        initial_idx = torch.randint(0, n_samples, (1,), device=data.device)
        centroids[0] = data[initial_idx]

        for i in range(1, k):
            dist_sq = torch.cdist(data, centroids[:i]).pow(2).min(dim=1)[0]
            probabilities = dist_sq / dist_sq.sum()
            selected_idx = torch.multinomial(probabilities, 1)
            centroids[i] = data[selected_idx]

        return centroids
    
    def _init_kmeans(self, data, k, max_iter=1000, atol=1e-4):
        centroids = self._init_kpp(data, k)
        
        for _ in range(max_iter):
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)
            
            new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(k)])
            
            if torch.allclose(centroids, new_centroids, atol=atol):
                break
            
            centroids = new_centroids
        
        return centroids
    
    def fit(self, X, max_iter=None, tol=None, n_init=None, random_state=None):
        """
        Fit the model to the data X.

        Parameters
        ----------
        X : torch.Tensor
            The input data.
        y : Ignored
            Not used, present for API consistency by convention.
        """
        if random_state is not None:
            self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Initialize parameters
        self._init_params()

        if max_iter is None:
            max_iter = self.max_iter

        if tol is None:
            tol = self.tol

        X = X.to(self.device)

        # Initialize means based on the specified method
        if self.means_init is not None:
            self.means_ = self.means_init.to(self.device)
        else:
            if self.init_params == 'random':
                self.means_ = self._init_krandom(X, self.n_components)
            elif self.init_params == 'points':
                self.means_ = self._init_kpoints(X, self.n_components)
            elif self.init_params == 'kpp':
                self.means_ = self._init_kpp(X, self.n_components)
            elif self.init_params == 'kmeans':
                self.means_ = self._init_kmeans(X, self.n_components)
            elif self.init_params == 'maxdist':
                self.means_ = self._init_maxdist(X, self.n_components)
            else:
                raise ValueError("Unsupported initialization method.")
            
        log_prob_norm = 0
        for n_iter in range(max_iter):
            prev_log_prob_norm = log_prob_norm
            
            # E-step
            resp, log_prob_norm = self._e_step(X)
            
            # M-step
            self._m_step(X, resp)
            
            if abs(log_prob_norm - prev_log_prob_norm) < self.tol:
                break
        
        self.converged_ = n_iter < self.max_iter - 1
        self.n_iter_ = n_iter
        self.lower_bound_ = log_prob_norm / X.size(0)

    
    def sample(self, n_samples=1):
        """
        Generate random samples from the fitted Gaussian distribution.
        """
        indices = torch.multinomial(self.weights_, n_samples, replacement=True)
        samples = []
        for i in indices:
            if self.covariance_type == 'full':
                dist = MultivariateNormal(self.means_[i], covariance_matrix=self.covariances_[i])
            elif self.covariance_type == 'diagonal':
                dist = MultivariateNormal(self.means_[i], covariance_matrix=torch.diag(self.covariances_[i]))
            elif self.covariance_type == 'tied':
                dist = MultivariateNormal(self.means_[i], covariance_matrix=self.covariances_)
            elif self.covariance_type == 'spherical':
                dist = MultivariateNormal(self.means_[i], covariance_matrix=torch.eye(self.n_features) * self.covariances_[i])
            else:
                raise ValueError("Unsupported covariance type")
            sample = dist.sample()
            samples.append(sample)
        samples = torch.stack(samples)
        return samples

    
    def predict(self, X):
        """
        Predict the labels for the data samples in X using the model.
        
        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to a single data point.
        
        Returns
        -------
        labels : torch.Tensor, shape (n_samples,)
            Component labels.
        """
        X = X.to(self.device)
        log_resp, _ = self._e_step(X)
        labels = torch.argmax(log_resp, dim=1)
        return labels
    
    def score_samples(self, X):
        """
        Compute the log likelihood of the data X under the model.
        
        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to a single data point.
        
        Returns
        -------
        log_prob : torch.Tensor, shape (n_samples,)
            Log likelihood of each sample in X.
        """
        _, log_prob_norm = self._e_step(X)
        return log_prob_norm
    
    def evaluate_clustering(self, X, true_labels=None, metrics=None):
        """
        Evaluate the clustering performance using various metrics.
        
        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            The input data.
        true_labels : torch.Tensor or array-like, shape (n_samples,), default=None
            The true labels of the data points. If None, only unsupervised metrics are calculated.
        metrics : list of str, default=None
            List of metrics to calculate. If None, calculates all metrics.
        
        Returns
        -------
        results : dict
            Dictionary containing the calculated metrics.
        """
        if metrics is None:
            metrics = [
                "rand_score",
                "adjusted_rand_score",
                "mutual_info_score",
                "normalized_mutual_info_score",
                "adjusted_mutual_info_score",
                "fowlkes_mallows_score",
                "homogeneity_score",
                "completeness_score",
                "v_measure_score",
                "silhouette_score",
                "davies_bouldin_index",
                "calinski_harabasz_score",
                "bic_score",
                "aic_score",
            ]

        pred_labels = self.predict(X).cpu()
        if true_labels is not None:
            true_labels = true_labels.cpu() if isinstance(true_labels, torch.Tensor) else true_labels

        results = {}
        if true_labels is not None:
            if "rand_score" in metrics:
                results["rand_score"] = ClusteringMetrics.rand_score(true_labels, pred_labels).item()
            if "adjusted_rand_score" in metrics:
                results["adjusted_rand_score"] = ClusteringMetrics.adjusted_rand_score(true_labels, pred_labels).item()
            if "mutual_info_score" in metrics:
                results["mutual_info_score"] = ClusteringMetrics.mutual_info_score(true_labels, pred_labels).item()
            if "adjusted_mutual_info_score" in metrics:
                results["adjusted_mutual_info_score"] = ClusteringMetrics.adjusted_mutual_info_score(true_labels, pred_labels).item()
            if "normalized_mutual_info_score" in metrics:
                results["normalized_mutual_info_score"] = ClusteringMetrics.normalized_mutual_info_score(true_labels, pred_labels).item()
            if "fowlkes_mallows_score" in metrics:
                results["fowlkes_mallows_score"] = ClusteringMetrics.fowlkes_mallows_score(true_labels, pred_labels)
            if "homogeneity_score" in metrics:
                results["homogeneity_score"] = ClusteringMetrics.homogeneity_score(true_labels, pred_labels)
            if "completeness_score" in metrics:
                results["completeness_score"] = ClusteringMetrics.completeness_score(true_labels, pred_labels)
            if "v_measure_score" in metrics:
                results["v_measure_score"] = ClusteringMetrics.v_measure_score(true_labels, pred_labels)
        
        if "silhouette_score" in metrics:
            results["silhouette_score"] = ClusteringMetrics.silhouette_score(X, pred_labels, self.n_components)
        if "davies_bouldin_index" in metrics:
            results["davies_bouldin_index"] = ClusteringMetrics.davies_bouldin_index(X, pred_labels, self.n_components)
        if "calinski_harabasz_score" in metrics:
            results["calinski_harabasz_score"] = ClusteringMetrics.calinski_harabasz_score(X, pred_labels, self.n_components)
        if "bic_score" in metrics:
            results["bic_score"] = ClusteringMetrics.bic_score(self.lower_bound_, X, self.n_components, self.covariance_type)
        if "aic_score" in metrics:
            results["aic_score"] = ClusteringMetrics.aic_score(self.lower_bound_, X, self.n_components, self.covariance_type)
        

        return results