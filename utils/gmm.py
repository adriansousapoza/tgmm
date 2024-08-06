import torch
from torch import nn
import utils.metrics

# reload
import importlib
importlib.reload(utils.metrics)

from torch.distributions import MultivariateNormal
from utils.metrics import ClusteringMetrics

class GaussianMixture(nn.Module):
    """
    Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm.

    Parameters
    ----------
    n_features : int
        The number of features in the dataset.
    n_components : int, default=1
        The number of mixture components.
    covariance_type : {'full', 'tied', 'diagonal', 'spherical'}, default='diagonal'
        The type of covariance parameters to use.
    tol : float, default=1e-5
        Convergence threshold.
    reg_covar : float, default=1e-10
        Non-negative regularization added to the diagonal of covariance.
    max_iter : int, default=1000
        The number of EM iterations to perform.
    n_init : int, default=1
        The number of initializations to perform.
    init_params : {'random', 'points', 'kpp', 'kmeans', 'maxdist'}, default='random'
        The method used to initialize the weights, the means, and the covariances.
    weights_init : tensor, default=None
        The initial weights.
    means_init : tensor, default=None
        The initial means.
    covariances_init : tensor, default=None
        The initial covariances.
    precisions_init : tensor, default=None
        The initial precisions.
    weights_prior : distribution or list of distributions, default=None
        The prior distribution for weights.
    means_prior : distribution or list of distributions, default=None
        The prior distribution for means.
    covariances_prior : distribution or list of distributions, default=None
        The prior distribution for covariances.
    precisions_prior : distribution or list of distributions, default=None
        The prior distribution for precisions.
    random_state : int, default=None
        The seed used by the random number generator.
    warm_start : bool, default=False
        If True, reuse the solution of the last fit.
    verbose : int, default=0
        Enable verbose output.
    verbose_interval : int, default=10
        Number of iteration done before the next print.
    device : {'cpu', 'cuda'}, default=None
        The device on which the model is run.

    Attributes
    ----------
    weights_ : tensor
        The weights of each mixture component.
    means_ : tensor
        The mean of each mixture component.
    covariances_ : tensor
        The covariance of each mixture component.
    precisions_ : tensor
        The precision of each mixture component.
    fitted_ : bool
        True if the model is fitted, False otherwise.
    converged_ : bool
        True if the model has converged, False otherwise.
    n_iter_ : int
        Number of step used by the best fit.
    lower_bound_ : float
        Log-likelihood of the best fit.

    Methods
    -------
    fit(X, max_iter=None, tol=None, n_init=None, random_state=None, warm_start=None):
        Fit the Gaussian mixture model.
    sample(n_samples=1):
        Generate random samples from the fitted Gaussian distribution.
    predict(X):
        Predict the labels for the data samples in X using trained model.
    score_samples(X):
        Return the per-sample likelihood of the data under the model.
    evaluate_clustering(X, true_labels=None, metrics=None):
        Evaluate supervised and unsupervised clustering metrics.
    """

    def __init__(
            self,
            n_features,
            n_components=1,
            covariance_type='full',
            tol=1e-5,
            reg_covar=1e-6,
            max_iter=1000,
            n_init=1,
            init_params='random',
            weights_init=None,
            means_init=None,
            covariances_init=None,
            precisions_init=None,
            weights_prior=None,
            means_prior=None,
            covariances_prior=None,
            precisions_prior=None,
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
        self.weights_prior = weights_prior
        self.means_prior = means_prior
        self.covariances_prior = covariances_prior
        self.precisions_prior = precisions_prior
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.precisions_ = None
        self.fitted_ = False

        self._init_params()

    def _init_params(self):
        """
        Initialize model parameters.
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
        
        # Initialize covariances and precisions based on the covariance type
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

    def _prior_log_prob(self, X):
        """
        Compute the log probability of the prior distributions.

        Parameters
        ----------
        X : tensor
            Input data.

        Returns
        -------
        log_resp_prior : tensor
            Log probability of the prior distributions.
        """

        n_samples, n_features = X.shape
        log_resp_prior = torch.zeros((n_samples, self.n_components), dtype=torch.float32, device=self.device)
        
        # Weights prior
        if self.weights_prior is not None:
            if hasattr(self.weights_prior, 'log_prob'):
                prior_log_prob = self.weights_prior.log_prob(self.weights_)
                log_resp_prior += prior_log_prob.unsqueeze(0).repeat(n_samples, 1)
            elif isinstance(self.weights_prior, torch.distributions.Distribution):
                prior_log_prob = self.weights_prior.log_prob(self.weights_)
                log_resp_prior += prior_log_prob.unsqueeze(0).repeat(n_samples, 1)
            else:
                raise ValueError("Unsupported weights prior type")
        
        # Means prior
        if self.means_prior is not None:
            if isinstance(self.means_prior, list):
                for k in range(self.n_components):
                    if hasattr(self.means_prior[k], 'log_prob'):
                        prior_log_prob = self.means_prior[k].log_prob(self.means_[k])
                        log_resp_prior[:, k] += prior_log_prob
                    elif isinstance(self.means_prior[k], torch.distributions.Distribution):
                        prior_log_prob = self.means_prior[k].log_prob(self.means_[k])
                        log_resp_prior[:, k] += prior_log_prob
                    else:
                        raise ValueError(f"Unsupported prior type for component {k}")
            elif hasattr(self.means_prior, 'log_prob'):
                prior_log_prob = self.means_prior.log_prob(self.means_)
                log_resp_prior += prior_log_prob.unsqueeze(0).repeat(n_samples, 1)
            elif isinstance(self.means_prior, torch.distributions.Distribution):
                prior_log_prob = self.means_prior.log_prob(self.means_)
                log_resp_prior += prior_log_prob.unsqueeze(0).repeat(n_samples, 1)
            else:
                raise ValueError("Unsupported prior type")
        
        if self.covariances_prior is not None and self.precisions_prior is not None:
            raise ValueError("Covariances and precisions prior cannot be used simultaneously")
        
        # Covariances prior
        if self.covariances_prior is not None:
            if isinstance(self.covariances_prior, list):
                for k in range(self.n_components):
                    if hasattr(self.covariances_prior[k], 'log_prob'):
                        prior_log_prob = self.covariances_prior[k].log_prob(self.covariances_[k])
                        log_resp_prior[:, k] += prior_log_prob
                    elif isinstance(self.covariances_prior[k], torch.distributions.Distribution):
                        prior_log_prob = self.covariances_prior[k].log_prob(self.covariances_[k])
                        log_resp_prior[:, k] += prior_log_prob
                    else:
                        raise ValueError(f"Unsupported covariances prior type for component {k}")
            elif hasattr(self.covariances_prior, 'log_prob'):
                prior_log_prob = self.covariances_prior.log_prob(self.covariances_)
                log_resp_prior += prior_log_prob.unsqueeze(0).repeat(n_samples, 1)
            elif isinstance(self.covariances_prior, torch.distributions.Distribution):
                prior_log_prob = self.covariances_prior.log_prob(self.covariances_)
                log_resp_prior += prior_log_prob.unsqueeze(0).repeat(n_samples, 1)
            else:
                raise ValueError("Unsupported covariances prior type")

        # Precisions prior
        elif self.precisions_prior is not None:
            if isinstance(self.precisions_prior, list):
                for k in range(self.n_components):
                    if hasattr(self.precisions_prior[k], 'log_prob'):
                        prior_log_prob = self.precisions_prior[k].log_prob(self.precisions_[k])
                        log_resp_prior[:, k] += prior_log_prob
                    elif isinstance(self.precisions_prior[k], torch.distributions.Distribution):
                        prior_log_prob = self.precisions_prior[k].log_prob(self.precisions_[k])
                        log_resp_prior[:, k] += prior_log_prob
                    else:
                        raise ValueError(f"Unsupported precisions prior type for component {k}")
            elif hasattr(self.precisions_prior, 'log_prob'):
                prior_log_prob = self.precisions_prior.log_prob(self.precisions_)
                log_resp_prior += prior_log_prob.unsqueeze(0).repeat(n_samples, 1)
            elif isinstance(self.precisions_prior, torch.distributions.Distribution):
                prior_log_prob = self.precisions_prior.log_prob(self.precisions_)
                log_resp_prior += prior_log_prob.unsqueeze(0).repeat(n_samples, 1)
            else:
                raise ValueError("Unsupported precisions prior type")
        
        return log_resp_prior

    def _e_step(self, X):
        """
        Perform the E-step of the EM algorithm.

        Parameters
        ----------
        X : tensor
            Input data.

        Returns
        -------
        resp : tensor
            Responsibilities for each data point and each component.
        log_prob_norm : tensor
            Log of the sum of the responsibilities for each data point.
        """
        
        def ensure_positive_definite(matrix, reg_covar):
            """
            Ensure that the covariance matrix is positive definite by attempting Cholesky decomposition.
            Increase the regularization parameter if the decomposition fails.

            Parameters
            ----------
            matrix : tensor
                Covariance matrix.
            reg_covar : float
                Regularization parameter.

            Returns
            -------
            cholesky : tensor
                Cholesky decomposition of the covariance matrix.
            reg_covar : float
                Updated regularization parameter.
            """
            try:
                cholesky = torch.linalg.cholesky(matrix)
                return cholesky, reg_covar
            except torch._C._LinAlgError:
                matrix += torch.eye(matrix.shape[0], device=matrix.device) * reg_covar
                try: 
                    cholesky = torch.linalg.cholesky(matrix)
                    return cholesky, reg_covar
                except torch._C._LinAlgError:  
                    while True:
                        try:
                            cholesky = torch.linalg.cholesky(matrix)
                            return cholesky, reg_covar
                        except torch._C._LinAlgError:
                            reg_covar = reg_covar * 10
                            matrix = matrix + torch.eye(matrix.shape[0], device=matrix.device) * reg_covar
                            print(f"Cholesky decomposition failed, increased reg_covar to {reg_covar}")
        
        X = X.to(self.device)
        n_samples, n_features = X.shape
        log_resp = torch.zeros((n_samples, self.n_components), dtype=torch.float32, device=self.device)

        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
        
        for k in range(self.n_components):
            mean = self.means_[k]
            
            if self.covariance_type == 'spherical':
                cov_matrix = self.covariances_[k] + self.reg_covar
                precision = 1.0 / cov_matrix
                log_det_cov = n_features * torch.log(cov_matrix)
            elif self.covariance_type == 'diagonal':
                cov_matrix = self.covariances_[k] + self.reg_covar
                precision = 1.0 / cov_matrix
                log_det_cov = torch.sum(torch.log(cov_matrix))
            elif self.covariance_type == 'tied':
                cov_matrix = self.covariances_ + torch.eye(n_features, device=self.device) * self.reg_covar
                precision = torch.inverse(cov_matrix)
                cholesky, self.reg_covar = ensure_positive_definite(cov_matrix, self.reg_covar)
                log_det_cov = 2 * torch.sum(torch.log(torch.diag(cholesky)))
            elif self.covariance_type == 'full':
                cov_matrix = self.covariances_[k]
                cholesky, self.reg_covar = ensure_positive_definite(cov_matrix, self.reg_covar)
                precision = torch.inverse(cov_matrix)
                log_det_cov = 2 * torch.sum(torch.log(torch.diag(cholesky)))
            else:
                raise ValueError("Unsupported covariance type")
            
            diff = X - mean
            if self.covariance_type in ['full', 'tied']:
                mahalanobis = torch.sum(diff @ precision * diff, dim=1)
            else:
                mahalanobis = torch.sum(diff * precision * diff, dim=1)

            log_prob = -0.5 * (n_features * log_2pi + log_det_cov + mahalanobis)
            log_resp[:, k] = torch.log(self.weights_[k]) + log_prob

        log_resp = log_resp + self._prior_log_prob(X)
        log_prob_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)
        log_resp = log_resp - log_prob_norm
        
        return torch.exp(log_resp), log_prob_norm.sum()


    
    def _m_step(self, X, resp):
        """
        Perform the M-step of the EM algorithm.

        Parameters
        ----------
        X : tensor
            Input data.
        resp : tensor
            Responsibilities for each data point and each component.
        """

        # Update weights
        self.weights_ = resp.sum(dim=0) / resp.sum()

        # Update means
        weighted_sum = torch.mm(resp.t(), X)
        weights_sum = resp.sum(dim=0).unsqueeze(-1)
        self.means_ = weighted_sum / weights_sum

        # Update covariances and precisions
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means_[k].unsqueeze(0)
                weighted_diff = diff.unsqueeze(2) * resp[:, k].unsqueeze(-1).unsqueeze(-1)
                cov_k = torch.matmul(weighted_diff, diff.unsqueeze(1)).sum(dim=0) / resp[:, k].sum()
                self.covariances_[k] = cov_k + torch.eye(self.n_features, device=self.device) * self.reg_covar
        elif self.covariance_type == 'tied':
            total_resp = resp.sum(dim=0)
            average_covariance = torch.zeros((self.n_features, self.n_features), device=self.device)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                average_covariance = average_covariance + torch.mm((resp[:, k].unsqueeze(1) * diff).t(), diff)
            self.covariances_ = average_covariance / total_resp.sum()
            self.covariances_ = self.covariances_ + torch.eye(self.n_features, device=self.device) * self.reg_covar
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
        else:
            print("Covariance type not supported (yet)")

    def _init_krandom(self, data, k):
        """
        Initialize means randomly based on data distribution.

        Parameters
        ----------
        data : tensor
            Input data.
        k : int
            Number of components.

        Returns
        -------
        samples : tensor
            Initialized means.
        """
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
        """
        Initialize means by randomly selecting points from the data.

        Parameters
        ----------
        data : tensor
            Input data.
        k : int
            Number of components.

        Returns
        -------
        samples : tensor
            Initialized means.
        """
        indices = torch.randperm(data.size(0), device=data.device)[:k]
        return data[indices]
    
    def _init_maxdist(self, data, k):
        """
        Initialize means using the maximum distance method.

        Parameters
        ----------
        data : tensor
            Input data.
        k : int
            Number of components.

        Returns
        -------
        centroids : tensor
            Initialized means.
        """
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
        """
        Initialize means using the k-means++ algorithm.

        Parameters
        ----------
        data : tensor
            Input data.
        k : int
            Number of components.

        Returns
        -------
        centroids : tensor
            Initialized means.
        """
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
        """
        Initialize means using the k-means algorithm.

        Parameters
        ----------
        data : tensor
            Input data.
        k : int
            Number of components.
        max_iter : int, default=1000
            Maximum number of iterations.
        atol : float, default=1e-4
            Convergence threshold.

        Returns
        -------
        centroids : tensor
            Initialized means.
        """
        centroids = self._init_kpp(data, k)
        
        for _ in range(max_iter):
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)
            
            new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(k)])
            
            if torch.allclose(centroids, new_centroids, atol=atol):
                break
            
            centroids = new_centroids
        
        return centroids
    
    def fit(self, X, max_iter=None, tol=None, n_init=None, random_state=None, warm_start=None):
        """
        Fit the Gaussian mixture model to the data.

        Parameters
        ----------
        X : tensor
            Input data.
        max_iter : int, default=None
            Maximum number of iterations.
        tol : float, default=None
            Convergence threshold.
        n_init : int, default=None
            Number of initializations.
        random_state : int, default=None
            Seed for random number generator.
        warm_start : bool, default=None
            If True, reuse the solution of the last fit.
        """

        if random_state is not None:
            self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        if warm_start is None:
            warm_start = self.warm_start

        if not self.warm_start or not self.fitted_:
            self._init_params()

        if max_iter is None:
            max_iter = self.max_iter

        if tol is None:
            tol = self.tol

        X = X.to(self.device)

        # Initialize means based on the specified method
        if not self.fitted_:
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
        self.fitted_ = True

    
    def sample(self, n_samples=1):
        """
        Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        samples : tensor
            Generated samples.
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
        Predict the labels for the data samples in X using the trained model.

        Parameters
        ----------
        X : tensor
            Input data.

        Returns
        -------
        labels : tensor
            Predicted labels.
        """
        X = X.to(self.device)
        log_resp, _ = self._e_step(X)
        labels = torch.argmax(log_resp, dim=1)
        return labels
    
    def score_samples(self, X):
        """
        Return the per-sample likelihood of the data under the model.

        Parameters
        ----------
        X : tensor
            Input data.

        Returns
        -------
        log_prob_norm : tensor
            Log-likelihood of the data.
        """
        _, log_prob_norm = self._e_step(X)
        return log_prob_norm
    
    def evaluate_clustering(self, X, true_labels=None, metrics=None):
        """
        Evaluate clustering metrics against true labels.

        Parameters
        ----------
        X : tensor
            Input data.
        true_labels : tensor, default=None
            True labels for the data.
        metrics : list of str, default=None
            List of metrics to evaluate.

        Returns
        -------
        results : dict
            Dictionary of metric results.
        """
        if metrics is None:
            metrics = [
                # supervised clustering metrics
                "rand_score",
                "adjusted_rand_score",
                "mutual_info_score",
                "normalized_mutual_info_score",
                "adjusted_mutual_info_score",
                "fowlkes_mallows_score",
                "homogeneity_score",
                "completeness_score",
                "v_measure_score",
                "purity_score",
                # classification metrics
                "classification_report",
                "confusion_matrix",
                # unsupervised clustering metrics
                "silhouette_score",
                "davies_bouldin_index",
                "calinski_harabasz_score",
                "dunn_index",
                "bic_score",
                "aic_score",
            ]

        pred_labels = self.predict(X).cpu()
        if true_labels is not None:
            true_labels = true_labels.cpu() if isinstance(true_labels, torch.Tensor) else true_labels

        results = {}
        if true_labels is not None:
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
        if "dunn_index" in metrics:
            results["dunn_index"] = ClusteringMetrics.dunn_index(X, pred_labels, self.n_components)

        return results