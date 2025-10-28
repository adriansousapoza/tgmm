import torch
import warnings

class GMMInitializer:
    r"""
    A utility class providing various initialization strategies for GMM parameters.

    This class defines several static methods to produce initial means, weights, and
    covariances for Gaussian Mixture Models from a dataset ``data`` (a 2D tensor of shape (N, D)):

    - :func:`random`
    - :func:`points`
    - :func:`kpp`
    - :func:`kmeans`
    - :func:`maxdist`

    **Mathematical Descriptions:**

    - **random:**  
      Computes the empirical mean $\bar{x}$ and covariance $\Sigma$ of ``data`` and draws
      initial centers as:
      $$
      \mu_i = \bar{x} + L z, \quad z \sim \mathcal{N}(0, I_d),
      $$
      where $ L $ is the Cholesky factor of $\Sigma$.

    - **points:**  
      Randomly selects $ k $ data points:
      $$
      \mu_i = x_{s_i}, \quad \text{for } s_i \in \text{random subset of } \{1, \dots, N\}.
      $$

    - **kpp (k-means++):**  
      Chooses the first center uniformly at random and subsequent centers with probability
      proportional to the squared distance to the nearest already chosen center:
      $$
      P(x_j) = \frac{D(x_j)}{\sum_{j=1}^{N} D(x_j)}, \quad \text{where } D(x_j) = \min_{l} \|x_j - \mu_l\|^2.
      $$

    - **kmeans:**  
      Runs the k-means algorithm starting from k-means++ initialization. At each iteration:
      $$
      c_j = \arg\min_i \|x_j - \mu_i\|^2, \quad \mu_i = \frac{1}{\|C_i\|} \sum_{x_j \in C_i} x_j,
      $$
      until convergence.

    - **maxdist:**  
      A modified k-means++ that selects subsequent centers as:
      $$
      \mu_i = \arg\max_{x} \min_{l < i} \|x - \mu_l\|,
      $$
      and then reselects the first center as:
      $$
      \mu_1 = \arg\max_{x} \min_{l=2}^{k} \|x - \mu_l\|.
      $$

    Example usage::

        from src.gmm_init import GMMInitializer
        
        data = torch.randn(1000, 2)  # Synthetic data
        k = 4
        init_means = GMMInitializer.random(data, k)
    """

    @staticmethod
    def random(data: torch.Tensor, k: int) -> torch.Tensor:
        r"""
        Randomly initialize cluster centers by sampling from the empirical
        distribution of ``data``.

        Mathematically, if $\bar{x}$ and $\Sigma$ are the sample mean and covariance
        of the data, then:
        $$
        \mu_i = \bar{x} + L z, \quad z \sim \mathcal{N}(0, I_d),
        $$
        where $L$ is the Cholesky factor of $\Sigma$.

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor of shape (N, D) representing the dataset.
        k : int
            Number of cluster centers to generate.

        Returns
        -------
        torch.Tensor
            A (k, D) tensor representing the initial cluster centers.
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

    @staticmethod
    def points(data: torch.Tensor, k: int) -> torch.Tensor:
        r"""
        Initialize cluster centers by randomly selecting existing data points.

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor of shape (N, D) representing the dataset.
        k : int
            Number of cluster centers to generate.

        Returns
        -------
        torch.Tensor
            A (k, D) tensor representing the initial cluster centers.
        """
        indices = torch.randperm(data.size(0), device=data.device)[:k]
        return data[indices]

    @staticmethod
    def kpp(data: torch.Tensor, k: int) -> torch.Tensor:
        r"""
        Initialize cluster centers using the k-means++ algorithm.

        The first center is chosen uniformly at random. Subsequent centers are chosen
        with probability proportional to the squared distance from the nearest existing
        center:
        $$
        P(x_j) = \frac{D(x_j)}{\sum_{j=1}^{N} D(x_j)}, \quad D(x_j) = \min_{l} \|x_j - \mu_l\|^2.
        $$

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor of shape (N, D) representing the dataset.
        k : int
            Number of cluster centers to generate.

        Returns
        -------
        torch.Tensor
            A (k, D) tensor representing the initial cluster centers.
        """
        n_samples, _ = data.shape
        centroids = torch.empty((k, data.size(1)), device=data.device)

        # Pick the first center uniformly at random
        initial_idx = torch.randint(0, n_samples, (1,), device=data.device)
        centroids[0] = data[initial_idx]

        for i in range(1, k):
            dist_sq = torch.cdist(data, centroids[:i]).pow(2).min(dim=1)[0]
            probabilities = dist_sq / dist_sq.sum()
            selected_idx = torch.multinomial(probabilities, 1)
            centroids[i] = data[selected_idx]

        return centroids

    @staticmethod
    def kmeans(data: torch.Tensor, k: int, max_iter: int = 1000, atol: float = 1e-4) -> torch.Tensor:
        r"""
        Initialize cluster centers by running the k-means algorithm on ``data``.

        Starting from a k-means++ initialization, k-means iteratively refines the centers by:
        
        1. **Assignment:**  
           $c_j = \arg\min_{i} \|x_j - \mu_i\|^2,$ for each data point $x_j$.
        
        2. **Update:**  
           $\mu_i = \frac{1}{\|C_i\|} \sum_{x_j \in C_i} x_j,$ where $C_i$ is the set of points assigned to center $\mu_i$.
        
        The algorithm stops when the centers move by less than the specified tolerance.

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor of shape (N, D) representing the dataset.
        k : int
            Number of cluster centers to generate.
        max_iter : int, optional
            Maximum number of iterations (default is 1000).
        atol : float, optional
            Convergence tolerance (default is $1\times10^{-4}$).

        Returns
        -------
        torch.Tensor
            A (k, D) tensor representing the final cluster centers.
        """
        centroids = GMMInitializer.kpp(data, k)

        for _ in range(max_iter):
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)
            new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(k)])
            if torch.allclose(centroids, new_centroids, atol=atol):
                break
            centroids = new_centroids

        return centroids

    @staticmethod
    def maxdist(data: torch.Tensor, k: int) -> torch.Tensor:
        r"""
        A modified k-means++ initialization that maximizes the minimum distance
        between centers.

        After randomly selecting the first center, each subsequent center is chosen as:
        $$
        \mu_i = \arg\max_{x \in \mathcal{D}} \min_{l=1,\dots,i-1} \|x - \mu_l\|,
        $$
        ensuring that the new center is as far as possible from the existing centers.
        Finally, the first center is reselected as:
        $$
        \mu_1 = \arg\max_{x \in \mathcal{D}} \min_{l=2}^{k} \|x - \mu_l\|.
        $$

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor of shape (N, D).
        k : int
            Number of cluster centers.

        Returns
        -------
        torch.Tensor
            A (k, D) tensor representing the initial cluster centers.
        """
        n_samples, _ = data.shape
        centroids = torch.empty((k, data.size(1)), device=data.device)

        initial_idx = torch.randint(0, n_samples, (1,), device=data.device)
        centroids[0] = data[initial_idx]

        for i in range(1, k):
            dist_sq = torch.cdist(data, centroids[:i]).pow(2)
            min_dist = dist_sq.min(dim=1)[0]
            selected_idx = torch.argmax(min_dist)
            centroids[i] = data[selected_idx]

        dist_sq_to_first = torch.cdist(data, centroids[1:]).pow(2)
        min_dist_to_first = dist_sq_to_first.min(dim=1)[0]
        new_first_idx = torch.argmax(min_dist_to_first)
        centroids[0] = data[new_first_idx]

        return centroids

    # ====================================================================
    # Weight Initialization Methods
    # ====================================================================
    
    @staticmethod
    def init_weights_uniform(n_components: int, device: torch.device) -> torch.Tensor:
        r"""
        Initialize weights uniformly (equal weights for all components).
        
        Parameters
        ----------
        n_components : int
            Number of mixture components.
        device : torch.device
            Device to create the tensor on.
            
        Returns
        -------
        torch.Tensor
            Uniform weights of shape (n_components,).
        """
        return torch.full(
            (n_components,),
            1.0 / n_components,
            dtype=torch.float32,
            device=device
        )
    
    @staticmethod
    def init_weights_random(n_components: int, device: torch.device) -> torch.Tensor:
        r"""
        Initialize weights randomly from a Dirichlet distribution.
        
        Parameters
        ----------
        n_components : int
            Number of mixture components.
        device : torch.device
            Device to create the tensor on.
            
        Returns
        -------
        torch.Tensor
            Random weights of shape (n_components,) that sum to 1.
        """
        alpha = torch.ones(n_components, device=device)
        weights = torch.distributions.Dirichlet(alpha).sample()
        return weights.float()
    
    @staticmethod
    def init_weights_from_clusters(data: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        r"""
        Initialize weights proportionally to cluster sizes based on k-means assignment.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (n_samples, n_features).
        means : torch.Tensor
            Cluster means of shape (n_components, n_features).
            
        Returns
        -------
        torch.Tensor
            Weights of shape (n_components,) proportional to cluster sizes.
        """
        if data.dim() == 1:
            data = data.unsqueeze(1)
        
        n_components = means.size(0)
        
        # Assign points to nearest mean
        distances = torch.cdist(data, means)
        labels = torch.argmin(distances, dim=1)
        
        # Count points per cluster
        counts = torch.bincount(labels, minlength=n_components).float()
        
        # Handle empty clusters
        counts = torch.clamp(counts, min=1e-10)
        
        # Normalize to get weights
        return counts / counts.sum()

    # ====================================================================
    # Covariance Initialization Methods
    # ====================================================================
    
    @staticmethod
    def init_covariances_eye(n_components: int, n_features: int, 
                            covariance_type: str, reg_covar: float, 
                            device: torch.device) -> torch.Tensor:
        r"""
        Initialize covariances as identity-like matrices/vectors.
        
        Parameters
        ----------
        n_components : int
            Number of mixture components.
        n_features : int
            Number of features.
        covariance_type : str
            Type of covariance ('full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical').
        reg_covar : float
            Regularization added to diagonal.
        device : torch.device
            Device to create tensors on.
            
        Returns
        -------
        torch.Tensor
            Initialized covariances with appropriate shape.
        """
        if covariance_type == 'full':
            out = []
            for _ in range(n_components):
                mat = torch.eye(n_features, device=device) * (1.0 + reg_covar)
                out.append(mat)
            return torch.stack(out, dim=0)
        elif covariance_type == 'diag':
            return torch.ones(n_components, n_features, device=device) * (1.0 + reg_covar)
        elif covariance_type == 'spherical':
            return torch.ones(n_components, device=device) * (1.0 + reg_covar)
        elif covariance_type == 'tied_full':
            return torch.eye(n_features, device=device) * (1.0 + reg_covar)
        elif covariance_type == 'tied_diag':
            return torch.ones(n_features, device=device) * (1.0 + reg_covar)
        elif covariance_type == 'tied_spherical':
            return torch.tensor(1.0 + reg_covar, device=device)
        else:
            raise ValueError(f"Unsupported covariance type: {covariance_type}")
    
    @staticmethod
    def init_covariances_random(n_components: int, n_features: int,
                               covariance_type: str, reg_covar: float,
                               device: torch.device) -> torch.Tensor:
        r"""
        Initialize covariances randomly as positive semi-definite matrices.
        
        Parameters
        ----------
        n_components : int
            Number of mixture components.
        n_features : int
            Number of features.
        covariance_type : str
            Type of covariance.
        reg_covar : float
            Regularization added to diagonal.
        device : torch.device
            Device to create tensors on.
            
        Returns
        -------
        torch.Tensor
            Random covariances with appropriate shape.
        """
        if covariance_type in ('full', 'tied_full'):
            def random_spd(dim):
                A = torch.randn(dim, dim, device=device)
                return A @ A.mT + reg_covar * torch.eye(dim, device=device)
            
            if covariance_type == 'full':
                return torch.stack([random_spd(n_features) for _ in range(n_components)], dim=0)
            else:
                return random_spd(n_features)
        elif covariance_type in ('diag', 'tied_diag'):
            shape = (n_components, n_features) if covariance_type == 'diag' else (n_features,)
            return torch.rand(shape, device=device) + reg_covar
        elif covariance_type in ('spherical', 'tied_spherical'):
            shape = (n_components,) if (covariance_type == 'spherical') else ()
            return torch.rand(shape, device=device) + (1.0 + reg_covar)
        else:
            raise ValueError(f"Unsupported covariance type: {covariance_type}")
    
    @staticmethod
    def init_covariances_global(data: torch.Tensor, n_components: int,
                               covariance_type: str, reg_covar: float) -> torch.Tensor:
        r"""
        Initialize covariances using global data covariance.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (n_samples, n_features).
        n_components : int
            Number of mixture components.
        covariance_type : str
            Type of covariance.
        reg_covar : float
            Regularization added to diagonal.
            
        Returns
        -------
        torch.Tensor
            Initialized covariances based on global data covariance.
        """
        if data.dim() == 1:
            data = data.unsqueeze(1)
        
        n_features = data.size(1)
        device = data.device
        
        # Compute global covariance matrix
        global_cov = torch.cov(data.T)
        global_cov += reg_covar * torch.eye(n_features, device=device)
        
        if covariance_type == 'full':
            return global_cov.unsqueeze(0).repeat(n_components, 1, 1)
        elif covariance_type == 'diag':
            diag_vals = torch.diagonal(global_cov)
            return diag_vals.unsqueeze(0).repeat(n_components, 1)
        elif covariance_type == 'spherical':
            avg_var = torch.mean(torch.diagonal(global_cov))
            return torch.full((n_components,), avg_var.item(), device=device)
        elif covariance_type == 'tied_full':
            return global_cov
        elif covariance_type == 'tied_diag':
            return torch.diagonal(global_cov)
        elif covariance_type == 'tied_spherical':
            return torch.mean(torch.diagonal(global_cov))
        else:
            raise ValueError(f"Unsupported covariance type: {covariance_type}")
    
    @staticmethod
    def init_covariances_empirical(data: torch.Tensor, means: torch.Tensor,
                                  covariance_type: str, reg_covar: float) -> torch.Tensor:
        r"""
        Initialize covariances empirically from cluster assignments.
        
        Assigns each data point to its nearest mean and computes cluster-wise
        empirical covariance matrices.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (n_samples, n_features).
        means : torch.Tensor
            Cluster means of shape (n_components, n_features).
        covariance_type : str
            Type of covariance.
        reg_covar : float
            Regularization added to diagonal.
            
        Returns
        -------
        torch.Tensor
            Empirical covariances with appropriate shape.
        """
        if data.dim() == 1:
            data = data.unsqueeze(1)
        
        n_samples, n_features = data.shape
        n_components = means.size(0)
        device = data.device
        
        # Assign points to nearest mean
        distances = torch.cdist(data, means)
        labels = torch.argmin(distances, dim=1)
        
        if covariance_type == 'full':
            new_covs = []
            for k in range(n_components):
                cluster_mask = (labels == k)
                if not torch.any(cluster_mask):
                    cov_k = torch.eye(n_features, device=device) * (1.0 + reg_covar)
                else:
                    cluster_data = data[cluster_mask]
                    cov_k = torch.cov(cluster_data.T)
                    # Ensure cov_k is always a matrix (for n_features=1, torch.cov returns scalar)
                    if cov_k.ndim == 0:
                        cov_k = cov_k.reshape(1, 1)
                    cov_k += reg_covar * torch.eye(n_features, device=device)
                new_covs.append(cov_k)
            return torch.stack(new_covs, dim=0)
            
        elif covariance_type == 'diag':
            new_covs = []
            for k in range(n_components):
                cluster_mask = (labels == k)
                if not torch.any(cluster_mask):
                    cov_k = torch.ones(n_features, device=device) * (1.0 + reg_covar)
                else:
                    cluster_data = data[cluster_mask]
                    cov_mat = torch.cov(cluster_data.T)
                    # Ensure cov_mat is always a matrix (for n_features=1, torch.cov returns scalar)
                    if cov_mat.ndim == 0:
                        cov_k = cov_mat.reshape(1) + reg_covar
                    else:
                        cov_k = torch.diagonal(cov_mat) + reg_covar
                new_covs.append(cov_k)
            return torch.stack(new_covs, dim=0)
            
        elif covariance_type == 'spherical':
            new_covs = []
            for k in range(n_components):
                cluster_mask = (labels == k)
                if not torch.any(cluster_mask):
                    cov_k = 1.0 + reg_covar
                else:
                    cluster_data = data[cluster_mask]
                    cov_mat = torch.cov(cluster_data.T)
                    # For n_features=1, torch.cov returns a scalar
                    if cov_mat.ndim == 0:
                        cov_k = max(cov_mat.item(), reg_covar)
                    else:
                        cov_k = max(torch.mean(torch.diagonal(cov_mat)).item(), reg_covar)
                new_covs.append(torch.tensor(cov_k, device=device))
            return torch.stack(new_covs, dim=0)
            
        elif covariance_type == 'tied_full':
            sum_cov = torch.zeros(n_features, n_features, device=device)
            for k in range(n_components):
                cluster_mask = (labels == k)
                cluster_data = data[cluster_mask]
                if cluster_data.size(0) > 0:
                    diff = cluster_data - cluster_data.mean(dim=0, keepdim=True)
                    sum_cov += diff.T @ diff
            sum_cov /= n_samples
            sum_cov += reg_covar * torch.eye(n_features, device=device)
            return sum_cov
            
        elif covariance_type == 'tied_diag':
            sum_diag = torch.zeros(n_features, device=device)
            for k in range(n_components):
                cluster_mask = (labels == k)
                cluster_data = data[cluster_mask]
                if cluster_data.size(0) > 0:
                    diff = cluster_data - cluster_data.mean(dim=0, keepdim=True)
                    sum_diag += (diff * diff).sum(dim=0)
            sum_diag /= n_samples
            sum_diag += reg_covar
            return sum_diag
            
        elif covariance_type == 'tied_spherical':
            total_sum = 0.0
            for k in range(n_components):
                cluster_mask = (labels == k)
                cluster_data = data[cluster_mask]
                if cluster_data.size(0) > 0:
                    diff = cluster_data - cluster_data.mean(dim=0, keepdim=True)
                    total_sum += diff.pow(2).sum().item()
            var = max(total_sum / (n_samples * n_features), reg_covar)
            return torch.tensor(var, device=device)
            
        else:
            raise ValueError(f"Unsupported covariance type: {covariance_type}")
