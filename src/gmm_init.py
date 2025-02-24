import torch

class GMMInitializer:
    r"""
    A utility class providing various initialization strategies for GMM means.

    This class defines several static methods to produce initial means for
    Gaussian Mixture Models from a dataset ``data`` (a 2D tensor of shape (N, D)):

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
      c_j = \arg\min_i \|x_j - \mu_i\|^2, \quad \mu_i = \frac{1}{|C_i|} \sum_{x_j \in C_i} x_j,
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
           $\displaystyle c_j = \arg\min_{i} \|x_j - \mu_i\|^2,$ for each data point $x_j$.
        
        2. **Update:**  
           $\displaystyle \mu_i = \frac{1}{|C_i|} \sum_{x_j \in C_i} x_j,$ where $C_i$ is the set of points assigned to center $\mu_i$.
        
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
