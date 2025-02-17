import torch

class GMMInitializer:
    r"""
    A utility class providing various initialization strategies for GMM means.

    This class defines several static methods to produce initial means for
    Gaussian Mixture Models from a dataset ``data``:

    - :func:`random`
    - :func:`points`
    - :func:`kpp`
    - :func:`kmeans`
    - :func:`maxdist`

    All methods expect a 2D data tensor (N x D) and a number ``k`` of desired
    cluster centers.

    Example usage::

        from myproject.gmm_initializer import GMMInitializer
        
        data = torch.randn(1000, 2)  # Synthetic data
        k = 4
        init_means = GMMInitializer.random(data, k)

    Methods
    -------
    random(data: torch.Tensor, k: int) -> torch.Tensor
        Randomly draw initial means based on the distribution of ``data``.
    points(data: torch.Tensor, k: int) -> torch.Tensor
        Select ``k`` points at random from ``data`` to serve as means.
    kpp(data: torch.Tensor, k: int) -> torch.Tensor
        Use the k-means++ procedure to choose initial means that are far apart.
    kmeans(data: torch.Tensor, k: int, max_iter=1000, atol=1e-4) -> torch.Tensor
        Run the k-means algorithm to find stable initial means.
    maxdist(data: torch.Tensor, k: int) -> torch.Tensor
        A modified k-means++ that maximizes the minimum distance between centroids
        and reevaluates the first cluster center.
    """

    @staticmethod
    def random(data: torch.Tensor, k: int) -> torch.Tensor:
        r"""
        Randomly initialize cluster centers by sampling from the empirical
        distribution of ``data``.

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor of shape (N, D) representing the dataset.
        k : int
            Number of cluster centers to generate.

        Returns
        -------
        means : torch.Tensor
            A (k, D) tensor representing the initial cluster centers.
        """
        mu = torch.mean(data, dim=0)
        if data.dim() == 1:
            # 1D data case
            cov = torch.var(data)
            samples = torch.randn(k, device=data.device) * torch.sqrt(cov)
        else:
            # 2D or higher
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
        means : torch.Tensor
            A (k, D) tensor representing the initial cluster centers.
        """
        indices = torch.randperm(data.size(0), device=data.device)[:k]
        return data[indices]

    @staticmethod
    def kpp(data: torch.Tensor, k: int) -> torch.Tensor:
        r"""
        Initialize cluster centers using the k-means++ algorithm.

        The first center is chosen uniformly at random from ``data``.
        Subsequent centers are chosen with probability proportional to
        their squared distance from the nearest existing center.

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor of shape (N, D) representing the dataset.
        k : int
            Number of cluster centers to generate.

        Returns
        -------
        means : torch.Tensor
            A (k, D) tensor representing the initial cluster centers.
        """
        n_samples, _ = data.shape
        centroids = torch.empty((k, data.size(1)), device=data.device)

        # Pick the first center
        initial_idx = torch.randint(0, n_samples, (1,), device=data.device)
        centroids[0] = data[initial_idx]

        for i in range(1, k):
            # Compute distances to the nearest center
            dist_sq = torch.cdist(data, centroids[:i]).pow(2).min(dim=1)[0]
            probabilities = dist_sq / dist_sq.sum()
            selected_idx = torch.multinomial(probabilities, 1)
            centroids[i] = data[selected_idx]

        return centroids

    @staticmethod
    def kmeans(data: torch.Tensor, k: int, max_iter: int = 1000, atol: float = 1e-4) -> torch.Tensor:
        r"""
        Initialize cluster centers by running the k-means algorithm on ``data``.

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor of shape (N, D) representing the dataset.
        k : int
            Number of cluster centers to generate.
        max_iter : int, optional
            Maximum number of iterations to run the k-means refinement.
        atol : float, optional
            Tolerance for convergence (if the centroids move less than this
            amount, the algorithm stops).

        Returns
        -------
        means : torch.Tensor
            A (k, D) tensor representing the final cluster centers after
            k-means refinement.
        """
        # Start with k-means++ initialization
        centroids = GMMInitializer.kpp(data, k)

        for _ in range(max_iter):
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)

            new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(k)])

            # Check if converged
            if torch.allclose(centroids, new_centroids, atol=atol):
                break
            centroids = new_centroids

        return centroids

    @staticmethod
    def maxdist(data: torch.Tensor, k: int) -> torch.Tensor:
        r"""
        A modified k-means++ that aims to maximize the minimum distance
        between chosen cluster centers. After selecting the first center
        randomly, each subsequent center is the point in ``data`` with
        the largest distance to any existing center. Finally, the
        first center is reevaluated based on the new centers.

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor of shape (N, D).
        k : int
            Number of cluster centers.

        Returns
        -------
        means : torch.Tensor
            A (k, D) tensor of initial cluster centers.
        """
        n_samples, _ = data.shape
        centroids = torch.empty((k, data.size(1)), device=data.device)

        # Pick the first center randomly
        initial_idx = torch.randint(0, n_samples, (1,), device=data.device)
        centroids[0] = data[initial_idx]

        # Pick subsequent centers by maximum distance
        for i in range(1, k):
            dist_sq = torch.cdist(data, centroids[:i]).pow(2)
            min_dist = dist_sq.min(dim=1)[0]
            selected_idx = torch.argmax(min_dist)
            centroids[i] = data[selected_idx]

        # Reevaluate the first center based on new sets of centers
        dist_sq_to_first = torch.cdist(data, centroids[1:]).pow(2)
        min_dist_to_first = dist_sq_to_first.min(dim=1)[0]
        new_first_idx = torch.argmax(min_dist_to_first)
        centroids[0] = data[new_first_idx]

        return centroids
