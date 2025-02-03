import torch

class GMMInitializer:
    @staticmethod
    def random(data: torch.Tensor, k: int) -> torch.Tensor:
        """
        Initialise means randomly based on data distribution.
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
        """
        Initialise means by randomly selecting points from the data.
        """
        indices = torch.randperm(data.size(0), device=data.device)[:k]
        return data[indices]

    @staticmethod
    def kpp(data: torch.Tensor, k: int) -> torch.Tensor:
        """
        Initialise means using the k-means++ algorithm.
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

    @staticmethod
    def kmeans(data: torch.Tensor, k: int, max_iter=1000, atol=1e-4) -> torch.Tensor:
        """
        Initialise means using the k-means algorithm.
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
        """
        Initialise means using a modified k-means++ algorithm
        that maximizes the minimum distance between centroids,
        with reevaluation of the first cluster center.
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
