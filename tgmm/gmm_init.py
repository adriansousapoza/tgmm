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
    - :func:`hdbscan` (and :func:`hdbscan_clusters`, which also returns the cluster labels)

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

    - **hdbscan:**
      Runs :class:`~tgmm.hdbscan.HDBSCAN` on the data and uses the mean of each cluster it finds as a
      centre, $\mu_j = \frac{1}{|C_j|}\sum_{x\in C_j}x$ (noise points are ignored). It needs no ``k`` to find the
      clusters, so it is the one method that can discover non-convex or very unequal clusters; ``k`` only decides how
      many of them are kept (the ``k`` largest) and how many extra centres have to be added (k-means++ seeding on the
      points no cluster claimed). The same clustering also gives the starting weights (cluster sizes) and covariances
      (cluster scatter) via :func:`init_weights_from_labels` and the ``labels`` argument of
      :func:`init_covariances_empirical`.

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
            samples = torch.randn(k, device=data.device, dtype=data.dtype) * torch.sqrt(cov)
        else:
            cov = torch.cov(data.t())
            samples = torch.randn(k, data.size(1), device=data.device, dtype=data.dtype) @ torch.linalg.cholesky(cov).t()
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
    def kpp(data: torch.Tensor, k: int, random_state: int = None) -> torch.Tensor:
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
        random_state : int, optional
            Random seed for reproducibility. If None, randomness is not controlled.

        Returns
        -------
        torch.Tensor
            A (k, D) tensor representing the initial cluster centers.
        """
        n_samples, _ = data.shape
        centroids = torch.empty((k, data.size(1)), device=data.device, dtype=data.dtype)

        # Create a generator if random_state is provided
        generator = None
        if random_state is not None:
            generator = torch.Generator(device=data.device)
            generator.manual_seed(random_state)

        # Pick the first center uniformly at random
        initial_idx = torch.randint(0, n_samples, (1,), device=data.device, generator=generator)
        centroids[0] = data[initial_idx]

        for i in range(1, k):
            dist_sq = torch.cdist(data, centroids[:i]).pow(2).min(dim=1)[0]
            total = dist_sq.sum()
            if bool(total > 0):
                probabilities = dist_sq / total
            else:  # every point coincides with a chosen center (k > number of distinct points): pick uniformly
                probabilities = torch.full_like(dist_sq, 1.0 / n_samples)
            selected_idx = torch.multinomial(probabilities, 1, generator=generator)
            centroids[i] = data[selected_idx]

        return centroids

    @staticmethod
    def kmeans(data: torch.Tensor, k: int, max_iter: int = 1000, atol: float = 1e-4, random_state: int = None) -> torch.Tensor:
        r"""
        Initialize cluster centers by running the k-means algorithm on ``data``.

        Starting from a k-means++ initialization, k-means iteratively refines the centers by:

        1. **Assignment:**
           $c_j = \arg\min_{i} \|x_j - \mu_i\|^2,$ for each data point $x_j$.

        2. **Update:**
           $\mu_i = \frac{1}{\|C_i\|} \sum_{x_j \in C_i} x_j,$ where $C_i$ is the set of points assigned to center $\mu_i$.
           A cluster that ends up empty is re-seeded at the data point farthest from its assigned center.

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
        random_state : int, optional
            Random seed for reproducibility. If None, randomness is not controlled.

        Returns
        -------
        torch.Tensor
            A (k, D) tensor representing the final cluster centers.
        """
        centroids = GMMInitializer.kpp(data, k, random_state=random_state)

        for _ in range(max_iter):
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)
            means = []
            empty = []
            for i in range(k):
                member = labels == i
                if bool(member.any()):
                    means.append(data[member].mean(dim=0))
                else:  # an empty cluster has no mean (0/0 -> NaN): re-seed it below
                    means.append(None)
                    empty.append(i)
            if empty:
                # Re-seed each empty cluster at the points farthest from their assigned centroid
                # (the standard remedy; leaves results unchanged whenever no cluster is empty).
                assigned = distances[torch.arange(data.shape[0], device=data.device), labels]
                order = torch.argsort(assigned, descending=True)
                for j, i in enumerate(empty):
                    means[i] = data[order[j % data.shape[0]]].clone()
            new_centroids = torch.stack(means)
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
        centroids = torch.empty((k, data.size(1)), device=data.device, dtype=data.dtype)

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
    def _validate_hdbscan_kwargs(hdbscan_kwargs):
        r"""
        Check the ``hdbscan_kwargs`` argument of :class:`~tgmm.gmm.GaussianMixture` and return a plain ``dict`` copy
        (``None`` stays ``None``). Allowed keys: ``min_cluster_size``, ``min_samples``, ``max_points`` and every other
        argument of :class:`~tgmm.hdbscan.HDBSCAN` except ``device`` (the clustering runs where the data is).
        """
        if hdbscan_kwargs is None:
            return None
        if not isinstance(hdbscan_kwargs, dict):
            raise TypeError(f"hdbscan_kwargs must be a dict or None, got {type(hdbscan_kwargs).__name__}.")
        import inspect

        from .hdbscan import HDBSCAN

        allowed = [p for p in inspect.signature(HDBSCAN.__init__).parameters if p not in ('self', 'device')]
        allowed.append('max_points')
        unknown = sorted(set(hdbscan_kwargs) - set(allowed), key=str)
        if unknown:
            raise ValueError(f"unknown hdbscan_kwargs key(s) {unknown}; allowed keys: {sorted(allowed)}.")
        kw = dict(hdbscan_kwargs)
        mcs, ms, mp = kw.get('min_cluster_size'), kw.get('min_samples'), kw.get('max_points')
        if mcs is not None and (isinstance(mcs, bool) or int(mcs) != mcs or int(mcs) < 2):
            raise ValueError(f"hdbscan_kwargs['min_cluster_size'] must be an integer >= 2, got {mcs!r}.")
        if ms is not None and (isinstance(ms, bool) or int(ms) != ms or int(ms) < 1):
            raise ValueError(f"hdbscan_kwargs['min_samples'] must be an integer >= 1, got {ms!r}.")
        if mp is not None and (isinstance(mp, bool) or int(mp) != mp or int(mp) < 2):
            raise ValueError(f"hdbscan_kwargs['max_points'] must be None or an integer >= 2, got {mp!r}.")
        return kw

    @staticmethod
    def _hdbscan_default_min_cluster_size(n: int, k: int) -> int:
        r"""
        Default ``min_cluster_size`` of the HDBSCAN initialisers: $\max\{10,\ \min\{\lfloor 0.02\,n\rfloor,\ \lfloor n/(5k)\rfloor\}\}$.

        2% of the points keeps HDBSCAN from calling small clumps clusters (and finds the four blobs of a typical
        2-D example), while $n/(5k)$ keeps it well below the size of a typical cluster ($\approx n/k$ points) when
        ``k`` is large, so that no cluster is lost to the "noise" label.
        """
        return max(10, min(int(0.02 * n), n // (5 * k)))

    @staticmethod
    def _hdbscan_partition(data: torch.Tensor, k: int, min_cluster_size=None, min_samples=None,
                           max_points: int = 5000, random_state: int = None, **hdbscan_kwargs):
        r"""
        The deterministic part of :func:`hdbscan_clusters`: cluster, keep the ``k`` largest clusters and return
        ``(labels, centres)`` with ``labels`` of shape (N,) in ``{-1, 0, ..., m-1}`` (``m <= k`` kept clusters, ordered by
        decreasing size; ``-1`` = noise or a dropped cluster) and ``centres`` of shape (m, D). ``m = 0`` when HDBSCAN
        finds no cluster. Contains no randomness once the (optional) subsample is fixed, so callers may cache it.
        """
        from .hdbscan import HDBSCAN

        n_samples, n_features = data.shape

        # HDBSCAN builds the full N x N distance matrix: for large N cluster a random subsample and give every other
        # point the label of its nearest subsampled point.
        if max_points is not None and n_samples > max_points:
            seed = random_state if random_state is not None else int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
            gen = torch.Generator(device='cpu')
            gen.manual_seed(int(seed))
            idx = torch.randperm(n_samples, generator=gen)[:int(max_points)].to(data.device)
            sub = data[idx]
        else:
            idx = None
            sub = data
        n_sub = sub.shape[0]

        if min_cluster_size is None:
            mcs = GMMInitializer._hdbscan_default_min_cluster_size(n_sub, k)
        else:
            mcs = max(2, int(round(int(min_cluster_size) * n_sub / n_samples)))
        if min_samples is None:
            min_samples = max(2, min(mcs, max(5, mcs // 2)))
        else:
            min_samples = int(min_samples)
            if idx is not None:
                min_samples = max(1, int(round(min_samples * n_sub / n_samples)))

        model = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, device=str(data.device), **hdbscan_kwargs)
        sub_labels = model.fit(sub).labels_.to(data.device)

        valid = sub_labels[sub_labels >= 0]
        if valid.numel() == 0:
            return torch.full((n_samples,), -1, dtype=torch.long, device=data.device), data.new_zeros((0, n_features))

        sizes = torch.bincount(valid)
        order = torch.argsort(sizes, descending=True, stable=True)[:k]          # the k largest clusters
        remap = torch.full((int(sizes.numel()),), -1, dtype=torch.long, device=data.device)
        remap[order] = torch.arange(order.numel(), device=data.device)
        sub_labels = torch.where(sub_labels >= 0, remap[sub_labels.clamp(min=0)], sub_labels)

        if idx is None:
            labels = sub_labels
        else:                                                                   # nearest subsampled point's label
            labels = torch.empty(n_samples, dtype=torch.long, device=data.device)
            for start in range(0, n_samples, 4096):
                chunk = data[start:start + 4096]
                labels[start:start + 4096] = sub_labels[torch.cdist(chunk, sub).argmin(dim=1)]
            labels[idx] = sub_labels

        m = int(order.numel())
        centres = torch.stack([data[labels == j].mean(dim=0) for j in range(m)])
        return labels, centres

    @staticmethod
    def _hdbscan_complete(data: torch.Tensor, labels: torch.Tensor, centres: torch.Tensor, k: int,
                          random_state: int = None) -> torch.Tensor:
        r"""
        Complete ``centres`` (m <= k rows) to ``k`` rows: the missing centres are drawn by k-means++ (probability
        proportional to the squared distance to the nearest existing centre) among the points no kept cluster claimed
        (noise first; all points if there is none). With ``m = 0`` (no cluster found) this is plain k-means++ on the data.
        """
        m = centres.shape[0]
        if m >= k:
            return centres[:k]
        if m == 0:
            return GMMInitializer.kpp(data, k, random_state=random_state)
        seed = random_state if random_state is not None else int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(seed))
        pool = data[labels < 0] if bool((labels < 0).any()) else data
        out = centres
        for _ in range(k - m):
            d2 = torch.cdist(pool, out).pow(2).min(dim=1)[0]
            if not bool(d2.sum() > 0):                      # the pool is already covered: draw from all the data
                pool = data
                d2 = torch.cdist(pool, out).pow(2).min(dim=1)[0]
            if bool(d2.sum() > 0):
                probs = (d2 / d2.sum()).cpu()
            else:                                           # every point coincides with a centre
                probs = torch.full((pool.shape[0],), 1.0 / pool.shape[0])
            j = int(torch.multinomial(probs.double(), 1, generator=gen).item())
            out = torch.cat([out, pool[j:j + 1]], dim=0)
        return out

    @staticmethod
    def hdbscan_clusters(data: torch.Tensor, k: int, min_cluster_size: int = None, min_samples: int = None,
                         max_points: int = 5000, random_state: int = None, **hdbscan_kwargs):
        r"""
        Initial centres from an HDBSCAN clustering, together with the clustering itself.

        :class:`~tgmm.hdbscan.HDBSCAN` is run on ``data`` (it needs no number of clusters) and every cluster it finds
        contributes its mean as a centre; noise points (label ``-1``) are ignored:
        $$
        \mu_j = \frac{1}{|C_j|}\sum_{x\in C_j} x .
        $$
        Handling of ``k``, with ``m`` the number of clusters found:

        - ``m > k``: the ``k`` largest clusters are kept (the points of the others become noise).
        - ``m < k``: the missing ``k - m`` centres are drawn by k-means++ seeding (probability proportional to the squared
          distance to the nearest existing centre) among the points no kept cluster claimed, noise points first.
        - ``m = 0`` (no cluster found, e.g. uniform data): the result is plain k-means++, see :func:`kpp`.

        Clusters are ordered by decreasing size, so row ``j`` of the centres is cluster ``j``; the extra centres come last
        and own no points. With ``m >= k`` the result is deterministic (and ``n_init > 1`` repeats the same start); the
        randomness enters only through the subsampling (see ``max_points``) and the k-means++ completion.

        Strengths and weaknesses: it finds non-convex and very unequal clusters and shrugs off outliers (they are noise),
        so it is the robust alternative to :func:`maxdist`; but it is a density method, so on heavily overlapping
        clusters (a Gaussian blob mixture with no density gaps) it may return a single cluster or none, and the
        centres are then only as good as the k-means++ completion. HDBSCAN needs the full ``N x N`` distance matrix
        (``O(N^2)`` memory), hence ``max_points``.

        Parameters
        ----------
        data : torch.Tensor
            Data of shape (N, D) (or (N,), treated as one feature).
        k : int
            Number of centres to return.
        min_cluster_size : int, optional
            Smallest group HDBSCAN accepts as a cluster. Default
            ``max(10, min(int(0.02 * N), N // (5 * k)))``: 2% of the points, but small enough for ``k`` clusters of
            about ``N / k`` points to survive. If ``data`` is subsampled it refers to the full data set and is scaled by
            the subsampling fraction.
        min_samples : int, optional
            HDBSCAN's core-point neighbourhood size. Default ``max(5, min_cluster_size // 2)`` (never above
            ``min_cluster_size``); a value you pass is scaled like ``min_cluster_size`` when the data is subsampled.
        max_points : int or None, default=5000
            If ``N > max_points`` the clustering runs on a random subsample of ``max_points`` points and every other point
            takes the label of its nearest subsampled point. ``None`` clusters all the data.
        random_state : int, optional
            Seed for the subsample and the k-means++ completion (the global torch RNG is used when ``None``).
        **hdbscan_kwargs
            Further keyword arguments of :class:`~tgmm.hdbscan.HDBSCAN` (``cluster_selection_method``,
            ``cluster_selection_epsilon``, ``metric``, ...).

        Returns
        -------
        means : torch.Tensor
            (k, D) centres, same device and dtype as ``data``.
        labels : torch.Tensor
            (N,) integer labels in ``{-1, 0, ..., k-1}``: the cluster of each point (``-1`` = noise, or a cluster
            that was dropped because more than ``k`` were found). Rows of ``means`` without members have no label.
        """
        if data.dim() == 1:
            data = data.unsqueeze(1)
        if data.dim() != 2:
            raise ValueError(f"data must be a 2D tensor of shape (N, D), got {tuple(data.shape)}.")
        if isinstance(k, bool) or int(k) != k or int(k) < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}.")
        k = int(k)
        if data.shape[0] < k:
            raise ValueError(f"cannot initialise k={k} centres from {data.shape[0]} points.")
        if min_cluster_size is not None and int(min_cluster_size) < 2:
            raise ValueError(f"min_cluster_size must be an integer >= 2, got {min_cluster_size!r}.")
        if min_samples is not None and int(min_samples) < 1:
            raise ValueError(f"min_samples must be an integer >= 1, got {min_samples!r}.")
        if max_points is not None and int(max_points) < 2:
            raise ValueError(f"max_points must be None or an integer >= 2, got {max_points!r}.")
        labels, centres = GMMInitializer._hdbscan_partition(
            data, k, min_cluster_size, min_samples, max_points, random_state, **hdbscan_kwargs)
        means = GMMInitializer._hdbscan_complete(data, labels, centres, k, random_state)
        return means.to(dtype=data.dtype), labels

    @staticmethod
    def hdbscan(data: torch.Tensor, k: int, min_cluster_size: int = None, min_samples: int = None,
                max_points: int = 5000, random_state: int = None, **hdbscan_kwargs) -> torch.Tensor:
        r"""
        Initial centres from an HDBSCAN clustering (the means of the clusters it finds).

        A thin wrapper of :func:`hdbscan_clusters` that returns only the ``(k, D)`` centres; see there for the handling of
        ``k``, the default ``min_cluster_size``, the subsampling and the other keyword arguments. Used by
        ``GaussianMixture(init_means='hdbscan')``.

        Returns
        -------
        torch.Tensor
            A (k, D) tensor of initial centres.
        """
        return GMMInitializer.hdbscan_clusters(
            data, k, min_cluster_size=min_cluster_size, min_samples=min_samples,
            max_points=max_points, random_state=random_state, **hdbscan_kwargs)[0]

    @staticmethod
    def init_weights_uniform(n_components: int, device: torch.device,
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
        r"""
        Initialize weights uniformly (equal weights for all components).

        Parameters
        ----------
        n_components : int
            Number of mixture components.
        device : torch.device
            Device to create the tensor on.
        dtype : torch.dtype, optional
            Floating point dtype for the tensor. (default: torch.float32)

        Returns
        -------
        torch.Tensor
            Uniform weights of shape (n_components,).
        """
        return torch.full(
            (n_components,),
            1.0 / n_components,
            dtype=dtype,
            device=device
        )

    @staticmethod
    def init_weights_random(n_components: int, device: torch.device,
                             dtype: torch.dtype = torch.float32) -> torch.Tensor:
        r"""
        Initialize weights randomly from a Dirichlet distribution.

        Parameters
        ----------
        n_components : int
            Number of mixture components.
        device : torch.device
            Device to create the tensor on.
        dtype : torch.dtype, optional
            Floating point dtype for the tensor. (default: torch.float32)

        Returns
        -------
        torch.Tensor
            Random weights of shape (n_components,) that sum to 1.
        """
        alpha = torch.ones(n_components, device=device, dtype=dtype)
        weights = torch.distributions.Dirichlet(alpha).sample()
        return weights.to(dtype)
    
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
        counts = torch.bincount(labels, minlength=n_components).to(data.dtype)
        
        # Handle empty clusters
        counts = torch.clamp(counts, min=1e-10)
        
        # Normalize to get weights
        return counts / counts.sum()

    @staticmethod
    def init_weights_from_labels(data: torch.Tensor, means: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        r"""
        Initialize weights proportionally to the sizes of the given hard clusters.

        ``labels`` assigns points to components ``0..K-1`` (``-1`` = unassigned, as returned by
        :func:`hdbscan_clusters`); unassigned points are given to their nearest mean, so the weights describe all the data:
        $$
        w_j = \frac{|C_j| + |\{x \text{ unassigned}: j = \arg\min_l \|x-\mu_l\|\}|}{N}.
        $$

        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (n_samples, n_features).
        means : torch.Tensor
            Component means of shape (n_components, n_features).
        labels : torch.Tensor
            Integer labels of shape (n_samples,) in ``{-1, 0, ..., n_components-1}``.

        Returns
        -------
        torch.Tensor
            Weights of shape (n_components,) that sum to 1.
        """
        if data.dim() == 1:
            data = data.unsqueeze(1)
        n_components = means.size(0)
        labels = labels.to(data.device)
        counts = torch.bincount(labels[labels >= 0], minlength=n_components).to(data.dtype)
        free = labels < 0
        if bool(free.any()):
            nearest = torch.cdist(data[free], means.to(data.device)).argmin(dim=1)
            counts = counts + torch.bincount(nearest, minlength=n_components).to(data.dtype)
        counts = torch.clamp(counts, min=1e-10)
        return counts / counts.sum()

    # ====================================================================
    # Covariance Initialization Methods
    # ====================================================================
    
    @staticmethod
    def init_covariances_eye(n_components: int, n_features: int,
                            covariance_type: str, reg_covar: float,
                            device: torch.device,
                            dtype: torch.dtype = torch.float32) -> torch.Tensor:
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
        dtype : torch.dtype, optional
            Floating point dtype for the tensors. (default: torch.float32)

        Returns
        -------
        torch.Tensor
            Initialized covariances with appropriate shape.
        """
        if covariance_type == 'full':
            out = []
            for _ in range(n_components):
                mat = torch.eye(n_features, device=device, dtype=dtype) * (1.0 + reg_covar)
                out.append(mat)
            return torch.stack(out, dim=0)
        elif covariance_type == 'diag':
            return torch.ones(n_components, n_features, device=device, dtype=dtype) * (1.0 + reg_covar)
        elif covariance_type == 'spherical':
            return torch.ones(n_components, device=device, dtype=dtype) * (1.0 + reg_covar)
        elif covariance_type == 'tied_full':
            return torch.eye(n_features, device=device, dtype=dtype) * (1.0 + reg_covar)
        elif covariance_type == 'tied_diag':
            return torch.ones(n_features, device=device, dtype=dtype) * (1.0 + reg_covar)
        elif covariance_type == 'tied_spherical':
            return torch.tensor(1.0 + reg_covar, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported covariance type: {covariance_type}")
    
    @staticmethod
    def init_covariances_random(n_components: int, n_features: int,
                               covariance_type: str, reg_covar: float,
                               device: torch.device,
                               dtype: torch.dtype = torch.float32) -> torch.Tensor:
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
        dtype : torch.dtype, optional
            Floating point dtype for the tensors. (default: torch.float32)

        Returns
        -------
        torch.Tensor
            Random covariances with appropriate shape.
        """
        if covariance_type in ('full', 'tied_full'):
            def random_spd(dim):
                A = torch.randn(dim, dim, device=device, dtype=dtype)
                return A @ A.mT + reg_covar * torch.eye(dim, device=device, dtype=dtype)

            if covariance_type == 'full':
                return torch.stack([random_spd(n_features) for _ in range(n_components)], dim=0)
            else:
                return random_spd(n_features)
        elif covariance_type in ('diag', 'tied_diag'):
            shape = (n_components, n_features) if covariance_type == 'diag' else (n_features,)
            return torch.rand(shape, device=device, dtype=dtype) + reg_covar
        elif covariance_type in ('spherical', 'tied_spherical'):
            shape = (n_components,) if (covariance_type == 'spherical') else ()
            return torch.rand(shape, device=device, dtype=dtype) + (1.0 + reg_covar)
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
        global_cov += reg_covar * torch.eye(n_features, device=device, dtype=data.dtype)

        if covariance_type == 'full':
            return global_cov.unsqueeze(0).repeat(n_components, 1, 1)
        elif covariance_type == 'diag':
            diag_vals = torch.diagonal(global_cov)
            return diag_vals.unsqueeze(0).repeat(n_components, 1)
        elif covariance_type == 'spherical':
            avg_var = torch.mean(torch.diagonal(global_cov))
            return torch.full((n_components,), avg_var.item(), device=device, dtype=data.dtype)
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
                                  covariance_type: str, reg_covar: float,
                                  labels: torch.Tensor = None) -> torch.Tensor:
        r"""
        Initialize covariances empirically from cluster assignments.
        
        Assigns each data point to its nearest mean (or uses the given ``labels``) and computes cluster-wise
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
        labels : torch.Tensor, optional
            Integer assignment of every point in ``{-1, 0, ..., n_components-1}``. Points labelled ``-1`` are left out of
            every covariance (HDBSCAN's noise, see :func:`hdbscan_clusters`). Default: nearest mean.
            
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
        
        # Assign points to nearest mean (unless the assignment is given)
        if labels is None:
            distances = torch.cdist(data, means)
            labels = torch.argmin(distances, dim=1)
        else:
            labels = labels.to(device)
        n_assigned = max(int((labels >= 0).sum().item()), 1)   # all the points, except those the caller left out (-1)

        # A singleton cluster has no sample covariance (torch.cov divides by n - 1 = 0 -> NaN); such clusters
        # fall back to the covariance of the whole data set (clusters with >= 2 points are unchanged).
        counts = torch.bincount(labels[labels >= 0], minlength=n_components)
        if bool(((counts == 1)).any()) and n_samples > 1:
            global_cov = torch.cov(data.T)
            if global_cov.ndim == 0:
                global_cov = global_cov.reshape(1, 1)
        else:
            global_cov = None

        if covariance_type == 'full':
            new_covs = []
            for k in range(n_components):
                cluster_mask = (labels == k)
                if not torch.any(cluster_mask):
                    cov_k = torch.eye(n_features, device=device, dtype=data.dtype) * (1.0 + reg_covar)
                elif counts[k] == 1 and global_cov is not None:
                    cov_k = global_cov.clone() + reg_covar * torch.eye(n_features, device=device, dtype=data.dtype)
                else:
                    cluster_data = data[cluster_mask]
                    cov_k = torch.cov(cluster_data.T)
                    # Ensure cov_k is always a matrix (for n_features=1, torch.cov returns scalar)
                    if cov_k.ndim == 0:
                        cov_k = cov_k.reshape(1, 1)
                    cov_k += reg_covar * torch.eye(n_features, device=device, dtype=data.dtype)
                new_covs.append(cov_k)
            return torch.stack(new_covs, dim=0)

        elif covariance_type == 'diag':
            new_covs = []
            for k in range(n_components):
                cluster_mask = (labels == k)
                if not torch.any(cluster_mask):
                    cov_k = torch.ones(n_features, device=device, dtype=data.dtype) * (1.0 + reg_covar)
                elif counts[k] == 1 and global_cov is not None:
                    cov_k = torch.diagonal(global_cov).clone() + reg_covar
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
                elif counts[k] == 1 and global_cov is not None:
                    cov_k = max(torch.mean(torch.diagonal(global_cov)).item(), reg_covar)
                else:
                    cluster_data = data[cluster_mask]
                    cov_mat = torch.cov(cluster_data.T)
                    # For n_features=1, torch.cov returns a scalar
                    if cov_mat.ndim == 0:
                        cov_k = max(cov_mat.item(), reg_covar)
                    else:
                        cov_k = max(torch.mean(torch.diagonal(cov_mat)).item(), reg_covar)
                new_covs.append(torch.tensor(cov_k, device=device, dtype=data.dtype))
            return torch.stack(new_covs, dim=0)

        elif covariance_type == 'tied_full':
            sum_cov = torch.zeros(n_features, n_features, device=device, dtype=data.dtype)
            for k in range(n_components):
                cluster_mask = (labels == k)
                cluster_data = data[cluster_mask]
                if cluster_data.size(0) > 0:
                    diff = cluster_data - cluster_data.mean(dim=0, keepdim=True)
                    sum_cov += diff.T @ diff
            sum_cov /= n_assigned
            sum_cov += reg_covar * torch.eye(n_features, device=device, dtype=data.dtype)
            return sum_cov

        elif covariance_type == 'tied_diag':
            sum_diag = torch.zeros(n_features, device=device, dtype=data.dtype)
            for k in range(n_components):
                cluster_mask = (labels == k)
                cluster_data = data[cluster_mask]
                if cluster_data.size(0) > 0:
                    diff = cluster_data - cluster_data.mean(dim=0, keepdim=True)
                    sum_diag += (diff * diff).sum(dim=0)
            sum_diag /= n_assigned
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
            var = max(total_sum / (n_assigned * n_features), reg_covar)
            return torch.tensor(var, device=device, dtype=data.dtype)
            
        else:
            raise ValueError(f"Unsupported covariance type: {covariance_type}")
