"""
PyTorch implementation of HDBSCAN (Hierarchical Density-Based Spatial Clustering 
of Applications with Noise).

This implementation closely follows the scikit-learn-contrib/hdbscan library
(BSD 3-Clause, https://github.com/scikit-learn-contrib/hdbscan) but uses PyTorch for
GPU acceleration. The condensed-tree construction, the cluster selection
(``'eom'`` and ``'leaf'``, ``cluster_selection_epsilon``, ``max_cluster_size``,
``allow_single_cluster``) and the labelling rules are adapted from that library's
``_hdbscan_tree.pyx`` (Copyright (c) 2015, Leland McInnes, BSD 3-Clause), re-derived
in NumPy/PyTorch; see the notes on ``min_samples`` in :class:`HDBSCAN` for the one
deliberate convention difference.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm


def prim_mst(distance_matrix: torch.Tensor) -> np.ndarray:
    r"""
    Minimum spanning tree over a dense `(n, n)` distance matrix, via Prim's
    algorithm starting from node 0.

    Factored out of `HDBSCAN._build_linkage_tree` (same algorithm, same
    code, zero behaviour change there) so
    `tgmm.metrics.ClusteringMetrics.dbcv` can reuse it for its per-cluster
    minimum spanning trees (Moulavi et al. 2014) without duplicating a
    Prim's-algorithm implementation.

    Parameters
    ----------
    distance_matrix : torch.Tensor
        Dense pairwise distances, shape (n, n).

    Returns
    -------
    mst_edges : numpy.ndarray
        Shape `(n - 1, 3)`, each row `[node_a, node_b, weight]`, in the
        order Prim's algorithm added them (**not** sorted by weight --
        callers that need a sorted MST, like `_build_linkage_tree`, sort
        it themselves).
    """
    device = distance_matrix.device
    n_samples = distance_matrix.shape[0]
    if n_samples < 2:
        return np.zeros((0, 3), dtype=np.float64)

    in_tree = torch.zeros(n_samples, dtype=torch.bool, device=device)
    min_distances = torch.full((n_samples,), float('inf'), dtype=distance_matrix.dtype, device=device)
    predecessors = torch.zeros(n_samples, dtype=torch.long, device=device)

    current_node = 0
    in_tree[0] = True

    mst_edges = []

    for _ in range(n_samples - 1):
        distances_from_current = distance_matrix[current_node]
        mask = ~in_tree
        update_mask = mask & (distances_from_current < min_distances)
        min_distances[update_mask] = distances_from_current[update_mask]
        predecessors[update_mask] = current_node

        temp_distances = min_distances.clone()
        temp_distances[in_tree] = float('inf')
        next_node = torch.argmin(temp_distances).item()

        mst_edges.append([
            predecessors[next_node].item(),
            next_node,
            min_distances[next_node].item()
        ])

        in_tree[next_node] = True
        current_node = next_node

    return np.array(mst_edges, dtype=np.float64)


class HDBSCAN:
    """
    HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications with Noise.
    
    PyTorch implementation that closely follows sklearn-contrib/hdbscan.
    
    Parameters
    ----------
    min_cluster_size : int, default=10
        The minimum number of samples in a group for that group to be considered 
        a cluster; groupings smaller than this size will be left as noise.
        
    min_samples : int, default=None
        The number of samples in a neighborhood for a point to be considered as 
        a core point. This includes the point itself (as in Campello et al.,
        2013), so the core distance is the distance to the ``min_samples``-th
        nearest point counting the point itself. When None, defaults to 
        min_cluster_size.

        .. note::
            scikit-learn-contrib/hdbscan counts neighbours *excluding* the point
            itself, so ``HDBSCAN(min_samples=k)`` here corresponds to
            ``hdbscan.HDBSCAN(min_samples=k - 1)`` there (the reference's own
            ``match_reference_implementation=True`` applies the same shift).
        
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances.
        Supported: 'euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'.
        
    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.
        
    allow_single_cluster : bool, default=False
        By default HDBSCAN will not produce a single cluster. Setting this to 
        True will override this and allow single cluster results. When the
        root of the condensed tree is the only selected cluster, a point is
        labelled with it only if it was still in the root at a density of at
        least ``1 / cluster_selection_epsilon`` (when the epsilon is non-zero),
        or at least the density at which the root's last member left it
        (otherwise); the remaining points are noise. (This is the reference
        implementation's rule.)
        
    cluster_selection_epsilon : float, default=0.0
        A distance threshold below which cluster *splits* are ignored: any
        selected cluster that was born at a distance smaller than this value
        is replaced by its first ancestor born at a distance of at least this
        value. ``0.0`` (default) disables the post-processing, so the result
        is the plain EOM / leaf selection. Must be ``>= 0``. When the only
        remaining candidate is the root of the tree it is kept only if
        ``allow_single_cluster=True``.
        
    max_cluster_size : int, default=0
        A limit to the size of clusters returned by the 'eom' algorithm
        (``0`` = no limit). With ``allow_single_cluster=True`` the limit also
        applies to the root, whose size is the total size of its child
        clusters.

    cluster_selection_method : {'eom', 'leaf'}, default='eom'
        How flat clusters are selected from the condensed tree: ``'eom'``
        (excess of mass, default) maximises total stability, ``'leaf'``
        selects every leaf of the condensed cluster tree (finer, more
        homogeneous clusters). ``cluster_selection_epsilon`` can be combined
        with either.
        
    device : str, default='cpu'
        Device to use for computation ('cpu' or 'cuda').
        
    p : float, default=2.0
        Parameter for Minkowski metric.
        
    Attributes
    ----------
    labels_ : torch.Tensor of shape (n_samples,)
        Cluster labels for each point. Noisy samples are labeled as -1.
        
    probabilities_ : torch.Tensor of shape (n_samples,)
        The strength with which each sample is a member of its assigned cluster.
        Noisy samples have probability zero.
        
    n_clusters_ : int
        The number of clusters found.
    """
    
    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: Optional[int] = None,
        metric: str = 'euclidean',
        alpha: float = 1.0,
        allow_single_cluster: bool = False,
        cluster_selection_epsilon: float = 0.0,
        max_cluster_size: int = 0,
        device: str = 'cpu',
        p: float = 2.0,
        cluster_selection_method: str = 'eom',
    ):
        if cluster_selection_method not in ('eom', 'leaf'):
            raise ValueError(
                f"cluster_selection_method must be 'eom' or 'leaf', got {cluster_selection_method!r}"
            )
        if cluster_selection_epsilon < 0:
            raise ValueError(
                f"cluster_selection_epsilon must be >= 0, got {cluster_selection_epsilon}"
            )
        self.cluster_selection_method = cluster_selection_method
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.metric = metric
        self.alpha = alpha
        self.allow_single_cluster = allow_single_cluster
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size if max_cluster_size > 0 else float('inf')
        self.device = device
        self.p = p
        
        # Attributes set during fit
        self.labels_ = None
        self.probabilities_ = None
        self.n_clusters_ = 0
        self._single_linkage_tree = None
        self._condensed_tree = None
        
    def fit(self, X: torch.Tensor) -> 'HDBSCAN':
        """Perform HDBSCAN clustering on X."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to(self.device)
        n_samples = X.shape[0]

        # Fewer than two points: nothing to link (and a core distance needs a neighbour)
        if n_samples < 2:
            self.labels_ = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
            self.probabilities_ = torch.zeros(n_samples, dtype=torch.float32, device=self.device)
            self.n_clusters_ = 0
            self._single_linkage_tree = np.zeros((0, 4), dtype=np.float64)
            self._condensed_tree = np.zeros(0, dtype=[('parent', np.intp), ('child', np.intp),
                                                      ('lambda_val', float), ('child_size', np.intp)])
            return self
        
        # Compute distance matrix
        distance_matrix = self._compute_distances(X)
        
        # Compute mutual reachability
        # min_samples counts the point itself, so it cannot exceed n (the reference clamps it the same way)
        mutual_reachability = self._mutual_reachability(
            distance_matrix, min(self.min_samples, n_samples), self.alpha)
        
        # Build MST and convert to linkage
        self._single_linkage_tree = self._build_linkage_tree(mutual_reachability)
        
        # Condense tree
        self._condensed_tree = self._condense_tree(self._single_linkage_tree, self.min_cluster_size)
        
        # Compute stability
        stability = self._compute_stability(self._condensed_tree)
        
        # Get clusters
        self.labels_, self.probabilities_ = self._get_clusters(
            self._condensed_tree,
            stability,
            self.allow_single_cluster,
            self.cluster_selection_epsilon,
            self.max_cluster_size,
            self.cluster_selection_method
        )
        
        # Count clusters
        unique_labels = torch.unique(self.labels_)
        self.n_clusters_ = (unique_labels >= 0).sum().item()
        
        return self
    
    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """Perform clustering on X and return cluster labels."""
        self.fit(X)
        return self.labels_
    
    def _compute_distances(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances."""
        if self.metric == 'euclidean':
            return torch.cdist(X, X, p=2)
        elif self.metric == 'manhattan':
            return torch.cdist(X, X, p=1)
        elif self.metric == 'chebyshev':
            return torch.cdist(X, X, p=float('inf'))
        elif self.metric == 'minkowski':
            return torch.cdist(X, X, p=self.p)
        elif self.metric == 'cosine':
            X_normalized = X / (X.norm(dim=1, keepdim=True) + 1e-8)
            cosine_sim = torch.mm(X_normalized, X_normalized.t())
            return 1 - cosine_sim
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def _mutual_reachability(self, distance_matrix: torch.Tensor, min_samples: int, alpha: float) -> torch.Tensor:
        """Compute mutual reachability distance matrix."""
        n_samples = distance_matrix.shape[0]
        
        # Get core distances (distance to k-th nearest neighbor)
        sorted_distances, _ = torch.sort(distance_matrix, dim=1)
        core_distances = sorted_distances[:, min_samples - 1]
        
        # Broadcast core distances
        core_i = core_distances.unsqueeze(1).expand(n_samples, n_samples)
        core_j = core_distances.unsqueeze(0).expand(n_samples, n_samples)
        
        # Mutual reachability
        mutual_reachability = torch.max(torch.max(core_i, core_j), distance_matrix)
        
        if alpha != 1.0:
            mutual_reachability = mutual_reachability / alpha
        
        return mutual_reachability
    
    def _build_linkage_tree(self, distance_matrix: torch.Tensor) -> np.ndarray:
        """Build minimum spanning tree and convert to linkage format."""
        n_samples = distance_matrix.shape[0]

        mst = prim_mst(distance_matrix)
        mst = mst[np.argsort(mst[:, 2])]

        # Convert to linkage tree using Union-Find
        linkage_tree = self._mst_to_linkage(mst, n_samples)

        return linkage_tree
    
    def _mst_to_linkage(self, mst: np.ndarray, n_samples: int) -> np.ndarray:
        """Convert MST to scipy-like linkage format using Union-Find."""
        parent = np.arange(2 * n_samples - 1, dtype=np.intp)
        size = np.ones(2 * n_samples - 1, dtype=np.intp)
        
        linkage = np.zeros((n_samples - 1, 4), dtype=np.float64)
        
        for i, (node_a, node_b, distance) in enumerate(mst):
            node_a, node_b = int(node_a), int(node_b)
            
            # Find roots
            root_a = self._find_root(parent, node_a)
            root_b = self._find_root(parent, node_b)
            
            new_node = n_samples + i
            parent[root_a] = new_node
            parent[root_b] = new_node
            
            cluster_size = size[root_a] + size[root_b]
            size[new_node] = cluster_size
            
            linkage[i] = [root_a, root_b, distance, cluster_size]
        
        return linkage
    
    def _find_root(self, parent: np.ndarray, node: int) -> int:
        """Find root in Union-Find structure."""
        while parent[node] != node:
            node = parent[node]
        return node
    
    def _condense_tree(self, hierarchy: np.ndarray, min_cluster_size: int) -> np.ndarray:
        """
        Condense tree by removing small clusters.
        Follows the hdbscan reference implementation exactly.
        """
        root = 2 * hierarchy.shape[0]
        num_points = root // 2 + 1
        next_label = num_points + 1
        
        # BFS from root
        node_list = self._bfs_from_hierarchy(hierarchy, root)
        
        relabel = np.empty(root + 1, dtype=np.intp)
        relabel[root] = num_points
        result_list = []
        ignore = np.zeros(len(node_list), dtype=bool)
        
        for node in node_list:
            if ignore[node] or node < num_points:
                continue
            
            children = hierarchy[node - num_points]
            left = int(children[0])
            right = int(children[1])
            
            if children[2] > 0.0:
                lambda_value = 1.0 / children[2]
            else:
                lambda_value = float('inf')
            
            if left >= num_points:
                left_count = int(hierarchy[left - num_points][3])
            else:
                left_count = 1
            
            if right >= num_points:
                right_count = int(hierarchy[right - num_points][3])
            else:
                right_count = 1
            
            if left_count >= min_cluster_size and right_count >= min_cluster_size:
                # Both children are clusters
                relabel[left] = next_label
                next_label += 1
                result_list.append((relabel[node], relabel[left], lambda_value, left_count))
                
                relabel[right] = next_label
                next_label += 1
                result_list.append((relabel[node], relabel[right], lambda_value, right_count))
                
            elif left_count < min_cluster_size and right_count < min_cluster_size:
                # Both children are too small
                for sub_node in self._bfs_from_hierarchy(hierarchy, left):
                    if sub_node < num_points:
                        result_list.append((relabel[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True
                
                for sub_node in self._bfs_from_hierarchy(hierarchy, right):
                    if sub_node < num_points:
                        result_list.append((relabel[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True
                    
            elif left_count < min_cluster_size:
                # Left is too small, right continues
                relabel[right] = relabel[node]
                for sub_node in self._bfs_from_hierarchy(hierarchy, left):
                    if sub_node < num_points:
                        result_list.append((relabel[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True
                    
            else:
                # Right is too small, left continues
                relabel[left] = relabel[node]
                for sub_node in self._bfs_from_hierarchy(hierarchy, right):
                    if sub_node < num_points:
                        result_list.append((relabel[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True
        
        dtype = np.dtype([('parent', np.intp), ('child', np.intp), 
                         ('lambda_val', float), ('child_size', np.intp)])
        return np.array(result_list, dtype=dtype)
    
    def _bfs_from_hierarchy(self, hierarchy: np.ndarray, bfs_root: int) -> list:
        """Perform BFS on hierarchy tree."""
        dim = hierarchy.shape[0]
        max_node = 2 * dim
        num_points = max_node - dim + 1
        
        to_process = [bfs_root]
        result = []
        
        while to_process:
            result.extend(to_process)
            to_process = [x - num_points for x in to_process if x >= num_points]
            if to_process:
                to_process = hierarchy[to_process, :2].flatten().astype(np.intp).tolist()
        
        return result
    
    def _compute_stability(self, condensed_tree: np.ndarray) -> dict:
        """Compute stability for each cluster."""
        if len(condensed_tree) == 0:
            return {}
        
        largest_child = condensed_tree['child'].max()
        smallest_cluster = condensed_tree['parent'].min()
        num_clusters = condensed_tree['parent'].max() - smallest_cluster + 1
        
        if largest_child < smallest_cluster:
            largest_child = smallest_cluster
        
        # Compute births (minimum lambda for each child)
        sorted_child_data = np.sort(condensed_tree[['child', 'lambda_val']], axis=0)
        births = np.full(largest_child + 1, np.nan, dtype=np.float64)
        
        current_child = -1
        min_lambda = 0
        
        for row in sorted_child_data:
            child = int(row['child'])
            lambda_val = float(row['lambda_val'])
            
            if child == current_child:
                min_lambda = min(min_lambda, lambda_val)
            elif current_child != -1:
                births[current_child] = min_lambda
                current_child = child
                min_lambda = lambda_val
            else:
                current_child = child
                min_lambda = lambda_val
        
        if current_child != -1:
            births[current_child] = min_lambda
        births[smallest_cluster] = 0.0
        
        # Compute stability
        result = np.zeros(num_clusters, dtype=np.float64)
        
        for i in range(len(condensed_tree)):
            parent = condensed_tree['parent'][i]
            lambda_val = condensed_tree['lambda_val'][i]
            child_size = condensed_tree['child_size'][i]
            result_index = parent - smallest_cluster
            
            result[result_index] += (lambda_val - births[parent]) * child_size
        
        stability_dict = {}
        for i, cluster_id in enumerate(range(smallest_cluster, condensed_tree['parent'].max() + 1)):
            stability_dict[cluster_id] = result[i]
        
        return stability_dict
    
    def _get_clusters(
        self,
        tree: np.ndarray,
        stability: dict,
        allow_single_cluster: bool,
        epsilon: float,
        max_cluster_size: float,
        cluster_selection_method: str = 'eom'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select flat clusters from the condensed tree.

        ``'eom'``: bottom-up excess-of-mass selection (a node is kept when it is
        at least as stable as the sum of its children and not larger than
        ``max_cluster_size``); ``'leaf'``: every leaf of the cluster tree. With
        a non-zero ``epsilon`` the selection is post-processed by
        :meth:`_epsilon_search`. Adapted from ``get_clusters`` in
        scikit-learn-contrib/hdbscan (BSD 3-Clause).
        """
        if len(tree) == 0:
            n_samples = len(self._single_linkage_tree) + 1
            labels = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
            probs = torch.zeros(n_samples, dtype=torch.float32, device=self.device)
            return labels, probs

        # Cluster ids are topologically ordered: parents have smaller ids than children
        if allow_single_cluster:
            node_list = sorted(stability.keys(), reverse=True)
        else:
            node_list = sorted(stability.keys(), reverse=True)[:-1]  # Exclude root

        cluster_tree = tree[tree['child_size'] > 1]
        is_cluster = {cluster: True for cluster in node_list}
        root = tree['parent'].min()

        cluster_sizes = {child: child_size for child, child_size in
                         zip(cluster_tree['child'], cluster_tree['child_size'])}
        if allow_single_cluster:
            # The root has no row of its own: its size is that of its child clusters.
            cluster_sizes[node_list[-1]] = int(np.sum(
                cluster_tree[cluster_tree['parent'] == node_list[-1]]['child_size']))

        if cluster_selection_method == 'eom':
            for node in node_list:
                child_selection = cluster_tree['parent'] == node
                subtree_stability = sum(
                    stability[child] for child in cluster_tree['child'][child_selection]
                    if child in stability
                )

                if (subtree_stability > stability[node] or
                        cluster_sizes.get(node, 0) > max_cluster_size):
                    is_cluster[node] = False
                    stability[node] = subtree_stability
                else:
                    # Mark all descendants as not clusters
                    for sub_node in self._bfs_from_cluster_tree(cluster_tree, node):
                        if sub_node != node:
                            is_cluster[sub_node] = False

            if epsilon != 0.0 and cluster_tree.shape[0] > 0:
                eom_clusters = [c for c in is_cluster if is_cluster[c]]
                if len(eom_clusters) == 1 and eom_clusters[0] == root:
                    # Only the root survived: it skips the epsilon check
                    selected = set(eom_clusters) if allow_single_cluster else set()
                else:
                    selected = self._epsilon_search(
                        set(eom_clusters), cluster_tree, epsilon, allow_single_cluster)
                is_cluster = {c: (c in selected) for c in is_cluster}

        else:  # 'leaf'
            leaves = set(self._cluster_tree_leaves(cluster_tree))
            if len(leaves) == 0:
                # The root has no child clusters: it is the only candidate (the reference
                # implementation marks it and then un-marks it again, returning all noise).
                selected = {root} if allow_single_cluster else set()
            elif epsilon != 0.0:
                selected = self._epsilon_search(leaves, cluster_tree, epsilon, allow_single_cluster)
            else:
                selected = leaves
            is_cluster = {c: (c in selected) for c in set(is_cluster) | selected}

        clusters = set(c for c in is_cluster if is_cluster[c])
        cluster_map = {c: n for n, c in enumerate(sorted(clusters))}

        labels = self._do_labelling(tree, clusters, cluster_map, allow_single_cluster, epsilon)
        probs = self._get_probabilities(tree, cluster_map, labels)

        return labels, probs

    def _bfs_from_cluster_tree(self, cluster_tree: np.ndarray, bfs_root: int) -> list:
        """BFS from cluster tree."""
        result = []
        to_process = [bfs_root]
        
        while len(to_process) > 0:
            result.extend(to_process)
            to_process = cluster_tree['child'][np.isin(cluster_tree['parent'], to_process)].tolist()
        
        return result

    def _cluster_tree_leaves(self, cluster_tree: np.ndarray) -> list:
        """Leaves (clusters without child clusters) of the condensed cluster tree."""
        if cluster_tree.shape[0] == 0:
            return []
        parents = set(cluster_tree['parent'].tolist())
        return [int(c) for c in cluster_tree['child'] if int(c) not in parents]

    def _traverse_upwards(
        self, cluster_tree: np.ndarray, epsilon: float, leaf: int, allow_single_cluster: bool
    ) -> int:
        """
        First ancestor of ``leaf`` that was born at a distance of at least
        ``epsilon`` (``1 / lambda_birth >= epsilon`` is what "born at distance"
        means); stops below the root unless ``allow_single_cluster``.
        """
        root = cluster_tree['parent'].min()
        parent = cluster_tree['parent'][cluster_tree['child'] == leaf][0]
        if parent == root:
            return int(parent) if allow_single_cluster else int(leaf)
        parent_eps = 1.0 / cluster_tree['lambda_val'][cluster_tree['child'] == parent][0]
        if parent_eps > epsilon:
            return int(parent)
        return self._traverse_upwards(cluster_tree, epsilon, int(parent), allow_single_cluster)

    def _epsilon_search(
        self, leaves: set, cluster_tree: np.ndarray, epsilon: float, allow_single_cluster: bool
    ) -> set:
        """
        Replace every cluster in ``leaves`` that was born at a distance below
        ``epsilon`` by its ancestor from :meth:`_traverse_upwards`; clusters born
        at or above ``epsilon`` are kept.
        """
        selected = []
        processed = set()
        for leaf in sorted(leaves):
            eps = 1.0 / cluster_tree['lambda_val'][cluster_tree['child'] == leaf][0]
            if eps < epsilon:
                if leaf not in processed:
                    epsilon_child = self._traverse_upwards(cluster_tree, epsilon, leaf, allow_single_cluster)
                    selected.append(epsilon_child)
                    for sub_node in self._bfs_from_cluster_tree(cluster_tree, epsilon_child):
                        if sub_node != epsilon_child:
                            processed.add(sub_node)
            else:
                selected.append(leaf)
        return set(selected)

    def _do_labelling(
        self,
        tree: np.ndarray,
        clusters: set,
        cluster_map: dict,
        allow_single_cluster: bool,
        epsilon: float = 0.0
    ) -> torch.Tensor:
        """Assign labels based on selected clusters."""
        root_cluster = tree['parent'].min()
        result = torch.full((root_cluster,), -1, dtype=torch.long, device=self.device)
        
        # Union-Find to propagate cluster membership
        parent = np.arange(tree['parent'].max() + 1, dtype=np.intp)
        
        for i in range(len(tree)):
            child = tree['child'][i]
            parent_node = tree['parent'][i]
            if child not in clusters:
                parent[child] = parent_node

        lambda_of_child = tree['lambda_val']
        child_of = tree['child']
        
        # Find clusters for each point
        for n in range(root_cluster):
            # Find root
            cluster = n
            while parent[cluster] != cluster:
                cluster = parent[cluster]
            
            if cluster < root_cluster:
                result[n] = -1
            elif cluster == root_cluster:
                # Single-cluster result: only points that stayed in the root until a
                # high enough density are members (reference implementation's rule).
                if len(clusters) == 1 and allow_single_cluster and cluster in cluster_map:
                    point_lambda = lambda_of_child[child_of == n][0]
                    if epsilon != 0.0:
                        member = point_lambda >= 1.0 / epsilon
                    else:
                        member = point_lambda >= lambda_of_child[tree['parent'] == cluster].max()
                    result[n] = cluster_map[cluster] if member else -1
                else:
                    result[n] = -1
            elif cluster in cluster_map:
                result[n] = cluster_map[cluster]
            else:
                result[n] = -1
        
        return result
    
    def _get_probabilities(
        self,
        tree: np.ndarray,
        cluster_map: dict,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute membership probabilities."""
        # Compute max lambda for each cluster
        deaths = self._max_lambdas(tree)
        
        result = torch.zeros(labels.shape[0], dtype=torch.float32, device=self.device)
        root_cluster = tree['parent'].min()
        
        reverse_map = {v: k for k, v in cluster_map.items()}
        
        for i in range(len(tree)):
            point = tree['child'][i]
            if point >= root_cluster:
                continue
            
            cluster_num = labels[point].item()
            if cluster_num == -1:
                continue
            
            cluster = reverse_map[cluster_num]
            max_lambda = deaths[cluster]
            lambda_val = tree['lambda_val'][i]
            
            if max_lambda == 0.0 or not np.isfinite(lambda_val):
                result[point] = 1.0
            else:
                lambda_val = min(lambda_val, max_lambda)
                result[point] = lambda_val / max_lambda
        
        return result
    
    def _max_lambdas(self, tree: np.ndarray) -> np.ndarray:
        """Compute maximum lambda for each parent."""
        sorted_parent_data = np.sort(tree[['parent', 'lambda_val']], axis=0)
        deaths = np.zeros(tree['parent'].max() + 1, dtype=np.float64)
        
        current_parent = -1
        max_lambda = 0
        
        for row in sorted_parent_data:
            parent = int(row['parent'])
            lambda_val = float(row['lambda_val'])
            
            if parent == current_parent:
                max_lambda = max(max_lambda, lambda_val)
            elif current_parent != -1:
                deaths[current_parent] = max_lambda
                current_parent = parent
                max_lambda = lambda_val
            else:
                current_parent = parent
                max_lambda = lambda_val
        
        deaths[current_parent] = max_lambda
        return deaths
