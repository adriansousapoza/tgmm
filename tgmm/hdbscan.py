"""
PyTorch implementation of HDBSCAN (Hierarchical Density-Based Spatial Clustering 
of Applications with Noise).

This implementation closely follows the scikit-learn-contrib/hdbscan library
but uses PyTorch for GPU acceleration.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm


class HDBSCAN:
    """
    HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications with Noise.
    
    PyTorch implementation that closely follows sklearn-contrib/hdbscan.
    
    Parameters
    ----------
    min_cluster_size : int, default=5
        The minimum number of samples in a group for that group to be considered 
        a cluster; groupings smaller than this size will be left as noise.
        
    min_samples : int, default=None
        The number of samples in a neighborhood for a point to be considered as 
        a core point. This includes the point itself. When None, defaults to 
        min_cluster_size.
        
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances.
        Supported: 'euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'.
        
    alpha : float, default=1.0
        A distance scaling parameter as used in robust single linkage.
        
    allow_single_cluster : bool, default=False
        By default HDBSCAN will not produce a single cluster. Setting this to 
        True will override this and allow single cluster results.
        
    cluster_selection_epsilon : float, default=0.0
        A distance threshold. Clusters below this value will be merged.
        
    max_cluster_size : int, default=0
        A limit to the size of clusters returned by the 'eom' algorithm.
        
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
    ):
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
        
        # Compute distance matrix
        distance_matrix = self._compute_distances(X)
        
        # Compute mutual reachability
        mutual_reachability = self._mutual_reachability(distance_matrix, self.min_samples, self.alpha)
        
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
            self.max_cluster_size
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
        
        # Prim's algorithm for MST
        in_tree = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        min_distances = torch.full((n_samples,), float('inf'), device=self.device)
        predecessors = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        
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
        
        mst = np.array(mst_edges, dtype=np.float64)
        
        # Sort by distance
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
        max_cluster_size: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract clusters from condensed tree using EOM (Excess of Mass)."""
        if len(tree) == 0:
            n_samples = len(self._single_linkage_tree) + 1
            labels = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
            probs = torch.zeros(n_samples, dtype=torch.float32, device=self.device)
            return labels, probs
        
        # Get node list
        if allow_single_cluster:
            node_list = sorted(stability.keys(), reverse=True)
        else:
            node_list = sorted(stability.keys(), reverse=True)[:-1]  # Exclude root
        
        cluster_tree = tree[tree['child_size'] > 1]
        is_cluster = {cluster: True for cluster in node_list}
        num_points = np.max(tree[tree['child_size'] == 1]['child']) + 1
        
        cluster_sizes = {child: child_size for child, child_size in 
                        zip(cluster_tree['child'], cluster_tree['child_size'])}
        
        # EOM: Select clusters by comparing stability
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
        
        clusters = set(c for c in is_cluster if is_cluster[c])
        cluster_map = {c: n for n, c in enumerate(sorted(clusters))}
        
        labels = self._do_labelling(tree, clusters, cluster_map, allow_single_cluster)
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
    
    def _do_labelling(
        self,
        tree: np.ndarray,
        clusters: set,
        cluster_map: dict,
        allow_single_cluster: bool
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
        
        # Find clusters for each point
        for n in range(root_cluster):
            # Find root
            cluster = n
            while parent[cluster] != cluster:
                cluster = parent[cluster]
            
            if cluster < root_cluster:
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
