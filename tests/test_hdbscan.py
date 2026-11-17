"""Pytest suite for tgmm's PyTorch HDBSCAN implementation, including parity with sklearn."""
import numpy as np
import torch
import pytest
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.metrics import adjusted_rand_score

from tgmm.hdbscan import HDBSCAN
from tgmm.synthetic_data import generate_gmm_data


def noisy_blob_data(seed=0):
    n_samples = [200, 150, 250]
    centers = [np.array([0, 0]), np.array([4, 4]), np.array([0, 4])]
    covs = [0.5 * np.eye(2), 0.4 * np.eye(2), 0.6 * np.eye(2)]
    X, y = generate_gmm_data(
        centers, covs, n_samples,
        random_state=seed, noise_ratio=0.15, noise_scale=1.5,
    )
    return X, y


class TestSklearnParity:
    """tgmm's HDBSCAN and sklearn's both define min_samples as including the
    point itself (the k-th nearest neighbor distance where k = min_samples,
    counting the point as its own 1st neighbor). With that convention matched,
    the two implementations should agree closely."""

    def test_matched_min_samples_agrees_with_sklearn(self):
        X, _ = noisy_blob_data(seed=0)
        X_np = X.cpu().numpy()

        torch_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', device='cpu')
        labels_torch = torch_model.fit_predict(X).cpu().numpy()

        sklearn_model = SklearnHDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
        labels_sklearn = sklearn_model.fit_predict(X_np)

        ari = adjusted_rand_score(labels_sklearn, labels_torch)
        assert ari > 0.99

        n_clusters_torch = torch_model.n_clusters_
        n_clusters_sklearn = len(set(labels_sklearn)) - (1 if -1 in labels_sklearn else 0)
        assert n_clusters_torch == n_clusters_sklearn

    def test_mismatched_min_samples_convention_disagrees(self):
        """Regression guard: min_samples=5 (torch) vs min_samples=4 (sklearn)
        looks like a self-inclusion offset but is actually just a parameter
        mismatch (sklearn's min_samples already includes the point itself),
        so it should NOT agree as well as the matched case above."""
        X, _ = noisy_blob_data(seed=0)
        X_np = X.cpu().numpy()

        torch_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', device='cpu')
        labels_torch = torch_model.fit_predict(X).cpu().numpy()

        sklearn_model = SklearnHDBSCAN(min_cluster_size=15, min_samples=4, metric='euclidean')
        labels_sklearn = sklearn_model.fit_predict(X_np)

        ari = adjusted_rand_score(labels_sklearn, labels_torch)
        assert ari < 0.99

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
    def test_matched_min_samples_agrees_across_metrics(self, metric):
        X, _ = noisy_blob_data(seed=1)
        X_np = X.cpu().numpy()

        torch_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric=metric, device='cpu')
        labels_torch = torch_model.fit_predict(X).cpu().numpy()

        sklearn_model = SklearnHDBSCAN(min_cluster_size=15, min_samples=5, metric=metric)
        labels_sklearn = sklearn_model.fit_predict(X_np)

        ari = adjusted_rand_score(labels_sklearn, labels_torch)
        assert ari > 0.95


class TestMinClusterSize:
    def test_larger_min_cluster_size_merges_or_drops_clusters(self):
        centers = [np.array([0, 0]), np.array([3, 3]), np.array([0, 3]), np.array([3, 0])]
        # generate_gmm_data used to sample with covariance cov @ cov.T, so 0.4 * I meant a variance of 0.16;
        # ask for 0.16 * I explicitly to keep the same (well-separated) data now that it honours `covs`.
        covs = [0.16 * np.eye(2) for _ in range(4)]
        X, _ = generate_gmm_data(centers, covs, [150, 100, 80, 50], random_state=42)

        n_clusters_by_size = {}
        for min_size in [10, 70, 100]:
            model = HDBSCAN(min_cluster_size=min_size, min_samples=5, device='cpu')
            model.fit_predict(X)
            n_clusters_by_size[min_size] = model.n_clusters_

        assert n_clusters_by_size[10] == 4
        assert n_clusters_by_size[70] == 3
        assert n_clusters_by_size[100] == 2


# ---------------------------------------------------------------------------
# n_clusters: exactly K non-nested clusters of the condensed tree (opt-in)
# ---------------------------------------------------------------------------
import itertools

from tgmm.hdbscan import InsufficientClustersError

_TREE_DTYPE = np.dtype([('parent', np.intp), ('child', np.intp), ('lambda_val', float), ('child_size', np.intp)])
SEPARATED = np.array([(0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (8.0, 8.0)])


def separated_blobs(n_per=100, seed=1):
    return generate_gmm_data([c for c in SEPARATED], [np.eye(2)] * 4, [n_per] * 4, random_state=seed)


def random_cluster_tree(rng, root=1000, max_nodes=11, p_split=0.7):
    """A random binary condensed cluster tree: (structured array of its cluster rows, stability dict, parent map)."""
    rows, parent_of, stability = [], {}, {root: float(rng.uniform(0.5, 5.0))}
    frontier, next_id = [root], root + 1
    while frontier and next_id - root < max_nodes:
        node = frontier.pop(0)
        if rng.random() < p_split and next_id - root + 2 <= max_nodes + 1:
            for _ in range(2):
                rows.append((node, next_id, 1.0, 10))
                parent_of[next_id] = node
                stability[next_id] = float(rng.uniform(0.1, 5.0))
                frontier.append(next_id)
                next_id += 1
    return np.array(rows, dtype=_TREE_DTYPE), stability, parent_of


def brute_force_best(stability, parent_of, root, K, allow_single=False):
    """Best total stability over every set of K non-nested clusters (the root only if allow_single and K == 1)."""
    candidates = [v for v in stability if v != root or allow_single]

    def ancestors(v):
        out = set()
        while v in parent_of:
            v = parent_of[v]
            out.add(v)
        return out

    anc = {v: ancestors(v) for v in candidates}
    best = None
    for combo in itertools.combinations(candidates, K):
        if any(a in anc[b] for a in combo for b in combo):
            continue
        total = sum(stability[v] for v in combo)
        if best is None or total > best:
            best = total
    return best


class TestExactK:
    def test_dp_matches_exhaustive_search_on_random_trees(self):
        rng = np.random.default_rng(0)
        checked = 0
        for _ in range(30):
            tree, stability, parent_of = random_cluster_tree(rng)
            if len(tree) == 0:
                continue
            root = int(tree['parent'].min())
            leaves = [v for v in stability if v != root and v not in set(tree['parent'].tolist())]
            for K in range(1, len(leaves) + 1):
                chosen = HDBSCAN()._select_k_clusters(tree, stability, K)
                assert len(chosen) == K and root not in chosen
                for a, b in itertools.permutations(chosen, 2):                      # non-nested
                    node = a
                    while node in parent_of:
                        node = parent_of[node]
                        assert node != b
                got = sum(stability[v] for v in chosen)
                assert got == pytest.approx(brute_force_best(stability, parent_of, root, K))
                checked += 1
        assert checked > 60

    def test_dp_with_the_root_as_a_candidate(self):
        tree = np.array([(100, 101, 1.0, 10), (100, 102, 1.0, 10)], dtype=_TREE_DTYPE)
        stability = {100: 9.0, 101: 2.0, 102: 3.0}
        h = HDBSCAN()
        assert h._select_k_clusters(tree, stability, 1, allow_single_cluster=True) == {100}   # the root is the most stable
        assert h._select_k_clusters(tree, stability, 1, allow_single_cluster=False) == {102}  # otherwise the best child
        assert h._select_k_clusters(tree, stability, 2, allow_single_cluster=True) == {101, 102}

    def test_dp_prefers_the_cluster_to_its_descendants_on_a_tie(self):
        tree = np.array([(100, 101, 1.0, 10), (100, 102, 1.0, 10), (101, 103, 2.0, 5), (101, 104, 2.0, 5)], dtype=_TREE_DTYPE)
        stability = {100: 1.0, 101: 3.0, 102: 1.0, 103: 1.5, 104: 1.5}
        assert HDBSCAN()._select_k_clusters(tree, stability, 2) == {101, 102}     # 101 ties with 103 + 104 and wins

    def test_dp_is_iterative_on_deep_trees(self):
        # a caterpillar tree 3000 levels deep must not hit the recursion limit
        rows, stability, nid = [], {1000: 1.0}, 1001
        node = 1000
        for _ in range(3000):
            rows += [(node, nid, 1.0, 10), (node, nid + 1, 1.0, 10)]
            stability[nid], stability[nid + 1] = 1.0, 0.5
            node, nid = nid, nid + 2
        tree = np.array(rows, dtype=_TREE_DTYPE)
        chosen = HDBSCAN()._select_k_clusters(tree, stability, 3)
        assert len(chosen) == 3

    @pytest.mark.parametrize("K", [2, 3, 4])
    def test_finds_exactly_k_clusters_on_separated_blobs(self, K):
        X, _ = separated_blobs()
        h = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=K).fit(X)
        found = sorted(set(h.labels_.tolist()) - {-1})
        assert h.n_clusters_ == K and found == list(range(K))

    def test_labels_are_pure_and_probabilities_valid(self):
        X, y = separated_blobs()
        h = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=4).fit(X)
        labels = h.labels_.numpy()
        assert adjusted_rand_score(y.numpy(), labels) == pytest.approx(1.0)
        p = h.probabilities_
        assert bool(((p >= 0) & (p <= 1)).all()) and bool((p[h.labels_ < 0] == 0).all())

    def test_fewer_clusters_than_blobs_leaves_the_rest_as_noise(self):
        X, _ = separated_blobs()
        h = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=2).fit(X)
        assert int((h.labels_ < 0).sum()) >= 100            # two blobs are clusters, the others are noise

    def test_deterministic(self):
        X, _ = separated_blobs()
        a = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=3).fit(X)
        b = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=3).fit(X)
        assert torch.equal(a.labels_, b.labels_) and torch.equal(a.probabilities_, b.probabilities_)

    def test_agrees_with_the_default_selection_when_k_is_the_number_of_blobs(self):
        X, _ = separated_blobs()
        default = HDBSCAN(min_cluster_size=10, min_samples=5).fit(X)
        exact = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=default.n_clusters_).fit(X)
        assert adjusted_rand_score(default.labels_.numpy(), exact.labels_.numpy()) == pytest.approx(1.0)

    def test_default_is_unchanged(self):
        X, _ = separated_blobs()
        assert HDBSCAN().n_clusters is None
        a = HDBSCAN(min_cluster_size=10, min_samples=5).fit(X)
        b = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=None).fit(X)
        assert torch.equal(a.labels_, b.labels_) and torch.equal(a.probabilities_, b.probabilities_)

    def test_too_many_clusters_asks_for_a_smaller_min_cluster_size(self):
        X, _ = separated_blobs()
        with pytest.raises(ValueError, match="min_cluster_size"):
            HDBSCAN(min_cluster_size=100, n_clusters=10).fit(X)
        with pytest.raises(InsufficientClustersError):
            HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=50).fit(X)

    def test_single_cluster(self):
        X, _ = separated_blobs()
        h = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=1).fit(X)
        assert h.n_clusters_ == 1 and int((h.labels_ >= 0).sum()) >= 100          # the most stable non-root cluster
        g = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=1, allow_single_cluster=True).fit(X)
        assert g.n_clusters_ == 1                                                # the root competes

    def test_one_point_cannot_be_clustered(self):
        with pytest.raises(ValueError, match="n_clusters"):
            HDBSCAN(n_clusters=1).fit(torch.zeros(1, 2))

    @pytest.mark.parametrize("bad", [0, -2, 2.5, "3", True, None.__class__])
    def test_invalid_n_clusters(self, bad):
        with pytest.raises(ValueError, match="n_clusters"):
            HDBSCAN(n_clusters=bad)

    def test_incompatible_with_a_selection_epsilon(self):
        with pytest.raises(ValueError, match="cluster_selection_epsilon"):
            HDBSCAN(n_clusters=3, cluster_selection_epsilon=0.5)

    def test_selection_method_is_ignored(self):
        X, _ = separated_blobs()
        a = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=4, cluster_selection_method='leaf').fit(X)
        b = HDBSCAN(min_cluster_size=10, min_samples=5, n_clusters=4, cluster_selection_method='eom').fit(X)
        assert torch.equal(a.labels_, b.labels_)
