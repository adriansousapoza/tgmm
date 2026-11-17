"""HDBSCAN as an initialisation method: ``GMMInitializer.hdbscan`` / ``hdbscan_clusters``,
``init_means='hdbscan'``, ``init_weights='hdbscan'`` and ``init_covariances='hdbscan'``."""
import numpy as np
import pytest
import torch

from tgmm import GaussianMixture, GMMInitializer, generate_gmm_data
from tgmm.hdbscan import HDBSCAN

CENTRES = np.array([(0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (8.0, 8.0)])


def blobs(n_per=(300, 300, 300, 300), seed=1, dtype=torch.float32):
    X, y = generate_gmm_data([c for c in CENTRES], [np.eye(2)] * 4, list(n_per), random_state=seed)
    return X.to(dtype), y


def with_outliers(X, frac=0.02, seed=0):
    rng = np.random.default_rng(seed)
    out = torch.tensor(rng.uniform(-12, 20, (int(round(frac * len(X))), 2)), dtype=X.dtype)
    return torch.cat([X, out])


def recovered(means, tol=1.5):
    """One initial mean in every blob: each mean lies within `tol` of its nearest true centre and the four means
    pick four different centres."""
    d = np.linalg.norm(np.asarray(means)[:, None] - CENTRES[None], axis=2)   # init x true
    nearest = d.argmin(axis=1)
    return bool(len(set(nearest.tolist())) == len(CENTRES) and (d.min(axis=1) < tol).all())


class TestShapeDtypeDevice:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_shapes_and_dtype(self, dtype):
        X, _ = blobs(dtype=dtype)
        means, labels = GMMInitializer.hdbscan_clusters(X, 4, random_state=0)
        assert means.shape == (4, 2) and means.dtype == dtype
        assert labels.shape == (X.shape[0],) and labels.dtype == torch.long
        assert int(labels.max()) <= 3 and int(labels.min()) >= -1
        assert GMMInitializer.hdbscan(X, 4, random_state=0).shape == (4, 2)

    def test_one_dimensional_input_is_one_feature(self):
        x = torch.cat([torch.randn(200) - 6, torch.randn(200) + 6])
        means = GMMInitializer.hdbscan(x, 2, random_state=0)
        assert means.shape == (2, 1)
        assert sorted(means.squeeze(1).tolist())[0] < -4 and sorted(means.squeeze(1).tolist())[1] > 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_data_stays_on_device(self):
        X, _ = blobs()
        means, labels = GMMInitializer.hdbscan_clusters(X.cuda(), 4, random_state=0)
        assert means.is_cuda and labels.is_cuda


class TestReproducibility:
    def test_same_random_state_same_means(self):
        X, _ = blobs()
        a = GMMInitializer.hdbscan(X, 4, random_state=3)
        b = GMMInitializer.hdbscan(X, 4, random_state=3)
        assert torch.equal(a, b)

    def test_completion_is_seeded(self):
        # two clusters found, five requested: the three extra centres come from k-means++ and follow the seed
        X = blobs()[0][:600]                           # the first two blobs
        a = GMMInitializer.hdbscan(X, 5, random_state=11)
        b = GMMInitializer.hdbscan(X, 5, random_state=11)
        c = GMMInitializer.hdbscan(X, 5, random_state=12)
        assert torch.equal(a, b)
        assert not torch.equal(a[2:], c[2:])

    def test_global_torch_seed_controls_it_when_random_state_is_none(self):
        X, _ = blobs()
        X = X[:600]
        torch.manual_seed(5)
        a = GMMInitializer.hdbscan(X, 5)
        torch.manual_seed(5)
        b = GMMInitializer.hdbscan(X, 5)
        assert torch.equal(a, b)

    def test_gaussian_mixture_fit_is_reproducible(self):
        X, _ = blobs()
        a = GaussianMixture(4, init_means='hdbscan', random_state=0).fit(X)
        b = GaussianMixture(4, init_means='hdbscan', random_state=0).fit(X)
        assert torch.allclose(a.means_, b.means_) and a.n_iter_ == b.n_iter_


class TestKHandling:
    def test_more_clusters_than_k_keeps_the_largest(self):
        X, _ = blobs(n_per=(400, 300, 200, 100))
        means, labels = GMMInitializer.hdbscan_clusters(X, 2, random_state=0)
        assert means.shape == (2, 2)
        # clusters are ordered by decreasing size: the 400- and 300-point blobs
        assert torch.allclose(means[0], torch.tensor(CENTRES[0], dtype=means.dtype), atol=0.5)
        assert torch.allclose(means[1], torch.tensor(CENTRES[1], dtype=means.dtype), atol=0.5)
        assert set(labels.unique().tolist()) <= {-1, 0, 1}
        assert int((labels == 0).sum()) > int((labels == 1).sum()) > 0
        # the dropped clusters' points are noise (-1)
        assert int((labels == -1).sum()) >= 250

    def test_fewer_clusters_than_k_adds_kpp_centres(self):
        X, _ = blobs()
        X = X[:600]                                   # two blobs only
        means, labels = GMMInitializer.hdbscan_clusters(X, 4, random_state=0)
        assert means.shape == (4, 2)
        assert set(labels.unique().tolist()) <= {-1, 0, 1}      # the extra centres own no points
        for extra in means[2:]:                                  # drawn from the data
            assert bool((X == extra).all(dim=1).any())
        assert not torch.equal(means[2], means[3])

    def test_extra_centres_prefer_noise_points(self):
        X, _ = blobs()
        X = X[:600]
        X = with_outliers(X, frac=0.02)
        means, labels = GMMInitializer.hdbscan_clusters(X, 4, random_state=0)
        noise = X[labels < 0]
        assert len(noise) > 0
        for extra in means[2:]:
            assert bool((noise == extra).all(dim=1).any())

    def test_no_cluster_falls_back_to_kpp(self):
        torch.manual_seed(0)
        X = torch.rand(100, 2)
        means, labels = GMMInitializer.hdbscan_clusters(X, 3, min_cluster_size=100, random_state=3)
        assert bool((labels == -1).all())
        assert torch.equal(means, GMMInitializer.kpp(X, 3, random_state=3))


class TestNoiseRobustness:
    def test_hdbscan_recovers_every_blob_despite_outliers_while_maxdist_does_not(self):
        X, _ = blobs()
        Xo = with_outliers(X, frac=0.02)
        hdb_ok = 0
        maxdist_ok = 0
        for seed in range(5):
            torch.manual_seed(seed)
            hdb_ok += recovered(GMMInitializer.hdbscan(Xo, 4, random_state=seed).numpy(), tol=1.5)
            maxdist_ok += recovered(GMMInitializer.maxdist(Xo, 4).numpy(), tol=4.5)   # farthest-first sits in the tails
        assert hdb_ok == 5
        assert maxdist_ok <= 1          # farthest-first grabs the outliers

    def test_without_outliers_maxdist_also_puts_one_mean_in_every_blob(self):
        X, _ = blobs()
        assert recovered(GMMInitializer.hdbscan(X, 4, random_state=0).numpy(), tol=1.5)
        assert recovered(GMMInitializer.maxdist(X, 4).numpy(), tol=4.5)


class TestGaussianMixtureIntegration:
    @pytest.mark.parametrize("cov_type", ["full", "diag", "spherical", "tied_full", "tied_diag", "tied_spherical",
                                           "EEV", "VEV"])
    def test_fit_converges_and_recovers_the_blobs(self, cov_type):
        X, _ = blobs()
        g = GaussianMixture(4, covariance_type=cov_type, init_means='hdbscan', init_weights='hdbscan',
                            init_covariances='hdbscan', random_state=0).fit(X)
        assert g.converged_
        assert recovered(g.means_.cpu().numpy(), tol=0.3)
        assert torch.allclose(g.weights_.sort().values, torch.full((4,), 0.25), atol=0.03)

    def test_means_only_and_outlier_contaminated_fit(self):
        X, _ = blobs()
        Xo = with_outliers(X, frac=0.02)
        g = GaussianMixture(4, init_means='hdbscan', random_state=0).fit(Xo)
        assert g.converged_
        assert recovered(g.initial_means_.cpu().numpy())

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_initial_parameters_come_from_the_clustering(self):
        X, _ = blobs(n_per=(400, 300, 200, 100))
        g = GaussianMixture(4, init_means='hdbscan', init_weights='hdbscan', init_covariances='hdbscan',
                            random_state=0, max_iter=1)
        g.fit(X)
        w = g.initial_weights_.cpu()
        assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)
        # components are ordered by cluster size: 400, 300, 200, 100 points of 1000
        assert torch.allclose(w, torch.tensor([0.4, 0.3, 0.2, 0.1]), atol=0.03)
        for k in range(4):
            assert torch.allclose(g.initial_covariances_[k].cpu(), torch.eye(2), atol=0.35)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_n_init_clusters_only_once(self, monkeypatch):
        X, _ = blobs()
        calls = {"n": 0}
        original = HDBSCAN.fit

        def counting_fit(self, data):
            calls["n"] += 1
            return original(self, data)

        monkeypatch.setattr(HDBSCAN, "fit", counting_fit)
        GaussianMixture(4, init_means='hdbscan', n_init=3, random_state=0).fit(X)
        assert calls["n"] == 1

    def test_weights_and_covariances_need_hdbscan_means(self):
        X, _ = blobs()
        with pytest.raises(ValueError, match="init_weights='hdbscan'.*init_means='hdbscan'"):
            GaussianMixture(4, init_means='kmeans', init_weights='hdbscan').fit(X)
        with pytest.raises(ValueError, match="init_covariances='hdbscan'.*init_means='hdbscan'"):
            GaussianMixture(4, init_means='kmeans', init_covariances='hdbscan').fit(X)

    def test_unsupported_method_message_lists_hdbscan(self):
        X, _ = blobs()
        with pytest.raises(ValueError, match="hdbscan"):
            GaussianMixture(4, init_means='nope').fit(X)


class TestHelpers:
    def test_weights_from_labels_gives_noise_to_the_nearest_mean(self):
        data = torch.tensor([[0.0, 0], [0.1, 0], [10, 0], [10.1, 0], [9.0, 0]])
        means = torch.tensor([[0.0, 0], [10.0, 0]])
        labels = torch.tensor([0, 0, 1, 1, -1])
        w = GMMInitializer.init_weights_from_labels(data, means, labels)
        assert torch.allclose(w, torch.tensor([0.4, 0.6]))
        assert torch.isclose(w.sum(), torch.tensor(1.0))

    def test_empirical_covariance_with_labels_leaves_noise_out(self):
        torch.manual_seed(0)
        a = torch.randn(200, 2)
        b = torch.randn(200, 2) * 2 + 10
        noise = torch.tensor([[1000.0, 1000.0], [-1000.0, 900.0]])
        data = torch.cat([a, b, noise])
        labels = torch.cat([torch.zeros(200), torch.ones(200), -torch.ones(2)]).long()
        means = torch.stack([a.mean(0), b.mean(0)])
        covs = GMMInitializer.init_covariances_empirical(data, means, 'full', 0.0, labels=labels)
        assert torch.allclose(covs[0], torch.cov(a.T), atol=1e-5)
        assert torch.allclose(covs[1], torch.cov(b.T), atol=1e-5)
        # without labels the far noise points are nearest-mean assigned and blow the covariance up
        covs_nn = GMMInitializer.init_covariances_empirical(data, means, 'full', 0.0)
        assert float(covs_nn[1].abs().max()) > 1000

    def test_empirical_covariance_default_is_unchanged(self):
        torch.manual_seed(0)
        data = torch.randn(300, 2)
        means = torch.randn(3, 2)
        explicit = GMMInitializer.init_covariances_empirical(
            data, means, 'full', 1e-6, labels=torch.cdist(data, means).argmin(1))
        default = GMMInitializer.init_covariances_empirical(data, means, 'full', 1e-6)
        assert torch.allclose(explicit, default)

    def test_subsampling_path(self):
        X, _ = blobs(n_per=(900, 900, 900, 900))
        means, labels = GMMInitializer.hdbscan_clusters(X, 4, max_points=800, random_state=0)
        assert labels.shape == (3600,)
        assert recovered(means.numpy())
        assert int((labels >= 0).sum()) > 3000


class TestInvalidArguments:
    @pytest.mark.parametrize("kwargs, match", [
        (dict(k=0), "positive integer"),
        (dict(k=1.5), "positive integer"),
        (dict(k=10 ** 6), "cannot initialise"),
        (dict(k=2, min_cluster_size=1), "min_cluster_size"),
        (dict(k=2, min_samples=0), "min_samples"),
        (dict(k=2, max_points=1), "max_points"),
    ])
    def test_bad_arguments(self, kwargs, match):
        X, _ = blobs()
        with pytest.raises(ValueError, match=match):
            GMMInitializer.hdbscan(X, **kwargs)

    def test_three_dimensional_data_is_rejected(self):
        with pytest.raises(ValueError, match="2D"):
            GMMInitializer.hdbscan(torch.randn(10, 2, 2), 2)


class TestDefaultRule:
    """min_cluster_size = max(10, min(int(0.02 n), n // (5 k))), min_samples = max(5, min_cluster_size // 2)."""

    @pytest.mark.parametrize("n, k, expected", [
        (3000, 4, 60),      # 2% of the points
        (3000, 50, 12),     # n // (5 k) takes over when k is large, so 50 clusters of ~60 points survive
        (200, 4, 10),       # never below 10
        (1200, 8, 24),
    ])
    def test_default_min_cluster_size(self, n, k, expected):
        assert GMMInitializer._hdbscan_default_min_cluster_size(n, k) == expected

    def test_many_small_clusters_are_not_lost_to_noise(self):
        # 50 blobs of 60 points: a flat 2% rule (60) would call most of them noise
        K, cols = 50, 8
        centres = [np.array([8.0 * (i % cols), 8.0 * (i // cols)]) for i in range(K)]
        X, _ = generate_gmm_data(centres, [np.eye(2) * 0.64] * K, [60] * K, random_state=3)
        means, labels = GMMInitializer.hdbscan_clusters(X, K, random_state=0)
        d = np.linalg.norm(np.stack(centres)[:, None] - means.numpy()[None], axis=2)
        assert int((d.min(axis=1) < 2.0).sum()) == K

    def test_tutorial_like_overlapping_blobs_give_four_clusters(self):
        covs = [np.eye(2), 0.25 * np.eye(2), np.array([[4.0, 0.0], [0.0, 0.25]]), np.array([[0.29, 1.1], [1.1, 4.25]])]
        cen = [np.array([0, 2]), np.array([2, -2]), np.array([0, 0]), np.array([2, 2])]
        X, _ = generate_gmm_data(cen, covs, [800, 200, 1000, 1000], random_state=0)
        _, labels = GMMInitializer.hdbscan_clusters(X, 4, random_state=0)
        assert int(labels.max()) + 1 == 4


class TestHdbscanKwargs:
    def test_default_is_none_and_stored(self):
        assert GaussianMixture(4).hdbscan_kwargs is None
        g = GaussianMixture(4, hdbscan_kwargs={'min_cluster_size': 40})
        assert g.hdbscan_kwargs == {'min_cluster_size': 40}

    def test_kwargs_tune_the_clustering(self):
        X, _ = blobs()
        small = GaussianMixture(4, init_means='hdbscan', init_weights='hdbscan', init_covariances='hdbscan',
                                hdbscan_kwargs={'min_cluster_size': 20}, random_state=0).fit(X)
        huge = GaussianMixture(4, init_means='hdbscan', hdbscan_kwargs={'min_cluster_size': 700}, random_state=0).fit(X)
        assert small.converged_ and recovered(small.means_.cpu().numpy(), tol=0.3)
        # a minimum cluster size above the blob size leaves no cluster: the start is plain k-means++ (still converges)
        assert huge.converged_

    def test_kwargs_change_the_number_of_clusters_found(self):
        X, _ = blobs()
        _, a = GMMInitializer.hdbscan_clusters(X, 4, random_state=0, min_cluster_size=20)
        _, b = GMMInitializer.hdbscan_clusters(X, 4, random_state=0, min_cluster_size=700)
        assert int(a.max()) + 1 == 4 and int(b.max()) + 1 == 0

    def test_weights_and_covariances_reuse_the_tuned_clustering(self):
        X, _ = blobs()
        g = GaussianMixture(4, init_means='hdbscan', init_weights='hdbscan', init_covariances='hdbscan',
                            hdbscan_kwargs={'min_cluster_size': 20, 'min_samples': 10}, max_iter=1, random_state=0)
        g.fit(X)
        _, labels, _ = g._hdbscan_cache
        assert int(labels.max()) + 1 == 4
        assert torch.allclose(g.initial_weights_.sum().cpu(), torch.tensor(1.0), atol=1e-5)

    def test_cache_key_includes_the_kwargs(self):
        X, _ = blobs()
        g = GaussianMixture(4, init_means='hdbscan', max_iter=1, random_state=0, hdbscan_kwargs={'min_cluster_size': 20})
        g.fit(X)
        first = g._hdbscan_cache
        g.hdbscan_kwargs = {'min_cluster_size': 700}      # same data, other settings: it must re-cluster
        g.fit(X)
        assert g._hdbscan_cache is not first
        assert int(g._hdbscan_cache[1].max()) + 1 == 0

    def test_n_init_still_clusters_once(self, monkeypatch):
        X, _ = blobs()
        calls = []
        real = HDBSCAN.fit
        monkeypatch.setattr(HDBSCAN, 'fit', lambda self, data, *a, **k: (calls.append(1), real(self, data, *a, **k))[1])
        GaussianMixture(4, init_means='hdbscan', n_init=3, max_iter=2, random_state=0,
                        hdbscan_kwargs={'min_cluster_size': 20}).fit(X)
        assert len(calls) == 1

    def test_max_points_key(self):
        X, _ = blobs(n_per=(900, 900, 900, 900))
        g = GaussianMixture(4, init_means='hdbscan', hdbscan_kwargs={'max_points': 800}, max_iter=1, random_state=0).fit(X)
        assert int(g._hdbscan_cache[1].max()) + 1 == 4

    def test_other_hdbscan_arguments_are_passed_on(self):
        X, _ = blobs()
        GaussianMixture(4, init_means='hdbscan', hdbscan_kwargs={'cluster_selection_method': 'leaf'},
                        max_iter=1, random_state=0).fit(X)
        with pytest.raises(ValueError, match="cluster_selection_method"):
            GaussianMixture(4, init_means='hdbscan', hdbscan_kwargs={'cluster_selection_method': 'nope'}, max_iter=1).fit(X)

    @pytest.mark.parametrize("bad, exc, match", [
        ([('min_cluster_size', 5)], TypeError, "dict or None"),
        ('min_cluster_size', TypeError, "dict or None"),
        ({'min_cluster_siz': 5}, ValueError, "unknown hdbscan_kwargs key"),
        ({'device': 'cpu'}, ValueError, "unknown hdbscan_kwargs key"),
        ({'min_cluster_size': 1}, ValueError, "min_cluster_size"),
        ({'min_cluster_size': 2.5}, ValueError, "min_cluster_size"),
        ({'min_samples': 0}, ValueError, "min_samples"),
        ({'max_points': 1}, ValueError, "max_points"),
    ])
    def test_invalid_input(self, bad, exc, match):
        with pytest.raises(exc, match=match):
            GaussianMixture(4, init_means='hdbscan', hdbscan_kwargs=bad)

    def test_unknown_key_message_lists_the_allowed_ones(self):
        with pytest.raises(ValueError) as e:
            GaussianMixture(4, hdbscan_kwargs={'nope': 1})
        assert 'min_cluster_size' in str(e.value) and 'max_points' in str(e.value)

    def test_the_dict_is_copied(self):
        kw = {'min_cluster_size': 20}
        g = GaussianMixture(4, hdbscan_kwargs=kw)
        kw['min_cluster_size'] = 999
        assert g.hdbscan_kwargs == {'min_cluster_size': 20}

    def test_save_and_load_roundtrip(self, tmp_path):
        X, _ = blobs()
        g = GaussianMixture(4, init_means='hdbscan', init_weights='hdbscan', init_covariances='hdbscan', random_state=0,
                            hdbscan_kwargs={'min_cluster_size': 20, 'min_samples': 10}).fit(X)
        path = str(tmp_path / 'gmm.pt')
        g.save(path)
        h = GaussianMixture.load(path)
        assert h.hdbscan_kwargs == {'min_cluster_size': 20, 'min_samples': 10}
        sd = g.save_state_dict()
        assert sd['hdbscan_kwargs'] == {'min_cluster_size': 20, 'min_samples': 10}
        k = GaussianMixture(4)
        k.load_state_dict(sd)
        assert k.hdbscan_kwargs == {'min_cluster_size': 20, 'min_samples': 10}

    def test_old_saved_models_without_the_key_load(self, tmp_path):
        X, _ = blobs()
        g = GaussianMixture(4, random_state=0, max_iter=5).fit(X)
        sd = g.save_state_dict()
        sd.pop('hdbscan_kwargs')
        k = GaussianMixture(4)
        k.load_state_dict(sd)
        assert k.hdbscan_kwargs is None
