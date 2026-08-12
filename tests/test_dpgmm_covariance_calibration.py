"""Regression tests for GaussianMixture.suggest_priors' covariance calibration.

Gibbs-mode fitting (n_components=None) always requires an NIW prior --
Neal's Algorithm 3 requires one to integrate out means/covariances
analytically, unlike EM mode which can run as plain MLE with no prior at
all. suggest_priors is the library's principled-starting-point helper for
that required prior; an earlier version of this project's (now-merged)
DPGMM class auto-resolved a default using this same logic, and an even
earlier version of that default used the *marginal* covariance of the
whole dataset, which conflated *between*-cluster spread with *within*-
cluster spread: for well-separated multi-modal data, the global covariance
is dominated by how far apart the cluster centers are, not by how tight
each cluster actually is, and the mean-prior cross term's coefficient
`(lambda0 * n_k) / (lambda0 + n_k)` converges to `lambda0` (not 0) as
`n_k -> infinity`, so it never shrinks away when a cluster's own mean sits
far from the global mean. Net effect: every component's fitted covariance
was systematically inflated, and *more* inflated the more cleanly
separated the clusters were -- backwards from what you'd want.

Reproduced on 10-d, 5-cluster synthetic data with true (identity) trace 10:
classical GaussianMixture (EM/MLE) recovered trace ~9.70; Gibbs mode with
the marginal-covariance prior recovered ~16.11 (well-separated, spread=8.0)
but only ~9.88 (overlapping, spread=2.5) -- confirming the inflation
tracked how much the global covariance overstated the true within-cluster
scale, not some fixed sampler bias. suggest_priors' pooled-within-cluster-
scatter estimator (this file's subject) fixes it.
"""
import numpy as np
import pytest
import torch

from tgmm import GaussianMixture
from tgmm.synthetic_data import generate_gmm_data


def make_isotropic_blobs(k, n_per_cluster, spread, d=10, seed=0):
    rng = np.random.RandomState(seed)
    centers = [rng.randn(d) * spread for _ in range(k)]
    covs = [np.eye(d) for _ in range(k)]
    X, labels = generate_gmm_data(centers, covs, [n_per_cluster] * k, random_state=seed)
    return X.double(), labels


@pytest.mark.parametrize("max_components", [None, 20])
def test_gibbs_covariance_not_inflated_on_well_separated_clusters(max_components):
    d = 10
    X, _ = make_isotropic_blobs(k=5, n_per_cluster=200, spread=8.0, d=d)
    extra = dict(max_iter=50, burn_in=15) if max_components is None else dict(max_iter=50)

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=5)
    model = GaussianMixture(n_components=None, covariance_type="full", alpha=1.0, random_state=0,
                             weight_threshold=10, device="cpu", max_components=max_components,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior,
                             **extra)
    model.fit(X)

    true_trace = float(d)  # true per-cluster covariance is identity
    mean_trace = model.covariances_[model.active_].diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()

    # Classical GaussianMixture (EM/MLE) on the same data recovers ~9.7;
    # the marginal-covariance-prior version of Gibbs mode reported ~16.1
    # here (a ~60% inflation) -- suggest_priors' pooled estimator fixes it.
    assert mean_trace == pytest.approx(true_trace, rel=0.25)


def test_gibbs_covariance_stays_accurate_on_overlapping_clusters():
    # Non-regression guard: the overlapping regime was already close to
    # unbiased even under the old marginal-covariance prior (~9.88 vs true
    # 10) -- suggest_priors must not make it worse.
    d = 10
    X, _ = make_isotropic_blobs(k=5, n_per_cluster=200, spread=2.5, d=d)

    mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
        GaussianMixture.suggest_priors(X, n_components=5)
    model = GaussianMixture(n_components=None, covariance_type="full", alpha=1.0, random_state=0,
                             weight_threshold=10, device="cpu", max_components=None, max_iter=50, burn_in=15,
                             mean_prior=mean_prior, mean_precision_prior=mean_precision_prior,
                             covariance_prior=covariance_prior, degrees_of_freedom_prior=degrees_of_freedom_prior)
    model.fit(X)

    true_trace = float(d)
    mean_trace = model.covariances_[model.active_].diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()
    assert mean_trace == pytest.approx(true_trace, rel=0.25)
