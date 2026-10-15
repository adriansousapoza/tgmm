"""generate_gmm_data must honour `covs` as covariance matrices (it used to produce cov @ cov.T)."""
import numpy as np
import pytest

from tgmm import generate_gmm_data


@pytest.mark.parametrize("cov", [
    np.eye(2), 0.5 * np.eye(2), np.array([[2.0, 0.0], [0.0, 0.5]]), np.array([[0.2, 0.5], [0.5, 2.0]]),
])
def test_sample_covariance_matches_requested(cov):
    X, _ = generate_gmm_data([np.zeros(2)], [cov], [200000], random_state=0)
    np.testing.assert_allclose(np.cov(X.numpy().T), cov, atol=0.02 * max(1.0, np.abs(cov).max()))


def test_singular_psd_covariance_is_supported():
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])
    X, _ = generate_gmm_data([np.zeros(2)], [cov], [5000], random_state=0)
    assert np.allclose(X[:, 0].numpy(), X[:, 1].numpy(), atol=1e-5)


def test_indefinite_covariance_raises():
    with pytest.raises(ValueError, match="semi-definite"):
        generate_gmm_data([np.zeros(2)], [np.array([[1.0, 2.0], [2.0, 1.0]])], [10])


def test_means_labels_and_seed_reproducibility():
    c = [np.array([0.0, 0.0]), np.array([5.0, 5.0])]
    cov = [np.eye(2), 2 * np.eye(2)]
    X1, y1 = generate_gmm_data(c, cov, [30, 20], random_state=1)
    X2, y2 = generate_gmm_data(c, cov, [30, 20], random_state=1)
    assert X1.shape == (50, 2) and (y1 == 1).sum() == 20
    assert np.array_equal(X1.numpy(), X2.numpy()) and np.array_equal(y1.numpy(), y2.numpy())
