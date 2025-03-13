import torch
import pytest
from tgmm import GaussianMixture

def test_gaussian_mixture_fit_predict():
    # Generate a synthetic dataset
    X = torch.randn(100, 2)
    
    # Instantiate a GMM with 3 components
    gmm = GaussianMixture(n_components=3, n_features=2)
    gmm.fit(X)
    
    # Predict cluster labels
    labels = gmm.predict(X)
    
    # Check that labels have the correct shape and values in the expected range
    assert labels.shape == (100,)
    assert labels.max() < 3 and labels.min() >= 0

if __name__ == '__main__':
    pytest.main([__file__])
