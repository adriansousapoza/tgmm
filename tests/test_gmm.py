#!/usr/bin/env python3
"""
Comprehensive test script for GaussianMixture model.
Tests all settings, combinations, and edge cases.
"""

import torch
import numpy as np
import tempfile
import os
from tgmm.gmm import GaussianMixture

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Test counters
total_tests = 0
passed_tests = 0
failed_tests = 0

def test(name):
    """Decorator to track test execution"""
    def decorator(func):
        def wrapper():
            global total_tests, passed_tests, failed_tests
            total_tests += 1
            try:
                func()
                passed_tests += 1
                print(f"  ✅ {name}")
                return True
            except AssertionError as e:
                failed_tests += 1
                print(f"  ❌ {name}: {e}")
                return False
            except Exception as e:
                failed_tests += 1
                print(f"  ❌ {name}: Unexpected error: {e}")
                return False
        return wrapper
    return decorator

# ============================================================================
# Data Generation
# ============================================================================

def generate_test_data(n_samples=200, n_features=3, n_clusters=3):
    """Generate synthetic clustered data"""
    X_list = []
    for i in range(n_clusters):
        mean = torch.randn(n_features) * 5
        cov = torch.eye(n_features) + torch.randn(n_features, n_features) * 0.1
        cov = cov @ cov.T  # Make positive definite
        X_cluster = torch.randn(n_samples // n_clusters, n_features) @ cov.T + mean
        X_list.append(X_cluster)
    return torch.cat(X_list, dim=0)

X_2d = generate_test_data(n_samples=200, n_features=2, n_clusters=3)
X_3d = generate_test_data(n_samples=200, n_features=3, n_clusters=4)
X_5d = generate_test_data(n_samples=300, n_features=5, n_clusters=5)
X_10d = generate_test_data(n_samples=400, n_features=10, n_clusters=3)

print("="*80)
print("COMPREHENSIVE GMM TEST SUITE")
print("="*80)

# ============================================================================
# 1. COVARIANCE TYPES
# ============================================================================

print("\n" + "="*80)
print("1. COVARIANCE TYPES")
print("="*80)

@test("full covariance")
def test_full_covariance():
    gmm = GaussianMixture(n_components=3, covariance_type='full')
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.covariances_.shape == (3, 3, 3)

@test("diag covariance")
def test_diag_covariance():
    gmm = GaussianMixture(n_components=3, covariance_type='diag')
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.covariances_.shape == (3, 3)

@test("spherical covariance")
def test_spherical_covariance():
    gmm = GaussianMixture(n_components=3, covariance_type='spherical')
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.covariances_.shape == (3,)

@test("tied_full covariance")
def test_tied_full_covariance():
    gmm = GaussianMixture(n_components=3, covariance_type='tied_full')
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.covariances_.shape == (3, 3)

@test("tied_diag covariance")
def test_tied_diag_covariance():
    gmm = GaussianMixture(n_components=3, covariance_type='tied_diag')
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.covariances_.shape == (3,)

@test("tied_spherical covariance")
def test_tied_spherical_covariance():
    gmm = GaussianMixture(n_components=3, covariance_type='tied_spherical')
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.covariances_.dim() == 0  # scalar

@test("tied alias for tied_full")
def test_tied_alias():
    gmm = GaussianMixture(n_components=3, covariance_type='tied')
    gmm.fit(X_3d)
    assert gmm.covariance_type == 'tied_full'
    assert gmm.fitted_

@test("isotropic alias for spherical")
def test_isotropic_alias():
    gmm = GaussianMixture(n_components=3, covariance_type='isotropic')
    gmm.fit(X_3d)
    assert gmm.covariance_type == 'spherical'
    assert gmm.fitted_

test_full_covariance()
test_diag_covariance()
test_spherical_covariance()
test_tied_full_covariance()
test_tied_diag_covariance()
test_tied_spherical_covariance()
test_tied_alias()
test_isotropic_alias()

# ============================================================================
# 2. MEAN INITIALIZATION METHODS
# ============================================================================

print("\n" + "="*80)
print("2. MEAN INITIALIZATION METHODS")
print("="*80)

@test("init_means='kmeans'")
def test_init_means_kmeans():
    gmm = GaussianMixture(n_components=3, init_means='kmeans')
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.means_.shape == (3, 3)

@test("init_means='kpp'")
def test_init_means_kpp():
    gmm = GaussianMixture(n_components=3, init_means='kpp')
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("init_means='random'")
def test_init_means_random():
    gmm = GaussianMixture(n_components=3, init_means='random')
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("init_means='points'")
def test_init_means_points():
    gmm = GaussianMixture(n_components=3, init_means='points')
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("init_means='maxdist'")
def test_init_means_maxdist():
    gmm = GaussianMixture(n_components=3, init_means='maxdist')
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("init_means=tensor")
def test_init_means_tensor():
    means = torch.randn(3, 3)
    gmm = GaussianMixture(n_components=3, init_means=means)
    gmm.fit(X_3d)
    assert gmm.fitted_
    # Compare on same device
    assert torch.allclose(gmm.initial_means_.cpu(), means.cpu(), atol=1e-5)

test_init_means_kmeans()
test_init_means_kpp()
test_init_means_random()
test_init_means_points()
test_init_means_maxdist()
test_init_means_tensor()

# ============================================================================
# 3. WEIGHT INITIALIZATION METHODS
# ============================================================================

print("\n" + "="*80)
print("3. WEIGHT INITIALIZATION METHODS")
print("="*80)

@test("init_weights='uniform'")
def test_init_weights_uniform():
    gmm = GaussianMixture(n_components=3, init_weights='uniform')
    gmm.fit(X_3d)
    assert gmm.fitted_
    expected = torch.ones(3, device=gmm.device) / 3
    assert torch.allclose(gmm.initial_weights_, expected, atol=1e-5)

@test("init_weights='random'")
def test_init_weights_random():
    gmm = GaussianMixture(n_components=3, init_weights='random')
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert torch.allclose(gmm.initial_weights_.sum(), torch.tensor(1.0), atol=1e-5)

@test("init_weights='kmeans'")
def test_init_weights_kmeans():
    gmm = GaussianMixture(n_components=3, init_weights='kmeans')
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert torch.allclose(gmm.initial_weights_.sum(), torch.tensor(1.0), atol=1e-5)

@test("init_weights=tensor")
def test_init_weights_tensor():
    weights = torch.tensor([0.5, 0.3, 0.2])
    gmm = GaussianMixture(n_components=3, init_weights=weights)
    gmm.fit(X_3d)
    assert gmm.fitted_
    # Weights are normalized, compare on same device
    assert torch.allclose(gmm.initial_weights_.cpu(), weights.cpu(), atol=1e-5)

test_init_weights_uniform()
test_init_weights_random()
test_init_weights_kmeans()
test_init_weights_tensor()

# ============================================================================
# 4. COVARIANCE INITIALIZATION METHODS
# ============================================================================

print("\n" + "="*80)
print("4. COVARIANCE INITIALIZATION METHODS")
print("="*80)

@test("init_covariances='empirical'")
def test_init_covariances_empirical():
    gmm = GaussianMixture(n_components=3, init_covariances='empirical')
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("init_covariances='eye'")
def test_init_covariances_eye():
    gmm = GaussianMixture(n_components=3, init_covariances='eye')
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("init_covariances='random'")
def test_init_covariances_random():
    gmm = GaussianMixture(n_components=3, init_covariances='random')
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("init_covariances='global'")
def test_init_covariances_global():
    gmm = GaussianMixture(n_components=3, init_covariances='global')
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("init_covariances=tensor (full)")
def test_init_covariances_tensor_full():
    covs = torch.stack([torch.eye(3) for _ in range(3)])
    gmm = GaussianMixture(n_components=3, covariance_type='full', init_covariances=covs)
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("init_covariances=tensor (diag)")
def test_init_covariances_tensor_diag():
    covs = torch.ones(3, 3)
    gmm = GaussianMixture(n_components=3, covariance_type='diag', init_covariances=covs)
    gmm.fit(X_3d)
    assert gmm.fitted_

test_init_covariances_empirical()
test_init_covariances_eye()
test_init_covariances_random()
test_init_covariances_global()
test_init_covariances_tensor_full()
test_init_covariances_tensor_diag()

# ============================================================================
# 5. COMBINATION TESTS
# ============================================================================

print("\n" + "="*80)
print("5. INITIALIZATION COMBINATIONS")
print("="*80)

@test("All empirical/data-driven initialization")
def test_all_empirical():
    gmm = GaussianMixture(
        n_components=3,
        init_means='kmeans',
        init_weights='kmeans',
        init_covariances='empirical'
    )
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("All random initialization")
def test_all_random():
    gmm = GaussianMixture(
        n_components=3,
        init_means='random',
        init_weights='random',
        init_covariances='random'
    )
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("Mixed string and tensor initialization")
def test_mixed_init():
    means = torch.randn(3, 3)
    gmm = GaussianMixture(
        n_components=3,
        init_means=means,
        init_weights='uniform',
        init_covariances='global'
    )
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("All explicit tensor initialization")
def test_all_tensor_init():
    means = torch.randn(3, 3)
    weights = torch.tensor([0.4, 0.4, 0.2])
    covs = torch.stack([torch.eye(3) * 1.5 for _ in range(3)])
    gmm = GaussianMixture(
        n_components=3,
        covariance_type='full',
        init_means=means,
        init_weights=weights,
        init_covariances=covs
    )
    gmm.fit(X_3d)
    assert gmm.fitted_

test_all_empirical()
test_all_random()
test_mixed_init()
test_all_tensor_init()

# ============================================================================
# 6. TRAINING PARAMETERS
# ============================================================================

print("\n" + "="*80)
print("6. TRAINING PARAMETERS")
print("="*80)

@test("max_iter parameter")
def test_max_iter():
    gmm = GaussianMixture(n_components=3, max_iter=5)
    gmm.fit(X_3d)
    assert gmm.n_iter_ <= 5

@test("tol parameter")
def test_tol():
    gmm = GaussianMixture(n_components=3, tol=1e-2)
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("reg_covar parameter")
def test_reg_covar():
    gmm = GaussianMixture(n_components=3, reg_covar=1e-4)
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("random_state parameter")
def test_random_state():
    gmm1 = GaussianMixture(n_components=3, random_state=42)
    gmm1.fit(X_3d)
    gmm2 = GaussianMixture(n_components=3, random_state=42)
    gmm2.fit(X_3d)
    assert torch.allclose(gmm1.means_, gmm2.means_, atol=1e-4)

@test("verbose parameter")
def test_verbose():
    gmm = GaussianMixture(n_components=3, verbose=True, verbose_interval=5)
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("warm_start parameter")
def test_warm_start():
    gmm = GaussianMixture(n_components=3, warm_start=True, max_iter=5)
    gmm.fit(X_3d)
    first_means = gmm.means_.clone()
    gmm.fit(X_3d)  # Should start from previous solution
    assert gmm.fitted_

@test("n_init parameter")
def test_n_init():
    gmm = GaussianMixture(n_components=3, n_init=3, random_state=42)
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.best_random_state_ is not None

test_max_iter()
test_tol()
test_reg_covar()
test_random_state()
test_verbose()
test_warm_start()
test_n_init()

# ============================================================================
# 7. CEM (Classification EM)
# ============================================================================

print("\n" + "="*80)
print("7. CLASSIFICATION EM (CEM)")
print("="*80)

@test("CEM algorithm")
def test_cem():
    gmm = GaussianMixture(n_components=3, cem=True)
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("CEM with different covariance types")
def test_cem_covariance_types():
    for cov_type in ['full', 'diag', 'spherical', 'tied_full']:
        gmm = GaussianMixture(n_components=3, cem=True, covariance_type=cov_type)
        gmm.fit(X_3d)
        assert gmm.fitted_, f"CEM failed with {cov_type}"

test_cem()
test_cem_covariance_types()

# ============================================================================
# 8. PRIORS (MAP Estimation)
# ============================================================================

print("\n" + "="*80)
print("8. PRIORS (MAP ESTIMATION)")
print("="*80)

@test("Weight prior (Dirichlet)")
def test_weight_prior():
    prior = torch.ones(3) * 2.0
    gmm = GaussianMixture(n_components=3, weight_concentration_prior=prior)
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.use_weight_prior

@test("Mean prior")
def test_mean_prior():
    mean_prior = torch.zeros(3)  # (n_features,) - will be broadcast
    gmm = GaussianMixture(
        n_components=3,
        n_features=3,
        mean_prior=mean_prior,
        mean_precision_prior=0.1
    )
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.use_mean_prior

@test("Covariance prior (Wishart)")
def test_covariance_prior():
    cov_prior = torch.eye(3)
    gmm = GaussianMixture(
        n_components=3,
        n_features=3,
        covariance_type='full',
        covariance_prior=cov_prior,
        degrees_of_freedom_prior=5.0
    )
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.use_covariance_prior

@test("All priors combined")
def test_all_priors():
    gmm = GaussianMixture(
        n_components=3,
        n_features=3,
        covariance_type='full',
        weight_concentration_prior=torch.ones(3) * 2.0,
        mean_prior=torch.zeros(3),  # (n_features,) - will be broadcast
        mean_precision_prior=0.1,
        covariance_prior=torch.eye(3),
        degrees_of_freedom_prior=5.0
    )
    gmm.fit(X_3d)
    assert gmm.fitted_

test_weight_prior()
test_mean_prior()
test_covariance_prior()
test_all_priors()

# ============================================================================
# 9. MODEL METHODS
# ============================================================================

print("\n" + "="*80)
print("9. MODEL METHODS")
print("="*80)

gmm_fitted = GaussianMixture(n_components=3)
gmm_fitted.fit(X_3d)

@test("predict method")
def test_predict():
    labels = gmm_fitted.predict(X_3d)
    assert labels.shape == (X_3d.shape[0],)
    assert labels.min() >= 0
    assert labels.max() < 3

@test("predict_proba method")
def test_predict_proba():
    # Ensure X is on same device as model
    X_test = X_3d.to(gmm_fitted.device)
    probs = gmm_fitted.predict_proba(X_test)
    assert probs.shape == (X_test.shape[0], 3)
    # Ensure comparison is on same device
    expected = torch.ones(X_test.shape[0], device=gmm_fitted.device)
    assert torch.allclose(probs.sum(dim=1), expected, atol=1e-5)

@test("score method")
def test_score():
    score = gmm_fitted.score(X_3d)
    assert isinstance(score, float)

@test("score_samples method")
def test_score_samples():
    scores = gmm_fitted.score_samples(X_3d)
    assert scores.shape == (X_3d.shape[0],)

@test("sample method")
def test_sample():
    samples, labels = gmm_fitted.sample(100)
    assert samples.shape == (100, 3)
    assert labels.shape == (100,)

# Note: bic and aic are in metrics.py, not methods of GaussianMixture

test_predict()
test_predict_proba()
test_score()
test_score_samples()
test_sample()

# ============================================================================
# 10. SAVE/LOAD
# ============================================================================

print("\n" + "="*80)
print("10. SAVE/LOAD")
print("="*80)

@test("save and load model")
def test_save_load():
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_3d)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        filepath = f.name
    
    try:
        # Save
        gmm.save(filepath)
        assert os.path.exists(filepath)
        
        # Load
        gmm_loaded = GaussianMixture.load(filepath)
        assert gmm_loaded.fitted_
        assert torch.allclose(gmm_loaded.means_, gmm.means_, atol=1e-5)
        assert torch.allclose(gmm_loaded.weights_, gmm.weights_, atol=1e-5)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@test("to_dict and load_state_dict")
def test_state_dict():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_3d)
    
    state = gmm.to_dict()
    assert isinstance(state, dict)
    assert 'means_' in state
    assert 'weights_' in state
    
    gmm2 = GaussianMixture(n_components=3)
    gmm2.load_state_dict(state)
    assert torch.allclose(gmm2.means_, gmm.means_, atol=1e-5)

test_save_load()
test_state_dict()

# ============================================================================
# 11. DIFFERENT DATA DIMENSIONS
# ============================================================================

print("\n" + "="*80)
print("11. DIFFERENT DATA DIMENSIONS")
print("="*80)

@test("2D data")
def test_2d_data():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_2d)
    assert gmm.fitted_
    assert gmm.means_.shape == (3, 2)

@test("3D data")
def test_3d_data():
    gmm = GaussianMixture(n_components=4)
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.means_.shape == (4, 3)

@test("5D data")
def test_5d_data():
    gmm = GaussianMixture(n_components=5)
    gmm.fit(X_5d)
    assert gmm.fitted_
    assert gmm.means_.shape == (5, 5)

@test("10D data")
def test_10d_data():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_10d)
    assert gmm.fitted_
    assert gmm.means_.shape == (3, 10)

@test("1D data")
def test_1d_data():
    X_1d = torch.randn(100, 1)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X_1d)
    assert gmm.fitted_
    assert gmm.means_.shape == (2, 1)

test_2d_data()
test_3d_data()
test_5d_data()
test_10d_data()
test_1d_data()

# ============================================================================
# 12. EDGE CASES
# ============================================================================

print("\n" + "="*80)
print("12. EDGE CASES")
print("="*80)

@test("Single component")
def test_single_component():
    gmm = GaussianMixture(n_components=1)
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("Many components")
def test_many_components():
    gmm = GaussianMixture(n_components=10)
    gmm.fit(X_10d)
    assert gmm.fitted_

@test("Small dataset")
def test_small_dataset():
    X_small = torch.randn(20, 2)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X_small)
    assert gmm.fitted_

@test("n_features specification")
def test_n_features():
    gmm = GaussianMixture(n_components=3, n_features=3)
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.n_features == 3

@test("Device specification (CPU)")
def test_device_cpu():
    gmm = GaussianMixture(n_components=3, device='cpu')
    gmm.fit(X_3d)
    assert gmm.fitted_
    assert gmm.device.type == 'cpu'

@test("Very small regularization")
def test_small_reg_covar():
    gmm = GaussianMixture(n_components=3, reg_covar=1e-10)
    gmm.fit(X_3d)
    assert gmm.fitted_

@test("Large regularization")
def test_large_reg_covar():
    gmm = GaussianMixture(n_components=3, reg_covar=1.0)
    gmm.fit(X_3d)
    assert gmm.fitted_

test_single_component()
test_many_components()
test_small_dataset()
test_n_features()
test_device_cpu()
test_small_reg_covar()
test_large_reg_covar()

# ============================================================================
# 13. ERROR HANDLING
# ============================================================================

print("\n" + "="*80)
print("13. ERROR HANDLING")
print("="*80)

@test("Deprecated init_params raises error")
def test_deprecated_init_params():
    try:
        gmm = GaussianMixture(n_components=3, init_params='kmeans')
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'init_params' in str(e)
        assert 'init_means' in str(e)

@test("Deprecated cov_init_method raises error")
def test_deprecated_cov_init_method():
    try:
        gmm = GaussianMixture(n_components=3, cov_init_method='empirical')
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'cov_init_method' in str(e)
        assert 'init_covariances' in str(e)

@test("Deprecated weights_init raises error")
def test_deprecated_weights_init():
    try:
        gmm = GaussianMixture(n_components=3, weights_init=torch.ones(3))
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'weights_init' in str(e)
        assert 'init_weights' in str(e)

@test("Invalid init_means method")
def test_invalid_init_means():
    try:
        gmm = GaussianMixture(n_components=3, init_means='invalid_method')
        gmm.fit(X_3d)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert 'invalid_method' in str(e)

@test("Wrong tensor shape for init_means")
def test_wrong_means_shape():
    try:
        gmm = GaussianMixture(n_components=3, init_means=torch.randn(2, 3))
        gmm.fit(X_3d)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert 'shape' in str(e).lower()

@test("Wrong tensor shape for init_weights")
def test_wrong_weights_shape():
    try:
        gmm = GaussianMixture(n_components=3, init_weights=torch.randn(2))
        gmm.fit(X_3d)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert 'shape' in str(e).lower()

@test("Invalid covariance type")
def test_invalid_covariance_type():
    try:
        gmm = GaussianMixture(n_components=3, covariance_type='invalid')
        gmm.fit(X_3d)
        assert False, "Should raise error"
    except Exception:
        pass  # Expected to fail

test_deprecated_init_params()
test_deprecated_cov_init_method()
test_deprecated_weights_init()
test_invalid_init_means()
test_wrong_means_shape()
test_wrong_weights_shape()
test_invalid_covariance_type()

# ============================================================================
# 14. STRESS TESTS
# ============================================================================

print("\n" + "="*80)
print("14. STRESS TESTS")
print("="*80)

@test("All covariance types with all init combinations")
def test_all_combinations():
    cov_types = ['full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical']
    mean_inits = ['kmeans', 'kpp', 'random']
    weight_inits = ['uniform', 'random', 'kmeans']
    cov_inits = ['empirical', 'eye', 'global']
    
    combinations_tested = 0
    for cov_type in cov_types:
        for mean_init in mean_inits:
            for weight_init in weight_inits:
                for cov_init in cov_inits:
                    gmm = GaussianMixture(
                        n_components=3,
                        covariance_type=cov_type,
                        init_means=mean_init,
                        init_weights=weight_init,
                        init_covariances=cov_init,
                        max_iter=10  # Keep it fast
                    )
                    gmm.fit(X_3d)
                    assert gmm.fitted_
                    combinations_tested += 1
    
    print(f"    Tested {combinations_tested} combinations successfully")

@test("Multiple random seeds")
def test_multiple_seeds():
    for seed in [0, 1, 42, 123, 999]:
        gmm = GaussianMixture(n_components=3, random_state=seed, max_iter=10)
        gmm.fit(X_3d)
        assert gmm.fitted_

@test("Different n_init values")
def test_different_n_init():
    for n_init in [1, 2, 5, 10]:
        gmm = GaussianMixture(n_components=3, n_init=n_init, random_state=42, max_iter=10)
        gmm.fit(X_3d)
        assert gmm.fitted_

test_all_combinations()
test_multiple_seeds()
test_different_n_init()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Total tests: {total_tests}")
print(f"Passed: {passed_tests} ✅")
print(f"Failed: {failed_tests} ❌")
print(f"Success rate: {100 * passed_tests / total_tests:.1f}%")
print("="*80)

if failed_tests == 0:
    print("\n🎉 ALL TESTS PASSED! The GMM implementation is working correctly.")
    exit(0)
else:
    print(f"\n⚠️  {failed_tests} test(s) failed. Please review the errors above.")
    exit(1)
