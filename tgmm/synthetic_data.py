"""Synthetic data generation for Gaussian Mixture Models."""

import numpy as np
import torch
from typing import List, Union, Optional


def _matrix_square_root(cov) -> np.ndarray:
    """A factor L with L @ L.T == cov: Cholesky if cov is positive definite, else the symmetric eigen-root."""
    cov = np.asarray(cov, dtype=np.float64)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh((cov + cov.T) / 2.0)
        if eigvals.min() < -1e-10 * max(1.0, abs(eigvals).max()):
            raise ValueError("covariance matrices must be positive semi-definite")
        return eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))


def generate_gmm_data(
    centers: Union[List[np.ndarray], np.ndarray],
    covs: Union[List[np.ndarray], np.ndarray],
    n_samples: Union[List[int], np.ndarray],
    device: str = 'cpu',
    random_state: Optional[int] = None,
    noise_ratio: float = 0.0,
    noise_scale: float = 1.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data from a Gaussian Mixture Model.
    
    Parameters
    ----------
    centers : list of np.ndarray or np.ndarray
        List of mean vectors for each component. Each element should be a 1D array
        of shape (n_features,).
    covs : list of np.ndarray or np.ndarray
        List of covariance matrices for each component. Each element should be a
        symmetric positive semi-definite 2D array of shape (n_features, n_features);
        the generated component has exactly this covariance.
    n_samples : list of int or np.ndarray
        Number of samples to generate for each component.
    device : str, default='cpu'
        Device to place the output tensors on ('cpu' or 'cuda').
    random_state : int, optional
        Random seed for reproducibility. If None, no seed is set.
    noise_ratio : float, default=0.0
        Fraction of noise points to add (uniformly distributed). For example, 
        0.1 adds 10% noise points relative to the total number of Gaussian samples.
    noise_scale : float, default=1.5
        Scale factor for the noise region. The noise will be uniformly distributed
        in a box that extends noise_scale times the range of the Gaussian data.
    
    Returns
    -------
    X : torch.Tensor
        Generated data of shape (total_samples, n_features).
    labels : torch.Tensor
        True component labels of shape (total_samples,).
    
    Examples
    --------
    >>> import numpy as np
    >>> centers = [np.array([0, 0]), np.array([5, 5])]
    >>> covs = [np.eye(2), 2 * np.eye(2)]
    >>> n_samples = [100, 150]
    >>> X, labels = generate_gmm_data(centers, covs, n_samples)
    >>> X.shape
    torch.Size([250, 2])
    
    >>> # With 10% uniform noise
    >>> X, labels = generate_gmm_data(centers, covs, n_samples, noise_ratio=0.1)
    >>> X.shape
    torch.Size([275, 2])
    >>> (labels == -1).sum()  # 25 noise points
    tensor(25)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert inputs to lists if they aren't already
    if isinstance(centers, np.ndarray) and centers.ndim == 1:
        centers = [centers]
    if isinstance(covs, np.ndarray) and covs.ndim == 2:
        covs = [covs]
    if isinstance(n_samples, (int, np.integer)):
        n_samples = [n_samples]
    
    # Validate inputs
    n_components = len(centers)
    if len(covs) != n_components or len(n_samples) != n_components:
        raise ValueError(
            f"Length mismatch: centers ({len(centers)}), "
            f"covs ({len(covs)}), n_samples ({len(n_samples)})"
        )
    
    # Generate samples for each component
    components = []
    for n, center, cov in zip(n_samples, centers, covs):
        # X = Z @ L^T + center with Z ~ N(0, I) and L L^T = cov, so Cov(X) = cov exactly.
        # (Earlier versions used X = Z @ cov, whose covariance is cov^T cov -- the *square* of cov.)
        samples = np.random.randn(n, len(center)) @ _matrix_square_root(cov).T + center
        components.append(samples)
    
    # Combine all components
    X = np.vstack(components)
    labels = np.concatenate([i * np.ones(n) for i, n in enumerate(n_samples)])
    
    # Add uniform noise if requested
    if noise_ratio > 0.0:
        total_samples = sum(n_samples)
        n_noise = int(total_samples * noise_ratio)
        
        if n_noise > 0:
            # Determine the bounding box of the data
            data_min = X.min(axis=0)
            data_max = X.max(axis=0)
            data_range = data_max - data_min
            
            # Expand the box by noise_scale
            noise_min = data_min - (noise_scale - 1) * data_range / 2
            noise_max = data_max + (noise_scale - 1) * data_range / 2
            
            # Generate uniform noise
            n_features = X.shape[1]
            noise_samples = np.random.uniform(
                noise_min, 
                noise_max, 
                size=(n_noise, n_features)
            )
            
            # Add noise to data
            X = np.vstack([X, noise_samples])
            labels = np.concatenate([labels, -1 * np.ones(n_noise)])
    
    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    return X_tensor, labels_tensor
