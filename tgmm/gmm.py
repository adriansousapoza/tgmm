import torch
from torch import nn
from torch.distributions import MultivariateNormal
from typing import Optional, Tuple
import warnings
import math
from .gmm_init import GMMInitializer
from scipy.stats import chi2



class GaussianMixture(nn.Module):
    r"""
    A Gaussian Mixture Model (GMM) based on Expectation-Maximisation (EM) implemented in PyTorch.

    This GMM supports:
    - The Expectation-Maximization (EM) algorithm
    - The Classification EM (CEM) algorithm as an alternative to standard EM
    - Multiple random initializations (n_init)
    - Configurable covariance types (full, diag, spherical, tied_full, tied_diag, tied_spherical)
    - Maximum Likelihood Estimation (MLE) and Maximum a Posteriori (MAP) estimation with priors

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. (default: 1)
    n_features : int, optional
        Dimensionality of the input data (number of features). If None, inferred from data. (default: None)
    covariance_type : str, optional
        Type of covariance parameters to use. Must be one of:
        'full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical'.
        Aliases: 'tied' → 'tied_full', 'isotropic' → 'spherical'. (default: 'full')
    max_iter : int, optional
        Maximum number of EM iterations to perform. (default: 1000)
    tol : float, optional
        Convergence threshold based on relative improvement in log-likelihood. (default: 1e-4)
    reg_covar : float, optional
        Non-negative regularization added to the diagonal of covariance matrices
        to prevent singularity. (default: 1e-6)
    n_init : int, optional
        Number of random initializations to try. The best run (highest log-likelihood)
        is kept. When n_init > 1, each initialization uses random_state + i as its seed
        to ensure both reproducibility and diversity. (default: 1)
    init_means : str or torch.Tensor, optional
        Method for initializing means, or a tensor of initial means.
        - If str: 'kmeans', 'kpp', 'random', 'points', 'maxdist'
        - If tensor: shape (n_components, n_features) or (n_features,) to broadcast
        (default: 'kmeans')
    init_weights : str or torch.Tensor, optional
        Method for initializing weights, or a tensor of initial weights.
        - If str: 'uniform', 'random', 'kmeans'
        - If tensor: shape (n_components,)
        (default: 'uniform')
    init_covariances : str or torch.Tensor, optional
        Method for initializing covariances, or a tensor of initial covariances.
        - If str: 'empirical', 'eye', 'random', 'global'
        - If tensor: shape depends on covariance_type (supports broadcasting)
          * full: (n_features, n_features) or (n_components, n_features, n_features)
          * diag: (n_features,) or (n_components, n_features)
          * spherical: scalar or (n_components,)
        (default: 'empirical')
    random_state : int or None, optional
        Random seed for reproducibility. If None, uses PyTorch's internal seed. 
        When n_init > 1, serves as the base seed (initialization i uses random_state + i).
        (default: None)
    warm_start : bool, optional
        If True, reuse the solution of the previous call to `fit` as initialization.
        (default: False)
    cem : bool, optional
        If True, use the Classification EM (hard assignment) algorithm instead of standard EM.
        (default: False)
    weight_concentration_prior : torch.Tensor or None, optional
        Dirichlet concentration prior for the mixture weights (MAP estimation).
        Shape: (n_components,) or scalar to broadcast. (default: None)
    mean_prior : torch.Tensor or None, optional
        Prior for the component means (MAP estimation). Must be used with mean_precision_prior.
        Shape: (n_features,) or (n_components, n_features). (default: None)
    mean_precision_prior : float or None, optional
        Precision (inverse variance) of the mean prior (MAP estimation). (default: None)
    covariance_prior : torch.Tensor or None, optional
        Prior for the covariances (MAP estimation). Must be used with degrees_of_freedom_prior.
        Shape depends on covariance_type (supports broadcasting like init_covariances).
        (default: None)
    degrees_of_freedom_prior : float or None, optional
        Degrees of freedom for the Wishart/Inverse-Wishart prior on covariances (MAP).
        Must be > n_features - 1. (default: None)
    verbose : bool, optional
        If True, print progress during EM iterations. (default: False)
    verbose_interval : int, optional
        Frequency (in iterations) at which to print progress when verbose=True. (default: 10)
    device : str or None, optional
        Device on which to run computations ('cpu' or 'cuda'). If None, uses GPU if
        available, otherwise CPU. (default: None)

    Attributes
    ----------
    weights_ : torch.Tensor
        Mixture component weights of shape (n_components,).
    means_ : torch.Tensor
        Mixture component means of shape (n_components, n_features).
    covariances_ : torch.Tensor
        Mixture component covariances. Shape depends on `covariance_type`.
    initial_weights_ : torch.Tensor
        Initial mixture component weights before EM optimization, shape (n_components,).
    initial_means_ : torch.Tensor
        Initial mixture component means before EM optimization, shape (n_components, n_features).
    initial_covariances_ : torch.Tensor
        Initial mixture component covariances before EM optimization.
    fitted_ : bool
        Whether the model has been fitted.
    converged_ : bool
        Whether the EM algorithm converged in the best run.
    n_iter_ : int
        Number of EM iterations performed in the best run.
    lower_bound_ : float
        Log-likelihood lower bound on the fitted data for the best run.
    best_random_state_ : int or None
        The random state that produced the best result when n_init > 1.
        Useful for reproducing the specific best initialization.
    """

    def __init__(
        self,
        # Core model parameters
        n_components: int = 1,
        n_features: int = None,
        covariance_type: str = 'full',
        
        # Convergence and training parameters
        max_iter: int = 1000,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        
        # Initialization parameters (accepts str method or torch.Tensor values)
        init_means='kmeans',
        init_weights='uniform',
        init_covariances='empirical',
        
        # Random state and restart options
        random_state: int = None,
        warm_start: bool = False,
        
        # Algorithm options
        cem: bool = False,
        
        # Prior parameters for MAP estimation
        weight_concentration_prior: torch.Tensor = None,
        mean_prior: torch.Tensor = None,
        mean_precision_prior: float = None,
        covariance_prior: torch.Tensor = None,
        degrees_of_freedom_prior: float = None,
        
        # Output and device options
        verbose: bool = False,
        verbose_interval: int = 10,
        device: str = None,
        
        **kwargs  # Catch deprecated parameters
    ):
        super().__init__()
        
        # ===================================================================
        # 1. Validate deprecated parameters
        # ===================================================================
        deprecated_params = {
            'init_params': ('init_means', 'init_means=\'kmeans\''),
            'cov_init_method': ('init_covariances', 'init_covariances=\'empirical\''),
            'weights_init': ('init_weights', 'init_weights=torch.tensor([...])'),
            'means_init': ('init_means', 'init_means=torch.tensor([...])'),
            'covariances_init': ('init_covariances', 'init_covariances=torch.tensor([...])')
        }
        
        for old_param, (new_param, example) in deprecated_params.items():
            if old_param in kwargs:
                raise TypeError(
                    f"Parameter '{old_param}' has been removed. "
                    f"Use '{new_param}' instead.\n"
                    f"The '{new_param}' parameter accepts both strings (method names) and tensors (explicit values).\n"
                    f"Example: {example}"
                )
        
        if kwargs:
            unexpected = ', '.join(f"'{k}'" for k in kwargs.keys())
            raise TypeError(f"GaussianMixture.__init__() got unexpected keyword argument(s): {unexpected}")

        # ===================================================================
        # 2. Store core model parameters
        # ===================================================================
        self.n_components = n_components
        self.n_features = n_features
        
        # Handle covariance type aliases
        if covariance_type == "tied":
            covariance_type = "tied_full"
        elif covariance_type == "isotropic":
            covariance_type = "spherical"
        self.covariance_type = covariance_type

        # ===================================================================
        # 3. Store convergence and training parameters
        # ===================================================================
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        
        # ===================================================================
        # 4. Store initialization parameters
        # ===================================================================
        self.init_means = init_means
        self.init_weights = init_weights
        self.init_covariances = init_covariances
        
        # ===================================================================
        # 5. Store random state and restart options
        # ===================================================================
        self.random_state = random_state
        self.warm_start = warm_start
        
        # ===================================================================
        # 6. Store algorithm options
        # ===================================================================
        self.cem = cem
        
        # ===================================================================
        # 7. Configure device
        # ===================================================================
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ===================================================================
        # 8. Store output options
        # ===================================================================
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        # ===================================================================
        # 9. Initialize and validate priors
        # ===================================================================
        self.use_weight_prior = weight_concentration_prior is not None
        self.use_mean_prior = (mean_prior is not None) and (mean_precision_prior is not None)
        self.use_covariance_prior = (covariance_prior is not None) and (degrees_of_freedom_prior is not None)

        self._init_priors(
            weight_concentration_prior,
            mean_prior,
            mean_precision_prior,
            covariance_prior,
            degrees_of_freedom_prior
        )

        # ===================================================================
        # 10. Initialize model state variables
        # ===================================================================
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.initial_weights_ = None
        self.initial_means_ = None
        self.initial_covariances_ = None
        self.fitted_ = False
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")
        self.best_random_state_ = None

    def _init_priors(
        self,
        weight_concentration_prior: Optional[torch.Tensor],
        mean_prior: Optional[torch.Tensor],
        mean_precision_prior: Optional[float],
        covariance_prior: Optional[torch.Tensor],
        degrees_of_freedom_prior: Optional[float]
    ):
        r"""
        Validate and store prior parameters for MAP estimation.

        This method validates the shapes and values of prior parameters and stores them
        for use during the M-step. Supports broadcasting for convenience.

        Parameters
        ----------
        weight_concentration_prior : torch.Tensor or None
            Dirichlet concentration prior for mixture weights.
        mean_prior : torch.Tensor or None
            Prior means for Gaussian components.
        mean_precision_prior : float or None
            Precision (inverse variance) for the mean prior.
        covariance_prior : torch.Tensor or None
            Prior for covariances (shape depends on covariance_type).
        degrees_of_freedom_prior : float or None
            Degrees of freedom for the Wishart/Inverse-Wishart prior.
        """
        # ---------------------------------------------------------------
        # Weight prior (Dirichlet)
        # ---------------------------------------------------------------
        if self.use_weight_prior:
            if not isinstance(weight_concentration_prior, torch.Tensor):
                weight_concentration_prior = torch.tensor(weight_concentration_prior, device=self.device)
            
            # Broadcast scalar or single value to all components
            if weight_concentration_prior.dim() == 0 or (weight_concentration_prior.dim() == 1 and weight_concentration_prior.numel() == 1):
                weight_concentration_prior = weight_concentration_prior.expand(self.n_components)
            elif weight_concentration_prior.dim() == 1 and weight_concentration_prior.numel() != self.n_components:
                raise ValueError(
                    f"weight_concentration_prior must be of shape ({self.n_components},) or a scalar, "
                    f"got {weight_concentration_prior.shape}."
                )
            self.weight_concentration_prior = weight_concentration_prior.to(self.device).float()
        else:
            self.weight_concentration_prior = None

        # ---------------------------------------------------------------
        # Mean prior (Gaussian)
        # ---------------------------------------------------------------
        if self.use_mean_prior:
            # Broadcast (n_features,) to (n_components, n_features)
            if mean_prior.shape == (self.n_features,):
                mean_prior = mean_prior.unsqueeze(0).expand(self.n_components, -1)
            elif mean_prior.shape != (self.n_components, self.n_features):
                raise ValueError(
                    f"mean_prior must be of shape ({self.n_components}, {self.n_features}) "
                    f"or ({self.n_features},). Got {mean_prior.shape}."
                )
            if mean_precision_prior <= 0:
                raise ValueError("mean_precision_prior must be > 0.")
            
            self.mean_prior = mean_prior.to(self.device).float()
            self.mean_precision_prior = float(mean_precision_prior)
        else:
            self.mean_prior = None
            self.mean_precision_prior = None

        # ---------------------------------------------------------------
        # Covariance prior (Wishart/Inverse-Wishart)
        # ---------------------------------------------------------------
        if self.use_covariance_prior:
            self.degrees_of_freedom_prior = float(degrees_of_freedom_prior)
            
            # Validate degrees of freedom
            if self.degrees_of_freedom_prior <= self.n_features - 1:
                raise ValueError(
                    f"degrees_of_freedom_prior must be > {self.n_features - 1}, "
                    f"got {self.degrees_of_freedom_prior}."
                )
            
            expected_shape = self._expected_covar_shape()
            
            # Handle broadcasting for non-tied covariance types
            if self.covariance_type == 'full':
                if covariance_prior.shape == (self.n_features, self.n_features):
                    # Broadcast single matrix to all components
                    covariance_prior = covariance_prior.unsqueeze(0).expand(
                        self.n_components, self.n_features, self.n_features
                    ).clone()
                elif covariance_prior.shape != expected_shape:
                    raise ValueError(
                        f"covariance_prior must be of shape {expected_shape} or "
                        f"({self.n_features}, {self.n_features}) for 'full' covariance. "
                        f"Got {covariance_prior.shape}."
                    )
            
            elif self.covariance_type == 'diag':
                if covariance_prior.shape == (self.n_features,):
                    # Broadcast single vector to all components
                    covariance_prior = covariance_prior.unsqueeze(0).expand(
                        self.n_components, self.n_features
                    ).clone()
                elif covariance_prior.shape != expected_shape:
                    raise ValueError(
                        f"covariance_prior must be of shape {expected_shape} or "
                        f"({self.n_features},) for 'diag' covariance. "
                        f"Got {covariance_prior.shape}."
                    )
            
            elif self.covariance_type == 'spherical':
                if covariance_prior.ndim == 0:
                    # Broadcast scalar to all components
                    covariance_prior = covariance_prior.expand(self.n_components).clone()
                elif covariance_prior.shape != expected_shape:
                    raise ValueError(
                        f"covariance_prior must be of shape {expected_shape} or a scalar "
                        f"for 'spherical' covariance. Got {covariance_prior.shape}."
                    )
            
            else:
                # Tied covariance types: no broadcasting (single shared covariance)
                if covariance_prior.shape != expected_shape:
                    raise ValueError(
                        f"covariance_prior must be of shape {expected_shape} for "
                        f"'{self.covariance_type}' covariance. Got {covariance_prior.shape}."
                    )
            
            self.covariance_prior = covariance_prior.to(self.device).float()
        else:
            self.degrees_of_freedom_prior = None
            self.covariance_prior = None

        # ---------------------------------------------------------------
        # Validate and report NIW prior usage
        # ---------------------------------------------------------------
        if self.use_mean_prior and self.use_covariance_prior:
            # Normal-Inverse-Wishart conjugate priors for joint estimation
            if self.verbose:
                print("INFO: Using Normal-Inverse-Wishart (NIW) conjugate priors for joint mean-covariance estimation.")
        elif self.use_mean_prior:
            if self.verbose:
                print("INFO: Using Gaussian prior for means only.")
        elif self.use_covariance_prior:
            if self.verbose:
                print("INFO: Using Inverse-Wishart prior for covariances only.")

    def _expected_covar_shape(self) -> Tuple:
        r"""
        Return the expected shape of covariances_ given self.covariance_type.

        Returns
        -------
        shape : Tuple
            The shape that self.covariances_ should have for the specified
            covariance type.
        """
        if self.covariance_type == 'full':
            return (self.n_components, self.n_features, self.n_features)
        elif self.covariance_type == 'diag':
            return (self.n_components, self.n_features)
        elif self.covariance_type == 'spherical':
            return (self.n_components,)
        elif self.covariance_type == 'tied_full':
            return (self.n_features, self.n_features)
        elif self.covariance_type == 'tied_diag':
            return (self.n_features,)
        elif self.covariance_type == 'tied_spherical':
            return ()  # Single scalar for entire dataset
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def _allocate_parameters(self, X: Optional[torch.Tensor] = None, set_random_state: bool = True):
        r"""
        Allocate and initialize model parameters (means, weights, covariances).

        Parameters are initialized in a specific order because some initialization
        methods depend on others (e.g., kmeans weight initialization needs means first).

        Parameters
        ----------
        X : torch.Tensor, optional
            Input data for data-based initialization methods. If None, uses random initialization.
        set_random_state : bool, optional
            Whether to set the random state. Set to False when random state is already
            set externally (e.g., for multiple initializations in n_init > 1). (default: True)

        Notes
        -----
        Initialization order:
        1. Means (required by some weight and covariance methods)
        2. Weights (may depend on means for kmeans method)
        3. Covariances (may depend on means for empirical method)
        """
        # ===============================================================
        # Set random seed if requested
        # ===============================================================
        if set_random_state and self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        # ===============================================================
        # 1. Initialize means
        # ===============================================================
        if isinstance(self.init_means, torch.Tensor):
            # User-provided tensor
            if self.init_means.shape != (self.n_components, self.n_features):
                raise ValueError(
                    f"init_means tensor must be shape ({self.n_components}, {self.n_features}), "
                    f"got {self.init_means.shape}."
                )
            self.means_ = self.init_means.to(self.device).float()
        
        elif isinstance(self.init_means, str):
            # Initialization method
            if X is None:
                # No data: fallback to random
                self.means_ = torch.randn(
                    self.n_components,
                    self.n_features,
                    device=self.device
                ).float()
            else:
                # Data-based initialization
                X_cpu = X.cpu()
                init_method = self.init_means.lower()
                
                if init_method == 'kmeans':
                    self.means_ = GMMInitializer.kmeans(X_cpu, self.n_components).to(self.device)
                elif init_method == 'kpp':
                    self.means_ = GMMInitializer.kpp(X_cpu, self.n_components).to(self.device)
                elif init_method == 'points':
                    self.means_ = GMMInitializer.points(X_cpu, self.n_components).to(self.device)
                elif init_method == 'maxdist':
                    self.means_ = GMMInitializer.maxdist(X_cpu, self.n_components).to(self.device)
                elif init_method == 'random':
                    self.means_ = GMMInitializer.random(X_cpu, self.n_components).to(self.device)
                else:
                    raise ValueError(
                        f"Unsupported init_means method: '{init_method}'. "
                        f"Supported: 'kmeans', 'kpp', 'points', 'maxdist', 'random'."
                    )
        else:
            raise TypeError(
                f"init_means must be a string (method name) or torch.Tensor, got {type(self.init_means)}"
            )

        self.initial_means_ = self.means_.clone().detach()

        # ===============================================================
        # 2. Initialize weights
        # ===============================================================
        if isinstance(self.init_weights, torch.Tensor):
            # User-provided tensor
            if self.init_weights.shape != (self.n_components,):
                raise ValueError(
                    f"init_weights tensor must be shape ({self.n_components},), "
                    f"got {self.init_weights.shape}."
                )
            weights = self.init_weights.to(self.device).float()
            if torch.sum(weights) < 1e-20:
                raise ValueError("Initial weights must sum to > 0.")
            self.weights_ = weights / torch.sum(weights)
        elif isinstance(self.init_weights, str):
            # Use initialization method from GMMInitializer
            init_method = self.init_weights.lower()
            
            if init_method in ('uniform', 'equal'):
                self.weights_ = GMMInitializer.init_weights_uniform(
                    self.n_components, self.device
                )
            elif init_method == 'random':
                self.weights_ = GMMInitializer.init_weights_random(
                    self.n_components, self.device
                )
            elif init_method == 'kmeans':
                if X is None:
                    warnings.warn(
                        "'kmeans' weight initialization requires data. "
                        "Falling back to uniform weights.",
                        UserWarning
                    )
                    self.weights_ = GMMInitializer.init_weights_uniform(
                        self.n_components, self.device
                    )
                else:
                    # Ensure data is on the same device as means for init_weights_from_clusters
                    X_for_weights = X.to(self.means_.device)
                    weights = GMMInitializer.init_weights_from_clusters(
                        X_for_weights, self.means_
                    )
                    self.weights_ = weights.to(self.device)
            else:
                raise ValueError(
                    f"Unsupported init_weights method: '{init_method}'. "
                    f"Supported methods: 'uniform', 'random', 'kmeans'."
                )
        else:
            raise TypeError(
                f"init_weights must be a string (method name) or torch.Tensor, got {type(self.init_weights)}"
            )

        # Store the initial weights for later access
        self.initial_weights_ = self.weights_.clone().detach()

        # ----------------------
        # 3) Allocate covariances
        # ----------------------
        if isinstance(self.init_covariances, torch.Tensor):
            # User provided explicit tensor
            expected_shape = self._expected_covar_shape()
            init_cov = self.init_covariances
            
            # Allow broadcasting: if user provides a single covariance matrix for non-tied types,
            # expand it to all components
            if self.covariance_type == 'full':
                # Can be (n_features, n_features) or (n_components, n_features, n_features)
                if init_cov.shape == (self.n_features, self.n_features):
                    # Broadcast to all components
                    init_cov = init_cov.unsqueeze(0).expand(
                        self.n_components, self.n_features, self.n_features
                    ).clone()
                elif init_cov.shape != expected_shape:
                    raise ValueError(
                        f"init_covariances tensor must be of shape {expected_shape} or "
                        f"({self.n_features}, {self.n_features}) for 'full' covariance type. "
                        f"Got {init_cov.shape}."
                    )
            elif self.covariance_type == 'diag':
                # Can be (n_features,) or (n_components, n_features)
                if init_cov.shape == (self.n_features,):
                    # Broadcast to all components
                    init_cov = init_cov.unsqueeze(0).expand(
                        self.n_components, self.n_features
                    ).clone()
                elif init_cov.shape != expected_shape:
                    raise ValueError(
                        f"init_covariances tensor must be of shape {expected_shape} or "
                        f"({self.n_features},) for 'diag' covariance type. "
                        f"Got {init_cov.shape}."
                    )
            elif self.covariance_type == 'spherical':
                # Can be a scalar or (n_components,)
                if init_cov.ndim == 0:
                    # Broadcast scalar to all components
                    init_cov = init_cov.expand(self.n_components).clone()
                elif init_cov.shape != expected_shape:
                    raise ValueError(
                        f"init_covariances tensor must be of shape {expected_shape} or a scalar "
                        f"for 'spherical' covariance type. Got {init_cov.shape}."
                    )
            else:
                # For tied types, no broadcasting needed
                if init_cov.shape != expected_shape:
                    raise ValueError(
                        f"init_covariances tensor must be of shape {expected_shape} for "
                        f"'{self.covariance_type}' covariance type. Got {init_cov.shape}."
                    )
            
            self.covariances_ = init_cov.to(self.device).float()
        elif isinstance(self.init_covariances, str):
            # Use initialization method from GMMInitializer
            init_method = self.init_covariances.lower()
            
            if init_method == 'eye':
                self.covariances_ = GMMInitializer.init_covariances_eye(
                    self.n_components, self.n_features, 
                    self.covariance_type, self.reg_covar, self.device
                )
            elif init_method == 'random':
                self.covariances_ = GMMInitializer.init_covariances_random(
                    self.n_components, self.n_features,
                    self.covariance_type, self.reg_covar, self.device
                )
            elif init_method == 'global':
                if X is None:
                    warnings.warn(
                        "Global covariance initialization requires data. "
                        "Falling back to identity initialization.",
                        UserWarning
                    )
                    self.covariances_ = GMMInitializer.init_covariances_eye(
                        self.n_components, self.n_features,
                        self.covariance_type, self.reg_covar, self.device
                    )
                else:
                    # Ensure data is on the correct device
                    X_for_cov = X.to(self.device)
                    covs = GMMInitializer.init_covariances_global(
                        X_for_cov, self.n_components,
                        self.covariance_type, self.reg_covar
                    )
                    self.covariances_ = covs.to(self.device)
            elif init_method == 'empirical':
                if X is None:
                    warnings.warn(
                        "Empirical covariance initialization requires data. "
                        "Falling back to identity initialization.",
                        UserWarning
                    )
                    self.covariances_ = GMMInitializer.init_covariances_eye(
                        self.n_components, self.n_features,
                        self.covariance_type, self.reg_covar, self.device
                    )
                else:
                    # Ensure data and means are on the same device
                    X_for_cov = X.to(self.means_.device)
                    covs = GMMInitializer.init_covariances_empirical(
                        X_for_cov, self.means_,
                        self.covariance_type, self.reg_covar
                    )
                    self.covariances_ = covs.to(self.device)
            else:
                raise ValueError(
                    f"Unsupported init_covariances method: '{init_method}'. "
                    f"Supported methods: 'eye', 'random', 'global', 'empirical'."
                )
        else:
            raise TypeError(
                f"init_covariances must be a string (method name) or torch.Tensor, "
                f"got {type(self.init_covariances)}"
            )

        # Store the initial covariances for later access
        self.initial_covariances_ = self.covariances_.clone().detach()

        # Mark that we've allocated
        self.fitted_ = False
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")


    def fit(
        self,
        X: torch.Tensor,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        random_state: Optional[int] = None,
        warm_start: Optional[bool] = None
    ) -> "GaussianMixture":
        r"""
        Fit the GMM to the data using the Expectation-Maximization algorithm.

        Supports multiple random initializations (n_init > 1) to find the best solution.
        The model with the highest log-likelihood is selected.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features).
        max_iter : int, optional
            Maximum number of EM iterations. Overrides `self.max_iter` if provided. (default: None)
        tol : float, optional
            Convergence tolerance. Overrides `self.tol` if provided. (default: None)
        random_state : int, optional
            Random seed. Overrides `self.random_state` if provided. (default: None)
        warm_start : bool, optional
            Whether to warm-start from previously fitted parameters. Overrides `self.warm_start`
            if provided. (default: None)

        Returns
        -------
        self : GaussianMixture
            The fitted model instance (allows method chaining).

        Raises
        ------
        ValueError
            If n_components > n_samples, or if parameters are invalid.
        """
        # ===============================================================
        # 1. Validate input parameters
        # ===============================================================
        if X.size(0) < self.n_components:
            raise ValueError(
                f"n_components={self.n_components} must be <= n_samples={X.size(0)}."
            )
        if self.n_components <= 0:
            raise ValueError(f"Invalid n_components: {self.n_components} (must be > 0).")
        if tol is not None and tol <= 0:
            raise ValueError(f"Invalid tol: {tol} (must be > 0).")
        if max_iter is not None and max_iter <= 0:
            raise ValueError(f"Invalid max_iter: {max_iter} (must be > 0).")

        # ===============================================================
        # 2. Set up parameters (use overrides or defaults)
        # ===============================================================
        warm_start = self.warm_start if warm_start is None else warm_start
        max_iter = self.max_iter if max_iter is None else max_iter
        tol = self.tol if tol is None else tol
        
        if random_state is not None:
            self.random_state = random_state

        # ===============================================================
        # 3. Prepare data
        # ===============================================================
        X = X.to(self.device)
        
        # Infer n_features from data if not set
        if self.n_features is None:
            self.n_features = X.shape[1]
        
        # Handle 1D data
        if X.dim() == 1:
            X = X.unsqueeze(1)
        
        # Validate feature dimension
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features}."
            )

        # ===============================================================
        # 4. Run multiple initializations (if n_init > 1)
        # ===============================================================
        best_lower_bound = -float("inf")
        best_params = None
        best_random_state = None

        # Warn about random state usage with multiple initializations
        if self.random_state is not None and self.n_init > 1:
            warnings.warn(
                f"With n_init={self.n_init} and random_state={self.random_state}, "
                f"initializations will use random states [{self.random_state}, "
                f"{self.random_state + 1}, ..., {self.random_state + self.n_init - 1}]. "
                f"The best initialization's random state will be stored in best_random_state_.",
                UserWarning
            )
        
        for init_idx in range(self.n_init):
            # Validate warm_start usage
            if warm_start and self.n_init > 1:
                warnings.warn(
                    "warm_start=True with n_init > 1 will not re-initialize parameters "
                    "for each run, which may lead to identical results.",
                    UserWarning
                )

            # Set different random state for each initialization
            # This ensures diversity while maintaining reproducibility
            if self.random_state is not None:
                current_random_state = self.random_state + init_idx
                torch.manual_seed(current_random_state)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_random_state)

            # Initialize parameters if needed
            if not warm_start or not self.fitted_ or init_idx > 0:
                self._allocate_parameters(X, set_random_state=False)

            # Run single EM optimization
            self._fit_single_run(X, max_iter, tol, run_idx=init_idx)

            # Warn about degenerate clusters
            if torch.any(self.weights_ < 1e-8):
                warnings.warn(
                    "Some cluster(s) have near-zero weight. This may indicate degenerate solutions.",
                    UserWarning
                )

            # Track best solution
            if self.lower_bound_ > best_lower_bound:
                best_lower_bound = self.lower_bound_
                best_random_state = (self.random_state + init_idx 
                                   if self.random_state is not None else None)
                best_params = (
                    self.weights_.clone(),
                    self.means_.clone(),
                    self.covariances_.clone(),
                    self.converged_,
                    self.n_iter_,
                    self.lower_bound_
                )

            self.fitted_ = True

        # ===============================================================
        # 5. Save best result
        # ===============================================================
        if best_params is not None:
            (self.weights_, self.means_, self.covariances_, 
             self.converged_, self.n_iter_, self.lower_bound_) = best_params
            self.best_random_state_ = best_random_state
        
        # Report which random state produced the best result
        if best_random_state is not None and self.n_init > 1:
            warnings.warn(
                f"Best result from random_state={best_random_state}. "
                f"To reproduce this specific result: use random_state={best_random_state} with n_init=1.",
                UserWarning
            )
        
        # Warn if convergence failed
        if not self.converged_:
            warnings.warn(
                "EM did not converge. Consider increasing max_iter or adjusting tol.",
                UserWarning
            )

        return self

    def _fit_single_run(
        self,
        X: torch.Tensor,
        max_iter: int,
        tol: float,
        run_idx: int = 0
    ):
        r"""
        Perform one complete EM or CEM optimization run.

        Standard EM: E-step → M-step → repeat
        CEM (Classification EM): E-step → C-step (hard assignment) → M-step → repeat

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features).
        max_iter : int
            Maximum number of iterations for this run.
        tol : float
            Convergence tolerance based on relative improvement in log-likelihood.
        run_idx : int, optional
            Initialization index (for logging purposes). (default: 0)
        """
        # ===============================================================
        # Prepare data
        # ===============================================================
        X = X.to(self.device)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features}.")

        # ===============================================================
        # Initialize tracking variables
        # ===============================================================
        prev_lower_bound = -float("inf")
        
        # Initial E-step to compute responsibilities
        resp, log_prob_norm = self._e_step(X)
        self.lower_bound_ = log_prob_norm.mean().item()
        
        # ===============================================================
        # EM/CEM iteration loop
        # ===============================================================
        for n_iter in range(max_iter):
            # Classification step (only for CEM)
            if self.cem:
                resp = self._c_step(resp)
                
            # Maximization step
            self._m_step(X, resp)

            # Check convergence
            rel_change = abs(self.lower_bound_ - prev_lower_bound) / (abs(prev_lower_bound) + 1e-20)
            if rel_change < tol:
                self.converged_ = True
                if self.verbose:
                    print(f"[Run {run_idx+1}] Iteration {n_iter}: "
                          f"log-likelihood={self.lower_bound_:.5f}, Converged!")
                break
                
            # Update for next iteration
            prev_lower_bound = self.lower_bound_
            resp, log_prob_norm = self._e_step(X)
            self.lower_bound_ = log_prob_norm.mean().item()

            # Verbose output
            if self.verbose and (n_iter % self.verbose_interval == 0):
                print(f"[Run {run_idx+1}] Iteration {n_iter}: "
                      f"log-likelihood={self.lower_bound_:.5f}")

        # ===============================================================
        # Final E-step after convergence
        # ===============================================================
        if self.converged_:
            resp, log_prob_norm = self._e_step(X)
            if self.cem:
                resp = self._c_step(resp)
            self.lower_bound_ = log_prob_norm.mean().item()
        else:
            warnings.warn(
                f"Run {run_idx+1}: EM did not converge after {max_iter} iterations.",
                UserWarning
            )

        self.n_iter_ = n_iter
    
    def _c_step(self, resp: torch.Tensor) -> torch.Tensor:
        r"""
        C-step (Classification): Convert soft responsibilities to hard assignments.
        
        Used in CEM (Classification EM) algorithm to assign each sample to exactly
        one component (the one with highest responsibility).
        
        Parameters
        ----------
        resp : torch.Tensor
            Soft responsibilities from E-step, shape (n_samples, n_components).
            Each row sums to 1.0 and represents posterior probabilities.
            
        Returns
        -------
        hard_resp : torch.Tensor
            Hard assignments (one-hot encoding), shape (n_samples, n_components).
            Each row has exactly one 1.0 and rest 0.0.
            
        Notes
        -----
        This converts the probabilistic assignments from E-step into deterministic
        assignments, making CEM a "hard" version of EM that can converge faster
        but may get stuck in local optima more easily.
        """
        # ===============================================================
        # Find best component for each sample
        # ===============================================================
        max_resp_indices = torch.argmax(resp, dim=1)  # (n_samples,)
        
        # ===============================================================
        # Create one-hot encoding for hard assignments
        # ===============================================================
        n_samples = resp.size(0)
        hard_resp = torch.zeros_like(resp)
        hard_resp[torch.arange(n_samples, device=resp.device), max_resp_indices] = 1.0
        
        return hard_resp

    # ===================================================================
    # E-step: Expectation
    # ===================================================================
    def _e_step(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        E-step: Compute responsibilities (posterior probabilities) for each component.

        For each sample, compute the posterior probability that it belongs to each
        component using Bayes' rule:
        
        .. math::
            γ(z_{ik}) = \frac{π_k N(x_i | μ_k, Σ_k)}{\sum_j π_j N(x_i | μ_j, Σ_j)}
        
        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).

        Returns
        -------
        resp : torch.Tensor
            Responsibilities for each sample w.r.t. each component,
            shape (n_samples, n_components). Each row sums to 1.0.
        log_prob_norm : torch.Tensor
            Log-likelihood for each sample (normalizing constant),
            shape (n_samples,). These are log p(x_i).
            
        Notes
        -----
        Computation is done in log-space for numerical stability using log-sum-exp trick.
        """
        # ===============================================================
        # 1. Compute log p(x|z) for each component
        # ===============================================================
        if self.covariance_type == 'full':
            log_prob = self._estimate_log_gaussian_full(X)
        elif self.covariance_type == 'diag':
            log_prob = self._estimate_log_gaussian_diag(X)
        elif self.covariance_type == 'spherical':
            log_prob = self._estimate_log_gaussian_spherical(X)
        elif self.covariance_type == 'tied_full':
            log_prob = self._estimate_log_gaussian_tied_full(X)
        elif self.covariance_type == 'tied_diag':
            log_prob = self._estimate_log_gaussian_tied_diag(X)
        elif self.covariance_type == 'tied_spherical':
            log_prob = self._estimate_log_gaussian_tied_spherical(X)
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

        # ===============================================================
        # 2. Add log π_k (log of mixing weights)
        # ===============================================================
        log_weights = torch.log(self.weights_ + 1e-20)
        log_prob = log_prob + log_weights.unsqueeze(0)  # (n_samples, n_components)
        
        # ===============================================================
        # 3. Normalize to get responsibilities (log-sum-exp trick)
        # ===============================================================
        log_prob_norm = torch.logsumexp(log_prob, dim=1)  # (n_samples,)
        log_resp = log_prob - log_prob_norm.unsqueeze(1)
        resp = torch.exp(log_resp)  # (n_samples, n_components)
        
        return resp, log_prob_norm

    # ===================================================================
    # Log-Gaussian Probability Estimation
    # ===================================================================
    # These methods compute log p(x|z_k, θ_k) for each sample-component pair
    # using the multivariate Gaussian density:
    #
    #   log p(x|μ,Σ) = -½[d·log(2π) + log|Σ| + (x-μ)ᵀΣ⁻¹(x-μ)]
    #
    # All computations use log-space for numerical stability.
    # ===================================================================
    
    def _estimate_log_gaussian_full(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for full covariance matrices.

        Each component has its own full covariance matrix Σₖ ∈ ℝ^(d×d).
        Uses Cholesky decomposition for numerical stability and efficiency.
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
            
        Notes
        -----
        Cholesky decomposition: Σ = LLᵀ where L is lower triangular.
        log|Σ| = 2·sum(log(diag(L)))
        Σ⁻¹(x-μ) solved via triangular solves.
        """
        # ===============================================================
        # Compute deviations from component means
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        
        # ===============================================================
        # Cholesky decomposition: Σ = LLᵀ
        # ===============================================================
        try:
            chol = torch.linalg.cholesky(self.covariances_)  # (n_components, n_features, n_features)
        except RuntimeError as e:
            raise ValueError(f"Cholesky decomposition failed. Covariances may not be positive definite: {e}")

        # ===============================================================
        # Compute log determinant: log|Σ| = 2·sum(log(diag(L)))
        # ===============================================================
        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1)  # (n_components,)
        
        # ===============================================================
        # Compute Mahalanobis distance: (x-μ)ᵀΣ⁻¹(x-μ)
        # ===============================================================
        diff_ = diff.unsqueeze(-1)  # (n_samples, n_components, n_features, 1)
        solve = torch.cholesky_solve(diff_, chol)  # Σ⁻¹(x-μ)
        mahal = (diff_ * solve).sum(dim=(2, 3))  # (n_samples, n_components)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
        return -0.5 * (self.n_features * log_2pi + log_det.unsqueeze(0) + mahal)

    def _estimate_log_gaussian_diag(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for diagonal covariance matrices.

        Each component has a diagonal covariance: Σₖ = diag(σ²ₖ₁, ..., σ²ₖd).
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
        """
        # ===============================================================
        # Compute deviations and precisions (inverse variances)
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        precisions = 1.0 / (self.covariances_ + 1e-20)  # (n_components, n_features)
        
        # ===============================================================
        # Log determinant and Mahalanobis distance
        # ===============================================================
        log_det = torch.sum(torch.log(self.covariances_ + 1e-20), dim=1)  # (n_components,)
        mahal = torch.sum(diff.pow(2) * precisions.unsqueeze(0), dim=2)  # (n_samples, n_components)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
        return -0.5 * (self.n_features * log_2pi + log_det.unsqueeze(0) + mahal)

    def _estimate_log_gaussian_spherical(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for spherical covariance matrices.

        Each component has spherical covariance: Σₖ = σ²ₖ·I.
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
        """
        # ===============================================================
        # Compute deviations and squared distances
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        sq_dist = torch.sum(diff.pow(2), dim=2)  # (n_samples, n_components)
        
        # ===============================================================
        # Scale by precision (1/σ²)
        # ===============================================================
        precisions = 1.0 / (self.covariances_ + 1e-20)  # (n_components,)
        mahal = sq_dist * precisions.unsqueeze(0)
        
        # ===============================================================
        # Log determinant: d·log(σ²)
        # ===============================================================
        log_det = self.n_features * torch.log(self.covariances_ + 1e-20)  # (n_components,)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
        return -0.5 * (self.n_features * log_2pi + log_det.unsqueeze(0) + mahal)

    def _estimate_log_gaussian_tied_full(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for tied full covariance.

        All components share the same full covariance matrix Σ ∈ ℝ^(d×d).
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
        """
        # ===============================================================
        # Compute deviations from component means
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        
        # ===============================================================
        # Cholesky decomposition of shared covariance
        # ===============================================================
        try:
            chol = torch.linalg.cholesky(self.covariances_)  # (n_features, n_features)
        except RuntimeError as e:
            raise ValueError(f"Tied full: Cholesky decomposition failed: {e}")

        # ===============================================================
        # Log determinant (same for all components)
        # ===============================================================
        log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()  # scalar
        
        # ===============================================================
        # Mahalanobis distance for all samples and components
        # ===============================================================
        diff_ = diff.unsqueeze(-1)  # (n_samples, n_components, n_features, 1)
        solve = torch.cholesky_solve(diff_, chol)
        mahal = (diff_ * solve).sum(dim=(2, 3))  # (n_samples, n_components)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
        return -0.5 * (self.n_features * log_2pi + log_det + mahal)

    def _estimate_log_gaussian_tied_diag(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for tied diagonal covariance.

        All components share the same diagonal covariance: Σ = diag(σ²₁, ..., σ²d).
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
        """
        # ===============================================================
        # Compute deviations
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        
        # ===============================================================
        # Shared diagonal covariance
        # ===============================================================
        cov_vector = self.covariances_ + 1e-20  # (n_features,)
        log_det = torch.sum(torch.log(cov_vector))  # scalar
        precisions = 1.0 / cov_vector  # (n_features,)
        
        # ===============================================================
        # Mahalanobis distance
        # ===============================================================
        mahal = torch.sum(diff.pow(2) * precisions, dim=2)  # (n_samples, n_components)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
        return -0.5 * (self.n_features * log_2pi + log_det + mahal)

    def _estimate_log_gaussian_tied_spherical(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log p(x|z,θ) for tied spherical covariance.

        All components share the same spherical covariance: Σ = σ²·I.
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
            
        Returns
        -------
        log_prob : torch.Tensor
            Log-probabilities, shape (n_samples, n_components).
        """
        # ===============================================================
        # Compute deviations and squared distances
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        sq_dist = torch.sum(diff.pow(2), dim=2)  # (n_samples, n_components)
        
        # ===============================================================
        # Shared spherical variance
        # ===============================================================
        var = self.covariances_ + 1e-20  # scalar
        prec = 1.0 / var
        mahal = sq_dist * prec
        log_det = self.n_features * torch.log(var)

        # ===============================================================
        # Combine into log-probability
        # ===============================================================
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
        return -0.5 * (self.n_features * log_2pi + log_det + mahal)

    # ---------------------------
    # M-step
    # ---------------------------
    # ===================================================================
    # M-step: Maximization
    # ===================================================================
    def _m_step(self, X: torch.Tensor, resp: torch.Tensor):
        r"""
        M-step: Update model parameters given current responsibilities.

        Updates weights, means, and covariances using either:
        - MLE (Maximum Likelihood Estimation) if no priors
        - MAP (Maximum A Posteriori) if priors are specified
        - NIW (Normal-Inverse-Wishart) conjugate updates if both mean and covariance priors

        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).
        resp : torch.Tensor
            Current responsibilities for each sample w.r.t. each component,
            shape (n_samples, n_components). From E-step or C-step.
            
        Notes
        -----
        The update strategy depends on which priors are specified:
        
        1. NIW conjugate (use_mean_prior=True AND use_covariance_prior=True):
           Joint update of means and covariances using conjugate prior formulas
           
        2. Independent priors:
           - Means: MAP if use_mean_prior=True, else MLE
           - Covariances: MAP if use_covariance_prior=True, else MLE
           
        3. Weights: Always MAP if use_weight_prior=True, else MLE
        """
        n_samples = X.size(0)
        
        # ===============================================================
        # 1. Compute effective sample counts per component
        # ===============================================================
        nk = resp.sum(dim=0) + 1e-20  # (n_components,)

        # ===============================================================
        # 2. Update mixing weights
        # ===============================================================
        if self.use_weight_prior:
            # MAP update with Dirichlet prior
            alpha = self.weight_concentration_prior  # (n_components,)
            total_alpha = alpha.sum()
            self.weights_ = (nk + alpha - 1.0) / (n_samples + total_alpha - self.n_components)
        else:
            # MLE update
            self.weights_ = nk / n_samples
        
        self.weights_.clamp_(min=1e-20)  # Ensure numerical stability

        # ===============================================================
        # 3. Update means and covariances
        # ===============================================================
        # Check if we're using NIW conjugate priors (both mean and covariance priors)
        use_niw = self.use_mean_prior and self.use_covariance_prior
        
        if use_niw:
            # NIW conjugate updates: joint update for means and covariances
            self._update_niw_conjugate(X, resp, nk)
        else:
            # Independent updates
            # Update means
            if self.use_mean_prior:
                # MAP update with Gaussian prior
                kappa0 = self.mean_precision_prior
                numerator = resp.t() @ X + kappa0 * self.mean_prior
                denom = nk.unsqueeze(1) + kappa0
                self.means_ = numerator / denom
            else:
                # MLE update
                self.means_ = (resp.t() @ X) / nk.unsqueeze(1)

            # Update covariances
            if self.use_covariance_prior:
                # MAP update with Wishart/Inverse-Wishart prior
                self._update_covariances_map(X, resp, nk)
            else:
                # MLE update
                self._update_covariances_mle(X, resp, nk)

    # ===================================================================
    # NIW (Normal-Inverse-Wishart) Conjugate Prior Updates
    # ===================================================================
    def _update_niw_conjugate(self, X: torch.Tensor, resp: torch.Tensor, nk: torch.Tensor):
        r"""
        Update means and covariances jointly using Normal-Inverse-Wishart (NIW) conjugate priors.
        
        The NIW prior is the conjugate prior for a multivariate normal with unknown
        mean and covariance. The posterior updates are:
        
        .. math::
            μₙ = \frac{λ μ₀ + n ȳ}{λ + n}
            
            λₙ = λ + n
            
            νₙ = ν + n
            
            Ψₙ = Ψ + S + \frac{λ n}{λ + n} (ȳ - μ₀)(ȳ - μ₀)^T
        
        where ȳ is the empirical mean, S is the scatter matrix, and (μ₀, λ, Ψ, ν)
        are the NIW hyperparameters.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).
        resp : torch.Tensor
            Responsibilities, shape (n_samples, n_components).
        nk : torch.Tensor
            Effective sample count per component, shape (n_components,).
            
        Notes
        -----
        This method routes to the appropriate covariance-type-specific NIW update.
        The covariance is computed as E[Σ] = Ψₙ / (νₙ - n_features - 1) for Inverse-Wishart.
        """
        # ===============================================================
        # Extract NIW prior parameters
        # ===============================================================
        mu0 = self.mean_prior  # (n_components, n_features)
        lambda0 = self.mean_precision_prior  # scalar
        psi0 = self.covariance_prior  # shape depends on covariance_type
        nu0 = self.degrees_of_freedom_prior  # scalar
        
        # ===============================================================
        # Compute empirical means (ȳ) for each component
        # ===============================================================
        empirical_means = (resp.t() @ X) / nk.unsqueeze(1)  # (n_components, n_features)
        
        # ===============================================================
        # Compute NIW posterior parameters
        # ===============================================================
        lambda_n = lambda0 + nk  # (n_components,)
        nu_n = nu0 + nk  # (n_components,)
        
        # Update means using NIW posterior mean
        mu_n = (lambda0 * mu0 + nk.unsqueeze(1) * empirical_means) / lambda_n.unsqueeze(1)
        self.means_ = mu_n
        
        # ===============================================================
        # Update covariances (route to covariance-type-specific method)
        # ===============================================================
        if self.covariance_type == 'full':
            self._update_niw_full(X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0)
        elif self.covariance_type == 'diag':
            self._update_niw_diag(X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0)
        elif self.covariance_type == 'spherical':
            self._update_niw_spherical(X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0)
        elif self.covariance_type == 'tied_full':
            self._update_niw_tied_full(X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0)
        elif self.covariance_type == 'tied_diag':
            self._update_niw_tied_diag(X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0)
        elif self.covariance_type == 'tied_spherical':
            self._update_niw_tied_spherical(X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0)
        else:
            raise ValueError(f"Unsupported covariance_type: {self.covariance_type}")

    # ===================================================================
    # MAP (Maximum A Posteriori) Covariance Updates
    # ===================================================================
    def _update_covariances_map(self, X, resp, nk):
        r"""
        Update covariances using MAP estimation with prior.

        Routes to appropriate covariance-type-specific MAP update method.
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
        resp : torch.Tensor
            Responsibilities, shape (n_samples, n_components).
        nk : torch.Tensor
            Effective sample counts, shape (n_components,).
        """
        if self.covariance_type == 'full':
            self._update_map_full(X, resp, nk)
        elif self.covariance_type == 'diag':
            self._update_map_diag(X, resp, nk)
        elif self.covariance_type == 'spherical':
            self._update_map_spherical(X, resp, nk)
        elif self.covariance_type == 'tied_full':
            self._update_map_tied_full(X, resp, nk)
        elif self.covariance_type == 'tied_diag':
            self._update_map_tied_diag(X, resp, nk)
        elif self.covariance_type == 'tied_spherical':
            self._update_map_tied_spherical(X, resp, nk)
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def _update_map_full(self, X, resp, nk):
        r"""
        MAP update for full covariance matrices.
        
        Uses Inverse-Wishart prior: Σₖ ~ IW(Ψ, ν).
        Posterior: Σₖ = (Ψ + S_k + prior_mean_term) / (ν + n_k + d)
        
        where S_k is the weighted scatter matrix and prior_mean_term accounts
        for the difference between empirical mean and prior mean when using
        mean priors.
        """
        # ===============================================================
        # Compute weighted scatter matrix S_k for each component
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        weighted_diff = resp.unsqueeze(-1).unsqueeze(-1) * diff.unsqueeze(3) * diff.unsqueeze(2)
        sum_diff = weighted_diff.sum(dim=0)  # (n_components, n_features, n_features)

        # ===============================================================
        # Add mean prior term if using Gaussian prior on means
        # ===============================================================
        if self.use_mean_prior:
            mean_diff = (self.means_ - self.mean_prior).unsqueeze(-1)  # (n_components, n_features, 1)
            prior_term = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1).unsqueeze(-1) \
                         * mean_diff @ mean_diff.transpose(-1, -2)
        else:
            prior_term = torch.zeros_like(sum_diff)

        # ===============================================================
        # Compute posterior covariance
        # ===============================================================
        df = self.degrees_of_freedom_prior + nk.unsqueeze(-1).unsqueeze(-1) + self.n_features

        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar * torch.eye(self.n_features, device=self.device).unsqueeze(0)
        ) / df

    def _update_map_diag(self, X, resp, nk):
        r"""
        MAP update for diagonal covariance matrices.
        
        Each dimension updated independently using Inverse-Gamma priors.
        """
        # ===============================================================
        # Compute weighted scatter (per dimension)
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        sum_diff = (resp.unsqueeze(-1) * diff.pow(2)).sum(dim=0)  # (n_components, n_features)

        # ===============================================================
        # Add mean prior term if using Gaussian prior on means
        # ===============================================================
        if self.use_mean_prior:
            mean_diff2 = (self.means_ - self.mean_prior).pow(2)
            prior_term = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1) * mean_diff2
        else:
            prior_term = torch.zeros_like(sum_diff)
            
        # ===============================================================
        # Compute posterior covariance
        # ===============================================================
        df = self.degrees_of_freedom_prior + nk.unsqueeze(-1) + self.n_features

        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar
        ) / df

    def _update_map_spherical(self, X, resp, nk):
        r"""
        MAP update for spherical covariance (single variance per component).
        
        Uses Inverse-Gamma prior on the shared variance.
        """
        # ===============================================================
        # Compute total weighted scatter
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        diff2 = diff.pow(2).sum(dim=2)  # (n_samples, n_components)
        sum_diff = (resp * diff2).sum(dim=0)  # (n_components,)

        # ===============================================================
        # Add mean prior term if using Gaussian prior on means
        # ===============================================================
        if self.use_mean_prior:
            mean_diff2 = (self.means_ - self.mean_prior).pow(2).sum(dim=1)
            prior_term = (nk / (nk + self.mean_precision_prior)) * mean_diff2
        else:
            prior_term = torch.zeros_like(sum_diff)
            
        # ===============================================================
        # Compute posterior covariance (averaged over dimensions)
        # ===============================================================
        df = self.degrees_of_freedom_prior + nk + self.n_features

        self.covariances_ = (
            self.covariance_prior + sum_diff + prior_term + self.reg_covar
        ) / (df * self.n_features)

    def _update_map_tied_full(self, X, resp, nk):
        r"""
        MAP update for tied full covariance (shared across all components).
        
        Pooled scatter matrix across all components with shared Inverse-Wishart prior.
        """
        # ===============================================================
        # Compute pooled scatter matrix across all components
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = torch.einsum('nk,nkd,nke->de', resp, diff, diff)  # (n_features, n_features)

        # ===============================================================
        # Add mean prior term (pooled across components)
        # ===============================================================
        if self.use_mean_prior:
            mean_diff = (self.means_ - self.mean_prior).unsqueeze(-1)
            prior_term = (
                (nk / (nk + self.mean_precision_prior)).unsqueeze(-1).unsqueeze(-1)
                * torch.matmul(mean_diff, mean_diff.transpose(-1, -2))
            )
            prior_term = prior_term.sum(dim=0)  # Sum across components
        else:
            prior_term = torch.zeros_like(sum_diff)
            
        # ===============================================================
        # Compute posterior covariance
        # ===============================================================
        df = self.degrees_of_freedom_prior + nk.sum() + self.n_features

        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar * torch.eye(self.n_features, device=self.device)
        ) / df

    def _update_map_tied_diag(self, X, resp, nk):
        r"""
        MAP update for tied diagonal covariance (shared across components).
        
        Per-dimension variance shared across all components.
        """
        # ===============================================================
        # Compute pooled scatter (per dimension)
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = torch.einsum('nk,nkd->d', resp, diff.pow(2))  # (n_features,)

        # ===============================================================
        # Add mean prior term (pooled across components)
        # ===============================================================
        if self.use_mean_prior:
            mean_diff2 = (self.means_ - self.mean_prior).pow(2)  # (n_components, n_features)
            prior_term = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1) * mean_diff2
            prior_term = prior_term.sum(dim=0)  # (n_features,)
        else:
            prior_term = torch.zeros_like(sum_diff)

        # ===============================================================
        # Compute posterior covariance
        # ===============================================================
        df = self.degrees_of_freedom_prior + nk.sum() + self.n_features

        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar
        ) / df

    def _update_map_tied_spherical(self, X, resp, nk):
        r"""
        MAP update for tied spherical covariance (single variance for all).
        
        Single shared variance across all components and dimensions.
        """
        # ===============================================================
        # Compute total pooled scatter
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        diff2 = diff.pow(2).sum(dim=2)  # (n_samples, n_components)
        sum_diff = torch.einsum('nk,nk->', resp, diff2)  # scalar

        # ===============================================================
        # Add mean prior term (pooled across components)
        # ===============================================================
        if self.use_mean_prior:
            mean_diff2 = (self.means_ - self.mean_prior).pow(2).sum(dim=1)  # (n_components,)
            prior_term = (nk / (nk + self.mean_precision_prior)) * mean_diff2
            prior_term_total = prior_term.sum()
        else:
            prior_term_total = 0.0

        # ===============================================================
        # Compute posterior covariance
        # ===============================================================
        df = self.degrees_of_freedom_prior + nk.sum() + self.n_features
        
        self.covariances_ = (
            self.covariance_prior + sum_diff + prior_term_total + self.reg_covar
        ) / (df * self.n_features)

    # ===================================================================
    # MLE (Maximum Likelihood Estimation) Covariance Updates
    # ===================================================================
    def _update_covariances_mle(self, X, resp, nk):
        r"""
        Update covariances using MLE (no prior).

        Routes to appropriate covariance-type-specific MLE update method.
        
        Parameters
        ----------
        X : torch.Tensor
            Data, shape (n_samples, n_features).
        resp : torch.Tensor
            Responsibilities, shape (n_samples, n_components).
        nk : torch.Tensor
            Effective sample counts, shape (n_components,).
        """
        if self.covariance_type == 'full':
            self._update_mle_full(X, resp, nk)
        elif self.covariance_type == 'diag':
            self._update_mle_diag(X, resp, nk)
        elif self.covariance_type == 'spherical':
            self._update_mle_spherical(X, resp, nk)
        elif self.covariance_type == 'tied_full':
            self._update_mle_tied_full(X, resp, nk)
        elif self.covariance_type == 'tied_diag':
            self._update_mle_tied_diag(X, resp, nk)
        elif self.covariance_type == 'tied_spherical':
            self._update_mle_tied_spherical(X, resp, nk)
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def _update_mle_full(self, X, resp, nk):
        r"""
        MLE update for full covariance matrices.
        
        Σₖ = (1/n_k) Σᵢ γᵢₖ(xᵢ - μₖ)(xᵢ - μₖ)ᵀ + reg·I
        """
        # ===============================================================
        # Compute weighted scatter matrix for each component
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        weighted_diff = resp.unsqueeze(-1).unsqueeze(-1) * diff.unsqueeze(3) * diff.unsqueeze(2)
        sum_diff = weighted_diff.sum(dim=0)  # (n_components, n_features, n_features)
        
        # ===============================================================
        # Normalize and add regularization
        # ===============================================================
        covs = sum_diff / nk.unsqueeze(-1).unsqueeze(-1)
        covs += self.reg_covar * torch.eye(self.n_features, device=self.device).unsqueeze(0)
        self.covariances_ = covs

    def _update_mle_diag(self, X, resp, nk):
        r"""
        MLE update for diagonal covariance matrices.
        
        σ²ₖd = (1/n_k) Σᵢ γᵢₖ(xᵢd - μₖd)² + reg
        """
        # ===============================================================
        # Compute weighted scatter per dimension
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = (resp.unsqueeze(-1) * diff.pow(2)).sum(dim=0)  # (n_components, n_features)
        
        # ===============================================================
        # Normalize and add regularization
        # ===============================================================
        cov_diag = sum_diff / nk.unsqueeze(-1)
        cov_diag += self.reg_covar
        self.covariances_ = cov_diag

    def _update_mle_spherical(self, X, resp, nk):
        r"""
        MLE update for spherical covariance (single variance per component).
        
        σ²ₖ = (1/(n_k·d)) Σᵢ γᵢₖ||xᵢ - μₖ||² + reg
        """
        # ===============================================================
        # Compute total weighted scatter
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        diff2 = diff.pow(2).sum(dim=2)  # (n_samples, n_components)
        sum_diff2 = (resp * diff2).sum(dim=0)  # (n_components,)
        
        # ===============================================================
        # Normalize by n_k * d and add regularization
        # ===============================================================
        cov_spherical = sum_diff2 / (nk * self.n_features)
        cov_spherical += self.reg_covar
        self.covariances_ = cov_spherical

    def _update_mle_tied_full(self, X, resp, nk):
        r"""
        MLE update for tied full covariance (shared across components).
        
        Σ = (1/N) Σₖ Σᵢ γᵢₖ(xᵢ - μₖ)(xᵢ - μₖ)ᵀ + reg·I
        """
        # ===============================================================
        # Compute pooled scatter matrix
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = torch.einsum('nk,nkd,nke->de', resp, diff, diff)  # (n_features, n_features)
        
        # ===============================================================
        # Normalize and add regularization
        # ===============================================================
        cov_tied = sum_diff / nk.sum()
        cov_tied += self.reg_covar * torch.eye(self.n_features, device=self.device)
        self.covariances_ = cov_tied

    def _update_mle_tied_diag(self, X, resp, nk):
        r"""
        MLE update for tied diagonal covariance (shared across components).
        
        σ²d = (1/N) Σₖ Σᵢ γᵢₖ(xᵢd - μₖd)² + reg
        """
        # ===============================================================
        # Compute pooled scatter per dimension
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = torch.einsum('nk,nkd->d', resp, diff.pow(2))  # (n_features,)
        
        # ===============================================================
        # Normalize and add regularization
        # ===============================================================
        cov_tied_diag = sum_diff / nk.sum()
        cov_tied_diag += self.reg_covar
        self.covariances_ = cov_tied_diag

    def _update_mle_tied_spherical(self, X, resp, nk):
        r"""
        MLE update for tied spherical covariance (single variance for all).
        
        σ² = (1/(N·d)) Σₖ Σᵢ γᵢₖ||xᵢ - μₖ||² + reg
        """
        # ===============================================================
        # Compute total pooled scatter
        # ===============================================================
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = torch.einsum('nk,nkd->', resp, diff.pow(2))  # scalar
        
        # ===============================================================
        # Normalize by N * d and add regularization
        # ===============================================================
        cov_tied_spherical = sum_diff / (nk.sum() * self.n_features)
        cov_tied_spherical += self.reg_covar
        self.covariances_ = cov_tied_spherical


    # ===================================================================
    # NIW (Normal-Inverse-Wishart) Specific Covariance Updates
    # ===================================================================
    # These methods implement the covariance component of the NIW posterior.
    # The mean component is handled in _update_niw_conjugate().
    # ===================================================================
    
    def _update_niw_full(self, X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0):
        r"""
        NIW posterior update for full covariance matrices.
        
        Computes: Ψₙ = Ψ₀ + S + (λ₀n/(λ₀+n))(ȳ - μ₀)(ȳ - μ₀)ᵀ
        Then: Σₖ = Ψₙ / νₙ
        
        Parameters
        ----------
        All parameters come from _update_niw_conjugate().
        """
        # ===============================================================
        # Compute scatter matrix S for each component
        # ===============================================================
        diff = X.unsqueeze(1) - empirical_means.unsqueeze(0)  # (n_samples, n_components, n_features)
        weighted_diff = resp.unsqueeze(-1).unsqueeze(-1) * diff.unsqueeze(3) * diff.unsqueeze(2)
        S = weighted_diff.sum(dim=0)  # (n_components, n_features, n_features)
        
        # ===============================================================
        # Compute cross-term: (λ₀n/(λ₀+n)) (ȳ - μ₀)(ȳ - μ₀)ᵀ
        # ===============================================================
        mean_diff = empirical_means - self.mean_prior  # (n_components, n_features)
        cross_term_coeff = (lambda0 * nk) / lambda_n  # (n_components,)
        cross_term = cross_term_coeff.unsqueeze(-1).unsqueeze(-1) * (
            mean_diff.unsqueeze(-1) @ mean_diff.unsqueeze(-2)
        )  # (n_components, n_features, n_features)
        
        # ===============================================================
        # NIW posterior scale matrix and regularization
        # ===============================================================
        psi_n = psi0 + S + cross_term
        psi_n += self.reg_covar * torch.eye(self.n_features, device=self.device).unsqueeze(0)
        
        # ===============================================================
        # Final covariance: E[Σ] = Ψₙ / νₙ
        # ===============================================================
        self.covariances_ = psi_n / nu_n.unsqueeze(-1).unsqueeze(-1)

    def _update_niw_diag(self, X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0):
        r"""
        NIW posterior update for diagonal covariance matrices.
        
        Each dimension updated independently using NIW formula.
        """
        # ===============================================================
        # Compute scatter matrix S (diagonal)
        # ===============================================================
        diff = X.unsqueeze(1) - empirical_means.unsqueeze(0)  # (n_samples, n_components, n_features)
        weighted_diff_sq = resp.unsqueeze(-1) * diff.pow(2)
        S = weighted_diff_sq.sum(dim=0)  # (n_components, n_features)
        
        # ===============================================================
        # Compute cross-term (per dimension)
        # ===============================================================
        mean_diff_sq = (empirical_means - self.mean_prior).pow(2)  # (n_components, n_features)
        cross_term_coeff = (lambda0 * nk) / lambda_n  # (n_components,)
        cross_term = cross_term_coeff.unsqueeze(-1) * mean_diff_sq  # (n_components, n_features)
        
        # ===============================================================
        # NIW posterior and regularization
        # ===============================================================
        psi_n = psi0 + S + cross_term
        psi_n += self.reg_covar
        
        # ===============================================================
        # Final covariance
        # ===============================================================
        self.covariances_ = psi_n / nu_n.unsqueeze(-1)

    def _update_niw_spherical(self, X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0):
        r"""
        NIW posterior update for spherical covariances.
        
        Single variance per component (isotropic).
        """
        # ===============================================================
        # Compute total scatter (sum over features)
        # ===============================================================
        diff = X.unsqueeze(1) - empirical_means.unsqueeze(0)  # (n_samples, n_components, n_features)
        weighted_diff_sq = resp.unsqueeze(-1) * diff.pow(2)
        S = weighted_diff_sq.sum(dim=(0, 2))  # (n_components,)
        
        # ===============================================================
        # Compute cross-term (total squared distance)
        # ===============================================================
        mean_diff_norm_sq = (empirical_means - self.mean_prior).pow(2).sum(dim=1)  # (n_components,)
        cross_term_coeff = (lambda0 * nk) / lambda_n  # (n_components,)
        cross_term = cross_term_coeff * mean_diff_norm_sq  # (n_components,)
        
        # ===============================================================
        # NIW posterior and regularization
        # ===============================================================
        psi_n = psi0 + S + cross_term
        psi_n += self.reg_covar * self.n_features
        
        # ===============================================================
        # Final covariance (averaged over dimensions)
        # ===============================================================
        self.covariances_ = psi_n / (nu_n * self.n_features)

    def _update_niw_tied_full(self, X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0):
        r"""
        NIW posterior update for tied full covariance.
        
        Single shared full covariance matrix across all components.
        """
        # ===============================================================
        # Compute pooled scatter matrix
        # ===============================================================
        diff = X.unsqueeze(1) - empirical_means.unsqueeze(0)  # (n_samples, n_components, n_features)
        S = torch.einsum('nk,nkd,nke->de', resp, diff, diff)  # (n_features, n_features)
        
        # ===============================================================
        # Compute pooled cross-term
        # ===============================================================
        mean_diff = empirical_means - self.mean_prior  # (n_components, n_features)
        cross_term_coeff = (lambda0 * nk) / lambda_n  # (n_components,)
        cross_term = torch.einsum('k,kd,ke->de', cross_term_coeff, mean_diff, mean_diff)  # (n_features, n_features)
        
        # ===============================================================
        # NIW posterior and regularization
        # ===============================================================
        psi_n = psi0 + S + cross_term
        psi_n += self.reg_covar * torch.eye(self.n_features, device=self.device)
        
        # ===============================================================
        # Final covariance (total degrees of freedom)
        # ===============================================================
        total_nu_n = self.degrees_of_freedom_prior + nk.sum()
        self.covariances_ = psi_n / total_nu_n

    def _update_niw_tied_diag(self, X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0):
        r"""
        NIW posterior update for tied diagonal covariance.
        
        Shared diagonal covariance across all components.
        """
        # ===============================================================
        # Compute pooled scatter (per dimension)
        # ===============================================================
        diff = X.unsqueeze(1) - empirical_means.unsqueeze(0)  # (n_samples, n_components, n_features)
        weighted_diff_sq = resp.unsqueeze(-1) * diff.pow(2)
        S = weighted_diff_sq.sum(dim=(0, 1))  # (n_features,)
        
        # ===============================================================
        # Compute pooled cross-term
        # ===============================================================
        mean_diff_sq = (empirical_means - self.mean_prior).pow(2)  # (n_components, n_features)
        cross_term_coeff = (lambda0 * nk) / lambda_n  # (n_components,)
        cross_term = torch.einsum('k,kd->d', cross_term_coeff, mean_diff_sq)  # (n_features,)
        
        # ===============================================================
        # NIW posterior and regularization
        # ===============================================================
        psi_n = psi0 + S + cross_term
        psi_n += self.reg_covar
        
        # ===============================================================
        # Final covariance
        # ===============================================================
        total_nu_n = self.degrees_of_freedom_prior + nk.sum()
        self.covariances_ = psi_n / total_nu_n

    def _update_niw_tied_spherical(self, X, resp, nk, empirical_means, lambda0, lambda_n, nu_n, psi0):
        r"""
        NIW posterior update for tied spherical covariance.
        
        Single shared variance across all components and dimensions.
        """
        # ===============================================================
        # Compute total pooled scatter
        # ===============================================================
        diff = X.unsqueeze(1) - empirical_means.unsqueeze(0)  # (n_samples, n_components, n_features)
        weighted_diff_sq = resp.unsqueeze(-1) * diff.pow(2)
        S = weighted_diff_sq.sum()  # scalar
        
        # ===============================================================
        # Compute pooled cross-term
        # ===============================================================
        mean_diff_norm_sq = (empirical_means - self.mean_prior).pow(2).sum(dim=1)  # (n_components,)
        cross_term_coeff = (lambda0 * nk) / lambda_n  # (n_components,)
        cross_term = (cross_term_coeff * mean_diff_norm_sq).sum()  # scalar
        
        # ===============================================================
        # NIW posterior and regularization
        # ===============================================================
        psi_n = psi0 + S + cross_term
        psi_n += self.reg_covar * nk.sum() * self.n_features
        
        # ===============================================================
        # Final covariance (total samples and features)
        # ===============================================================
        total_nu_n = self.degrees_of_freedom_prior + nk.sum()
        self.covariances_ = psi_n / (total_nu_n * self.n_features)

    # ===================================================================
    # Prediction and Scoring Methods
    # ===================================================================
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Predict cluster labels for samples (hard assignment).

        Assigns each sample to the component with maximum posterior probability.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).

        Returns
        -------
        labels : torch.Tensor
            Cluster labels for each sample, shape (n_samples,).
            Each label is an integer in [0, n_components-1].
            
        Warnings
        --------
        Issues warning if model hasn't been fitted or didn't converge.
        
        See Also
        --------
        predict_proba : Get soft assignments (posterior probabilities).
        """
        if not self.fitted_:
            warnings.warn("GMM has not been fitted. Results may be unreliable.", UserWarning)
        elif not self.converged_:
            warnings.warn("GMM did not converge. Results may be unreliable.", UserWarning)
            
        resp, _ = self._e_step(X.to(self.device))
        return torch.argmax(resp, dim=1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Get posterior probabilities for each component (soft assignment).

        Returns the responsibility of each component for each sample:
        γ(z_{ik}) = p(z_k | x_i, θ)
        
        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).

        Returns
        -------
        resp : torch.Tensor
            Posterior probabilities (responsibilities), shape (n_samples, n_components).
            Each row sums to 1.0.
            
        Warnings
        --------
        Issues warning if model hasn't been fitted or didn't converge.
        
        See Also
        --------
        predict : Get hard assignments (argmax of probabilities).
        """
        if not self.fitted_:
            warnings.warn("GMM has not been fitted. Results may be unreliable.", UserWarning)
        elif not self.converged_:
            warnings.warn("GMM did not converge. Results may be unreliable.", UserWarning)
            
        resp, _ = self._e_step(X.to(self.device))
        return resp

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log-likelihood for each sample.

        Returns log p(x_i | θ) for each sample under the fitted GMM.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).

        Returns
        -------
        log_prob : torch.Tensor
            Log-likelihood for each sample, shape (n_samples,).
            Higher values indicate better fit to the model.
            
        Warnings
        --------
        Issues warning if model hasn't been fitted or didn't converge.
        
        See Also
        --------
        score : Get average log-likelihood across all samples.
        """
        if not self.fitted_:
            warnings.warn("GMM has not been fitted. Results may be unreliable.", UserWarning)
        elif not self.converged_:
            warnings.warn("GMM did not converge. Results may be unreliable.", UserWarning)
            
        _, log_prob_norm = self._e_step(X.to(self.device))
        return log_prob_norm

    def score(self, X: torch.Tensor) -> float:
        r"""
        Compute average log-likelihood of data.

        Useful for model selection and comparison.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).

        Returns
        -------
        score : float
            Mean log-likelihood: (1/n) Σᵢ log p(xᵢ | θ).
            Higher values indicate better fit.
            
        See Also
        --------
        score_samples : Get per-sample log-likelihoods.
        """
        return self.score_samples(X).mean().item()

    # ===================================================================
    # Sampling Method
    # ===================================================================
    
    def sample(self, n_samples: int = 1, component: int = None, std_radius: float = None, 
               std_range: Tuple[float, float] = None, confidence: float = None,
               confidence_range: Tuple[float, float] = None,
               center_point: torch.Tensor = None, center_radius: float = None,
               max_attempts_per_sample: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Generate new samples from the fitted GMM.

        Supports both standard sampling and constrained sampling with rejection.
        Can sample from all components (according to weights) or a specific component,
        with optional constraints on distance from mean or a center point.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
            
        component : int, optional
            If specified, samples only from this component. If None, samples 
            from all components according to their weights.
            
        std_radius : float, optional
            Only return samples within this many standard deviations from the
            component mean (Mahalanobis distance ≤ std_radius). Uses rejection
            sampling. Cannot be combined with other distance constraints.
            
        std_range : tuple of (float, float), optional
            Only return samples whose Mahalanobis distance falls within
            (min_std, max_std). Examples:
            - (1.0, 2.0): between 1 and 2 standard deviations
            - (3.0, inf): further than 3 standard deviations (outliers)
            Cannot be combined with other distance constraints.
            
        confidence : float, optional
            Only return samples within the confidence ellipse (e.g., 0.95 for 95%).
            Converted to std_radius using χ² distribution. For 2D data, 0.95 ≈ 2.45σ.
            Cannot be combined with other distance constraints.
            
        confidence_range : tuple of (float, float), optional
            Only return samples between two confidence ellipses. Example:
            (0.68, 0.95) returns samples between 68% and 95% confidence regions.
            Cannot be combined with other distance constraints.
            
        center_point : torch.Tensor, optional
            Only return samples within center_radius Euclidean distance from this
            point. Shape (n_features,). Must be used with center_radius.
            Cannot be combined with other distance constraints.
            
        center_radius : float, optional
            Euclidean distance threshold from center_point.
            Must be used with center_point.
            
        max_attempts_per_sample : int, default=1000
            Maximum rejection sampling attempts per sample before raising error.
            Increase for very restrictive constraints.

        Returns
        -------
        samples : torch.Tensor
            Generated samples, shape (n_samples, n_features).
        indices : torch.Tensor
            Component index for each sample, shape (n_samples,).
            
        Raises
        ------
        ValueError
            If parameter combinations are invalid or constraints conflict.
        RuntimeError
            If rejection sampling fails to find valid samples.
            
        Warnings
        --------
        Issues warning if model hasn't been fitted or didn't converge.
        """
        # ===============================================================
        # 1. Check model state
        # ===============================================================
        if not self.fitted_:
            warnings.warn("GMM has not been fitted. Results may be unreliable.", UserWarning)
        elif not self.converged_:
            warnings.warn("GMM did not converge. Results may be unreliable.", UserWarning)

        # ===============================================================
        # 2. Validate constraint parameter combinations
        # ===============================================================
        constraint_params = [std_radius, std_range, confidence, confidence_range, 
                            (center_point, center_radius)]
        non_none_params = [p for p in constraint_params 
                          if p is not None and (not isinstance(p, tuple) or 
                                               all(x is not None for x in p))]
        
        if len(non_none_params) > 1:
            raise ValueError(
                "Cannot specify more than one of: std_radius, std_range, confidence, "
                "confidence_range, or center constraints. Use only one."
            )
        
        # ===============================================================
        # 3. Validate and convert std_range
        # ===============================================================
        if std_range is not None:
            if len(std_range) != 2:
                raise ValueError("std_range must be a tuple of exactly 2 values (min_std, max_std)")
            min_std, max_std = std_range
            if min_std < 0:
                raise ValueError("std_range minimum must be non-negative")
            if max_std <= min_std:
                raise ValueError("std_range maximum must be greater than minimum")
        
        # ===============================================================
        # 4. Convert confidence to std_radius
        # ===============================================================
        if confidence is not None:
            if not (0 < confidence < 1):
                raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
            std_radius = self._confidence_to_std_radius(confidence)
        
        # ===============================================================
        # 5. Convert confidence_range to std_range
        # ===============================================================
        if confidence_range is not None:
            if len(confidence_range) != 2:
                raise ValueError(
                    "confidence_range must be a tuple of exactly 2 values "
                    "(min_confidence, max_confidence)"
                )
            min_conf, max_conf = confidence_range
            if not (0 <= min_conf < 1) or not (0 < max_conf <= 1):
                raise ValueError(
                    f"confidence_range values must be in [0, 1), got ({min_conf}, {max_conf})"
                )
            if max_conf <= min_conf:
                raise ValueError("confidence_range maximum must be greater than minimum")
            
            # Convert to std_range (handle min_conf=0 special case)
            min_std = 0.0 if min_conf == 0.0 else self._confidence_to_std_radius(min_conf)
            max_std = self._confidence_to_std_radius(max_conf)
            std_range = (min_std, max_std)

        # ===============================================================
        # 6. Validate center point constraints
        # ===============================================================
        if (center_point is None) != (center_radius is None):
            raise ValueError("center_point and center_radius must be specified together")
        
        if center_point is not None:
            center_point = center_point.to(self.device)
            if center_point.dim() != 1 or center_point.size(0) != self.n_features:
                raise ValueError(
                    f"center_point must be a 1D tensor with {self.n_features} features, "
                    f"got shape {center_point.shape}"
                )
            if center_radius <= 0:
                raise ValueError(f"center_radius must be positive, got {center_radius}")

        # ===============================================================
        # 7. Select component indices
        # ===============================================================
        if component is not None:
            # Validate component index
            if not (0 <= component < self.n_components):
                raise ValueError(
                    f"component must be between 0 and {self.n_components - 1}, got {component}"
                )
            # Sample only from specified component
            indices = torch.full((n_samples,), component, dtype=torch.long, device=self.device)
        else:
            # Choose components according to mixture weights
            indices = torch.multinomial(self.weights_, n_samples, replacement=True)
        
        # ===============================================================
        # 8a. Standard sampling (no distance constraints)
        # ===============================================================
        if std_radius is None and std_range is None and center_point is None:
            means = self.means_[indices]
            covariances = self._build_covariances_for_sampling(indices, n_samples)
            samples = MultivariateNormal(means, covariance_matrix=covariances).sample()
            return samples, indices
        
        # ===============================================================
        # 8b. Rejection sampling (with distance constraints)
        # ===============================================================
        else:
            # Define distance check function and description
            if std_radius is not None:
                if std_radius <= 0:
                    raise ValueError("std_radius must be positive")
                distance_check = lambda sample, comp_idx: (
                    self._compute_mahalanobis_distance(
                        sample.unsqueeze(0), 
                        torch.tensor([comp_idx], device=self.device)
                    ).item() <= std_radius
                )
                constraint_desc = f"within {std_radius} standard deviations"
                
            elif std_range is not None:
                min_std, max_std = std_range
                distance_check = lambda sample, comp_idx: (
                    min_std <= self._compute_mahalanobis_distance(
                        sample.unsqueeze(0), 
                        torch.tensor([comp_idx], device=self.device)
                    ).item() <= max_std
                )
                if max_std == float('inf'):
                    constraint_desc = f"further than {min_std} standard deviations"
                else:
                    constraint_desc = f"between {min_std} and {max_std} standard deviations"
                    
            else:  # center_point is not None
                distance_check = lambda sample, comp_idx: (
                    torch.norm(sample - center_point).item() <= center_radius
                )
                constraint_desc = f"within {center_radius} units from center point"
            
            # Perform rejection sampling
            valid_samples = []
            valid_indices = []
            
            for i in range(n_samples):
                comp_idx = indices[i].item()
                attempts = 0
                
                while attempts < max_attempts_per_sample:
                    # Generate candidate sample
                    mean = self.means_[comp_idx:comp_idx+1]  # Keep batch dimension
                    cov = self._build_covariances_for_sampling(
                        torch.tensor([comp_idx], device=self.device), 1
                    )
                    sample = MultivariateNormal(mean, covariance_matrix=cov).sample()
                    
                    # Accept if constraint satisfied
                    if distance_check(sample.squeeze(0), comp_idx):
                        valid_samples.append(sample.squeeze(0))
                        valid_indices.append(comp_idx)
                        break
                    
                    attempts += 1
                
                # Raise error if max attempts exceeded
                if attempts >= max_attempts_per_sample:
                    raise RuntimeError(
                        f"Could not generate sample {i+1} {constraint_desc} "
                        f"after {max_attempts_per_sample} attempts. "
                        f"Constraint is too restrictive. Try increasing "
                        f"max_attempts_per_sample or relaxing the constraint."
                    )
            
            # Stack results
            samples = torch.stack(valid_samples)
            indices = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
            return samples, indices

    # ===================================================================
    # Helper Methods for Sampling
    # ===================================================================
    
    def _build_covariances_for_sampling(self, indices, n_samples):
        r"""
        Construct batch of full covariance matrices for sampling.

        Converts the stored covariance format (which may be tied, diagonal, or spherical)
        into full (n_features × n_features) covariance matrices suitable for sampling.

        Parameters
        ----------
        indices : torch.Tensor
            Component indices for each sample, shape (n_samples,).
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        covs : torch.Tensor
            Full covariance matrices, shape (n_samples, n_features, n_features).
        """
        # ===============================================================
        # Component-specific covariances
        # ===============================================================
        if self.covariance_type == 'full':
            # Already in full format
            return self.covariances_[indices]

        elif self.covariance_type == 'diag':
            # Convert diagonal to full matrix (zeros off-diagonal)
            return torch.diag_embed(self.covariances_[indices])

        elif self.covariance_type == 'spherical':
            # σ²·I for each component
            eye = torch.eye(self.n_features, device=self.device).unsqueeze(0)
            return eye * self.covariances_[indices].view(-1, 1, 1)

        # ===============================================================
        # Tied (shared) covariances
        # ===============================================================
        elif self.covariance_type == 'tied_full':
            # Same full matrix for all samples
            return self.covariances_.unsqueeze(0).expand(n_samples, -1, -1)

        elif self.covariance_type == 'tied_diag':
            # Same diagonal for all samples
            diag_mat = torch.diag_embed(self.covariances_)
            return diag_mat.unsqueeze(0).expand(n_samples, -1, -1)

        elif self.covariance_type == 'tied_spherical':
            # Same σ²·I for all samples
            eye = torch.eye(self.n_features, device=self.device).unsqueeze(0)
            return eye * self.covariances_

        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def _confidence_to_std_radius(self, confidence: float) -> float:
        r"""
        Convert confidence level to standard deviation radius.
        
        For multivariate Gaussian, the squared Mahalanobis distance follows χ²
        distribution with d degrees of freedom. This finds the radius containing
        the specified probability mass.
        
        For example:
        - confidence=0.95 in 2D → radius ≈ 2.45σ (95% of points within ellipse)
        - confidence=0.68 in 2D → radius ≈ 1.51σ (68% of points within ellipse)
        
        Parameters
        ----------
        confidence : float
            Confidence level in (0, 1), e.g., 0.95 for 95%.
            
        Returns
        -------
        std_radius : float
            Standard deviation radius: sqrt(χ²_d(confidence))
        """
        if not (0 < confidence < 1):
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

        # χ² quantile for given confidence and dimensionality
        chi2_quantile = chi2.ppf(confidence, df=self.n_features)
        return math.sqrt(chi2_quantile)

    def _compute_mahalanobis_distance(self, samples: torch.Tensor, component_indices: torch.Tensor) -> torch.Tensor:
        r"""
        Compute Mahalanobis distance from samples to component means.
        
        Mahalanobis distance accounts for covariance structure:
        d²(x, μ) = (x - μ)ᵀ Σ⁻¹ (x - μ)
        
        Parameters
        ----------
        samples : torch.Tensor
            Sample points, shape (n_samples, n_features).
        component_indices : torch.Tensor
            Component for each sample, shape (n_samples,).
            
        Returns
        -------
        distances : torch.Tensor
            Mahalanobis distances, shape (n_samples,).
            Distance of 1.0 = one standard deviation from mean.
        """
        # ===============================================================
        # Compute deviations from component means
        # ===============================================================
        means = self.means_[component_indices]  # (n_samples, n_features)
        diff = samples - means  # (n_samples, n_features)
        
        # ===============================================================
        # Compute d² = diff^T Σ⁻¹ diff (depends on covariance type)
        # ===============================================================
        if self.covariance_type == 'full':
            # Component-specific full covariances
            covs = self.covariances_[component_indices]  # (n_samples, n_features, n_features)
            diff_expanded = diff.unsqueeze(-1)  # (n_samples, n_features, 1)
            try:
                inv_covs = torch.inverse(covs)
            except RuntimeError:
                # Handle singular matrices
                inv_covs = torch.pinverse(covs)
            mahal_sq = torch.bmm(torch.bmm(diff.unsqueeze(1), inv_covs), diff_expanded).squeeze()
            
        elif self.covariance_type == 'diag':
            # Component-specific diagonal: d² = Σᵢ (xᵢ - μᵢ)² / σᵢ²
            vars = self.covariances_[component_indices]  # (n_samples, n_features)
            mahal_sq = torch.sum(diff.pow(2) / vars, dim=1)
            
        elif self.covariance_type == 'spherical':
            # Component-specific spherical: d² = ||x - μ||² / σ²
            vars = self.covariances_[component_indices]  # (n_samples,)
            mahal_sq = torch.sum(diff.pow(2), dim=1) / vars
            
        elif self.covariance_type == 'tied_full':
            # Shared full covariance
            try:
                inv_cov = torch.inverse(self.covariances_)
            except RuntimeError:
                inv_cov = torch.pinverse(self.covariances_)
            mahal_sq = torch.sum(diff * torch.matmul(diff, inv_cov), dim=1)
            
        elif self.covariance_type == 'tied_diag':
            # Shared diagonal covariance
            mahal_sq = torch.sum(diff.pow(2) / self.covariances_, dim=1)
            
        elif self.covariance_type == 'tied_spherical':
            # Shared spherical covariance
            mahal_sq = torch.sum(diff.pow(2), dim=1) / self.covariances_
            
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")
            
        return torch.sqrt(mahal_sq)

    # ===================================================================
    # Model Persistence (Save/Load)
    # ===================================================================
    
    def save(self, filepath: str):
        r"""
        Save GMM model to disk.

        Saves all model parameters, configuration, training state, and priors
        to a PyTorch file that can be reloaded later.

        Parameters
        ----------
        filepath : str
            Path where to save the model. Typically ends with '.pth' or '.pt'.
            
        See Also
        --------
        load : Class method to load a saved model.
        save_state_dict : Get state dictionary without saving to file.
        """
        state_dict = {
            # ===============================================================
            # Model parameters (fitted)
            # ===============================================================
            'weights_': self.weights_,
            'means_': self.means_,
            'covariances_': self.covariances_,
            
            # ===============================================================
            # Initial parameters (for reproducibility)
            # ===============================================================
            'initial_weights_': self.initial_weights_,
            'initial_means_': self.initial_means_,
            'initial_covariances_': self.initial_covariances_,
            
            # ===============================================================
            # Model configuration
            # ===============================================================
            'n_components': self.n_components,
            'n_features': self.n_features,
            'covariance_type': self.covariance_type,
            'tol': self.tol,
            'reg_covar': self.reg_covar,
            'max_iter': self.max_iter,
            'init_means': self.init_means,
            'init_weights': self.init_weights,
            'init_covariances': self.init_covariances,
            'n_init': self.n_init,
            'random_state': self.random_state,
            'warm_start': self.warm_start,
            'verbose': self.verbose,
            'verbose_interval': self.verbose_interval,
            'cem': self.cem,
            
            # ===============================================================
            # Training state
            # ===============================================================
            'fitted_': self.fitted_,
            'converged_': self.converged_,
            'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_,
            
            # ===============================================================
            # Prior settings
            # ===============================================================
            'use_weight_prior': self.use_weight_prior,
            'use_mean_prior': self.use_mean_prior,
            'use_covariance_prior': self.use_covariance_prior,
            'weight_concentration_prior': self.weight_concentration_prior,
            'mean_prior': self.mean_prior,
            'mean_precision_prior': self.mean_precision_prior,
            'covariance_prior': self.covariance_prior,
            'degrees_of_freedom_prior': self.degrees_of_freedom_prior,
        }
        
        torch.save(state_dict, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = None) -> "GaussianMixture":
        r"""
        Load GMM model from disk.

        Creates a new GaussianMixture instance with all parameters and state
        restored from a saved file.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.
        device : str, optional
            Device to load model tensors on ('cpu' or 'cuda'). 
            If None, uses device from saved model or defaults to GPU if available.

        Returns
        -------
        model : GaussianMixture
            The loaded GMM model, ready for prediction or continued training.
            
        Notes
        -----
        Handles backward compatibility with older saved models that used
        deprecated parameter names like 'init_params' and 'cov_init_method'.
        
        See Also
        --------
        save : Save a model to disk.
        load_state_dict : Load from a state dictionary.
        """
        # ===============================================================
        # Load state dictionary from file
        # ===============================================================
        if device is None:
            state_dict = torch.load(filepath, weights_only=False)
        else:
            state_dict = torch.load(filepath, map_location=device, weights_only=False)
        
        # ===============================================================
        # Handle backward compatibility
        # ===============================================================
        if 'init_params' in state_dict and 'init_means' not in state_dict:
            state_dict['init_means'] = state_dict['init_params']
        if 'cov_init_method' in state_dict and 'init_covariances' not in state_dict:
            state_dict['init_covariances'] = state_dict['cov_init_method']
        if 'init_weights' not in state_dict:
            state_dict['init_weights'] = 'uniform'
        
        # ===============================================================
        # Create new instance with saved configuration
        # ===============================================================
        model = cls(
            n_components=state_dict['n_components'],
            n_features=state_dict['n_features'],
            covariance_type=state_dict['covariance_type'],
            tol=state_dict['tol'],
            reg_covar=state_dict['reg_covar'],
            max_iter=state_dict['max_iter'],
            init_means=state_dict['init_means'],
            init_weights=state_dict['init_weights'],
            init_covariances=state_dict['init_covariances'],
            n_init=state_dict['n_init'],
            random_state=state_dict['random_state'],
            warm_start=state_dict['warm_start'],
            verbose=state_dict['verbose'],
            verbose_interval=state_dict['verbose_interval'],
            device=device,
            weight_concentration_prior=state_dict['weight_concentration_prior'],
            mean_prior=state_dict['mean_prior'],
            mean_precision_prior=state_dict['mean_precision_prior'],
            covariance_prior=state_dict['covariance_prior'],
            degrees_of_freedom_prior=state_dict['degrees_of_freedom_prior'],
            cem=state_dict['cem'],
        )
        
        # ===============================================================
        # Load trained parameters and state
        # ===============================================================
        model.weights_ = state_dict['weights_']
        model.means_ = state_dict['means_']
        model.covariances_ = state_dict['covariances_']
        model.initial_weights_ = state_dict['initial_weights_']
        model.initial_means_ = state_dict['initial_means_']
        model.initial_covariances_ = state_dict['initial_covariances_']
        model.fitted_ = state_dict['fitted_']
        model.converged_ = state_dict['converged_']
        model.n_iter_ = state_dict['n_iter_']
        model.lower_bound_ = state_dict['lower_bound_']
        
        # ===============================================================
        # Load prior flags
        # ===============================================================
        model.use_weight_prior = state_dict['use_weight_prior']
        model.use_mean_prior = state_dict['use_mean_prior']
        model.use_covariance_prior = state_dict['use_covariance_prior']
        
        return model

    def save_state_dict(self) -> dict:
        r"""
        Get model state as a dictionary (PyTorch-style).
        
        Useful for custom saving/loading workflows or integration with
        other PyTorch models.

        Returns
        -------
        state_dict : dict
            Dictionary containing all model parameters, configuration,
            and training state.
            
        See Also
        --------
        load_state_dict : Load from a state dictionary.
        to_dict : Get a simplified dictionary (basic parameters only).
        save : Save directly to a file.
        """
        return {
            # ===============================================================
            # Model parameters
            # ===============================================================
            'weights_': self.weights_,
            'means_': self.means_,
            'covariances_': self.covariances_,
            
            # ===============================================================
            # Initial parameters
            # ===============================================================
            'initial_weights_': self.initial_weights_,
            'initial_means_': self.initial_means_,
            'initial_covariances_': self.initial_covariances_,
            
            # ===============================================================
            # Model configuration
            # ===============================================================
            'n_components': self.n_components,
            'n_features': self.n_features,
            'covariance_type': self.covariance_type,
            'tol': self.tol,
            'reg_covar': self.reg_covar,
            'max_iter': self.max_iter,
            'init_means': self.init_means,
            'init_weights': self.init_weights,
            'init_covariances': self.init_covariances,
            'n_init': self.n_init,
            'random_state': self.random_state,
            'warm_start': self.warm_start,
            'verbose': self.verbose,
            'verbose_interval': self.verbose_interval,
            'cem': self.cem,
            
            # ===============================================================
            # Training state
            # ===============================================================
            'fitted_': self.fitted_,
            'converged_': self.converged_,
            'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_,
            
            # ===============================================================
            # Prior settings
            # ===============================================================
            'use_weight_prior': self.use_weight_prior,
            'use_mean_prior': self.use_mean_prior,
            'use_covariance_prior': self.use_covariance_prior,
            'weight_concentration_prior': self.weight_concentration_prior,
            'mean_prior': self.mean_prior,
            'mean_precision_prior': self.mean_precision_prior,
            'covariance_prior': self.covariance_prior,
            'degrees_of_freedom_prior': self.degrees_of_freedom_prior,
        }

    def load_state_dict(self, state_dict: dict):
        r"""
        Load model state from a dictionary (PyTorch-style).

        Updates the current model instance with parameters and configuration
        from the state dictionary.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing model parameters and metadata.
            Typically obtained from save_state_dict().
            
        Warnings
        --------
        Issues warning if loading a state dict with different n_components
        than the current model.
        
        Notes
        -----
        Handles backward compatibility with older state dictionaries.
        
        See Also
        --------
        save_state_dict : Get state dictionary from current model.
        load : Load model from file (creates new instance).
        """
        # ===============================================================
        # Validate compatibility
        # ===============================================================
        if hasattr(self, 'n_components') and self.n_components != state_dict['n_components']:
            warnings.warn(
                f"n_components mismatch: current={self.n_components}, "
                f"loaded={state_dict['n_components']}"
            )
        
        # ===============================================================
        # Handle backward compatibility
        # ===============================================================
        if 'init_params' in state_dict and 'init_means' not in state_dict:
            state_dict['init_means'] = state_dict['init_params']
        if 'cov_init_method' in state_dict and 'init_covariances' not in state_dict:
            state_dict['init_covariances'] = state_dict['cov_init_method']
        if 'init_weights' not in state_dict:
            state_dict['init_weights'] = 'uniform'
        
        # ===============================================================
        # Update configuration
        # ===============================================================
        self.n_components = state_dict['n_components']
        self.n_features = state_dict['n_features']
        self.covariance_type = state_dict['covariance_type']
        self.tol = state_dict['tol']
        self.reg_covar = state_dict['reg_covar']
        self.max_iter = state_dict['max_iter']
        self.init_means = state_dict['init_means']
        self.init_weights = state_dict['init_weights']
        self.init_covariances = state_dict['init_covariances']
        self.n_init = state_dict['n_init']
        self.random_state = state_dict['random_state']
        self.warm_start = state_dict['warm_start']
        self.verbose = state_dict['verbose']
        self.verbose_interval = state_dict['verbose_interval']
        self.cem = state_dict['cem']
        
        # ===============================================================
        # Load parameters
        # ===============================================================
        self.weights_ = state_dict['weights_']
        self.means_ = state_dict['means_']
        self.covariances_ = state_dict['covariances_']
        self.initial_weights_ = state_dict['initial_weights_']
        self.initial_means_ = state_dict['initial_means_']
        self.initial_covariances_ = state_dict['initial_covariances_']
        
        # ===============================================================
        # Load training state
        # ===============================================================
        self.fitted_ = state_dict['fitted_']
        self.converged_ = state_dict['converged_']
        self.n_iter_ = state_dict['n_iter_']
        self.lower_bound_ = state_dict['lower_bound_']
        
        # ===============================================================
        # Load prior settings
        # ===============================================================
        self.use_weight_prior = state_dict['use_weight_prior']
        self.use_mean_prior = state_dict['use_mean_prior']
        self.use_covariance_prior = state_dict['use_covariance_prior']
        self.weight_concentration_prior = state_dict['weight_concentration_prior']
        self.mean_prior = state_dict['mean_prior']
        self.mean_precision_prior = state_dict['mean_precision_prior']
        self.covariance_prior = state_dict['covariance_prior']
        self.degrees_of_freedom_prior = state_dict['degrees_of_freedom_prior']

    def to_dict(self) -> dict:
        r"""
        Alias for save_state_dict() for convenience.
        
        Returns the state dictionary of the model.

        Returns
        -------
        state_dict : dict
            Dictionary containing model parameters and metadata.
        """
        return self.save_state_dict()