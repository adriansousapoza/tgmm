import torch
from torch import nn
from torch.distributions import MultivariateNormal
from typing import Optional, Tuple
from .gmm_init import GMMInitializer

EPS = 1e-20  # Small constant for numerical stability

class GaussianMixture(nn.Module):
    r"""
    A Gaussian Mixture Model (GMM) based on Expectation-Maximisation (EM) implemented in PyTorch.

    This GMM supports:
    
    - The Expectation-Maximization (EM) algorithm.
    - Multiple initializations (n_init).
    - Configurable covariance types (full, diag, spherical, tied_full, tied_diag, tied_spherical).
    - Maximum Likelihood Estimation (MLE) and Maximum a Posteriori (MAP) estimation (i.e. with Priors).

    Parameters
    ----------
    n_features : int
        Dimensionality of the input data (number of features).
    n_components : int, optional
        Number of mixture components (default: 1).
    covariance_type : str, optional
        Type of covariance parameters to use. Must be one of:
        'full', 'diag', 'spherical', 'tied_full', 'tied_diag', 'tied_spherical'.
        (default: 'full')
    tol : float, optional
        Convergence threshold for EM (relative improvement in log-likelihood).
        (default: 1e-4)
    reg_covar : float, optional
        Non-negative regularization added to the diagonal of covariance.
        Helps keep covariance matrices from becoming singular. (default: 1e-6)
    max_iter : int, optional
        Maximum number of EM iterations to perform. (default: 1000)
    init_params : str, optional
        Method for initializing means (kmeans, random, points, kpp, maxdist).
        (default: 'kmeans')
    weights_init : torch.Tensor or None, optional
        User-provided initial component weights of shape (n_components,).
        If None, weights are set uniformly. (default: None)
    means_init : torch.Tensor or None, optional
        User-provided initial component means of shape (n_components, n_features).
        If None, means are randomized according to `init_params`. (default: None)
    covariances_init : torch.Tensor or None, optional
        User-provided initial covariances. Shape depends on `covariance_type`. (default: None)
    n_init : int, optional
        Number of random initializations to try. The best run (highest log-likelihood)
        is kept. (default: 1)
    random_state : int or None, optional
        Random seed for reproducibility. If None, uses PyTorch's internal seed. (default: None)
    warm_start : bool, optional
        If True, reuse the solution of the previous call to `fit` as initialization.
        (default: False)
    verbose : bool, optional
        If True, print progress during EM iterations. (default: False)
    verbose_interval : int, optional
        Frequency (in iterations) at which to print progress. (default: 10)
    device : str or None, optional
        Device on which to run computations ('cpu' or 'cuda'). If None, uses GPU if
        available, otherwise CPU. (default: None)
    weight_concentration_prior : torch.Tensor or None, optional
        Dirichlet concentration prior for the mixture weights, used in MAP. (default: None)
    mean_prior : torch.Tensor or None, optional
        Prior for the means, used in MAP. If provided, must match `mean_precision_prior`.
        (default: None)
    mean_precision_prior : float or None, optional
        Precision of the mean prior, used in MAP. (default: None)
    covariance_prior : torch.Tensor or None, optional
        Prior on covariance(s), used in MAP. If provided, must match
        `degrees_of_freedom_prior`. (default: None)
    degrees_of_freedom_prior : float or None, optional
        Degrees of freedom in the Wishart prior for covariances (MAP). (default: None)
    cov_init_method : str, optional
        Method for initializing covariances if `covariances_init` is None.
        Supported: 'eye', 'random', 'empirical'. (default: 'eye')

    Attributes
    ----------
    weights_ : torch.Tensor
        Mixture component weights of shape (n_components,).
    means_ : torch.Tensor
        Mixture component means of shape (n_components, n_features).
    covariances_ : torch.Tensor
        Mixture component covariances. Shape depends on `covariance_type`.
    fitted_ : bool
        Whether the model has been fitted. This attribute is important when using `warm_start`.
    converged_ : bool
        Whether the EM algorithm has converged in the best run.
    n_iter_ : int
        Number of EM iterations performed in the best run.
    lower_bound_ : float
        Log-likelihood lower bound on the fitted data for the best run.
    """

    def __init__(
        self,
        n_components: int = 1,
        n_features: int = None,
        covariance_type: str = 'full',
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        max_iter: int = 1000,
        init_params: str = 'kmeans',
        cov_init_method: str = 'eye',
        weights_init: torch.Tensor = None,
        means_init: torch.Tensor = None,
        covariances_init: torch.Tensor = None,
        n_init: int = 1,
        random_state: int = None,
        warm_start: bool = False,
        verbose: bool = False,
        verbose_interval: int = 10,
        device: str = None,
        weight_concentration_prior: torch.Tensor = None,
        mean_prior: torch.Tensor = None,
        mean_precision_prior: float = None,
        covariance_prior: torch.Tensor = None,
        degrees_of_freedom_prior: float = None,
    ):
        super().__init__()

        # Basic settings
        self.n_features = n_features
        self.n_components = n_components

        # Allow "tied" as alias for "tied_full"
        if covariance_type == "tied":
            covariance_type = "tied_full"
        self.covariance_type = covariance_type

        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.covariances_init = covariances_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        # Additional features
        self.n_init = n_init
        self.cov_init_method = cov_init_method

        # Device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if any prior is given
        self.use_weight_prior = weight_concentration_prior is not None
        self.use_mean_prior = (mean_prior is not None) and (mean_precision_prior is not None)
        self.use_covariance_prior = (covariance_prior is not None) and (degrees_of_freedom_prior is not None)

        # Store priors
        self._init_priors(
            weight_concentration_prior,
            mean_prior,
            mean_precision_prior,
            covariance_prior,
            degrees_of_freedom_prior
        )

        # Initialize internal parameters for mixture
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.fitted_ = False
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")

    def _init_priors(
        self,
        weight_concentration_prior: Optional[torch.Tensor],
        mean_prior: Optional[torch.Tensor],
        mean_precision_prior: Optional[float],
        covariance_prior: Optional[torch.Tensor],
        degrees_of_freedom_prior: Optional[float]
    ):
        r"""
        Validate and store prior parameters (if MAP is used).

        Parameters
        ----------
        weight_concentration_prior : torch.Tensor or None
            Dirichlet prior for the mixture weights.
        mean_prior : torch.Tensor or None
            Prior means for the Gaussian components.
        mean_precision_prior : float or None
            Scalar precision factor for the mean prior.
        covariance_prior : torch.Tensor or None
            Prior for covariances, shape depends on `covariance_type`.
        degrees_of_freedom_prior : float or None
            Degrees of freedom for the Wishart prior on covariances.
        """
        if self.use_weight_prior:
            # Convert to torch.Tensor if not already.
            if not isinstance(weight_concentration_prior, torch.Tensor):
                weight_concentration_prior = torch.tensor(weight_concentration_prior, device=self.device)
            # If a single value is provided, replicate it to shape (n_components,)
            if weight_concentration_prior.dim() == 0 or (weight_concentration_prior.dim() == 1 and weight_concentration_prior.numel() == 1):
                weight_concentration_prior = weight_concentration_prior.expand(self.n_components)
            elif weight_concentration_prior.dim() == 1 and weight_concentration_prior.numel() != self.n_components:
                raise ValueError(
                    f"weight_concentration_prior must be of shape ({self.n_components},) or a single value, "
                    f"got {weight_concentration_prior.shape}."
                )
            self.weight_concentration_prior = weight_concentration_prior.to(self.device).float()
        else:
            self.weight_concentration_prior = None

        if self.use_mean_prior:
            if mean_prior.shape == (self.n_features,):
                mean_prior = mean_prior.unsqueeze(0).expand(self.n_components, -1)
            elif mean_prior.shape != (self.n_components, self.n_features):
                raise ValueError(
                    "mean_prior must be of shape (n_components, n_features) or (n_features,). "
                    f"Got {mean_prior.shape}."
                )
            if mean_precision_prior <= 0:
                raise ValueError("mean_precision_prior must be > 0.")
            self.mean_prior = mean_prior.to(self.device).float()
            self.mean_precision_prior = float(mean_precision_prior)
        else:
            self.mean_prior = None
            self.mean_precision_prior = None

        if self.use_covariance_prior:
            self.degrees_of_freedom_prior = float(degrees_of_freedom_prior)
            if self.degrees_of_freedom_prior <= self.n_features - 1:
                raise ValueError(
                    "degrees_of_freedom_prior must be greater than "
                    f"{self.n_features - 1}, got {self.degrees_of_freedom_prior}."
                )
            expected_shape = self._expected_covar_shape()
            if covariance_prior.shape != expected_shape:
                raise ValueError(
                    f"covariance_prior must be of shape {expected_shape} for "
                    f"'{self.covariance_type}' covariance type. Got {covariance_prior.shape}."
                )
            self.covariance_prior = covariance_prior.to(self.device).float()
        else:
            self.degrees_of_freedom_prior = None
            self.covariance_prior = None

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

    def _allocate_parameters(self, X: Optional[torch.Tensor] = None):
        r"""
        Allocate and initialize model parameters (weights, means, covariances).

        If X is provided, we can do data-based initialization of means
        using self.init_params (kmeans, kpp, etc.). Otherwise, we fall back
        to random or user-provided means_init.
        """
        # Seed control
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        # ----------------------
        # 1) Allocate weights
        # ----------------------
        if self.weights_init is not None:
            if self.weights_init.shape != (self.n_components,):
                raise ValueError(
                    f"weights_init must be shape ({self.n_components},), got {self.weights_init.shape}."
                )
            weights = self.weights_init.to(self.device).float()
            if torch.sum(weights) < EPS:
                raise ValueError("Initial weights must sum to > 0.")
            self.weights_ = weights / torch.sum(weights)
        else:
            self.weights_ = torch.full(
                (self.n_components,),
                1.0 / self.n_components,
                dtype=torch.float32,
                device=self.device
            )

        # ----------------------
        # 2) Allocate means
        # ----------------------
        # (If user provided means_init, we trust that. Otherwise we do random or data-based init.)
        if self.means_init is not None:
            if self.means_init.shape != (self.n_components, self.n_features):
                raise ValueError(
                    f"means_init must be shape ({self.n_components}, {self.n_features}), "
                    f"got {self.means_init.shape}."
                )
            self.means_ = self.means_init.to(self.device).float()
        else:
            # default fallback => random normal
            self.means_ = torch.randn(
                self.n_components,
                self.n_features,
                device=self.device
            ).float()

            # if we have X, we can do data-based initialization
            if X is not None:
                # call the helper for data-based init
                self._init_means_from_gmminitializer(X)

        # ----------------------
        # 3) Allocate covariances
        # ----------------------
        if self.covariances_init is not None:
            self._check_covariance_init_shape(self.covariances_init)
            self.covariances_ = self.covariances_init.to(self.device).float()
        else:
            self._init_default_covariances()

        # Mark that we've allocated
        self.fitted_ = False
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")

    def _init_means_from_gmminitializer(self, X: torch.Tensor):
        r"""
        If means_ is still random (and means_init is None),
        use self.init_params to call the appropriate GMMInitializer method on X.
        """
        init_method = self.init_params.lower()

        X_cpu = X.cpu()  # GMMInitializer typically works on CPU
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
            # fallback => do nothing; we already have random normal means
            pass

    def _check_covariance_init_shape(self, cov_init: torch.Tensor):
        r"""
        Validate the shape of a given covariance initialization against
        `self.covariance_type`.

        Parameters
        ----------
        cov_init : torch.Tensor
            The user-provided initial covariances to validate.
        """
        expected_shape = self._expected_covar_shape()
        if cov_init.shape != expected_shape:
            raise ValueError(
                f"covariances_init must be of shape {expected_shape} for '{self.covariance_type}'. "
                f"Got {cov_init.shape}."
            )

    def _init_default_covariances(self):
        r"""
        Initialize covariances if the user did not provide any, based on
        `cov_init_method` and the chosen `covariance_type`.
        """
        if self.cov_init_method == "eye":
            self._init_covar_eye()
        elif self.cov_init_method == "random":
            self._init_covar_random()
        elif self.cov_init_method == "empirical":
            self._init_covar_empirical(self.means_)
        else:
            raise ValueError(f"Unsupported cov_init_method: {self.cov_init_method}")

    def _init_covar_eye(self):
        r"""
        Initialize covariances as identity-like matrices/vectors, with `self.reg_covar`
        added to avoid singular matrices.
        """
        if self.covariance_type == 'full':
            out = []
            for _ in range(self.n_components):
                mat = torch.eye(self.n_features, device=self.device)
                mat *= (1.0 + self.reg_covar)
                out.append(mat)
            self.covariances_ = torch.stack(out, dim=0)

        elif self.covariance_type == 'diag':
            self.covariances_ = torch.ones(
                self.n_components, self.n_features, device=self.device
            ) * (1.0 + self.reg_covar)

        elif self.covariance_type == 'spherical':
            self.covariances_ = torch.ones(
                self.n_components, device=self.device
            ) * (1.0 + self.reg_covar)

        elif self.covariance_type == 'tied_full':
            mat = torch.eye(self.n_features, device=self.device)
            mat *= (1.0 + self.reg_covar)
            self.covariances_ = mat

        elif self.covariance_type == 'tied_diag':
            self.covariances_ = torch.ones(self.n_features, device=self.device) * (1.0 + self.reg_covar)

        elif self.covariance_type == 'tied_spherical':
            self.covariances_ = torch.tensor(1.0 + self.reg_covar, device=self.device)

        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def _init_covar_random(self):
        r"""
        Initialize covariances randomly. This can be refined or replaced
        with more sophisticated approaches if desired.
        """
        if self.covariance_type in ('full', 'tied_full'):
            def random_spd(dim):
                A = torch.randn(dim, dim, device=self.device)
                return A @ A.mT + self.reg_covar * torch.eye(dim, device=self.device)
            
            if self.covariance_type == 'full':
                out = [random_spd(self.n_features) for _ in range(self.n_components)]
                self.covariances_ = torch.stack(out, dim=0)
            else:
                self.covariances_ = random_spd(self.n_features)

        elif self.covariance_type in ('diag', 'tied_diag'):
            shape = (self.n_components, self.n_features) if self.covariance_type == 'diag' else (self.n_features,)
            self.covariances_ = torch.rand(shape, device=self.device) + self.reg_covar

        elif self.covariance_type in ('spherical', 'tied_spherical'):
            shape = (self.n_components,) if (self.covariance_type == 'spherical') else ()
            val = torch.rand(shape, device=self.device) + (1.0 + self.reg_covar)
            self.covariances_ = val
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")
        
    def _init_covar_empirical(self, X: torch.Tensor):
        r"""
        Initialize each covariance by assigning data points to their nearest mean,
        then computing the cluster-wise empirical covariance matrices (with regularization).
        Useful when `init_params == 'kmeans'`.
        """
        assert self.init_params == 'kmeans', (
            "Empirical covariance initialization requires k-means init for means."
        )
        
        X = X.to(self.device)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        n_samples = X.size(0)
        
        distances = torch.cdist(X, self.means_)  # (n_samples, n_components)
        labels = torch.argmin(distances, dim=1)  # (n_samples,)
        
        if self.covariance_type == 'full':
            new_covs = []
            
            for k in range(self.n_components):
                cluster_mask = (labels == k)
                if not torch.any(cluster_mask):
                    cov_k = torch.eye(self.n_features, device=self.device) * (1.0 + self.reg_covar)
                else:
                    cluster_data = X[cluster_mask]
                    cov_k = torch.cov(cluster_data.T)
                    cov_k += self.reg_covar * torch.eye(self.n_features, device=self.device)
                new_covs.append(cov_k)
            
            self.covariances_ = torch.stack(new_covs, dim=0)

        elif self.covariance_type == 'diag':
            new_covs = []
            
            for k in range(self.n_components):
                cluster_mask = (labels == k)
                if not torch.any(cluster_mask):
                    cov_k = torch.ones(self.n_features, device=self.device) * (1.0 + self.reg_covar)
                else:
                    cluster_data = X[cluster_mask]
                    cov_mat = torch.cov(cluster_data.T)
                    diag_vals = torch.diagonal(cov_mat)
                    diag_vals += self.reg_covar
                    cov_k = diag_vals
                new_covs.append(cov_k)
            
            self.covariances_ = torch.stack(new_covs, dim=0)

        elif self.covariance_type == 'spherical':
            new_covs = []
            
            for k in range(self.n_components):
                cluster_mask = (labels == k)
                if not torch.any(cluster_mask):
                    cov_k = 1.0 + self.reg_covar
                else:
                    cluster_data = X[cluster_mask]
                    cov_mat = torch.cov(cluster_data.T)
                    val = torch.mean(torch.diagonal(cov_mat))
                    val = max(val.item(), self.reg_covar)
                    cov_k = val
                new_covs.append(torch.tensor(cov_k, device=self.device))
            
            self.covariances_ = torch.stack(new_covs, dim=0)

        elif self.covariance_type == "tied_full":
            sum_cov = torch.zeros(self.n_features, self.n_features, device=self.device)
            
            for k in range(self.n_components):
                cluster_mask = (labels == k)
                cluster_data = X[cluster_mask]
                if cluster_data.size(0) > 0:
                    diff = cluster_data - cluster_data.mean(dim=0, keepdim=True)
                    cov_k = diff.T @ diff
                    sum_cov += cov_k
            
            sum_cov /= n_samples
            sum_cov += self.reg_covar * torch.eye(self.n_features, device=self.device)
            
            self.covariances_ = sum_cov

        elif self.covariance_type == "tied_diag":
            sum_diag = torch.zeros(self.n_features, device=self.device)
            
            for k in range(self.n_components):
                cluster_mask = (labels == k)
                cluster_data = X[cluster_mask]
                if cluster_data.size(0) > 0:
                    diff = cluster_data - cluster_data.mean(dim=0, keepdim=True)
                    sum_diag += (diff * diff).sum(dim=0)
            
            sum_diag /= n_samples
            sum_diag += self.reg_covar
            self.covariances_ = sum_diag

        elif self.covariance_type == "tied_spherical":
            total_sum = 0.0
            
            for k in range(self.n_components):
                cluster_mask = (labels == k)
                cluster_data = X[cluster_mask]
                if cluster_data.size(0) > 0:
                    diff = cluster_data - cluster_data.mean(dim=0, keepdim=True)
                    total_sum += diff.pow(2).sum().item()
            
            var = total_sum / (n_samples * self.n_features)
            var = max(var, self.reg_covar)
            self.covariances_ = torch.tensor(var, device=self.device)
        
        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

    def fit(self,
            X: torch.Tensor,
            max_iter: Optional[int] = None,
            tol: Optional[float] = None,
            random_state: Optional[int] = None,
            warm_start: Optional[bool] = None) -> "GaussianMixture":
        r"""
        Fit the GMM to the data using the EM algorithm, possibly across multiple
        random initializations.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features).
        max_iter : int, optional
            Maximum number of EM iterations. If None, uses `self.max_iter`. (default: None)
        tol : float, optional
            Convergence tolerance. If None, uses `self.tol`. (default: None)
        random_state : int or None, optional
            Random seed. Overrides `self.random_state` if provided. (default: None)
        warm_start : bool or None, optional
            Whether to warm-start from previously fitted parameters in multi-init
            settings. If None, uses `self.warm_start`. (default: None)

        Returns
        -------
        self : GaussianMixture
            The fitted model instance.
        """

        if X.size(0) < self.n_components:
            raise ValueError(
                f"n_components={self.n_components} should be <= n_samples={X.size(0)}."
            )
        if self.n_components <= 0:
            raise ValueError(f"Invalid value for n_components: {self.n_components}")
        if tol is not None and tol <= 0:
            raise ValueError(f"Invalid value for tol: {tol}")
        if max_iter is not None and max_iter <= 0:
            raise ValueError(f"Invalid value for max_iter: {max_iter}")

        if warm_start is None:
            warm_start = self.warm_start

        if max_iter is None:
            max_iter = self.max_iter

        if tol is None:
            tol = self.tol

        best_lower_bound = -float("inf")
        best_params = None

        if random_state is not None:
            self.random_state = random_state

        X = X.to(self.device)
        if self.n_features is None:
            self.n_features = X.shape[1]
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but n_features={self.n_features}.")
        
        self._allocate_parameters()

        for init_idx in range(self.n_init):
            if warm_start and self.n_init > 1:
                raise UserWarning("Leaving warm_start=True with n_init>1 will not re-initialize parameters for each run.")

            # 1) Allocate parameters (including data-based means if needed)
            #    do this if not warm-starting or if it's the first run or if multiple inits
            if not warm_start or not self.fitted_ or init_idx > 0:
                self._allocate_parameters(X)

            # 2) Run one EM
            self._fit_single_run(X, max_iter, tol, run_idx=init_idx)

            # warning for degenerate clusters
            if torch.any(self.weights_ < 1e-8):
                raise UserWarning("Some cluster(s) have near-zero weight. Check for degenerate solutions.")
                
            # 3) Track best solution
            if self.lower_bound_ > best_lower_bound:
                best_lower_bound = self.lower_bound_
                best_params = (
                    self.weights_.clone(),
                    self.means_.clone(),
                    self.covariances_.clone(),
                    self.converged_,
                    self.n_iter_,
                    self.lower_bound_
                )

            self.fitted_ = True

        # Save best result
        if best_params is not None:
            self.weights_, self.means_, self.covariances_, self.converged_, self.n_iter_, self.lower_bound_ = best_params
        
        # warning if we didn't converge
        if not self.converged_:
            raise UserWarning("EM algorithm did not converge. Try increasing max_iter or lowering tol.")

    def _fit_single_run(self, X: torch.Tensor, max_iter: int, tol: float, run_idx: int = 0):
        r"""
        Perform one run of the EM algorithm (E-step + M-step).

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features).
        max_iter : int
            Maximum iterations for this run.
        tol : float
            Convergence tolerance (relative improvement in log-likelihood).
        run_idx : int
            Which initialization run we are on (useful for logging).
        """

        X = X.to(self.device)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features, but n_features={self.n_features}.")

        prev_lower_bound = None
        for n_iter in range(max_iter):
            resp, log_prob_norm = self._e_step(X)
            self.lower_bound_ = log_prob_norm.mean().item()

            self._m_step(X, resp)

            if prev_lower_bound is not None:
                rel_change = abs(self.lower_bound_ - prev_lower_bound) / (abs(prev_lower_bound) + EPS)
                if rel_change < tol:
                    self.converged_ = True
                    if self.verbose:
                        print(f"[InitRun {run_idx}] Converged at iteration {n_iter}, lower bound={self.lower_bound_:.5f}")
                    break
            prev_lower_bound = self.lower_bound_

            if self.verbose and (n_iter % self.verbose_interval == 0):
                print(f"[InitRun {run_idx}] Iter {n_iter}, lower bound: {self.lower_bound_:.5f}")

        self.n_iter_ = n_iter

    # ---------------------------
    # E-step
    # ---------------------------
    def _e_step(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        E-step: compute the responsibilities (posterior probabilities of components)
        and the per-sample log-likelihood.

        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).

        Returns
        -------
        resp : torch.Tensor
            Responsibilities for each sample w.r.t. each component, shape
            (n_samples, n_components).
        log_prob_norm : torch.Tensor
            Log-sum-exp of the weighted probabilities for each sample,
            shape (n_samples,).
        """
        log_weights = torch.log(self.weights_ + EPS)

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

        # Add log(weights)
        log_prob = log_prob + log_weights.unsqueeze(0)
        log_prob_norm = torch.logsumexp(log_prob, dim=1)
        log_resp = log_prob - log_prob_norm.unsqueeze(1)
        resp = torch.exp(log_resp)
        return resp, log_prob_norm

    # ------ Full ------
    def _estimate_log_gaussian_full(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log-probabilities under a full covariance model for each
        sample-component pair.

        Uses a Cholesky factorization for numerical stability.
        """
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        try:
            chol = torch.linalg.cholesky(self.covariances_)
        except RuntimeError as e:
            raise ValueError(f"Cholesky failed: {e}")

        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1)
        diff_ = diff.unsqueeze(-1)
        solve = torch.cholesky_solve(diff_, chol)
        mahal = (diff_ * solve).sum(dim=(2, 3))

        return -0.5 * (
            self.n_features * torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
            + log_det.unsqueeze(0)
            + mahal
        )

    # ------ Diag ------
    def _estimate_log_gaussian_diag(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log-probabilities under a diagonal covariance model for each
        sample-component pair.
        """
        precisions = 1.0 / (self.covariances_ + EPS)
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        log_det = torch.sum(torch.log(self.covariances_ + EPS), dim=1)
        mahal = torch.sum(diff.pow(2) * precisions.unsqueeze(0), dim=2)

        return -0.5 * (
            self.n_features * torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
            + log_det.unsqueeze(0)
            + mahal
        )

    # ------ Spherical ------
    def _estimate_log_gaussian_spherical(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log-probabilities under a spherical covariance model for each
        sample-component pair.
        """
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        precisions = 1.0 / (self.covariances_ + EPS)
        mahal = torch.sum(diff.pow(2), dim=2) * precisions.unsqueeze(0)
        log_det = self.n_features * torch.log(self.covariances_ + EPS)

        return -0.5 * (
            self.n_features * torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
            + log_det.unsqueeze(0)
            + mahal
        )

    # ------ Tied Full ------
    def _estimate_log_gaussian_tied_full(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log-probabilities under a tied_full covariance model for each
        sample-component pair.
        """
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        try:
            chol = torch.linalg.cholesky(self.covariances_)
        except RuntimeError as e:
            raise ValueError(f"Tied full: Cholesky failed: {e}")

        log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()
        diff_ = diff.unsqueeze(-1)
        solve = torch.cholesky_solve(diff_, chol)
        mahal = (diff_ * solve).sum(dim=(2, 3))

        return -0.5 * (
            self.n_features * torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
            + log_det
            + mahal
        )

    # ------ Tied Diag ------
    def _estimate_log_gaussian_tied_diag(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log-probabilities under a tied_diag covariance model for each
        sample-component pair.
        """
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        cov_vector = self.covariances_ + EPS
        log_det = torch.sum(torch.log(cov_vector))
        precisions = 1.0 / cov_vector
        mahal = torch.sum(diff.pow(2) * precisions, dim=2)

        return -0.5 * (
            self.n_features * torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
            + log_det
            + mahal
        )

    # ------ Tied Spherical ------
    def _estimate_log_gaussian_tied_spherical(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute log-probabilities under a tied_spherical covariance model for each
        sample-component pair.
        """
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        var = self.covariances_ + EPS
        prec = 1.0 / var
        mahal = torch.sum(diff.pow(2), dim=2) * prec
        log_det = self.n_features * torch.log(var)

        return -0.5 * (
            self.n_features * torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
            + log_det
            + mahal
        )

    # ---------------------------
    # M-step
    # ---------------------------
    def _m_step(self, X: torch.Tensor, resp: torch.Tensor):
        r"""
        M-step: update the parameters (weights, means, covariances) based on
        the current responsibilities.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_features).
        resp : torch.Tensor
            Current responsibilities for each sample w.r.t. each component
            (from E-step).
        """
        n_samples = X.size(0)
        nk = resp.sum(dim=0) + EPS

        # Update weights (MAP or MLE)
        if self.use_weight_prior:
            alpha = self.weight_concentration_prior
            total_alpha = alpha.sum()
            self.weights_ = (nk + alpha - 1.0) / (n_samples + total_alpha - self.n_components)
        else:
            self.weights_ = nk / n_samples
        self.weights_.clamp_(min=EPS)

        # Update means (MAP or MLE)
        if self.use_mean_prior:
            kappa0 = self.mean_precision_prior
            numerator = resp.t() @ X + kappa0 * self.mean_prior
            denom = nk.unsqueeze(1) + kappa0
            self.means_ = numerator / denom
        else:
            self.means_ = (resp.t() @ X) / nk.unsqueeze(1)

        # Update covariances (MAP or MLE)
        if self.use_covariance_prior:
            self._update_covariances_map(X, resp, nk)
        else:
            self._update_covariances_mle(X, resp, nk)

    def _update_covariances_map(self, X, resp, nk):
        r"""
        Update covariances with MAP estimation, depending on `covariance_type`.
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

    def _update_covariances_mle(self, X, resp, nk):
        r"""
        Update covariances using MLE, depending on `covariance_type`.
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

    # ---------- MAP updates ----------
    # MAP Full
    def _update_map_full(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        weighted_diff = resp.unsqueeze(-1).unsqueeze(-1) * diff.unsqueeze(3) * diff.unsqueeze(2)
        sum_diff = weighted_diff.sum(dim=0)

        mean_diff = (self.means_ - self.mean_prior).unsqueeze(-1)
        prior_term = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1).unsqueeze(-1) \
                     * mean_diff @ mean_diff.transpose(-1, -2)

        df = self.degrees_of_freedom_prior + nk.unsqueeze(-1).unsqueeze(-1) + self.n_features

        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar * torch.eye(self.n_features, device=self.device).unsqueeze(0)
        ) / df

    # MAP Diag
    def _update_map_diag(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = (resp.unsqueeze(-1) * diff.pow(2)).sum(dim=0)  # (K, D)

        mean_diff2 = (self.means_ - self.mean_prior).pow(2)
        prior_term = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1) * mean_diff2
        df = self.degrees_of_freedom_prior + nk.unsqueeze(-1) + self.n_features

        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar
        ) / df

    # MAP Spherical
    def _update_map_spherical(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        diff2 = diff.pow(2).sum(dim=2)
        sum_diff = (resp * diff2).sum(dim=0)

        mean_diff2 = (self.means_ - self.mean_prior).pow(2).sum(dim=1)
        prior_term = (nk / (nk + self.mean_precision_prior)) * mean_diff2
        df = self.degrees_of_freedom_prior + nk + self.n_features

        self.covariances_ = (
            self.covariance_prior + sum_diff + prior_term + self.reg_covar
        ) / (df * self.n_features)

    # MAP Tied Full
    def _update_map_tied_full(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = torch.einsum('nk,nkd,nke->de', resp, diff, diff)

        mean_diff = (self.means_ - self.mean_prior).unsqueeze(-1)
        prior_term = (
            (nk / (nk + self.mean_precision_prior)).unsqueeze(-1).unsqueeze(-1)
            * torch.matmul(mean_diff, mean_diff.transpose(-1, -2))
        )
        prior_term = prior_term.sum(dim=0)  # sum across components
        df = self.degrees_of_freedom_prior + nk.sum() + self.n_features

        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar * torch.eye(self.n_features, device=self.device)
        ) / df

    # MAP Tied Diag
    def _update_map_tied_diag(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = torch.einsum('nk,nkd->d', resp, diff.pow(2))

        mean_diff2 = (self.means_ - self.mean_prior).pow(2)
        # shape (K, D)
        prior_term = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1) * mean_diff2
        prior_term = prior_term.sum(dim=0)  # shape (D,)

        df = self.degrees_of_freedom_prior + nk.sum() + self.n_features

        # covariance_prior is shape (n_features,)
        self.covariances_ = (
            self.covariance_prior
            + sum_diff
            + prior_term
            + self.reg_covar
        ) / df

    # MAP Tied Spherical
    def _update_map_tied_spherical(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        diff2 = diff.pow(2).sum(dim=2)
        sum_diff = torch.einsum('nk,nk->', resp, diff2)

        mean_diff2 = (self.means_ - self.mean_prior).pow(2).sum(dim=1)
        # shape (K,)
        prior_term = (nk / (nk + self.mean_precision_prior)) * mean_diff2
        prior_term_total = prior_term.sum()

        df = self.degrees_of_freedom_prior + nk.sum() + self.n_features
        self.covariances_ = (
            self.covariance_prior + sum_diff + prior_term_total + self.reg_covar
        ) / (df * self.n_features)

    # ---------- MLE updates  ----------
    # MLE Full
    def _update_mle_full(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        weighted_diff = resp.unsqueeze(-1).unsqueeze(-1) * diff.unsqueeze(3) * diff.unsqueeze(2)
        sum_diff = weighted_diff.sum(dim=0)
        covs = sum_diff / nk.unsqueeze(-1).unsqueeze(-1)
        covs += self.reg_covar * torch.eye(self.n_features, device=self.device).unsqueeze(0)
        self.covariances_ = covs

    # MLE Diag
    def _update_mle_diag(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = (resp.unsqueeze(-1) * diff.pow(2)).sum(dim=0)
        cov_diag = sum_diff / nk.unsqueeze(-1)
        cov_diag += self.reg_covar
        self.covariances_ = cov_diag

    # MLE Spherical
    def _update_mle_spherical(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        diff2 = diff.pow(2).sum(dim=2)
        sum_diff2 = (resp * diff2).sum(dim=0)
        cov_spherical = sum_diff2 / (nk * self.n_features)
        cov_spherical += self.reg_covar
        self.covariances_ = cov_spherical

    # MLE Tied Full
    def _update_mle_tied_full(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = torch.einsum('nk,nkd,nke->de', resp, diff, diff)
        cov_tied = sum_diff / nk.sum()
        cov_tied += self.reg_covar * torch.eye(self.n_features, device=self.device)
        self.covariances_ = cov_tied

    # MLE Tied Diag
    def _update_mle_tied_diag(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = torch.einsum('nk,nkd->d', resp, diff.pow(2))
        cov_tied_diag = sum_diff / nk.sum()
        cov_tied_diag += self.reg_covar
        self.covariances_ = cov_tied_diag

    # MLE Tied Spherical
    def _update_mle_tied_spherical(self, X, resp, nk):
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        sum_diff = torch.einsum('nk,nkd->', resp, diff.pow(2))
        cov_tied_spherical = sum_diff / (nk.sum() * self.n_features)
        cov_tied_spherical += self.reg_covar
        self.covariances_ = cov_tied_spherical

    # ---------------------------
    # Prediction / Scoring
    # ---------------------------
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Predict cluster labels for each sample in X (i.e., argmax of responsibilities).

        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).

        Returns
        -------
        labels : torch.Tensor
            Cluster labels for each sample, shape (n_samples,).
        """
        if not self.fitted_:
            raise ValueError("Call fit() before predict().")
        resp, _ = self._e_step(X.to(self.device))
        return torch.argmax(resp, dim=1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Return the posterior probabilities (responsibilities) of each sample
        belonging to each component.

        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).

        Returns
        -------
        resp : torch.Tensor
            Responsibilities, shape (n_samples, n_components).
        """
        if not self.fitted_:
            raise ValueError("Call fit() before predict_proba().")
        resp, _ = self._e_step(X.to(self.device))
        return resp

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Compute per-sample log-likelihood under the model.

        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).

        Returns
        -------
        log_prob : torch.Tensor
            Log-likelihood for each sample, shape (n_samples,).
        """
        if not self.fitted_:
            raise ValueError("Call fit() before score_samples().")
        _, log_prob_norm = self._e_step(X.to(self.device))
        return log_prob_norm

    def score(self, X: torch.Tensor) -> float:
        r"""
        Compute the average log-likelihood of X under the model.

        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_samples, n_features).

        Returns
        -------
        score : float
            Mean log-likelihood across all samples.
        """
        return self.score_samples(X).mean().item()

    def sample(self, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Generate new samples from the fitted GMM.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        samples : torch.Tensor
            Generated samples of shape (n_samples, n_features).
        indices : torch.Tensor
            The indices of the component each sample came from, shape (n_samples,).
        """
        if not self.fitted_:
            raise ValueError("Call fit() before sample().")

        # Choose components
        indices = torch.multinomial(self.weights_, n_samples, replacement=True)
        means = self.means_[indices]

        # Construct covariance
        covariances = self._build_covariances_for_sampling(indices, n_samples)
        samples = MultivariateNormal(means, covariance_matrix=covariances).sample()
        return samples, indices

    def _build_covariances_for_sampling(self, indices, n_samples):
        r"""
        Construct a batch of covariance matrices for the chosen component indices.

        Parameters
        ----------
        indices : torch.Tensor
            Indices of the chosen components, shape (n_samples,).
        n_samples : int
            Number of total samples to be generated.

        Returns
        -------
        covs : torch.Tensor
            A batch of covariance matrices of shape (n_samples, n_features, n_features).
        """
        if self.covariance_type == 'full':
            return self.covariances_[indices]

        elif self.covariance_type == 'diag':
            return torch.diag_embed(self.covariances_[indices])

        elif self.covariance_type == 'spherical':
            eye = torch.eye(self.n_features, device=self.device).unsqueeze(0)
            return eye * self.covariances_[indices].view(-1, 1, 1)

        elif self.covariance_type == 'tied_full':
            # single shared => repeat
            return self.covariances_.unsqueeze(0).expand(n_samples, -1, -1)

        elif self.covariance_type == 'tied_diag':
            diag_mat = torch.diag_embed(self.covariances_)
            return diag_mat.unsqueeze(0).expand(n_samples, -1, -1)

        elif self.covariance_type == 'tied_spherical':
            eye = torch.eye(self.n_features, device=self.device).unsqueeze(0)
            return eye * self.covariances_

        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")
