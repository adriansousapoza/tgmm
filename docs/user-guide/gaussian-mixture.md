# Gaussian Mixture Model

The `GaussianMixture` class is the core component of tgmm, implementing the Expectation-Maximization (EM) algorithm for fitting Gaussian Mixture Models with optional Bayesian priors (MAP estimation).

## Overview

A Gaussian Mixture Model represents a probability distribution as a weighted sum of Gaussian components:

$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
$$

where:

- $K$ is the number of components
- $\pi_k$ are the mixing weights ($\sum_k \pi_k = 1$)
- $\mu_k$ are the component means
- $\Sigma_k$ are the component covariances

## Basic Usage

```python
from tgmm import GaussianMixture
import torch

# Create GMM
gmm = GaussianMixture(
    n_components=3,      # Number of Gaussian components
    n_features=2,        # Data dimensionality
    covariance_type='full'  # Type of covariance matrix
)

# Fit to data
gmm.fit(X)

# Make predictions
labels = gmm.predict(X)              # Hard assignments
probabilities = gmm.predict_proba(X)  # Soft assignments
log_likelihood = gmm.score(X)         # Average log-likelihood
```

## Covariance Types

tgmm supports six different covariance types, offering trade-offs between flexibility and computational cost:

### Component-Specific Covariances

Each component has its own covariance matrix:

#### Full Covariance

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='full')
```

- **Shape**: $(K, d, d)$ where $K$ is number of components, $d$ is dimensionality
- **Parameters**: $K \cdot d \cdot (d+1)/2$ (due to symmetry)
- **Use**: Most flexible; clusters can have arbitrary orientation and shape
- **Computational Cost**: Highest

#### Diagonal Covariance

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='diag')
```

- **Shape**: $(K, d)$
- **Parameters**: $K \cdot d$
- **Use**: Axis-aligned ellipsoids; assumes feature independence
- **Computational Cost**: Medium

#### Spherical Covariance

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='spherical')
```

- **Shape**: $(K,)$ (single variance per component)
- **Parameters**: $K$
- **Use**: Spherical/circular clusters; all features have same variance
- **Computational Cost**: Lowest

### Tied Covariances

All components share the same covariance structure:

#### Tied Full

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='tied_full')
```

- **Shape**: $(d, d)$
- **Parameters**: $d \cdot (d+1)/2$
- **Use**: All clusters have same shape/orientation, different centers

#### Tied Diagonal

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='tied_diag')
```

- **Shape**: $(d,)$
- **Parameters**: $d$
- **Use**: All clusters are axis-aligned ellipsoids with same shape

#### Tied Spherical

```python
gmm = GaussianMixture(n_components=3, n_features=2, covariance_type='tied_spherical')
```

- **Shape**: Scalar
- **Parameters**: 1
- **Use**: All clusters are spheres with same radius

## Estimation Methods

### Maximum Likelihood Estimation (MLE)

Standard EM algorithm without priors:

```python
gmm = GaussianMixture(n_components=3, n_features=2)
gmm.fit(X)
```

### Maximum A Posteriori (MAP) with Priors

Bayesian approach with conjugate priors:

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    # Dirichlet prior on weights
    weight_concentration_prior=torch.ones(3) * 2.0,
    # Gaussian prior on means
    mean_prior=torch.zeros(3, 2),
    mean_precision_prior=0.01,
    # Inverse-Wishart prior on covariances
    covariance_prior=torch.eye(2).unsqueeze(0).repeat(3, 1, 1),
    degrees_of_freedom_prior=3.0
)
gmm.fit(X)
```

See the [Priors Tutorial](../notebooks/09_priors.ipynb) for details.

### Normal-Inverse-Wishart (NIW) Conjugate Prior

When both mean and covariance priors are specified, tgmm uses the NIW conjugate prior:

```python
# NIW automatically used when both priors are set
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    mean_prior=torch.zeros(3, 2),
    mean_precision_prior=0.1,
    covariance_prior=torch.eye(2).unsqueeze(0).repeat(3, 1, 1),
    degrees_of_freedom_prior=5.0
)
```

## Initialization Strategies

Good initialization is critical for EM convergence:

### K-means Initialization (Recommended)

```python
gmm = GaussianMixture(n_components=3, n_features=2, init_means='kmeans')
```

Runs a pure-PyTorch Lloyd's algorithm (started from a k-means++ initialization) to initialize component means -- no scikit-learn dependency.

### K-means++ Initialization

```python
gmm = GaussianMixture(n_components=3, n_features=2, init_means='kpp')
```

More sophisticated initialization that spreads initial centers apart.

### Random Initialization

```python
gmm = GaussianMixture(n_components=3, n_features=2, init_means='random')
```

Randomly selects data points as initial means.

### Custom Initialization

```python
initial_means = torch.tensor([[0, 0], [5, 5], [0, 5]], dtype=torch.float32)
gmm = GaussianMixture(n_components=3, n_features=2, init_means=initial_means)
```

### Multiple Random Restarts

To avoid local minima:

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    n_init=10,  # Run EM 10 times with different initializations
    random_state=42
)
```

## Fitting Algorithms

`algorithm` selects which EM variant drives `fit()`. One sentence each on
when to reach for them:

| `algorithm` | When to use it |
| --- | --- |
| `'em'` (default) | Standard soft-assignment EM -- the right default for most fits. |
| `'cem'` | Classification EM (hard assignments between the E- and M-steps) -- converges faster and gives a crisper partition, at the cost of being more prone to local minima than soft EM. |
| `'sem'` | Stochastic EM -- samples a hard assignment from the responsibilities each iteration instead of taking the argmax; use it to escape poor local optima, or to detect an overspecified `n_components` via `n_empty_components_`. |
| `'saem'` | Stochastic Approximation EM -- SEM with a decaying step size blended across iterations, so it *does* converge pointwise (unlike plain SEM); use it when you want SEM's exploration with a convergent stopping point. |

`temperature` (deterministic annealing) and `step_size` (online/mini-batch
EM) are independent knobs layered on top of these -- see below.

### Classification EM (CEM)

```python
gmm = GaussianMixture(n_components=3, n_features=2, algorithm='cem')
gmm.fit(X)
```

!!! note "`cem=True` is deprecated"
    The older `cem=True` keyword still works (it's treated as
    `algorithm='cem'`), but emits a `DeprecationWarning` -- prefer
    `algorithm='cem'` in new code.

### Stochastic EM (SEM)

```python
gmm = GaussianMixture(n_components=3, n_features=2, algorithm='sem', max_iter=200)
gmm.fit(X)

print(gmm.log_likelihood_history_)   # per-sweep log-likelihood trace
print(gmm.n_empty_components_)       # > 0 re-seeds suggest n_components is too large
```

SEM never checks convergence -- it always runs the full `max_iter` sweeps
(same convention as Gibbs mode), since the sampled hard assignment keeps
the iterate moving even near a mode. The final parameters come from
`sem_estimate` applied to the sweeps at or after `sem_burn_in` (default
`max_iter // 5`): `'best'` (default) keeps the highest-log-likelihood
post-burn-in sweep; `'mean'` averages parameters across all of them, which
assumes no label switching between sweeps -- prefer `'best'` whenever
`n_components` might be overspecified.

### Stochastic Approximation EM (SAEM)

```python
gmm = GaussianMixture(n_components=3, n_features=2, algorithm='saem', max_iter=200)
gmm.fit(X)
print(gmm.converged_)
```

SAEM blends each sweep's sampled-assignment statistics into a running
average with a decaying step size (`step_size`, defaulting to `'auto'`
for this algorithm), so `tol`-based convergence applies exactly as in
ordinary EM/CEM -- the "SEM that converges."

### Deterministic Annealing (DAEM)

```python
gmm = GaussianMixture(
    n_components=3, n_features=2,
    temperature=(0.1, 30),  # (beta_0, n_anneal)
)
gmm.fit(X)
print(gmm.annealing_schedule_)  # beta actually used at each E-step
```

Works with `algorithm` in `{'em', 'cem', 'sem'}`. Tempers the E-step's
joint log-density by a `beta` that rises geometrically from `beta_0` to
`1.0` over the first `n_anneal` iterations (then holds at `1.0`);
convergence checks are suspended while `beta < 1.0`. Use this when
initialization is unreliable, e.g. several components starting out close
together -- annealing smooths the likelihood surface early on, when
standard EM would otherwise settle into the first nearby local optimum.
It is a heuristic, not a guarantee: an initialization that is *exactly*
symmetric (e.g. every mean identical to machine precision) stays
symmetric under annealing too, since tempering a perfectly symmetric
responsibility distribution is still perfectly symmetric -- some
real-world asymmetry in the starting point, even a small one, is what
annealing needs to amplify.

### Mini-Batch and Online EM

```python
gmm = GaussianMixture(
    n_components=3, n_features=2,
    batch_size=256, step_size='auto',  # or a float, or a callable t -> gamma_t
    max_iter=10,  # epochs
)
gmm.fit(X)  # shuffles X into batch_size-sized mini-batches each epoch
print(gmm.log_likelihood_history_)  # one entry per epoch
```

For streaming data, `partial_fit` continues training after an initial
`fit()` call (or explicit parameter initialization):

```python
gmm = GaussianMixture(n_components=3, n_features=2, batch_size=256, step_size='auto')
gmm.fit(first_batch)
for batch in stream_of_batches:
    gmm.partial_fit(batch)
```

`step_size=None` (the default) keeps `batch_size` as a pure memory
optimization (exact EM, see [Large Datasets](#large-datasets-batch_size)
above) -- setting `step_size` is what switches to online EM. `partial_fit`
assumes a roughly constant batch size across calls; unlike `fit()`'s own
epoch loop, it does not rescale each batch against a known total dataset
size (there isn't one in a true streaming setting).

## Supervised Fitting

If you already know the true class of every point, `fit` can skip EM/CEM
entirely and fit one Gaussian per class directly from the labeled data —
pass `labels`:

```python
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X, labels=y)  # y: shape (n_samples,), one of 3 distinct values

print(gmm.classes_)      # the 3 distinct label values seen, sorted
print(gmm.converged_)    # always True
print(gmm.n_iter_)       # always 1
```

This is the fully-labeled limit of Classification EM: instead of assigning
each point to its highest-responsibility component (`argmax(resp)`), the
assignment comes from the true label. Since that assignment never depends
on the current parameter estimates, there's nothing left to iterate — a
single M-step already recovers the same per-class mean/covariance/weight
you'd get from Classification EM if it had converged with the correct
partition. `n_components` must equal the number of distinct values in
`labels`; label values don't need to be contiguous integers (`classes_[k]`
recovers the original label for component `k`). `n_init` and `warm_start`
are ignored — the result is deterministic. Any configured `mean_prior` /
`covariance_prior` still applies, giving a regularized supervised fit for
classes with few samples.

See the [Supervised Fitting notebook](../notebooks/05_supervised_gmm.ipynb) for
a worked comparison of plain EM, CEM, and supervised fitting on the same
synthetic data.

## Gibbs Sampling: Inferring the Number of Components

If you don't know `K` ahead of time, `GaussianMixture(n_components=None,
...)` switches from EM to a collapsed Gibbs sampler that infers the number
of components from the data during `fit`, instead of requiring it fixed in
advance:

```python
from tgmm import GaussianMixture

# A principled starting point for the NIW prior Gibbs mode requires,
# estimated from a quick k-means pass over the data.
mean_prior, mean_precision_prior, covariance_prior, degrees_of_freedom_prior = \
    GaussianMixture.suggest_priors(X, n_components=10)

gmm = GaussianMixture(
    n_components=None,       # switches to Gibbs sampling
    covariance_type='full',  # 'full', 'diag', or 'spherical' -- no tied_* support
    max_components=20,       # upper bound on tracked components
    alpha=1.0,                # Dirichlet process concentration
    mean_prior=mean_prior,
    mean_precision_prior=mean_precision_prior,
    covariance_prior=covariance_prior,
    degrees_of_freedom_prior=degrees_of_freedom_prior,
    max_iter=50,               # an exact sweep count in Gibbs mode -- see below
)
gmm.fit(X)

print(gmm.n_components_)  # the inferred component count
print(gmm.active_)        # which of the max_components slots are in use
```

Unlike EM, Gibbs mode has **no default prior**: an earlier version of this
project auto-resolved one, and the auto-resolved default measurably
distorted fitted covariances, so `__init__` requires `mean_prior`,
`mean_precision_prior`, `covariance_prior`, and `degrees_of_freedom_prior`
explicitly whenever `n_components=None`. `GaussianMixture.suggest_priors(X,
n_components=<a rough guess>)` gives a sensible starting point -- inspect
and adjust it before passing it in if needed.

Key parameters (see the [API reference](../api/gaussian-mixture.md) for
full detail):

- `max_components`: upper bound on the number of components tracked while sampling (default `20`). Set to `None` for the unbounded variant, which creates and destroys components on the fly instead of using a fixed number of slots.
- `alpha`: Dirichlet process concentration -- higher values make the sampler more willing to propose new components.
- `init_k`: number of components to seed the sampler with.
- `weight_threshold`: a component is treated as inactive (excluded from `n_components_`/`active_`) once its *expected point count*, `weight * n_samples`, drops to or below this value -- a count threshold, not a raw weight.
- `burn_in`: unbounded variant only -- sweeps excluded from `n_components_history_`, the per-sweep trace of how many components were active. It does not affect the fitted parameters themselves, which always come from the final sweep regardless of `burn_in`.

Gibbs sampling has no EM-style convergence check, so `max_iter` is an
*exact* sweep count rather than an early-stoppable cap. Each sweep is an
O(n_samples) Python-level loop over every point, so the EM-oriented default
of 1000 can be far slower than necessary for Gibbs mode -- pass an
explicit, usually much smaller, value.

See the [Dirichlet Process Mixture notebook](../notebooks/11_dpgmm.ipynb) for
the stick-breaking/Gibbs-sampling math, a step-by-step walkthrough of the
sampler itself, and a worked comparison against classical EM; see [the
design
doc](https://github.com/adriansousapoza/tgmm/blob/main/docs/superpowers/specs/2026-07-28-dpgmm-design.md)
for the full derivation and rationale.

## Convergence Control

```python
gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    max_iter=200,     # Maximum EM iterations
    tol=1e-4,         # Convergence tolerance on log-likelihood change
    verbose=True,     # Print progress
    verbose_interval=10  # Print every 10 iterations
)
gmm.fit(X)

# Check convergence
print(f"Converged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")
print(f"Final log-likelihood: {gmm.lower_bound_:.2f}")
```

By default, `tol` is checked against the *relative* change in log-likelihood
between iterations (`convergence='relative'`). `convergence='aitken'` uses an
Aitken-accelerated extrapolation of the asymptotic log-likelihood instead,
which can detect convergence earlier on fits that improve slowly but
steadily:

```python
gmm = GaussianMixture(n_components=3, n_features=2, convergence='aitken')
gmm.fit(X)
```

## Large Datasets: `batch_size`

For datasets too large to comfortably fit the E-step/M-step in memory at
once, `batch_size` processes `X` in row chunks instead of all at once:

```python
gmm = GaussianMixture(n_components=10, n_features=64, batch_size=4096)
gmm.fit(X)  # same result as batch_size=None, bounded peak memory
```

This changes nothing about the result (up to ordinary floating-point
summation-order differences) -- it's purely a memory/speed knob. Peak memory
is roughly `O(batch_size * n_components * n_features)` regardless of how
many rows `X` has.

## Sampling from the Model

Generate new samples from the fitted distribution:

```python
# Basic sampling
samples, component_ids = gmm.sample(100)

# Sample from specific component
samples, _ = gmm.sample(50, component=0)

# Sample within confidence region
samples, _ = gmm.sample(100, confidence=0.95)

# Sample within standard deviation range
samples, _ = gmm.sample(100, std_range=(1.0, 2.0))

# Sample outliers (beyond 3 std)
outliers, _ = gmm.sample(20, std_range=(3.0, float('inf')))
```

## Model Persistence

Save and load trained models:

```python
# Save model
gmm.save('my_model.pth')

# Load model
loaded_gmm = GaussianMixture.load('my_model.pth', device='cuda')

# Or use state dict (PyTorch-style)
state_dict = gmm.save_state_dict()
torch.save(state_dict, 'model_state.pth')

# Load state dict
new_gmm = GaussianMixture(n_components=3, n_features=2)
new_gmm.load_state_dict(torch.load('model_state.pth'))
```

## GPU Acceleration

```python
# Automatic device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gmm = GaussianMixture(
    n_components=3,
    n_features=2,
    device=device
)

# Data is automatically moved to the correct device
gmm.fit(X)  # X can be on CPU or GPU
```

## Attributes

After fitting, the model exposes:

- `gmm.weights_`: Mixing weights, shape $(K,)$
- `gmm.means_`: Component means, shape $(K, d)$
- `gmm.covariances_`: Component covariances (shape depends on type)
- `gmm.converged_`: Whether EM converged
- `gmm.n_iter_`: Number of EM iterations performed
- `gmm.lower_bound_`: Final average log-likelihood
- `gmm.fitted_`: Whether model has been fitted

## Complete API Reference

For full details on all parameters and methods, see the [API Reference](../api/gaussian-mixture.md).
