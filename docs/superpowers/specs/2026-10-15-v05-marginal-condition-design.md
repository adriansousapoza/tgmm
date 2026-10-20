# v0.5 item 1.3: `marginal` and `condition`

Date: 2026-10-15. Plan section 1.3 of the v0.5 roadmap. Branch `v05-1-3-marginal-condition`.

## What

Two methods on a fitted `GaussianMixture`, and one new result class:

- `gmm.marginal(dims)` -> a new fitted `GaussianMixture` over the selected coordinates.
- `gmm.condition(x_obs, obs_dims)` -> `ConditionalGaussianMixture`, the distribution of the remaining coordinates given the
  observed ones (one mixture per row of `x_obs`). Its `mean()` is Gaussian mixture regression (GMR).
- `StudentTMixture.marginal(dims)` (exact, trivial). `StudentTMixture.condition` raises `NotImplementedError`.

## Math

Component `k` has mean `mu_k` and (dense) covariance `S_k`. Split the coordinates into observed `o` and missing `m`.

Marginal over coordinates `dims`: component `k` is `N(mu_k[dims], S_k[dims, dims])`, weights unchanged.

Conditional on `x_o`: Gaussian conditioning component by component,

```
A_k      = S_k,mo S_k,oo^{-1}                     (d_m x d_o)
m_ik     = mu_k,m + A_k (x_o,i - mu_k,o)          conditional mean, (n, K, d_m)
C_k      = S_k,mm - A_k S_k,om                    conditional covariance, (K, d_m, d_m): independent of the sample
w_ik     ∝ pi_k N(x_o,i ; mu_k,o, S_k,oo)         conditional weights, softmax over k
```

so `p(y | x_o,i) = sum_k w_ik N(y ; m_ik, C_k)`. GMR: `E[y | x_o,i] = sum_k w_ik m_ik`. All computations are batched
Cholesky / `solve` over `K` (and `n` for the weights); no Python loop over samples.

## API and conventions

- `dims` / `obs_dims`: sequences of distinct integers in `[0, d)` (negative indices are not accepted); non-empty; for
  `condition` the observed set must also leave at least one coordinate missing. The **order is kept**: `marginal`'s output
  coordinate `j` is input coordinate `dims[j]`, and column `j` of `x_obs` corresponds to `obs_dims[j]`.
  `missing_dims` is the **sorted** complement of `obs_dims`; the conditional distribution is over those coordinates in
  increasing order.
- `x_obs` is `(n, len(obs_dims))`; a 1-D vector is treated as a single row.
- `ConditionalGaussianMixture` attributes: `weights_ (n, K)`, `means_ (n, K, d_m)`, `covariances_ (K, d_m, d_m)`,
  `obs_dims`, `missing_dims`, `n_components`, `n_features` (= `d_m`), `device`, `dtype`.
  Methods: `mean() -> (n, d_m)`; `log_prob(Y) -> (n,)` for `Y (n, d_m)`; `sample(n_draws) -> (n, n_draws, d_m)`;
  `as_gmm(i) -> GaussianMixture` over the missing coordinates (full covariance), for row `i`.
- Covariance types: every type (`full`, `diag`, `spherical`, `tied_*`, and all 14 GPCM codes) is supported through its
  assembled dense covariance. `marginal` keeps `diag`/`spherical`/`tied_*` (restricting a diagonal or spherical covariance
  keeps it diagonal/spherical with the same variances) and returns `covariance_type='full'` for the 8 GPCM codes without a
  classic name (the volume/shape/orientation decomposition does not survive restriction). The result of `condition` is
  always dense.
- Fitted bookkeeping of the new model: `fitted_ = converged_ = True`, `n_iter_ = 0`, `lower_bound_ = nan`; `dtype`, `device`,
  `reg_covar`, `missing` are copied. No priors are copied (the marginal is a fitted model, not a re-fit).

## Not supported (raises `NotImplementedError`)

- Gibbs-mode fits (`n_components=None`): the active-component bookkeeping has no meaning for a derived model.
- `noise=True` models: the uniform noise component has no Gaussian-mixture representation after conditioning (and
  `marginal` would silently drop it). Both methods raise rather than change the model's meaning.
- `StudentTMixture.condition`: the conditional of a multivariate t is a t with degrees of freedom `nu + d_o` and a scale
  inflated by `(nu + delta_o) / (nu + d_o)`, which depends on the sample, so it is not a member of this class. (`marginal`
  is exact: a t marginal has the same `nu` and the restricted scale matrix.)
- `VonMisesFisherMixture`: not closed under marginalisation or conditioning; no methods are added.

## Tests (`tests/test_v05_condition.py`)

- K = 1 matches the closed-form bivariate normal conditional; `mean()` equals `sum_k w_ik m_ik`.
- `marginal` log-likelihood equals `logsumexp_k log pi_k + log N` of scipy marginals; checked for every covariance type and
  GPCM code; covariance types kept/converted as specified.
- Consistency: conditioning the full model on dims `O` equals conditioning the *marginal* over `O ∪ M'` on `O` and keeping
  coordinates `M'` (the extra coordinates are integrated out), i.e. `condition` commutes with `marginal` of the unobserved
  coordinates.
- GMR on `y = sin(x) + noise` with K = 8: RMSE against the noiseless curve < 0.1.
- `log_prob` integrates to ~1 on a 1-D grid; `as_gmm(i).score_samples` equals `log_prob`; `sample` moments match the
  analytic mixture mean/variance.
- Validation errors (duplicates, out of range, empty, observed = everything), Gibbs/noise raise, Student-t marginal exact
  and `condition` raises, dtype/device honoured.
