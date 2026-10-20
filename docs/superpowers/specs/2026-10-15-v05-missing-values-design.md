# v0.5 item 1.2: missing values (`missing='em'`)

Date: 2026-10-15. Implements section 1.2 of the v0.5 plan. Stacked on item 1.1 (noise component).

## What

`GaussianMixture` (all six classic covariance types and the 14 GPCM codes) and `StudentTMixture` accept
`missing: str = 'error'` (default) or `'em'`. With `'em'`, `NaN` entries of `X` are treated as missing at random
(MAR) and the likelihood is maximised over the *observed* entries by EM (Ghahramani & Jordan, 1994; Hunt &
Jorgensen, 2003). Everything that consumes data accepts incomplete rows: `fit`, `predict`, `predict_proba`,
`score_samples`, `score`, `bic`/`aic`/`icl`. New method `impute(X)`.

- `missing='error'` (default): a `NaN` in `X` raises `ValueError` (message points at `missing='em'`). Before this item a
  `NaN` silently propagated into `NaN` parameters; no fit on finite data changes (tested bit-identical).
- `missing='em'` on data with **no** `NaN` follows exactly the same code path as `'error'`: results are identical.
- Rows whose entries are *all* missing and columns that are entirely `NaN` raise `ValueError` (a row with no observed
  entry carries no information about the cluster; drop it).

## Maths (Gaussian)

Group rows by missingness pattern (`torch.unique(isnan(X), dim=0)`, `P` patterns; `UserWarning` when `P > 1000`).
For a pattern with observed index set `o` (size `p`) and missing index set `m` (size `q`) and component `k`:

- **Marginal** (E-step): `log N(x_o; mu_o, Sigma_oo)`, so `log p(x_i | k) = -1/2 [p log 2 pi + log|Sigma_oo| + delta_o]`
  with `delta_o = (x_o - mu_o)^T Sigma_oo^{-1} (x_o - mu_o)`.
- **Conditional** (M-step statistics): with `A_k = Sigma_mo Sigma_oo^{-1}`,
  `xhat_m = mu_m + A_k (x_o - mu_o)`, `C_k = Sigma_mm - A_k Sigma_om` (independent of the row, per pattern).
- Completed row `xtilde_ik` = `x_i` with the missing block replaced by `xhat_m` (observed entries unchanged).
- Sufficient statistics (`SufficientStats`, so `batch_size`, SAEM etc. work): `nk = sum_i r_ik`,
  `s1_k = sum_i r_ik xtilde_ik`, and `s2_k` = centred second moment about `ybar_k = s1_k / nk`:
  `s2_k = sum_i r_ik (xtilde_ik - ybar_k)(xtilde_ik - ybar_k)^T + sum_i r_ik C_{k, pattern(i)}` where `C` is placed in the
  `m x m` block. `s2` is exactly the quantity `E[sum r (x - ybar)(x - ybar)^T | x_o, theta_old]`, so the existing M-step
  formulas (MLE, MAP, NIW, GPCM) apply unchanged: every covariance type's M-step reads `(nk, s1, s2)` only.
  The statistics use the *current* parameters, which are the ones the E-step just used.
- `diag` / `spherical` / `tied_diag` / `tied_spherical`: `Sigma_mo = 0`, so `xhat_m = mu_m`, `C = diag(Sigma_mm)`; only
  per-feature variances are touched (no `d x d` solves). For `spherical` types `s2` is the trace-reduced second moment the
  M-step already expects, so the extra term is `q * sigma_k^2`.
- Dense covariance types (`full`, `tied_full`, the 8 non-classic GPCM codes, stored assembled as `(K, d, d)`) use batched
  Cholesky solves per pattern, vectorised over rows and components. A Python loop runs over patterns only.

`impute(X)` returns `X` with each `NaN` replaced by `sum_k tau_ik xhat_ik` (posterior-weighted conditional mean); observed
entries are untouched.

## Maths (Student-t)

Scale-mixture form: `w ~ Gamma(nu/2, nu/2)`, `x | w, k ~ N(mu_k, Sigma_k / w)`. The observed block is
`t_p(mu_o, Sigma_oo, nu)`, so the marginal log-density is the usual t log-density with `d -> p` and `delta -> delta_o`.
Given `x_o` and `k`:

- `w | x_o ~ Gamma((nu + p)/2, (nu + delta_o)/2)`: `u = E[w | x_o] = (nu + p) / (nu + delta_o)`,
  `E[ln w | x_o] = psi((nu + p)/2) - ln((nu + delta_o)/2)`.
- `x_m | x_o, w ~ N(xhat_m, Sigma_{m|o} / w)` with the *same* `xhat_m` and `Sigma_{m|o} = C_k` as the Gaussian case (the
  conditional mean does not depend on `w`). Hence `E[w x_m | x_o] = u xhat_m` and
  `E[w (x_m - c)(x_m - c)^T | x_o] = u (xhat_m - c)(xhat_m - c)^T + C_k` (because `E[w * 1/w] = 1`).
  (Lin, Lee & Ho 2006; McLachlan & Krishnan 2008, ch. 5.)
- ECM statistics: `m_k = sum_i tau_ik u_ik`, `s1_k = sum_i tau_ik u_ik xtilde_ik`, `ybar_k = s1_k / m_k`,
  `s2_k = sum_i tau_ik u_ik (xtilde_ik - ybar_k)(xtilde_ik - ybar_k)^T + sum_i tau_ik C_{k, pattern(i)}` (**no `u` factor on
  `C`**). `mu_k = ybar_k`, `Sigma_k = s2_k / n_k` as before.
- Degrees of freedom: the exact conditional maximiser of the expected complete-data log-likelihood in `nu` solves
  `1 - psi(nu/2) + ln(nu/2) + c_k = 0` with `c_k = (1/n_k) sum_i tau_ik [psi((nu_old + p_i)/2) - ln((nu_old + p_i)/2) + ln u_ik - u_ik]`
  (the textbook Peel & McLachlan form, evaluated at the previous `nu_old`, with each row's own `p_i`). When data have missing
  entries this form is used (the complete-data code path, which evaluates the digamma pair at the candidate `nu`, is kept
  untouched for complete data). Both share the same fixed points.
- `scale_update='ecme'` with `missing='em'` raises `NotImplementedError`: its `m_k = n_k` fixed-point identity relies on
  `sum_i r u delta = d n_k`, which no longer holds with a varying number of observed dimensions.
- Monotonicity: each CM step maximises the expected complete-data log-likelihood given the E-step expectations, so the
  observed-data log-likelihood is non-decreasing (ECM). Tested for `dof` fixed and `per_component`.

## Initialisation

`_allocate_parameters` (k-means etc.) runs on the **column-mean-imputed** data (`nanmean` per column), never on `NaN`s.
SEM's empty-component re-seeding uses the same imputed copy.

## Supported / unsupported

| combination | behaviour |
| --- | --- |
| `algorithm in {'em','cem','sem','saem'}`, `temperature`, `batch_size`, `n_init`, `warm_start` | supported |
| `noise=True` | `NotImplementedError` at construction (the noise density needs the volume of each row's *observed* subspace; out of scope) |
| `step_size` with `em`/`cem` (online EM), `partial_fit` | `NotImplementedError` |
| Gibbs mode (`n_components=None`) | `NotImplementedError` |
| `fit(X, labels=...)` (supervised / semi-supervised) | `NotImplementedError` |
| `StudentTMixture(scale_update='ecme')` | `NotImplementedError` |
| `VonMisesFisherMixture` | `missing='em'` raises `ValueError` (not a Gaussian-type family); `missing='error'` still validates `NaN` |

Persistence: `missing` is stored by `save` / `save_state_dict` / `to_dict` and restored (older files default to `'error'`).

## Per-feature weights

The plan mentions a per-feature weight of 0 for independent-feature families (Phase 2 and diagonal Gaussians). That hook
belongs to item 1.4 (`sample_weight` of shape `(n, d)`); it is deliberately not built here. Diagonal Gaussians already get
the exact marginalisation described above, which is equivalent.

## Tests (`tests/test_v05_missing.py`)

K=1 closed-form marginal/conditional checks against `scipy.stats` and a per-row NumPy reference EM (Gaussian and t);
20% MCAR recovery (means, ARI, monotone log-likelihood) for each covariance type; no-`NaN` identity; `impute`;
`algorithm` parametrisation; chunked == unchunked; every unsupported combination; save/load.
