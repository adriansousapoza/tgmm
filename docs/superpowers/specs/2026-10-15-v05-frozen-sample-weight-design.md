# v0.5 item 1.4: `frozen_components` and `sample_weight`

Date: 2026-10-15. Branch `v05-1-4-frozen-sample-weight`. Plan section 1.4.

## 1. `frozen_components`

`frozen_components: Sequence[int] = ()` on `BaseMixture` and therefore on `GaussianMixture`, `StudentTMixture` and
`VonMisesFisherMixture`. A frozen component keeps its per-component parameters (`means_`, `covariances_`, and `dof_` /
`kappa_` for the other families) at their *initial* values for the whole fit; its **mixing weight still updates**, and so
does the noise weight of item 1.1. The weights keep summing to 1 (`weights_.sum() + noise_weight_ == 1`).

* Initial values come from `init_means` / `init_covariances` / ... (tensors), or from the previous fit with
  `warm_start=True`. With string initialisers a frozen component keeps whatever k-means (etc.) produced.
* Implementation: after every call of `_m_step_from_stats` the frozen components' per-component tensors are restored
  (`BaseMixture._m_step_frozen`). Everything that depends on the M-step therefore holds for free: priors (MAP / NIW) never
  shrink a frozen component, `batch_size`, `cem`/`sem`/`saem`, `temperature`, online EM, noise, missing values.
* SEM's empty-component re-seeding never touches a frozen component.
* Not supported (`ValueError`, tested): tied covariance types and the 8 GPCM codes without a classic name (some factor is
  shared across components, so one component cannot be held fixed), `StudentTMixture(dof='shared')` and
  `VonMisesFisherMixture(concentration='shared')` (a shared scalar), Gibbs mode (`n_components=None`), and invalid indices
  (non-integers, out of range, duplicates).
* Persistence: stored by `save`/`load`, `to_dict`/`save_state_dict`/`load_state_dict`.
* `PositiveUnlabeledMixture(anchor='fixed')` is now implemented with it: the positive block is frozen at its labeled-only
  fit. This **fixes an existing bug**: the old hook copied the block-local snapshot, including its local weight `1.0`,
  into `weights_` after every M-step, so `weights_` did not sum to 1 (e.g. `[0.60, 1.00]`) and `class_prior_` was biased.
  `frozen_components` may not be passed through `**family_kwargs` of `PositiveUnlabeledMixture`.
* `inertia` (pomegranate) is deliberately not added: `step_size` (online EM / SAEM, v0.4 Phase 2.3) is the same control
  on the statistics.

## 2. `fit(X, sample_weight=None)`

`sample_weight` is the last keyword of `fit` (existing positional order unchanged). Validated: finite, non-negative,
shape `(n,)` or `(n, d)`, positive total.

### 2.1 Row weights, shape `(n,)`

Row `i` counts `w_i` times: its responsibilities are scaled by `w_i` before the sufficient statistics, so `n_k`, `s1`, `s2`
(and the Student-t `u`-weighted sums, the vMF resultant vector) are weighted. The sample count of the M-step is
`n_eff = sum_i w_i` (mixing weights `pi_k = n_k / n_eff`; MAP priors use `n_eff` too), the noise weight is the weighted mean
noise responsibility, and the convergence criterion / `lower_bound_` is the weighted mean log-likelihood
`sum_i w_i log p(x_i) / sum_i w_i`. Consequences (all tested): integer weights equal duplicating rows; weight 0 equals
dropping the row; a constant weight `c` changes nothing; all-ones is bit-identical to `sample_weight=None`.

* Initialisation (k-means etc.) ignores weights except that rows with weight 0 are excluded.
* `bic` / `aic` / `icl` / `score` take only `X` and are unweighted.
* SEM samples assignments from the (unweighted) responsibilities, statistics are weighted; the empty-component test uses
  unweighted counts.
* `bounding box` of the noise component is computed from all rows, including weight-0 rows.
* Works with `missing='em'`, `noise=True`, `batch_size` (the weights are chunked with `X`), `n_init`, `warm_start` and
  `algorithm in {'em','cem','sem','saem'}`.
* `NotImplementedError`: Gibbs mode, `labels=...`, online EM (`step_size` with `em`/`cem`).

### 2.2 Per-feature weights, shape `(n, d)`

Accepted where the model factorises per feature: `GaussianMixture(covariance_type='diag'|'spherical')` and
`StudentTMixture(covariance_type='diag')`, **maximum likelihood only** (no mean / covariance / NIW priors). Entry `x_ij`
carries weight `w_ij`; `w_ij = 0` makes feature `j` invisible for row `i` (the missing-value trick, with exact
marginalisation because the features are independent given the component). Other model types raise `ValueError`.

Gaussian, `diag`:

    log p(x_i | k) = -1/2 sum_j w_ij [ log 2 pi + log s_kj + (x_ij - m_kj)^2 / s_kj ]
    m_kj = sum_i r_ik w_ij x_ij / N_kj,   s_kj = sum_i r_ik w_ij (x_ij - m_kj)^2 / N_kj,   N_kj = sum_i r_ik w_ij

`spherical` shares one variance, `s_k = sum_i r_ik sum_j w_ij (x_ij - m_kj)^2 / sum_j N_kj`. The mixing weights use the
unweighted responsibilities, `pi_k = sum_i r_ik / n`. (Row weights times feature weights: pass the elementwise product.)

Student-t, `diag`: the effective dimension of row `i` is `p_i = sum_j w_ij`; the density is the t density with
`d -> p_i`, `log|Sigma| -> sum_j w_ij log s_kj` and `delta_ik = sum_j w_ij (x_ij - m_kj)^2 / s_kj`;
`u_ik = (nu_k + p_i) / (nu_k + delta_ik)`; `m_kj = sum_i tau_ik u_ik w_ij x_ij / sum_i tau_ik u_ik w_ij`;
`s_kj = sum_i tau_ik u_ik w_ij (x_ij - m_kj)^2 / sum_i tau_ik w_ij` (`'ecm'`) or divided by `sum_i tau_ik u_ik w_ij`
(`'ecme'`); `nu_k` uses the exact per-row digamma form of item 1.2 with `d -> p_i`.

Errors / `NotImplementedError`: priors, `noise=True`, `missing='em'`, Gibbs mode, `labels=`, online EM. The weights enter
both the E-step and the M-step statistics (`SufficientStats.aux['nk_feat']`).

## 3. Tests

`tests/test_v05_frozen.py`: duplication / zero / constant-weight equivalences for all six classic Gaussian covariance
types, Student-t and vMF; priors, noise and missing values with weights; algorithm parametrisation; `batch_size`
equality; NumPy reference EMs for the per-feature cases; frozen components bit-identical to their initial values (all
families, all algorithms, priors, SEM re-seeding, warm start, noise); unsupported combinations; persistence; the PU
`anchor='fixed'` normalisation fix.
