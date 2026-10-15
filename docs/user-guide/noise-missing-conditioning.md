# Noise, Missing Values and Conditioning

Five features that every mixture family inherits from `BaseMixture` (or, for the Gaussian-specific ones, from
`GaussianMixture`) and that change *what the fit is allowed to do* rather than which density it fits:

| Feature | What it gives you | Section |
| --- | --- | --- |
| `noise=True` | a parameter-free uniform component that absorbs stray points | [Uniform noise component](#uniform-noise-component) |
| `missing='em'` | maximum likelihood from rows with `NaN` entries, and `impute` | [Missing values](#missing-values) |
| `marginal` / `condition` | marginal and conditional distributions, Gaussian mixture regression | [Marginals and conditionals](#marginals-and-conditionals) |
| `frozen_components`, `sample_weight` | pin known components; weighted observations | [Frozen components and sample weights](#frozen-components-and-sample-weights) |
| `Component` protocol, `GenericMixture` | bring your own component family | [Custom components](#custom-components-the-component-protocol-and-genericmixture) |

Notation as elsewhere: $n$ samples, $d$ features, $K$ components, responsibilities $r_{ik}$, $n_k=\sum_i r_{ik}$. The
[tutorial](../notebooks/19_noise_missing_conditioning.ipynb) runs every example below.

## Uniform noise component

Real data often contain points that no cluster explains. A Gaussian component forced to cover them widens and drifts.
Banfield & Raftery (1993) add one more component that is **uniform over a region of volume $V$**:

$$
p(x)=\sum_{k=1}^{K}\pi_k f_k(x)+\frac{\pi_0}{V},\qquad \sum_{k=1}^{K}\pi_k+\pi_0=1 .
$$

```python
from tgmm import GaussianMixture

gmm = GaussianMixture(n_components=3, noise=True).fit(X)
labels = gmm.predict(X)        # -1 where the noise component is the most probable
p_noise = gmm.noise_proba(X)   # posterior probability of the noise component
gmm.noise_weight_              # pi_0
```

**E-step.** A column $\log\pi_0-\log V$ is appended to the $(n,K)$ component log-densities, so the responsibilities have
shape $(n,K+1)$ and the last column is $\tau_{i0}$. $\log V$ is a constant, so the noise log-density is the same at every $x$ (also
outside the data's bounding box: new points beyond the box are not given zero noise density).

**M-step.** $\pi_0=\sum_i\tau_{i0}/n$. The $K$ components are updated from the first $K$ columns by the family's usual
M-step, so `weights_` $=n_k/n$ sums to $1-\pi_0$ rather than to 1.

!!! note "`weights_` and `noise_weight_`"
    With `noise=True`, `weights_` keeps shape $(K,)$ and holds the **component** weights, which sum to
    `1 - noise_weight_`. This keeps `sample`, `plot_gmm`, `plot_vmf` and `save`/`load` working unchanged.

| Argument | Default | Meaning |
| --- | --- | --- |
| `noise` | `False` | switch the component on; with `False` every fit is bit-identical to before |
| `noise_volume` | `'bbox'` | `'bbox'`: volume of the axis-aligned bounding box of the fit data, $\prod_j(\max_j-\min_j)$ (each extent floored at $10^{-12}$); or a positive float, a user-supplied hypervolume |
| `noise_prior` | `0.05` | initial $\pi_0\in(0,1)$ |
| `noise_init` | `None` | per-sample initial flag of length $n$. A `bool` tensor is a mask (`True` = noise); any other dtype is read as cluster labels with `-1` = noise, so `HDBSCAN(...).labels_` can be passed as is. It overrides `noise_prior`, and the initial means and covariances come from the unflagged rows only |

**Family hook.** The noise log-density is `-log V` for `GaussianMixture` (all covariance types and GPCM codes) and `StudentTMixture`.
For `VonMisesFisherMixture` it is $-\log A_d$ with $A_d=2\pi^{d/2}/\Gamma(d/2)$ the surface area of the unit sphere $S^{d-1}$ (the uniform density
on the sphere), independent of `noise_volume`. Families without bounded continuous support raise `ValueError`.

**Behaviour.** `predict_proba(X)` is $(n,K+1)$; `score_samples`, `bic`, `aic`, `icl` and `select_model` include the noise term (one extra
parameter). `sample` draws a noise point with probability $\pi_0$, uniform in the stored bounding box `noise_bounds_` (uniform on the sphere for vMF), with
component index `-1`. Works with `em`, `cem`, `sem`, `saem`, `temperature`, `batch_size`, `n_init`, `warm_start` and persistence.

!!! warning "Unsupported combinations (they raise)"
    Gibbs mode (`n_components=None`); `fit(X, labels=...)` (a point cannot be labelled "known noise": `-1` still means unlabelled);
    online EM (`step_size`) and `partial_fit`; `weight_concentration_prior`; constrained `GaussianMixture.sample` variants
    (`component=`, `confidence=`, ...); `missing='em'`.

## Missing values

With `missing='em'`, `NaN` entries of `X` are **missing at random** and EM maximises the likelihood of the *observed* entries
(Ghahramani & Jordan, 1994; Hunt & Jorgensen, 2003). It is available for `GaussianMixture` (all six classic covariance types and all 14 GPCM codes) and
`StudentTMixture`. `fit`, `predict`, `predict_proba`, `score_samples`, `score` and `bic`/`aic`/`icl` then accept incomplete rows.

```python
gmm = GaussianMixture(n_components=2, missing='em').fit(X_with_nans)
X_filled = gmm.impute(X_with_nans)
```

For a row with observed index set $o$ and missing index set $m$, and component $k$:

- **E-step (marginal likelihood):** $\log p(x_i\mid k)=-\tfrac12\big[p\log2\pi+\log|\Sigma_{k,oo}|+\delta_o\big]$ with
  $\delta_o=(x_o-\mu_{k,o})^\top\Sigma_{k,oo}^{-1}(x_o-\mu_{k,o})$ and $p=|o|$.
- **Conditional mean and covariance of the hole**, with $A_k=\Sigma_{k,mo}\Sigma_{k,oo}^{-1}$:
$$
\hat x_{m}=\mu_{k,m}+A_k(x_o-\mu_{k,o}),\qquad C_k=\Sigma_{k,mm}-A_k\Sigma_{k,om}.
$$
- **Completed statistics.** With $\tilde x_{ik}$ the row with the hole replaced by $\hat x_m$: $n_k=\sum_ir_{ik}$, $s_{1,k}=\sum_ir_{ik}\tilde x_{ik}$, $\bar y_k=s_{1,k}/n_k$ and
$$
s_{2,k}=\sum_i r_{ik}(\tilde x_{ik}-\bar y_k)(\tilde x_{ik}-\bar y_k)^\top+\sum_i r_{ik}\,C_{k,\text{pattern}(i)}\quad(C\text{ in the }m\times m\text{ block}),
$$
  which is exactly $\mathbb E[\sum_ir_{ik}(x_i-\bar y_k)(x_i-\bar y_k)^\top\mid x_o,\theta^{\text{old}}]$. Every M-step formula of the
  [Gaussian guide](gaussian-mixture.md) (MLE, MAP, NIW, GPCM) reads only $(n_k,s_1,s_2)$, so it applies unchanged.

Rows are grouped by **missingness pattern** (`torch.unique(torch.isnan(X), dim=0)`), so the cost is one batched solve per pattern per component, not per row
(`UserWarning` when more than 1000 distinct patterns). For `diag`/`spherical` types $\Sigma_{mo}=0$, so $\hat x_m=\mu_m$ and $C=\operatorname{diag}\Sigma_{mm}$:
no solves at all. `impute(X)` replaces each `NaN` by $\sum_k\tau_{ik}\hat x_{ik}$ and leaves observed entries untouched.

**Student-t.** With the scale-mixture form ($w\sim\mathrm{Gamma}(\nu/2,\nu/2)$, $x\mid w\sim\mathcal N(\mu,\Sigma/w)$) the observed block is a $t_p$ and
$u_{ik}=(\nu_k+p)/(\nu_k+\delta_o)$ uses the number $p$ of observed entries. The conditional mean does not depend on $w$, so
$\mathbb E[w\,x_m\mid x_o]=u\,\hat x_m$ and $\mathbb E[w(x_m-c)(x_m-c)^\top\mid x_o]=u(\hat x_m-c)(\hat x_m-c)^\top+C_k$ (the conditional covariance carries **no** factor $u$). Degrees
of freedom use the textbook form per row with that row's own $p$. `scale_update='ecme'` is not available with `missing='em'`.

**Initialisation** runs on the column-mean-imputed data. **Defaults:** with `missing='error'` a `NaN` raises a `ValueError` pointing at `missing='em'` (earlier
versions silently propagated `NaN` into the parameters); `missing='em'` on data without `NaN` gives results identical to `'error'`.

!!! warning "What raises"
    Rows with no observed entry and entirely-`NaN` columns (`ValueError`: they carry no information); `noise=True`, Gibbs mode,
    `labels=`, online EM / `partial_fit` (`NotImplementedError`); `VonMisesFisherMixture` (`ValueError`).

## Marginals and conditionals

A fitted `GaussianMixture` is a joint density, so it can be restricted and conditioned.

**`gmm.marginal(dims)`** returns a new fitted `GaussianMixture` over the chosen coordinates: component $k$ becomes $\mathcal N(\mu_k[\text{dims}],\Sigma_k[\text{dims},\text{dims}])$ with the same
weights. `diag`, `spherical` and `tied_*` keep their type; `full` and the GPCM codes without a classic name return `full` (the volume/shape/orientation decomposition does not
survive restriction). `StudentTMixture.marginal` is exact too (same $\nu$, restricted scale matrix).

**`gmm.condition(x_obs, obs_dims)`** returns a `ConditionalGaussianMixture`: for every row $x_{o,i}$ the distribution of the remaining coordinates,

$$
p(y\mid x_{o,i})=\sum_{k=1}^{K}w_{ik}\,\mathcal N\!\big(y;\,m_{ik},\,C_k\big),\qquad
w_{ik}\propto\pi_k\,\mathcal N(x_{o,i};\mu_{k,o},\Sigma_{k,oo}),
$$

$$
m_{ik}=\mu_{k,m}+\Sigma_{k,mo}\Sigma_{k,oo}^{-1}(x_{o,i}-\mu_{k,o}),\qquad C_k=\Sigma_{k,mm}-\Sigma_{k,mo}\Sigma_{k,oo}^{-1}\Sigma_{k,om}.
$$

The weights depend on the observed values; the conditional covariances do not. All steps are batched Cholesky solves, with no loop over samples.

```python
cond = gmm.condition(x_obs, obs_dims=[0])
cond.mean()        # (n, d_m): Gaussian mixture regression, E[y | x] = sum_k w_ik m_ik
cond.log_prob(Y)   # (n,) conditional log-density (a single-row conditional broadcasts against many Y)
cond.sample(50)    # (n, 50, d_m)
cond.as_gmm(i)     # conditional of row i as an ordinary GaussianMixture over the missing coordinates
```

`obs_dims` keeps the order you give (column $j$ of `x_obs` is coordinate `obs_dims[j]`); the conditional is over the **sorted complement**. Attributes: `weights_ (n, K)`, `means_ (n, K, d_m)`,
`covariances_ (K, d_m, d_m)`, `obs_dims`, `missing_dims`. The mean is a regression curve; the *distribution* can be multimodal, which a single regression mean would hide.

!!! warning "Not supported"
    Gibbs-mode and `noise=True` models (a noise component has no Gaussian-mixture form after conditioning); `StudentTMixture.condition` (the conditional t has degrees of
    freedom $\nu+d_o$ and a sample-dependent scale, so it is not a member of this class); `VonMisesFisherMixture`.

## Frozen components and sample weights

**`frozen_components=[k, ...]`** keeps component $k$'s means and covariances (or `dof_`, `kappa_`) at their **initial** values for the whole fit. Mixing weights, and the
noise weight, still update, and $\sum_k\pi_k+\pi_0=1$ is preserved. Initial values come from `init_means`/`init_covariances` or `warm_start`. Priors never shrink a frozen
component and SEM never re-seeds it. Not supported (`ValueError`): tied covariance types and GPCM codes without a classic name, `dof='shared'`, `concentration='shared'`, Gibbs mode.
`PositiveUnlabeledMixture(anchor='fixed')` is built on it.

**`fit(X, sample_weight=w)`**

- *Row weights*, shape $(n,)$: responsibilities are scaled by $w_i$ before the sufficient statistics, so $n_k$, $s_1$, $s_2$ (and the Student-t and vMF analogues) are weighted. The effective sample count is
  $n_{\text{eff}}=\sum_iw_i$ (mixing weights $\pi_k=n_k/n_{\text{eff}}$; MAP priors use $n_{\text{eff}}$), and `lower_bound_` is $\sum_iw_i\log p(x_i)/\sum_iw_i$. Hence integer weights equal
  duplicating rows, a weight of 0 drops the row, a constant rescaling changes nothing, and all-ones is bit-identical to no weights. Initialisation ignores weights (except to exclude zero-weight rows),
  and `bic`/`aic`/`icl`/`score` take only `X`.
- *Per-feature weights*, shape $(n,d)$: for models that factorise over features (`GaussianMixture` `'diag'`/`'spherical'`, `StudentTMixture` `'diag'`), maximum likelihood only. A weight $w_{ij}=0$ hides feature
  $j$ of row $i$ (exact marginalisation). For `'diag'`:
$$
\log p(x_i\mid k)=-\tfrac12\sum_jw_{ij}\Big[\log2\pi+\log s_{kj}+\frac{(x_{ij}-m_{kj})^2}{s_{kj}}\Big],\quad
m_{kj}=\frac{\sum_ir_{ik}w_{ij}x_{ij}}{N_{kj}},\ \ s_{kj}=\frac{\sum_ir_{ik}w_{ij}(x_{ij}-m_{kj})^2}{N_{kj}},
$$
  with $N_{kj}=\sum_ir_{ik}w_{ij}$. Other models raise `ValueError`.

Row weights work with `em`/`cem`/`sem`/`saem`, `batch_size`, `n_init`, `warm_start`, `noise=True` and `missing='em'` (each on its own: the two cannot be combined); they raise `NotImplementedError` in Gibbs mode, with `labels=` and with online EM.

!!! note "No `inertia`"
    pomegranate's `inertia` is deliberately not added: `step_size` (online EM / SAEM) is the same control on the statistics.

## Custom components: the `Component` protocol and `GenericMixture`

`tgmm.mixtures.protocol.Component` is a runtime-checkable `typing.Protocol` that describes a mixture component as EM sees it (the API shape is borrowed from pomegranate 1.x, MIT; tgmm does not import it):

| Method | Role |
| --- | --- |
| `log_probability(X)` | $\log f_k(x_i)$, shape `(n,)` |
| `summarize(X, sample_weight)` | **accumulate** weighted statistics of one chunk (`sample_weight` = responsibility $\times$ row weight) |
| `from_summaries()` | update the parameters from everything summarised and reset the accumulators |

Optional members: `sample(n)`, `n_parameters()` (fallback: the number of scalar entries in the object's public tensors), `frozen`, `reseed(x)`.

```python
from tgmm import GenericMixture

gm = GenericMixture(components=[Exponential(0.5), Normal1D(5.0, 1.0)]).fit(X)
```

**M-step.** For every data chunk and every non-frozen component, `summarize(X_chunk, resp_chunk[:, k])`; then `from_summaries()` once. Because `summarize` accumulates, `batch_size` works and
chunked equals unchunked. Mixing weights are $\pi_k=n_k/n$ as in every family.

**Autograd components.** A `torch.nn.Module` with learnable parameters and a `log_prob(X)` method (and no `summarize`) is wrapped automatically. Its M-step maximises
$Q_k(\theta_k)=\sum_iw_{ik}\log f_k(x_i\mid\theta_k)/\sum_iw_{ik}$ with a few L-BFGS steps (`BaseMixture._gem_step`) and **keeps the update only if $Q_k$ increases**, so the log-likelihood stays monotone (a generalised EM step).

**Decisions.** The components you pass *are* the starting point (no k-means); `n_init` must be 1; frozen parameters still count in BIC; `em`, `cem`, `sem` (with `sem_estimate='best'`) and `temperature` work, while
`saem`, online EM and `partial_fit` raise `NotImplementedError` (opaque `summarize` state cannot be blended); `noise=True` needs a float `noise_volume`; per-feature weights and `missing='em'` raise `ValueError`;
`save`/`load` pickle the components, so their classes must be importable.

## Related

- [Gaussian Mixture Model](gaussian-mixture.md): the E- and M-steps these features modify; [Other Mixture Families](other-mixture-families.md); [Semi-Supervised and PU Fitting](semi-supervised-pu.md).
- API: [`ConditionalGaussianMixture`](../api/conditional-gaussian-mixture.md), [`GenericMixture`](../api/generic-mixture.md).

## References

Banfield, J. D. & Raftery, A. E. (1993). Model-based Gaussian and non-Gaussian clustering. *Biometrics*, 49(3). Ghahramani, Z. & Jordan, M. I. (1994). Supervised learning from incomplete data via an EM approach. *NeurIPS 6*.
Hunt, L. & Jorgensen, M. (2003). Mixture model clustering for mixed data with missing information. *CSDA*, 41. Lin, T. I., Lee, J. C. & Ho, H. J. (2006). On fast supervised learning for normal mixture models with missing information. *Pattern Recognition*, 39. McLachlan, G. & Krishnan, T. (2008). *The EM Algorithm and Extensions*, ch. 5.
Sung, H. G. (2004). *Gaussian Mixture Regression and Classification*. PhD thesis, Rice University. Design notes: `docs/superpowers/specs/2026-10-15-v05-*.md`.
