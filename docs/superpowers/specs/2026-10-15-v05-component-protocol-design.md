# v0.5 item 1.5: component protocol and `GenericMixture`

Date: 2026-10-15. Implements item 1.5 of the v0.5 roadmap **and** the optional v0.4 section 4.4
("`GenericMixture`"), which was never built; the roadmap's 1.5 assumes it. Both live in one class.

## What

`tgmm.mixtures.protocol.Component` is a `typing.Protocol` (runtime checkable) describing a mixture
component as the EM algorithm sees it, and `tgmm.GenericMixture(components=[...])` is a
`BaseMixture` over `K` such objects. The API is borrowed from pomegranate 1.x (MIT), whose
distributions are `log_probability` / `summarize` / `from_summaries` objects.

```python
class Component(Protocol):
    def log_probability(self, X) -> Tensor: ...        # (n,)   log f_k(x_i)
    def summarize(self, X, sample_weight) -> None: ... # accumulate weighted statistics of one chunk
    def from_summaries(self) -> None: ...              # update parameters from everything summarised, then reset
    # optional: sample(n) -> (n, d);  n_parameters() -> int;  frozen: bool
```

Two kinds of component are accepted by `GenericMixture`:

1. **Protocol components** (anything with the three methods above). The E-step stacks
   `log_probability` into an `(n, K)` matrix. The M-step calls, for every row chunk and every
   non-frozen `k`, `summarize(X_chunk, resp_chunk[:, k])` (weights = responsibility times the row
   weight of item 1.4), then `from_summaries()` once per M-step. `summarize` must therefore
   *accumulate* (it is called once per chunk) and `from_summaries` must update the parameters and
   clear the accumulators. This is exactly the shape that makes `batch_size` work for free.
2. **Autograd components** (v0.4 section 4.4): a `torch.nn.Module` with learnable parameters and a
   `log_prob(X) -> (n,)` method, without `summarize`. They are wrapped in the internal adapter
   `_AutogradComponent`, which implements the protocol: `summarize` stores references to the
   chunk and its weights, and `from_summaries` maximises the expected complete-data
   log-likelihood

   $$Q_k(\theta_k)=\sum_i w_{ik}\,\log f_k(x_i\mid\theta_k)\Big/\sum_i w_{ik}$$

   by `BaseMixture._gem_step` (L-BFGS, up to 20 steps), **accepting the update only if $Q_k$
   increases**. The M-step is thus a *generalised* EM step and the log-likelihood stays monotone.
   A Laplace mixture, a Riemannian normal with a fixed metric, or a Student-t with learnable
   location/scale are one `nn.Module` away.

An object that has both `summarize` and `log_prob` is treated as a protocol component.

## Model and updates

$p(x)=\sum_{k=1}^{K}\pi_k f_k(x)$. Mixing weights: $\pi_k=n_k/n$ with $n_k=\sum_i r_{ik}$ (row weights
and the noise column of item 1.1 behave as in every other family, so `weights_` sums to
`1 - noise_weight_`). Component parameters: whatever the component's `from_summaries` does.

## Decisions

- **Initial state.** The user-supplied components *are* the starting point: there is no k-means
  initialisation (the library cannot know how to build a component from data). The constructor
  stores deep copies; every `fit` starts from fresh deep copies of those initial states
  (`warm_start=True` continues from the fitted `components_`). The fitted components are
  `GenericMixture.components_` (for autograd components, `components_[k].module`).
- **`n_init`.** Only `n_init=1` is accepted (`ValueError` otherwise): restarts would all begin from
  the same deep copy. Build several models from differently initialised components instead.
- **`init_weights`**: `'uniform'` (default) or a tensor. No `init_means`.
- **Frozen components.** Component `k` is skipped (no `summarize`, no `from_summaries`) when `k`
  is in `frozen_components` (item 1.4) **or** its `frozen` attribute is truthy. Its mixing weight
  still updates. Frozen parameters are still counted by `_n_parameters` (as in the other families).
- **`_n_parameters`** $=\sum_k$ `n_parameters()` $+\,(K-1)$ (+1 with a noise component). If a
  component has no `n_parameters()`, the fallback counts the scalar entries of the tensors held in
  its public attributes (names not starting with `_`; for an autograd component, its
  `requires_grad` parameters). Implement `n_parameters()` when in doubt.
- **Algorithms.** `em`, `cem`, `sem` (with `sem_estimate='best'`; `'mean'` would average
  arbitrary component state and raises), `temperature` (DAEM), `batch_size`. `saem` and online EM
  (`step_size`, `partial_fit`) need *additive* statistics that can be blended; a
  `summarize`/`from_summaries` object keeps opaque state, so they raise `NotImplementedError`.
  During SEM, a component whose sampled count falls below 1 is re-seeded only if it defines an
  optional `reseed(x)` method (called with the worst-fit point); otherwise it is left unchanged,
  and it must tolerate a (near-)zero total weight in `from_summaries`.
- **`sample_weight`.** Row weights multiply the responsibilities before they reach `summarize`
  (item 1.4); per-feature weights raise `ValueError` (a generic component has no per-feature
  factorisation).
- **Noise (1.1).** Needs a finite volume: `noise=True` requires `noise_volume=<float>`
  (`NotImplementedError` for `'bbox'`: a generic density has no bounded support in general);
  noise points are sampled uniformly in the bounding box of the fit data.
- **Missing values (1.2).** `missing='em'` is not available (`ValueError`); `NaN` input raises.
- **Labels.** Semi-supervised EM and the fully labelled one-component-per-class fit work through
  `BaseMixture`; `components_per_class > 1` with fully labelled data would need to refit blocks of a
  different size and raises `NotImplementedError`.
- **Sampling.** `sample(n)` uses each component's optional `sample(n)`; a component without it
  raises `NotImplementedError`.
- **Persistence.** `save`/`load` pickle the components with `torch.save`; the component classes must
  be importable when loading. `load_state_dict`-style partial persistence is not offered.
- **Gibbs mode** (`n_components=None`) does not exist for this class (`components` fixes `K`).
- **pomegranate.** Not a dependency. The protocol imports nothing from it; the parity test uses
  `pytest.importorskip('pomegranate')` and an adapter written in the test file. Whether pomegranate
  belongs in the `[dev]` extras or stays a CI-only install is left to the project owner.

## Out of scope

Ready-made components in the library (the tests define `DiagNormal`, `Exponential`, ...), per-feature
weights, `n_init > 1`, SAEM / online EM, constrained `GaussianMixture`-style sampling.
