# Phase 2: Stochastic EM, deterministic annealing, SAEM, mini-batch EM — Design

Part of the [tgmm v0.4 roadmap](../../../tgmm%20v0.4%20roadmap%20coding-agent%20plan.md).
Builds on [[Phase 0]](2026-10-14-phase0-sufficient-stats-design.md)'s
`SufficientStats`/`algorithm`/`batch_size` plumbing; independent of Phase 1.

## Scope recap

1. SEM (`algorithm='sem'`): `_s_step`, no pointwise convergence,
   `sem_burn_in`/`sem_estimate`, empty-component re-seeding,
   `log_likelihood_history_`.
2. Deterministic annealing (`temperature`): tempered E-step, suspended
   convergence while annealing, works with `'em'|'cem'|'sem'`.
3. Mini-batch / online EM (`step_size`, `partial_fit`).
4. SAEM (`algorithm='saem'`): SEM + the online blend from (3).
5. `algorithm` accepts `'em'|'cem'|'sem'|'saem'`; Gibbs mode rejects
   anything but `'em'`.

## The one mechanical issue every later piece depends on: blending `SufficientStats`

`SufficientStats.__add__` sums `n_samples` (correct for Phase 0's chunking
use case: disjoint row-chunks of one dataset should sum to that dataset's
total `N`). Online EM's and SAEM's running-statistics blend,

```
s <- (1 - gamma_t) * s + gamma_t * (n / b) * s_batch
```

is a different operation: `s` and `s_batch` both already represent
"statistics as if accumulated over the full dataset" (that's what the
`n/b` rescaling of the batch's own stats is *for*), and should **stay** at
that same nominal `n_samples = n` after blending -- not have their
`n_samples` fields added together (`n + n = 2n`, growing without bound
over iterations). `scale()` already leaves `n_samples` untouched by
design (see the Phase 0 doc), so `s.scale(1 - gamma) + s_batch.scale(gamma
* n / b)` computes the right `nk`/`s1`/`s2` tensors via `__add__`, but
needs its `.n_samples` field overwritten back to `n` immediately
afterward (`SufficientStats` isn't frozen, so this is a plain
attribute assignment). A tiny shared helper does this once:

```python
@staticmethod
def _blend_stats(running, new, gamma, n_samples, new_scale=1.0):
    blended = running.scale(1.0 - gamma) + new.scale(gamma * new_scale)
    blended.n_samples = n_samples
    return blended
```

used by both mini-batch/online EM (`new_scale = n / b`) and SAEM
(`new_scale = 1.0`, since SAEM's "batch" is the whole dataset every
iteration -- see below).

## 2.1 SEM

- `_s_step(resp)`: `z = torch.multinomial(resp, 1).squeeze(1); return
  F.one_hot(z, resp.shape[1]).to(resp.dtype)`. No explicit `torch.Generator`
  needed -- `fit()`'s n_init loop already calls `torch.manual_seed(...)`
  before `_fit_single_run`, so `torch.multinomial`'s draw comes from that
  same seeded global stream, matching "seed through the existing
  random_state generator" without extra plumbing.
- Separate loop (`_fit_single_run_sem`), not folded into the EM/CEM loop:
  SEM always runs the full `max_iter` sweeps (no early stopping -- same
  documented convention as Gibbs mode, including `converged_` being
  unconditionally `True` afterward), and needs to retain every
  post-burn-in iterate's parameters (for `sem_estimate='mean'`) or track a
  running best (`='best'`), neither of which fits the EM/CEM loop's
  break-on-convergence structure.
- `sem_burn_in` resolves to `max_iter // 5` when `None`, per the plan.
  **Both** `sem_estimate` options restrict to post-burn-in iterates (re-read
  closely: "'best' (keep the iterate with the highest observed
  log-likelihood **after burn-in**)" -- not the global best from iteration
  0). If `sem_burn_in >= max_iter` (no post-burn-in iterates at all), falls
  back to the single last iterate for both options, rather than raising.
- `log_likelihood_history_`: every per-sweep log-likelihood, including the
  initial (pre-loop) E-step and the burn-in period -- needed for the
  notebook's "SEM trace shows the stationary band" plot, which has to show
  the whole trace to be legible as a trace.
- Empty-component re-seeding: `nk = z_onehot.sum(dim=0)`; for every `k`
  with `nk[k] < 1`, assign that component's mean to the data point with
  the **worst** (lowest) log-likelihood among points *not already claimed*
  by another re-seed in the same sweep (`torch.argsort(log_prob_norm)`,
  taking the first `n_empty` indices) -- using the single worst point for
  every emptied component would collide if more than one empties in the
  same sweep. Covariance re-seeds to a "global covariance" shaped to match
  `covariance_type` (`full`: `torch.cov(X.T)`; `diag`: per-feature
  variance; `spherical`: mean variance) for the three non-tied types only
  -- `tied_*` covariance isn't indexed per-component, so there's nothing
  to re-seed per empty slot there; only the mean re-seeds for tied types.
  One `UserWarning` per sweep with any re-seeds, naming the count;
  `n_empty_components_` accumulates the total across the whole run.

## 2.2 Deterministic annealing (DAEM)

- `_e_step`/`_e_step_chunk` gain `beta: float = 1.0`: `log_prob = beta *
  (log_prob + log_weights)` computed *before* `logsumexp`, exactly "one
  multiplication... before the logsumexp" as specified. `_e_step`'s
  chunking wrapper threads `beta` through unchanged.
- `_beta_for_iteration(n_iter)`: `self.temperature is None -> 1.0`;
  otherwise geometric interpolation from `beta_0` at `n_iter=0` to `1.0` at
  `n_iter=n_anneal`, held at `1.0` thereafter:
  `beta_0 ** (1 - min(n_iter, n_anneal) / n_anneal)`.
- Wired into the EM/CEM loop (`_fit_single_run`) and the SEM loop
  (`_fit_single_run_sem`) identically: `beta` for the *next* E-step is
  resolved once per iteration; the convergence check (EM/CEM: the usual
  `_check_convergence`; SEM: N/A, it never checks) is skipped whenever the
  *current* `beta < 1.0`. Not wired into SAEM or plain online EM -- the
  plan says DAEM "works with 'em', 'cem' and 'sem'" and doesn't mention
  the other two; SAEM already has its own convergence mechanism via
  `gamma_t -> 0`.
- `annealing_schedule_`: the list of `beta` values actually used, one per
  E-step call in the run (length `n_iter + 2`: the initial E-step plus one
  per loop iteration up to and including the break).

## 2.3 Mini-batch / online EM

- New constructor arg `step_size: Optional[float | str | Callable[[int],
  float]] = None`. `None` (default): Phase 0's behaviour is unchanged
  (`batch_size` alone just chunks for memory, exact EM). Set (a float, the
  string `'auto'`, or a callable `t -> gamma_t`): switches `fit()` to
  online EM for `algorithm='em'`. Requires `batch_size` to also be set
  (there is no batch to be "mini" without one) -- raises `ValueError` at
  fit time otherwise. `'auto'` resolves to `gamma_t = (t + 2) ** -0.7`,
  the plan's documented default (Cappé & Moulines recommend exponents in
  `(0.5, 1]`; `0.7` sits in that range).
- `_fit_single_run_online`: `max_iter` epochs, each epoch a fresh
  `torch.randperm(n)` (seeded globally, same mechanism as SEM's sampling)
  sliced into `batch_size`-sized chunks. Per batch: `_e_step` on just that
  batch, `_compute_stats` on just that batch (`n_samples = b`), blend into
  the running accumulator via `_blend_stats(..., new_scale=n/b)`, one
  `_m_step_from_stats` on the blended result. `t` (the step-size clock)
  increments per batch, not per epoch. Log-likelihood on the *full*
  dataset is recorded once per epoch (into `log_likelihood_history_`) for
  monitoring -- not used for early stopping, since the plan only says
  "`max_iter` counts epochs" for this path, nothing about a stopping rule.
  `converged_ = True` unconditionally after completing all epochs (same
  "nothing to fail at" convention as Gibbs/SEM).
- `partial_fit(X_batch)`: one such blend-and-M-step update, for streaming
  use after an initial `fit()` call (or explicit parameter
  initialization) has already set `weights_`/`means_`/`covariances_` --
  raises `RuntimeError` if called on an unfitted model. Continues the same
  running accumulator and step counter `fit()` would have built
  (`self._online_running_stats_`, `self._online_step_count_`, both
  private - `partial_fit` is unusable without a preceding `fit()` or
  equivalent manual initialization in this design, since the running
  accumulator has to start somewhere). Uses `new_scale = 1.0` (no `n/b`
  rescaling): a true streaming caller doesn't know a fixed total `N` up
  front, so each `partial_fit` batch is treated as a complete update in
  its own right rather than rescaled against a dataset size that doesn't
  exist in this calling pattern. This is a deliberate asymmetry from the
  `fit()`-driven path (documented on `partial_fit` itself) -- the two
  aren't meant to produce bit-identical results, both match the plan's
  text, which only specifies the `n/b` formula for `fit()`'s own epoch
  loop.

## 2.4 SAEM

SAEM's "batch" is the **whole dataset**, every iteration -- unlike mini-batch
EM, there is no sub-sampling. `_fit_single_run_saem`:

```
E-step (full X) -> S-step (full X) -> _compute_stats(X, z_onehot, batch_size=self.batch_size)
  -> blend with _blend_stats(..., new_scale=1.0)  # n/b = n/n = 1
  -> _m_step_from_stats
```

`self.batch_size` here is purely Phase 0's memory-chunking optimization
(exact, not online) -- SAEM's own stochasticity comes entirely from the
`_s_step` sampling, not from row sub-sampling. `tol`/`_check_convergence`
apply exactly as in the EM/CEM loop (`gamma_t -> 0` is what makes this
converge, per the plan). If `step_size is None` when `algorithm='saem'`,
defaults to `'auto'` rather than raising -- SAEM's whole mechanism *requires*
a decaying step size, so requiring the caller to redundantly also pass
`step_size='auto'` after already choosing `algorithm='saem'` would be
pure friction for the documented, recommended default.

## 2.5 Wiring and acceptance

- `algorithm` validation (`__init__`) extends to `('em', 'cem', 'sem',
  'saem')`. The Phase-0-added `NotImplementedError` guard for
  `algorithm='sem'` in `fit()` is removed (now implemented);
  `algorithm='saem'` needs no equivalent guard since it didn't exist as an
  accepted value before this phase.
- Gibbs mode (`n_components=None`) now validates `algorithm` for the first
  time: anything but `'em'` raises `ValueError` at `__init__`, naming why
  (Gibbs sampling has no S-step/stochastic-approximation/annealing
  analogue in this codebase).
- `_fit_single_run` becomes a dispatcher: `'sem' ->
  _fit_single_run_sem`, `'saem' -> _fit_single_run_saem`, `step_size is
  not None and algorithm == 'em' -> _fit_single_run_online`, else the
  existing EM/CEM loop (now also carrying the DAEM `beta` wiring).

## Non-goals for this phase

- No empty-component re-seeding for online EM or SAEM (the plan's
  re-seeding paragraph is under SEM specifically).
- No early stopping for the pure online-EM path beyond "epochs = `max_iter`"
  (not specified).
- No interaction between `temperature` and SAEM/online EM (not specified).
