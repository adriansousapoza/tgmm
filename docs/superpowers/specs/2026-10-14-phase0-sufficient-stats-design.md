# Phase 0: sufficient-statistics M-step, algorithm switch, memory fixes — Design

Part of the [tgmm v0.4 roadmap](../../../tgmm%20v0.4%20roadmap%20coding-agent%20plan.md).
This phase is a pure refactor: every later phase (SEM, SAEM, mini-batch EM,
GPCM, Student-t/vMF, label clamping) needs the M-step to consume accumulated
statistics rather than `(X, resp)` directly, so this does that refactor
first, with zero behaviour change for existing callers.

## Goals

- Introduce `tgmm.stats.SufficientStats`, a small accumulator
  (`nk`, `s1`, `s2`, `n_samples`) that any M-step can be rewritten to consume.
- Split `GaussianMixture._m_step(X, resp)` into
  `_compute_stats(X, resp, batch_size=None)` (builds a `SufficientStats`) and
  `_m_step_from_stats(stats)` (the existing MLE/MAP/NIW branches, rewritten
  to read `nk`/`s1`/`s2` instead of re-touching `X`/`resp`). `_m_step` becomes
  a one-line wrapper over the two, kept for the one remaining direct caller
  (the supervised-fit path in `fit()`).
- Fix the `(n_samples, n_components, n_features, n_features)` memory
  blow-up in `_update_mle_full`, `_update_map_full`, `_update_niw_full` by
  computing the full-covariance second moment via
  `torch.einsum('nk,nkd,nke->kde', resp, diff, diff)` instead of forming
  `diff.unsqueeze(3) * diff.unsqueeze(2)` elementwise.
- Add `algorithm: str = 'em'` (`'em' | 'cem' | 'sem'`), with `cem=True`
  mapped to `algorithm='cem'` plus a `DeprecationWarning`, and `'sem'`
  accepted but not runnable until Phase 2.
- Add `batch_size: Optional[int] = None` so the E-step and stats
  accumulation can run in row chunks, bounding peak memory to
  `O(batch_size · n_components · n_features)` regardless of `n_samples`.
- Add `convergence: str = 'relative' | 'aitken'`, an alternative
  Aitken-accelerated stopping rule for `_fit_single_run`.

## `SufficientStats` (`tgmm/stats.py`)

```python
@dataclass
class SufficientStats:
    nk: torch.Tensor          # (K,)   sum_i r_ik
    s1: torch.Tensor          # (K, d) sum_i r_ik x_i
    s2: torch.Tensor          # shape depends on covariance_type; see below
    n_samples: int
```

`s2` is **always computed around each component's own empirical mean for
this E-step**, `ȳ_k = s1_k / nk_k` — not around whatever `self.means_`
currently holds. Shapes: `full` → `(K, d, d)`; `diag` → `(K, d)`;
`spherical` → `(K,)`. For `tied_*` covariance types, `s2` is still computed
per-component at the `full`/`diag`/`spherical` shape above (means stay
per-component for tied types — only the covariance is shared); the M-step
pools it across `k` (`stats.s2.sum(dim=0)`) before dividing by the pooled
`nk.sum()`, exactly mirroring what the pre-refactor tied `_update_*_tied_*`
methods did by looping the per-component `diff` through a pooling `einsum`.

This choice (centering on `ȳ_k`, not `self.means_`) is what makes
`__add__`/chunking exact: chunks differ only in which rows contributed to
`nk`/`s1`, and `s2` for a *global* `ȳ_k` can't be obtained by summing
per-chunk second moments centered on *per-chunk* means (that under-counts
the between-chunk-mean spread — the classical parallel-variance pitfall).
`_compute_stats` sidesteps this with a **two-pass** chunked accumulation:
pass 1 sums `nk`/`s1` over chunks (exact, since summation is linear and the
reference point doesn't matter yet); pass 2, now that the true `ȳ_k` is
known, re-walks the chunks computing `s2` centered on that fixed `ȳ_k` and
sums the results (also exact, since the center is the same constant in
every chunk). Peak memory stays `O(batch_size · K · d)` — the second pass
costs roughly 2x the FLOPs of a naive single pass, traded for exactness.

`__add__` sums all four fields elementwise (`n_samples` included — it is
only used to track how many rows of `X`/`resp` a `SufficientStats`
instance summarizes, for `_compute_stats`'s own chunking and, come Phase 2,
for mini-batch accumulation bookkeeping).

`scale(factor)` multiplies `nk`, `s1`, `s2` by `factor` and leaves
`n_samples` untouched — it exists for Phase 2's online-EM running-statistics
blend (`s ← (1−γ)s + γ·(n/b)·s_batch`), which needs to reweight an
accumulator's contribution without claiming it suddenly summarizes a
different number of physical rows. Not exercised by anything in Phase 0.

## Reconstructing each M-step branch from `(nk, s1, s2)`

The non-NIW branches need the covariance centered on `self.means_` — which,
whenever a mean prior shrinks the mean away from `ȳ_k` (MAP, independent
prior), is *not* the same point `s2` is centered on. Rather than re-touch
`X`, `_m_step_from_stats` reconstructs the needed scatter algebraically
(parallel-axis / König–Huygens): for any component mean estimate `c_k`,

```
Σ_i r_ik (x_i - c_k)(x_i - c_k)^T  =  s2_k + nk_k (ȳ_k - c_k)(ȳ_k - c_k)^T
```

(the cross term vanishes because `Σ_i r_ik (x_i - ȳ_k) = 0` by definition of
`ȳ_k`). This is implemented once, as `_scatter_about_means(stats, nk, ybar,
base_type, pool_tied)`, shared by the MLE and MAP covariance updates:

- **MLE** (`use_mean_prior=False`): `self.means_ = ȳ_k`, so the mismatch
  term `nk_k(ȳ_k - c_k)^2` is exactly zero and `s2` is used as-is — bit
  identical to today's `diff = X - self.means_` computation, just already
  reduced during `_compute_stats`.
- **MAP, independent priors** (`use_mean_prior=True`,
  `use_covariance_prior` either way): `self.means_` is the shrunk MAP mean,
  so the mismatch term is generally non-zero and supplies exactly the
  correction the pre-refactor code computed directly (its `prior_term`
  variable, in the mean-prior branch of `_update_map_full/diag/spherical`,
  *is* this same `nk(ȳ-μ_MAP)^2` quantity — compare the two if in doubt).
- **NIW**: both the scatter term `S_k` and the cross term
  `(λ₀ n_k / λ_n)(ȳ_k − μ₀)(ȳ_k − μ₀)ᵀ` in the conjugate update are already
  expressed in terms of `ȳ_k` in the pre-refactor code (its own
  `empirical_means` variable) — so this branch needs no correction at all,
  `S_k = stats.s2` directly, and `ȳ_k = stats.s1 / nk` replaces the
  separate `resp.t() @ X / nk` computation. This is the simplest branch and
  the strongest argument that centering `s2` on `ȳ_k` is the right choice.

Weights are unaffected by any of this (computed from `nk`/`n_samples`
exactly as before). `_m_step` becomes:

```python
def _m_step(self, X, resp):
    self._m_step_from_stats(self._compute_stats(X, resp, batch_size=self.batch_size))
```

## `algorithm` and backward compatibility with `cem`

```python
algorithm: str = 'em'   # 'em' | 'cem' | 'sem'
cem: bool = False        # deprecated
```

- `cem=True` → `warnings.warn(DeprecationWarning)`, then treated as
  `algorithm='cem'`. If the caller *also* passed an explicit `algorithm`
  that isn't `'em'` or `'cem'` (i.e. `'sem'`), that's a genuine conflict →
  `ValueError`. `algorithm='em'` (the default) plus `cem=True` is not
  treated as a conflict — that's just the old calling convention.
- `self.cem` is still set (`True` iff `self.algorithm == 'cem'`) and still
  read nowhere except `__init__`/`save`/`load` bookkeeping —
  `_fit_single_run` now dispatches on `self.algorithm`, per the plan.
- `algorithm='sem'` is accepted at construction (so Phase 2 code that
  already sets it doesn't need an `__init__` change later) but `fit()`
  raises `NotImplementedError` if it's actually used before Phase 2 lands.
- Gibbs mode (`n_components=None`) does not yet validate `algorithm` at all
  (today it silently ignores `self.cem` the same way) — Phase 2.5 is where
  that rejection is an explicit acceptance criterion; adding it early would
  pre-empt that phase's own test for it.

## `batch_size` (E-step and stats chunking)

`_e_step`'s existing body becomes `_e_step_chunk` (no logic change); the
public `_e_step(X)` loops over `range(0, n, batch_size)` and concatenates
when `self.batch_size` is set and smaller than `X.shape[0]`, else calls
`_e_step_chunk(X)` once. This is exact, not approximate — softmax
normalization is per-row, so chunking rows changes nothing about any row's
result. `_compute_stats` chunks internally the same way (see above). No
caller of `_e_step`/`_m_step` needed to change for this.

## `convergence` (Aitken acceleration)

`_fit_single_run` tracks the last two lower bounds it compared against
(`prev_lower_bound`, `prev_prev_lower_bound`) and calls
`_check_convergence(current, previous, previous2, tol)`:

- `convergence='relative'` (default): exactly today's
  `abs(current - previous) / (abs(previous) + 1e-20) < tol` — so the
  default path is untouched, byte for byte.
- `convergence='aitken'`: `a = (current - previous) / (previous -
  previous2)`; if `previous2` is unavailable yet, or `|previous - previous2|`
  is ~0, or `a >= 1`, falls back to the relative rule for that iteration
  (per the plan's guard). Otherwise `l_inf = previous + (current -
  previous) / (1 - a)` and converges when `l_inf - current < tol`.

## Memory regression test: `tracemalloc` doesn't work here

The plan suggests `tracemalloc` for the CPU path. Checked empirically
before writing the test: `tracemalloc` instruments only CPython's own
allocator domain, and PyTorch CPU tensors are allocated through
`c10::alloc_cpu` (plain `malloc`), entirely outside that domain —
allocating a 3.3 GB tensor and checking `tracemalloc.get_traced_memory()`
reports ~750 *bytes* of traced memory. A test built on it would pass
whether or not the `(n, K, d, d)` blow-up was actually fixed, i.e. it
wouldn't be testing anything. `resource.getrusage(...).ru_maxrss` (peak
RSS) does see it (verified: ~3.1 GB delta for the same tensor) and is used
instead for the CPU path; `torch.cuda.max_memory_allocated()` is used as
specified when CUDA is available (not exercised in this environment — no
GPU here). Because `ru_maxrss` is a process-lifetime high-water mark that
can't be reset mid-process, the fit under test runs in a subprocess so an
unrelated earlier peak can't mask a regression.

## Pre/post bit-identity testing

`_m_step_from_stats` can't be compared against "the old `_m_step`" directly
once the old code is gone. Instead, `tests/fixtures/phase0_pre_refactor_fits.pt`
was generated by running the *pre-refactor* code once (seeded data, 15 EM
iterations, all 6 covariance types × {MLE, MAP, NIW}) and saving
`weights_`/`means_`/`covariances_`/`lower_bound_` plus the input `X`. Tests
re-fit the same `X` under the refactored code with identical settings and
assert `atol=1e-10` agreement in float64.

## Non-goals for this phase

- No change to `_fit_gibbs*` (Gibbs-sampling mode) — it does not use
  `_m_step` at all.
- No `aux` field on `SufficientStats` yet (Phase 4 adds it for
  family-specific accumulators — adding an unused dict now would be
  speculative).
- No `partial_fit`, mini-batch blending, or `step_size` — Phase 2.
- No Gibbs-mode `algorithm` validation — Phase 2.5.
