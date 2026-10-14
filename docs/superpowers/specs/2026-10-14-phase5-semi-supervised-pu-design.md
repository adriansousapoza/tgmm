# Phase 5: semi-supervised and positive-unlabeled fitting — Design

Part of the [tgmm v0.4 roadmap](../../../tgmm%20v0.4%20roadmap%20coding-agent%20plan.md).
Depends on Phase 2 (done) and Phase 4 (done -- `BaseMixture`/`StudentTMixture`/
`VonMisesFisherMixture`). Final phase before the `0.4.0` release.

⚠ Decision point the plan itself flags: `PositiveUnlabeledMixture`'s default
`anchor`. Asked the user directly; answer was **`'prior'`** (matching the
plan's own code-signature default) -- see [[v0_4_roadmap_progress]].

## 5.1 Semi-supervised fitting: generalizing `fit(X, labels=...)` into `BaseMixture`

Pre-Phase-5, `GaussianMixture.fit(X, labels=...)` only existed on
`GaussianMixture` itself, required every row labeled, and required exactly
one component per class. This phase generalizes it into `BaseMixture` (so
every family gets it) and adds two things: `components_per_class` (more
than one component per class, MclustDA-style) and `-1` for "label unknown"
(semi-supervised EM instead of a one-shot M-step).

### Fully labeled path (`_fit_supervised`): two sub-cases

- **Every class's block is exactly 1 component** (the default, and the
  historical behavior): a single one-hot-responsibility M-step over the
  *whole* dataset at once, exactly as `GaussianMixture` always did it (now
  moved to `BaseMixture`, generic via the `_m_step`/`_allocate_parameters`
  abstract hooks). This is the only sub-case that correctly supports tied
  covariance types (`tied_full`/`tied_diag`/`tied_spherical`), since tied
  covariance is pooled across *all* components by a single M-step call
  regardless of which class each component belongs to.
- **Any class's block has more than 1 component**: each class's own block
  is fit independently by running the *ordinary unsupervised* EM loop
  (`BaseMixture.fit`'s own machinery, reused via a new helper
  `_fit_component_block`) on just that class's rows, with `n_components`
  temporarily set to that class's block size. The resulting per-class
  parameter sets are concatenated (via `_get_param_snapshot`/
  `_apply_snapshot`, so this works for *any* family's parameter set,
  not just Gaussian's) into the full model, each class's `weights_`
  rescaled by `n_c / n_total` so the overall mixture reflects true class
  proportions. **Scope reduction**: this sub-case raises `ValueError` for
  tied covariance types -- "tied" means "one covariance shared by every
  component in the whole model," which is incompatible with fitting each
  class's block as an independent sub-mixture (there's no single matrix
  to assemble from N independently-tied per-class fits). Not tested by
  the plan's own test list, and not worth the extra design complexity of
  threading partial-sharing through the block-assembly step for this
  release.

`_fit_component_block` temporarily reassigns `self.n_components` (and
`self.random_state`, if given), runs `BaseMixture.fit` directly (not
`self.fit`, to avoid re-entering a subclass's own `fit` override), then
restores the instance's original configuration and fitted state in a
`finally` block -- the same "swap down, restore after" idiom
`GaussianMixture._active_e_step` already uses for the Gibbs-mode active-
components restriction.

`GaussianMixture`'s pre-existing sparse-component warning (too few points
in one class to reliably estimate `'full'`/`'diag'`/`'spherical'`
covariance) is preserved via a new hook, `_post_supervised_fit_warnings
(component_counts)`, defaulting to a no-op in `BaseMixture` and overridden
by `GaussianMixture` with its existing check. `GaussianMixture`'s old
micro-optimization of swapping `init_means`/`init_covariances` to cheap
placeholders before the one-shot M-step (since their *values* are
immediately overwritten) is **not** preserved in the generalized path --
for the `components_per_class=1` fast path there's no init step at all
(straight to the M-step, as before); the `>1` sub-case needs a real init
every class-block, same cost any other unsupervised fit pays.

### Semi-supervised path (`_fit_semisupervised`): any `-1` present

New per-iteration mechanics, used uniformly across families via existing
abstract hooks (`_component_log_prob`, `_e_step`, `_m_step`):

- `class_of_component_` (shape `(n_components,)`): which class (0-indexed
  into `classes_`) each component belongs to.
- `is_labeled` (shape `(n_samples,)`, bool) and `label_class_idx` (shape
  `(n_samples,)`, long, garbage/unused where `is_labeled` is `False`).
- `_clamp_responsibilities(resp, is_labeled, block_mask, label_confidence)`:
  for labeled rows, zero responsibility outside the row's class block and
  renormalize inside it (one-hot when the block has 1 component);
  `label_confidence < 1` blends `label_confidence` mass on the renormalized
  in-block posterior with `1 - label_confidence` on the *original*
  (unclamped) out-of-block posterior, rescaled to sum to that remainder --
  exactly the plan's "keeps mass c on its class block and distributes
  1 − c over the other components by the unclamped posterior." Unlabeled
  rows pass through untouched.
- The monitored objective (for convergence and `n_init` selection) is the
  plan's own: `sum_labeled log sum_{k in block(c_i)} pi_k f_k(x_i) +
  sum_unlabeled log sum_k pi_k f_k(x_i)` -- computed directly from
  `_component_log_prob` (masking to `-inf` outside each labeled row's
  block before `logsumexp`), not from the clamped responsibilities
  (clamping is a *statistics-accumulation* device, not part of the
  likelihood itself).
- `unlabeled_weight` multiplies **unlabeled rows'** (clamped, which for
  unlabeled rows is a no-op) responsibilities before the M-step only --
  i.e. only the statistics accumulation step sees the reweighted
  responsibilities; the monitored objective above is never reweighted.
  Implemented as a single `resp.clone(); resp[~is_labeled] *=
  unlabeled_weight` right before `_m_step`, so no family's `_compute_stats`
  needs to know this mechanism exists.
- `algorithm='cem'`: `_c_step` (hard-assign) is applied only to unlabeled
  rows; labeled rows keep their (soft, renormalized) clamped
  responsibility, matching the plan's "CEM/SEM/SAEM leave clamped rows
  untouched."

**Scope reduction**: semi-supervised fitting supports `algorithm in ('em',
'cem')` only for this release. `'sem'`/`'saem'`/online EM
(`step_size` set) combined with partial labels raise `NotImplementedError`
with a clear explanation. Generalizing the S-step/running-statistics-blend
machinery to also respect per-row label clamping is a real extension (the
sampled hard assignment and the running accumulator both need the same
"leave labeled rows alone" treatment SEM/CEM need, but interacting
correctly with burn-in/best-or-mean selection and the online mini-batch
loop respectively is substantially more surface area), and none of the
plan's own Phase 5.1 tests exercise it. Tracked the same way GPCM's
MAP/Gibbs gaps and Phase 5.3's unbounded-negative-block stretch goal are:
a documented, deliberate release-scope boundary, not an oversight.

### `components_per_class` resolution

`components_per_class: int | dict[Any, int] = 1`. Resolved inside
`fit` (not `__init__`, since the actual class labels aren't known until
then): `classes_ = unique(labels[labels != -1], sorted=True)`; an `int`
gives every class that same block size; a `dict` is indexed by the
original label value (not its position in `classes_`). The total must
equal `self.n_components`, else `ValueError`. `class_of_component_` lays
out blocks in `classes_`'s sorted order, contiguously -- the plan's own
"MclustDA-style... contiguous block."

## 5.2 `PositiveUnlabeledMixture`

Lives in new `tgmm/semi_supervised.py` (per the plan's file-layout note).
A thin wrapper per the plan: builds the underlying mixture
(`GaussianMixture`/`StudentTMixture`/`VonMisesFisherMixture` depending on
`family`) with `components_per_class={1: n_components_positive, 0:
n_components_negative}`, labels `1` on `positive_mask` and `-1` elsewhere,
and fits it through the Phase 5.1 path above.

- `anchor='fixed'`: fit the positive block once (a plain supervised fit
  on just the labeled positives, via `_fit_component_block`), then run
  semi-supervised EM with the positive block's parameters frozen after
  every M-step (restored from the initial fit each iteration; only
  `weights_` for those components -- i.e. the class prior -- is allowed to
  keep moving). Works for every `family`.
- `anchor='prior'` (**default**, per the user's confirmation): the same
  initial fit on labeled positives, then used as an NIW prior
  (`mean_prior`/`covariance_prior` = that fit's own mean/covariance,
  `mean_precision_prior = degrees_of_freedom_prior = n_labeled`) for a
  *fresh* `GaussianMixture` that then runs ordinary semi-supervised EM
  (the NIW conjugate update, not a frozen block, pulls the positive
  block's parameters toward the labeled-only fit without hard-freezing
  them). **Scope reduction: Gaussian-only.** NIW conjugate priors are a
  `GaussianMixture`-specific mechanism (`mean_prior`/`covariance_prior`
  don't exist as concepts for `StudentTMixture`/`VonMisesFisherMixture`);
  `anchor='prior'` with `family != 'gaussian'` raises `ValueError` pointing
  at `'fixed'`/`'free'` instead.
- `anchor='free'`: plain semi-supervised EM, no special treatment of the
  positive block beyond its initialization from the labeled positives.
  Works for every `family`. Expected (and, per the plan, only
  *documented*, not asserted in tests) to let the positive block drift
  toward the negatives on strongly overlapping data -- that's the point
  of having `'fixed'`/`'prior'` as the identifiability anchors.
- SCAR (Selected Completely At Random; Elkan & Noto, 2008) is assumed and
  documented: labeled positives are a uniform random subsample of all true
  positives, not a biased one.
- `class_prior_`: mean, over unlabeled rows only, of the summed
  responsibility on the positive block -- the estimated fraction of
  *unlabeled* rows that are actually positive. `predict_proba_positive(X)`
  sums responsibility on the positive block for arbitrary `X`;
  `predict_positive(X, threshold=0.5)` thresholds it.

### 5.3 (stretch, out of scope)

`n_components_negative=None` (unbounded Gibbs sampler for the negative
block, positive block held fixed) is explicitly out of scope per the plan
itself. `PositiveUnlabeledMixture(n_components_negative=None, ...)` raises
`NotImplementedError` with the plan's own explanation and a pointer to
this doc.

## Non-goals for this phase

- `algorithm in ('sem', 'saem')` or online EM (`step_size`) combined with
  partial labels (`-1` present) -- `NotImplementedError`, see above.
- Tied covariance types (`tied_full`/`tied_diag`/`tied_spherical`) with
  any `components_per_class` block size `> 1` -- `ValueError`, see above.
- `anchor='prior'` for `family != 'gaussian'` -- `ValueError`, see above.
- `n_components_negative=None` (5.3) -- `NotImplementedError`, per the
  plan's own scoping.
