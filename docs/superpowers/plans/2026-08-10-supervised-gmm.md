# Supervised GMM Fitting via Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `GaussianMixture.fit(X, labels=...)` fit one Gaussian component per distinct label directly from ground-truth labels (instead of estimating the assignment via EM/CEM), then demonstrate it against plain EM and CEM in a new comparison notebook.

**Architecture:** A new `labels=None` keyword argument on `GaussianMixture.fit`. When given, a new private helper `_validate_labels` maps arbitrary label values to contiguous component indices (`0..n_components-1`, recorded in `self.classes_`), builds a one-hot responsibility matrix from those indices, and calls the *existing* `_m_step(X, resp)` exactly once — no EM/CEM loop, no `n_init`. This is safe because `_m_step` recomputes `weights_`/`means_`/`covariances_` entirely from `(X, resp)` on every call (verified: none of `_update_mle_full`, `_update_covariances_map`, `_update_niw_conjugate` read the *previous* value of `self.covariances_`/`self.means_` — only the fresh `self.means_` set earlier in the same `_m_step` call), so a fixed one-hot `resp` is already at the M-step's fixed point after one call.

**Tech Stack:** Python 3, PyTorch (`.venv` at repo root has `torch==2.12.1+cpu`, `pytest==9.1.1`, `nbconvert==7.17.1`), pytest, Jupyter notebooks via `nbconvert --execute` for headless verification.

## Global Constraints

- `labels=None` (the default) must be a byte-for-byte no-op change to existing unsupervised `fit()` behavior — every existing test in `tests/test_gmm.py` must keep passing unmodified.
- Supervised fitting requires exactly one Gaussian component per distinct label: `n_components` must equal the number of distinct values in `labels`, or `fit` raises `ValueError`. No multiple-components-per-class support (that's MDA/MclustDA — out of scope, see design doc `docs/superpowers/specs/2026-08-10-supervised-gmm-design.md`).
- No semi-supervised / partial-label path (out of scope, see design doc Non-goals).
- Reuse `_m_step` as-is — do not modify it. Do not modify `_c_step`, `_e_step`, or the unsupervised `_fit_single_run` loop.
- Run all commands with the repo's venv active: `source .venv/bin/activate` (confirmed present, `torch`/`pytest` importable there; system `python3` has neither).

---

### Task 1: Supervised fit core (`labels` parameter, `_validate_labels`, docstrings)

**Files:**
- Modify: `tgmm/gmm.py:714-751` (`fit` signature + docstring), `tgmm/gmm.py:801-805` (insert supervised branch after data prep, before the `n_init` loop), new private method `_validate_labels` (add directly above `_c_step` at `tgmm/gmm.py:983`)
- Modify: `tgmm/gmm.py:96-125` (class docstring `Attributes` section — add `classes_`)
- Test: `tests/test_gmm.py` (new section appended at end of file, after the existing `# === 14. Stress tests ===` section which currently ends at line 542)

**Interfaces:**
- Produces: `GaussianMixture.fit(self, X, labels=None, max_iter=None, tol=None, random_state=None, warm_start=None) -> "GaussianMixture"` — `labels` accepts any 1D tensor/array-like of length `n_samples`.
- Produces: `GaussianMixture._validate_labels(self, labels, X: torch.Tensor) -> torch.Tensor` — returns a `torch.long` tensor of component indices in `[0, n_components)`, and sets `self.classes_` (a sorted `torch.Tensor` of the distinct original label values, so `self.classes_[k]` is the original label for component `k`).
- Produces: `self.classes_` attribute, present only after a supervised fit.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gmm.py` (after the last line, currently `test_different_n_init_values`'s body ending at line 542 — add a blank line then this section):

```python
# ============================================================================
# 15. Supervised fitting (labels)
# ============================================================================

def generate_labeled_test_data(n_per_class=60, n_features=3, n_classes=3, seed=7):
    """Generate synthetic per-class clustered data with known labels."""
    generator = torch.Generator().manual_seed(seed)
    X_list, y_list = [], []
    for c in range(n_classes):
        mean = torch.randn(n_features, generator=generator) * 5
        cov = torch.eye(n_features) + torch.randn(n_features, n_features, generator=generator) * 0.1
        cov = cov @ cov.T  # positive definite
        X_c = torch.randn(n_per_class, n_features, generator=generator) @ cov.T + mean
        X_list.append(X_c)
        y_list.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)


X_SUP, Y_SUP = generate_labeled_test_data()


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical", "tied_full"])
def test_supervised_fit_recovers_per_class_mean(covariance_type):
    gmm = GaussianMixture(n_components=3, covariance_type=covariance_type)
    gmm.fit(X_SUP, labels=Y_SUP)
    assert gmm.fitted_
    for c in range(3):
        expected_mean = X_SUP[Y_SUP == c].mean(dim=0)
        assert torch.allclose(gmm.means_[c].cpu(), expected_mean, atol=1e-4)


def test_supervised_fit_recovers_per_class_full_covariance():
    gmm = GaussianMixture(n_components=3, covariance_type="full", reg_covar=0.0)
    gmm.fit(X_SUP, labels=Y_SUP)
    for c in range(3):
        X_c = X_SUP[Y_SUP == c]
        expected_cov = torch.cov(X_c.T, correction=0)
        assert torch.allclose(gmm.covariances_[c].cpu(), expected_cov, atol=1e-4)


def test_supervised_fit_recovers_class_weights():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_SUP, labels=Y_SUP)
    expected_weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    assert torch.allclose(gmm.weights_.cpu(), expected_weights, atol=1e-4)


def test_supervised_fit_converges_in_one_iteration():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_SUP, labels=Y_SUP)
    assert gmm.converged_
    assert gmm.n_iter_ == 1


def test_supervised_fit_predict_proba_sums_to_one():
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_SUP, labels=Y_SUP)
    probs = gmm.predict_proba(X_SUP)
    assert torch.allclose(probs.sum(dim=1).cpu(), torch.ones(X_SUP.size(0)), atol=1e-5)


def test_supervised_fit_noncontiguous_labels_round_trip():
    y_noncontig = Y_SUP * 3 + 2  # {0, 1, 2} -> {2, 5, 8}
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_SUP, labels=y_noncontig)
    assert torch.equal(gmm.classes_.cpu(), torch.tensor([2, 5, 8]))
    preds = gmm.predict(X_SUP)
    mapped = gmm.classes_.cpu()[preds.cpu()]
    # Well-separated synthetic classes: predictions should match original labels almost always.
    assert (mapped == y_noncontig).float().mean() > 0.95


def test_supervised_fit_wrong_n_components_raises():
    gmm = GaussianMixture(n_components=2)  # only 2 slots for 3 distinct labels
    with pytest.raises(ValueError, match="n_components"):
        gmm.fit(X_SUP, labels=Y_SUP)


def test_supervised_fit_wrong_labels_length_raises():
    gmm = GaussianMixture(n_components=3)
    with pytest.raises(ValueError, match="n_samples"):
        gmm.fit(X_SUP, labels=Y_SUP[:-1])


def test_supervised_fit_with_mean_prior_shrinks_estimate():
    """A mean prior pulled toward zero should measurably shrink the supervised
    per-class mean estimate compared to the unregularized MLE, confirming
    _m_step's prior handling is actually exercised by the supervised path."""
    gmm_mle = GaussianMixture(n_components=3)
    gmm_mle.fit(X_SUP, labels=Y_SUP)

    gmm_map = GaussianMixture(
        n_components=3,
        mean_prior=torch.zeros(3),
        mean_precision_prior=5.0,
    )
    gmm_map.fit(X_SUP, labels=Y_SUP)

    mle_norm = gmm_mle.means_.norm(dim=1)
    map_norm = gmm_map.means_.norm(dim=1)
    assert torch.all(map_norm < mle_norm)


def test_unsupervised_fit_unaffected_by_labels_param_default():
    """labels=None (the default) must be unchanged from before this feature existed."""
    gmm_a = GaussianMixture(n_components=3, random_state=0)
    gmm_a.fit(X_3D)
    gmm_b = GaussianMixture(n_components=3, random_state=0)
    gmm_b.fit(X_3D, labels=None)
    assert torch.allclose(gmm_a.means_.cpu(), gmm_b.means_.cpu())
    assert torch.allclose(gmm_a.weights_.cpu(), gmm_b.weights_.cpu())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate
python -m pytest tests/test_gmm.py -k "supervised" -v
```

Expected: every `test_supervised_fit_*` FAILs with `TypeError: fit() got an unexpected keyword argument 'labels'` (the `test_unsupervised_fit_unaffected_by_labels_param_default` test fails the same way). This confirms the tests actually exercise not-yet-written code.

- [ ] **Step 3: Implement `_validate_labels` and the supervised `fit` branch**

In `tgmm/gmm.py`, add the new method directly above `_c_step` (currently at line 983):

```python
    def _validate_labels(self, labels, X: torch.Tensor) -> torch.Tensor:
        r"""
        Validate and map user-provided class labels to contiguous component
        indices for supervised fitting.

        Parameters
        ----------
        labels : array-like
            Per-sample class labels, shape (n_samples,). Values do not need
            to already be contiguous or zero-indexed; the mapping used is
            recorded in `self.classes_`.
        X : torch.Tensor
            The data being fit, used only to check `labels` has a matching
            length.

        Returns
        -------
        component_idx : torch.Tensor
            Long tensor of shape (n_samples,) with values in
            [0, n_components), where component `k` corresponds to the
            original label `self.classes_[k]`.

        Raises
        ------
        ValueError
            If `labels` isn't 1D of length `n_samples`, or if the number of
            distinct labels doesn't equal `self.n_components`.
        """
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        labels = labels.to(device=self.device)

        if labels.dim() != 1 or labels.size(0) != X.size(0):
            raise ValueError(
                f"labels must be a 1D array of length n_samples={X.size(0)}, "
                f"got shape {tuple(labels.shape)}."
            )

        classes = torch.unique(labels, sorted=True)
        if classes.numel() != self.n_components:
            raise ValueError(
                f"Supervised fit requires exactly one Gaussian component per "
                f"distinct label: found {classes.numel()} unique label(s) in "
                f"`labels` but n_components={self.n_components}."
            )

        self.classes_ = classes
        return torch.searchsorted(classes, labels).long()
```

Then in `fit` (`tgmm/gmm.py`), change the signature at line 714-721 from:

```python
    def fit(
        self,
        X: torch.Tensor,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        random_state: Optional[int] = None,
        warm_start: Optional[bool] = None
    ) -> "GaussianMixture":
```

to:

```python
    def fit(
        self,
        X: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        random_state: Optional[int] = None,
        warm_start: Optional[bool] = None
    ) -> "GaussianMixture":
```

Add to the docstring's `Parameters` section (right after the existing `X` entry, before `max_iter`):

```
        labels : torch.Tensor or array-like, optional
            Per-sample ground-truth class labels, shape (n_samples,). If
            given, `fit` performs **supervised fitting**: one Gaussian
            component is fit per distinct label directly from that label's
            data (the classification-likelihood limit of CEM where the
            classification step uses the true label instead of
            ``argmax(resp)`` — see the "Supervised Fitting" section of the
            user guide). Requires ``n_components`` to equal the number of
            distinct values in `labels`. Label values don't need to be
            contiguous or zero-indexed (see `classes_`). Bypasses the
            EM/CEM loop and `n_init` entirely: the fit is a single M-step
            and is deterministic, since the assignment never depends on the
            current parameter estimates. (default: None)
```

Add to the docstring's `Raises` section (find it below `Returns` in the same docstring):

```
        ValueError
            If `labels` is given and its length doesn't match `X`, or its
            number of distinct values doesn't equal `n_components`.
```

Then insert the supervised branch immediately after the existing feature-dimension check (currently the last lines of data prep, `tgmm/gmm.py:800-804`:
```python
        # Validate feature dimension
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {self.n_features}."
            )
```
) and before the `# 4. Run multiple initializations (if n_init > 1)` comment:

```python
        # ===============================================================
        # 3b. Supervised fit: bypass EM/CEM entirely when labels are given
        # ===============================================================
        if labels is not None:
            component_idx = self._validate_labels(labels, X)
            self._allocate_parameters(X)

            n_samples = X.size(0)
            resp = torch.zeros(n_samples, self.n_components, device=self.device, dtype=self.dtype)
            resp[torch.arange(n_samples, device=self.device), component_idx] = 1.0

            self._m_step(X, resp)

            _, log_prob_norm = self._e_step(X)
            self.lower_bound_ = log_prob_norm.mean().item()
            self.converged_ = True
            self.n_iter_ = 1
            self.best_random_state_ = None
            self.fitted_ = True

            if torch.any(self.weights_ < 1e-8):
                warnings.warn(
                    "Some class(es) have near-zero weight after supervised "
                    "fitting (very few samples for that label). This may "
                    "indicate a degenerate component.",
                    UserWarning
                )

            return self

```

Finally, add `classes_` to the class docstring's `Attributes` section (`tgmm/gmm.py`, in the block starting `Attributes\n    ----------` around line 96), right after `best_random_state_`:

```
    classes_ : torch.Tensor
        Sorted distinct label values seen by the most recent supervised
        `fit(X, labels=...)` call; `classes_[k]` is the original label for
        component `k`. Only set after a supervised fit — absent otherwise.
        Not persisted by `save`/`load` or `save_state_dict`/`load_state_dict`.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
python -m pytest tests/test_gmm.py -v
```

Expected: all tests pass, including every `test_supervised_fit_*` and the pre-existing 241 tests (no regressions).

- [ ] **Step 5: Commit**

```bash
git add tgmm/gmm.py tests/test_gmm.py
git commit -m "$(cat <<'EOF'
Add supervised GMM fitting via labels

fit(X, labels=...) fits one Gaussian component per distinct label
directly from that label's data, bypassing the EM/CEM loop -- the
classification-likelihood limit of CEM where the C-step uses the true
label instead of argmax(resp). Reuses _m_step unmodified; a single call
is already the fixed point since the label-derived assignment never
depends on the current parameter estimates.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: User-guide documentation

**Files:**
- Modify: `docs/user-guide/gaussian-mixture.md:214-228` (insert new section between the existing `## Classification EM (CEM)` section and `## Convergence Control`)

**Interfaces:**
- Consumes: `GaussianMixture.fit(X, labels=...)` and `self.classes_` from Task 1 (must be complete and merged first — this task documents the exact signature/behavior Task 1 implements).

- [ ] **Step 1: Write the new section**

In `docs/user-guide/gaussian-mixture.md`, insert immediately after the existing CEM section's closing line (`CEM tends to converge faster but may be more prone to local minima.`, currently line 227) and before `## Convergence Control` (currently line 229):

```markdown

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

See `notebooks/supervised_gmm.ipynb` for a worked comparison of plain EM,
CEM, and supervised fitting on the same synthetic data.
```

- [ ] **Step 2: Verify the doc site builds**

```bash
source .venv/bin/activate
python -m mkdocs build --strict 2>&1 | tail -30
```

Expected: build succeeds with no new warnings/errors attributable to `docs/user-guide/gaussian-mixture.md` (pre-existing unrelated warnings, if any, are not this task's concern — only confirm nothing new appears about this file).

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/gaussian-mixture.md
git commit -m "$(cat <<'EOF'
Document supervised GMM fitting in the user guide

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Comparison notebook (`notebooks/supervised_gmm.ipynb`)

**Files:**
- Create: `notebooks/supervised_gmm.ipynb`

**Interfaces:**
- Consumes: `GaussianMixture.fit(X, labels=...)` / `.classes_` (Task 1), `generate_gmm_data` (`tgmm/synthetic_data.py`), `plot_gmm`, `dynamic_figsize`, `match_predicted_to_true_labels` (`tgmm/plotting.py`), `ClusteringMetrics.adjusted_rand_score` / `.purity_score` / `.classification_report` (`tgmm/metrics.py`).

- [ ] **Step 1: Create the notebook**

Create `notebooks/supervised_gmm.ipynb` as nbformat 4 JSON (mirror `notebooks/cem.ipynb`'s cell/metadata structure) with these cells, in order:

1. **Markdown:**
```markdown
# Supervised GMM vs. EM vs. CEM

This notebook compares three ways of fitting a `GaussianMixture` on the
same synthetic data:

- **EM**: standard soft-assignment fitting, no label information.
- **CEM**: Classification EM — hard assignment via `argmax(resp)` each
  iteration, no label information.
- **Supervised**: `fit(X, labels=y)` — the assignment comes from the true
  label instead of `argmax(resp)`, converging in a single M-step (see the
  "Supervised Fitting" section of the user guide, and
  `docs/superpowers/specs/2026-08-10-supervised-gmm-design.md` for the
  literature this is grounded in: Celeux & Govaert 1992, McLachlan & Peel
  2000, Fraley & Raftery 2002).

The data reuses 3 of the 4 Gaussian components from `notebooks/gmm.ipynb`
(same centers/covariances/sample counts), reduced to 3 classes so the
comparison lines up directly with 3-way supervised labels.
```

2. **Code** (setup — same pattern as `cem.ipynb`):
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

import os
os.chdir('../')

import tgmm
import importlib
importlib.reload(tgmm)

from tgmm import (
    GaussianMixture, plot_gmm, dynamic_figsize, generate_gmm_data,
    match_predicted_to_true_labels, ClusteringMetrics,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random_state = 42
np.random.seed(random_state)
torch.manual_seed(random_state)
print('Using device:', device)
```

3. **Markdown:**
```markdown
## Data: 3 of the 4 Gaussians from `gmm.ipynb`

Same centers/covariances/sample counts as `notebooks/gmm.ipynb`, dropping
the `(2, -2)` component and keeping the other three (spherical, diagonal,
and full covariance, moderately overlapping).
```

4. **Code:**
```python
centers = [np.array([0, 2]), np.array([0, 0]), np.array([2, 2])]
covs = [
    1.0 * np.eye(2),                    # spherical covariance
    np.array([[2, 0], [0, 0.5]]),       # diagonal covariance
    np.array([[0.2, 0.5], [0.5, 2]]),   # full covariance
]
n_samples = [800, 1000, 1000]
legend_labels = [f'Class {i}' for i in range(len(n_samples))]

X_tensor, y_tensor = generate_gmm_data(centers, covs, n_samples, device=device, random_state=random_state)
X = X_tensor.cpu().numpy()
y_true = y_tensor.cpu().numpy()
n_components = len(n_samples)

plot_gmm(X=X, true_labels=y_true, title='True Classes', legend_labels=legend_labels,
         show_ellipses=False, show_means=False)
plt.show()
```

5. **Markdown:**
```markdown
## Fitting all three models

EM and CEM don't see `y_tensor` at all; only the supervised fit does.
```

6. **Code:**
```python
gmm_em = GaussianMixture(n_components=n_components, init_means='random', n_init=10, random_state=random_state)
t0 = time.time()
gmm_em.fit(X_tensor)
time_em = time.time() - t0

gmm_cem = GaussianMixture(n_components=n_components, init_means='random', n_init=10, random_state=random_state, cem=True)
t0 = time.time()
gmm_cem.fit(X_tensor)
time_cem = time.time() - t0

gmm_sup = GaussianMixture(n_components=n_components)
t0 = time.time()
gmm_sup.fit(X_tensor, labels=y_tensor)
time_sup = time.time() - t0

print(f"EM:         iters={gmm_em.n_iter_:4d}  converged={gmm_em.converged_}  time={time_em:.4f}s  mean log-lik={gmm_em.score(X_tensor):.4f}")
print(f"CEM:        iters={gmm_cem.n_iter_:4d}  converged={gmm_cem.converged_}  time={time_cem:.4f}s  mean log-lik={gmm_cem.score(X_tensor):.4f}")
print(f"Supervised: iters={gmm_sup.n_iter_:4d}  converged={gmm_sup.converged_}  time={time_sup:.4f}s  mean log-lik={gmm_sup.score(X_tensor):.4f}")
print(f"\nSupervised gmm.classes_: {gmm_sup.classes_.cpu().tolist()}")
```

7. **Markdown:**
```markdown
## Visual comparison

EM/CEM cluster indices are arbitrary, so they're remapped to the true
labels via the Hungarian algorithm (`match_predicted_to_true_labels`)
purely for coloring — the supervised fit needs no remapping since its
component index already *is* the label (`classes_ == [0, 1, 2]`).
```

8. **Code:**
```python
pred_em = match_predicted_to_true_labels(y_tensor, gmm_em.predict(X_tensor))
pred_cem = match_predicted_to_true_labels(y_tensor, gmm_cem.predict(X_tensor))
pred_sup = gmm_sup.predict(X_tensor)

figsize = dynamic_figsize(1, 4)
fig, axes = plt.subplots(1, 4, figsize=figsize)
fig.suptitle('True labels vs. EM vs. CEM vs. Supervised')

plot_gmm(X=X, true_labels=y_true, ax=axes[0], title='True Classes',
         show_ellipses=False, show_means=False)
plot_gmm(X=X, gmm=gmm_em, ax=axes[1], title=f'EM\n({gmm_em.n_iter_} iters)',
         color_by_cluster=True, match_labels_to_true=True, true_labels=y_true,
         ellipse_std_devs=[3], ellipse_fill=True, ellipse_alpha=0.1)
plot_gmm(X=X, gmm=gmm_cem, ax=axes[2], title=f'CEM\n({gmm_cem.n_iter_} iters)',
         color_by_cluster=True, match_labels_to_true=True, true_labels=y_true,
         ellipse_std_devs=[3], ellipse_fill=True, ellipse_alpha=0.1)
plot_gmm(X=X, gmm=gmm_sup, ax=axes[3], title='Supervised\n(1 iter)',
         color_by_cluster=True, match_labels_to_true=True, true_labels=y_true,
         ellipse_std_devs=[3], ellipse_fill=True, ellipse_alpha=0.1)

plt.tight_layout()
plt.show()
```

9. **Markdown:**
```markdown
## Quantitative comparison

`adjusted_rand_score` and `purity_score` are permutation-invariant (they
don't need the Hungarian remap), so they're computed on the raw
predictions; `classification_report` needs matching label spaces, so it
uses the remapped EM/CEM predictions and the supervised model's raw
predictions (already label-aligned).
```

10. **Code:**
```python
raw_pred_em = gmm_em.predict(X_tensor)
raw_pred_cem = gmm_cem.predict(X_tensor)

results = {
    'EM': (raw_pred_em, pred_em),
    'CEM': (raw_pred_cem, pred_cem),
    'Supervised': (pred_sup, pred_sup),
}

print(f"{'Model':<12} {'ARI':>8} {'Purity':>8} {'Mean log-lik':>14} {'Iters':>7}")
for name, (raw_pred, aligned_pred) in results.items():
    ari = ClusteringMetrics.adjusted_rand_score(y_tensor, raw_pred)
    purity = ClusteringMetrics.purity_score(y_tensor, raw_pred)
    model = {'EM': gmm_em, 'CEM': gmm_cem, 'Supervised': gmm_sup}[name]
    print(f"{name:<12} {ari:8.4f} {purity:8.4f} {model.score(X_tensor):14.4f} {model.n_iter_:7d}")

print("\nPer-class report (precision / recall / f1), supervised model:")
report_sup = ClusteringMetrics.classification_report(y_tensor, pred_sup)
for label, metrics in sorted(report_sup.items()):
    print(f"  class {label}: precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1-score']:.3f} support={metrics['support']}")
```

11. **Markdown:**
```markdown
## Takeaways

- **Supervised fitting converges in exactly 1 iteration**, regardless of
  how much the classes overlap — because the assignment is fixed by the
  labels rather than estimated, `fit` is doing a single closed-form
  per-class MLE (see the design doc for why this is mathematically the
  fixed point of CEM with a label-driven C-step, not an approximation of
  it).
- **EM and CEM need the labels remapped** to compare against ground truth
  (arbitrary cluster indexing); the supervised fit doesn't, since
  `classes_` ties each component directly to its original label.
- On this moderately-overlapping 3-class data, expect supervised fitting
  to log-likelihood-match or exceed CEM (which itself approximates the
  true partition via `argmax(resp)`) since it uses the *exact* partition
  instead of an estimated one — the gap between CEM and supervised is a
  direct measure of how many points CEM's E/C-step assignment actually
  got wrong relative to ground truth.
```

- [ ] **Step 2: Execute the notebook headlessly to verify it runs end-to-end**

```bash
source .venv/bin/activate
cd notebooks
jupyter nbconvert --to notebook --execute --inplace supervised_gmm.ipynb
cd ..
```

Expected: exits 0, no exceptions. Then spot-check the executed notebook's outputs are non-empty (the printed metrics table has 3 rows, the 4-panel figure rendered) — open `notebooks/supervised_gmm.ipynb` and confirm each code cell has an `outputs` array populated (non-empty) after execution, in particular that the two `print(...)` cells (Step 6 and Step 10 above) show real numbers rather than an empty output list, which would indicate the cell silently produced nothing.

- [ ] **Step 3: Commit**

```bash
git add notebooks/supervised_gmm.ipynb
git commit -m "$(cat <<'EOF'
Add notebook comparing EM, CEM, and supervised GMM fitting

Reuses 3 of the 4 Gaussians from notebooks/gmm.ipynb, fits all three
variants on the same data, and compares them visually (matched-label
ellipses) and quantitatively (ARI, purity, per-class precision/recall,
log-likelihood, iteration count).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
