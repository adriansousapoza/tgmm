r"""
Positive-unlabeled (PU) learning on top of Phase 5.1's semi-supervised
`fit(X, labels=...)`. See
docs/superpowers/specs/2026-10-14-phase5-semi-supervised-pu-design.md.
"""
from typing import Optional

import torch

from .gmm import GaussianMixture
from .mixtures.student_t import StudentTMixture
from .mixtures.vmf import VonMisesFisherMixture

_VALID_FAMILIES = ('gaussian', 'student_t', 'vmf')
_VALID_ANCHORS = ('fixed', 'prior', 'free')
_FAMILY_CLASSES = {'gaussian': GaussianMixture, 'student_t': StudentTMixture, 'vmf': VonMisesFisherMixture}


def _assign_slice(model, snapshot: dict, idx):
    for key, value in snapshot.items():
        getattr(model, key)[idx] = value


def _niw_blend_positive_block(model, X: torch.Tensor, resp_pos: torch.Tensor, idx, mu0, psi0, lambda0: float, nu0: float):
    r"""
    NIW posterior update (same formulas as
    `GaussianMixture._update_niw_full`/`_diag`/`_spherical`, re-derived
    here since those write the model's *entire* `means_`/`covariances_`
    rather than a slice), applied only to the positive block's `idx`
    components, using `mu0`/`psi0` (the labeled-only fit, with
    `psi0 = nu0 * cov0`) as the prior and `lambda0 = nu0 = n_labeled`.
    """
    stats = model._compute_stats(X, resp_pos)
    nk = stats.nk + 1e-20
    ybar = stats.s1 / nk.unsqueeze(1)
    lambda_n = lambda0 + nk
    nu_n = nu0 + nk
    mu_n = (lambda0 * mu0 + nk.unsqueeze(1) * ybar) / lambda_n.unsqueeze(1)

    cross_term_coeff = (lambda0 * nk) / lambda_n
    if model.covariance_type == 'full':
        mean_diff = ybar - mu0
        cross = cross_term_coeff.view(-1, 1, 1) * torch.einsum('kd,ke->kde', mean_diff, mean_diff)
        psi_n = psi0 + stats.s2 + cross
        psi_n = psi_n + model.reg_covar * torch.eye(model.n_features, device=model.device, dtype=model.dtype).unsqueeze(0)
        cov_n = psi_n / nu_n.view(-1, 1, 1)
    elif model.covariance_type == 'diag':
        mean_diff_sq = (ybar - mu0).pow(2)
        cross = cross_term_coeff.unsqueeze(-1) * mean_diff_sq
        psi_n = psi0 + stats.s2 + cross + model.reg_covar
        cov_n = psi_n / nu_n.unsqueeze(-1)
    elif model.covariance_type == 'spherical':
        mean_diff_norm_sq = (ybar - mu0).pow(2).sum(dim=1)
        cross = cross_term_coeff * mean_diff_norm_sq
        psi_n = psi0 + stats.s2 + cross + model.reg_covar * model.n_features
        cov_n = psi_n / (nu_n * model.n_features)
    else:
        raise ValueError(
            f"anchor='prior' supports covariance_type in ('full', 'diag', 'spherical'), "
            f"got {model.covariance_type!r}."
        )

    model.means_[idx] = mu_n
    model.covariances_[idx] = cov_n


class PositiveUnlabeledMixture:
    r"""
    Positive-unlabeled (PU) learning (Elkan & Noto, 2008) via a mixture
    model. A subset of rows is known to be the positive class (e.g.
    confirmed cancer cells); every other row is unlabeled and may be
    positive or negative. Models the unlabeled density as
    :math:`\pi f_P(x) + (1-\pi) f_N(x)` and jointly estimates :math:`\pi`
    (`class_prior_`) and both densities, under the SCAR (Selected
    Completely At Random) assumption: labeled positives are a uniform
    random subsample of *all* true positives, not a biased one.

    A thin wrapper over Phase 5.1's semi-supervised `fit(X, labels=...)`:
    builds the chosen `family`'s mixture with
    `components_per_class={1: n_components_positive, 0:
    n_components_negative}`, labels `1` on `positive_mask` and `-1`
    (unlabeled) elsewhere, and fits it through that path.

    Parameters
    ----------
    n_components_positive : int, default=1
        Number of components in the positive block.
    n_components_negative : int, default=5
        Number of components in the negative block. `None` (an unbounded
        Gibbs sampler for the negative block with the positive block held
        fixed) is out of scope for this release.
    family : {'gaussian', 'student_t', 'vmf'}, default='gaussian'
        Which mixture family to use for both blocks.
    anchor : {'fixed', 'prior', 'free'}, default='prior'
        How strongly the positive block stays tied to the labeled
        positives -- needed because :math:`\pi` is identifiable only if
        the negative block can't absorb the positive one.

        - `'fixed'`: fit the positive block once on the labeled positives
          and freeze it completely; semi-supervised EM then estimates only
          :math:`\pi` and the negative block (classical mixture-
          proportion estimation). Works for every `family`.
        - `'prior'` (default): fit the positive block once on the labeled
          positives, then use that fit as an NIW prior
          (`mean_precision_prior = degrees_of_freedom_prior = n_labeled`)
          so unlabeled positives can still refine it each M-step without
          drifting toward the negatives. **Gaussian-only** (`covariance_type`
          in `'full'`/`'diag'`/`'spherical'`) -- NIW conjugate priors
          aren't a concept for `'student_t'`/`'vmf'`.
        - `'free'`: the positive block is initialized from the labeled-
          only fit, then plain semi-supervised EM, no further special
          treatment. Works for every `family`. Can let the positive block
          drift toward the negatives on strongly overlapping data --
          documented, not guarded against.
    **family_kwargs
        Forwarded to the underlying mixture's constructor (e.g.
        `covariance_type='diag'`, `random_state=0`, `max_iter=200`).
        Must not include `n_components` or `components_per_class` (set
        internally).

    Attributes
    ----------
    class_prior_ : float or None
        Estimated fraction of *unlabeled* rows that are positive: the
        mean, over unlabeled rows only, of their total responsibility on
        the positive block. `None` before `fit`.
    model_ : BaseMixture or None
        The underlying fitted mixture. `None` before `fit`. Any attribute
        or method not defined on `PositiveUnlabeledMixture` itself (e.g.
        `bic`, `score`, `sample`) is forwarded to this.
    """

    def __init__(
        self,
        n_components_positive: int = 1,
        n_components_negative: Optional[int] = 5,
        family: str = 'gaussian',
        anchor: str = 'prior',
        **family_kwargs,
    ):
        if n_components_negative is None:
            raise NotImplementedError(
                "n_components_negative=None (an unbounded Gibbs sampler for the "
                "negative block, with the positive block held fixed) is out of "
                "scope for this release -- see Phase 5.3 in "
                "docs/superpowers/specs/2026-10-14-phase5-semi-supervised-pu-design.md."
            )
        if n_components_positive <= 0 or n_components_negative <= 0:
            raise ValueError("n_components_positive and n_components_negative must both be > 0.")
        if family not in _VALID_FAMILIES:
            raise ValueError(f"family must be one of {_VALID_FAMILIES}, got {family!r}")
        if anchor not in _VALID_ANCHORS:
            raise ValueError(f"anchor must be one of {_VALID_ANCHORS}, got {anchor!r}")
        if anchor == 'prior' and family != 'gaussian':
            raise ValueError(
                "anchor='prior' requires family='gaussian' -- NIW conjugate priors "
                "are a GaussianMixture-specific mechanism, not available for "
                "'student_t'/'vmf'. Use anchor='fixed' or anchor='free' instead."
            )
        for forbidden in ('n_components', 'components_per_class'):
            if forbidden in family_kwargs:
                raise ValueError(
                    f"{forbidden} is set internally by PositiveUnlabeledMixture "
                    f"(from n_components_positive/n_components_negative); do not pass it."
                )

        self.n_components_positive = n_components_positive
        self.n_components_negative = n_components_negative
        self.family = family
        self.anchor = anchor
        self.family_kwargs = family_kwargs

        self.class_prior_ = None
        self.model_ = None
        self._positive_idx = None

    def fit(self, X: torch.Tensor, positive_mask) -> "PositiveUnlabeledMixture":
        r"""
        Fit on `X` with `positive_mask` indicating the labeled positives.

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
        positive_mask : array-like of bool, shape (n_samples,)
            `True` for rows known to be positive; `False` for unlabeled
            rows (which may be positive or negative).

        Returns
        -------
        self
        """
        if not torch.is_tensor(positive_mask):
            positive_mask = torch.as_tensor(positive_mask)
        positive_mask = positive_mask.bool()
        if positive_mask.dim() != 1 or positive_mask.size(0) != X.size(0):
            raise ValueError(
                f"positive_mask must be a 1D boolean array of length n_samples={X.size(0)}, "
                f"got shape {tuple(positive_mask.shape)}."
            )
        n_labeled = int(positive_mask.sum().item())
        if n_labeled == 0:
            raise ValueError("positive_mask has no True entries -- at least one labeled positive is required.")

        n_total = self.n_components_positive + self.n_components_negative
        model = _FAMILY_CLASSES[self.family](
            n_components=n_total,
            components_per_class={1: self.n_components_positive, 0: self.n_components_negative},
            **self.family_kwargs,
        )

        # classes_ will resolve to [0, 1] (sorted): the negative block
        # (n_components_negative components) comes first, the positive
        # block (n_components_positive components) last.
        positive_idx = slice(self.n_components_negative, n_total)

        labels = torch.where(
            positive_mask,
            torch.ones(X.shape[0], dtype=torch.long, device=positive_mask.device),
            torch.full((X.shape[0],), -1, dtype=torch.long, device=positive_mask.device),
        )

        X_on_device = X.to(model.device)
        if model.n_features is None:
            model.n_features = X_on_device.shape[1]
        model.dtype = X_on_device.dtype
        positive_snapshot = model._fit_component_block(
            X_on_device[positive_mask.to(model.device)], self.n_components_positive, random_state=model.random_state,
        )

        if self.anchor == 'fixed':
            def hook(X_full, resp):
                _assign_slice(model, positive_snapshot, positive_idx)
        elif self.anchor == 'free':
            def hook(X_full, resp):
                if resp is None:
                    _assign_slice(model, positive_snapshot, positive_idx)
        else:  # 'prior'
            mu0 = positive_snapshot['means_']
            cov0 = positive_snapshot['covariances_']
            lambda0 = nu0 = float(n_labeled)
            psi0 = nu0 * cov0

            def hook(X_full, resp):
                if resp is None:
                    model.means_[positive_idx] = mu0
                    model.covariances_[positive_idx] = cov0
                    return
                _niw_blend_positive_block(
                    model, X_full, resp[:, positive_idx], positive_idx, mu0, psi0, lambda0, nu0,
                )

        model._post_mstep_hook = hook
        model.fit(X, labels=labels)
        model._post_mstep_hook = None

        self.model_ = model
        self._positive_idx = positive_idx

        resp = model.predict_proba(X)
        unlabeled_resp = resp[~positive_mask.to(model.device)]
        self.class_prior_ = unlabeled_resp[:, positive_idx].sum(dim=1).mean().item()

        return self

    def predict_proba_positive(self, X: torch.Tensor) -> torch.Tensor:
        r"""Total responsibility on the positive block for each row of `X`."""
        if self.model_ is None:
            raise RuntimeError("PositiveUnlabeledMixture is not fitted yet; call fit(X, positive_mask) first.")
        resp = self.model_.predict_proba(X)
        return resp[:, self._positive_idx].sum(dim=1)

    def predict_positive(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        r"""Hard positive/negative call: `predict_proba_positive(X) >= threshold`."""
        return self.predict_proba_positive(X) >= threshold

    def __getattr__(self, name):
        model = self.__dict__.get('model_')
        if model is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")
        return getattr(model, name)
