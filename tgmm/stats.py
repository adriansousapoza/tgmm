"""
Sufficient statistics for the Gaussian mixture M-step.

See docs/superpowers/specs/2026-10-14-phase0-sufficient-stats-design.md for
the design rationale -- in particular why `s2` is always centered on each
component's own empirical mean for the current E-step (`s1 / nk`), and how
that choice makes chunked accumulation exact.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class SufficientStats:
    r"""
    Accumulated sufficient statistics for one (chunk of a) mixture E-step,
    per component :math:`k`.

    Attributes
    ----------
    nk : torch.Tensor, shape (n_components,)
        :math:`n_k = \sum_i r_{ik}`.
    s1 : torch.Tensor, shape (n_components, n_features)
        :math:`\sum_i r_{ik} x_i`.
    s2 : torch.Tensor
        Covariance-type-specific second moment, centered on each
        component's empirical mean :math:`\bar y_k = s1_k / n_k` for
        *this* accumulation (not on whatever the model's current
        ``means_`` holds): ``full`` -> ``(n_components, n_features,
        n_features)`` = :math:`\sum_i r_{ik} (x_i - \bar y_k)(x_i - \bar
        y_k)^T`; ``diag`` -> ``(n_components, n_features)``; ``spherical``
        -> ``(n_components,)``. Tied covariance types use the same shape as
        their non-tied base type -- the pooling across components happens
        in the M-step, not here.
    n_samples : int
        Number of rows this instance summarizes.
    aux : dict[str, torch.Tensor], optional
        Family-specific accumulators that don't fit the Gaussian-shaped
        ``nk``/``s1``/``s2`` fields above -- e.g. `StudentTMixture`'s
        :math:`u`-weighted sums. ``None`` (default) for every family that
        doesn't need it, in particular `GaussianMixture`, whose behavior is
        completely unaffected by this field's existence. ``__add__``/
        ``scale`` merge/scale it key-by-key when present on either side.
    """

    nk: torch.Tensor
    s1: torch.Tensor
    s2: torch.Tensor
    n_samples: int
    aux: Optional[Dict[str, torch.Tensor]] = None

    def __add__(self, other: "SufficientStats") -> "SufficientStats":
        if not isinstance(other, SufficientStats):
            return NotImplemented
        return SufficientStats(
            nk=self.nk + other.nk,
            s1=self.s1 + other.s1,
            s2=self.s2 + other.s2,
            n_samples=self.n_samples + other.n_samples,
            aux=self._merge_aux(other),
        )

    def scale(self, factor: float) -> "SufficientStats":
        r"""
        Scale ``nk``/``s1``/``s2``/``aux`` by ``factor``; ``n_samples`` is
        left untouched since it tracks how many physical rows contributed,
        not a quantity meaningful to reweight. Used by online/mini-batch EM
        (Phase 2) to blend a running accumulator with a new batch's
        statistics.
        """
        return SufficientStats(
            nk=self.nk * factor,
            s1=self.s1 * factor,
            s2=self.s2 * factor,
            n_samples=self.n_samples,
            aux=None if self.aux is None else {k: v * factor for k, v in self.aux.items()},
        )

    def _merge_aux(self, other: "SufficientStats") -> Optional[Dict[str, torch.Tensor]]:
        if self.aux is None and other.aux is None:
            return None
        self_aux = self.aux or {}
        other_aux = other.aux or {}
        keys = set(self_aux) | set(other_aux)
        merged = {}
        for key in keys:
            if key in self_aux and key in other_aux:
                merged[key] = self_aux[key] + other_aux[key]
            else:
                merged[key] = self_aux.get(key, other_aux.get(key))
        return merged
