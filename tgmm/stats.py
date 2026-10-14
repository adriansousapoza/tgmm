"""
Sufficient statistics for the Gaussian mixture M-step.

See docs/superpowers/specs/2026-10-14-phase0-sufficient-stats-design.md for
the design rationale -- in particular why `s2` is always centered on each
component's own empirical mean for the current E-step (`s1 / nk`), and how
that choice makes chunked accumulation exact.
"""
from dataclasses import dataclass

import torch


@dataclass
class SufficientStats:
    r"""
    Accumulated sufficient statistics for one (chunk of a) Gaussian mixture
    E-step, per component :math:`k`.

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
    """

    nk: torch.Tensor
    s1: torch.Tensor
    s2: torch.Tensor
    n_samples: int

    def __add__(self, other: "SufficientStats") -> "SufficientStats":
        if not isinstance(other, SufficientStats):
            return NotImplemented
        return SufficientStats(
            nk=self.nk + other.nk,
            s1=self.s1 + other.s1,
            s2=self.s2 + other.s2,
            n_samples=self.n_samples + other.n_samples,
        )

    def scale(self, factor: float) -> "SufficientStats":
        r"""
        Scale ``nk``/``s1``/``s2`` by ``factor``; ``n_samples`` is left
        untouched since it tracks how many physical rows contributed, not
        a quantity meaningful to reweight. Used by online/mini-batch EM
        (Phase 2) to blend a running accumulator with a new batch's
        statistics.
        """
        return SufficientStats(
            nk=self.nk * factor,
            s1=self.s1 * factor,
            s2=self.s2 * factor,
            n_samples=self.n_samples,
        )
