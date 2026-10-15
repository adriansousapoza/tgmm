r"""
The component protocol behind `tgmm.GenericMixture`.

A mixture component is, from the point of view of EM, three operations: evaluate its log-density,
*summarise* a weighted chunk of data, and *update its parameters from the summaries*. This module
defines that interface as a `typing.Protocol` so that any object -- yours, or one adapted from another
library -- can be a component without inheriting from anything in tgmm.

The three-method shape (``log_probability`` / ``summarize`` / ``from_summaries``) follows the API of
pomegranate 1.x (https://pomegranate.readthedocs.io, MIT licence), whose distributions are designed
exactly for out-of-core EM; the design is credited to that project. tgmm does not import pomegranate.

See docs/superpowers/specs/2026-10-15-v05-component-protocol-design.md.
"""
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class Component(Protocol):
    r"""
    What `GenericMixture` needs from a mixture component.

    Required methods
    ----------------
    log_probability(X) -> Tensor of shape (n,)
        :math:`\log f_k(x_i)` for every row of `X`.
    summarize(X, sample_weight) -> None
        Accumulate the weighted sufficient statistics of one chunk: `sample_weight` has shape `(n,)`
        and holds the (row-weighted) responsibility of this component for each row. It is called once
        per row chunk (`batch_size` splits an M-step into several chunks), so it must *add to* what
        it already holds.
    from_summaries() -> None
        Update the parameters from everything summarised since the last call, then reset the
        accumulators. A component that received (almost) no weight should keep its parameters.

    Optional members (looked up with ``getattr``; not part of the runtime check)
    ----------------------------------------------------------------------------
    sample(n) -> Tensor of shape (n, d)
        Needed by `GenericMixture.sample`.
    n_parameters() -> int
        Number of free scalar parameters, used by `bic`/`aic`/`icl`. Without it, the number of scalar
        entries of the tensors held in the object's public attributes is used (see
        `component_n_parameters`).
    frozen : bool
        If truthy, the component's parameters are never updated.
    reseed(x) -> None
        Called during SEM with the worst-fit data point when the component's sampled count drops
        below one.
    """

    def log_probability(self, X: torch.Tensor) -> torch.Tensor: ...

    def summarize(self, X: torch.Tensor, sample_weight: torch.Tensor) -> None: ...

    def from_summaries(self) -> None: ...


def component_is_frozen(component) -> bool:
    r"""Whether `component` declares itself frozen (``component.frozen`` is truthy)."""
    return bool(getattr(component, 'frozen', False))


def component_n_parameters(component) -> int:
    r"""
    Number of free parameters of a component: ``component.n_parameters()`` if it exists, else the
    number of scalar entries of the tensors held in its public attributes (names that do not start
    with ``_``; for an ``nn.Module`` attribute, its parameters that require grad). Implement
    ``n_parameters()`` whenever the object stores anything else as a public tensor.
    """
    method = getattr(component, 'n_parameters', None)
    if callable(method):
        return int(method())
    total = 0
    for name, value in vars(component).items():
        if name.startswith('_'):
            continue
        if isinstance(value, nn.Module):
            total += sum(p.numel() for p in value.parameters() if p.requires_grad)
        elif torch.is_tensor(value):
            total += int(value.numel())
    return total
