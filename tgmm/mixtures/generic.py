r"""
`GenericMixture`: a mixture of user-supplied components.

Components are either objects that implement the `tgmm.mixtures.protocol.Component` protocol
(``log_probability`` / ``summarize`` / ``from_summaries``, the pomegranate 1.x API) or autograd
components (a ``torch.nn.Module`` with learnable parameters and a ``log_prob(X)`` method), whose
M-step is a generalised EM step on the expected complete-data log-likelihood
(`BaseMixture._gem_step`, accepted only on improvement).

See docs/superpowers/specs/2026-10-15-v05-component-protocol-design.md.
"""
import copy
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..stats import SufficientStats
from .base import BaseMixture
from .protocol import Component, component_is_frozen, component_n_parameters


class _AutogradComponent:
    r"""
    Adapter that makes an ``nn.Module`` with ``log_prob(X) -> (n,)`` a protocol component.

    ``summarize`` stores references to the chunk and its weights; ``from_summaries`` maximises
    :math:`Q(\theta)=\sum_i w_i \log f(x_i\mid\theta)/\sum_i w_i` with `BaseMixture._gem_step`
    (L-BFGS, accepted only if :math:`Q` increases), then drops the references. The wrapped module is
    available as ``.module``.
    """

    n_steps = 20

    def __init__(self, module: nn.Module):
        self.module = module
        self._X = []
        self._w = []

    @property
    def frozen(self) -> bool:
        return bool(getattr(self.module, 'frozen', False))

    def log_probability(self, X: torch.Tensor) -> torch.Tensor:
        return self.module.log_prob(X)

    def summarize(self, X: torch.Tensor, sample_weight: torch.Tensor) -> None:
        self._X.append(X)
        self._w.append(sample_weight)

    def from_summaries(self) -> None:
        if not self._X:
            return
        X = self._X[0] if len(self._X) == 1 else torch.cat(self._X)
        w = self._w[0] if len(self._w) == 1 else torch.cat(self._w)
        self._X, self._w = [], []
        total = w.sum()
        params = [p for p in self.module.parameters() if p.requires_grad]
        if not params or float(total) <= 1e-12:
            return

        def objective():
            return -(w * self.module.log_prob(X)).sum() / total

        BaseMixture._gem_step(objective, params, n_steps=self.n_steps)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)

    def sample(self, n: int) -> torch.Tensor:
        method = getattr(self.module, 'sample', None)
        if not callable(method):
            raise NotImplementedError(
                f"the wrapped {type(self.module).__name__} has no sample(n) method; "
                "GenericMixture.sample needs every component to provide one."
            )
        return method(n)


def _wrap_component(obj, index: int):
    r"""A protocol component as is, an ``nn.Module`` with ``log_prob`` through the adapter; else TypeError."""
    if isinstance(obj, Component):
        return obj
    if isinstance(obj, nn.Module) and callable(getattr(obj, 'log_prob', None)):
        return _AutogradComponent(obj)
    raise TypeError(
        f"components[{index}] ({type(obj).__name__}) is neither a protocol component (it needs the methods "
        "log_probability, summarize and from_summaries) nor a torch.nn.Module with a log_prob(X) method."
    )


class GenericMixture(BaseMixture):
    r"""
    Mixture :math:`p(x)=\sum_k \pi_k f_k(x)` of user-supplied components.

    Parameters
    ----------
    components : sequence of components
        The `K` components, which are also the **starting point** of the fit (there is no data-driven
        initialisation: tgmm cannot know how to build your component from data). Each is either

        * a protocol component (`tgmm.mixtures.protocol.Component`): the E-step stacks
          ``log_probability``; the M-step calls ``summarize(X_chunk, resp_chunk[:, k])`` for every row
          chunk and then ``from_summaries()`` once, so ``batch_size`` works unchanged; or
        * an ``nn.Module`` with learnable parameters and ``log_prob(X) -> (n,)``: its M-step maximises
          :math:`\sum_i r_{ik}\log f_k(x_i)` by L-BFGS and accepts the update only if the objective
          increases (a generalised EM step, so the log-likelihood stays monotone).

        The constructor keeps deep copies; every `fit` starts from fresh copies (with
        ``warm_start=True`` from the fitted `components_`).
    init_weights : {'uniform'} or torch.Tensor, optional
        Initial mixing weights. (default: 'uniform')
    n_init : int, optional
        Must be 1 (restarts would all begin from the same components). (default: 1)
    algorithm : {'em', 'cem', 'sem'}, optional
        `'saem'` and online EM (`step_size`) need additive, blendable statistics and raise
        `NotImplementedError`; `sem_estimate='mean'` raises `ValueError`. (default: 'em')
    noise, noise_volume, noise_prior, noise_init : see `BaseMixture`
        `noise=True` requires a float `noise_volume`.
    frozen_components : sequence of int, optional
        Components that are never updated (their mixing weights still are). A component whose
        ``frozen`` attribute is truthy is frozen too.
    max_iter, tol, random_state, warm_start, batch_size, temperature, convergence, sem_burn_in,
    components_per_class, unlabeled_weight, label_confidence, verbose, verbose_interval, device :
        as in `GaussianMixture`.

    Attributes
    ----------
    components_ : list
        The fitted components (for autograd components: adapters, with the module in ``.module``).
    weights_ : torch.Tensor, shape (n_components,)
    """

    def __init__(
        self,
        components: Sequence,
        n_features: Optional[int] = None,
        init_weights='uniform',
        max_iter: int = 1000,
        tol: float = 1e-4,
        n_init: int = 1,
        random_state: Optional[int] = None,
        warm_start: bool = False,
        algorithm: str = 'em',
        batch_size: Optional[int] = None,
        temperature: Optional[Tuple[float, int]] = None,
        convergence: str = 'relative',
        step_size=None,
        sem_burn_in: Optional[int] = None,
        sem_estimate: str = 'best',
        components_per_class=1,
        unlabeled_weight: float = 1.0,
        label_confidence: float = 1.0,
        noise: bool = False,
        noise_volume='bbox',
        noise_prior: float = 0.05,
        noise_init=None,
        missing: str = 'error',
        frozen_components: Sequence[int] = (),
        verbose: bool = False,
        verbose_interval: int = 10,
        device: Optional[str] = None,
    ):
        components = list(components)
        if len(components) == 0:
            raise ValueError("components must contain at least one component.")
        wrapped = [_wrap_component(c, i) for i, c in enumerate(components)]

        super().__init__(
            n_components=len(wrapped), n_features=n_features, max_iter=max_iter, tol=tol, n_init=n_init,
            init_means=None, init_weights=init_weights, random_state=random_state, warm_start=warm_start,
            algorithm=algorithm, batch_size=batch_size, temperature=temperature, convergence=convergence,
            step_size=step_size, sem_burn_in=sem_burn_in, sem_estimate=sem_estimate,
            components_per_class=components_per_class, unlabeled_weight=unlabeled_weight,
            label_confidence=label_confidence, noise=noise, noise_volume=noise_volume,
            noise_prior=noise_prior, noise_init=noise_init, missing=missing,
            frozen_components=frozen_components, verbose=verbose, verbose_interval=verbose_interval,
            device=device,
        )

        if n_init != 1:
            raise ValueError(
                f"GenericMixture supports only n_init=1 (got {n_init}): every restart would begin from the "
                "same user-supplied components. Build several models from differently initialised components."
            )
        if algorithm == 'sem' and sem_estimate == 'mean':
            raise ValueError(
                "sem_estimate='mean' averages component parameters, which is undefined for generic "
                "components; use sem_estimate='best'."
            )
        self._check_algorithm_supported()
        if self.noise and isinstance(self.noise_volume, str):
            raise NotImplementedError(
                "GenericMixture(noise=True) needs noise_volume=<float>: a generic density has no bounding box "
                "that could define the volume of the noise region."
            )
        if not (isinstance(init_weights, str) and init_weights == 'uniform') and not torch.is_tensor(init_weights):
            raise ValueError(f"init_weights must be 'uniform' or a tensor, got {init_weights!r}.")

        self._initial_components = copy.deepcopy(wrapped)
        self.components_ = None
        self.__dict__['_frozen_ready'] = True

    # ===================================================================
    # Guards
    # ===================================================================
    def _check_algorithm_supported(self):
        if self.algorithm == 'saem':
            raise NotImplementedError(
                "GenericMixture does not support algorithm='saem': stochastic approximation blends additive "
                "sufficient statistics, but summarize/from_summaries components keep opaque state. "
                "Use 'em', 'cem' or 'sem'."
            )
        if self.step_size is not None:
            raise NotImplementedError(
                "GenericMixture does not support online EM (step_size set): it needs blendable sufficient "
                "statistics. Use batch_size alone for memory-bounded exact EM."
            )

    def _fit_single_run(self, X, max_iter, tol, run_idx=0):
        self._check_algorithm_supported()
        return super()._fit_single_run(X, max_iter, tol, run_idx)

    def partial_fit(self, X_batch):
        raise NotImplementedError("GenericMixture does not support partial_fit (online EM); see fit(..., batch_size=...).")

    def _fit_component_block(self, X_block, n_block, random_state):
        raise NotImplementedError(
            "components_per_class > 1 with fully labelled data would refit blocks of a different size, which is "
            "not possible for a fixed list of user-supplied components; use partially labelled data (some -1 "
            "labels) or components_per_class=1."
        )

    # ===================================================================
    # BaseMixture hooks
    # ===================================================================
    def _is_frozen(self, k: int) -> bool:
        return k in self.frozen_components or component_is_frozen(self.components_[k])

    def _allocate_parameters(self, X: Optional[torch.Tensor] = None, set_random_state: bool = True):
        K = self.n_components
        self.components_ = copy.deepcopy(self._initial_components)
        if torch.is_tensor(self.init_weights):
            w = self.init_weights.to(device=self.device, dtype=self.dtype)
            if w.shape != (K,):
                raise ValueError(f"init_weights must have shape ({K},), got {tuple(w.shape)}.")
            self.weights_ = w / w.sum()
        else:
            self.weights_ = torch.full((K,), 1.0 / K, device=self.device, dtype=self.dtype)
        self.fitted_ = False
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")

    def _component_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cols = [c.log_probability(X).reshape(-1).to(device=X.device, dtype=X.dtype) for c in self.components_]
        return torch.stack(cols, dim=1)

    def _compute_stats(self, X: torch.Tensor, resp: torch.Tensor, batch_size: Optional[int] = None) -> SufficientStats:
        r"""Feed every non-frozen component its weighted rows chunk by chunk (`summarize`); the returned
        statistics only carry `nk` (the mixing weights' sufficient statistic)."""
        n, K = resp.shape
        step = n if (batch_size is None or batch_size >= n) else batch_size
        active = [k for k in range(K) if not self._is_frozen(k)]
        for start in range(0, n, step):
            Xc, rc = X[start:start + step], resp[start:start + step]
            for k in active:
                self.components_[k].summarize(Xc, rc[:, k])
        empty = torch.zeros(K, 0, dtype=resp.dtype, device=resp.device)
        return SufficientStats(nk=resp.sum(dim=0), s1=empty, s2=empty, n_samples=n)

    def _m_step_from_stats(self, stats: SufficientStats):
        nk = stats.nk + 1e-20
        self.weights_ = (nk / stats.n_samples).clamp_(min=1e-20)
        for k in range(self.n_components):
            if not self._is_frozen(k):
                self.components_[k].from_summaries()

    def _n_parameters(self) -> int:
        comps = self.components_ if self.components_ is not None else self._initial_components
        total = sum(component_n_parameters(c) for c in comps) + (self.n_components - 1)
        return int(total + (1 if self.noise else 0))

    def _sample_component(self, k: int, n: int) -> torch.Tensor:
        method = getattr(self.components_[k], 'sample', None)
        if not callable(method):
            raise NotImplementedError(
                f"component {k} ({type(self.components_[k]).__name__}) has no sample(n) method; "
                "GenericMixture.sample needs every component to provide one."
            )
        return method(n).reshape(n, -1).to(device=self.device, dtype=self.dtype)

    def _reseed_empty_components(self, X, log_prob_norm, empty_mask) -> int:
        r"""SEM: components with an optional `reseed(x)` method get the worst-fit points; the others are
        left unchanged (they must tolerate a near-zero total weight in `from_summaries`)."""
        idx = torch.nonzero(empty_mask, as_tuple=True)[0]
        worst = torch.argsort(log_prob_norm)[:idx.shape[0]]
        for j, k in enumerate(idx.tolist()):
            method = getattr(self.components_[k], 'reseed', None)
            if callable(method):
                method(X[worst[j]])
        return int(idx.shape[0])

    # ===================================================================
    # Snapshots (n_init loop, SEM's best post-burn-in sweep)
    # ===================================================================
    def _get_param_snapshot(self) -> dict:
        snapshot = {'weights_': self.weights_.clone(), 'components_': copy.deepcopy(self.components_)}
        if self.noise:
            snapshot['noise_weight_'] = self.noise_weight_.clone()
        return snapshot

    # ===================================================================
    # Persistence: components are pickled, so their classes must be importable when loading.
    # ===================================================================
    def to_dict(self) -> dict:
        config = {
            'n_features': self.n_features, 'max_iter': self.max_iter, 'tol': self.tol,
            'random_state': self.random_state, 'warm_start': self.warm_start, 'algorithm': self.algorithm,
            'batch_size': self.batch_size, 'temperature': self.temperature, 'convergence': self.convergence,
            'sem_burn_in': self.sem_burn_in, 'sem_estimate': self.sem_estimate,
            'components_per_class': self.components_per_class, 'unlabeled_weight': self.unlabeled_weight,
            'label_confidence': self.label_confidence, 'noise': self.noise, 'noise_volume': self.noise_volume,
            'noise_prior': self.noise_prior, 'frozen_components': list(self.frozen_components),
            'verbose': self.verbose, 'verbose_interval': self.verbose_interval,
        }
        return {
            'config': config,
            'initial_components': self._initial_components,
            'components_': self.components_,
            'weights_': self.weights_,
            'noise_weight_': self.noise_weight_,
            'noise_bounds_': self.noise_bounds_,
            'dtype': self.dtype,
            'fitted_': self.fitted_, 'converged_': self.converged_, 'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_, 'n_components_': self.n_components_,
            'classes_': self.classes_, 'class_of_component_': self.class_of_component_,
        }

    def save(self, filepath: str):
        r"""Save the model (including its components, pickled) to `filepath`; see `load`."""
        torch.save(self.to_dict(), filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "GenericMixture":
        r"""Load a model saved with `save`. The component classes must be importable."""
        kwargs = {} if device is None else {'map_location': device}
        state = torch.load(filepath, weights_only=False, **kwargs)
        model = cls(components=state['initial_components'], device=device, **state['config'])
        model.dtype = state['dtype']
        model.components_ = state['components_']
        model.weights_ = state['weights_']
        model.noise_weight_ = state['noise_weight_']
        model.noise_bounds_ = state['noise_bounds_']
        for name in ('fitted_', 'converged_', 'n_iter_', 'lower_bound_', 'n_components_', 'classes_',
                     'class_of_component_'):
            setattr(model, name, state[name])
        if model.fitted_:
            model.active_ = torch.ones(model.n_components, dtype=torch.bool, device=model.device)
        return model
