"""
Model selection sweeps for `GaussianMixture`: fit a grid of
(n_components, covariance_type) combinations and compare them by BIC, AIC,
or ICL.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .gmm import GaussianMixture
from .covariance.gpcm import GPCM_CODES

_VALID_CRITERIA = ('bic', 'aic', 'icl')


@dataclass
class ModelSelectionResult:
    r"""
    The result of a :func:`select_model` sweep.

    Attributes
    ----------
    table : list of dict
        One row per `(n_components, covariance_type)` combination tried,
        each a dict with keys `n_components`, `covariance_type`, `bic`,
        `aic`, `icl`, `log_likelihood`, `converged`. A combination whose
        fit raised has every score set to `None` (and `converged=None`)
        rather than being omitted or propagating the exception.
    best_ : GaussianMixture or None
        The fitted model minimising the sweep's `criterion`. `None` if
        every combination in the sweep raised.
    models_ : dict
        Maps `(n_components, covariance_type)` -> the fitted
        `GaussianMixture` for every combination that fit successfully
        (failed combinations are absent, not `None`-valued, since there is
        no model object to return for them).
    """

    table: List[Dict[str, Any]]
    best_: Optional[GaussianMixture]
    models_: Dict[Tuple[int, str], GaussianMixture]

    def to_dataframe(self):
        r"""
        `self.table` as a `pandas.DataFrame`. Imports `pandas` lazily --
        it is not a runtime dependency of `tgmm` otherwise.
        """
        import pandas as pd
        return pd.DataFrame(self.table)


def select_model(
    X,
    n_components: Iterable[int] = range(1, 11),
    covariance_types: Iterable[str] = GPCM_CODES,
    criterion: str = 'bic',
    n_init: int = 1,
    random_state: Optional[int] = None,
    **gmm_kwargs,
) -> ModelSelectionResult:
    r"""
    Fit `GaussianMixture(n_components=K, covariance_type=type, ...)` for
    every `(K, type)` pair in the Cartesian product of `n_components` and
    `covariance_types`, and return the comparison.

    Parameters
    ----------
    X : torch.Tensor
        Data to fit, shape (n_samples, n_features).
    n_components : iterable of int, optional
        Component counts to try. (default: `range(1, 11)`)
    covariance_types : iterable of str, optional
        Covariance types to try, as tgmm names or mclust GPCM codes
        (case-insensitive) -- see `GaussianMixture`'s `covariance_type`.
        (default: all 14 mclust GPCM codes, i.e. the full
        `tgmm.covariance.gpcm.GPCM_CODES`; this is the mclust-style
        model-selection workflow the plan describes)
    criterion : str, optional
        Which score to minimise when choosing `best_`: `'bic'`, `'aic'`,
        or `'icl'`. (default: `'bic'`)
    n_init : int, optional
        Forwarded to every `GaussianMixture`. (default: 1)
    random_state : int, optional
        Forwarded to every `GaussianMixture`. (default: None)
    **gmm_kwargs
        Any other `GaussianMixture` constructor argument, forwarded to
        every fit in the sweep (e.g. `max_iter`, `tol`, `reg_covar`).

    Returns
    -------
    ModelSelectionResult

    Notes
    -----
    A combination whose fit raises (e.g. a singular covariance at a large
    `K`) is recorded in `.table` with every score `None` and is excluded
    from the `criterion`-minimising search -- the sweep as a whole does not
    fail because one combination did.
    """
    if criterion not in _VALID_CRITERIA:
        raise ValueError(f"criterion must be one of {_VALID_CRITERIA}, got {criterion!r}")

    table: List[Dict[str, Any]] = []
    models: Dict[Tuple[int, str], GaussianMixture] = {}
    best_score = float('inf')
    best_model: Optional[GaussianMixture] = None

    for k in n_components:
        for covariance_type in covariance_types:
            row: Dict[str, Any] = {
                'n_components': k,
                'covariance_type': covariance_type,
                'bic': None,
                'aic': None,
                'icl': None,
                'log_likelihood': None,
                'converged': None,
            }
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    n_init=n_init,
                    random_state=random_state,
                    **gmm_kwargs,
                )
                gmm.fit(X)
                row['bic'] = gmm.bic(X)
                row['aic'] = gmm.aic(X)
                row['icl'] = gmm.icl(X)
                row['log_likelihood'] = gmm.lower_bound_
                row['converged'] = gmm.converged_
                models[(k, covariance_type)] = gmm

                score = row[criterion]
                if score is not None and score < best_score:
                    best_score = score
                    best_model = gmm
            except Exception:
                pass

            table.append(row)

    return ModelSelectionResult(table=table, best_=best_model, models_=models)
