# PositiveUnlabeledMixture API

`PositiveUnlabeledMixture` wraps the semi-supervised `fit(X, labels=...)`
from `BaseMixture` for positive-unlabeled (PU) learning: a subset of rows
is known to be positive, everything else is unlabeled. See the
[Semi-Supervised and Positive-Unlabeled Fitting](../user-guide/semi-supervised-pu.md)
user guide page for the terms (SCAR, mixture proportion estimation,
`anchor`) and when to reach for each option.

::: tgmm.semi_supervised.PositiveUnlabeledMixture
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
