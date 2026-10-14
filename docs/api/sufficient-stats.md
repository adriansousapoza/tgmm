# SufficientStats API

`SufficientStats` is the accumulator the `GaussianMixture` M-step reads
from (`nk`, `s1`, `s2`, `n_samples`); see the [sufficient-statistics
design
doc](https://github.com/adriansousapoza/tgmm/blob/main/docs/superpowers/specs/2026-10-14-phase0-sufficient-stats-design.md)
for the derivation. Most users won't construct one directly -- it's public
mainly so later mixture families and streaming/mini-batch fitting can
share the same accumulator type.

::: tgmm.stats.SufficientStats
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
