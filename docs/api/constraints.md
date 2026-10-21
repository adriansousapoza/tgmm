# tgmm.constraints API

The exact block solvers behind `GaussianMixture`'s `weight_bounds`, `mean_bounds` and `covariance_bounds` (see
[Fixed values and bounds](../user-guide/gaussian-mixture.md#fixed-values-and-bounds) for the mathematics). Each maximises the
expected complete-data log-likelihood over one parameter block, so the constrained M-step is an ECM step. Most users never call them
directly.

::: tgmm.constraints.water_fill_weights
    options:
      show_root_heading: true
      heading_level: 2

::: tgmm.constraints.box_qp_coordinate_ascent
    options:
      show_root_heading: true
      heading_level: 2

::: tgmm.constraints.clip_eigenvalues
    options:
      show_root_heading: true
      heading_level: 2
