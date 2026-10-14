# GPCM Covariance API

`mstep_gpcm` is the M-step used internally by `GaussianMixture` for the 8
new GPCM covariance types (`'VEI'`, `'EVI'`, `'VEE'`, `'EVE'`, `'EEV'`,
`'VVE'`, `'VEV'`, `'EVV'`) -- most users won't call it directly, just pass
the code as `covariance_type`. See the
[GPCM Covariance Models](../user-guide/gpcm-covariance-models.md) user
guide for an overview.

::: tgmm.covariance.gpcm.mstep_gpcm
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: tgmm.covariance.gpcm.GPCMResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
