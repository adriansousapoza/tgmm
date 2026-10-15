# ConditionalGaussianMixture API

`ConditionalGaussianMixture` is the result of `GaussianMixture.condition(x_obs, obs_dims)`: the distribution of the
unobserved coordinates given the observed ones, one Gaussian mixture per conditioning row. Its `mean()` is Gaussian
mixture regression. `GaussianMixture.marginal(dims)` (and `StudentTMixture.marginal`) return the marginal mixture over a
subset of the coordinates. See the [design
note](https://github.com/adriansousapoza/tgmm/blob/main/docs/superpowers/specs/2026-10-15-v05-marginal-condition-design.md)
for the equations and conventions.

::: tgmm.mixtures.conditional.ConditionalGaussianMixture
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
