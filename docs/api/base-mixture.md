# BaseMixture API

`BaseMixture` is the shared base class for every tgmm mixture family
(`GaussianMixture`, `StudentTMixture`, `VonMisesFisherMixture`): the
EM/CEM/SEM/SAEM/online-EM loop, `n_init` best-run selection,
`predict`/`predict_proba`/`score`/`score_samples`/`bic`/`aic`/`icl`, and a
generic `sample()` shell all live here, behind a small set of abstract
methods each family implements
(`_component_log_prob`/`_compute_stats`/`_m_step_from_stats`/
`_allocate_parameters`/`_n_parameters`/`_sample_component`). Most users
won't instantiate it directly or need to know it exists -- it's public
mainly so the shared machinery has one documented home, and so a custom
mixture family can be built on it directly. See the [Other Mixture
Families](../user-guide/other-mixture-families.md) user guide page and the
[design
doc](https://github.com/adriansousapoza/tgmm/blob/main/docs/superpowers/specs/2026-10-14-phase4-base-mixture-studentt-vmf-design.md)
for the rationale behind what's shared here vs. what each family keeps as
its own override (`GaussianMixture`'s `fit`/`sample`/`save`/`load` in
particular).

::: tgmm.mixtures.base.BaseMixture
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
