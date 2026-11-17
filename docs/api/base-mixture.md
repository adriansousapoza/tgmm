# BaseMixture API

`BaseMixture` is the shared base class of tgmm's mixture models (`GaussianMixture`,
`ConditionalGaussianMixture` and `GenericMixture`): the
EM/CEM/SEM/SAEM/online-EM loop, `n_init` best-run selection,
`predict`/`predict_proba`/`score`/`score_samples`/`bic`/`aic`/`icl`, and a
generic `sample()` shell all live here, behind a small set of abstract
methods each component model implements
(`_component_log_prob`/`_compute_stats`/`_m_step_from_stats`/
`_allocate_parameters`/`_n_parameters`/`_sample_component`). Most users
won't instantiate it directly or need to know it exists -- it's public
mainly so the shared machinery has one documented home, and so a custom
mixture can be built on it directly (see [GenericMixture](generic-mixture.md)).
`GaussianMixture` overrides `fit`/`sample`/`save`/`load` with its richer implementations.
For the Gaussian model itself, see the [Introduction](../notebooks/01_introduction.ipynb) tutorial.

::: tgmm.mixtures.base.BaseMixture
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
