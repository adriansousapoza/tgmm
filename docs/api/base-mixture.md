# BaseMixture API

`BaseMixture` is the shared base class of tgmm's Gaussian mixture models (`GaussianMixture`
and `ConditionalGaussianMixture`): the
EM/CEM/SEM/SAEM/online-EM loop, `n_init` best-run selection and
`predict`/`predict_proba`/`score`/`score_samples`/`bic`/`aic`/`icl` all live here,
behind a small set of abstract methods that the concrete model implements (see the
source below). Most users won't instantiate it directly or need to know it
exists -- it's public mainly so the shared machinery has one documented home.
`GaussianMixture` overrides `fit`/`sample`/`save`/`load` with its richer implementations.
For the Gaussian model itself, see the [Introduction](../notebooks/01_introduction.ipynb) tutorial.

::: tgmm.mixtures.base.BaseMixture
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
