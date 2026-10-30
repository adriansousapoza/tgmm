# Model Selection API

`select_model` fits a grid of `(n_components, covariance_type)` combinations
and compares them by BIC, AIC, or ICL; `ModelSelectionResult` is what it
returns. See the [Model Selection tutorial](../notebooks/14_model_selection.ipynb)
for a worked example.

::: tgmm.model_selection.select_model
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: tgmm.model_selection.ModelSelectionResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
