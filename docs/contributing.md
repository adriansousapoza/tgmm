# Contributing

tgmm isn't yet set up to accept external contributions through a formal process (no
contribution guidelines, issue templates, or review workflow are in place yet).

If you're interested in contributing, please reach out directly to
[asp@di.ku.dk](mailto:asp@di.ku.dk).

## Tutorials are marimo notebooks

The tutorials under `notebooks/` are [marimo](https://marimo.io) notebooks, i.e. plain Python files
(`notebooks/NN_name.py`), and they are the single source of truth. The `.ipynb` pages the documentation site
renders are generated from them by CI (`marimo export ipynb ... --include-outputs`) into `docs/notebooks/`,
which is git-ignored -- never commit an `.ipynb`.

- Conventions (header with PEP 723 metadata, unique global names, figures as last expressions, the
  `TGMM_DOCS_FAST` budgets) are in `notebooks/README.md`.
- `make notebooks-check` runs `marimo check notebooks/` (must be clean) and `make notebooks-test` runs every notebook
  as a script with `TGMM_DOCS_FAST=1` (each must finish in under 3 minutes). CI runs both on every pull request.
- `make docs` exports the notebooks and serves the site locally, exactly as CI builds it.
- The interactive explainer pages (`docs/explainers/`) run in the reader's browser through `mkdocs-marimo`
  (Pyodide), so they may import only NumPy, SciPy, matplotlib and marimo -- never `tgmm` or `torch`.
- **Wrap every explainer fence in `<div markdown="0">`.** `mkdocs-marimo` inserts each island as an inline tag, which
  Markdown puts in a `<p>`; browsers close a `<p>` when they meet a `<div>`, so any output built from `mo.hstack`,
  `mo.vstack` or other layout elements silently loses its island (the cells then fail with `NameError` in Pyodide).
  `<div markdown="0">` around the fence prevents the paragraph.

