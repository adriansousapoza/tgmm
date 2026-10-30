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
- Interactivity lives in the notebooks themselves: use `mo.ui` controls, mark the control cells with `# docs: hide-cell`, and make
  sure every control has a default so the static export and `python notebooks/NN_name.py` still work. Each tutorial page links to a
  molab session, where the controls run with real tgmm.

