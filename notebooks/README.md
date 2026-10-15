# tgmm tutorial notebooks

The tutorials are [marimo](https://marimo.io) notebooks: plain Python files (`NN_name.py`), one per
tutorial, numbered in the suggested reading order (see the
[tutorial index](../docs/tutorials/index.md)). They are the source of truth. The `.ipynb` files that
the documentation site renders are build artifacts, generated in CI (and by `make docs`) into
`docs/notebooks/`, which is git-ignored.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/adriansousapoza/tgmm/blob/main/notebooks/01_gmm.py)
*Every tutorial page on the docs site carries an "Open in molab" badge that runs the real notebook, torch
included, on a hosted server.*

## Running a notebook

```bash
pip install -e .                       # from a clone: tgmm plus its dependencies
pip install marimo
marimo edit notebooks/01_gmm.py        # interactive editor
python notebooks/01_gmm.py             # or just run it as a script
```

Without a clone: `uvx marimo edit --sandbox notebooks/01_gmm.py` builds an isolated environment from the
notebook's own header (PEP 723 metadata) and installs `tgmm` from PyPI.

## Conventions

- **Header.** Every notebook starts with PEP 723 inline metadata. `torch` carries the marker
  `sys_platform != 'emscripten'` (PyTorch has no WebAssembly wheel), which keeps marimo's WASM lint
  (MW003) quiet and `marimo edit --sandbox` working. Add any other import (`scipy`, `pandas`, `seaborn`, ...)
  to the `dependencies` list. The file uses `app = marimo.App(width="medium")`.
- **Plumbing cells.** The first two code cells (`import marimo as mo`, and the `FAST` / `MAX_ITER` / `N_INIT`
  config cell) start with the comment `# docs: hide-cell`; the docs build strips them from the rendered page.
- **Page layout.** Title and introduction (`# Title`, "What you'll learn", "Prerequisites", "See also"),
  `## Setup`, the single imports-and-seeds cell, the body, then `## Summary` and `**Next:**` links. The seed
  is `SEED`, keeping the value the notebook has always used.
- **`TGMM_DOCS_FAST`.** CI sets `TGMM_DOCS_FAST=1`. The config cell turns it into `FAST`, `MAX_ITER` and
  `N_INIT` (`100` / `1` when set, `1000` / `5` otherwise). Every slow call -- `max_iter`, `n_init`, sample
  counts, number of sweeps -- takes its budget from these names (or from a notebook-specific
  `FAST`-dependent constant), so the committed defaults stay realistic for local use. Each notebook must run
  in under three minutes with the flag set on a CPU runner.
- **Unique names.** marimo requires every global name to be defined in exactly one cell. Use distinct names
  (`gmm_full`, `gmm_diag`) for values used later, and underscore-prefixed names (`_gmm`, `_fig`, `_ax`,
  `_k`) for cell-local scratch values. Do not wrap cells in functions just to silence the error.
- **Figures.** A figure is shown when it is the cell's last expression (`plt.gcf()` or `fig`), not
  `plt.show()`. One figure per cell; split a cell that draws several.
- **Cell order is execution order.** The exported `.ipynb` is read top to bottom.
- **No `os.chdir`.** `tgmm` is imported as an installed package (editable install in a clone).
- **Checks.** `marimo check notebooks/` must be clean (no cycles, no multiple definitions, no unused-import
  warnings), and `python notebooks/NN_name.py` must exit 0.

## Tests

`tests/test_notebooks.py` runs every notebook as a script with `TGMM_DOCS_FAST=1` (180 s timeout,
`MPLBACKEND=Agg`) and asserts exit code 0. It is excluded from the default `pytest` run:

```bash
pytest -m notebooks            # or: make notebooks-test
marimo check notebooks/        # or: make notebooks-check
```

## Building the documentation locally

```bash
make docs        # export every notebook to docs/notebooks/*.ipynb, then `mkdocs serve`
```

CI does the same on every push to `main` (`.github/workflows/docs.yml`), with `--strict` and the badge
injection step in between.
