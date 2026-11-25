# tgmm tutorial notebooks

The tutorials are [marimo](https://marimo.io) notebooks: plain Python files (`NN_name.py`), one per
tutorial, numbered in the suggested reading order (see the
[tutorial index](../docs/tutorials/index.md)). They are the source of truth. The `.ipynb` files that
the documentation site renders are build artifacts, generated in CI (and by `make docs`) into
`docs/notebooks/`, which is git-ignored.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/adriansousapoza/tgmm/blob/main/notebooks/01_introduction.py)
*Every tutorial page on the docs site carries an "Open in molab" badge that runs the real notebook, torch
included, on a hosted server.*

## Running a notebook

```bash
pip install -e .                       # from a clone: tgmm plus its dependencies
pip install marimo
marimo edit notebooks/01_introduction.py        # interactive editor
python notebooks/01_introduction.py             # or just run it as a script
```

Without a clone: `uvx marimo edit --sandbox notebooks/01_introduction.py` builds an isolated environment from the
notebook's own header (PEP 723 metadata) and installs `tgmm` from PyPI.

## Conventions

- **Header.** Every notebook starts with PEP 723 inline metadata. `torch` carries the marker
  `sys_platform != 'emscripten'` (PyTorch has no WebAssembly wheel), which keeps marimo's WASM lint
  (MW003) quiet and `marimo edit --sandbox` working. Add any other import (`scipy`, `pandas`, `seaborn`, ...)
  to the `dependencies` list. The file uses `app = marimo.App(width="medium")`.
- **Interactive controls.** `mo.ui` controls (sliders, dropdowns, the blob designer in `01_introduction`) go in their own cells,
  marked `# docs: hide-cell` so the static docs page does not show widgets that cannot work there. Every control needs a default, so the
  exported page and `python notebooks/NN_name.py` show the default result. The controls run in the molab link at the top of every
  docs page (real tgmm, so the sliders work) and in `marimo edit` locally.
- **Controls must be reactive.** marimo only re-runs dependent cells when a control's value changes if the control is a top-level
  variable or sits inside `mo.ui.array` / `mo.ui.dictionary` / `mo.ui.batch`. Controls nested in a plain Python list or dict render
  fine but never update anything, so build groups of controls with `mo.ui.array`. Number fields (`mo.ui.number`) commit when they lose
  focus or on Enter. Check a new control by changing it in `marimo run` and confirming the output follows.
- **Animations are assets.** Show an EM run (or anything that changes over iterations) as one video, not as a slider or a row of
  snapshots, so the static docs page and the interactive notebook show the same thing and a reader can start, stop and scrub it. The
  notebook never renders it: the code lives in `scripts/make_assets/` and writes `notebooks/assets/<name>.mp4` (plus a `.jpg` poster
  and a `.json` with the numbers the prose quotes); the notebook calls `show_video('<name>')`, which embeds the file in the page (or
  falls back to the copy on GitHub). Regenerate with `make assets` after changing a script, and commit the files
  (`tests/test_assets.py` checks that every shown video exists). See `notebooks/assets/README.md`.
- **Interactive controls at the end.** Put every `mo.ui` control, and the figures that exist only to show it, in a final section
  `## Try it yourself (interactive version)` after the Summary, so the static page reads as one story.
- **Shared dataset.** The default four-blob data is defined once, in `tutorial_data.py` (a plain module next to the notebooks, not a
  tutorial). `01_introduction` builds and explains it; every later notebook that needs it does `from tutorial_data import
  make_default_data` and has no data section of its own. Use `--ignore-scripts` with `marimo check` so the module is not parsed as a
  notebook, and keep the defaults in `01_introduction.py` equal to the module's (`tests/test_tutorial_data.py` checks this).
- **One narrative.** The tutorials are the documentation (there is no separate user guide or explainer set): put the mathematics, the
  code and the figures in the notebook, and link to the API reference (`../api/*.md`) for signatures.
- **Plumbing cells.** The first two code cells (`import marimo as mo`, and the `FAST` / `MAX_ITER` / `N_INIT`
  config cell) start with the comment `# docs: hide-cell`; the docs build strips them from the rendered page.
- **Page layout.** Keep it simple: a `# Title`, one or two sentences of introduction, `## Setup`, the single
  imports-and-seeds cell, then the body (see "Page structure" below). The seed is `SEED`, keeping the value the
  notebook has always used.
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
- **Checks.** `marimo check --ignore-scripts notebooks/` must be clean (no cycles, no multiple definitions, no unused-import
  warnings), and `python notebooks/NN_name.py` must exit 0.

## Figure style (the same in every tutorial and every video)

The tgmm plotting theme is made for papers (16 pt titles, 20 pt figure titles, 300 dpi); the tutorials override it with one block, applied
right after `tgmm` is imported in the setup cell (the block is identical in every notebook; `tests/test_style.py` checks it):

```python
# --- figure style (see notebooks/README.md) ---
DOCS_RC = {'figure.dpi': 100, 'figure.titlesize': 12, 'axes.titlesize': 10.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
           'legend.fontsize': 8, 'lines.linewidth': 1.4, 'lines.markersize': 5}
plt.rcParams.update(DOCS_RC)
```

- **Size.** A figure is at most 11 inches wide (the docs column is scaled down, so anything wider has unreadable text). Single panel
  `(6.5, 4.5)`; two or three panels in a row `(11, 4.5)` / `(11, 3.8)`; more panels: a 2-column grid `(11, 4.5 * rows)` (never four or
  more panels in one row). If an export would exceed 8 MB, lower only `figure.dpi` and say why in a comment.
- **Titles.** One short title per panel (taken from rcParams: do not pass `fontsize`), a `suptitle` only for multi-panel figures, one line.
- **Colours.** Components use tgmm's `CATEGORICAL_PALETTE` (what `plot_gmm` uses); sequential data `viridis`; noise or unassigned points
  grey; red marks the initial guess; orange the current estimate; black the fitted means/centres and the real values. No pink or magenta.
- **Markers.** Fitted component means and centres: black hexagon `'h'`, `s=25` (the `plot_gmm` default, never above 40). Initial guess or
  chosen seeds: red cross `'x'`, `s=30`, linewidth 1.2. Current estimate (videos): orange octagon `'8'`, `s=30`, white edge. Real or true
  values: black diamond `'D'`, `s=20`. Never stars. Data points `s=4`-`8` with `alpha=0.5`.
- **Lines.** 1.4 pt; solid for estimated, dashed for real/true; covariance ellipses show the 95% region (or 1, 2 and 3 standard deviations
  where the levels matter).
- **Videos** (`scripts/make_assets/`): 11 inches wide at 150 dpi with the compact fonts of `_video.RC` (they are scaled down further) and
  the same markers and colours.

## Page structure (the same in every tutorial)

A `# Title` and one or two sentences of introduction; `## Setup`; the numbered body sections; `## Try it yourself (interactive version)` if the
notebook has controls; `## References` if the text cites papers. Nothing else: no "What you'll learn", "Prerequisites", "See also", "Summary" or
"Next" / "Where to go" sections (links to other tutorials belong inline, in the sentence that needs them).

## Tests

`tests/test_notebooks.py` runs every notebook as a script with `TGMM_DOCS_FAST=1` (180 s timeout,
`MPLBACKEND=Agg`) and asserts exit code 0. It is excluded from the default `pytest` run:

```bash
pytest -m notebooks            # or: make notebooks-test
marimo check --ignore-scripts notebooks/   # or: make notebooks-check
```

## Building the documentation locally

```bash
make docs        # export every notebook to docs/notebooks/*.ipynb, then `mkdocs serve`
```

CI does the same on every push to `main` (`.github/workflows/docs.yml`), with `--strict` and the badge
injection step in between.
