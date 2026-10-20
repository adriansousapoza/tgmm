#!/usr/bin/env python
"""Post-process the notebooks exported by `marimo export ipynb` before MkDocs renders them.

For every ``*.ipynb`` in the given directory (default ``docs/notebooks``):

* remove the plumbing cells that start with ``# docs: hide-cell`` (``import marimo as mo`` and the
  ``TGMM_DOCS_FAST`` config cell), which are not part of the tutorial;
* insert an "Open in molab" badge as the first markdown cell, linking to the notebook's ``.py`` source
  on GitHub (https://molab.marimo.io/github/...). Never the ``/wasm`` variant: tgmm needs torch.

Idempotent and offline: running it twice changes nothing, and it never touches the network.

Usage: python scripts/postprocess_notebooks.py [DIR]
"""
import sys
from pathlib import Path

import nbformat

REPO = "adriansousapoza/tgmm"
BADGE = (
    "[![Open in molab](https://marimo.io/molab-shield.svg)]"
    "(https://molab.marimo.io/github/{repo}/blob/main/notebooks/{stem}.py)"
)
HIDE_MARKER = "# docs: hide-cell"
BADGE_PREFIX = "[![Open in molab]"


def process(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    before = nbformat.writes(nb)

    nb.cells = [c for c in nb.cells if not (c.cell_type == "code" and c.source.lstrip().startswith(HIDE_MARKER))]

    badge = BADGE.format(repo=REPO, stem=path.stem)
    has_badge = any(c.cell_type == "markdown" and c.source.startswith(BADGE_PREFIX) for c in nb.cells)
    if not has_badge:
        nb.cells.insert(0, nbformat.v4.new_markdown_cell(badge))
    else:  # refresh the link if the repo or file name changed
        for c in nb.cells:
            if c.cell_type == "markdown" and c.source.startswith(BADGE_PREFIX):
                c.source = badge

    nbformat.validate(nb)
    after = nbformat.writes(nb)
    if after != before:
        nbformat.write(nb, path)
        return True
    return False


def main(argv):
    directory = Path(argv[1]) if len(argv) > 1 else Path("docs/notebooks")
    paths = sorted(directory.glob("*.ipynb"))
    if not paths:
        sys.exit(f"no .ipynb files in {directory}")
    changed = sum(process(p) for p in paths)
    print(f"post-processed {len(paths)} notebooks ({changed} modified)")


if __name__ == "__main__":
    main(sys.argv)
