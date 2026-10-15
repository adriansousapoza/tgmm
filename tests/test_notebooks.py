"""Run every tutorial notebook (``notebooks/NN_name.py``, marimo) as a plain script.

marimo notebooks are Python scripts, so no marimo server is needed: each case runs
``python notebooks/NN_name.py`` in a subprocess with ``TGMM_DOCS_FAST=1`` (the reduced budgets CI uses),
a non-interactive matplotlib backend and a 180 s timeout, and asserts exit code 0.

Slow, so excluded from the default run (``addopts = -m "not notebooks"`` in pytest.ini)::

    pytest -m notebooks                       # all notebooks
    pytest -m notebooks -k 01_gmm             # one notebook
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((REPO / "notebooks").glob("[0-9][0-9]_*.py"))
TIMEOUT_SECONDS = 180

pytestmark = pytest.mark.notebooks


def test_notebooks_are_present():
    assert NOTEBOOKS, "no notebooks/NN_name.py files found"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_runs_as_script(notebook):
    env = dict(
        os.environ,
        TGMM_DOCS_FAST="1",
        MPLBACKEND="Agg",
        PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
    )
    try:
        result = subprocess.run(
            [sys.executable, str(notebook)],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{notebook.name} exceeded {TIMEOUT_SECONDS} s under TGMM_DOCS_FAST=1\n{exc.stderr or ''}")
    assert result.returncode == 0, (
        f"{notebook.name} exited with {result.returncode}\n--- stderr (tail) ---\n{result.stderr[-4000:]}"
    )
