"""notebooks/01_introduction.py defines the default blobs inline; they must equal the shared notebooks/tutorial_data.py."""
import ast
import sys
from pathlib import Path

import numpy as np

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"


def _inline_defaults():
    tree = ast.parse((NOTEBOOKS / "01_introduction.py").read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("default_n_samples", "default_centers", "default_covs"):
                found[name] = eval(compile(ast.Expression(node.value), name, "eval"), {"np": np})
    return found


def test_notebook_01_defaults_match_shared_module():
    sys.path.insert(0, str(NOTEBOOKS))
    try:
        import tutorial_data as td
    finally:
        sys.path.remove(str(NOTEBOOKS))
    inline = _inline_defaults()
    assert set(inline) == {"default_n_samples", "default_centers", "default_covs"}
    assert inline["default_n_samples"] == td.N_SAMPLES
    for a, b in zip(inline["default_centers"], td.CENTERS):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(inline["default_covs"], td.COVS):
        np.testing.assert_allclose(a, b)
    assert len(inline["default_covs"]) == len(td.COVS)
