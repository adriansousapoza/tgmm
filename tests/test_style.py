"""The tutorials share one figure style (notebooks/README.md, 'Figure style'): this keeps it from drifting."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.py"))
SCRIPTS = sorted((ROOT / "scripts" / "make_assets").glob("*.py"))
README = (ROOT / "notebooks" / "README.md").read_text()
MAX_WIDTH = 11.0  # inches
MAX_MEAN_SIZE = 40


def _squash(text):
    return re.sub(r"\s+", "", text)


def _canonical_block():
    block = re.search(r"```python\n(# --- figure style.*?)```", README, re.S)
    assert block, "the style block is missing from notebooks/README.md"
    return block.group(1)


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_applies_the_style_block(nb):
    source = _squash(nb.read_text())
    for line in _canonical_block().strip().splitlines():
        if line.startswith("#"):
            continue
        assert _squash(line) in source or _squash(line.strip().rstrip(",")) in source, f"{nb.name} lacks the style block line: {line.strip()}"
    assert "# --- figure style (see notebooks/README.md) ---" in nb.read_text()


@pytest.mark.parametrize("path", NOTEBOOKS + SCRIPTS, ids=lambda p: p.name)
def test_no_stars_and_no_oversized_markers(path):
    text = path.read_text()
    assert not re.search(r"""marker\s*=\s*['"]\*['"]""", text), f"{path.name}: star markers are not used (symmetric markers only)"
    assert "dynamic_figsize" not in text, f"{path.name}: use the standard figure sizes instead of dynamic_figsize"
    for size in re.findall(r"\bmean_size\s*=\s*([0-9.]+)", text):
        assert float(size) <= MAX_MEAN_SIZE, f"{path.name}: mean_size={size} (the style allows at most {MAX_MEAN_SIZE})"
    for dpi in re.findall(r"figure\.dpi['\"]\s*\]?\s*[:=]\s*([0-9]+)", text):
        assert int(dpi) <= 100, f"{path.name}: figure.dpi={dpi}"


@pytest.mark.parametrize("path", NOTEBOOKS + SCRIPTS, ids=lambda p: p.name)
def test_figures_are_at_most_11_inches_wide(path):
    for width in re.findall(r"figsize\s*=\s*\(\s*([0-9.]+)\s*,", path.read_text()):
        assert float(width) <= MAX_WIDTH, f"{path.name}: figsize width {width} in > {MAX_WIDTH} in"


# ---- page structure (notebooks/README.md, 'Page structure'): a title, a short introduction, the body, 'Try it yourself', 'References'
BOILERPLATE = [
    (re.compile(r"^\s*#{1,6}\s*(Summary|Next steps?|Where to [Gg]o\b.*|What you'll learn|Prerequisites|See also)\s*$", re.M), "boilerplate heading"),
    (re.compile(r"\*\*(What you'll learn|Prerequisites|See also|Next|Where to [Gg]o[^*]*):?\*\*", re.I), "boilerplate label"),
]


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_boilerplate_sections(nb):
    text = nb.read_text()
    for pattern, what in BOILERPLATE:
        match = pattern.search(text)
        assert not match, f"{nb.name}: {what} {match.group(0).strip()!r}: keep the page simple (see notebooks/README.md, 'Page structure')"
