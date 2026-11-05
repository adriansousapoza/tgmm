"""Every video a tutorial shows is a finished file in notebooks/assets/ (rendered by scripts/make_assets/, see notebooks/assets/README.md)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
ASSETS = NOTEBOOKS / "assets"
CALLS = re.compile(r"""show_video\(\s*['"]([A-Za-z0-9_]+)['"]""")


def _videos():
    found = {}
    for nb in sorted(NOTEBOOKS.glob("[0-9][0-9]_*.py")):
        for name in CALLS.findall(nb.read_text()):
            found.setdefault(name, []).append(nb.name)
    return found


def test_notebooks_load_videos_from_assets():
    assert _videos(), "no notebook calls show_video(...): did the loader change?"


@pytest.mark.parametrize("name", sorted(_videos()))
def test_video_asset_exists(name):
    for ext in ("mp4", "jpg"):
        path = ASSETS / f"{name}.{ext}"
        assert path.is_file() and path.stat().st_size > 1000, f"{path} is missing (used by {_videos()[name]}); run `make assets`"
    assert (ASSETS / f"{name}.mp4").stat().st_size < 3_000_000, "keep each video under 3 MB (it is embedded in the exported page)"


def test_every_asset_script_output_is_used():
    """No orphaned videos: each mp4 in the folder is shown by some notebook."""
    used = set(_videos())
    for mp4 in ASSETS.glob("*.mp4"):
        assert mp4.stem in used, f"{mp4.name} is not shown by any notebook"
