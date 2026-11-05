"""Shared helpers for the scripts that render the videos shown in the tutorials (notebooks/assets/*.mp4).

A tutorial notebook only *loads* a finished video; the code that produces it lives in this folder, so a notebook runs in seconds. Every
video is accompanied by a poster frame (``<name>.jpg``) and, where the notebook quotes numbers from the run, a ``<name>.json`` file.

Run the scripts through ``make assets`` (needs ``pip install -r scripts/make_assets/requirements.txt``).
"""
import io
import json
import pathlib
import subprocess
import tempfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ASSETS = pathlib.Path(__file__).resolve().parents[2] / "notebooks" / "assets"

# Compact fonts: the figure is drawn 11 inches wide and the page scales it down to the notebook column. No tight bbox: the tgmm theme
# saves with bbox 'tight', which would crop every frame differently.
RC = {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
      "figure.titlesize": 11, "savefig.bbox": "standard"}
DPI = 150
FPS = 4
HOLD = 8  # the last frame is repeated, so the end of a run stays on screen for a moment


def pick_frames(n_iterations, max_frames=60):
    """Iterations to show, 0..n_iterations, denser early on (where an iterative algorithm changes fastest); both ends are always included."""
    m = min(max_frames, n_iterations + 1)
    return [int(i) for i in np.unique(np.round(n_iterations * np.linspace(0, 1, m) ** 2).astype(int))]


def render_frames(draw, frames, rc=None, dpi=DPI):
    """``draw(t)`` returns a matplotlib figure; every frame is rendered on a canvas of the same size. Returns a list of RGB PIL images."""
    images = []
    for t in frames:
        with plt.rc_context({**RC, **(rc or {})}):
            fig = draw(t)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        images.append(Image.open(buf).convert("RGB"))
    return images


def encode_mp4(images, fps=FPS, hold=HOLD, crf=21):
    """H.264 (yuv420p, constant frame rate, ``+faststart``) of the frames, with the ffmpeg binary that ships with imageio-ffmpeg."""
    import imageio_ffmpeg

    w, h = images[0].size
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "video.mp4"  # a file, not a pipe: '+faststart' moves the index to the front
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
               "-i", "-", "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for img in images + [images[-1]] * hold:
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err}")
        return path.read_bytes()


def write_asset(name, draw, frames, info=None, fps=FPS, hold=HOLD, crf=21, rc=None):
    """Render the frames and write ``notebooks/assets/<name>.mp4`` and ``<name>.jpg`` (poster); ``info`` (a dict) goes to ``<name>.json``."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    images = render_frames(draw, frames, rc=rc)
    data = encode_mp4(images, fps=fps, hold=hold, crf=crf)
    (ASSETS / f"{name}.mp4").write_bytes(data)
    images[0].save(ASSETS / f"{name}.jpg", format="JPEG", quality=85)
    if info is not None:
        (ASSETS / f"{name}.json").write_text(json.dumps({**info, "frames": len(images), "fps": fps}, indent=2, default=float))
    print(f"{name}: {len(images)} frames, {len(data) / 1e6:.2f} MB, size {images[0].size}")
    return data
