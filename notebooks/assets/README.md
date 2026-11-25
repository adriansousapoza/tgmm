# Pre-rendered assets of the tutorials

The tutorials only **load** the videos in this folder; they never render them. A simulation that animates dozens of EM or k-means
iterations takes minutes, so the code that produces each video lives in `scripts/make_assets/` and the finished files are committed here.

For every asset `<name>` there are up to three files:

| File | What it is |
| --- | --- |
| `<name>.mp4` | the video: H.264, yuv420p, 4 frames per second, `+faststart` (plays in every browser; no autoplay, the reader can play, pause and scrub) |
| `<name>.jpg` | the poster frame (the first frame), shown before the video is played |
| `<name>.json` | the numbers the notebook quotes in its prose (iterations, log-likelihoods, ...), so the text always matches the video |

| Asset | Used by | Rendered by |
| --- | --- | --- |
| `em_1d` | `01_introduction`: EM on a 1-D mixture from a bad start | `scripts/make_assets/make_em_videos.py` |
| `em_2d` | `01_introduction`: EM on the four default blobs, started with all components around (0, 0) | `scripts/make_assets/make_em_videos.py` |
| `kmeans_pp` | `18_kmeans_plus_plus`: k-means++ seeding step by step | `scripts/make_assets/make_seeding_videos.py` |
| `kmeans` | `19_kmeans`: Lloyd's algorithm iteration by iteration | `scripts/make_assets/make_seeding_videos.py` |

## How a notebook loads them

Each notebook has a small hidden cell (`show_video(name)`, `asset_info(name)`) that reads `notebooks/assets/<name>.mp4`, `.jpg` and `.json`
next to the notebook and embeds the video in the page as a `<video controls>` (base64, so the exported docs page is self-contained). If the files
are not next to the notebook (for example a hosted copy of a single notebook), it falls back to the same names under
`https://raw.githubusercontent.com/adriansousapoza/tgmm/main/notebooks/assets/`.

## Regenerating

After changing a script in `scripts/make_assets/` (or the data it uses, `notebooks/tutorial_data.py`), render the videos again and commit the result:

```bash
pip install -r scripts/make_assets/requirements.txt   # numpy, matplotlib, scipy, pillow, imageio-ffmpeg (a pip-bundled ffmpeg)
make assets                                           # or: python scripts/make_assets/make_em_videos.py, ... make_seeding_videos.py
```

This takes a few minutes. The scripts are deterministic (fixed seeds), so running them twice gives the same videos. Nothing in the docs build
re-renders them: if a script changes and the assets are not regenerated, the page shows the old video.
