# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "tgmm",
#     "torch; sys_platform != 'emscripten'",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "pillow",
#     "imageio-ffmpeg",
# ]
#
# [tool.marimo.display]
# dataframes = "plain"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    # docs: hide-cell
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # docs: hide-cell
    import os as _os

    # TGMM_DOCS_FAST=1 (set by CI) shrinks the slow calls; the defaults below are the realistic local ones.
    FAST = _os.environ.get("TGMM_DOCS_FAST") == "1"
    N_SEEDS = 60 if FAST else 300  # independent seedings per method in the experiments below (seeding is cheap: no EM)
    return FAST, N_SEEDS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # k-means++: How the Seeding Works

    `init_means='kpp'` and `init_means='kmeans'` (see [3. Initialization](03_initialization.ipynb)) both start from **k-means++ seeding**: a way of choosing $k$ starting centres from the data that spreads them out *without* being fooled by outliers. This tutorial builds the algorithm step by step, shows what its sampling probabilities look like, checks that the step-by-step version reproduces `GMMInitializer.kpp` exactly, and measures why the squared distance is the right weight. The next tutorial, [21. k-means](21_kmeans.ipynb), shows what happens after the seeding.

    **What you'll learn**

    - The k-means++ algorithm and its sampling probability $P(x)\propto D(x)^2$
    - Why the *squared* distance: uniform sampling ignores distance, farthest-first (`maxdist`) over-reacts to outliers, $D^2$ sits in between and carries a guarantee
    - How to read the probability map at each step, as a figure and as a video you can pause and scrub
    - How seeding quality is measured (the k-means cost of the seeds) and how the methods compare

    **Prerequisites:** [1. Introduction](01_introduction.ipynb) and [3. Initialization](03_initialization.ipynb).

    **See also:** [21. k-means](21_kmeans.ipynb), the [`GMMInitializer` API reference](../api/gmm-initializer.md).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    The next cell imports what the notebook needs and fixes the random seeds. It expects `tgmm` to be importable: `pip install tgmm`, or `pip install -e .` from a clone of the repository. Opened with `marimo edit --sandbox`, the dependencies listed in the header of this file are installed automatically.
    """)
    return


@app.cell
def _():
    import base64
    import io
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    from tgmm import GMMInitializer, generate_gmm_data
    from tutorial_data import make_default_data

    warnings.filterwarnings('ignore')
    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return GMMInitializer, Image, SEED, base64, device, generate_gmm_data, io, make_default_data, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data are the default four blobs of [1. Introduction](01_introduction.ipynb), and we seed $k=4$ centres.
    """)
    return


@app.cell
def _(SEED, device, make_default_data):
    X_tensor, _ = make_default_data(SEED, device=device)
    X = X_tensor.cpu().numpy()
    K = 4
    return K, X, X_tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The algorithm

    Seeding means picking $k$ of the data points as starting centres $c_1,\dots,c_k$. Write

    $$
    D(x)=\min_{l<i}\lVert x-c_l\rVert
    $$

    for the distance from a point $x$ to the **nearest centre chosen so far**. k-means++ (Arthur & Vassilvitskii, 2007) is:

    1. Choose $c_1$ **uniformly at random** from the $N$ data points.
    2. For $i=2,\dots,k$: draw the next centre $c_i$ from the data with probability
       $$
       P(x_j)=\frac{D(x_j)^2}{\sum_{m=1}^{N}D(x_m)^2}.
       $$

    That is all. A point that is already a centre has $D=0$, so it can never be chosen twice, and a point far from every centre is likely to be chosen. The probability is a *sample*, not an argmax, so every point that is not yet covered keeps a chance.

    The tgmm implementation (`GMMInitializer.kpp(data, k, random_state)`) does exactly this with a seeded `torch.Generator`: `torch.randint` for $c_1$ and `torch.multinomial` on the probabilities for each later centre. The function below does the same step by step and keeps the probabilities it used, so that we can look at them.
    """)
    return


@app.cell
def _(torch):
    def kpp_trace(data, k, seed, power=2.0):
        """k-means++ seeding, step by step: the chosen indices and, for every step, the distances D(x) and the sampling probabilities.

        ``power`` is the exponent in P(x) ~ D(x)**power: 2 is k-means++ (and mirrors ``GMMInitializer.kpp`` call for call), 0 is uniform
        sampling, ``float('inf')`` is farthest-first (always the point with the largest D, like ``GMMInitializer.maxdist``).
        """
        n = data.shape[0]
        gen = torch.Generator(device=data.device)
        gen.manual_seed(seed)
        idx = [int(torch.randint(0, n, (1,), device=data.device, generator=gen))]
        dist = [None]
        prob = [torch.full((n,), 1.0 / n, device=data.device)]  # step 1: every point has probability 1/N
        for _ in range(1, k):
            _d2 = torch.cdist(data, data[idx]).pow(2).min(dim=1)[0]
            if power == float('inf'):
                _p = torch.zeros_like(_d2)
                _sel = int(torch.argmax(_d2))
                _p[_sel] = 1.0
            else:
                _w = _d2 if power == 2 else _d2.pow(power / 2)
                _p = _w / _w.sum()
                _sel = int(torch.multinomial(_p, 1, generator=gen))
            idx.append(_sel)
            dist.append(_d2.sqrt())
            prob.append(_p)
        return {'idx': idx, 'dist': dist, 'prob': prob}

    return (kpp_trace,)


@app.cell
def _(GMMInitializer, K, X_tensor, kpp_trace, torch):
    _same = [torch.equal(X_tensor[kpp_trace(X_tensor, K, _s)['idx']], GMMInitializer.kpp(X_tensor, K, random_state=_s)) for _s in range(20)]
    print(f'step-by-step kpp_trace == GMMInitializer.kpp for seeds 0..19: {all(_same)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two agree centre for centre (the check above compares the 4 centres for 20 different seeds), so what follows is what `init_means='kpp'` really does.

    ## Step by step on the four blobs

    Each panel shows the situation **just before** the next centre is drawn: every point is coloured and sized by its sampling probability relative to the largest one, $P(x)/\max P$, the centres chosen so far are the red stars, and the black ring marks the point that was then sampled. In step 1 all points are equally likely. From step 2 on, the points near a star are almost impossible to draw and the points far from every star are the likely ones.
    """)
    return


@app.cell
def _(np):
    def rel_weight(trace, i):
        """Sampling probability of step i relative to its maximum (for farthest-first, whose probability is one point, the distance)."""
        _p = trace['prob'][i].cpu().numpy()
        if i > 0 and _p.max() >= 1.0 - 1e-12:  # a one-hot probability (farthest-first): show D/max D instead
            _d = trace['dist'][i].cpu().numpy()
            return _d / _d.max()
        return _p / _p.max()

    def draw_step(ax, X, trace, i, show_pick=True):
        """The probability map of step i: points coloured and sized by P(x)/max P, the chosen centres (stars) and the sampled point (ring)."""
        _w = rel_weight(trace, i)
        _order = np.argsort(_w)
        _size = np.full(len(_w), 6.0) if i == 0 else 3 + 24 * _w
        _sc = ax.scatter(X[_order, 0], X[_order, 1], c=_w[_order], s=_size[_order], cmap='viridis', vmin=0, vmax=1, alpha=0.85, linewidths=0)
        _chosen = trace['idx'][:i]
        if _chosen:
            ax.scatter(X[_chosen, 0], X[_chosen, 1], marker='*', s=260, c='red', edgecolor='k', zorder=4)
        if show_pick:
            _p = trace['idx'][i]
            ax.scatter(X[_p, 0], X[_p, 1], s=230, facecolor='none', edgecolor='k', linewidth=2, zorder=5)
        ax.set_aspect('equal')
        return _sc

    return draw_step, rel_weight


@app.cell
def _(K, SEED, X_tensor, kpp_trace):
    trace_main = kpp_trace(X_tensor, K, SEED)
    return (trace_main,)


@app.cell
def _(K, X, draw_step, plt, trace_main):
    _fig, _axs = plt.subplots(3, 2, figsize=(9, 11.5), sharex=True, sharey=True, layout='constrained')
    for _i in range(K):
        _sc = draw_step(_axs.flat[_i], X, trace_main, _i)
        _axs.flat[_i].set_title('step 1: the first centre is uniform' if _i == 0 else f'step {_i + 1}: sample with $P(x)\\propto D(x)^2$')
    _ax = _axs.flat[K]
    _ax.scatter(X[:, 0], X[:, 1], s=4, color='0.65', alpha=0.6, linewidths=0)
    _ax.scatter(X[trace_main['idx'], 0], X[trace_main['idx'], 1], marker='*', s=260, c='red', edgecolor='k', zorder=4)
    for _n, _p in enumerate(trace_main['idx']):
        _ax.annotate(str(_n + 1), (X[_p, 0], X[_p, 1]), textcoords='offset points', xytext=(8, 6), fontsize=9, weight='bold')
    _ax.set_aspect('equal')
    _ax.set_title(f'the {K} seeds (numbered in the order drawn)')
    _axs.flat[K + 1].axis('off')
    _fig.colorbar(_sc, ax=_axs, shrink=0.55, location='right', label='relative sampling probability  $P(x)/\\max P$')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same run as a video

    Press play, or drag the timeline to scrub through the steps. Each step has two frames: first the probability map, then the same map with the sampled point ringed. The right-hand plot sorts the points by their distance $D$ to the nearest centre and shows how much of the sampling probability the nearest $x\%$ of the points carry. A straight diagonal means uniform sampling; the $D^2$ weighting bends the curve down, so that the farthest points carry far more than their share.
    """)
    return


@app.cell
def _(Image, base64, io, mo, plt):
    ANIM_RC = {'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'figure.titlesize': 11, 'savefig.bbox': 'standard'}
    ANIM_DPI = 150  # an 11-inch figure is 1650 px wide: crisp when the page scales it down to the notebook column

    def _encode_mp4(imgs, fps, hold):
        """H.264 video (yuv420p, constant frame rate) of the frames, with the ffmpeg binary that ships with the imageio-ffmpeg package."""
        import pathlib
        import subprocess
        import tempfile

        import imageio_ffmpeg

        _w, _h = imgs[0].size
        with tempfile.TemporaryDirectory() as _dir:
            _path = pathlib.Path(_dir) / 'anim.mp4'  # a file, not a pipe: '+faststart' moves the index to the front, so playback can start at once
            _cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{_w}x{_h}', '-r', str(fps), '-i', '-',
                    '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(_path)]
            _proc = subprocess.Popen(_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for _img in imgs + [imgs[-1]] * hold:
                _proc.stdin.write(_img.tobytes())
            _proc.stdin.close()
            _err = _proc.stderr.read().decode(errors='replace')
            if _proc.wait() != 0:
                raise RuntimeError(f'ffmpeg failed: {_err}')
            return _path.read_bytes()

    def anim_from_frames(draw, frames, fps=2, hold=4):
        """Render ``draw(t)`` (it returns a matplotlib figure) for every t in ``frames`` into a video the reader can play, pause and scrub.

        Every frame has the same size (the tgmm theme saves with a tight bounding box, which would crop each frame differently). The video is
        embedded as a ``<video controls>`` (no autoplay); without a usable ffmpeg it falls back to a lossless animated WebP.
        """
        _imgs = []
        for _t in frames:
            with plt.rc_context(ANIM_RC):
                _fig = draw(_t)
                _buf = io.BytesIO()
                _fig.savefig(_buf, format='png', dpi=ANIM_DPI)
            plt.close(_fig)
            _imgs.append(Image.open(_buf).convert('RGB'))
        try:
            _data = _encode_mp4(_imgs, fps, hold)
            _poster = io.BytesIO()
            _imgs[0].save(_poster, format='JPEG', quality=85)
            _html = (f'<video controls loop muted playsinline preload="metadata" poster="data:image/jpeg;base64,{base64.b64encode(_poster.getvalue()).decode()}" '
                     f'style="width:100%;height:auto"><source src="data:video/mp4;base64,{base64.b64encode(_data).decode()}" type="video/mp4"></video>')
        except (ImportError, OSError, RuntimeError):  # no usable ffmpeg: a lossless animated WebP instead
            _out = io.BytesIO()
            _imgs[0].save(_out, format='WEBP', save_all=True, append_images=_imgs[1:], lossless=True, method=4, loop=0, duration=[1000 // fps] * (len(_imgs) - 1) + [2500])
            _data = _out.getvalue()
            _html = f'<img src="data:image/webp;base64,{base64.b64encode(_data).decode()}" style="width:100%;height:auto">'
        return mo.Html(_html), len(_imgs), len(_data)

    return (anim_from_frames,)


@app.cell
def _(GMMInitializer, K, SEED, X_tensor, torch):
    def potential(data, centres):
        """The k-means cost of a set of centres: the sum over all points of the squared distance to the nearest centre."""
        return float(torch.cdist(data, centres).pow(2).min(dim=1)[0].sum())

    phi_kmeans = potential(X_tensor, GMMInitializer.kmeans(X_tensor, K, random_state=SEED))  # the cost after Lloyd's algorithm (tutorial 21)
    return phi_kmeans, potential


@app.cell
def _(K, X, X_tensor, anim_from_frames, draw_step, np, phi_kmeans, plt, potential, trace_main):
    _plan = [(_i, _mode) for _i in range(K) for _mode in ('prob', 'pick')] + [(K, 'final')]
    _phi_seed = potential(X_tensor, X_tensor[trace_main['idx']])

    def draw_kpp_frame(f):
        _i, _mode = _plan[f]
        _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={'width_ratios': [1.7, 1]})
        if _mode == 'final':
            _ax.scatter(X[:, 0], X[:, 1], s=4, color='0.65', alpha=0.6, linewidths=0)
            _ax.scatter(X[trace_main['idx'], 0], X[trace_main['idx'], 1], marker='*', s=260, c='red', edgecolor='k', zorder=4)
            _ax.set_aspect('equal')
            _ax.set_title(f'seeding done: the {K} centres')
            _ax2.axis('off')
            _ax2.text(0.02, 0.6, f'k-means cost of the seeds\n  $\\phi_{{seed}}$ = {_phi_seed:,.0f}\n\nafter Lloyd\'s algorithm\n(tutorial 21)\n  $\\phi_{{k-means}}$ = {phi_kmeans:,.0f}', fontsize=10, va='center')
        else:
            draw_step(_ax, X, trace_main, _i, show_pick=(_mode == 'pick'))
            _what = 'sampled point (ring) becomes centre' if _mode == 'pick' else 'probability map before the draw, centre'
            _ax.set_title(f'step {_i + 1} of {K}: {_what} {_i + 1}' if _i else f'step 1 of {K}: the first centre is a uniformly random point')
            _p = trace_main['prob'][_i].cpu().numpy()
            if _i == 0:
                _ax2.plot([0, 1], [0, 1], color='0.5')
                _ax2.text(0.5, 0.2, 'every point has\nprobability 1/N', ha='center')
            else:
                _order = np.argsort(trace_main['dist'][_i].cpu().numpy())
                _ax2.plot([0, 1], [0, 1], color='0.6', ls='--', label='uniform sampling')
                _ax2.plot(np.arange(1, len(_p) + 1) / len(_p), np.cumsum(_p[_order]), color='C3', lw=2, label='$P(x)\\propto D(x)^2$')
                _far = 1 - np.cumsum(_p[_order])[int(0.9 * len(_p)) - 1]
                _ax2.text(0.03, 0.62, f'the farthest 10% of the points\ncarry {_far:.0%} of the probability', fontsize=9)
                _ax2.legend(loc='lower right')
            _ax2.set_xlabel('fraction of the points, nearest to the centres first')
            _ax2.set_ylabel('probability carried')
            _ax2.set_title('where the sampling probability sits')
            _ax2.set_xlim(0, 1)
            _ax2.set_ylim(0, 1.02)
        _fig.tight_layout()
        return _fig

    video_kpp, n_frames_kpp, video_kpp_bytes = anim_from_frames(draw_kpp_frame, range(len(_plan)), fps=1, hold=3)
    print(f'k-means++ video: {n_frames_kpp} frames, {video_kpp_bytes / 1e3:.0f} kB')
    video_kpp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why the *squared* distance?

    Sampling $c_i$ with probability proportional to $D(x)^p$ gives a family of seeding rules, and three of them are familiar:

    | exponent $p$ | rule | what it does |
    |---|---|---|
    | $0$ | **uniform** (`points`) | ignores the centres: two seeds often land in the same blob and a small blob is easily missed |
    | $2$ | **k-means++** (`kpp`) | far points are likely, near points are not, but every uncovered point keeps a chance |
    | $\infty$ | **farthest-first** (`maxdist`) | always the single farthest point: deterministic, and the farthest point is typically an outlier |

    The squared distance is not arbitrary. The quantity k-means minimises is the **cost** (or *potential*)

    $$
    \phi(C)=\sum_{j=1}^{N}\min_{c\in C}\lVert x_j-c\rVert^2 ,
    $$

    and weighting by $D^2$ means choosing a point with probability proportional to its *contribution to the current cost*. Arthur & Vassilvitskii (2007) prove that this makes the seeding already competitive with the best possible clustering: if $\phi_{\mathrm{opt}}$ is the cost of an optimal set of $k$ centres, then

    $$
    \mathbb{E}\big[\phi(C_{\text{k-means++}})\big]\;\le\;8\,(\ln k+2)\,\phi_{\mathrm{opt}} .
    $$

    Uniform seeding has no such guarantee: it can be arbitrarily far from optimal. The table below puts numbers on the three rules for the seeds drawn in step 3 of the run above (two centres chosen, the third is about to be drawn): how much of the sampling probability sits on the farthest 1% of the points.
    """)
    return


@app.cell
def _(mo, np, trace_main):
    _d = trace_main['dist'][2].cpu().numpy()  # distance to the nearest of the first two centres
    _top = np.argsort(_d)[-max(1, len(_d) // 100):]  # the farthest 1% of the points
    _rows = []
    for _name, _p in (('uniform', 0.0), ('$D$', 1.0), ('$D^2$ (k-means++)', 2.0), ('$D^4$', 4.0)):
        _w = _d ** _p if _p else np.ones_like(_d)
        _rows.append(f'| {_name} ({"p = " + format(_p, "g")}) | {_w[_top].sum() / _w.sum():.1%} |')
    _rows.append('| farthest-first ($p=\\infty$) | 100% (all of it, on one point) |')
    mo.md('| sampling probability $\\propto$ | share on the farthest 1% of the points |\n|---|---|\n' + '\n'.join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The share grows steadily with $p$: from the $1\%$ that uniform sampling gives, through $D^2$, to all of it for farthest-first. $D^2$ is the middle way: it concentrates the draw on the uncovered region (which is what makes the seeding spread out) but never on one point alone, so a single outlier cannot take over.

    ## How good are the seeds?

    Two quick experiments, both cheap because seeding needs no EM. The **cost ratio** is $\phi$ of the seeds divided by the cost $\phi_{\text{k-means}}$ after running Lloyd's algorithm to convergence (1.0 would be a perfect seeding; a ratio near 1 means Lloyd has little left to do). The **one centre per blob** rate uses a clean dataset of four unit-variance blobs 8 apart: a seeding succeeds when the four seeds fall in four different blobs and each lies within distance 4 of its blob's centre (half the distance between neighbouring blobs), so a seed placed on an outlier does not count. The last column adds 1% uniform outliers to the same data.
    """)
    return


@app.cell
def _(GMMInitializer, K, N_SEEDS, X_tensor, device, generate_gmm_data, mo, np, phi_kmeans, potential, torch):
    def seed_centres(method, data, k, s):
        torch.manual_seed(s)
        if method == 'kpp':
            return GMMInitializer.kpp(data, k, random_state=s)
        return getattr(GMMInitializer, method)(data, k)

    _true = np.array([(0, 0), (8, 0), (0, 8), (8, 8)], float)
    X_clean, _ = generate_gmm_data([c for c in _true], [np.eye(2)] * 4, [300] * 4, device=device, random_state=1)
    _out = torch.tensor(np.random.default_rng(0).uniform(-12, 20, (12, 2)), dtype=X_clean.dtype, device=device)
    X_outl = torch.cat([X_clean, _out])
    _true_t = torch.tensor(_true, dtype=X_clean.dtype, device=device)

    def one_per_blob(data, method):
        _ok = 0
        for _s in range(N_SEEDS):
            _c = seed_centres(method, data, 4, _s)
            _d, _blob = torch.cdist(_c, _true_t).min(dim=1)  # each seed's distance to, and index of, its nearest true blob centre
            _ok += int(len(set(_blob.tolist())) == 4 and bool((_d < 4).all()))  # four different blobs, every seed inside its blob (< half the spacing)
        return _ok / N_SEEDS

    results = {}
    _rows = []
    for _method in ('points', 'random', 'kpp', 'maxdist'):
        _ratio = np.array([potential(X_tensor, seed_centres(_method, X_tensor, K, _s)) / phi_kmeans for _s in range(N_SEEDS)])
        results[_method] = (float(np.median(_ratio)), one_per_blob(X_clean, _method), one_per_blob(X_outl, _method))
        _rows.append(f'| `{_method}` | {results[_method][0]:.2f} | {results[_method][1]:.0%} | {results[_method][2]:.0%} |')
    mo.md(f'Median over {N_SEEDS} seedings per method:\n\n| `init_means=` | cost ratio $\\phi_{{seed}}/\\phi_{{k-means}}$ (default blobs) | one centre per blob (clean) | one centre per blob (+1% outliers) |\n|---|---|---|---|\n' + '\n'.join(_rows))
    return (results,)


@app.cell(hide_code=True)
def _(mo, results):
    mo.md(rf"""
    Reading the table:

    - **Cost ratio.** `kpp` seeds cost about {results['kpp'][0]:.1f} times the converged k-means cost, against {results['points'][0]:.1f} for uniformly chosen data points and {results['random'][0]:.1f} for `random`, which draws its centres from a Gaussian fitted to the whole dataset and so ignores the clusters. `maxdist` ({results['maxdist'][0]:.1f}) is high on these overlapping blobs because it reaches for the extreme points.
    - **One centre per blob, clean data.** Farthest-first finds all four blobs in {results['maxdist'][1]:.0%} of the runs, k-means++ in {results['kpp'][1]:.0%}, uniform points in {results['points'][1]:.0%}: when the clusters are compact and well separated, being deterministic about the farthest point pays off.
    - **With 1% outliers.** The picture reverses: farthest-first drops to {results['maxdist'][2]:.0%} (it spends its centres on the outliers) while k-means++ keeps {results['kpp'][2]:.0%}, because the outliers are only a tiny share of the total $D^2$ mass.

    That robustness is why `kpp` (and `kmeans`, which starts from it) is the default choice and `maxdist` is reserved for clean, well-separated data.

    ## Practical notes

    - **Cost.** Each step computes the distances from all $N$ points to the new centre, so seeding costs $O(Nkd)$: the same as *one* iteration of Lloyd's algorithm. It is inherently sequential (the probabilities of step $i$ depend on the centre drawn in step $i-1$). For very large data a parallel variant, k-means$\parallel$ (Bahmani et al., 2012), draws several centres per round.
    - **Greedy k-means++.** The common refinement (used by scikit-learn) draws a few candidates per step and keeps the one that lowers the cost most. It is not in `GMMInitializer`; `n_init` restarts play a similar role.
    - **Seeding is not the end.** The seeds are data points, not cluster means. `init_means='kmeans'` therefore polishes them with Lloyd's algorithm ([21. k-means](21_kmeans.ipynb)), and EM then refines everything including weights and covariances.

    ## Summary

    - k-means++ draws the first centre uniformly and every later centre with probability $\propto D(x)^2$, the squared distance to the nearest centre so far.
    - `kpp_trace` above reproduces `GMMInitializer.kpp` exactly; the probability maps show how the draw moves from region to region.
    - $D^2$ sits between uniform sampling (ignores the centres) and farthest-first (all weight on one point): it spreads the seeds out, has a guarantee $\mathbb{{E}}[\phi]\le 8(\ln k+2)\phi_{{\mathrm{{opt}}}}$, and is robust to outliers.

    **Next:** [21. k-means](21_kmeans.ipynb), what Lloyd's algorithm does with the seeds. Back to [3. Initialization](03_initialization.ipynb).

    ## References

    - Arthur, D. and Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding. *Proceedings of the 18th ACM-SIAM Symposium on Discrete Algorithms*, 1027-1035.
    - Gonzalez, T. F. (1985). Clustering to minimize the maximum intercluster distance. *Theoretical Computer Science* 38, 293-306 (farthest-first traversal).
    - Bahmani, B., Moseley, B., Vattani, A., Kumar, R. and Vassilvitskii, S. (2012). Scalable k-means++. *Proceedings of the VLDB Endowment* 5(7), 622-633.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it yourself (interactive version)

    The page you are reading shows the defaults. In the interactive version of this notebook (open it with the link at the top of the page, or run `marimo edit notebooks/20_kmeans_plus_plus.py` from a clone) the controls below change the number of centres $k$, the random seed and the sampling rule, and the probability maps update. The defaults here ($k=6$, seed 1) differ from the run above. The results are static figures: no video is made for your own settings, which would be too slow.
    """)
    return


@app.cell
def _(mo):
    # docs: hide-cell
    k_try = mo.ui.number(start=2, stop=12, step=1, value=6, label='number of centres k', debounce=True)
    seed_try = mo.ui.number(start=0, stop=999, step=1, value=1, label='seed', debounce=True)
    power_try = mo.ui.dropdown(
        options={'D² (k-means++)': 2.0, 'uniform (p = 0)': 0.0, 'D (p = 1)': 1.0, 'D⁴ (p = 4)': 4.0, 'farthest-first (p = ∞)': float('inf')},
        value='D² (k-means++)',
        label='sampling probability ∝',
    )
    return k_try, power_try, seed_try


@app.cell
def _(k_try, mo, power_try, seed_try):
    # docs: hide-cell
    mo.hstack([k_try, seed_try, power_try], justify='start')
    return


@app.cell
def _(X, X_tensor, draw_step, k_try, kpp_trace, np, plt, potential, power_try, seed_try):
    _k = int(k_try.value)
    _trace = kpp_trace(X_tensor, _k, int(seed_try.value), power=float(power_try.value))
    _nrows = (_k + 1) // 2
    _fig, _axs = plt.subplots(_nrows, 2, figsize=(9, 3.9 * _nrows), sharex=True, sharey=True, squeeze=False, layout='constrained')
    _sc = None
    for _i, _ax in enumerate(_axs.ravel()):
        if _i < _k:
            _sc = draw_step(_ax, X, _trace, _i)
            _ax.set_title(f'pick {_i + 1}')
        else:
            _ax.axis('off')
    _fig.colorbar(_sc, ax=_axs, shrink=0.5, location='right', label='relative sampling probability (farthest-first: relative distance $D$)')
    _fig.suptitle(f'k = {_k}, seed {int(seed_try.value)}: cost of these seeds $\\phi_{{seed}}$ = {potential(X_tensor, X_tensor[_trace["idx"]]):,.0f}')
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
