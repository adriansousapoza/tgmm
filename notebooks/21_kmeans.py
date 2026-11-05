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
    N_SEEDS = 30 if FAST else 100  # independent runs per start in the sensitivity experiment below
    return FAST, N_SEEDS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # k-means: How Lloyd's Algorithm Works

    `init_means='kmeans'` (see [3. Initialization](03_initialization.ipynb)) starts EM from the centres found by **k-means**, computed with Lloyd's algorithm. This tutorial builds the algorithm step by step on the four-blob data, shows every iteration as a picture and as a video you can pause and scrub, checks that the step-by-step version reproduces `GMMInitializer.kmeans` exactly, and explains how k-means relates to a Gaussian mixture and where it falls short.

    **What you'll learn**

    - The two steps of Lloyd's algorithm (assign every point to its nearest centre, move every centre to the mean of its points) and the objective they minimise
    - Why the objective can only go down, and why the algorithm therefore stops
    - What the tgmm implementation does at the edges: the starting centres, empty clusters, the stopping rule
    - How k-means is the hard-assignment limit of EM for a Gaussian mixture, and why its clusters are Voronoi cells
    - Why it is a good initialiser but not a good model: its sensitivity to the start and its spherical bias

    **Prerequisites:** [1. Introduction](01_introduction.ipynb), [3. Initialization](03_initialization.ipynb) and [20. k-means++](20_kmeans_plus_plus.ipynb), which explains the seeding that k-means starts from.

    **See also:** [7. Classification EM (CEM)](07_cem.ipynb) for the same hard-assignment idea with full covariances, and the [`GMMInitializer` API reference](../api/gmm-initializer.md).
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
    from matplotlib.colors import ListedColormap
    from PIL import Image

    from tgmm import ClusteringMetrics, GaussianMixture, GMMInitializer
    from tutorial_data import make_default_data

    warnings.filterwarnings('ignore')
    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (
        ClusteringMetrics,
        GMMInitializer,
        GaussianMixture,
        Image,
        ListedColormap,
        SEED,
        base64,
        device,
        io,
        make_default_data,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data are the default four blobs of [1. Introduction](01_introduction.ipynb), and we look for $k=4$ clusters.
    """)
    return


@app.cell
def _(SEED, device, make_default_data):
    X_tensor, y_tensor = make_default_data(SEED, device=device)
    X = X_tensor.cpu().numpy()
    K = 4
    return K, X, X_tensor, y_tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The algorithm

    k-means looks for $k$ centres $\mu_1,\dots,\mu_k$ and an assignment $c_j\in\{1,\dots,k\}$ of every point to a centre that make the **within-cluster sum of squares** (the *inertia*) as small as possible:

    $$
    J(\mu, c)=\sum_{j=1}^{N}\lVert x_j-\mu_{c_j}\rVert^2 .
    $$

    Minimising $J$ over both at once is hard, but each half is easy when the other is fixed. **Lloyd's algorithm** (1957, published 1982) alternates the two:

    1. **Assignment step.** With the centres fixed, give every point to its nearest centre:
       $$
       c_j=\arg\min_{i}\lVert x_j-\mu_i\rVert^2 .
       $$
    2. **Update step.** With the assignments fixed, move every centre to the mean of its cluster $C_i=\{j: c_j=i\}$:
       $$
       \mu_i=\frac{1}{|C_i|}\sum_{j\in C_i}x_j .
       $$

    **Why $J$ can only go down.** The assignment step picks, for every point separately, the centre that makes its term smallest, so it cannot increase $J$. The update step cannot increase $J$ either: for a cluster with mean $\bar x$ and any centre $\mu$,
    $$
    \sum_{j\in C}\lVert x_j-\mu\rVert^2=\sum_{j\in C}\lVert x_j-\bar x\rVert^2+|C|\,\lVert\bar x-\mu\rVert^2 ,
    $$
    which is smallest exactly at $\mu=\bar x$. There are finitely many ways to partition $N$ points, and $J$ strictly decreases unless the partition stays the same, so the algorithm reaches a fixed point after finitely many iterations. The fixed point is a **local** minimum of $J$, not necessarily the global one, which is why the starting centres matter. Each iteration costs $O(Nkd)$.

    ### What tgmm does at the edges

    `GMMInitializer.kmeans(data, k, max_iter, atol, random_state)`:

    - **Start.** The centres start from k-means++ seeding, `GMMInitializer.kpp(data, k, random_state)` ([20. k-means++](20_kmeans_plus_plus.ipynb)).
    - **Empty clusters.** If a cluster loses all its points it has no mean (0/0). tgmm re-seeds it at the data point that is farthest from its own assigned centre, the usual remedy.
    - **Stopping rule.** It stops when `torch.allclose(old_centres, new_centres, atol=atol)` holds (`atol` $=10^{-4}$ by default, together with PyTorch's default relative tolerance $10^{-5}$), or after `max_iter` iterations, and it returns the **old** centres of that last iteration (they differ from the new ones by less than the tolerance).

    The function below does exactly this, step by step, and keeps what each iteration did.
    """)
    return


@app.cell
def _(torch):
    def lloyd_trace(data, init, max_iter=1000, atol=1e-4):
        """Lloyd's algorithm from the centres ``init``, keeping every iteration; mirrors ``GMMInitializer.kmeans`` (which starts from kpp seeding).

        Per iteration t it records the centres at the start of the iteration (``centroids[t]``), the assignment of the points to them
        (``labels[t]``), the inertia of that assignment (``inertia[t]``) and the centres after the update step (``updated[t]``).
        ``final`` are the centres the library returns.
        """
        _c = init.clone()
        _k = _c.shape[0]
        _cents, _labs, _J, _upd = [], [], [], []
        for _ in range(max_iter):
            _dist = torch.cdist(data, _c)
            _lab = _dist.argmin(dim=1)
            _cents.append(_c.cpu().numpy())
            _labs.append(_lab.cpu().numpy())
            _J.append(float(_dist.min(dim=1)[0].pow(2).sum()))
            _means, _empty = [], []
            for _i in range(_k):
                _member = _lab == _i
                if bool(_member.any()):
                    _means.append(data[_member].mean(dim=0))
                else:
                    _means.append(None)
                    _empty.append(_i)
            if _empty:  # re-seed each empty cluster at the points farthest from their assigned centre
                _assigned = _dist[torch.arange(data.shape[0], device=data.device), _lab]
                _order = torch.argsort(_assigned, descending=True)
                for _n, _i in enumerate(_empty):
                    _means[_i] = data[_order[_n % data.shape[0]]].clone()
            _new = torch.stack(_means)
            _upd.append(_new.cpu().numpy())
            if torch.allclose(_c, _new, atol=atol):
                break
            _c = _new
        return {'centroids': _cents, 'labels': _labs, 'inertia': _J, 'updated': _upd, 'final': _c}

    return (lloyd_trace,)


@app.cell
def _(GMMInitializer, K, X_tensor, lloyd_trace, torch):
    _same = [torch.equal(lloyd_trace(X_tensor, GMMInitializer.kpp(X_tensor, K, random_state=_s))['final'], GMMInitializer.kmeans(X_tensor, K, random_state=_s)) for _s in range(20)]
    print(f'step-by-step lloyd_trace == GMMInitializer.kmeans for seeds 0..19: {all(_same)}')
    return


@app.cell
def _(GMMInitializer, K, SEED, X_tensor, lloyd_trace, np):
    trace_main = lloyd_trace(X_tensor, GMMInitializer.kpp(X_tensor, K, random_state=SEED))
    n_iter = len(trace_main['inertia'])
    _drops = np.diff(trace_main['inertia'])
    print(f'{n_iter} iterations; inertia {trace_main["inertia"][0]:,.0f} -> {trace_main["inertia"][-1]:,.0f}; largest increase between iterations: {max(0.0, _drops.max()):.2e} (never goes up)')
    return n_iter, trace_main


@app.cell(hide_code=True)
def _(mo, n_iter, trace_main):
    mo.md(rf"""
    The two lines above say what to expect: the step-by-step version returns the same centres as the library for 20 different seeds, and on the default seed the algorithm stops after {n_iter} iterations with the inertia falling from {trace_main['inertia'][0]:,.0f} (the k-means++ seeds, each point assigned to its nearest seed) to {trace_main['inertia'][-1]:,.0f}. It never rises.

    ## Iteration by iteration

    Every picture shows the state at the **start** of an iteration. The shaded regions are the **Voronoi cells** of the current centres (the set of locations nearer to one centre than to any other, which is exactly the assignment step applied to every possible point), the points are coloured by the cluster they belong to, the stars are the current centres and the arrows show where the update step is about to move them. The thin lines are the paths the centres have taken so far.
    """)
    return


@app.cell
def _(ListedColormap, X, np, plt):
    PALETTE = [plt.cm.tab10(_i) for _i in range(10)]
    _pad = 0.6
    XLIM = (X[:, 0].min() - _pad, X[:, 0].max() + _pad)
    YLIM = (X[:, 1].min() - _pad, X[:, 1].max() + _pad)
    _gx, _gy = np.meshgrid(np.linspace(*XLIM, 240), np.linspace(*YLIM, 240))
    GRID = np.stack([_gx.ravel(), _gy.ravel()], axis=1)
    GRID_SHAPE = _gx.shape

    def draw_lloyd(ax, trace, t, show_arrows=True):
        """The state at the start of iteration t: Voronoi cells, points coloured by cluster, centres, update arrows and centre paths."""
        _c = trace['centroids'][t]
        _k = _c.shape[0]
        _cells = np.argmin(((GRID[:, None, :] - _c[None]) ** 2).sum(-1), axis=1).reshape(GRID_SHAPE)
        ax.contourf(np.linspace(*XLIM, GRID_SHAPE[1]), np.linspace(*YLIM, GRID_SHAPE[0]), _cells, levels=np.arange(_k + 1) - 0.5, colors=[PALETTE[_i % 10] for _i in range(_k)], alpha=0.16)
        _lab = trace['labels'][t]
        ax.scatter(X[:, 0], X[:, 1], s=4, c=[PALETTE[_i % 10] for _i in _lab], alpha=0.65, linewidths=0)
        _path = np.stack(trace['centroids'][:t + 1])  # (t+1, k, d)
        for _i in range(_k):
            ax.plot(_path[:, _i, 0], _path[:, _i, 1], color=PALETTE[_i % 10], lw=1.2, alpha=0.9)
        ax.scatter(_c[:, 0], _c[:, 1], marker='*', s=240, c=[PALETTE[_i % 10] for _i in range(_k)], edgecolor='k', zorder=5)
        if show_arrows:
            _u = trace['updated'][t]
            for _i in range(_k):
                if np.linalg.norm(_u[_i] - _c[_i]) > 1e-3:
                    ax.annotate('', xy=_u[_i], xytext=_c[_i], arrowprops={'arrowstyle': '->', 'color': 'k', 'lw': 1.3}, zorder=6)
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect('equal')

    def draw_inertia(ax, trace, t):
        """The inertia of every iteration (grey) and the part reached so far (black), with the current iteration marked."""
        _J = np.array(trace['inertia'])
        ax.plot(np.arange(len(_J)), _J, color='0.8', lw=1)
        ax.plot(np.arange(t + 1), _J[:t + 1], color='k', lw=1.6, marker='o', ms=3)
        ax.scatter([t], [_J[t]], color='magenta', edgecolor='white', s=70, zorder=5)
        ax.set_xlabel('iteration')
        ax.set_ylabel('inertia $J$')
        ax.set_title('the objective, never increasing')
        ax.set_xlim(-0.5, len(_J) - 0.5)

    return PALETTE, draw_inertia, draw_lloyd


@app.cell
def _(draw_inertia, draw_lloyd, n_iter, np, plt, trace_main):
    _shown = sorted(set(int(_t) for _t in np.clip([0, 1, 2, 4, n_iter - 1], 0, n_iter - 1)))
    _fig, _axs = plt.subplots(3, 2, figsize=(9, 11.5), layout='constrained')
    for _ax, _t in zip(_axs.ravel(), _shown):
        draw_lloyd(_ax, trace_main, _t)
        _ax.set_title(f'iteration {_t}' + (' (the k-means++ seeds)' if _t == 0 else ' (converged)' if _t == n_iter - 1 else ''))
    for _ax in _axs.ravel()[len(_shown):-1]:
        _ax.axis('off')
    draw_inertia(_axs.ravel()[-1], trace_main, n_iter - 1)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same run as a video

    Press play, or drag the timeline to scrub through the iterations. The left plot is the state at the start of each iteration (as above); the right plot is the inertia, which falls fast at first and then flattens out as the centres settle.
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
def _(FAST, anim_from_frames, draw_inertia, draw_lloyd, n_iter, np, plt, trace_main):
    _cap = 20 if FAST else 40  # at most this many frames; longer runs are subsampled (denser early, where k-means moves most)
    _frames = list(range(n_iter)) if n_iter <= _cap else [int(_t) for _t in np.unique(np.round((n_iter - 1) * np.linspace(0, 1, _cap) ** 1.6).astype(int))]

    def draw_lloyd_frame(t):
        _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={'width_ratios': [1.7, 1]})
        draw_lloyd(_ax, trace_main, t)
        _ax.set_title(f'iteration {t} of {n_iter - 1}: assignment (colours), then update (arrows)')
        draw_inertia(_ax2, trace_main, t)
        _ax2.set_title(f'inertia $J$ = {trace_main["inertia"][t]:,.0f}')
        _fig.tight_layout()
        return _fig

    video_kmeans, n_frames_kmeans, video_kmeans_bytes = anim_from_frames(draw_lloyd_frame, _frames, fps=2, hold=6)
    print(f'k-means video: {n_frames_kmeans} frames, {video_kmeans_bytes / 1e3:.0f} kB')
    video_kmeans
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## k-means and the Gaussian mixture

    k-means is the **hard-assignment limit** of EM for a Gaussian mixture. Take a mixture with equal weights $\pi_k=1/k$ and the same spherical covariance $\Sigma_k=\sigma^2I$ for every component. The E-step responsibilities are

    $$
    r_{ik}=\frac{\exp\!\big(-\lVert x_i-\mu_k\rVert^2/2\sigma^2\big)}{\sum_l\exp\!\big(-\lVert x_i-\mu_l\rVert^2/2\sigma^2\big)} ,
    $$

    and as $\sigma\to0$ they become $0/1$: $r_{ik}\to1$ for the nearest centre and $0$ for all others. That is the assignment step. The M-step mean $\mu_k=\sum_ir_{ik}x_i/\sum_ir_{ik}$ then becomes the plain mean of the cluster, which is the update step. So Lloyd's algorithm is EM with soft assignments replaced by hard ones and with the covariances frozen to $\sigma^2I$. Allowing every cluster its own covariance while keeping the hard assignments gives **Classification EM** ([7. Classification EM (CEM)](07_cem.ipynb)); keeping soft assignments gives the full GMM.

    That relationship explains both why k-means is a good initialiser and why it is not a good model:

    - **A good initialiser.** Its centres are cheap to compute, sit inside the clusters and are already a fixed point of a hard version of EM, which is close to where EM wants to go.
    - **A biased model.** Because every cluster is judged by the Euclidean distance to its centre, k-means implicitly assumes round clusters of similar size. Its cells are the convex Voronoi regions above, with straight boundaries halfway between centres. An elongated or tilted cluster, like the thin diagonal blob here, cannot be represented by one centre and gets cut up.

    The next cell measures the second point on the default data: the adjusted Rand index (ARI, 1 is a perfect match with the true blobs) of the k-means partition against that of a full-covariance GMM started from the same k-means centres.
    """)
    return


@app.cell
def _(ClusteringMetrics, GaussianMixture, K, SEED, X_tensor, mo, trace_main, y_tensor):
    _labels_km = trace_main['labels'][-1]
    _ari_km = ClusteringMetrics.adjusted_rand_score(y_tensor, y_tensor.new_tensor(_labels_km))
    _gmm = GaussianMixture(n_components=K, covariance_type='full', init_means='kmeans', random_state=SEED).fit(X_tensor)
    _ari_gmm = ClusteringMetrics.adjusted_rand_score(y_tensor, _gmm.predict(X_tensor))
    ari_km, ari_gmm = float(_ari_km), float(_ari_gmm)
    mo.md(f'| method | ARI against the true blobs |\n|---|---|\n| k-means (Lloyd) | {ari_km:.3f} |\n| GMM with full covariances, started from the k-means centres | {ari_gmm:.3f} |')
    return ari_gmm, ari_km


@app.cell(hide_code=True)
def _(ari_gmm, ari_km, mo):
    mo.md(rf"""
    The mixture ends up closer to the truth ({ari_gmm:.2f} against {ari_km:.2f}) because it can describe the stretched blob with its own covariance, which k-means cannot. That is the reason for using k-means only to *start* EM, not instead of it.

    ## How sensitive is k-means to its start?

    Lloyd's algorithm finds a local minimum of $J$, so different starting centres can end in different ones. The experiment below runs it from many starts of two kinds and records the inertia it ends at: from **k-means++ seeds** (what tgmm uses) and from **uniformly chosen data points** (the `points` initialiser), for the $k=4$ of this tutorial and for a harder $k=8$. A run counts as successful when it reaches an inertia within 0.5% of the best one found for that $k$.
    """)
    return


@app.cell
def _(GMMInitializer, N_SEEDS, X_tensor, lloyd_trace, mo, np, torch):
    def _start(kind, k, s):
        if kind == 'kpp':
            return GMMInitializer.kpp(X_tensor, k, random_state=s)
        _g = torch.Generator(device=X_tensor.device)
        _g.manual_seed(s)
        return X_tensor[torch.randperm(X_tensor.shape[0], device=X_tensor.device, generator=_g)[:k]]

    sens = {}
    _rows = []
    for _k in (4, 8):
        _runs = {}
        for _kind in ('kpp', 'points'):
            _res = [lloyd_trace(X_tensor, _start(_kind, _k, _s)) for _s in range(N_SEEDS)]
            _runs[_kind] = (np.array([_r['inertia'][-1] for _r in _res]), np.array([len(_r['inertia']) for _r in _res]))
        _best = min(_runs['kpp'][0].min(), _runs['points'][0].min())
        for _kind, _label in (('kpp', 'k-means++ seeds'), ('points', 'uniform data points')):
            _J, _its = _runs[_kind]
            _ok = float((_J < 1.005 * _best).mean())
            sens[(_k, _kind)] = (_ok, float(_J.max() / _best), float(_its.mean()))
            _rows.append(f'| {_k} | {_label} | {_best:,.0f} | {_ok:.0%} | {_J.max() / _best:.3f} | {_its.mean():.0f} |')
    mo.md(f'Over {N_SEEDS} runs per start:\n\n| $k$ | start | best inertia found | runs reaching it (within 0.5%) | worst final inertia / best | mean iterations |\n|---|---|---|---|---|---|\n' + '\n'.join(_rows))
    return (sens,)


@app.cell(hide_code=True)
def _(mo, sens):
    mo.md(rf"""
    With $k=4$ the four blobs leave little room for a bad solution: {sens[(4, 'kpp')][0]:.0%} of the runs from k-means++ seeds and {sens[(4, 'points')][0]:.0%} of the runs from uniform points reach the best inertia. With $k=8$ it is different: only {sens[(8, 'kpp')][0]:.0%} (k-means++ seeds) and {sens[(8, 'points')][0]:.0%} (uniform points) of the runs reach the best one, and the worst run ends {sens[(8, 'kpp')][1] - 1:.1%} and {sens[(8, 'points')][1] - 1:.1%} above it. The number of local minima grows with $k$, and a careful seeding such as k-means++ does not remove them, so what helps is to restart. The practical consequences:

    - Run k-means several times and keep the lowest inertia (this is what `n_init` does for the whole mixture, see [3. Initialization](03_initialization.ipynb)).
    - The number of clusters $k$ is an input, not something k-means can choose: the inertia always falls as $k$ grows, so it cannot tell the right $k$ (model selection is the subject of [13. Model Selection](13_model_selection.ipynb)).
    - Euclidean distance is scale dependent: standardise features that are measured in different units before clustering.

    ## Summary

    - Lloyd's algorithm alternates two steps, assign every point to its nearest centre and move every centre to the mean of its points. The inertia $J=\sum_j\lVert x_j-\mu_{{c_j}}\rVert^2$ never increases, and the algorithm stops at a local minimum after finitely many iterations.
    - `lloyd_trace` above reproduces `GMMInitializer.kmeans` exactly: the same k-means++ start, the same empty-cluster remedy and the same stopping rule.
    - Lloyd's algorithm only finds a local minimum, and there are more of them the larger $k$ is, so restart it (`n_init`) rather than trust one run.
    - The clusters are Voronoi cells. k-means is the hard-assignment, equal-spherical-covariance limit of EM, so it is a good way to start a mixture but a poor model for elongated or unequal clusters.

    **Next:** back to [3. Initialization](03_initialization.ipynb), or on to [7. Classification EM (CEM)](07_cem.ipynb) for hard assignments with full covariances.

    ## References

    - Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory* 28(2), 129-137.
    - Arthur, D. and Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding. *Proceedings of the 18th ACM-SIAM Symposium on Discrete Algorithms*, 1027-1035.
    - Celeux, G. and Govaert, G. (1992). A classification EM algorithm for clustering and two stochastic versions. *Computational Statistics & Data Analysis* 14(3), 315-332.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it yourself (interactive version)

    The page you are reading shows the defaults. In the interactive version of this notebook (open it with the link at the top of the page, or run `marimo edit notebooks/21_kmeans.py` from a clone) the controls below change the number of clusters $k$, the random seed and the kind of start (k-means++ seeds or uniformly chosen data points), and the pictures update. The defaults here ($k=6$, seed 1) differ from the run above. The results are static figures: no video is made for your own settings, which would be too slow.
    """)
    return


@app.cell
def _(mo):
    # docs: hide-cell
    k_try = mo.ui.number(start=2, stop=8, step=1, value=6, label='number of clusters k', debounce=True)
    seed_try = mo.ui.number(start=0, stop=999, step=1, value=1, label='seed', debounce=True)
    start_try = mo.ui.dropdown(options={'k-means++ seeds': 'kpp', 'uniform data points': 'points'}, value='k-means++ seeds', label='start')
    return k_try, seed_try, start_try


@app.cell
def _(k_try, mo, seed_try, start_try):
    # docs: hide-cell
    mo.hstack([k_try, seed_try, start_try], justify='start')
    return


@app.cell
def _(GMMInitializer, X_tensor, draw_inertia, draw_lloyd, k_try, lloyd_trace, np, plt, seed_try, start_try, torch):
    _k, _s = int(k_try.value), int(seed_try.value)
    if start_try.value == 'kpp':
        _init = GMMInitializer.kpp(X_tensor, _k, random_state=_s)
    else:
        _g = torch.Generator(device=X_tensor.device)
        _g.manual_seed(_s)
        _init = X_tensor[torch.randperm(X_tensor.shape[0], device=X_tensor.device, generator=_g)[:_k]]
    _trace = lloyd_trace(X_tensor, _init)
    _n = len(_trace['inertia'])
    _shown = sorted(set(int(_t) for _t in np.clip([0, 1, 2, 4, _n - 1], 0, _n - 1)))
    _fig, _axs = plt.subplots(3, 2, figsize=(9, 11.5), layout='constrained')
    for _ax, _t in zip(_axs.ravel(), _shown):
        draw_lloyd(_ax, _trace, _t)
        _ax.set_title(f'iteration {_t}' + (' (start)' if _t == 0 else ' (converged)' if _t == _n - 1 else ''))
    for _ax in _axs.ravel()[len(_shown):-1]:
        _ax.axis('off')
    draw_inertia(_axs.ravel()[-1], _trace, _n - 1)
    _fig.suptitle(f'k = {_k}, seed {_s}: {_n} iterations, final inertia {_trace["inertia"][-1]:,.0f}')
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
