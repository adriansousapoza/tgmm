"""Render the two videos of tutorials 20 and 21: how k-means++ seeding and Lloyd's algorithm (k-means) work.

    python scripts/make_assets/make_seeding_videos.py        # or: make assets

writes notebooks/assets/kmeans_pp.{mp4,jpg,json} and notebooks/assets/kmeans.{mp4,jpg,json}. The notebooks only load these files; the
step-by-step traces below mirror ``GMMInitializer.kpp`` and ``GMMInitializer.kmeans`` call for call (the notebooks check that on 20 seeds),
and everything is seeded, so running the script again reproduces the same videos. Everything runs on the CPU, so the result does not
depend on whether a GPU is present.
"""
import pathlib
import sys
import warnings

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))

import matplotlib.pyplot as plt  # noqa: E402
from _video import write_asset  # noqa: E402
from tutorial_data import make_default_data  # noqa: E402

from tgmm import GMMInitializer  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 0  # the seed the notebooks use for their static figures
K = 4
X_TENSOR, _ = make_default_data(SEED, device="cpu")
X = X_TENSOR.numpy()


# --------------------------------------------------------------------------------------------------------------------------------
# k-means++ seeding (tutorial 20)
# --------------------------------------------------------------------------------------------------------------------------------
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
        d2 = torch.cdist(data, data[idx]).pow(2).min(dim=1)[0]
        if power == float("inf"):
            p = torch.zeros_like(d2)
            sel = int(torch.argmax(d2))
            p[sel] = 1.0
        else:
            w = d2 if power == 2 else d2.pow(power / 2)
            p = w / w.sum()
            sel = int(torch.multinomial(p, 1, generator=gen))
        idx.append(sel)
        dist.append(d2.sqrt())
        prob.append(p)
    return {"idx": idx, "dist": dist, "prob": prob}


def rel_weight(trace, i):
    """Sampling probability of step i relative to its maximum (for farthest-first, whose probability is one point, the distance)."""
    p = trace["prob"][i].cpu().numpy()
    if i > 0 and p.max() >= 1.0 - 1e-12:  # a one-hot probability (farthest-first): show D/max D instead
        d = trace["dist"][i].cpu().numpy()
        return d / d.max()
    return p / p.max()


def draw_step(ax, X, trace, i, show_pick=True):
    """The probability map of step i: points coloured and sized by P(x)/max P, the chosen centres (stars) and the sampled point (ring)."""
    w = rel_weight(trace, i)
    order = np.argsort(w)
    size = np.full(len(w), 6.0) if i == 0 else 3 + 24 * w
    sc = ax.scatter(X[order, 0], X[order, 1], c=w[order], s=size[order], cmap="viridis", vmin=0, vmax=1, alpha=0.85, linewidths=0)
    chosen = trace["idx"][:i]
    if chosen:
        ax.scatter(X[chosen, 0], X[chosen, 1], marker="*", s=260, c="red", edgecolor="k", zorder=4)
    if show_pick:
        p = trace["idx"][i]
        ax.scatter(X[p, 0], X[p, 1], s=230, facecolor="none", edgecolor="k", linewidth=2, zorder=5)
    ax.set_aspect("equal")
    return sc


def potential(data, centres):
    """The k-means cost of a set of centres: the sum over all points of the squared distance to the nearest centre."""
    return float(torch.cdist(data, centres).pow(2).min(dim=1)[0].sum())


def make_kpp_video():
    trace = kpp_trace(X_TENSOR, K, SEED)
    phi_kmeans = potential(X_TENSOR, GMMInitializer.kmeans(X_TENSOR, K, random_state=SEED))  # the cost after Lloyd's algorithm (tutorial 21)
    phi_seed = potential(X_TENSOR, X_TENSOR[trace["idx"]])
    plan = [(i, mode) for i in range(K) for mode in ("prob", "pick")] + [(K, "final")]
    far10 = []  # share of the sampling probability carried by the farthest 10% of the points, per step

    def draw_frame(f):
        i, mode = plan[f]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={"width_ratios": [1.7, 1]})
        if mode == "final":
            ax.scatter(X[:, 0], X[:, 1], s=4, color="0.65", alpha=0.6, linewidths=0)
            ax.scatter(X[trace["idx"], 0], X[trace["idx"], 1], marker="*", s=260, c="red", edgecolor="k", zorder=4)
            ax.set_aspect("equal")
            ax.set_title(f"seeding done: the {K} centres")
            ax2.axis("off")
            ax2.text(0.02, 0.6, f"k-means cost of the seeds\n  $\\phi_{{seed}}$ = {phi_seed:,.0f}\n\nafter Lloyd's algorithm\n(tutorial 21)\n  $\\phi_{{k-means}}$ = {phi_kmeans:,.0f}",
                     fontsize=10, va="center")
        else:
            draw_step(ax, X, trace, i, show_pick=(mode == "pick"))
            what = "sampled point (ring) becomes centre" if mode == "pick" else "probability map before the draw, centre"
            ax.set_title(f"step {i + 1} of {K}: {what} {i + 1}" if i else f"step 1 of {K}: the first centre is a uniformly random point")
            p = trace["prob"][i].cpu().numpy()
            if i == 0:
                ax2.plot([0, 1], [0, 1], color="0.5")
                ax2.text(0.5, 0.2, "every point has\nprobability 1/N", ha="center")
            else:
                order = np.argsort(trace["dist"][i].cpu().numpy())
                ax2.plot([0, 1], [0, 1], color="0.6", ls="--", label="uniform sampling")
                ax2.plot(np.arange(1, len(p) + 1) / len(p), np.cumsum(p[order]), color="C3", lw=2, label="$P(x)\\propto D(x)^2$")
                far = 1 - np.cumsum(p[order])[int(0.9 * len(p)) - 1]
                if mode == "prob":
                    far10.append(float(far))
                ax2.text(0.03, 0.62, f"the farthest 10% of the points\ncarry {far:.0%} of the probability", fontsize=9)
                ax2.legend(loc="lower right")
            ax2.set_xlabel("fraction of the points, nearest to the centres first")
            ax2.set_ylabel("probability carried")
            ax2.set_title("where the sampling probability sits")
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1.02)
        fig.tight_layout()
        return fig

    # far10 is filled while the frames are drawn; write_asset dumps the json after rendering
    info = {"seed": SEED, "k": K, "steps": K, "centre_indices": trace["idx"], "phi_seed": phi_seed, "phi_kmeans": phi_kmeans,
            "far10_share_per_step": far10}
    write_asset("kmeans_pp", draw_frame, range(len(plan)), info=info, fps=1, hold=3, crf=18)


# --------------------------------------------------------------------------------------------------------------------------------
# k-means (Lloyd's algorithm, tutorial 21)
# --------------------------------------------------------------------------------------------------------------------------------
def lloyd_trace(data, init, max_iter=1000, atol=1e-4):
    """Lloyd's algorithm from the centres ``init``, keeping every iteration; mirrors ``GMMInitializer.kmeans`` (which starts from kpp seeding).

    Per iteration t it records the centres at the start of the iteration (``centroids[t]``), the assignment of the points to them
    (``labels[t]``), the inertia of that assignment (``inertia[t]``) and the centres after the update step (``updated[t]``).
    ``final`` are the centres the library returns.
    """
    c = init.clone()
    k = c.shape[0]
    cents, labs, inertia, upd = [], [], [], []
    for _ in range(max_iter):
        dist = torch.cdist(data, c)
        lab = dist.argmin(dim=1)
        cents.append(c.cpu().numpy())
        labs.append(lab.cpu().numpy())
        inertia.append(float(dist.min(dim=1)[0].pow(2).sum()))
        means, empty = [], []
        for i in range(k):
            member = lab == i
            if bool(member.any()):
                means.append(data[member].mean(dim=0))
            else:
                means.append(None)
                empty.append(i)
        if empty:  # re-seed each empty cluster at the points farthest from their assigned centre
            assigned = dist[torch.arange(data.shape[0], device=data.device), lab]
            order = torch.argsort(assigned, descending=True)
            for n, i in enumerate(empty):
                means[i] = data[order[n % data.shape[0]]].clone()
        new = torch.stack(means)
        upd.append(new.cpu().numpy())
        if torch.allclose(c, new, atol=atol):
            break
        c = new
    return {"centroids": cents, "labels": labs, "inertia": inertia, "updated": upd, "final": c}


PALETTE = [plt.cm.tab10(i) for i in range(10)]
PAD = 0.6
XLIM = (X[:, 0].min() - PAD, X[:, 0].max() + PAD)
YLIM = (X[:, 1].min() - PAD, X[:, 1].max() + PAD)
GX, GY = np.meshgrid(np.linspace(*XLIM, 240), np.linspace(*YLIM, 240))
GRID = np.stack([GX.ravel(), GY.ravel()], axis=1)
GRID_SHAPE = GX.shape


def draw_lloyd(ax, trace, t, show_arrows=True):
    """The state at the start of iteration t: Voronoi cells, points coloured by cluster, centres, update arrows and centre paths."""
    c = trace["centroids"][t]
    k = c.shape[0]
    cells = np.argmin(((GRID[:, None, :] - c[None]) ** 2).sum(-1), axis=1).reshape(GRID_SHAPE)
    ax.contourf(np.linspace(*XLIM, GRID_SHAPE[1]), np.linspace(*YLIM, GRID_SHAPE[0]), cells, levels=np.arange(k + 1) - 0.5,
                colors=[PALETTE[i % 10] for i in range(k)], alpha=0.16)
    lab = trace["labels"][t]
    ax.scatter(X[:, 0], X[:, 1], s=4, c=[PALETTE[i % 10] for i in lab], alpha=0.65, linewidths=0)
    path = np.stack(trace["centroids"][:t + 1])  # (t+1, k, d)
    for i in range(k):
        ax.plot(path[:, i, 0], path[:, i, 1], color=PALETTE[i % 10], lw=1.2, alpha=0.9)
    ax.scatter(c[:, 0], c[:, 1], marker="*", s=240, c=[PALETTE[i % 10] for i in range(k)], edgecolor="k", zorder=5)
    if show_arrows:
        u = trace["updated"][t]
        for i in range(k):
            if np.linalg.norm(u[i] - c[i]) > 1e-3:
                ax.annotate("", xy=u[i], xytext=c[i], arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.3}, zorder=6)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")


def draw_inertia(ax, trace, t):
    """The inertia of every iteration (grey) and the part reached so far (black), with the current iteration marked."""
    J = np.array(trace["inertia"])
    ax.plot(np.arange(len(J)), J, color="0.8", lw=1)
    ax.plot(np.arange(t + 1), J[:t + 1], color="k", lw=1.6, marker="o", ms=3)
    ax.scatter([t], [J[t]], color="magenta", edgecolor="white", s=70, zorder=5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("inertia $J$")
    ax.set_title("the objective, never increasing")
    ax.set_xlim(-0.5, len(J) - 0.5)


def make_kmeans_video():
    trace = lloyd_trace(X_TENSOR, GMMInitializer.kpp(X_TENSOR, K, random_state=SEED))
    n_iter = len(trace["inertia"])
    cap = 40  # at most this many frames; longer runs are subsampled (denser early, where k-means moves most)
    frames = list(range(n_iter)) if n_iter <= cap else [int(t) for t in np.unique(np.round((n_iter - 1) * np.linspace(0, 1, cap) ** 1.6).astype(int))]

    def draw_frame(t):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={"width_ratios": [1.7, 1]})
        draw_lloyd(ax, trace, t)
        ax.set_title(f"iteration {t} of {n_iter - 1}: assignment (colours), then update (arrows)")
        draw_inertia(ax2, trace, t)
        ax2.set_title(f"inertia $J$ = {trace['inertia'][t]:,.0f}")
        fig.tight_layout()
        return fig

    info = {"seed": SEED, "k": K, "iterations": n_iter, "inertia_start": trace["inertia"][0], "inertia_end": trace["inertia"][-1],
            "iterations_shown": frames}
    write_asset("kmeans", draw_frame, frames, info=info, fps=2, hold=6, crf=18)


if __name__ == "__main__":
    make_kpp_video()
    make_kmeans_video()
