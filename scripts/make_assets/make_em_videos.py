"""Render the two EM videos of tutorial 01 (notebooks/assets/em_1d.mp4 and em_2d.mp4).

    python scripts/make_assets/make_em_videos.py        # from the repository root; or: make assets

The tutorial only *loads* the finished videos. Both simulations are deterministic (fixed seeds); run the script again after changing it.

* ``em_1d``: EM on a 1-D mixture of three Gaussians from a deliberately bad start, one E/M step at a time; each frame shows the fit against
  the real components, the responsibilities, the average log-likelihood and, for each component, its log-likelihood surface over
  (mean, std) with the path EM has taken.
* ``em_2d``: EM on the four default blobs (notebooks/tutorial_data.py), started with all components close together around (0, 0);
  each frame shows the responsibilities, the hard assignments, and the means trace with one 95% covariance ellipse per component.
"""
import pathlib
import sys
import warnings

import numpy as np
import torch
from matplotlib.patches import Ellipse
from scipy.stats import chi2, norm

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))  # tutorial_data
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # _video

import matplotlib.pyplot as plt  # noqa: E402

import _video  # noqa: E402
from tgmm import GaussianMixture  # noqa: E402  (importing tgmm applies the plotting theme the tutorials use)
from tutorial_data import make_default_data  # noqa: E402

warnings.filterwarnings("ignore", message=".*EM did not converge.*")

SEED = 0
COLORS = ["#e41a1c", "#ff7f00", "#984ea3", "#4daf4a", "#377eb8", "#a65628"]  # one colour per 2-D component


# ------------------------------------------------------------------------------------------------------------------ 1-D example
def simulate_1d(max_iter=1000, grid=50):
    """The 1-D example of the tutorial: three Gaussians, a bad start (all means near 10), EM one E/M step at a time."""
    real_means, real_stds, sizes = [-2, 2, 6], [2, 1, 3], [2000, 1000, 3000]
    z = np.random.RandomState(SEED).standard_normal(12000)[6000:]  # the same fixed slice of a seeded stream the notebook plots
    data = np.concatenate([real_means[0] + real_stds[0] * z[:2000], real_means[1] + real_stds[1] * z[2000:3000], real_means[2] + real_stds[2] * z[3000:6000]])
    true_taus = [s / sum(sizes) for s in sizes]

    X = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)
    init_means = torch.tensor([[9.5], [10.0], [10.5]], dtype=torch.float32)
    init_cov = torch.tensor([[[2.5 ** 2]], [[2.5 ** 2]], [[2.5 ** 2]]], dtype=torch.float32)
    init_w = torch.tensor([0.33, 0.33, 0.33], dtype=torch.float32)
    gmm = GaussianMixture(n_features=1, n_components=3, covariance_type="full", init_means=init_means, init_covariances=init_cov,
                          init_weights=init_w, device="cpu")
    gmm._allocate_parameters(X, set_random_state=False)
    means, stds, taus = [init_means.squeeze().numpy()], [np.sqrt(init_cov[:, 0, 0].numpy())], [init_w.numpy()]  # state after iteration 0, 1, 2, ...
    resp, lp = gmm._e_step(X)
    ll = [lp.mean().item()]
    converged = False
    for _ in range(max_iter):
        gmm._m_step(X, resp)
        resp, lp = gmm._e_step(X)
        means.append(gmm.means_.squeeze().numpy())
        stds.append(torch.sqrt(gmm.covariances_[:, 0, 0]).numpy())
        taus.append(gmm.weights_.numpy())
        ll.append(lp.mean().item())
        if abs(ll[-1] - ll[-2]) < 1e-9:
            converged = True
            break
    means, stds, taus, ll = np.array(means), np.array(stds), np.array(taus), np.array(ll)

    def loglik(m, s, w):
        return np.sum(np.log(sum(w[k] * norm.pdf(data, m[k], s[k]) for k in range(3))))

    mean_range = np.linspace(data.min(), data.max(), grid)
    std_range = np.linspace(0.01, 5, grid)
    surfaces = [np.zeros((grid, grid)) for _ in range(3)]
    for i, m in enumerate(mean_range):
        for j, s in enumerate(std_range):
            for k in range(3):  # vary component k's (mean, std); the others stay at their converged values
                mk, sk = list(means[-1]), list(stds[-1])
                mk[k], sk[k] = m, s
                surfaces[k][j, i] = loglik(mk, sk, taus[-1])
    return dict(data=data, real_means=real_means, real_stds=real_stds, true_taus=true_taus, means=means, stds=stds, taus=taus, ll=ll,
                mean_range=mean_range, std_range=std_range, surfaces=surfaces, converged=converged)


def make_draw_1d(sim):
    data, means, stds, taus, ll = sim["data"], sim["means"], sim["stds"], sim["taus"], sim["ll"]
    real_means, real_stds, true_taus = sim["real_means"], sim["real_stds"], sim["true_taus"]
    hist_peak = np.histogram(data, bins=30, density=True)[0].max()

    def draw(t):
        colors = ["r", "b", "g"]
        fig = plt.figure(figsize=(11, 6.4))
        gs = fig.add_gridspec(2, 3)
        ax_fit, ax_resp, ax_ll = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
        x = np.linspace(data.min(), data.max(), 1000)
        ax_fit.hist(data, bins=30, density=True, alpha=0.6, color="g", label="Data histogram")
        est, real = 0, 0
        for k in range(3):
            comp = taus[t, k] * norm.pdf(x, means[t, k], stds[t, k])
            ax_fit.plot(x, comp, color=colors[k], linestyle="-", label=f"Gaussian {k + 1} (estimated)")
            ax_fit.plot(x, true_taus[k] * norm.pdf(x, real_means[k], real_stds[k]), color=colors[k], linestyle="--", label=f"Gaussian {k + 1} (real)")
            est = est + comp
            real = real + true_taus[k] * norm.pdf(x, real_means[k], real_stds[k])
        ax_fit.plot(x, est, "k-", label="GMM (estimated)")
        ax_fit.plot(x, real, "k--", label="GMM (real)")
        ax_fit.set_title("Data and Gaussian Distributions")
        ax_fit.set_ylim(0, 1.7 * max(hist_peak, real.max()))  # headroom for the legend
        ax_fit.legend(ncol=2, loc="upper left", handlelength=1.6, columnspacing=0.9)
        # responsibilities r_ik of the parameters after iteration t (the E-step formula, evaluated directly)
        dens = taus[t] * norm.pdf(data[:, None], means[t], stds[t])
        r = dens / np.maximum(dens.sum(axis=1, keepdims=True), 1e-300)
        for k in range(3):
            ax_resp.scatter(data, r[:, k], color=colors[k], label=f"Responsibility of cluster {k + 1}", alpha=0.6, marker="x", s=12)
        ax_resp.set_title("Responsibilities")
        ax_resp.legend()
        # average log-likelihood after each iteration so far (the quantity EM never decreases)
        ax_ll.plot(ll, color="lightgray", linewidth=1)
        ax_ll.plot(np.arange(t + 1), ll[:t + 1], color="k", linewidth=1.5, label="Up to this iteration")
        ax_ll.scatter([t], [ll[t]], color="orange", edgecolor="white", linewidths=0.5, s=32, marker="8", zorder=5, label="Current iteration")
        lo, hi = ll[min(2, len(ll) - 1):].min(), ll.max()  # zoom past the (terrible) initial guess so the slow climb is visible
        pad = 0.1 * max(hi - lo, 1e-3)
        ax_ll.set_ylim(lo - pad, hi + pad)
        ax_ll.set_xlim(0, len(ll) - 1)
        if ll[0] < lo - pad:
            ax_ll.text(0.07, 0.04, f"iteration 0: {ll[0]:.2f} (off scale)", transform=ax_ll.transAxes, fontsize=7, va="bottom")
        ax_ll.set_title("Average log-likelihood")
        ax_ll.set_xlabel("EM iteration")
        ax_ll.set_ylabel("Average log-likelihood")
        ax_ll.legend(loc="center right")
        for k in range(3):
            ax = fig.add_subplot(gs[1, k])
            cs = ax.contourf(sim["mean_range"], sim["std_range"], sim["surfaces"][k], levels=15, cmap="viridis")
            fig.colorbar(cs, ax=ax).ax.set_ylabel("Log-Likelihood")
            ax.scatter(real_means[k], real_stds[k], color="black", edgecolor="white", linewidths=0.5, label="Real mean and std", s=20, marker="D")
            ax.scatter(means[0, k], stds[0, k], color="red", label="Initial mean and std", s=28, marker="x", linewidths=1.1)
            ax.plot(means[:t + 1, k], stds[:t + 1, k], color="red", linewidth=1.2, label="Path up to this iteration")
            ax.scatter(means[t, k], stds[t, k], color="orange", edgecolor="white", linewidths=0.5, s=32, marker="8", zorder=5, label="Current estimate")
            ax.set_title(f"Log-Likelihood vs. Mean{k + 1} and Std{k + 1}")
            ax.set_xlabel("Mean")
            ax.set_ylabel("Standard Deviation")
            ax.legend(loc="lower left")
        fig.suptitle(f"EM iteration {t} of {len(ll) - 1}")  # the log-likelihood has its own panel
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    return draw


# ------------------------------------------------------------------------------------------------------------------ 2-D example
JITTER_STD = 0.2     # the components start at (0, 0) plus this much seeded noise
JITTER_SEED = 0


def simulate_2d(max_iter=1000):
    """EM on the default blobs, started with all four components close together around (0, 0); the state is recorded after every iteration."""
    X, y = make_default_data(SEED, "cpu")
    K = 4
    g = torch.Generator().manual_seed(JITTER_SEED)
    init_means = JITTER_STD * torch.randn(K, 2, generator=g)
    init_cov = torch.cov(X.T).unsqueeze(0).repeat(K, 1, 1)  # every component starts as wide as the whole data set
    gmm = GaussianMixture(n_features=2, n_components=K, covariance_type="full", init_means=init_means, init_covariances=init_cov,
                          init_weights=torch.full((K,), 1.0 / K), device="cpu")
    gmm._allocate_parameters(X)
    resp, lp = gmm._e_step(X)
    prev = lp.mean().item()
    states = {0: (gmm.means_.clone(), gmm.covariances_.clone(), resp.clone())}
    ll = {0: prev}
    converged = False
    for it in range(1, max_iter + 1):
        gmm._m_step(X, resp)
        resp, lp = gmm._e_step(X)
        cur = lp.mean().item()
        states[it] = (gmm.means_.clone(), gmm.covariances_.clone(), resp.clone())
        ll[it] = cur
        if abs(cur - prev) < 1e-6:
            converged = True
            break
        prev = cur
    return dict(X=X.numpy(), states=states, ll=ll, init_means=init_means.numpy(), converged=converged)


def make_draw_2d(sim):
    X, states, ll = sim["X"], sim["states"], sim["ll"]
    last = max(states)

    DASH = (0, (4, 1.5))
    R95 = float(chi2.ppf(0.95, df=2)) ** 0.5  # Mahalanobis radius of the ellipse that encloses 95% of a 2-D Gaussian: 2.4477 standard deviations
    def draw(t):
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.9), squeeze=False)
        axs = axs[0]
        means, covs, resp = states[t]
        K = means.shape[0]
        for k in range(K):
            axs[0].plot(resp[:, k].numpy(), label=f"Cluster {k + 1}", color=COLORS[k], marker=".", markersize=2, linestyle="None", alpha=0.5)
        axs[0].legend(markerscale=4)
        axs[0].set_title(f"Responsibilities, iteration {t}")
        axs[0].set_xlabel("Data point index")
        axs[0].set_ylabel("Probability")
        # colour each point by the component with the highest responsibility
        assignment = resp.max(dim=1)[1].numpy()
        axs[1].scatter(X[:, 0], X[:, 1], c=[COLORS[c] for c in assignment], s=10, alpha=0.5)
        axs[1].set_title(f"Hard assignments, iteration {t}")
        axs[1].set_xlabel("X1")
        axs[1].set_ylabel("X2")
        # data (small, black), trace of the means, and the 95% ellipse of every component
        axs[2].scatter(X[:, 0], X[:, 1], c="black", s=1.5, alpha=0.45, linewidths=0)
        for k in range(K):
            eigvals, eigvecs = torch.linalg.eigh(covs[k])
            angle = torch.atan2(eigvecs[1, 0], eigvecs[0, 0]).item() * 180.0 / torch.pi
            width, height = 2 * torch.sqrt(eigvals).numpy()
            axs[2].add_patch(Ellipse(means[k].numpy(), R95 * width, R95 * height, angle=angle, edgecolor=COLORS[k], facecolor="none", lw=1.3, linestyle=DASH))
            trace = torch.stack([states[i][0][k] for i in range(t + 1)]).numpy()
            axs[2].plot(trace[:, 0], trace[:, 1], color=COLORS[k], linestyle=(0, (1.5, 2.5)), lw=0.7, alpha=0.9)
            axs[2].scatter(trace[-1, 0], trace[-1, 1], color=COLORS[k], marker="o", s=7, edgecolor="white", linewidths=0.3, zorder=5)
            axs[2].scatter(trace[0, 0], trace[0, 1], color=COLORS[k], marker="x", s=13, linewidths=0.7, zorder=5)
        axs[2].set_title(f"GMM, iteration {t}")
        axs[2].set_xlabel("X1")
        axs[2].set_ylabel("X2")
        axs[2].set_xlim(-8.5, 8.5)
        axs[2].set_ylim(-4.5, 8.5)
        axs[2].scatter([], [], color="black", marker="x", s=13, linewidths=0.7, label="Initial means")
        axs[2].scatter([], [], color="black", marker="o", s=7, label="Current means")
        axs[2].plot([], [], color="black", linestyle=DASH, lw=1.3, label="95% region of each component")
        axs[2].plot([], [], color="black", linestyle=(0, (1.5, 2.5)), lw=0.7, label="Trace of means")
        axs[2].legend(loc="upper left", fontsize=6)
        fig.suptitle(f"EM iteration {t} of {last}: average log-likelihood {ll[t]:.4f}")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        return fig

    return draw


def main():
    sim1 = simulate_1d()
    n1 = len(sim1["ll"]) - 1
    _video.write_asset("em_1d", make_draw_1d(sim1), _video.pick_frames(n1), info=dict(
        iterations=n1, converged=sim1["converged"], ll_first=float(sim1["ll"][0]), ll_last=float(sim1["ll"][-1]),
        n_points=int(len(sim1["data"])), final_means=sim1["means"][-1].tolist(), final_stds=sim1["stds"][-1].tolist(), final_weights=sim1["taus"][-1].tolist(),
        largest_drop=float(max(0.0, -np.diff(sim1["ll"]).min()))))

    sim2 = simulate_2d()
    n2 = max(sim2["states"])
    _video.write_asset("em_2d", make_draw_2d(sim2), _video.pick_frames(n2, max_frames=40), crf=24, info=dict(
        iterations=n2, converged=sim2["converged"], ll_first=float(sim2["ll"][0]), ll_last=float(sim2["ll"][n2]),
        init_means=sim2["init_means"].tolist(), jitter_std=JITTER_STD, jitter_seed=JITTER_SEED))


if __name__ == "__main__":
    main()
