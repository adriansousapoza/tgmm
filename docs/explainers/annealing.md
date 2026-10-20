# Deterministic Annealing

<!-- Mirrors `GaussianMixture(..., temperature=(beta_0, n_anneal))`; see
     ../user-guide/gaussian-mixture.md#deterministic-annealing-daem (E-step: tgmm/mixtures/base.py `_e_step_chunk`,
     schedule: `_beta_for_iteration`). Reference NumPy code, not tgmm. -->

This demo takes a few seconds to load and runs entirely in your browser (NumPy, SciPy and matplotlib via
Pyodide; no tgmm or PyTorch involved).

EM can get stuck when its components start on top of each other. **Deterministic annealing EM** (Ueda &
Nakano, 1998) tempers the E-step with an inverse temperature $\beta\in(0,1]$:

$$
r_{ik}(\beta)=\frac{\big(\pi_k\,\mathcal N(x_i\mid\mu_k,\Sigma_k)\big)^{\beta}}{\sum_j\big(\pi_j\,\mathcal N(x_i\mid\mu_j,\Sigma_j)\big)^{\beta}},
\qquad
\beta_t=\begin{cases}\beta_0^{\,1-t/n_{\text{anneal}}}, & t<n_{\text{anneal}}\\ 1, & t\ge n_{\text{anneal}}.\end{cases}
$$

Small $\beta$ spreads every point's responsibility over all components (a smooth, nearly unimodal objective); $\beta$ then
rises geometrically to 1 over $n_{\text{anneal}}=30$ iterations. Below, three Gaussians are fitted to three well-separated
blobs from a **saddle start**: all three means are jittered (σ = 0.3) around the *same* blob, each with covariance $I$ -- a
classic EM failure mode (a poor local optimum that plain EM never leaves). Pick
$\beta_0$ and the algorithm, then press the button to restart from a fresh jitter.

<div markdown="0">
```python {marimo}
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

_rng = np.random.default_rng(0)
X = np.vstack([_rng.normal(c, 1.0, (150, 2)) for c in [(0, 0), (8, 0), (4, 7)]])  # fixed data, 3 blobs
K, N_ANNEAL, N_ITER = 3, 30, 150

def log_norm(X, mu, S):
    L = np.linalg.cholesky(S); z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (z ** 2).sum(0) - np.log(np.diag(L)).sum() - np.log(2 * np.pi)

def fit(hard, beta0, seed):
    """EM (hard=False) or CEM (hard=True) with the DAEM schedule and tgmm's stopping rule; collapsed init. Returns means, covs, labels, entropy trace, final mean log-lik."""
    mu = 0.3 * np.random.default_rng(seed).normal(size=(K, 2))  # all three means start on top of blob 0
    S, pi, H = np.stack([np.eye(2)] * K), np.full(K, 1 / K), []
    beta = lambda t: 1.0 if t >= N_ANNEAL or beta0 >= 1 else beta0 ** (1 - t / N_ANNEAL)
    def e_step(b):
        lp = np.stack([np.log(pi[k]) + log_norm(X, mu[k], S[k]) for k in range(K)], 1)
        t = b * lp; r = np.exp(t - t.max(1, keepdims=True))
        return r / r.sum(1, keepdims=True), np.logaddexp.reduce(lp, 1).mean()
    r, ll = e_step(beta(0)); prev = -np.inf
    for t in range(N_ITER):
        H.append(-(r * np.log(r + 1e-12)).sum(1).mean())
        if hard: r = np.eye(K)[r.argmax(1)]            # C-step: argmax is unchanged by beta > 0
        nk = r.sum(0) + 1e-9; pi, mu = nk / len(X), (r.T @ X) / nk[:, None]
        S = np.stack([(r[:, k, None] * (X - mu[k])).T @ (X - mu[k]) / nk[k] + 1e-6 * np.eye(2) for k in range(K)])
        if beta(t) >= 1 and abs(ll - prev) / (abs(prev) + 1e-20) < 1e-4: break   # tgmm: convergence only checked at beta = 1
        prev = ll; r, ll = e_step(beta(t + 1))
    r, ll = e_step(1.0)
    return mu, S, r.argmax(1), H, ll
```
</div>

<div markdown="0">
```python {marimo}
beta0 = mo.ui.slider(0.05, 1.0, step=0.05, value=0.7, label="starting temperature β₀", debounce=True)
algo = mo.ui.dropdown(["EM", "CEM"], value="EM", label="algorithm")
run = mo.ui.button(value=0, on_click=lambda v: v + 1, label="Run from collapsed init")
mo.hstack([beta0, algo, run], justify="start")
```
</div>

<div markdown="0">
```python {marimo}
res = {"annealed": fit(algo.value == "CEM", beta0.value, 2 + run.value),
       "plain": fit(algo.value == "CEM", 1.0, 2 + run.value)}
res["ll"] = (algo.value, beta0.value, 2 + run.value, res["annealed"][4], res["plain"][4])
```
</div>

<div markdown="0">
```python {marimo}
def draw(ax, out, title):
    mu, S, lab, _, ll = out
    ax.scatter(*X.T, c=lab, cmap="viridis", s=8, alpha=0.6)
    for m, C in zip(mu, S):
        w, V = np.linalg.eigh(C)
        ax.add_patch(Ellipse(m, *(4 * np.sqrt(w)), angle=np.degrees(np.arctan2(V[1, 0], V[0, 0])), fill=False, lw=2, color="k"))
    ax.set_title(f"{title}\nmean log-likelihood {ll:.3f}"); ax.set_aspect("equal")
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
draw(axs[0], res["plain"], "β = 1 (plain " + algo.value + ")")
draw(axs[1], res["annealed"], f"annealed, β₀ = {beta0.value:g}")
axs[2].plot(res["plain"][3], label="β = 1"); axs[2].plot(res["annealed"][3], label="annealed")
axs[2].axvline(N_ANNEAL, color="0.6", ls=":"); axs[2].set_xlabel("iteration"); axs[2].set_ylabel("mean responsibility entropy (nats)")
axs[2].set_title("how diffuse the assignments are"); axs[2].legend()
fig
```
</div>

<div markdown="0">
```python {marimo}
get_log, set_log = mo.state([])
```
</div>

<div markdown="0">
```python {marimo}
set_log(lambda log: (log + [res["ll"]])[-6:])
```
</div>

<div markdown="0">
```python {marimo}
_gain = res["annealed"][4] - res["plain"][4]
mo.vstack([mo.md(f"**Final mean log-likelihood:** annealed {res['annealed'][4]:.3f} vs. β = 1: {res['plain'][4]:.3f} "
                 f"(difference {_gain:+.3f}). " + ("Hard assignments: the argmax of the tempered posterior equals the argmax of the plain one, so β changes nothing for CEM."
                                                   if algo.value == "CEM" else "")),
           mo.md("**Recent runs** (algorithm, β₀, jitter seed, annealed LL, β = 1 LL)"),
           mo.ui.table([dict(zip(("algo", "β₀", "seed", "annealed", "β=1"), (a, b, s, round(x, 3), round(y, 3)))) for a, b, s, x, y in get_log()], selection=None)])
```
</div>

## What to look for

* **EM, β₀ ≈ 0.7 (default):** the tempered E-step lets the components see more than the blob they started on, and this
  jitter ends in the three-blob solution (higher log-likelihood), while plain EM at $\beta=1$ settles in a poor local
  optimum. The right-hand plot shows how the entropy of the assignments evolves as $\beta\to1$ (dotted line: end of annealing).
* **It is a heuristic, not a guarantee.** Press the button a few times: with other jitters annealing ties, or even loses
  to plain EM. And **too hot a start collapses the fit**: for $\beta_0\lesssim0.3$ the means are pulled together until all
  three components describe the same blob (log-likelihood far below either alternative). A near-symmetric state
  stays symmetric under tempering, so annealing needs some real asymmetry in the start to amplify.
* **CEM:** assignments are hard, and $\arg\max_k r_{ik}(\beta)$ does not depend on $\beta>0$, so annealing has no effect
  (this is also what `tgmm` does: the C-step follows the tempered E-step).
* This page mirrors `tgmm`'s loop: the convergence test is only applied once $\beta=1$, and the reported value is the true
  mean log-likelihood of the final fit (while $\beta<1$, `tgmm` records the *tempered* normaliser).

See the [Stochastic EM, Annealing & SAEM tutorial](../notebooks/14_stochastic_em.ipynb) for the same idea with `tgmm`.
