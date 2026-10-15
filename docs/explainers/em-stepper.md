# EM Step by Step

This demo takes a few seconds to load and runs entirely in your browser (NumPy and matplotlib via Pyodide; no tgmm or PyTorch is involved).
It mirrors the E- and M-steps of [`GaussianMixture`](../user-guide/gaussian-mixture.md#e-step-responsibilities) in one dimension, together with
its `algorithm='em' | 'cem' | 'sem'` options (see [Fitting algorithms](../user-guide/gaussian-mixture.md#fitting-algorithms)).

Two Gaussian components are fitted to 200 fixed points, starting from deliberately bad means. **E-step** computes the responsibilities

$$r_{ik}=\frac{\pi_k\,\mathcal N(x_i\mid\mu_k,\sigma_k^2)}{\sum_j\pi_j\,\mathcal N(x_i\mid\mu_j,\sigma_j^2)},$$

and **M-step** re-estimates the parameters from them, with $n_k=\sum_i r_{ik}$:

$$\pi_k=\frac{n_k}{n},\qquad \mu_k=\frac{1}{n_k}\sum_i r_{ik}x_i,\qquad \sigma_k^2=\frac{1}{n_k}\sum_i r_{ik}(x_i-\mu_k)^2.$$

The algorithms differ only in the E-step's output: **EM** keeps the soft $r_{ik}$, **CEM** replaces them by the hard $\arg\max$ assignment, **SEM** *samples* each
assignment from $r_{i\cdot}$ (seeded, so the page is reproducible). The M-step is applied to whatever the E-step produced.

<div markdown="0">
```python {marimo}
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt

_g = np.random.default_rng(0)
x = np.concatenate([_g.normal(0.0, 1.0, 120), _g.normal(4.0, 0.8, 80)])
n = len(x)

def pdf(z, mu, sig):
    return np.exp(-0.5 * ((z[:, None] - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

def loglik(p):
    return float(np.log((p["pi"] * pdf(x, p["mu"], p["sig"])).sum(1)).sum())

def e_step(p, mode):
    dens = p["pi"] * pdf(x, p["mu"], p["sig"])
    r = dens / dens.sum(1, keepdims=True)
    if mode == "CEM":
        r = np.eye(2)[r.argmax(1)]
    elif mode == "SEM":                       # sample z_i ~ Categorical(r_i), seeded by the draw counter
        r = np.eye(2)[(np.random.default_rng(100 + p["draw"]).random(n) > r[:, 0]).astype(int)]
    return {**p, "r": r, "fresh": True, "draw": p["draw"] + 1}

def m_step(p):
    r = p["r"]; nk = r.sum(0); ok = nk > 1e-9; nk = np.where(ok, nk, 1.0)
    mu = np.where(ok, (r * x[:, None]).sum(0) / nk, p["mu"])
    sig = np.where(ok, np.maximum(np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk), 0.05), p["sig"])
    q = {**p, "mu": mu, "sig": sig, "pi": np.where(ok, nk / n, 1e-9), "fresh": False, "it": p["it"] + 1}
    return {**q, "ll": p["ll"] + [loglik(q)]}

def step(p, mode, kind):
    if kind == "run":
        for _ in range(10):
            p = m_step(e_step(p, mode))
        return p
    if kind == "E" or not p["fresh"]:
        p = e_step(p, mode)
    return m_step(p) if kind == "M" else p

INIT = {"mu": np.array([1.5, 2.5]), "sig": np.array([1.5, 1.5]), "pi": np.array([0.5, 0.5]),
        "r": None, "fresh": False, "it": 0, "draw": 0}
INIT["ll"] = [loglik(INIT)]
get_s, set_s = mo.state(INIT)
```
</div>

<div markdown="0">
```python {marimo}
mode = mo.ui.dropdown(["EM", "CEM", "SEM"], value="EM", label="algorithm", on_change=lambda _: set_s(INIT))
b_e = mo.ui.button(label="E-step", on_click=lambda _: set_s(lambda p: step(p, mode.value, "E")))
b_m = mo.ui.button(label="M-step", on_click=lambda _: set_s(lambda p: step(p, mode.value, "M")))
b_run = mo.ui.button(label="Run 10", on_click=lambda _: set_s(lambda p: step(p, mode.value, "run")))
b_reset = mo.ui.button(label="Reset", on_click=lambda _: set_s(INIT))
mo.hstack([mode, b_e, b_m, b_run, b_reset], justify="start")
```
</div>

Changing the algorithm resets the fit. **M-step** first runs an E-step if the responsibilities are stale, so *E then M* is one EM iteration, and **Run 10** does ten.

<div markdown="0">
```python {marimo}
_p = get_s()
_g = np.linspace(-4, 8, 400)
_bins = np.linspace(-4, 8, 41)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
if _p["r"] is None:
    ax1.hist(x, _bins, color="0.7")
else:
    ax1.hist([x, x], _bins, weights=[_p["r"][:, 0], _p["r"][:, 1]], stacked=True, color=["#2a78d6", "#eb6834"])
_dens = _p["pi"] * pdf(_g, _p["mu"], _p["sig"]) * n * (_bins[1] - _bins[0])
for _k, _c in enumerate(["#2a78d6", "#eb6834"]):
    ax1.plot(_g, _dens[:, _k], "--", color=_c)
ax1.plot(_g, _dens.sum(1), color="k")
ax1.set_title("data coloured by responsibility; fitted densities")
ax2.plot(_p["ll"], "o-", color="k")
ax2.set_xlabel("iteration"); ax2.set_ylabel("log-likelihood"); ax2.set_title(f"log-likelihood trace ({mode.value})")
fig
```
</div>

<div markdown="0">
```python {marimo}
_q = get_s()
_next = "M-step (responsibilities are fresh)" if _q["fresh"] else "E-step"
mo.md("| | component 1 | component 2 |\n|---|---|---|\n"
      + f"| $\\mu_k$ | {_q['mu'][0]:.2f} | {_q['mu'][1]:.2f} |\n| $\\sigma_k$ | {_q['sig'][0]:.2f} | {_q['sig'][1]:.2f} |\n"
      + f"| $\\pi_k$ | {_q['pi'][0]:.2f} | {_q['pi'][1]:.2f} |\n\n"
      + f"Iterations: **{_q['it']}**; log-likelihood **{_q['ll'][-1]:.2f}**; next step: **{_next}**.")
```
</div>

What to look for: with **EM** the log-likelihood never decreases, and from this symmetric-ish start it first crawls along a plateau before climbing; **CEM** jumps to a (crisp, hard-partition) fixed point within a step or two and then stops changing;
**SEM** keeps jittering around a stationary level because each assignment is resampled (that is why `tgmm` never declares SEM converged and offers `'saem'` for a convergent variant).
