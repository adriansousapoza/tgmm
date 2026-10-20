# Design Your Own Mixture

This demo takes a few seconds to load (up to a minute on a slow connection, because the first fit runs in your browser) and runs entirely in your browser (NumPy and matplotlib via Pyodide; no tgmm or PyTorch is involved).
It mirrors how [tutorial 01](../notebooks/01_gmm.ipynb) builds its synthetic data with `tgmm.generate_gmm_data`, and the EM fit it then runs with
[`GaussianMixture`](../api/gaussian-mixture.md) (`covariance_type='full'`).

Each blob is a Gaussian with a centre $\mu_k$, a covariance $\Sigma_k=R(\theta_k)\,\operatorname{diag}(s_{k1}^2,s_{k2}^2)\,R(\theta_k)^{\top}$ (standard
deviations $s_{k1},s_{k2}$ along two axes rotated by $\theta_k$), and a number of points $n_k$, so the mixing weights are $\pi_k=n_k/n$. Points are drawn as
$x=\mu_k+L_kz$ with $z\sim\mathcal N(0,I)$ and $L_kL_k^{\top}=\Sigma_k$ (a Cholesky factor), which is what `generate_gmm_data` does, so the covariance you see in the table is the
covariance you get. Pick a preset, or move the sliders (they update when released); **Reset blobs** returns to the chosen preset. Preset values are rounded to the slider grid, so the tutorial blobs are reproduced to within about 1% for the covariance entries.

<div markdown="0">
```python {marimo}
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt

def from_cov(S):                       # covariance -> (s1, s2, theta in degrees) on the slider grid
    w, V = np.linalg.eigh(np.asarray(S, float))
    th = np.degrees(np.arctan2(V[1, 1], V[0, 1])) % 180
    q = lambda v: round(round(float(v) / 0.005) * 0.005, 3)
    return q(np.sqrt(w[1])), q(np.sqrt(w[0])), (round(float(th) * 2) / 2) % 180

_C = [(0, 2), (2, -2), (0, 0), (2, 2)]
_S = [np.eye(2), 0.5 * np.eye(2), np.array([[2, 0], [0, 0.5]]), np.array([[0.2, 0.5], [0.5, 2]])]
_N = [800, 200, 1000, 1000]
PRESETS = {
    "Tutorial 01 (current)": [(*c, *from_cov(S), n) for c, S, n in zip(_C, _S, _N)],
    "Tutorial 01 before the covariance fix": [(*c, *from_cov(S.T @ S), n) for c, S, n in zip(_C, _S, _N)],
    "Three well separated blobs": [(-4, -3, 0.8, 0.8, 0, 500), (4, -3, 1.2, 0.5, 30, 400), (0, 4, 0.7, 0.7, 0, 300)],
    "Overlapping blobs": [(0, 0, 1.0, 1.0, 0, 500), (1.5, 0.5, 1.0, 1.0, 0, 500), (0.5, 1.5, 1.0, 1.0, 0, 500)],
}
EXTRA = [(-5, 4, 0.6, 0.6, 0, 300), (5, 4, 0.6, 0.6, 0, 300)]          # defaults for blobs beyond the preset's K
POOL = [np.random.default_rng(i).standard_normal((1500, 2)) for i in range(5)]   # fixed normals: sliders never reshuffle points
get_p, set_p = mo.state("Tutorial 01 (current)")
get_tick, set_tick = mo.state(0)
preset = mo.ui.dropdown(list(PRESETS), value="Tutorial 01 (current)", label="preset", on_change=set_p)
reset = mo.ui.button(label="Reset blobs", on_click=lambda _: set_tick(lambda t: t + 1))
mo.hstack([preset, reset], justify="start")
```
</div>

The preset *Tutorial 01 before the covariance fix* uses the squares $\Sigma^{\top}\Sigma$ of the requested matrices: before the fix, `generate_gmm_data` sampled `Z @ cov`, so a requested `cov` really produced $\operatorname{cov}^{\top}\operatorname{cov}$.

<div markdown="0">
```python {marimo}
_ = get_tick()
_P = PRESETS[get_p()]
_spec = [("x", -6, 6, 0.25), ("y", -6, 6, 0.25), ("s₁", 0.05, 3, 0.005), ("s₂", 0.05, 3, 0.005), ("θ (°)", 0, 179.5, 0.5), ("n", 0, 1500, 50)]
sliders = [[mo.ui.slider(lo, hi, step=st, value=v, label=f"{nm}{i + 1}", debounce=True, show_value=True)
            for (nm, lo, hi, st), v in zip(_spec, (_P[i] if i < len(_P) else EXTRA[i - len(_P)]))] for i in range(5)]
kblobs = mo.ui.dropdown([str(k) for k in range(1, 6)], value=str(len(_P)), label="number of blobs K")
kblobs
```
</div>

<div markdown="0">
```python {marimo}
K = int(kblobs.value)
mo.vstack([mo.vstack([mo.md(f"**Blob {i + 1}**"), mo.hstack(sliders[i][:3], justify="start"), mo.hstack(sliders[i][3:], justify="start")])
           for i in range(K)])
```
</div>

<div markdown="0">
```python {marimo}
def cov_of(s1, s2, th):
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([s1 ** 2, s2 ** 2]) @ R.T

blobs = [[w.value for w in sliders[i]] for i in range(K)]
covs = [cov_of(b[2], b[3], b[4]) for b in blobs]
counts = [int(b[5]) for b in blobs]
n_tot = sum(counts)
_kind = lambda b, S: "spherical" if abs(b[2] - b[3]) < 1e-9 else ("diagonal (axis-aligned)" if abs(S[0, 1]) < 1e-9 else "full")
_rows = [f"| {i + 1} | $\\begin{{pmatrix}}{T[0, 0]:.2f}&{T[0, 1]:.2f}\\\\{T[0, 1]:.2f}&{T[1, 1]:.2f}\\end{{pmatrix}}$ | {_kind(b, S)} | {n} | {n / max(n_tot, 1):.2f} |"
         for i, (b, S, T, n) in enumerate(zip(blobs, covs, [np.round(S, 2) + 0.0 for S in covs], counts))]   # T: rounded for display
mo.md("| blob | $\\Sigma_k$ | type | $n_k$ | $\\pi_k$ |\n|---|---|---|---|---|\n" + "\n".join(_rows) + f"\n\nTotal: **{n_tot}** points.")
```
</div>

<div markdown="0">
```python {marimo}
X = np.vstack([np.array(b[:2]) + POOL[i][:n] @ np.linalg.cholesky(S).T for i, (b, S, n) in enumerate(zip(blobs, covs, counts))]) if n_tot else None
y = np.concatenate([np.full(n, i) for i, n in enumerate(counts)]) if n_tot else None

def em(X, k, iters=100, tol=1e-6, reg=1e-6):          # full-covariance EM, k-means++ start (seeded)
    n, d = X.shape
    g = np.random.default_rng(0)
    mu = X[[g.integers(n)]]
    for _ in range(k - 1):
        d2 = ((X[:, None] - mu[None]) ** 2).sum(-1).min(1)
        mu = np.vstack([mu, X[g.choice(n, p=d2 / d2.sum() if d2.sum() > 0 else None)]])
    for _ in range(5):
        a = ((X[:, None] - mu[None]) ** 2).sum(-1).argmin(1)
        mu = np.array([X[a == j].mean(0) if (a == j).any() else mu[j] for j in range(k)])
    pi, S, hist = np.full(k, 1 / k), np.stack([np.cov(X.T) + reg * np.eye(d)] * k), []
    for it in range(iters):
        L = np.linalg.cholesky(S)
        z = np.linalg.solve(L, (X[:, None] - mu).transpose(1, 2, 0))                  # (k, d, n)
        lp = np.log(pi)[:, None] - 0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1))[:, None] - 0.5 * (z ** 2).sum(1)
        m = lp.max(0)
        ll = m + np.log(np.exp(lp - m).sum(0))
        hist.append(ll.sum())
        r = np.exp(lp - ll)
        if it and abs(hist[-1] - hist[-2]) < tol * n:
            break
        nk = r.sum(1) + 1e-12
        mu, pi = (r @ X) / nk[:, None], nk / n
        dx = X[None] - mu[:, None]
        S = np.einsum("kn,knd,kne->kde", r, dx, dx) / nk[:, None, None] + reg * np.eye(d)
    return dict(mu=mu, S=S, pi=pi, r=r, ll=hist[-1], it=it + 1, bic=-2 * hist[-1] + (k * d + k * d * (d + 1) // 2 + k - 1) * np.log(n))

fits = {k: em(X, k) for k in range(1, 7)} if n_tot >= 12 else None
```
</div>

## Fit a mixture to your data

<div markdown="0">
```python {marimo}
kfit = mo.ui.slider(1, 6, value=int(kblobs.value), label="components to fit", debounce=True, show_value=True)
kfit
```
</div>

<div markdown="0">
```python {marimo}
if fits is None:
    _out = mo.md("**Add at least 12 points** (raise some $n_k$) to fit a mixture.")
else:
    _t = np.linspace(0, 2 * np.pi, 120)
    _ring = np.stack([np.cos(_t), np.sin(_t)])
    _col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    _f = fits[kfit.value]
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for _i, _S in enumerate(covs):                                           # left: what you designed
        _m = y == _i
        _ax[0].scatter(X[_m, 0], X[_m, 1], s=5, alpha=0.5, color=_col[_i])
    _ax[0].set_title(f"the data you designed (K = {K}, n = {n_tot})")
    _lab = _f["r"].argmax(0)
    for _j in range(kfit.value):                                             # right: the EM fit
        _m = _lab == _j
        _ax[1].scatter(X[_m, 0], X[_m, 1], s=5, alpha=0.5, color=_col[_j])
        for _s in (1, 2):
            _e = _f["mu"][_j][:, None] + _s * np.linalg.cholesky(_f["S"][_j]) @ _ring
            _ax[1].plot(_e[0], _e[1], color="k", lw=1.2 if _s == 1 else 0.7)
    _ax[1].set_title(f"EM fit with {kfit.value} components (1σ and 2σ)")
    for _a in _ax:
        _a.set_aspect("equal")
    _out = _fig
_out
```
</div>

<div markdown="0">
```python {marimo}
if fits is None:
    _msg = ""
else:
    _best = min(fits, key=lambda k: fits[k]["bic"])
    _f = fits[kfit.value]
    _msg = (f"Fit with **{kfit.value}** components: log-likelihood per sample **{_f['ll'] / n_tot:.3f}**, BIC **{_f['bic']:.1f}**, EM iterations **{_f['it']}**.\n\n"
            f"BIC for K = 1…6: " + ", ".join(f"{k}: {fits[k]['bic']:.0f}" for k in fits) + f". **BIC picks K = {_best}** (you designed {K} blobs).")
mo.md(_msg)
```
</div>

What to look for: with the *well separated* preset the BIC minimum sits at the true $K=3$, and fitting fewer components merges blobs while fitting more splits one of them. With the
*overlapping* preset BIC typically prefers fewer components than you designed, because heavily overlapping blobs look like a single broader one: the same
effect that makes the choice of $K$ hard in practice (see the [model selection tutorial](../notebooks/13_model_selection.ipynb)). Make one blob tiny (a small $n_k$) and watch it disappear from the BIC preference first.
The code budget for this page is about 150 lines of Python.
