# Clustering Metrics and Model Selection

<!-- Mirrors `tgmm.metrics.ClusteringMetrics` (silhouette_score, calinski_harabasz_score, davies_bouldin_index, dunn_index,
     wb_index) and `GaussianMixture.bic/icl`; see ../user-guide/clustering-metrics.md and
     ../user-guide/gaussian-mixture.md#model-selection. Reference NumPy code, not tgmm. -->

This demo takes a few seconds to load and runs entirely in your browser (NumPy, SciPy and matplotlib via Pyodide; no
tgmm or PyTorch involved).

Three Gaussian blobs, a k-means clustering with the $K$ you choose, and every index of
[`ClusteringMetrics`](../user-guide/clustering-metrics.md) recomputed live with the same formulas. Move the
**separation** slider to watch the indices degrade as the blobs overlap, change $K$ to see which indices are fooled, and
switch on **noise** (uniform background points). tgmm treats noise by *dropping* the points labelled as noise and scaling
the index by $f=(n-n_{\text{noise}})/n$ (multiplied for higher-is-better, divided for lower-is-better indexes); the second
column below does that, the first lets k-means swallow the noise into clusters.

<div markdown="0">
```python {marimo}
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt

def blobs(sep, noise):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 1.0, (80, 2)) for c in np.array([[0, 0], [1, 0], [0.5, 0.87]]) * sep]); y = np.repeat([0, 1, 2], 80)
    if noise:  # 30 uniform background points, label -1
        X = np.vstack([X, rng.uniform(X.min(0) - 3, X.max(0) + 3, (30, 2))]); y = np.r_[y, -np.ones(30, int)]
    return X, y

def kmeans(X, k):
    C = X[np.random.default_rng(1).choice(len(X), k, replace=False)]
    for _ in range(50):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab

def indices(X, lab, noise_label=-1):
    keep = lab != noise_label; Xk, l = X[keep], np.unique(lab[keep], return_inverse=True)[1]; f = keep.mean(); K = l.max() + 1
    D = np.linalg.norm(Xk[:, None] - Xk[None], axis=-1); cnt = np.bincount(l, minlength=K); i = np.arange(len(l))
    sums = np.stack([D[:, l == j].sum(1) for j in range(K)], 1); a = sums[i, l] / np.maximum(cnt[l] - 1, 1)
    m = sums / cnt; m[i, l] = np.inf; b = m.min(1)
    sil = np.where(cnt[l] <= 1, 0, (b - a) / np.maximum(a, b)).mean()
    C = np.stack([Xk[l == j].mean(0) for j in range(K)]); own = np.linalg.norm(Xk - C[l], axis=1)
    ssw, ssb = (own ** 2).sum(), (cnt * ((C - Xk.mean(0)) ** 2).sum(1)).sum()
    M = np.linalg.norm(C[:, None] - C[None], axis=-1); R = (np.bincount(l, own, K) / cnt)[:, None] + (np.bincount(l, own, K) / cnt)[None]
    R = R / np.where(M == 0, np.inf, M); np.fill_diagonal(R, -np.inf)
    same = l[:, None] == l[None]
    up = {"Silhouette": sil, "Calinski-Harabasz": ssb / (K - 1) / (ssw / (len(l) - K)), "Dunn": D[~same].min() / D[same].max()}
    down = {"Davies-Bouldin": R.max(1).mean(), "WB-index": K * ssw / ssb}
    return {**{k: v * f for k, v in up.items()}, **{k: v / f for k, v in down.items()}}

def bic_icl(X, k):
    """EM for a k-component full-covariance GMM (k-means init); returns BIC and hard-ICL (lower is better)."""
    n, d = X.shape; r = np.eye(k)[kmeans(X, k)]; prev = -np.inf
    for _ in range(300):
        nk = r.sum(0) + 1e-9; pi, mu = nk / n, (r.T @ X) / nk[:, None]
        S = [(r[:, j, None] * (X - mu[j])).T @ (X - mu[j]) / nk[j] + 1e-6 * np.eye(d) for j in range(k)]
        lp = np.stack([np.log(pi[j]) - 0.5 * np.linalg.slogdet(S[j])[1] - np.log(2 * np.pi)
                       - 0.5 * np.einsum("ni,ij,nj->n", X - mu[j], np.linalg.inv(S[j]), X - mu[j]) for j in range(k)], 1)
        ll = np.logaddexp.reduce(lp, 1); r = np.exp(lp - ll[:, None])
        if abs(ll.sum() - prev) < 1e-8 * abs(ll.sum()): break
        prev = ll.sum()
    bic = -2 * ll.sum() + (k * d * (d + 1) / 2 + k * d + k - 1) * np.log(n)   # free parameters of a full-covariance GMM
    return bic, bic - 2 * np.log(r.max(1)).sum()                              # ICL = BIC + 2 * entropy of the hard assignment
```
</div>

<div markdown="0">
```python {marimo}
sep = mo.ui.slider(1.0, 8.0, step=0.5, value=4.0, label="cluster separation", debounce=True)
k = mo.ui.slider(2, 8, step=1, value=3, label="K for k-means", debounce=True)
noise = mo.ui.switch(label="add noise points")
mo.hstack([sep, k, noise], justify="start")
```
</div>

<div markdown="0">
```python {marimo}
X, y = blobs(sep.value, noise.value)
lab = kmeans(X, k.value)
lab_noise = np.where(y == -1, -1, lab)   # noise known (e.g. from HDBSCAN): label -1
m_plain = indices(X, lab, noise_label=-1)
m_noise = indices(X, lab_noise, noise_label=-1)
crit = np.array([bic_icl(X, kk) for kk in range(1, 7)])
```
</div>

<div markdown="0">
```python {marimo}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
axs[0].scatter(*X.T, c=lab_noise, cmap="viridis", s=12); axs[0].set_aspect("equal")
axs[0].set_title(f"k-means, K = {k.value}" + (" (noise in grey-purple: label −1)" if noise.value else ""))
for j, name in enumerate(("BIC", "ICL")):
    axs[1].plot(range(1, 7), crit[:, j], marker="o", label=name); axs[1].plot(1 + crit[:, j].argmin(), crit[:, j].min(), "k*", ms=14)
axs[1].set_xlabel("number of GMM components K"); axs[1].set_ylabel("criterion (lower is better)"); axs[1].legend()
axs[1].set_title(f"BIC picks K = {1 + crit[:, 0].argmin()}, ICL picks K = {1 + crit[:, 1].argmin()}")
fig
```
</div>

<div markdown="0">
```python {marimo}
_arrow = {"Silhouette": "↑", "Calinski-Harabasz": "↑", "Dunn": "↑", "Davies-Bouldin": "↓", "WB-index": "↓"}
mo.md("| index | better | noise swallowed by clusters | noise labelled −1 (dropped, scaled) |\n|---|---|---|---|\n" +
      "\n".join(f"| {n} | {_arrow[n]} | {m_plain[n]:.3f} | {m_noise[n]:.3f} |" for n in m_plain) +
      f"\n\nICL − BIC at K = 3: {crit[2, 1] - crit[2, 0]:.1f} (twice the assignment entropy; it grows with overlap).")
```
</div>

## What to look for

* **Separation:** as the blobs overlap, silhouette and Calinski–Harabasz fall and Davies–Bouldin and WB rise. **Dunn** is the
  most erratic (it rests on one closest pair and one widest cluster), which is why `tgmm`'s documentation ranks it behind the
  others; at low separation its best $K$ jumps around while silhouette and CH keep pointing at $K=3$.
* **K:** at high separation almost every index peaks at the true $K=3$; when the blobs overlap, WB-index and Davies–Bouldin
  (and Dunn) are the ones that wander.
* **Noise:** the noise points are known here, so the second column drops them (label $-1$) and scales by
  $(n-n_{\text{noise}})/n$. Calinski–Harabasz, WB and Dunn improve a lot; silhouette and Davies–Bouldin barely move.
  The GMM panel cannot drop noise: it tends to spend an extra component on the background, so BIC and ICL often pick
  $K=4$ or $5$ when noise is switched on.
* **BIC vs. ICL:** both are $-2\log L+p\log n$; ICL adds $-2\sum_i\log\max_k\tau_{ik}$, which is near 0 for well-separated
  components and large when they overlap. Around separation $\approx3$ ICL already prefers fewer components than BIC,
  because it penalises solutions whose clusters are not clearly distinguishable (see the
  [Model Selection tutorial](../notebooks/13_model_selection.ipynb)).
