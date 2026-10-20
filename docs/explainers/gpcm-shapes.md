# GPCM Covariance Shapes

This demo takes a few seconds to load and runs entirely in your browser (NumPy and matplotlib via Pyodide; no tgmm or PyTorch is involved).
It mirrors the covariance parameterisation of
[`GaussianMixture(covariance_type=...)`](../user-guide/gpcm-covariance-models.md) (API: [`tgmm.covariance.gpcm`](../api/covariance-gpcm.md)).

Each component's covariance is split into three factors,

$$\Sigma_k=\lambda_k\,D_k\,A_k\,D_k^{\top},\qquad \det A_k=1,\quad D_k^{\top}D_k=I,$$

a **volume** $\lambda_k$ (size), a **shape** $A_k=\operatorname{diag}(a_{k1},a_{k2})$ (elongation, $a_{k1}/a_{k2}$ is the axis ratio) and an
**orientation** $D_k$ (rotation by $\theta_k$). For two components, choose for each factor whether the components share it
(**E**, *equal*: component 2 copies component 1), let it **V**ary (each has its own sliders) or, for shape and orientation, fix it to the
**I**dentity (**I**: a circle, or no rotation). The three letters give the model's mclust code.

<div markdown="0">
```python {marimo}
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt

vol_mode = mo.ui.dropdown(["E", "V"], value="V", label="volume")
shape_mode = mo.ui.dropdown(["E", "V", "I"], value="E", label="shape")
orient_mode = mo.ui.dropdown(["E", "V", "I"], value="V", label="orientation")
lam1 = mo.ui.slider(0.2, 3.0, step=0.1, value=1.0, label="λ₁")
lam2 = mo.ui.slider(0.2, 3.0, step=0.1, value=2.0, label="λ₂")
ratio1 = mo.ui.slider(1.0, 12.0, step=0.5, value=4.0, label="a₁/a₂ (comp. 1)")
ratio2 = mo.ui.slider(1.0, 12.0, step=0.5, value=4.0, label="a₁/a₂ (comp. 2)")
ang1 = mo.ui.slider(0, 180, step=5, value=30, label="θ₁ (°)")
ang2 = mo.ui.slider(0, 180, step=5, value=120, label="θ₂ (°)")
mo.vstack([
    mo.hstack([vol_mode, shape_mode, orient_mode], justify="start"),
    mo.hstack([lam1, ratio1, ang1], justify="start"),
    mo.hstack([lam2, ratio2, ang2], justify="start"),
])
```
</div>

Component 2's slider is ignored whenever its factor is **E** (it copies component 1), and both shape sliders are ignored for shape **I**
(both orientation sliders for orientation **I**).

<div markdown="0">
```python {marimo}
VALID = {"EII", "VII", "EEI", "VEI", "EVI", "VVI", "EEE", "VEE", "EVE", "EEV", "VVE", "VEV", "EVV", "VVV"}
K, d = 2, 2
_v, _s, _o = vol_mode.value, shape_mode.value, orient_mode.value
code = _v + _s + _o
l1, l2 = lam1.value, (lam1.value if _v == "E" else lam2.value)
r1 = 1.0 if _s == "I" else ratio1.value
r2 = 1.0 if _s == "I" else (r1 if _s == "E" else ratio2.value)
t1 = 0.0 if _o == "I" else ang1.value
t2 = 0.0 if _o == "I" else (t1 if _o == "E" else ang2.value)
n_params = ({"E": 1, "V": K}[_v] + {"I": 0, "E": d - 1, "V": K * (d - 1)}[_s]
            + {"I": 0, "E": d * (d - 1) // 2, "V": K * d * (d - 1) // 2}[_o])

def make_cov(lam, ratio, deg):
    a = np.array([np.sqrt(ratio), 1 / np.sqrt(ratio)])      # det A = 1
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    D = np.array([[c, -s], [s, c]])
    return lam * D @ np.diag(a) @ D.T

covs = [make_cov(l1, r1, t1), make_cov(l2, r2, t2)]
verdict = ("one of the 14 GPCM models" if code in VALID else
           "**not a valid GPCM model**: with a spherical (identity) shape the orientation is meaningless, so use orientation I")
mo.md(f"### mclust code `{code}` — {verdict}\n\nFree covariance parameters for $d=2$, $K=2$: **{n_params}** "
      f"(volume + shape + orientation contributions; $\\det\\Sigma_1={np.linalg.det(covs[0]):.2f}$, $\\det\\Sigma_2={np.linalg.det(covs[1]):.2f}$).")
```
</div>

<div markdown="0">
```python {marimo}
_t = np.linspace(0, 2 * np.pi, 200)
_circle = np.stack([np.cos(_t), np.sin(_t)])
fig, ax = plt.subplots(figsize=(7, 4))
for _k, (_S, _c, _x0) in enumerate(zip(covs, ["#2a78d6", "#eb6834"], [-3.0, 3.0])):
    _L = np.linalg.cholesky(_S)
    for _n in (1, 2):
        ax.plot(_x0 + _n * (_L @ _circle)[0], _n * (_L @ _circle)[1], color=_c, lw=2 if _n == 1 else 1)
    ax.plot(_x0, 0, "o", color=_c)
ax.set_aspect("equal"); ax.set_xlim(-7, 7); ax.set_ylim(-3.5, 3.5)
ax.set_title(f"1σ and 2σ contours of Σ₁ (blue) and Σ₂ (orange), model {code}")
fig
```
</div>

Things to try: with `EII` / `VII` both shapes are circles (identity), `EEE` shares everything (the same ellipse twice), and `VVV` frees all three factors.
The parameter count is what `bic`/`aic` charge for the covariances, which is why constrained models win on small data
(see the [Covariance Structures tutorial](../notebooks/15_covariance_models.ipynb)).
