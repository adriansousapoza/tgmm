# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "tgmm",
#     "torch; sys_platform != 'emscripten'",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
#     "scipy",
# ]
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
    MAX_ITER = 100 if FAST else 1000
    N_INIT = 1 if FAST else 5
    return (FAST,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stochastic EM, Deterministic Annealing, SAEM and Online EM

    Plain EM only climbs: from a bad starting point it converges to the nearest local optimum, and the hard assignments of CEM make this worse. This notebook collects the variants of the EM loop that `GaussianMixture` offers beyond EM and CEM ([06 -- CEM](06_cem.ipynb)): `algorithm='sem'` (Stochastic EM, Celeux & Diebolt 1985), `temperature` (Deterministic Annealing EM, Ueda & Nakano 1998), `algorithm='saem'` (Stochastic Approximation EM, Delyon, Lavielle & Moulines 1999), mini-batch and online EM (Cappé & Moulines 2009), and the convergence rules (`convergence`, `tol`).

    **What you'll learn**

    - how a saddle initialization traps EM and CEM at a poor local optimum
    - how SEM (sampling a hard assignment from the responsibilities each sweep) and DAEM (tempering the E-step) escape it, and how to try annealing yourself
    - why SEM's log-likelihood trace never settles to a point but to a "stationary band"
    - how SAEM adds a decaying step size so it converges like ordinary EM
    - how `batch_size` bounds memory exactly, and how `step_size` turns it into online EM
    - how the `'relative'` and `'aitken'` stopping rules differ

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb) (EM), [06 -- CEM](06_cem.ipynb).

    **See also:** the [`GaussianMixture` API reference](../api/gaussian-mixture.md) for every argument used here.
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
    import warnings

    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from tgmm import GaussianMixture, plot_gmm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f'Using device: {device}')
    return GaussianMixture, SEED, np, plot_gmm, plt, torch, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The variants at a glance

    `algorithm` selects the EM variant that drives `fit()`. Each variant feeds some matrix of (soft or hard) assignments into the same M-step, which computes the statistics $n_k,\bar y_k,W_k$ from it exactly as in [01 -- Introduction](01_introduction.ipynb). `temperature` (annealing) and `step_size` (online EM) are separate knobs layered on top.

    | `algorithm` | What goes into the M-step | When to use it |
    | --- | --- | --- |
    | `'em'` (default) | soft responsibilities $r_{ik}$ | the right default |
    | `'cem'` | hard one-hot $\mathbb 1[k=\arg\max_j r_{ij}]$ | faster, crisper partition; more local-optimum prone |
    | `'sem'` | a hard label *sampled* from $r_{i\cdot}$ | escape poor optima; diagnose an overspecified $K$ |
    | `'saem'` | statistics averaged over sampled sweeps with step $\gamma_t$ | SEM's exploration with a convergent stopping point |

    ### Stochastic EM (SEM)

    SEM replaces the soft assignment by **one draw per sample**, $z_i\sim\operatorname{Categorical}(r_{i1},\dots,r_{iK})$, and runs the M-step on the resulting one-hot matrix. The iterates form a Markov chain that keeps exploring instead of settling, so SEM can leave poor local optima. Details of the implementation:

    - SEM has no pointwise convergence: `fit()` always runs exactly `max_iter` sweeps and sets `converged_=True`.
    - A component that receives fewer than one sampled point in a sweep has its mean re-seeded at one of the worst-fit data points (and, for `full`/`diag`/`spherical`, its covariance reset to the data covariance). Each re-seed is counted in `n_empty_components_`, so a non-zero count hints that `n_components` is too large.
    - The returned parameters are computed from the sweeps at or after `sem_burn_in` (default `max(1, max_iter // 5)`). `sem_estimate='best'` (default) keeps the sweep with the highest log-likelihood; `'mean'` averages the parameters over all of them, which presumes no label switching between sweeps, so prefer `'best'` when $K$ may be overspecified.

    ### Deterministic annealing (DAEM)

    DAEM tempers the posterior with an inverse temperature $\beta\in(0,1]$:

    $$
    r_{ik}(\beta)=\frac{\big(\pi_k\,\mathcal N(x_i\mid\mu_k,\Sigma_k)\big)^{\beta}}{\sum_j\big(\pi_j\,\mathcal N(x_i\mid\mu_j,\Sigma_j)\big)^{\beta}} .
    $$

    At $\beta\to0$ every sample is assigned almost uniformly to all components (a smooth, nearly unimodal objective); at $\beta=1$ this is the ordinary E-step. `temperature=(beta_0, n_anneal)` starts at $\beta_0\in(0,1)$ and raises $\beta$ geometrically to 1 over `n_anneal` iterations:

    $$
    \beta_t=\begin{cases}\beta_0^{\,1-t/n_{\text{anneal}}}, & t<n_{\text{anneal}},\\[2pt] 1, & t\ge n_{\text{anneal}}.\end{cases}
    $$

    It works with `algorithm` in `{'em', 'cem', 'sem'}`. While $\beta<1$ the convergence check is suspended, and the entries of `log_likelihood_history_` recorded then are *tempered* normalizers, not true log-likelihoods: compare only the entries from $\beta=1$ on. The schedule actually used is stored in `annealing_schedule_`. Annealing is a heuristic, not a guarantee: a perfectly symmetric start (e.g. identical means) stays symmetric under any $\beta$, so some asymmetry in the initialization is needed for annealing to amplify.

    ### Stochastic Approximation EM (SAEM)

    SAEM averages the *sufficient statistics* of the sampled sweeps with a decreasing step size $\gamma_t$ (Robbins-Monro):

    $$
    \tilde S_t=(1-\gamma_t)\,\tilde S_{t-1}+\gamma_t\,S\big(z^{(t)}\big),\qquad \tilde S_0=S\big(z^{(0)}\big),
    $$

    where $z^{(t)}$ is the sampled one-hot assignment of sweep $t$, $S(z)=\big(n_k,\ \sum_i z_{ik}x_i,\ W_k\big)_{k}$ are its sufficient statistics, and the M-step maps $\tilde S_t$ to the parameters. The default `step_size='auto'` is $\gamma_t=(t+2)^{-0.7}$ (so $\sum_t\gamma_t=\infty$ while $\sum_t\gamma_t^2<\infty$); a positive float gives a constant step and a callable $t\mapsto\gamma_t$ a custom schedule. Because the averaged statistics converge, the usual `tol` test applies: SAEM is "the SEM that converges".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A known saddle initialization

    Three well-separated true clusters, but all three components are initialized jittered around a
    *single* one of them (cluster 0) -- a classic EM failure mode: two components end up fighting
    over the one cluster they started near, while the third true cluster is left with no component
    to claim it. Standard (soft) EM and CEM both get stuck here; SEM's stochastic sampling and DAEM's
    tempered E-step both escape it, as shown below.
    """)
    return


@app.cell
def _(plt, torch):
    torch.manual_seed(0)
    centers = [(0.0, 0.0), (8.0, 0.0), (4.0, 7.0)]
    X = torch.cat([torch.randn(150, 2) + torch.tensor(c) for c in centers])

    torch.manual_seed(3)
    init_means = torch.tensor([0.0, 0.0]).unsqueeze(0).repeat(3, 1) + torch.randn(3, 2) * 0.3
    init_covariances = torch.stack([torch.eye(2)] * 3)

    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], s=8, alpha=0.4, label='data')
    plt.scatter(init_means[:, 0], init_means[:, 1], c='red', marker='x', s=120, label='initial means (all 3)')
    plt.legend()
    plt.title('Saddle initialization: all 3 components start near one cluster')
    plt.gca().set_aspect('equal')
    plt.gcf()
    return X, init_covariances, init_means


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit EM, CEM, SEM, DAEM and SAEM from the identical initialization
    """)
    return


@app.cell
def _(GaussianMixture, SEED, X, init_covariances, init_means, warnings):
    common_kwargs = dict(
        n_components=3, n_features=2, random_state=SEED, max_iter=100,
        init_means=init_means.clone(), init_covariances=init_covariances.clone(),
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # EM/CEM may report that they did not converge within max_iter
        gmm_em = GaussianMixture(algorithm='em', **common_kwargs).fit(X)
        gmm_cem = GaussianMixture(algorithm='cem', **common_kwargs).fit(X)
        gmm_sem = GaussianMixture(algorithm='sem', **common_kwargs).fit(X)
        gmm_daem = GaussianMixture(temperature=(0.9, 30), **common_kwargs).fit(X)
        gmm_saem_saddle = GaussianMixture(algorithm='saem', **common_kwargs).fit(X)

    for _name, _gmm in [('EM', gmm_em), ('CEM', gmm_cem), ('SEM', gmm_sem), ('DAEM', gmm_daem), ('SAEM', gmm_saem_saddle)]:
        print(f'{_name:5s} final log-likelihood: {_gmm.lower_bound_:.4f}')
    return common_kwargs, gmm_cem, gmm_daem, gmm_em, gmm_sem


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log-likelihood traces, plotted together

    EM and CEM plateau early, stuck at the same (worse) local optimum. SEM
    climbs to the *better* optimum and stays there; unlike EM it never stops, and at this scale its
    tail looks flat, but it actually keeps fluctuating in a narrow "stationary band" (zoomed in below),
    since every iteration re-samples a hard assignment instead of converging to a fixed point. DAEM's trace shows the tempered,
    lower-likelihood early iterations (while `beta < 1`) before it locks onto the better solution as
    `beta` reaches `1.0`.
    """)
    return


@app.cell
def _(gmm_cem, gmm_daem, gmm_em, gmm_sem, plt):
    plt.figure(figsize=(8, 5))
    plt.plot(gmm_em.log_likelihood_history_, label='EM', marker='.')
    plt.plot(gmm_cem.log_likelihood_history_, label='CEM', marker='.')
    plt.plot(gmm_sem.log_likelihood_history_, label='SEM', alpha=0.8)
    plt.plot(gmm_daem.log_likelihood_history_, label='DAEM', linewidth=2)
    plt.xlabel('iteration / sweep')
    plt.ylabel('mean log-likelihood')
    plt.title('EM vs CEM vs SEM vs DAEM from the same saddle initialization')
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zooming in: SEM's stationary band

    Unlike EM/CEM/DAEM, SEM never converges pointwise -- `fit()` always runs the full `max_iter`
    sweeps. The tail of its trace fluctuates in a narrow band
    around the best achievable log-likelihood instead of settling exactly, since each sweep samples a
    fresh hard assignment from the responsibilities rather than taking the `argmax`.
    """)
    return


@app.cell
def _(gmm_daem, gmm_sem, plt):
    _tail = gmm_sem.log_likelihood_history_[-50:]
    plt.figure(figsize=(8, 3))
    plt.plot(_tail, marker='.')
    plt.axhline(gmm_daem.lower_bound_, color='C3', linestyle='--', label='DAEM/best optimum')
    plt.xlabel('sweep (last 50)')
    plt.ylabel('mean log-likelihood')
    plt.title(f'SEM stationary band (std = {_tail.std().item():.2e})')
    plt.legend()
    plt.gcf()
    return


@app.cell
def _(gmm_daem, gmm_sem):
    print('n_empty_components_:', gmm_sem.n_empty_components_)
    print('annealing_schedule_ (first 10 values):', [round(b, 4) for b in gmm_daem.annealing_schedule_[:10]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From a stationary band to one answer: `sem_estimate`

    A noisy trace still has to be reduced to one set of parameters. `sem_estimate='best'` keeps the sweep with the highest log-likelihood, `'mean'` averages the parameters over the post-burn-in sweeps. Here the components do not swap labels, so the two agree:
    """)
    return


@app.cell
def _(GaussianMixture, X, common_kwargs, mo, warnings):
    _rows = ['| `sem_estimate` | final log-likelihood |', '|---|---|']
    for _estimate in ('best', 'mean'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _g = GaussianMixture(algorithm='sem', sem_estimate=_estimate, **common_kwargs).fit(X)
        _rows.append(f'| `{_estimate}` | {_g.lower_bound_:.4f} |')
    mo.md('\n'.join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try annealing yourself

    Pick the starting temperature $\beta_0$, the number of annealing iterations, the algorithm, and the jitter of the initial means (the seed). The cell fits the saddle data twice from that **same** start, once plainly and once with `temperature=(beta_0, n_anneal)`, and compares the two fits. The right panel plots the log-likelihood only from the iterations where $\beta=1$ (before that, the recorded values are tempered normalizers).

    The controls only respond in a running notebook (`marimo edit`, or the molab link at the top of the page); on the static docs page they show the default setting.
    """)
    return


@app.cell
def _(mo):
    beta0_ui = mo.ui.slider(0.1, 0.95, step=0.05, value=0.9, label='starting temperature beta_0', debounce=True, show_value=True)
    n_anneal_ui = mo.ui.slider(10, 60, step=5, value=30, label='annealing iterations n_anneal', debounce=True, show_value=True)
    jitter_seed_ui = mo.ui.slider(0, 9, step=1, value=3, label='jitter seed of the initial means', debounce=True, show_value=True)
    algorithm_ui = mo.ui.dropdown(['em', 'cem', 'sem'], value='em', label='algorithm')
    mo.vstack([mo.hstack([beta0_ui, n_anneal_ui], justify='start'), mo.hstack([jitter_seed_ui, algorithm_ui], justify='start')])
    return algorithm_ui, beta0_ui, jitter_seed_ui, n_anneal_ui


@app.cell
def _(GaussianMixture, SEED, X, algorithm_ui, beta0_ui, jitter_seed_ui, n_anneal_ui, torch, warnings):
    _gen = torch.Generator().manual_seed(int(jitter_seed_ui.value))
    _means0 = 0.3 * torch.randn(3, 2, generator=_gen)  # all three means start jittered around cluster 0 at the origin
    _kw = dict(
        n_components=3, n_features=2, random_state=SEED, max_iter=150, algorithm=algorithm_ui.value,
        init_means=_means0, init_covariances=torch.stack([torch.eye(2)] * 3),
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gmm_plain = GaussianMixture(**_kw).fit(X)
        gmm_annealed = GaussianMixture(temperature=(float(beta0_ui.value), int(n_anneal_ui.value)), **_kw).fit(X)
    return gmm_annealed, gmm_plain


@app.cell
def _(X, algorithm_ui, beta0_ui, gmm_annealed, gmm_plain, n_anneal_ui, np, plot_gmm, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.4))
    _Xn = X.numpy()
    for _ax, _g, _title in ((_axes[0], gmm_plain, f'beta = 1 (plain {algorithm_ui.value.upper()})'), (_axes[1], gmm_annealed, f'annealed, beta_0 = {beta0_ui.value:g}, n_anneal = {n_anneal_ui.value}')):
        plot_gmm(_Xn, gmm=_g, ax=_ax, color_by_cluster=True, ellipse_std_devs=[2], ellipse_fill=False, ellipse_line_width=2, title=f'{_title}\nmean log-likelihood {_g.lower_bound_:.3f}', legend=False)
        _ax.set_aspect('equal')
    _hist_plain = gmm_plain.log_likelihood_history_.numpy()
    _hist_ann = gmm_annealed.log_likelihood_history_.numpy()
    _beta = np.asarray(gmm_annealed.annealing_schedule_)
    _on = _beta >= 1.0 if len(_beta) == len(_hist_ann) else np.ones(len(_hist_ann), dtype=bool)
    _axes[2].plot(_hist_plain, label='plain')
    _axes[2].plot(np.nonzero(_on)[0], _hist_ann[_on], label='annealed (beta = 1 part)')
    _axes[2].axvline(n_anneal_ui.value, color='0.6', ls=':')
    _axes[2].set_xlabel('iteration')
    _axes[2].set_ylabel('mean log-likelihood')
    _axes[2].set_title('log-likelihood (dotted: end of annealing)')
    _axes[2].legend()
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(gmm_annealed, gmm_plain, mo):
    mo.md(f"""
    **Final mean log-likelihood:** annealed {gmm_annealed.lower_bound_:.3f} vs. plain {gmm_plain.lower_bound_:.3f} (difference {gmm_annealed.lower_bound_ - gmm_plain.lower_bound_:+.3f}). The three-blob optimum has mean log-likelihood about -3.98 for this data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.** With `algorithm='em'` and `n_anneal=30`, $\beta_0=0.9$ reached the three-blob optimum from every jitter seed in 2-7 that we tried; $\beta_0=0.7$ did so for some seeds and not others; and a *hot* start, $\beta_0=0.3$, collapsed the fit for all of them (the means are pulled together until all three components describe the same blob, log-likelihood about -5.3, far below either alternative). So annealing is a heuristic: too little smoothing changes nothing, too much destroys the structure it was meant to find. Two more things to try with the default start. With `'cem'`, the C-step takes the `argmax` of the tempered posterior, which is the same as the plain `argmax` for every $\beta>0$, so annealing cannot change the hard partitions along the way; the only difference is that the convergence test is suspended while $\beta<1$, so the annealed run keeps iterating a little longer (here it ends at -4.4455 instead of -4.4475, still the poor optimum). With `'sem'`, the sampling already escapes the saddle without annealing (-3.981 either way).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SAEM: the "SEM that converges"

    SAEM blends each sweep's sampled-assignment statistics into a running average with a decaying
    step size (`step_size`, defaulting to `'auto'` for this algorithm), so -- unlike plain SEM --
    `tol`-based convergence applies and `fit()` can stop early. Started from the saddle above, SAEM's own stochastic trajectory settles into a poor local optimum of its own (final log-likelihood -4.71, worse than EM's -4.45; see the fits above): a reminder that none of these methods *guarantee* escaping every bad initialization, they each just change the odds. What SAEM
    reliably does is match plain EM when both are given the same, reasonable starting point -- here two moderately overlapping clusters, started from the default k-means initialization:
    """)
    return


@app.cell
def _(GaussianMixture, plt, torch):
    torch.manual_seed(1)
    X2 = torch.cat([torch.randn(2000, 2) + c for c in
                    [torch.tensor([0.0, 0.0]), torch.tensor([3.0, 0.0])]])

    gmm_em2 = GaussianMixture(n_components=2, n_features=2, algorithm='em', random_state=0, max_iter=200)
    gmm_em2.fit(X2)

    gmm_saem2 = GaussianMixture(n_components=2, n_features=2, algorithm='saem', random_state=0, max_iter=200)
    gmm_saem2.fit(X2)

    print('EM means:\n', gmm_em2.means_)
    print('SAEM means:\n', gmm_saem2.means_)
    print(f'SAEM converged: {gmm_saem2.converged_}, iterations: {gmm_saem2.n_iter_}')

    plt.figure(figsize=(8, 4))
    plt.plot(gmm_saem2.log_likelihood_history_, marker='.')
    plt.xlabel('iteration')
    plt.ylabel('mean log-likelihood')
    plt.title('SAEM log-likelihood trace (converges, unlike plain SEM)')
    plt.gcf()
    return (X2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The step-size schedule

    `step_size` sets the weight $\gamma_t$ given to the newest sweep: `'auto'` is $\gamma_t=(t+2)^{-0.7}$, a float is a constant step, and a callable gives any schedule. The table runs SAEM with four schedules on the same data at a tight tolerance (`tol=1e-6`) and compares the final log-likelihood with the long-run EM optimum:
    """)
    return


@app.cell
def _(GaussianMixture, X2, mo, warnings):
    _schedules = {"'auto' ($(t+2)^{-0.7}$)": 'auto', 'constant 0.5': 0.5, 'constant 0.1': 0.1, r'callable $1/(t+1)$': lambda t: 1.0 / (t + 1)}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _optimum = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=5000, tol=1e-12).fit(X2).lower_bound_
        _rows = ['| `step_size` | sweeps | converged | gap to the EM optimum |', '|---|---|---|---|']
        for _label, _step in _schedules.items():
            _g = GaussianMixture(n_components=2, n_features=2, algorithm='saem', step_size=_step, random_state=0, max_iter=300, tol=1e-6).fit(X2)
            _rows.append(f'| {_label} | {_g.n_iter_} | {_g.converged_} | {_optimum - _g.lower_bound_:.1e} |')
    mo.md('\n'.join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every schedule converges to within a few $10^{-4}$ of the EM optimum here; what differs is the number of sweeps and the exact landing point. The small constant step 0.1 needs the most sweeps because it averages over many of them, while the quickly decaying $1/(t+1)$ stops soonest but lands farthest from the optimum. The Robbins-Monro conditions ($\sum\gamma_t=\infty$, $\sum\gamma_t^2<\infty$) are what guarantee convergence in general; `'auto'` satisfies them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mini-batch and online EM

    With `batch_size` alone, the E-step and the statistics accumulation run in row chunks. This is **exact** (a per-row softmax and additive statistics) and only bounds the peak memory at $O(\text{batch\_size}\cdot K\cdot d)$. Adding `step_size` switches to **online EM**: in each epoch the data are shuffled, and for every mini-batch $B$ the statistics $S_B$ are computed from the current parameters, rescaled to the full data size, blended into a running estimate with step $\gamma_t$ ($t$ counts mini-batches globally), and the M-step is run from it:

    $$
    \tilde S\leftarrow(1-\gamma_t)\,\tilde S+\gamma_t\,\frac{n}{|B|}\,S_B ,\qquad
    \theta\leftarrow\text{M-step}(\tilde S).
    $$

    `max_iter` counts epochs; there is no `tol` test (`converged_` is set to `True`), and `log_likelihood_history_` has one entry per epoch. `step_size` requires `batch_size`.

    A caveat on blended scatter: $W_k$ is centered on each accumulation's *own* mean. SAEM and online EM blend $W_k$ (and $n_k$, $\sum_i r_{ik}x_i$) linearly, which is exact for the first two but only approximate for $W_k$ while the means are still shifting, because the true scatter about the blended mean also contains a between-accumulation term. The approximation vanishes as the iterates stabilize.

    Below: three overlapping blobs with different scales, fitted by standard EM, by chunked exact EM, and by online EM.
    """)
    return


@app.cell
def _(FAST, torch):
    torch.manual_seed(2)
    _n_each = 2000 if FAST else 5000
    X_big = torch.cat([torch.randn(_n_each, 2) * s + m for s, m in [(1.0, torch.tensor([0.0, 0.0])), (0.5, torch.tensor([4.0, 0.0])), (0.8, torch.tensor([2.0, 4.0]))]])
    X_big = X_big[torch.randperm(len(X_big))]
    print(f'{len(X_big)} points')
    return (X_big,)


@app.cell
def _(GaussianMixture, X_big, mo, warnings):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gmm_full = GaussianMixture(n_components=3, n_features=2, random_state=0, max_iter=200).fit(X_big)
        gmm_chunked = GaussianMixture(n_components=3, n_features=2, random_state=0, max_iter=200, batch_size=1000).fit(X_big)
        gmm_online = GaussianMixture(n_components=3, n_features=2, random_state=0, max_iter=10, batch_size=256, step_size='auto').fit(X_big)

    _max_diff = max(
        (gmm_full.means_ - gmm_chunked.means_).abs().max().item(),
        (gmm_full.weights_ - gmm_chunked.weights_).abs().max().item(),
        (gmm_full.covariances_ - gmm_chunked.covariances_).abs().max().item(),
    )
    mo.md(f"""
    | fit | iterations / epochs | final mean log-likelihood |
    |---|---|---|
    | standard EM | {gmm_full.n_iter_} | {gmm_full.lower_bound_:.5f} |
    | `batch_size=1000` (exact, chunked) | {gmm_chunked.n_iter_} | {gmm_chunked.lower_bound_:.5f} |
    | online EM (`batch_size=256`, `step_size='auto'`) | {gmm_online.n_iter_} epochs | {gmm_online.lower_bound_:.5f} |

    The chunked fit matches the standard one up to floating-point summation order (largest absolute difference in any weight, mean or covariance entry: {_max_diff:.1e}). Online EM, after {gmm_online.n_iter_} passes over the data, reaches a log-likelihood within {abs(gmm_online.lower_bound_ - gmm_full.lower_bound_):.1e} of standard EM's.
    """)
    return (gmm_online,)


@app.cell
def _(gmm_online, plt):
    plt.figure(figsize=(7, 3.5))
    plt.plot(range(1, len(gmm_online.log_likelihood_history_) + 1), gmm_online.log_likelihood_history_, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('mean log-likelihood (full data)')
    plt.title('Online EM: log-likelihood after each epoch')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convergence control

    Let $L_t=\frac1n\ell(\theta^{(t)})$ be the average log-likelihood (`lower_bound_`). `convergence='relative'` (the default) stops when

    $$
    \frac{|L_t-L_{t-1}|}{|L_{t-1}|}<\texttt{tol}.
    $$

    `convergence='aitken'` uses Aitken's acceleration. EM's log-likelihood sequence is asymptotically geometric, so with $a_t=\dfrac{L_t-L_{t-1}}{L_{t-1}-L_{t-2}}$ (when $a_t<1$) the limit is estimated as

    $$
    L_\infty=L_{t-1}+\frac{L_t-L_{t-1}}{1-a_t},
    $$

    and the fit stops when $L_\infty-L_t<\texttt{tol}$, an *absolute* gap to the extrapolated limit. It falls back to the relative test until three values exist, or whenever $a_t\ge1$. `fit()` also accepts per-call overrides of `max_iter`, `tol`, `random_state` and `warm_start` (continue from the previous fit's parameters) without changing the model's defaults.

    The table compares the two rules on two heavily overlapping clusters, where EM crawls: for each `tol` it lists the iterations used and the remaining gap to the true optimum (a long run with `tol=1e-12`).
    """)
    return


@app.cell
def _(GaussianMixture, mo, torch, warnings):
    torch.manual_seed(1)
    _X_slow = torch.cat([torch.randn(2000, 2) + c for c in [torch.tensor([0.0, 0.0]), torch.tensor([2.0, 0.0])]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _optimum = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=5000, tol=1e-12).fit(_X_slow).lower_bound_
        _rows = ['| `tol` | relative: iterations | relative: gap to optimum | aitken: iterations | aitken: gap to optimum |', '|---|---|---|---|---|']
        for _tol in (1e-3, 1e-4, 1e-5, 1e-6):
            _r = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=5000, tol=_tol, convergence='relative').fit(_X_slow)
            _a = GaussianMixture(n_components=2, n_features=2, random_state=0, max_iter=5000, tol=_tol, convergence='aitken').fit(_X_slow)
            _rows.append(f'| {_tol:g} | {_r.n_iter_} | {_optimum - _r.lower_bound_:.1e} | {_a.n_iter_} | {_optimum - _a.lower_bound_:.1e} |')
    mo.md('\n'.join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At the same `tol`, the Aitken rule runs a few iterations longer here and stops closer to the optimum: its tolerance is a statement about the *remaining* distance to the limit, while the relative rule only looks at the last step, which is small whenever progress is slow, even if the limit is still far away.

    ## Summary

    - **EM** (default) and **CEM** (hard assignments) both converge to the nearest local optimum, so from the saddle initialization they stay stuck.
    - **SEM** escapes by sampling hard assignments instead of taking the `argmax`, at the cost of never settling to a single point -- use `sem_estimate='best'` (or `'mean'`) over the post-burn-in sweeps to get a final answer out of a noisy trace, and read `n_empty_components_` as a hint that $K$ is too large.
    - **DAEM** escapes by tempering the E-step early on, smoothing the likelihood surface while the fit is still deciding where to settle; the starting temperature matters (too hot collapses the fit).
    - **SAEM** gets SEM's exploration with a decaying step size, so it converges like ordinary EM and matches EM closely from a reasonable initialization.
    - `batch_size` alone is an exact memory bound; with `step_size` it becomes online EM, which approaches the full-data optimum in a few epochs.
    - `convergence='aitken'` makes `tol` a bound on the remaining gap to the limit.
    - None of the escape methods *guarantees* escaping every bad initialization; they change the odds.

    **Next:** [16 -- Student-t Mixtures](16_student_t.ipynb), the first non-Gaussian family, or back to [14 -- Model Selection](14_model_selection.ipynb).
    """)
    return


if __name__ == "__main__":
    app.run()
