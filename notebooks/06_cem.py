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
#     "seaborn",
#     "tqdm",
#     "ipywidgets",
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
    # Classification EM (CEM)

    This notebook compares the **Classification EM (CEM)** algorithm with standard **Expectation-Maximization (EM)** for Gaussian Mixture Models. CEM inserts a hard classification step between the E-step and the M-step, so each point updates only the component it is most likely to belong to. In `tgmm` it is selected with `GaussianMixture(algorithm='cem')`.

    **What you'll learn**

    - How the C-step changes the EM iteration, and which objective CEM maximizes (the *classification* likelihood, not the mixture likelihood)
    - How to read the two likelihoods side by side while EM and CEM iterate
    - How EM and CEM behave on heavily overlapping clusters (accuracy, speed, and the bias of CEM's covariance estimates)
    - How they behave on well-separated clusters, where CEM's hard assignments lose little
    - When CEM is a sensible choice

    **Prerequisites:** [01 -- Introduction](01_introduction.ipynb) (the model, the E-step and the M-step).

    **See also:** [15 -- Stochastic EM, Annealing and SAEM](15_stochastic_em.ipynb) for the other EM variants (`'sem'`, `'saem'`, `temperature`), and the [`GaussianMixture` API reference](../api/gaussian-mixture.md).
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
    import time
    import warnings

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment

    from tgmm import GaussianMixture, plot_gmm, dynamic_figsize, generate_gmm_data
    from tutorial_data import make_default_data

    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (
        GaussianMixture,
        SEED,
        device,
        dynamic_figsize,
        generate_gmm_data,
        linear_sum_assignment,
        make_default_data,
        np,
        plot_gmm,
        plt,
        time,
        torch,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model behind CEM

    Recall from [01 -- Introduction](01_introduction.ipynb) that EM maximizes the **mixture log-likelihood**

    $$
    \ell(\theta)=\sum_{i=1}^{n}\log\sum_{k=1}^{K}\pi_k\,\mathcal N(x_i\mid\mu_k,\Sigma_k),
    $$

    and that its E-step computes the responsibilities $r_{ik}=\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)\big/\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)$.

    The CEM algorithm (Celeux and Govaert, 1992) instead maximizes the **classification log-likelihood** over the parameters *and* a hard labelling $z_i\in\{1,\dots,K\}$:

    $$
    \ell_C(\theta,z)=\sum_{i=1}^{n}\log\big(\pi_{z_i}\,\mathcal N(x_i\mid\mu_{z_i},\Sigma_{z_i})\big).
    $$

    It does so by coordinate ascent, which gives the three steps of one iteration:

    1. **E-step**: compute the responsibilities $r_{ik}$ for the current $\theta$.
    2. **C-step**: assign each observation to its most probable cluster (the MAP operator, formula below).
    3. **M-step**: maximize $\ell_C(\theta,\hat z)$ over $\theta$. This is the usual M-step with $r_{ik}$ replaced by the one-hot $\hat z_{ik}$: $n_k=\sum_i\hat z_{ik}$, $\pi_k=n_k/n$, $\mu_k$ the cluster mean, and the covariance from the cluster scatter, with the closed forms for each `covariance_type` given in [02 -- Covariance Types](02_covariance_types.ipynb).

    The C-step sets
    $$\hat z_{ik}=\begin{cases}1 & k=\arg\max_j r_{ij},\\ 0 & \text{otherwise.}\end{cases}$$
    Because $r_{ik}\propto\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)$ with a normalizer that does not depend on $k$, this is exactly $\arg\max_{z}\ell_C(\theta,z)$ for fixed $\theta$.

    Each of the two coordinate steps can only increase $\ell_C$, so **$\ell_C$ never decreases** (the notebook checks this below). The mixture likelihood $\ell$ has no such guarantee under CEM: it is not the objective.

    ### How the two likelihoods are related

    Writing $\log\sum_k a_k=\sum_k r_k\log a_k-\sum_k r_k\log r_k$ with $a_k=\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)$ and $r_k=a_k/\sum_ja_j$ gives

    $$
    \ell(\theta)=\sum_{i,k}r_{ik}\log\big(\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)\big)-\sum_{i,k}r_{ik}\log r_{ik}.
    $$

    With the one-hot $\hat z$ in place of $r$ the first term becomes $\ell_C$ and the entropy term $-\sum r_{ik}\log r_{ik}$ is zero, so CEM drops the term EM keeps. For well-separated clusters $r$ is almost one-hot and the term is negligible; for overlapping clusters it is large, and CEM's estimates are **biased**: each cluster is estimated from the points nearest to it only, which shrinks the fitted covariances (measured below).

    ### Key properties

    - **K-means-like.** With equal weights and a shared spherical covariance, the C-step is "assign to the nearest mean" and the M-step is "recompute the cluster means", i.e. Lloyd's k-means. `tgmm` also updates the weights, so `covariance_type='tied_spherical'` with `algorithm='cem'` is k-means up to the weights.
    - **Fast.** CEM generally needs fewer iterations (measured below).
    - **Biased and more local-minimum prone** than EM when clusters overlap.
    - **`lower_bound_` is still the mixture log-likelihood.** `tgmm` reports the average mixture log-likelihood after a CEM fit too, so EM and CEM fits are directly comparable; the classification likelihood is computed explicitly below.

    ### When to use CEM

    - **Well-separated clusters**, with similar proportions
    - **Fast clustering**, when speed or a crisp partition matters more than the exact maximum-likelihood estimates
    - **Initialisation**: a CEM run can seed an EM run when k-means does not work well (see [03 -- Initialization](03_initialization.ipynb))
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CEM in `tgmm`

    CEM is one keyword away: `GaussianMixture(..., algorithm='cem')`. Here it is on the default four blobs of [01 -- Introduction](01_introduction.ipynb), next to standard EM from the same seed.

    **Note:** the older `cem=True` keyword is deprecated. It still works (it is treated as `algorithm='cem'`) but emits a `DeprecationWarning`, so use `algorithm='cem'`.
    """)
    return


@app.cell
def _(GaussianMixture, SEED, device, dynamic_figsize, make_default_data, plot_gmm, plt):
    _X_default, _ = make_default_data(SEED, device=device)
    _gmm_em = GaussianMixture(n_components=4, covariance_type='full', random_state=0, max_iter=200, device=device).fit(_X_default)
    _gmm_cem = GaussianMixture(n_components=4, covariance_type='full', algorithm='cem', random_state=0, max_iter=200, device=device).fit(_X_default)
    _fig, _axes = plt.subplots(1, 2, figsize=dynamic_figsize(1, 2))
    plot_gmm(_X_default.cpu().numpy(), gmm=_gmm_em, ax=_axes[0], title=f'Standard EM ({_gmm_em.n_iter_} iters, LL={_gmm_em.lower_bound_:.3f})')
    plot_gmm(_X_default.cpu().numpy(), gmm=_gmm_cem, ax=_axes[1], title=f'CEM ({_gmm_cem.n_iter_} iters, LL={_gmm_cem.lower_bound_:.3f})')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 1: Overlapping Clusters

    In this first experiment, we create **overlapping clusters** with varying covariance structures (1000 samples each):

    - **Component 1**: centre (0, 2), spherical covariance ($\sigma^2 = 0.6$)
    - **Component 2**: centre (2, -2), spherical covariance ($\sigma^2 = 0.3$)
    - **Component 3**: centre (0, 0), diagonal covariance $\mathrm{diag}(1.2,\,0.3)$
    - **Component 4**: centre (2, 2), full covariance with correlation, $\begin{pmatrix}0.12 & 0.3\\ 0.3 & 1.2\end{pmatrix}$

    This scenario tests how EM vs CEM handle **ambiguous cluster boundaries** where observations could reasonably belong to multiple clusters. Each algorithm is run from several `init_means='points'` starts and the best run (highest log-likelihood) is kept.
    """)
    return


@app.cell
def _(FAST, GaussianMixture, device, time):
    N_RESTARTS = 10 if FAST else 20

    def fit_and_evaluate_gmm(X_tensor, n_components, covariance_type='full', cem=False, seed=1, iters=N_RESTARTS):
        """Fit a GMM with either EM or CEM from `iters` seeds; return the best run and the average time, iterations and log-likelihood."""
        best_gmm = None
        total_time = total_iters = total_ll = 0
        for i in range(iters):
            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, tol=1e-10, reg_covar=1e-06, max_iter=1000, init_means='points', random_state=seed + i, device=device, algorithm='cem' if cem else 'em')
            start_time = time.time()
            gmm.fit(X_tensor)
            total_time += time.time() - start_time
            total_iters += gmm.n_iter_
            total_ll += gmm.lower_bound_
            if best_gmm is None or gmm.lower_bound_ > best_gmm.lower_bound_:
                best_gmm = gmm
        return best_gmm, {'time': total_time / iters, 'iters': total_iters / iters, 'avg_ll': total_ll / iters, 'best_ll': best_gmm.lower_bound_}

    def summary_table(stats_em, stats_cem):
        rows = ['| | avg. time (s) | avg. iterations | best log-likelihood | avg. log-likelihood |', '|---|---|---|---|---|']
        for name, s in (('EM', stats_em), ('CEM', stats_cem)):
            rows.append(f"| {name} | {s['time']:.3f} | {s['iters']:.0f} | {s['best_ll']:.3f} | {s['avg_ll']:.3f} |")
        return '\n'.join(rows)
    return N_RESTARTS, fit_and_evaluate_gmm, summary_table


@app.cell
def _(SEED, device, generate_gmm_data, np):
    _n_samples = [1000, 1000, 1000, 1000]
    centers_overlap = [np.array([0, 2]), np.array([2, -2]), np.array([0, 0]), np.array([2, 2])]
    covs_overlap = [0.6 * np.eye(2), 0.3 * np.eye(2), np.array([[1.2, 0], [0, 0.3]]), np.array([[0.12, 0.3], [0.3, 1.2]])]
    X_tensor, _y_tensor = generate_gmm_data(centers_overlap, covs_overlap, _n_samples, device=device, random_state=SEED)
    X = X_tensor.cpu().numpy()
    labels = _y_tensor.cpu().numpy()
    return X, X_tensor, centers_overlap, covs_overlap, labels


@app.cell
def _(SEED, X_tensor, fit_and_evaluate_gmm):
    gmm_em, stats_em = fit_and_evaluate_gmm(X_tensor, 4, cem=False, seed=SEED)
    gmm_cem, stats_cem = fit_and_evaluate_gmm(X_tensor, 4, cem=True, seed=SEED)
    return gmm_cem, gmm_em, stats_cem, stats_em


@app.cell(hide_code=True)
def _(mo, stats_cem, stats_em, summary_table):
    mo.md(f"""
    ### Results Analysis: Overlapping Clusters

    {summary_table(stats_em, stats_cem)}

    With overlapping clusters the expected pattern is:

    - **CEM converges faster** (here {stats_em['iters'] / stats_cem['iters']:.1f}x fewer iterations on average) but reaches a **lower log-likelihood** than EM (average {stats_cem['avg_ll']:.3f} vs {stats_em['avg_ll']:.3f}; best {stats_cem['best_ll']:.3f} vs {stats_em['best_ll']:.3f}), because the mixture likelihood is not what it maximizes
    - **EM finds better optima** of that likelihood but needs more iterations
    - **CEM's hard assignments** are suboptimal when cluster boundaries are ambiguous

    This matches the theoretical expectation that CEM trades accuracy for speed when components overlap.
    """)
    return


@app.cell
def _(X, dynamic_figsize, gmm_cem, gmm_em, labels, plot_gmm, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=dynamic_figsize(1, 3))
    _fig.suptitle('Comparing the best fit of EM vs CEM')
    plot_gmm(X=X, true_labels=labels, ax=_axes[0], title='True Components', show_ellipses=False, show_means=False)
    plot_gmm(X=X, gmm=gmm_em, ax=_axes[1], title=f'Standard EM Clustering\n(Iterations: {gmm_em.n_iter_})', color_by_cluster=True, match_labels_to_true=True, ellipse_std_devs=[3], ellipse_fill=True, ellipse_alpha=0.1)
    plot_gmm(X=X, gmm=gmm_cem, ax=_axes[2], title=f'Classification EM Clustering\n(Iterations: {gmm_cem.n_iter_})', color_by_cluster=True, match_labels_to_true=True, ellipse_std_devs=[3], ellipse_fill=True, ellipse_alpha=0.1)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CEM shrinks the covariances

    Because CEM estimates each cluster from the points nearest to it, the tails that overlap a neighbouring cluster are cut off and the fitted covariances come out smaller than EM's. The table pairs every fitted component with the true blob it is closest to and compares the trace of $\Sigma_k$ (the total variance):
    """)
    return


@app.cell
def _(covs_overlap, centers_overlap, gmm_cem, gmm_em, linear_sum_assignment, mo, np):
    _true_means = np.stack(centers_overlap).astype(float)
    _rows = {}
    for _name, _g in (('EM', gmm_em), ('CEM', gmm_cem)):
        _d = np.linalg.norm(_g.means_.cpu().numpy()[:, None] - _true_means[None], axis=2)
        _fit_idx, _true_idx = linear_sum_assignment(_d)
        _rows[_name] = {int(j): float(np.trace(_g.covariances_[i].cpu().numpy())) for i, j in zip(_fit_idx, _true_idx)}
    _lines = ['| blob | true tr(Σ) | EM tr(Σ) | CEM tr(Σ) |', '|---|---|---|---|']
    for _j in range(len(covs_overlap)):
        _lines.append(f"| {_j + 1} | {np.trace(covs_overlap[_j]):.3f} | {_rows['EM'][_j]:.3f} | {_rows['CEM'][_j]:.3f} |")
    mo.md('\n'.join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Which likelihood does each algorithm increase?

    To see the objectives directly, refit from one fixed (k-means) start with `max_iter` = 1, 2, 3, ... (EM and CEM are deterministic given the initialization, so each fit is the prefix of the longer one) and evaluate both likelihoods after each iteration, per sample:

    $$
    \bar\ell(\theta)=\frac1n\sum_i\log\sum_k\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k),\qquad
    \bar\ell_C(\theta)=\frac1n\sum_i\max_k\log\big(\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)\big),
    $$

    where $\bar\ell_C$ is the classification likelihood at the C-step's optimal labelling.
    """)
    return


@app.cell
def _(FAST, GaussianMixture, X_tensor, device, torch, warnings):
    def two_likelihoods(gmm, X_t):
        """Mean mixture log-likelihood and mean classification log-likelihood of a fitted full-covariance GMM."""
        _lp = torch.stack([torch.log(gmm.weights_[k]) + torch.distributions.MultivariateNormal(gmm.means_[k], gmm.covariances_[k]).log_prob(X_t) for k in range(gmm.n_components)], dim=1)
        return torch.logsumexp(_lp, dim=1).mean().item(), _lp.max(dim=1).values.mean().item()

    trace_steps = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20] if FAST else [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]
    traces = {}
    for _alg in ('em', 'cem'):
        _rows = []
        for _m in trace_steps:
            _g = GaussianMixture(n_components=4, init_means='kmeans', random_state=1, tol=1e-30, max_iter=_m, algorithm=_alg, device=device)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # "did not converge" is expected: we stop on purpose
                _g.fit(X_tensor)
            _rows.append(two_likelihoods(_g, X_tensor))
        traces[_alg] = torch.tensor(_rows)
    return trace_steps, traces


@app.cell
def _(plt, trace_steps, traces):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    for _alg, _label in (('em', 'EM'), ('cem', 'CEM')):
        _axes[0].plot(trace_steps, traces[_alg][:, 0], marker='o', label=_label)
        _axes[1].plot(trace_steps, traces[_alg][:, 1], marker='o', label=_label)
    _axes[0].set_title(r'Mixture log-likelihood $\bar\ell(\theta_t)$ (EM maximizes this)')
    _axes[1].set_title(r'Classification log-likelihood $\bar\ell_C(\theta_t)$ (CEM maximizes this)')
    for _ax in _axes:
        _ax.set_xlabel('iteration t')
        _ax.legend()
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo, trace_steps, traces):
    _cem_lc_steps = traces['cem'][1:, 1] - traces['cem'][:-1, 1]
    _best_l = 'EM' if traces['em'][-1, 0] > traces['cem'][-1, 0] else 'CEM'
    _best_lc = 'EM' if traces['em'][-1, 1] > traces['cem'][-1, 1] else 'CEM'
    mo.md(f"""
    At the last checkpoint (iteration {trace_steps[-1]}) EM has mixture log-likelihood {traces['em'][-1, 0]:.4f} and classification log-likelihood {traces['em'][-1, 1]:.4f}; CEM has {traces['cem'][-1, 0]:.4f} and {traces['cem'][-1, 1]:.4f}. So the higher **mixture** likelihood belongs to **{_best_l}** and the higher **classification** likelihood to **{_best_lc}**: each algorithm is ahead on the objective it optimizes. The classification likelihood of the CEM iterates **never decreases** (smallest change between consecutive checkpoints: {_cem_lc_steps.min().item():+.2e}), as the two-step coordinate ascent guarantees.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 2: Well-Separated Clusters

    In this second experiment, we create **well-separated clusters** by moving the centres far apart (1000 samples each, same covariances as before):

    - **Component 1**: (0, 10), spherical covariance
    - **Component 2**: (10, -20), spherical covariance
    - **Component 3**: (0, 0), diagonal covariance
    - **Component 4**: (10, 10), full covariance with correlation

    This is the **ideal case for CEM**, where cluster boundaries are clear and unambiguous. According to theory:

    > "When the mixture components are well separated with similar proportions, the CEM algorithm is expected to provide a relevant clustering" (Celeux and Govaert, 1992)

    **Expected behaviour:** similar final accuracy for both algorithms, and CEM converging faster.
    """)
    return


@app.cell
def _(SEED, device, generate_gmm_data, np):
    _centers = [np.array([0, 10]), np.array([10, -20]), np.array([0, 0]), np.array([10, 10])]
    _covs = [0.6 * np.eye(2), 0.3 * np.eye(2), np.array([[1.2, 0], [0, 0.3]]), np.array([[0.12, 0.3], [0.3, 1.2]])]
    X_sep_tensor, _y_sep = generate_gmm_data(_centers, _covs, [1000] * 4, device=device, random_state=SEED)
    X_sep = X_sep_tensor.cpu().numpy()
    labels_sep = _y_sep.cpu().numpy()
    return X_sep, X_sep_tensor, labels_sep


@app.cell
def _(X_sep_tensor, fit_and_evaluate_gmm):
    gmm_em_sep, stats_em_sep = fit_and_evaluate_gmm(X_sep_tensor, 4, cem=False)
    gmm_cem_sep, stats_cem_sep = fit_and_evaluate_gmm(X_sep_tensor, 4, cem=True)
    return gmm_cem_sep, gmm_em_sep, stats_cem_sep, stats_em_sep


@app.cell(hide_code=True)
def _(mo, stats_cem_sep, stats_em_sep, summary_table):
    mo.md(f"""
    ### Results Analysis: Well-Separated Clusters

    {summary_table(stats_em_sep, stats_cem_sep)}

    - **The best run of each algorithm reaches the same optimum** (best log-likelihood {stats_em_sep['best_ll']:.3f} for EM, {stats_cem_sep['best_ll']:.3f} for CEM). Averaged over the random restarts both are pulled down by a few poor local optima (some `init_means='points'` starts put two means in the same cluster), and neither is consistently better
    - **CEM needs fewer iterations** ({stats_em_sep['iters'] / stats_cem_sep['iters']:.1f}x fewer on average) and less time
    - **The partitions are essentially identical** when clusters are well separated

    The hard-assignment strategy costs nothing when cluster boundaries are unambiguous.
    """)
    return


@app.cell
def _(
    X_sep,
    dynamic_figsize,
    gmm_cem_sep,
    gmm_em_sep,
    labels_sep,
    plot_gmm,
    plt,
):
    _fig, _axes = plt.subplots(1, 3, figsize=dynamic_figsize(1, 3))
    _fig.suptitle('Comparing the best fit of EM vs CEM')
    plot_gmm(X=X_sep, true_labels=labels_sep, ax=_axes[0], title='True Components', show_ellipses=False, show_means=False)
    plot_gmm(X=X_sep, gmm=gmm_em_sep, ax=_axes[1], title=f'Standard EM Clustering\n(Iterations: {gmm_em_sep.n_iter_})', color_by_cluster=True, match_labels_to_true=True, ellipse_std_devs=[3], ellipse_fill=True, ellipse_alpha=0.1)
    plot_gmm(X=X_sep, gmm=gmm_cem_sep, ax=_axes[2], title=f'Classification EM Clustering\n(Iterations: {gmm_cem_sep.n_iter_})', color_by_cluster=True, match_labels_to_true=True, ellipse_std_devs=[3], ellipse_fill=True, ellipse_alpha=0.1)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - CEM replaces the soft E-step posterior with a hard assignment (the C-step) and maximizes the classification likelihood $\ell_C(\theta,z)$ rather than the mixture likelihood $\ell(\theta)$; $\ell_C$ is nondecreasing, $\ell$ is not.
    - With overlapping clusters, EM's soft assignments reach a higher mixture log-likelihood and larger (less truncated) covariances, while CEM stops earlier on a cruder partition.
    - With well-separated clusters the two reach the same best solution and CEM needs fewer iterations.
    - Use `algorithm='cem'` when speed or a crisp partition matters more than the exact maximum-likelihood fit; `lower_bound_` stays the mixture log-likelihood, so fits remain comparable.

    **Next:** [07 -- Supervised Fitting](07_supervised_gmm.ipynb), which adds label information, or [15 -- Stochastic EM](15_stochastic_em.ipynb) for the stochastic variants.
    """)
    return


if __name__ == "__main__":
    app.run()
