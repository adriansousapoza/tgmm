# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "tgmm",
#     "torch; sys_platform != 'emscripten'",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
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
    return FAST, N_INIT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Semi-Supervised and Positive-Unlabeled Fitting

    Often only some rows have labels: a few confirmed examples, or a handful of annotated classes among mostly raw data. `fit(X, labels=...)` accepts partial labels (`-1` marks an unknown row) on every mixture family via the shared `BaseMixture`, running semi-supervised EM in which labeled rows keep their known component assignment while unlabeled rows are soft-assigned as usual. `PositiveUnlabeledMixture` handles the harder case where *only positives* are ever labeled (positive-unlabeled learning, Elkan & Noto, 2008).

    **What you'll learn**

    - how much a small labeled fraction (5-20%) helps cluster recovery on ambiguous, overlapping data
    - how to run PU learning with a Student-t mixture and read the estimated class prior `class_prior_` and per-row positive probabilities
    - how the SCAR assumption (labeled positives are a uniform random subsample of all positives) makes this identifiable, and what `anchor` does
    - the equations behind both fits (the labeled-class likelihood, the clamped E-step, the NIW anchor)

    **Prerequisites:** [07 -- Supervised Fitting](07_supervised_gmm.ipynb), [16 -- Student-t Mixtures](16_student_t.ipynb), [17 -- Von Mises-Fisher Mixtures](17_von_mises_fisher.ipynb).

    **Notation:** $n$ samples $x_i$, $K$ components with weights $\pi_k$ and densities $f_k$ (Gaussian, Student-t or von Mises-Fisher), responsibilities $r_{ik}$, effective counts $n_k=\sum_i r_{ik}$.
    See also the API page for [`PositiveUnlabeledMixture`](../api/positive-unlabeled-mixture.md).
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
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from tgmm import GaussianMixture, StudentTMixture, PositiveUnlabeledMixture
    from tgmm.metrics import ClusteringMetrics
    from tgmm.plotting import plot_gmm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plt.rcParams['figure.dpi'] = 100  # tgmm's theme uses 300 dpi; marimo embeds figures at 2x, so keep outputs small
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f'Using device: {device}')
    return (
        ClusteringMetrics,
        GaussianMixture,
        PositiveUnlabeledMixture,
        SEED,
        plot_gmm,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: semi-supervised fitting on three overlapping blobs

    Three overlapping 2-D blobs where unsupervised EM's cluster <-> class correspondence is
    genuinely ambiguous. We compare fits at 5%, 20%, and 100% of rows labeled, against plain
    unsupervised EM (0% labeled) on the same data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The model: classes own blocks of components

    Let the classes be $c\in\{1,\dots,C\}$ (these are `classes_`, the sorted distinct label values). The $K$ components are partitioned into contiguous **blocks** $B_c$, one per class:
    with `components_per_class=m` every class owns $m$ components ($K=mC$), and with a dict, class $c$ owns `components_per_class[c]` of them (the idea of mclust's MclustDA).
    `class_of_component_[k]` says which class component $k$ belongs to.

    An observation *without* a label has the ordinary mixture density, and one with known class $c$ has the restriction of that sum to its block:

    $$
    p(x)=\sum_{k=1}^{K}\pi_kf_k(x),\qquad p(x,c)=\sum_{k\in B_c}\pi_kf_k(x).
    $$

    The weights $\pi_k$ are therefore joint class-and-component probabilities (they sum to one over *all* components), and the class proportions of the labeled data influence them. With $L$ the
    labeled rows (class $c_i$ for $i\in L$) and $U$ the unlabeled ones, the semi-supervised log-likelihood is

    $$
    \ell_\text{SS}(\theta)=\sum_{i\in L}\log\sum_{k\in B_{c_i}}\pi_kf_k(x_i)\;+\;\sum_{i\in U}\log\sum_{k=1}^{K}\pi_kf_k(x_i),
    $$

    which is exactly what `lower_bound_` reports for a semi-supervised fit (averaged over rows).

    ```python
    labels = torch.full((X.shape[0],), -1)      # -1 = unknown
    labels[known_idx] = known_labels
    gmm = GaussianMixture(n_components=3).fit(X, labels=labels)
    gmm.predict_class(X)                         # class labels, not raw component indices
    ```

    `n_components` must equal the total the `components_per_class` layout implies (for the default `1`, the number of distinct labels), and label values need not be contiguous integers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The algorithm

    **Fully labeled (no `-1`): one exact M-step.** Each row's class never depends on the current parameters. With one component per class the responsibilities are one-hot, and a single
    M-step gives the per-class weight $\pi_c=n_c/n$, mean and covariance: the fully-labeled limit of Classification EM (`n_iter_` is `1`, `converged_` is `True`). With $m_c>1$ components, that class's block
    is fit by ordinary unsupervised EM on *only* its rows and its weights are rescaled by $n_c/n$ so that they sum to the class proportion (tied covariance types cannot be assembled from independent blocks
    and raise a `ValueError`). Any configured `mean_prior` / `covariance_prior` still applies, which gives a regularised supervised fit for classes with few samples.

    **Partially labeled: semi-supervised EM.** The **E-step** computes the ordinary posterior $r_{ik}\propto\pi_kf_k(x_i)$ for every row, then *clamps* the labeled rows to their class block: for $i\in L$,

    $$
    r_{ik}\;\leftarrow\;\begin{cases}\dfrac{\pi_kf_k(x_i)}{\sum_{j\in B_{c_i}}\pi_jf_j(x_i)}&k\in B_{c_i},\\[2mm]0&k\notin B_{c_i},\end{cases}
    $$

    (one-hot when the block has one component). A labeled row can then never "vote" for another class but, with $m_c>1$, still splits softly across its own class's components; unlabeled rows keep the full
    posterior. This is the exact E-step for $\ell_\text{SS}$, because the posterior over components given that $x_i$ belongs to class $c_i$ is the in-block renormalisation above. The **M-step** is the family's usual
    one run on the clamped $r_{ik}$: for the Gaussian, $n_k$, $\bar y_k=\sum_ir_{ik}x_i/n_k$ and $W_k=\sum_ir_{ik}(x_i-\bar y_k)(x_i-\bar y_k)^\top$ feed the updates of
    [01 -- Introduction](01_introduction.ipynb) (the Student-t and vMF updates are those of [16](16_student_t.ipynb) and [17](17_von_mises_fisher.ipynb)). With `unlabeled_weight=1` and `label_confidence=1` this is plain EM on $\ell_\text{SS}$,
    so the log-likelihood never decreases. A few labels per class often resolve what unsupervised EM cannot: which cluster is which class, and where the boundary falls on overlapping data. The initialisation itself is still
    unsupervised (e.g. k-means) and the clamping then pulls each block onto its class, so use `n_init > 1` if a poor start matters.

    **Two weighting knobs.**

    - `unlabeled_weight` $=w$ (default `1.0`) multiplies every unlabeled row's responsibilities before the M-step, $r_{ik}\leftarrow w\,r_{ik}$ for $i\in U$, so unlabeled rows contribute $w\sum_{i\in U}r_{ik}$ to $n_k$, $\bar y_k$ and $W_k$ while labeled rows count fully.
      $w=0$ ignores the unlabeled data (a fit on the labeled rows alone) and $w=1$ treats all rows equally. A small $w$ protects against a mis-specified component density letting the (typically far more numerous) unlabeled rows swamp the labels.
    - `label_confidence` $=c$ (default `1.0`) is for labels you do not fully trust. A labeled row's responsibility becomes the blend

    $$
    r_{i\cdot}=c\;r^{\text{in-block}}_{i\cdot}+(1-c)\;r^{\text{out-of-block}}_{i\cdot},\qquad
    r^{\text{out-of-block}}_{ik}=\frac{\pi_kf_k(x_i)\,\mathbb 1[k\notin B_{c_i}]}{\sum_{j\notin B_{c_i}}\pi_jf_j(x_i)},
    $$

    &nbsp;&nbsp;&nbsp;&nbsp;i.e. mass $c$ stays on the stated class and mass $1-c$ goes to the *other* classes' components in proportion to their posterior. It is a heuristic for occasional mislabeling, **not** part of the likelihood:
    `lower_bound_` still evaluates $\ell_\text{SS}$ with the labels taken at face value.

    **CEM.** With `algorithm='cem'`, unlabeled rows are hard-assigned to their highest-responsibility component each iteration, while labeled rows keep their clamped responsibilities.

    **Scope.** Partial labels support `algorithm='em'` and `'cem'` only (`'sem'`, `'saem'` and online EM with `step_size` raise `NotImplementedError`). Gibbs mode (`n_components=None`) is not supported with `labels` (a `ValueError`: there is no fixed-$K$
    target), nor are tied covariance types combined with `components_per_class > 1`, `sample_weight`, `noise=True` or `missing='em'`.
    """)
    return


@app.cell
def _(SEED, plt, torch):
    _gen = torch.Generator().manual_seed(SEED)
    n_per = 200
    true_centers = torch.tensor([[0.0, 0.0], [3.0, 3.0], [3.0, -3.0]])
    X = torch.cat([torch.randn(n_per, 2, generator=_gen) * 1.6 + true_centers[0], torch.randn(n_per, 2, generator=_gen) * 1.6 + true_centers[1], torch.randn(n_per, 2, generator=_gen) * 1.6 + true_centers[2]])
    true_labels = torch.cat([torch.full((n_per,), _k) for _k in range(3)])
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', s=10)
    plt.title('3 overlapping blobs (true labels)')
    plt.gcf()
    return X, true_labels


@app.cell
def _(ClusteringMetrics, FAST, GaussianMixture, X, torch, true_labels):
    def partial_labels(labels, frac, seed):
        gen = torch.Generator().manual_seed(seed)
        mask = torch.rand(labels.shape[0], generator=gen) < frac
        return (torch.where(mask, labels, torch.full_like(labels, -1)), mask)
    label_fracs = [0.0, 0.05, 0.2, 1.0]
    # init_means='random' + n_init=1 (rather than the usual kmeans + n_init>1)
    # so unsupervised EM's local-optimum quality genuinely depends on luck --
    # otherwise a good initialization alone already resolves this data well
    # regardless of labels, since ARI is itself invariant to how clusters are
    # *numbered* (only labels fix that, which ARI doesn't measure).
    aris_by_frac = {frac: [] for frac in label_fracs}
    example_models = {}
    n_seeds = 4 if FAST else 10
    for seed in range(n_seeds):
        for _frac in label_fracs:
            labels = None if _frac == 0.0 else partial_labels(true_labels, _frac, seed=100 + seed)[0]
            _gmm = GaussianMixture(n_components=3, init_means='random', random_state=seed, n_init=1, max_iter=200)
            _gmm.fit(X, labels=labels)
            aris_by_frac[_frac].append(ClusteringMetrics.adjusted_rand_score(true_labels, _gmm.predict(X)))
            if seed == 0:
                example_models[_frac] = _gmm
    for _frac in label_fracs:
        vals = aris_by_frac[_frac]
        print(f'{int(_frac * 100):3d}% labeled: mean ARI = {sum(vals) / len(vals):.3f} (over {n_seeds} random-init seeds)')
    return example_models, label_fracs


@app.cell
def _(
    ClusteringMetrics,
    X,
    example_models,
    label_fracs,
    plot_gmm,
    plt,
    true_labels,
):
    _fig, _axes = plt.subplots(1, 4, figsize=(20, 5))
    for _ax, _frac in zip(_axes, label_fracs):
        _gmm = example_models[_frac]
        _ari = ClusteringMetrics.adjusted_rand_score(true_labels, _gmm.predict(X))
        plot_gmm(X, gmm=_gmm, ax=_ax, title=f'{int(_frac * 100)}% labeled (seed 0, ARI={_ari:.2f})')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even a handful of labels (5-20%) measurably improves cluster recovery over plain
    unsupervised EM on this ambiguous, overlapping data (here, deliberately run from a single
    random initialization so unsupervised EM's local-optimum quality varies -- with a strong
    initializer like k-means and several restarts, as `GaussianMixture` uses by default, plain
    unsupervised EM would already resolve this particular dataset well on its own, and the only
    thing labels would add is knowing *which* cluster is which class, which `adjusted_rand_score`
    itself can't measure since it's invariant to how clusters are numbered).
    """)
    return


@app.cell
def _(GaussianMixture, SEED, X, torch, true_labels):
    _full = GaussianMixture(n_components=3, random_state=SEED).fit(X, labels=true_labels)
    _proportions = torch.bincount(true_labels).float() / len(true_labels)
    _class_means = torch.stack([X[true_labels == _c].mean(0) for _c in range(3)])
    print(f'fully labeled, one component per class: n_iter_ = {_full.n_iter_}, converged_ = {_full.converged_}')
    print('weights_ equal the class proportions n_c / n:', torch.allclose(_full.weights_, _proportions, atol=1e-6))
    print('means_ equal the per-class sample means    :', torch.allclose(_full.means_, _class_means, atol=1e-5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With everything labeled there is nothing to iterate: one M-step returns exactly the per-class proportions and sample means.

    ### More than one component per class

    Real classes are not always unimodal. `components_per_class` (an `int`, or a `dict` keyed by label value) gives a class more than one component. `predict(X)` returns raw component indices;
    `predict_class(X)` sums each row's responsibility over every class block and returns the class with the largest total, mapped back through `classes_`:

    $$
    \hat c(x)=\arg\max_c\sum_{k\in B_c}r_k(x).
    $$

    When `components_per_class` is a dict its *keys* define the full set of classes (a key need not appear in `labels`), and every labeled value must be one of those keys. Below, class 0 is **bimodal**
    (two blobs, one on each side) and class 1 is a single blob; only about 9% of the rows are labeled. Started from a single unsupervised initialisation the fit can land in a poor optimum, and `n_init`
    (which keeps the run with the highest semi-supervised log-likelihood) fixes that.
    """)
    return


@app.cell
def _(GaussianMixture, torch):
    _g = torch.Generator().manual_seed(1)
    _a = torch.cat([torch.randn(150, 2, generator=_g) * 0.6 + torch.tensor([-4.0, 0.0]), torch.randn(150, 2, generator=_g) * 0.6 + torch.tensor([4.0, 0.0])])
    _b = torch.randn(300, 2, generator=_g) * 0.6 + torch.tensor([0.0, 4.0])
    X_multi = torch.cat([_a, _b])
    y_multi = torch.cat([torch.zeros(300, dtype=torch.long), torch.ones(300, dtype=torch.long)])
    labels_multi = torch.where(torch.rand(600, generator=_g) < 0.1, y_multi, torch.full_like(y_multi, -1))

    for _n_init, _seed in [(1, 1), (5, 1)]:
        _m = GaussianMixture(n_components=3, components_per_class={0: 2, 1: 1}, random_state=_seed, n_init=_n_init).fit(X_multi, labels=labels_multi)
        _acc = float((_m.predict_class(X_multi) == y_multi).float().mean())
        print(f'n_init={_n_init}: class accuracy {_acc:.2f}, lower_bound_ = {_m.lower_bound_:.3f}, '
              f'classes_ = {_m.classes_.tolist()}, class_of_component_ = {_m.class_of_component_.tolist()}')
    print(f'{int((labels_multi != -1).sum())} of {len(labels_multi)} rows labeled')

    _full_multi = GaussianMixture(n_components=3, components_per_class={0: 2, 1: 1}, random_state=1).fit(X_multi, labels=y_multi)
    print('fully labeled: weight of each class block =', [round(float(_full_multi.weights_[_full_multi.class_of_component_ == _c].sum()), 3) for _c in (0, 1)], '(the class proportions)')
    return X_multi, labels_multi, y_multi


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The two knobs on imperfect labels

    To see `unlabeled_weight` and `label_confidence` at work, take the three overlapping blobs again, label 15% of the rows and **flip 20% of those labels** to a random class, then average the class accuracy
    of `predict_class` over several random single initialisations.
    """)
    return


@app.cell
def _(FAST, GaussianMixture, X, true_labels, torch):
    def _accuracy(**knobs):
        _accs = []
        for _s in range(4 if FAST else 10):
            _gen = torch.Generator().manual_seed(100 + _s)
            _lab = torch.where(torch.rand(len(true_labels), generator=_gen) < 0.15, true_labels, torch.full_like(true_labels, -1))
            _flip = (_lab >= 0) & (torch.rand(len(true_labels), generator=_gen) < 0.2)
            _lab[_flip] = torch.randint(0, 3, (int(_flip.sum()),), generator=_gen)
            _m = GaussianMixture(n_components=3, init_means='random', n_init=1, random_state=_s, max_iter=200, **knobs).fit(X, labels=_lab)
            _accs.append(float((_m.predict_class(X) == true_labels).float().mean()))
        return sum(_accs) / len(_accs)

    knob_table = {'defaults': {}, 'unlabeled_weight=0.3': dict(unlabeled_weight=0.3), 'unlabeled_weight=0 (labeled rows only)': dict(unlabeled_weight=0.0),
                  'label_confidence=0.8': dict(label_confidence=0.8), 'label_confidence=0.6': dict(label_confidence=0.6)}
    for _name, _knobs in knob_table.items():
        print(f'{_name:42s} mean class accuracy = {_accuracy(**_knobs):.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `unlabeled_weight` mostly trades how much the unlabeled rows may move the fit against the labels. `label_confidence` is a blunt instrument: it moves mass off the stated class for *every* labeled row, not only the wrong ones, so it can
    hurt when the labels are mostly right, as the table shows for this data. Use it when you know labels are noisy, and check the effect on held-out data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: positive-unlabeled learning on a "cancer cells vs. everything else" simulation

    A `StudentTMixture`-based PU model: a minority "cancer cell" population with heavier-tailed
    within-class variation (some genuinely atypical cells), embedded in a larger population of
    several other, unlabeled cell types. Only a small fraction of the true positives are
    confirmed-labeled; everything else is unlabeled.

    **The model.** `PositiveUnlabeledMixture` models the unlabeled density as

    $$
    p_U(x)=\pi\,f_P(x)+(1-\pi)\,f_N(x),
    $$

    with $f_P$ and $f_N$ themselves mixtures (a positive block of `n_components_positive` components and a negative block of `n_components_negative`), and estimates both densities and $\pi$ jointly:

    ```python
    pu = PositiveUnlabeledMixture(n_components_positive=1, n_components_negative=5)
    pu.fit(X, positive_mask)                # bool mask, True = confirmed positive
    pu.class_prior_                         # estimated fraction of *unlabeled* rows that are positive
    pu.predict_proba_positive(X)            # P(positive) for arbitrary X
    pu.predict_positive(X, threshold=0.5)
    ```

    **How it is fit.** It is a thin wrapper over the semi-supervised fit above: it builds the chosen `family`'s mixture with `components_per_class={1: n_components_positive, 0: n_components_negative}` (so the negative
    block comes *first* in component order, the positive block last), labels `1` on `positive_mask` and `-1` everywhere else, and fits it. The labeled positives are clamped to the positive block, each unlabeled row keeps the full
    posterior over both blocks, and with $P$ the positive block

    $$
    P(\text{positive}\mid x)=\sum_{k\in P}r_k(x)\quad(\texttt{predict\_proba\_positive}),\qquad
    \hat\pi=\frac1{|U|}\sum_{i\in U}\sum_{k\in P}r_{ik}\quad(\texttt{class\_prior\_}),
    $$

    i.e. `class_prior_` is the average positive-block responsibility over the *unlabeled* rows only. Do not read it off `weights_`: the mixture weights are estimated over *all* rows, labeled positives included, so the positive
    block's total weight is $\approx(n_L+\hat\pi|U|)/n$, larger than $\hat\pi$.

    **SCAR.** This assumes the labeled positives are a uniform random sample of *all* true positives (*Selected Completely At Random*; Elkan & Noto, 2008), not a biased subset such as "the easiest-to-confirm cases". Under SCAR the labeled
    positives are draws from $f_P$ itself, which is what allows them to be clamped to the positive block. If they were selected some other way, `class_prior_` and the fitted densities can be biased: SCAR is a real assumption, not a formality.
    Estimating $\pi$, the mixing proportion of the unobserved positive class inside the unlabeled pool, is also called **mixture proportion estimation**. A related, distinct problem is **semi-supervised novelty detection** (Blanchard, Lee & Scott, 2010):
    given a labeled "normal" sample and an unlabeled mix of normal and novel points, decide *which* unlabeled points are novel. It is closer to `anchor='fixed'` below (the labeled class acts as a known, unmoving reference density) than to the `'prior'` default.
    """)
    return


@app.cell
def _(SEED, torch):
    d = 10
    _gen = torch.Generator().manual_seed(SEED)
    n_positive_true = 150
    n_negative_true = 600
    pi_true = n_positive_true / (n_positive_true + n_negative_true)
    positive_center = torch.full((d,), 3.0)
    X_positive = torch.randn(n_positive_true, d, generator=_gen) * 1.0 + positive_center
    neg_centers = [torch.full((d,), -3.0), torch.full((d,), -3.0) * torch.tensor([1.0] * 3 + [-1.0] * 7)]
    X_negative = torch.cat([torch.randn(n_negative_true // 2, d, generator=_gen) * 1.2 + c for c in neg_centers])
    X_pu = torch.cat([X_positive, X_negative])
    true_positive = torch.cat([torch.ones(n_positive_true, dtype=torch.bool), torch.zeros(X_negative.shape[0], dtype=torch.bool)])
    label_frac = 0.15
    gen2 = torch.Generator().manual_seed(SEED + 1)
    positive_indices = true_positive.nonzero(as_tuple=True)[0]
    n_labeled = int(label_frac * positive_indices.numel())
    labeled_idx = positive_indices[torch.randperm(positive_indices.numel(), generator=gen2)[:n_labeled]]
    positive_mask = torch.zeros(X_pu.shape[0], dtype=torch.bool)
    positive_mask[labeled_idx] = True
    print(f'True class prior (fraction positive): {pi_true:.3f}')
    print(f'Labeled positives: {positive_mask.sum().item()} / {true_positive.sum().item()} true positives')
    return X_pu, positive_mask, true_positive


@app.cell
def _(
    N_INIT,
    PositiveUnlabeledMixture,
    SEED,
    X_pu,
    positive_mask,
    true_positive,
):
    pu = PositiveUnlabeledMixture(n_components_positive=1, n_components_negative=2, family='student_t', anchor='fixed', random_state=SEED, n_init=min(3, N_INIT), max_iter=200)
    pu.fit(X_pu, positive_mask)
    pred_positive = pu.predict_positive(X_pu)
    tp = (pred_positive & true_positive).sum().item()
    fp = (pred_positive & ~true_positive).sum().item()
    fn = (~pred_positive & true_positive).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pi_unlabeled_true = true_positive[~positive_mask].float().mean().item()
    print(f'Estimated class_prior_ (fraction of UNLABELED rows that are positive): {pu.class_prior_:.3f}')
    print(f'True fraction of unlabeled rows that are positive:                      {pi_unlabeled_true:.3f}')
    print(f'F1 of predict_positive on all rows (vs. true positives):                {f1:.3f}')
    return (pu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The positive-probability map

    Projecting onto the first two dimensions (where the positive/negative separation is
    strongest by construction): confirmed-labeled positives are marked with a star, and color
    shows `predict_proba_positive` for every row -- the model's estimate of how likely each
    *unlabeled* row is to actually be positive.
    """)
    return


@app.cell
def _(X_pu, plt, positive_mask, pu):
    proba_positive = pu.predict_proba_positive(X_pu)
    plt.figure(figsize=(6, 5))
    _sc = plt.scatter(X_pu[:, 0], X_pu[:, 1], c=proba_positive, cmap='RdYlBu_r', s=12, vmin=0, vmax=1)
    plt.scatter(X_pu[positive_mask, 0], X_pu[positive_mask, 1], marker='*', s=120, facecolors='none', edgecolors='black', linewidths=1.2, label='labeled positive')
    plt.colorbar(_sc, label='P(positive)')
    plt.legend()
    plt.title('PositiveUnlabeledMixture: positive-probability map')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Picking `anchor`

    The mixture proportion $\pi$ is only identifiable if the negative block cannot simply absorb the positive one (a sufficiently flexible negative block can explain every unlabeled point and drive $\hat\pi\to0$).
    `anchor` controls how tightly the positive block stays tied to the labeled positives. All three variants start from a first fit of the positive block on the labeled positives alone, $(\mu_P^0,\Sigma_P^0)$:

    | `anchor` | positive block | corresponds to |
    | --- | --- | --- |
    | `'fixed'` | frozen at the labeled-only fit after every M-step (through `frozen_components`); only the negative block and the weights keep moving | classical mixture-proportion estimation / semi-supervised novelty detection |
    | `'prior'` (default) | used as an NIW prior of strength $n_L$ (below), so unlabeled positives can refine it without drifting towards the negatives. **Gaussian only** | a softer version of the same: genuinely positive but different-looking unlabeled points can still pull the estimate, within reason |
    | `'free'` | initialised from the labeled-only fit, then plain semi-supervised EM with no further anchoring | plain PU learning with no identifiability safeguard: can drift towards the negatives on strongly overlapping data |

    For `'prior'`, after each M-step the positive block's parameters are replaced by an NIW posterior (the conjugate update of [10 -- Priors](10_priors.ipynb)) with prior mean $\mu_P^0$, prior scale $\Psi_0=n_L\Sigma_P^0$ and
    prior strength $\lambda_0=\nu_0=n_L$ (the number of labeled positives), using the positive block's current responsibilities:

    $$
    \mu_P=\frac{\lambda_0\mu_P^0+n_P\bar y_P}{\lambda_0+n_P},\qquad
    \Sigma_P=\frac{\Psi_0+W_P+\frac{\lambda_0n_P}{\lambda_0+n_P}(\bar y_P-\mu_P^0)(\bar y_P-\mu_P^0)^\top}{\nu_0+n_P},
    $$

    where $n_P$, $\bar y_P$, $W_P$ are that block's effective count, weighted mean and scatter. The labeled positives enter twice, once through the prior and once as data with responsibility 1, which anchors the block strongly: the prior is worth about
    $n_L$ pseudo-observations.

    `family='gaussian'` (default), `'student_t'` or `'vmf'` picks the mixture family for both blocks: `'student_t'` is a reasonable choice when either class has outliers, `'vmf'` when `X` is L2-normalised directional data. `anchor='prior'` is
    Gaussian-only (NIW conjugate priors are not a concept for the other two families), so use `'fixed'` or `'free'` with `'student_t'` / `'vmf'`. Any keyword not listed (`covariance_type`, `random_state`, `max_iter`, ...) is forwarded to the underlying
    mixture's constructor, and anything not defined on `PositiveUnlabeledMixture` itself (`bic`, `score`, `sample`, ...) is forwarded to the fitted mixture, `pu.model_`. `n_components_negative=None` (an unbounded Gibbs sampler for the negative block) is not
    implemented and raises `NotImplementedError`. The cell below fits the variants on the data above.
    """)
    return


@app.cell
def _(N_INIT, PositiveUnlabeledMixture, SEED, X_pu, positive_mask, true_positive):
    _truth = float(true_positive[~positive_mask].float().mean())
    print(f'true fraction of unlabeled rows that are positive: {_truth:.3f}\n')
    print(f"{'family':10s} {'anchor':7s} {'class_prior_':>13s} {'F1':>6s}")
    for _family, _anchor in [('gaussian', 'fixed'), ('gaussian', 'prior'), ('gaussian', 'free'), ('student_t', 'fixed'), ('student_t', 'free')]:
        _m = PositiveUnlabeledMixture(n_components_positive=1, n_components_negative=2, family=_family, anchor=_anchor,
                                      random_state=SEED, n_init=min(3, N_INIT), max_iter=200).fit(X_pu, positive_mask)
        _pred = _m.predict_positive(X_pu)
        _tp, _fp, _fn = (_pred & true_positive).sum().item(), (_pred & ~true_positive).sum().item(), (~_pred & true_positive).sum().item()
        _f1 = 2 * _tp / max(2 * _tp + _fp + _fn, 1)
        print(f'{_family:10s} {_anchor:7s} {_m.class_prior_:13.3f} {_f1:6.3f}')
    try:
        PositiveUnlabeledMixture(family='student_t', anchor='prior')
    except ValueError as _err:
        print('\nstudent_t with anchor="prior":', _err)
    return


@app.cell
def _(GaussianMixture, X_pu, positive_mask, torch, true_positive):
    _labeled_only = GaussianMixture(n_components=1, random_state=0).fit(X_pu[positive_mask])
    _fresh = X_pu[true_positive & ~positive_mask]                      # true positives the model was NOT told about
    _diff = _fresh - _labeled_only.means_[0]
    _maha2 = ((_diff @ torch.linalg.inv(_labeled_only.covariances_[0])) * _diff).sum(1)
    print(f'labeled-only fit uses {int(positive_mask.sum())} points in {X_pu.shape[1]} dimensions')
    print(f'mean squared Mahalanobis distance of the {len(_fresh)} unlabeled true positives to it: {float(_maha2.mean()):.1f}  (about {X_pu.shape[1]} if the fit were exact)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On this data `anchor='fixed'` underestimates $\pi$ badly, while `'prior'` and `'free'` recover it. The reason is visible above: a full covariance estimated from a few labeled points in 10 dimensions is noisy, and its
    inverse **inflates Mahalanobis distances**, so unlabeled positives the model was never told about look far from the frozen positive block and are explained by the negative block instead. Letting the positive
    block adapt (inside an NIW prior, or freely) repairs that. Do not read this as "`'free'` is best": it has no safeguard against drifting towards the negatives and simply did not drift on these well-separated classes.
    With more labeled positives, or a diagonal / spherical `covariance_type`, `'fixed'` suffers less.
    """)
    return


@app.cell
def _(pu, positive_mask, X_pu):
    _positive_block = pu.model_.class_of_component_ == 1
    _w_positive = float(pu.weights_[_positive_block].sum())
    _n_l, _n_u, _n = int(positive_mask.sum()), int((~positive_mask).sum()), X_pu.shape[0]
    print(f'total weight of the positive block (weights_) : {_w_positive:.4f}')
    print(f'(n_L + class_prior_ * |U|) / n                 : {(_n_l + pu.class_prior_ * _n_u) / _n:.4f}   -- the same number, so weights_ is not the class prior')
    print(f'class_prior_ itself                            : {pu.class_prior_:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Caveats

    - **`class_prior_` quality** depends on the negative block being able to model the negative class without wandering into the positive region. More negative components make that easier *and* make identifiability harder, so use
      `anchor='fixed'` or `'prior'` and check that `class_prior_` is stable across `n_components_negative`.
    - Labeled positives are clamped to the positive block, so with the default `label_confidence=1` a mislabeled "positive" is never corrected. To allow that, pass `label_confidence=c` with $c<1$ through `**family_kwargs`; the unlabeled rows are handled as usual.
    - `predict_proba_positive` uses the fitted mixture weights, which include the labeled positives (see above), whereas `class_prior_` is the right estimate of the positive fraction in a fresh unlabeled sample.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - Semi-supervised fitting (`fit(X, labels=...)` with `-1` for unknown) works on every tgmm mixture family, and a modest labeling budget can noticeably improve cluster recovery on ambiguous, overlapping data.
    - `PositiveUnlabeledMixture` estimates the class prior (the fraction of *unlabeled* rows that are positive) and a per-row positive probability from only a small labeled positive subset, under the SCAR assumption -- useful whenever confirming a negative is expensive or impossible, but confirming a positive is not.
    - `anchor` controls how tightly the positive block stays tied to the labeled positives, which is what keeps the problem identifiable.

    **Next:** [19 -- Noise, Missing Values and Conditioning](19_noise_missing_conditioning.ipynb), or back to the [tutorial overview](../tutorials/index.md).

    **References**

    - Elkan, C. & Noto, K. (2008). Learning classifiers from only positive and unlabeled data. *KDD*.
    - Fraley, C. & Raftery, A. E. (2002). Model-based clustering, discriminant analysis, and density estimation. *JASA*, 97(458), 611-631 (MclustDA).
    - Blanchard, G., Lee, G. & Scott, C. (2010). Semi-supervised novelty detection. *JMLR*, 11, 2973-3009.
    """)
    return


if __name__ == "__main__":
    app.run()
