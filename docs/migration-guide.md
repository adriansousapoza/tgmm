# Migration Guide: Sphinx to MkDocs

This guide helps users transition from the old Sphinx-based documentation to the new MkDocs Material documentation.

## What Changed?

### Documentation System

- **Old**: Sphinx with ReadTheDocs theme (`.rst` files)
- **New**: MkDocs with Material theme (`.md` files)

### Why MkDocs Material?

1. **Modern Design**: Clean, responsive interface with light/dark mode
2. **Better Navigation**: Improved search, tabs, and mobile experience
3. **Easier Authoring**: Markdown is simpler than reStructuredText
4. **Built-in Features**: Code copy buttons, annotations, better syntax highlighting
5. **Faster Builds**: Quicker documentation generation

## Finding Content

### Homepage

=== "Old (Sphinx)"
    ```
    https://tgmm.readthedocs.io/en/latest/
    ```

=== "New (MkDocs)"
    ```
    https://tgmm.readthedocs.io/
    ```

### Getting Started

| Old Location | New Location |
|-------------|--------------|
| Installation instructions in README | [Installation Guide](getting-started/installation.md) |
| Basic usage in README | [Quick Start Guide](getting-started/quickstart.md) |

### User Guides

| Old Location | New Location |
|-------------|--------------|
| `source/gaussian_mixture.rst` | [Gaussian Mixture Model](user-guide/gaussian-mixture.md) |
| `source/gmm_initializer.rst` | [GMM Initializer](user-guide/gmm-initializer.md) |
| `source/clustering_metrics.rst` | [Clustering Metrics](user-guide/clustering-metrics.md) |
| `source/plotting.rst` | [Plotting & Visualization](user-guide/plotting.md) |

### Tutorials

All Jupyter notebook tutorials remain in the same location:

| Tutorial | Location |
|---------|----------|
| Basic GMM | [notebooks/gmm.ipynb](notebooks/gmm.ipynb) |
| Metrics & Evaluation | [notebooks/metrics.ipynb](notebooks/metrics.ipynb) |
| Using Priors (MAP) | [notebooks/priors.ipynb](notebooks/priors.ipynb) |
| CEM Algorithm | [notebooks/cem.ipynb](notebooks/cem.ipynb) |
| Visualization | [notebooks/visualise.ipynb](notebooks/visualise.ipynb) |
| PCA Plotting Demo | [notebooks/pca_plotting.ipynb](notebooks/pca_plotting.ipynb) |
| NIW Priors Comparison | [notebooks/niw_priors_comparison.ipynb](notebooks/niw_priors_comparison.ipynb) |
| Sampling | [notebooks/sampling.ipynb](notebooks/sampling.ipynb) |

### API Reference

API documentation is now auto-generated from docstrings:

| Module | New Location |
|--------|--------------|
| `GaussianMixture` | [API: GaussianMixture](api/gaussian-mixture.md) |
| `GMMInitializer` | [API: GMMInitializer](api/gmm-initializer.md) |
| `ClusteringMetrics` | [API: ClusteringMetrics](api/clustering-metrics.md) |
| `plotting` | [API: plotting](api/plotting.md) |

## New Features

### Enhanced Search

The new documentation includes:

- **Better search**: More accurate results with highlighting
- **Search suggestions**: Auto-complete as you type
- **Section search**: Find specific sections within pages

### Code Blocks

All code blocks now have:

- **Copy buttons**: Click to copy code
- **Syntax highlighting**: Better color schemes
- **Line numbers**: Optional line numbering
- **Annotations**: Hover tooltips for explanations

Example:

```python
from tgmm import GaussianMixture
import torch

# Create GMM instance
gmm = GaussianMixture(
    n_components=3,  # (1)!
    n_features=2,
    covariance_type='full'
)
```

1. Number of Gaussian components

### Tabs

Content organized in tabs for better comparison:

=== "Full Covariance"
    ```python
    gmm = GaussianMixture(covariance_type='full')
    ```

=== "Diagonal Covariance"
    ```python
    gmm = GaussianMixture(covariance_type='diag')
    ```

=== "Spherical Covariance"
    ```python
    gmm = GaussianMixture(covariance_type='spherical')
    ```

### Admonitions

Important information highlighted with admonitions:

!!! tip "GPU Acceleration"
    Move your data and model to GPU for faster computation:
    ```python
    X_gpu = X.to('cuda')
    gmm = gmm.to('cuda')
    ```

!!! warning "Numerical Stability"
    Use regularization to prevent singular covariance matrices:
    ```python
    gmm = GaussianMixture(reg_covar=1e-6)
    ```

!!! note "Best Practice"
    Always check convergence after fitting:
    ```python
    if not gmm.converged_:
        print("Warning: Model did not converge")
    ```

### Math Rendering

Improved LaTeX math rendering:

Inline math: $p(\mathbf{x} | \theta)$

Display math:

$$
p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

### Dark Mode

Toggle between light and dark themes using the icon in the header.

### Mobile Friendly

Responsive design works great on mobile devices with:

- Collapsible navigation
- Touch-friendly interface
- Optimized layout

## For Contributors

### Building Documentation Locally

=== "Old (Sphinx)"
    ```bash
    cd docs
    make html
    open _build/html/index.html
    ```

=== "New (MkDocs)"
    ```bash
    pip install -r docs/requirements.txt
    mkdocs serve
    # Visit http://127.0.0.1:8000
    ```

### Writing Documentation

=== "Old (Sphinx/RST)"
    ```rst
    Section Title
    =============

    Subsection
    ----------

    Some text with ``code`` and a `link <url>`_.

    .. code-block:: python

        def example():
            pass

    .. note::
        This is a note
    ```

=== "New (MkDocs/Markdown)"
    ```markdown
    # Section Title

    ## Subsection

    Some text with `code` and a [link](url).

    ```python
    def example():
        pass
    ```

    !!! note
        This is a note
    ```

### File Locations

| Type | Old Location | New Location |
|------|-------------|--------------|
| Configuration | `docs/conf.py` | `mkdocs.yml` |
| Source files | `docs/source/*.rst` | `docs/user-guide/*.md`, `docs/api/*.md` |
| Static files | `docs/_static/` | `docs/assets/` |
| Build output | `docs/_build/html/` | `site/` |

## Breaking Changes

### Deep Linking

Some URLs have changed. Update bookmarks:

| Old URL Pattern | New URL Pattern |
|----------------|-----------------|
| `/en/latest/source/gaussian_mixture.html` | `/user-guide/gaussian-mixture/` |
| `/en/latest/source/modules.html` | `/api/gaussian-mixture/` |

### API Documentation Format

- **Old**: Manually written API docs
- **New**: Auto-generated from docstrings using mkdocstrings

This ensures documentation always matches the code.

## Getting Help

If you can't find what you're looking for:

1. Use the **search bar** (top of page)
2. Check the **navigation menu**
3. Visit the [GitHub repository](https://github.com/adriansousapoza/TorchGMM)
4. Open an [issue](https://github.com/adriansousapoza/TorchGMM/issues)

## Feedback

We'd love to hear your thoughts on the new documentation:

- [Open a discussion](https://github.com/adriansousapoza/TorchGMM/discussions) for general feedback
- [Report issues](https://github.com/adriansousapoza/TorchGMM/issues) for bugs or missing content
- [Contribute improvements](contributing.md) via pull requests

---

Welcome to the new TorchGMM documentation! 🎉
