# Installation

## Requirements

- **Python**: 3.9 or higher (the minimum supported by PyTorch 2.5)
- **PyTorch**: 2.5 or higher (`torch>=2.5.1` in `requirements.txt`)
- **NumPy**: for numerical operations
- **SciPy**: special functions and distributions (the von Mises-Fisher Bessel functions, chi-squared quantiles, Hungarian label matching)
- **Matplotlib** and **seaborn**: `tgmm.plotting`, which `import tgmm` loads
- **tqdm**: progress bars in `HDBSCAN`

`import tgmm` imports all of the above at start-up, so all must be installed. If a plain `pip install tgmm` does
not pull them in automatically, install them explicitly:

```bash
pip install scipy matplotlib seaborn tqdm
```

## Installation Methods

### From PyPI (Recommended)

The easiest way to install tgmm is using pip:

```bash
pip install tgmm
```

This will install the latest stable release along with all required dependencies.

### From Source (Development)

For the latest features or if you want to contribute:

```bash
# Clone the repository
git clone https://github.com/adriansousapoza/tgmm.git
cd tgmm

# Install in editable mode
pip install -e .
```

The `-e` flag installs in editable mode, so changes to the source code are immediately available.

### With GPU Support

tgmm automatically uses CUDA if available. To enable GPU acceleration:

1. **Install CUDA-enabled PyTorch** following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/)

2. **Verify CUDA availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0)}")
   ```

3. **Use GPU in tgmm**:
   ```python
   from tgmm import GaussianMixture
   
   gmm = GaussianMixture(n_components=3, n_features=2, device='cuda')
   gmm.fit(X)  # X will be automatically moved to GPU
   ```

## Optional Dependencies

For running the tutorial notebooks (scikit-learn is used by some notebooks for datasets and PCA, pandas for the
`select_model(...).to_dataframe()` tables):

```bash
pip install jupyter scikit-learn pandas tqdm
# from a source checkout, the "examples" extra installs tqdm and pandas:
pip install -e ".[examples]"
```

For running the test suite (`mpmath` supplies reference Bessel values for the von Mises-Fisher tests):

```bash
pip install -e ".[test]"
```

For building this documentation:

```bash
pip install -e ".[docs]"   # mkdocs, mkdocs-material, mkdocstrings, mkdocs-jupyter, pymdown-extensions
mkdocs serve
```

## Verification

Verify your installation:

```python
import tgmm
print(f"tgmm version: {tgmm.__version__}")

# Test basic functionality
from tgmm import GaussianMixture
import torch

X = torch.randn(100, 2)
gmm = GaussianMixture(n_components=3, n_features=2)
gmm.fit(X)
labels = gmm.predict(X)
print(f"Fitted GMM with {gmm.n_components} components")
print(f"Converged: {gmm.converged_}")
```

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Ensure all dependencies are installed
pip install torch numpy scipy matplotlib seaborn tqdm

# Reinstall tgmm
pip uninstall tgmm
pip install tgmm
```

### CUDA Errors

If you have CUDA-related issues:

1. **Check PyTorch CUDA installation**:
   ```python
   import torch
   print(torch.version.cuda)  # Should show CUDA version
   ```

2. **Fall back to CPU**:
   ```python
   gmm = GaussianMixture(n_components=3, n_features=2, device='cpu')
   ```

### Version Conflicts

If you have version conflicts with dependencies:

```bash
# Create a fresh virtual environment
python -m venv tgmm_env
source tgmm_env/bin/activate  # On Windows: tgmm_env\Scripts\activate

# Install tgmm
pip install tgmm
```

## Next Steps

- **[Quick Start Guide](quickstart.md)** - Fit your first model
- **[User Guide](../user-guide/gaussian-mixture.md)** - The models and the mathematics behind them
- **[Tutorials](../tutorials/index.md)** - Runnable notebooks
- **[API Reference](../api/gaussian-mixture.md)** - Complete API documentation
