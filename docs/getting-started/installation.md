# Installation

## Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.5 or higher
- **NumPy**: For numerical operations
- **SciPy**: For statistical functions and some initialization methods

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
git clone https://github.com/adriansousapoza/TorchGMM.git
cd TorchGMM

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

For running tutorials and visualization:

```bash
pip install jupyter matplotlib seaborn
```

For building documentation:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-jupyter
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
pip install torch numpy scipy scikit-learn

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

Next steps: Check out the [Quick Start Guide](quickstart.md) or explore the [Tutorials](../tutorials/index.md).
- **[API Reference](../api/gaussian-mixture.md)** - Complete API documentation
