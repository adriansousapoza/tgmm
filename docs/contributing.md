# Contributing to TorchGMM

Thank you for your interest in contributing to TorchGMM! This guide will help you get started.

## Getting Started

### Development Installation

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/TorchGMM.git
cd TorchGMM
```

3. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install in development mode:

```bash
pip install -e .
pip install -r requirements.txt
```

5. Install development dependencies:

```bash
pip install pytest pytest-cov black flake8 mypy
```

## Development Workflow

### Creating a Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/` for new features
- `bugfix/` for bug fixes
- `docs/` for documentation changes
- `refactor/` for code refactoring

### Code Style

We follow PEP 8 guidelines with some specific conventions:

#### Formatting

Use `black` for automatic formatting:

```bash
black tgmm/
```

#### Linting

Check code quality with `flake8`:

```bash
flake8 tgmm/ --max-line-length=88 --ignore=E203,W503
```

#### Type Hints

Use type hints for function signatures:

```python
def fit(
    self, 
    X: torch.Tensor, 
    y: Optional[torch.Tensor] = None
) -> 'GaussianMixture':
    """Fit the GMM to data."""
    ...
```

Check with `mypy`:

```bash
mypy tgmm/
```

### Documentation

#### Docstrings

Use NumPy-style docstrings:

```python
def method_name(param1: int, param2: str) -> bool:
    """
    Short description.

    Longer description explaining the method in detail.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> result = method_name(42, "test")
    >>> print(result)
    True
    """
    ...
```

#### Building Documentation

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

Build and serve locally:

```bash
mkdocs serve
```

View at `http://127.0.0.1:8000`

### Testing

#### Running Tests

Run all tests:

```bash
pytest
```

Run specific test file:

```bash
pytest tests/test_gmm.py
```

Run with coverage:

```bash
pytest --cov=tgmm --cov-report=html
```

View coverage report: `open htmlcov/index.html`

#### Writing Tests

Place tests in the `tests/` directory. Use descriptive names:

```python
import torch
from tgmm import GaussianMixture

def test_fit_basic():
    """Test basic fitting functionality."""
    X = torch.randn(100, 2)
    gmm = GaussianMixture(n_components=3, n_features=2)
    gmm.fit(X)
    assert gmm.converged_

def test_predict_shape():
    """Test that predict returns correct shape."""
    X = torch.randn(100, 2)
    gmm = GaussianMixture(n_components=3, n_features=2)
    gmm.fit(X)
    labels = gmm.predict(X)
    assert labels.shape == (100,)
```

### Commit Messages

Write clear, descriptive commit messages:

```
Add feature for weighted samples in GMM fitting

- Add sample_weight parameter to fit() method
- Update E-step to handle weighted samples
- Add tests for weighted fitting
- Update documentation
```

Format:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed explanation with bullet points

## Submitting Changes

### Pull Request Process

1. Update your branch with latest main:

```bash
git checkout main
git pull upstream main
git checkout your-branch
git rebase main
```

2. Run all tests and checks:

```bash
pytest
black tgmm/
flake8 tgmm/
mypy tgmm/
```

3. Push to your fork:

```bash
git push origin your-branch
```

4. Create a pull request on GitHub

### Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: Explain what changed and why
- **Tests**: Include tests for new features
- **Documentation**: Update docs for user-facing changes
- **Single concern**: One feature/fix per PR

Example PR description:

```markdown
## Summary
Adds support for weighted samples in GMM fitting.

## Changes
- Add `sample_weight` parameter to `fit()` method
- Modify E-step to incorporate sample weights
- Update M-step for weighted parameter estimation

## Tests
- Added `test_weighted_fit()` in `tests/test_gmm.py`
- Added `test_weighted_predict()` for consistency check

## Documentation
- Updated `fit()` docstring with `sample_weight` parameter
- Added example in user guide

Closes #42
```

## Code Review

### What to Expect

- Reviewers will provide constructive feedback
- May request changes before merging
- Be responsive and collaborative
- Learn from the review process

### Reviewing Code

When reviewing others' PRs:

- Be respectful and constructive
- Test the changes locally
- Check for edge cases
- Verify documentation updates
- Ensure tests are adequate

## Areas for Contribution

### Feature Additions

- New initialization methods
- Additional covariance structures
- Alternative inference algorithms
- Performance optimizations

### Bug Fixes

- Check [Issues](https://github.com/TorchGMM/TorchGMM/issues) for known bugs
- Report new bugs with minimal reproducible examples

### Documentation

- Improve existing docs
- Add tutorials and examples
- Fix typos and clarify explanations

### Testing

- Increase test coverage
- Add edge case tests
- Performance benchmarks

### Performance

- Profile code for bottlenecks
- Optimize critical sections
- Add GPU-specific optimizations

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/TorchGMM/TorchGMM/discussions)
- **Bugs**: Open an [Issue](https://github.com/TorchGMM/TorchGMM/issues)
- **Chat**: Join our community chat (if available)

## Code of Conduct

### Our Standards

- Be welcoming and inclusive
- Respect differing viewpoints
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks
- Public or private harassment
- Publishing others' private information

## Recognition

Contributors will be:

- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes for significant contributions
- Acknowledged in academic citations for major features

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to TorchGMM! 🎉
