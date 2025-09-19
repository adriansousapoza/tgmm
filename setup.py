from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements from requirements.txt and filter for core dependencies."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            lines = f.read().splitlines()
        
        # Filter for core dependencies (torch, numpy, matplotlib)
        # Exclude development dependencies like setuptools, sphinx, etc.
        core_deps = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Include only torch, numpy, matplotlib
                if any(line.lower().startswith(dep) for dep in ['torch', 'numpy', 'matplotlib']):
                    core_deps.append(line)
        return core_deps
    return []

def read_docs_requirements():
    """Read docs requirements from docs/requirements.txt and filter for docs dependencies."""
    docs_requirements_path = os.path.join(os.path.dirname(__file__), 'docs', 'requirements.txt')
    if os.path.exists(docs_requirements_path):
        with open(docs_requirements_path, 'r') as f:
            lines = f.read().splitlines()
        
        # Filter for docs-specific dependencies
        docs_deps = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Include sphinx and pygments for docs
                if any(line.lower().startswith(dep) for dep in ['sphinx', 'pygments']):
                    docs_deps.append(line)
        return docs_deps
    return ["Sphinx>=7.1.0,<8.2.0", "Pygments>=2.19.1"]

setup(
    name="tgmm",
    version="0.1.7",
    author="Your Name",
    author_email="adrian.sousapoza@gmail.com",
    description="A Gaussian Mixture Model (GMM) based on Expectation-Maximisation (EM) implemented in PyTorch",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/adriansousapoza/TorchGMM",
    packages=find_packages(),
    install_requires=read_requirements(),
    extras_require={
        "docs": read_docs_requirements(),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
