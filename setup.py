from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements from requirements.txt and filter for core dependencies."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            lines = f.read().splitlines()
        
        # Filter for core dependencies (torch, numpy, matplotlib)
        core_deps = []
        in_core_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('# Core dependencies'):
                in_core_section = True
                continue
            elif line.startswith('#') and in_core_section:
                in_core_section = False
                continue
            elif in_core_section and line and not line.startswith('#'):
                core_deps.append(line)
        
        # Fallback to pattern matching if sections aren't found
        if not core_deps:
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    if any(line.lower().startswith(dep) for dep in ['torch', 'numpy', 'matplotlib']):
                        core_deps.append(line)
        return core_deps
    return []

def read_docs_requirements():
    """Read docs requirements from requirements.txt."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            lines = f.read().splitlines()
        
        # Filter for docs dependencies
        docs_deps = []
        in_docs_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('# Documentation dependencies'):
                in_docs_section = True
                continue
            elif line.startswith('#') and in_docs_section:
                in_docs_section = False
                continue
            elif in_docs_section and line and not line.startswith('#'):
                docs_deps.append(line)
        
        return docs_deps
    return ["Sphinx>=7.1.0,<8.2.0", "Pygments>=2.19.1"]

def read_examples_requirements():
    """Read example/notebook requirements from requirements.txt."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            lines = f.read().splitlines()
        
        # Filter for example dependencies
        example_deps = []
        in_examples_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('# Example/notebook dependencies'):
                in_examples_section = True
                continue
            elif line.startswith('#') and in_examples_section:
                in_examples_section = False
                continue
            elif in_examples_section and line and not line.startswith('#'):
                example_deps.append(line)
        
        return example_deps
    return []

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
        "examples": read_examples_requirements(),
        "all": read_docs_requirements() + read_examples_requirements(),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
