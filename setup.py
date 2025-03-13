from setuptools import setup, find_packages

setup(
    name="tgmm",
    version="0.1.5",
    author="Your Name",
    author_email="adrian.sousapoza@gmail.com",
    description="A Gaussian Mixture Model (GMM) based on Expectation-Maximisation (EM) implemented in PyTorch",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/adriansousapoza/TorchGMM",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.1",
        "numpy>=1.23.0",
        "matplotlib>=3.10.1"
    ],
    extras_require={
        "docs": [
            "Sphinx>=8.2.3",
            "Pygments>=2.19.1",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
