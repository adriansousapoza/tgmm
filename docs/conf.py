# conf.py

# Import standard libraries for file and system operations.
import os
import sys

# Add the parent directory to sys.path so that Sphinx can locate your project's modules.
sys.path.insert(0, os.path.abspath('..'))

# Import PythonLexer from pygments for syntax highlighting.
from pygments.lexers import PythonLexer

# Import lexers from Sphinx highlighting to register a custom lexer.
from sphinx.highlighting import lexers

# Register the PythonLexer for blocks marked as 'ipython3' to ensure proper highlighting of IPython code.
lexers['ipython3'] = PythonLexer()

# -- Project information -----------------------------------------------------

# The name of the project.
project = 'TorchGMM'
# Copyright information.
copyright = '2025, Adrián A. Sousa-Poza'
# The author of the project.
author = 'Adrián A. Sousa-Poza'
# The release version of the project.
release = '0.1.0'

# -- General configuration ---------------------------------------------------

# Extensions are Sphinx modules that add extra functionality.
extensions = [
    'sphinx.ext.autodoc',   # Automatically documents your Python code.
    'sphinx.ext.napoleon',  # Supports NumPy and Google style docstrings.
    'sphinx.ext.viewcode',  # Adds links to highlighted source code.
    'nbsphinx',             # Allows inclusion of Jupyter Notebook (.ipynb) files.
    'pydata_sphinx_theme'   # The PyData Sphinx Theme for a modern look.
]

# List of patterns, relative to the source directory, that Sphinx will ignore.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# Specify the HTML theme. Here we switch from the default to the PyData Sphinx Theme.
html_theme = 'pydata_sphinx_theme'

# Paths that contain custom static files (such as style sheets). They are copied into the final HTML build.
html_static_path = ['_static']

# -- Options for Napoleon (docstring style) ----------------------------------

# If True, use the "Parameters" section header in docstrings.
napoleon_use_param = True
# If False, do not include an automatically generated "Returns" header; you can specify it manually.
napoleon_use_rtype = False
# If True, use the "Attributes" section header for class attributes.
napoleon_use_ivar = True
