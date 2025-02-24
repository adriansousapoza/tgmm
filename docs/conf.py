# conf.py

import os
import sys

# Add the parent directory to sys.path so that Sphinx can locate your project's modules.
sys.path.insert(0, os.path.abspath('..'))

# Import PythonLexer from pygments for syntax highlighting.
from pygments.lexers import PythonLexer
from sphinx.highlighting import lexers
lexers['ipython3'] = PythonLexer()

# -- Project information -----------------------------------------------------

project = 'TorchGMM'
copyright = '2025, Adrián A. Sousa-Poza'
author = 'Adrián A. Sousa-Poza'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',   # Automatically documents your Python code.
    'sphinx.ext.napoleon',  # Supports NumPy and Google style docstrings.
    'sphinx.ext.viewcode',  # Adds links to highlighted source code.
    'nbsphinx',             # Allows inclusion of Jupyter Notebook (.ipynb) files.
    'sphinx_rtd_theme',     # Read the Docs theme.
]

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    # Optional: customize theme options, for example:
    # "collapse_navigation": False,
    # "navigation_depth": 4,
}

# Remove the "View page source" link.
html_show_sourcelink = False

# Paths that contain custom static files (e.g., style sheets)
html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Options for Napoleon (docstring style) ----------------------------------

napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True
