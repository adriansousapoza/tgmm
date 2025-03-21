import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from pygments.lexers import PythonLexer
from sphinx.highlighting import lexers
lexers['ipython3'] = PythonLexer()

project = 'TorchGMM'
copyright = '2025, Adrián A. Sousa-Poza'
author = 'Adrián A. Sousa-Poza'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.imgmath',  # Add this line for $-style math
    'nbsphinx',
    'sphinx_rtd_theme',
]

# Enable $-style math syntax:
mathjax3_config = {
    'tex': {'inlineMath': [['$', '$'], ['\\(', '\\)']]}
}

# Note: if you see warnings about "File `anyfontsize.sty` not found," you may need to install
# additional LaTeX packages that provide that .sty file, or remove references to it in your
# Sphinx configuration. The warnings are unrelated to using $ for math, but come from the
# LaTeX installation missing certain files.

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_theme_options = {}
html_show_sourcelink = False
html_static_path = ['_static']
html_css_files = ['custom.css']

napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True
