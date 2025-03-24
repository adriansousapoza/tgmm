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
    'sphinx.ext.mathjax',   # For dollar-sign math
    # 'sphinx.ext.imgmath', # Omit or comment out this line to avoid image-based math
    'nbsphinx',
    'sphinx_rtd_theme',
]

# Configure MathJax to allow $...$ inline and $$...$$ display math
mathjax3_config = {
    'tex': {
        'inlineMath': [
            ['$', '$'],        # dollar-sign inline math
            ['\\(', '\\)']     # optional: \( ... \) as well
        ],
        'displayMath': [
            ['$$', '$$'],      # dollar-sign display math
            ['\\[', '\\]']     # optional: \[ ... \]
        ]
    }
}
# By default, ReadTheDocs will load MathJax from its CDN:
mathjax_path = (
    'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
)

exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_theme_options = {}
html_show_sourcelink = False
html_static_path = ['_static']
html_css_files = ['custom.css']

# Napoleon settings
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True
