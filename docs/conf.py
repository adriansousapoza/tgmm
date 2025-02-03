# conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # So Sphinx can find utils/ modules


project = 'TorchGMM'
copyright = '2025, Adrián A. Sousa-Poza'
author = 'Adrián A. Sousa-Poza'
release = '0.1.0'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']
