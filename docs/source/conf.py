import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'langesim_optim'
copyright = '2025, Gabriel Tellez'
author = 'Gabriel Tellez'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# AutoDoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
add_module_names = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# GitHub Pages settings
github_repo = os.environ.get('GITHUB_REPOSITORY', 'GabrielTellez/langesim_optim')
github_user = github_repo.split('/')[0] if '/' in github_repo else 'GabrielTellez'
github_repo_name = github_repo.split('/')[1] if '/' in github_repo else 'langesim_optim'

html_baseurl = f'/{github_repo_name}/'
html_context = {
    'display_github': True,
    'github_user': github_user,
    'github_repo': github_repo_name,
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
} 