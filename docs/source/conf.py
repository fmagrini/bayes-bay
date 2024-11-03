# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import bayesbay as bb
from docutils import nodes
from docutils.parsers.rst import roles


def underline_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    """Enable underlined text"""
    return [nodes.emphasis(rawtext, text, classes=['underline'])], []

def bold_underline_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    """Enable bold-underlined text"""
    return [nodes.emphasis(rawtext, text, classes=['bold-underline'])], []

roles.register_local_role('underline', underline_role)
roles.register_local_role('bold-underline', bold_underline_role)


# -- Project information -----------------------------------------------------
project = "BayesBay"
copyright = f"{datetime.date.today().year}, InLab, BayesBay development team"
version = "dev" if "dev" in bb.__version__ else f"v{bb.__version__}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",        # Ensure MathJax support for LaTeX
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "myst_nb", 
    "sphinxcontrib.mermaid"
]

templates_path = ['_templates']
exclude_patterns = [
    "tutorials/jupyter_cache/*"
]

intersphinx_mapping = {
    "joblib": ("https://joblib.readthedocs.io/en/stable", None),
}

# settings for the sphinx-copybutton extension
copybutton_prompt_text = ">>> "

autodoc_typehints = 'none'

nb_execution_mode = "off"

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_title = f'{project} <span class="project-version">{version}</span>'
html_short_title = project
html_theme = 'sphinx_rtd_theme'
html_static_path = ["_static"]
html_css_files = ["style.css", "custom.css"]
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "fmagrini", # Username
    "github_repo": "bayes-bridge", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/source/", # Path in the checkout to the docs root
}

latex_elements = {
    'preamble': r'''
\usepackage{titlesec}
\titleformat{\rubric}[display]
  {\normalfont\Large\bfseries}
  {\thesection}{1em}{}
'''
}
