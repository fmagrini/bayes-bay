# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import bayesbridge as bb


# -- Project information -----------------------------------------------------
project = "Bayes Bridge"
copyright = f"{datetime.date.today().year}, InLab, BayesBridge development team"
version = "dev" if "dev" in bb.__version__ else f"v{bb.__version__}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "myst_nb", 
    "sphinx.ext.napoleon",
]

templates_path = ['_templates']
exclude_patterns = []


# settings for the sphinx-copybutton extension
copybutton_prompt_text = ">>> "

autodoc_typehints = 'none'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_title = f'{project} <span class="project-version">{version}</span>'
html_short_title = project
html_theme = 'sphinx_book_theme'
html_static_path = ["_static"]
html_css_files = ["style.css"]
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "fmagrini", # Username
    "github_repo": "bayes-bridge", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/source/", # Path in the checkout to the docs root
}
