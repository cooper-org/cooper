# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys

# sys.path.insert(0, os.path.abspath("."))

import cooper.version

# -- Project information -----------------------------------------------------

project = "Cooper"
copyright = "2022, Cooper Developers"
author = "Cooper Developers"

# The full version, including alpha/beta/rc tags
release = f"v{cooper.version.version}"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

mathjax3_config = {
    "extensions": ["tex2jax.js"],
    "TeX": {
        "Macros": {
            "argmin": "\\DeclareMathOperator*{\\argmin}{\\mathbf{arg\\,min}}",
            "argmax": "\\DeclareMathOperator*{\\argmin}{\\mathbf{arg\\,max}}",
            "bs": "\\newcommand{\\bs}[1]{\\boldsymbol{#1}}",
        },
    },
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\(", "\)"]],
    },
    "jax": ["input/TeX", "output/HTML-CSS"],
    "displayAlign": "left",
}


# Handle Latex-style references
bibtex_reference_style = "author_year"
bibtex_bibfiles = ["references.bib"]

todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "analytics_id": "",
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "navigation_with_keys": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

# intersphinx maps
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../../tutorials/scripts/",
    # "doc_module": "cooper",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_tutorials",
    # "backreferences_dir": os.path.join("modules", "generated"),
    # "show_memory": True,
    # "reference_url": {"cooper": None},
    # "filename_pattern": r"/plot_\.py",
    "ignore_pattern": r"__init__\.py|.*_utils.py",
    "line_numbers": True,
    # "run_stale_examples": True,
}
