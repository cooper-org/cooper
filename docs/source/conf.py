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

import cooper

# -- Project information -----------------------------------------------------

project = "Cooper"
copyright = "2022, The Cooper Developers"
author = "The Cooper Developers"

# The full version, including alpha/beta/rc tags
release = f"v{cooper.__version__}"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.napoleon", # napoleon on top of autodoc: https://stackoverflow.com/a/66930447 might correct some warnings
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
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

source_suffix = ['.md']

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]

# Handle Latex-style references
bibtex_encoding = "latin"
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
    "logo_only": False,
    "display_version": True,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_with_keys": True,
    "prev_next_buttons_location": "bottom",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

html_logo = "_static/cooper_logo_200px.png"

# intersphinx maps
intersphinx_mapping = {
    # "python": ("https://docs.python.org/3", None),
    "python": ("https://python.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../../tutorials/scripts/",
    # "doc_module": "cooper",
    "gallery_dirs": "auto_tutorials",  # path to save gallery generated examples
    # "backreferences_dir": os.path.join("modules", "generated"),
    # "show_memory": True,
    # "reference_url": {"cooper": None},
    # "filename_pattern": r"/plot_\.py",
    "ignore_pattern": r"__init__\.py|.*_utils.py",
    "line_numbers": True,
    # "run_stale_examples": True,
}
