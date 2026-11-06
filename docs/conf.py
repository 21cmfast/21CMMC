"""Sphinx configuration for the 21cmMC docs."""

import py21cmmc

project = "21cmMC"
author = "The 21cmFAST collaboration"
release = py21cmmc.__version__
version = ".".join(release.split(".")[:2])

extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
html_theme = "furo"
exclude_patterns = ["_build"]

# The tutorials are MyST notebooks, run on every build; an error in one fails it.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 600
myst_enable_extensions = ["dollarmath", "amsmath"]
