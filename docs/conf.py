"""Sphinx configuration for the 21cmMC docs."""

import py21cmmc

project = "21cmMC"
author = "The 21cmFAST collaboration"
release = py21cmmc.__version__
version = ".".join(release.split(".")[:2])

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
html_theme = "furo"
exclude_patterns = ["_build"]
