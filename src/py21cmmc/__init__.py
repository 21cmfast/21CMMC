"""21cmMC: Bayesian inference for 21cmFAST and the 21-cm signal.

This is the v2 development branch. The v1 code lives on the ``v1.x`` branch and in
the 1.x releases on PyPI.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("21cmMC")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "unknown"

__all__ = ["__version__"]
