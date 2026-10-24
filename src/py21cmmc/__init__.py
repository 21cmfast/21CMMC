"""21cmMC: Bayesian inference for 21cmFAST and the 21-cm signal."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("21cmMC")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "unknown"

__all__ = ["__version__"]
