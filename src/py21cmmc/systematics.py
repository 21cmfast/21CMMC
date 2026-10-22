"""Priors on a positive systematic power added to each band power.

Used by :class:`~py21cmmc.likelihood.LikelihoodPspec` (``systematics=...``), which
marginalises the systematic analytically in each bin. The prior width in a bin with
noise ``sigma`` at wavenumber ``k`` [1/Mpc] is

    width = scale * sigma * (k / k0) ** (-k_slope)

``Flat(np.inf)`` (a flat prior on [0, inf)) gives the HERA upper-limit likelihood.
Each prior can also draw systematics, e.g. to add them to a mock (:meth:`sample`).
"""
import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri


def log_diff_ndtr(a, b):
    """Log of Phi(a) - Phi(b) for a > b, with Phi the standard normal CDF."""
    flip = b > 0
    a, b = np.where(flip, -b, a), np.where(flip, -a, b)
    return log_ndtr(a) + np.log1p(-np.exp(log_ndtr(b) - log_ndtr(a)))


class Systematics:
    """Base class of the priors on the systematic power in each bin.

    Parameters
    ----------
    scale : float
        Width of the prior at k = k0, in units of the noise of the bin.
    k_slope : float
        The width scales as (k / k0) ** (-k_slope).
    k0 : float
        Pivot wavenumber [1/Mpc].
    rho : float
        Correlation between neighbouring k bins of the systematics drawn by
        :meth:`sample` (the likelihood treats the bins as independent).
    """

    def __init__(self, scale=1.0, k_slope=0.0, k0=0.2, rho=0.3):
        self.scale = scale
        self.k_slope = k_slope
        self.k0 = k0
        self.rho = rho

    def width(self, sigma, k):
        """Width of the prior in bins with noise sigma at k [1/Mpc]."""
        return self.scale * sigma * (k / self.k0) ** (-self.k_slope)

    def sample(self, sigma, k, rng):
        """One draw of the systematic in each k bin."""
        i = np.arange(len(k))
        corr = self.rho ** np.abs(i[:, None] - i[None, :])
        z = np.linalg.cholesky(corr) @ rng.standard_normal(len(k))
        return self.width(sigma, k) * self.unit_ppf(ndtr(z))

    def unit_ppf(self, u):
        """Inverse CDF of the prior with unit width."""
        raise NotImplementedError

    def lnlike(self, d, m, sigma, k):
        """Log-likelihood of data d given model m in each bin, systematic marginalised."""
        raise NotImplementedError


class Flat(Systematics):
    """Systematic uniform in [0, width]. ``Flat(np.inf)`` is the HERA likelihood."""

    def unit_ppf(self, u):
        """Inverse CDF of the prior with unit width."""
        if np.isinf(self.scale):
            raise ValueError("Cannot sample from an unbounded flat prior.")
        return u

    def lnlike(self, d, m, sigma, k):
        """Log-likelihood of data d given model m in each bin, systematic marginalised."""
        x = (d - m) / sigma
        if np.isinf(self.scale):
            return log_ndtr(x)
        w = self.width(sigma, k)
        return log_diff_ndtr(x, x - w / sigma) - np.log(w)


class HalfNormal(Systematics):
    """Systematic distributed as the absolute value of N(0, width^2)."""

    def unit_ppf(self, u):
        """Inverse CDF of the prior with unit width."""
        return ndtri(0.5 + 0.5 * u)

    def lnlike(self, d, m, sigma, k):
        """Log-likelihood of data d given model m in each bin, systematic marginalised."""
        tau = self.width(sigma, k)
        omega = np.sqrt(sigma**2 + tau**2)
        y = (d - m) / omega
        return (
            np.log(2 / omega)
            - 0.5 * y**2
            - 0.5 * np.log(2 * np.pi)
            + log_ndtr(tau / sigma * y)
        )


class Exponential(Systematics):
    """Systematic exponentially distributed, with mean width."""

    def unit_ppf(self, u):
        """Inverse CDF of the prior with unit width."""
        return -np.log1p(-u)

    def lnlike(self, d, m, sigma, k):
        """Log-likelihood of data d given model m in each bin, systematic marginalised."""
        lam = self.width(sigma, k)
        x = d - m
        return (
            -np.log(lam)
            + sigma**2 / (2 * lam**2)
            - x / lam
            + log_ndtr(x / sigma - sigma / lam)
        )
