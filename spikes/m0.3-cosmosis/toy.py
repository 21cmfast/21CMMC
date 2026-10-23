"""Analytic toy posterior shared by every M0.3 spike script.

A correlated 5D Gaussian likelihood inside a uniform box prior. The box is wide
enough that the evidence is analytic to high accuracy, so samplers can be checked
against known moments and ln Z. Derived values include a scalar and an 8-element
array, standing in for blobs such as a power spectrum.
"""

import os
import time

import numpy as np

NDIM = 5
LOW, HIGH = -10.0, 10.0
MEAN = np.array([0.5, -1.0, 0.0, 2.0, -0.5])
SIGMA = np.array([0.5, 1.0, 0.8, 1.5, 0.3])
_CORR = np.full((NDIM, NDIM), 0.4) + 0.6 * np.eye(NDIM)
COV = _CORR * np.outer(SIGMA, SIGMA)
_ICOV = np.linalg.inv(COV)
_LOGNORM = -0.5 * (NDIM * np.log(2 * np.pi) + np.linalg.slogdet(COV)[1])

# Analytic ln Z for a normalised Gaussian inside a uniform box that contains it.
LOGZ = -NDIM * np.log(HIGH - LOW)

VEC_LEN = 8
_K = np.arange(1, VEC_LEN + 1, dtype=float)

# Optional artificial cost per evaluation, to mimic an expensive simulator.
SLEEP = float(os.environ.get("TOY_SLEEP", "0"))


def derived(p):
    """Return the derived values the pipeline should store for parameters ``p``."""
    p = np.asarray(p, dtype=float)
    return {"sum": float(p.sum()), "vec": p[0] * _K + p[1]}


def log_like(p):
    """Log-likelihood only."""
    if SLEEP:
        time.sleep(SLEEP)
    r = np.asarray(p, dtype=float) - MEAN
    return float(_LOGNORM - 0.5 * r @ _ICOV @ r)


def log_like_with_derived(p):
    """Return ``(lnL, derived)`` in the form CosmoSIS's from_likelihood_function expects."""
    return log_like(p), derived(p)
