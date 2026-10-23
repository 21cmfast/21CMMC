"""Tests of the neutral fraction likelihood."""

import numpy as np
import pytest

from py21cmmc import LikelihoodNeutralFraction


def test_neutral_fraction_spline():
    lk = LikelihoodNeutralFraction(redshift=5.9, xHI=0.06, xHI_sigma=0.05)
    lk._require_spline = True
    zs = np.array([5.0, 6.0, 7.0])
    model = {
        "redshifts": zs,
        "xHI": np.array([[0.0, 0.2, 0.5], [0.0, 0.1, 0.5]]),
        "err": np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]),
    }
    lnl = lk.computeLikelihood(model)
    xhi = np.array([0.18, 0.09])  # linear interpolation at z = 5.9
    sigma = np.sqrt(0.05**2 + np.array([0, 0.1]) ** 2)
    assert np.allclose(lnl, -0.5 * ((0.06 - xhi) / sigma) ** 2)

    # a model the spline cannot be built from raises (it used to print and go on)
    model["redshifts"] = zs[::-1]
    model["err"] = np.zeros_like(model["xHI"])
    with pytest.raises(ValueError):
        lk.computeLikelihood(model)
