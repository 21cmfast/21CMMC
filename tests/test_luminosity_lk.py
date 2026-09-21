"""Tests of the luminosity function likelihood."""

import os

import numpy as np
import pytest

from py21cmmc import (
    CoreLuminosityFunction,
    LikelihoodLuminosityFunction,
    build_computation_chain,
)


def test_single_datafile(tmpdir):
    direc = tmpdir.mkdir("test_single_datafile")
    dfile = direc.join("test_datafile.npz").strpath
    nfile = direc.join("test_noisefile.npz").strpath

    core = CoreLuminosityFunction(redshift=[7, 8, 9], sigma=np.ones((3, 1)))
    lk = LikelihoodLuminosityFunction(simulate=True, datafile=dfile, noisefile=nfile)
    chain = build_computation_chain(core, lk, setup=True)

    assert os.path.exists(dfile)

    assert isinstance(lk.data, dict)
    assert isinstance(lk.noise, dict)

    model = lk.get_fiducial_model()
    assert isinstance(model, dict)
    assert lk.data["Muv"].shape[1] == 3
    assert model["Muv"].shape[1] == 3

    chain({})

    lk = LikelihoodLuminosityFunction(datafile=dfile, noisefile=nfile)
    chain = build_computation_chain(core, lk, setup=True)

    assert isinstance(lk.data, dict)

    model = lk.get_fiducial_model()
    assert isinstance(model, dict)

    assert lk.data["Muv"].shape[1] == 3
    assert model["Muv"].shape[1] == 3

    chain({})


def test_multi_datafile(tmpdir):
    direc = tmpdir.mkdir("test_single_datafile")

    dfile7 = direc.join("test_datafile7.npz").strpath
    dfile8 = direc.join("test_datafile8.npz").strpath
    nfile7 = direc.join("test_noisefile7.npz").strpath
    nfile8 = direc.join("test_noisefile8.npz").strpath

    core7 = CoreLuminosityFunction(
        redshift=[7], n_muv_bins=100, sigma=np.ones((1, 1)), name="7"
    )
    core8 = CoreLuminosityFunction(
        redshift=[8], n_muv_bins=80, sigma=np.ones((1, 1)), name="8"
    )

    lk7 = LikelihoodLuminosityFunction(
        simulate=True, datafile=dfile7, noisefile=nfile7, name="7"
    )
    lk8 = LikelihoodLuminosityFunction(
        simulate=True, datafile=dfile8, noisefile=nfile8, name="8"
    )

    chain = build_computation_chain([core7, core8], [lk7, lk8], setup=True)

    assert isinstance(lk7.data, dict)
    assert lk7.data["Muv"].shape[1] == 1
    assert lk8.data["Muv"].shape[1] == 1

    assert lk8.data["Muv"][0].shape[1] != lk7.data["Muv"][0].shape[1]
    chain({})

    lk7read = LikelihoodLuminosityFunction(datafile=dfile7, noisefile=nfile7, name="7")
    lk8read = LikelihoodLuminosityFunction(datafile=dfile8, noisefile=nfile8, name="8")
    chain = build_computation_chain([core7, core8], [lk7read, lk8read], setup=True)

    assert isinstance(lk7read.data, dict)
    assert lk7read.data["Muv"].shape[1] == 1
    assert lk8read.data["Muv"].shape[1] == 1

    assert lk8read.data["Muv"][0].shape[1] != lk7read.data["Muv"][0].shape[1]
    chain({})


def test_create_mock():
    core = CoreLuminosityFunction(redshift=[7])
    lk = LikelihoodLuminosityFunction(simulate=True)
    with pytest.raises(ValueError):
        build_computation_chain(core, lk, setup=True)


def test_computeLikelihood_floors_too_few_valid_points():
    """A model LF that's ~all NaN at a redshift must be rejected, not crash."""
    from py21cmmc.likelihood import _LOGLIKE_FLOOR

    core = CoreLuminosityFunction(redshift=[7], sigma=np.ones((1, 1)))
    lk = LikelihoodLuminosityFunction(simulate=True)
    build_computation_chain(core, lk, setup=True)

    model = lk.get_fiducial_model()

    # Leave only 2 valid (non-NaN) Muv bins: a cubic spline needs more
    # points than its degree (k=3, so >=4), so this used to crash with a
    # scipy dfitpack error; it should now just floor the likelihood.
    model["lfunc"] = np.full_like(model["lfunc"], np.nan)
    model["lfunc"][..., :2] = 1.0

    lnl = lk.computeLikelihood(model)
    assert lnl[0] == _LOGLIKE_FLOOR


def test_computeLikelihood_floors_extreme_values():
    """An ill-conditioned model LF must not produce a non-finite/huge lnl."""
    from py21cmmc.likelihood import _LOGLIKE_FLOOR

    core = CoreLuminosityFunction(redshift=[7], sigma=np.ones((1, 1)))
    lk = LikelihoodLuminosityFunction(simulate=True)
    build_computation_chain(core, lk, setup=True)

    model = lk.get_fiducial_model()

    # A model log-luminosity-function of 150 everywhere is a perfectly
    # valid (non-NaN) spline fit, but predicts 10**150 galaxies/mag/Mpc^3 --
    # comparing that against realistic data blows up to an astronomically
    # large (but still finite) chi^2, which must be floored rather than fed
    # to a sampler as-is.
    model["lfunc"] = np.full_like(model["lfunc"], 150.0)

    lnl = lk.computeLikelihood(model)
    assert lnl[0] == _LOGLIKE_FLOOR
