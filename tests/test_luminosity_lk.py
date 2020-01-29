from py21cmmc import (
    LikelihoodLuminosityFunction,
    CoreLuminosityFunction,
    build_computation_chain,
)
import os
import pytest
import numpy as np
import shutil


def test_single_datafile(tmpdir):
    direc = tmpdir.mkdir("test_single_datafile")
    dfile = direc.join("test_datafile.npz").strpath
    nfile = direc.join("test_noisefile.npz").strpath

    core = CoreLuminosityFunction(redshift=[7, 8, 9], sigma=np.ones((3, 1)))
    lk = LikelihoodLuminosityFunction(simulate=True, datafile=dfile, noisefile=nfile)
    chain = build_computation_chain(core, lk, setup=True)

    assert os.path.exists(dfile)
    shutil.copy(dfile, "/home/steven/")

    assert isinstance(lk.data, dict)
    assert isinstance(lk.noise, dict)

    model = lk.get_fiducial_model()
    assert isinstance(model, dict)

    assert len(lk.data["Muv"]) == 3
    assert len(model["Muv"]) == 3

    chain({})

    lk = LikelihoodLuminosityFunction(datafile=dfile, noisefile=nfile)
    chain = build_computation_chain(core, lk, setup=True)

    assert isinstance(lk.data, dict)

    model = lk.get_fiducial_model()
    assert isinstance(model, dict)

    assert len(lk.data["Muv"]) == 3
    assert len(model["Muv"]) == 3

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
    assert len(lk7.data["Muv"]) == 1
    assert len(lk8.data["Muv"]) == 1

    assert len(lk8.data["Muv"][0]) != len(lk7.data["Muv"][0])
    chain({})

    lk7read = LikelihoodLuminosityFunction(datafile=dfile7, noisefile=nfile7, name="7")
    lk8read = LikelihoodLuminosityFunction(datafile=dfile8, noisefile=nfile8, name="8")
    chain = build_computation_chain([core7, core8], [lk7read, lk8read], setup=True)

    assert isinstance(lk7read.data, dict)
    assert len(lk7read.data["Muv"]) == 1
    assert len(lk8read.data["Muv"]) == 1

    assert len(lk8read.data["Muv"][0]) != len(lk7read.data["Muv"][0])
    chain({})


def test_create_mock():
    core = CoreLuminosityFunction(redshift=[7])
    lk = LikelihoodLuminosityFunction(simulate=True)
    with pytest.raises(ValueError):
        build_computation_chain(core, lk, setup=True)
