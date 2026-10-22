import pytest

import numpy as np
import os
import shutil

from py21cmmc import (
    Core21cmEMU,
    CoreLuminosityFunction,
    LikelihoodLuminosityFunction,
    build_computation_chain,
)
from py21cmmc.cosmoHammer import Params


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


def test_interpolate_uvlf():
    zs = np.array([6.0, 8.0, 10.0])
    uvlfs = np.array([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]])  # (n_z, n_Muv)
    lf = Core21cmEMU.interpolate_uvlf(uvlfs, zs, [6, 7, 9.5, 10])
    assert np.allclose(lf, [[1, 2], [2, 3], [6, 7], [7, 8]])

    # vectorized over parameters, and the redshifts need not be sorted
    lf = Core21cmEMU.interpolate_uvlf(
        np.stack([uvlfs, 2 * uvlfs])[:, ::-1], zs[::-1], 7
    )
    assert np.allclose(lf, [[[2, 3]], [[4, 6]]])

    for z in [5.9, 10.1]:
        with pytest.raises(ValueError):
            Core21cmEMU.interpolate_uvlf(uvlfs, zs, z)


V3_PARAMS = {
    "F_STAR10": -1.3,
    "ALPHA_STAR": 0.5,
    "t_STAR": 0.5,
    "F_ESC10": -1.0,
    "ALPHA_ESC": -0.5,
    "F_STAR7_MINI": -2.5,
    "F_ESC7_MINI": -2.0,
    "L_X": 40.0,
    "L_X_MINI": 40.0,
    "NU_X_THRESH": 500.0,
    "SIGMA_8": 0.8,
}


def emulated_uvlfs(**lk_kwargs):
    """Emulated UV LFs (log10) at the emulated redshifts, and lk.reduce_data for lk_kwargs."""
    lk = LikelihoodLuminosityFunction(**lk_kwargs)
    chain = build_computation_chain([Core21cmEMU()], [lk], setup=True)
    params = Params(*[(k, np.array([v, v + 0.1])) for k, v in V3_PARAMS.items()])
    ctx = chain.build_model_data(params)
    uvlfs, zs = ctx.get("UVLFs"), ctx.get("UVLF_redshifts")
    return uvlfs.value, list(getattr(zs, "value", zs)), lk.reduce_data(ctx)["lfunc"]


def test_emulator_uvlf_interpolation():
    # at an emulated redshift, interpolation returns the emulated LF
    uvlfs, zs, lfunc = emulated_uvlfs(z=10)
    assert np.allclose(lfunc[:, 0], uvlfs[:, zs.index(10)], equal_nan=True)
    _, _, lfunc_nearest = emulated_uvlfs(z=10, interpolate_z=False)
    assert np.allclose(lfunc, lfunc_nearest, equal_nan=True)

    # between emulated redshifts (10 and 12 for 21cmEMUv3), log10 phi is linear in z
    _, _, lfunc = emulated_uvlfs(z=11, telescope="JWST")
    expected = 0.5 * (uvlfs[:, zs.index(10)] + uvlfs[:, zs.index(12)])
    assert np.allclose(lfunc[:, 0], expected, equal_nan=True)


def test_uvlf_redshift_outside_emulator():
    with pytest.raises(ValueError):
        build_computation_chain(
            [Core21cmEMU(emulator_name="acg")],
            [LikelihoodLuminosityFunction(z=11, telescope="JWST")],
            setup=True,
        )
