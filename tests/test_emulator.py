"""Tests of the 21cmEMU emulator integration."""


def test_emulator_runs():
    import numpy as np
    from py21cmemu import Emulator

    # This codebase currently only supports the 'acg' (v1, 9-parameter)
    # emulator model; the default is now 'mcg' (v3, 11 parameters).
    emu = Emulator(emulator="acg")
    emu.predict(np.random.rand(9))


def test_emu_compat():
    import py21cmmc as mcmc
    from py21cmmc import core, likelihood

    lk = likelihood.Likelihood1DPowerLightcone.from_builtin_data("HERA_H1C_IDR3")

    c21cmemu = core.Core21cmEMU()
    chain = mcmc.build_computation_chain(
        [c21cmemu],
        [lk],
        setup=True,
    )
    chain({})
