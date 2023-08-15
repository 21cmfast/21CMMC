def test_emulator_runs():
    import numpy as np
    from py21cmemu import Emulator

    emu = Emulator()
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
