import numpy as np

from py21cmmc import (
    Likelihood1DPowerLightconeUpper,
    LikelihoodPlanck,
    analyse,
    build_computation_chain,
    core,
    likelihood,
    mcmc,
    run_mcmc,
)


def test_hera_lk():

    lk = Likelihood1DPowerLightconeUpper.from_builtin_data("HERA_H1C_IDR3")

    c21cmemu = core.Core21cmEMU()
    chain = build_computation_chain(
        [c21cmemu],
        [
            lk,
        ],
        setup=True,
    )
    chain({})
