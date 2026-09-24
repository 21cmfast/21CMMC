"""Tests of the zeus-mcmc sampler integration."""

import numpy as np

import py21cmmc as p21mc
from py21cmmc import mcmc


def test_zeus(tmp_path):
    core = p21mc.CoreCoevalModule(
        redshift=[7, 8, 9],
        user_params={"HII_DIM": 50, "BOX_LEN": 125.0},
        regenerate=False,
        change_seed_every_iter=False,
    )

    datafiles = [tmp_path / f"simple_mcmc_data_{z}.npz" for z in core.redshift]

    likelihood = p21mc.Likelihood1DPowerCoeval(
        datafile=datafiles,
        noisefile=None,
        logk=False,
        min_k=0.1,
        max_k=1.0,
        simulate=True,
    )

    chain = mcmc.run_mcmc(
        core,
        likelihood,
        datadir=str(tmp_path),
        model_name="SimpleTest",
        params={
            "HII_EFF_FACTOR": [30.0, 10.0, 50.0, 3.0],
            "ION_Tvir_MIN": [4.7, 4, 6, 0.1],
        },
        reuse_burnin=False,
        continue_sampling=False,
        use_zeus=True,
        nsteps=1,
    )

    fchain = chain.get_chain(flat=True)
    H2, TV = fchain.T
    print(np.mean(H2), np.std(H2))
    print(np.mean(TV), np.std(TV))