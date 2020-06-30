import os

import pytest

import numpy as np

import py21cmmc as mcmc
from py21cmfast import LightCone
from py21cmmc.cosmoHammer import CosmoHammerSampler
from py21cmmc.cosmoHammer import HDFStorageUtil
from py21cmmc.cosmoHammer import LikelihoodComputationChain
from py21cmmc.cosmoHammer import Params


@pytest.fixture(scope="module")
def core():
    return mcmc.CoreCoevalModule(
        redshift=9,
        user_params={"HII_DIM": 35, "DIM": 70},
        cache_mcmc=False,
        cache_init=False,
    )


@pytest.fixture(scope="module")
def likelihood_coeval(tmpdirec):
    return mcmc.Likelihood1DPowerCoeval(
        simulate=True, datafile=os.path.join(tmpdirec.strpath, "likelihood_coeval.npz")
    )


@pytest.fixture(scope="module")
def default_params():
    return {"HII_EFF_FACTOR": [30.0, 10.0, 50.0, 3.0], "ION_Tvir_MIN": [4.7, 2, 8, 0.1]}


def test_core_coeval_not_setup():
    core = mcmc.CoreCoevalModule(redshift=9)

    with pytest.raises(mcmc.NotAChain):
        core.chain


@pytest.fixture(scope="module")
def lc_core():
    return mcmc.CoreLightConeModule(
        redshift=7.0, max_redshift=8.0, user_params={"HII_DIM": 35, "DIM": 70}
    )


@pytest.fixture(scope="module")
def lc_core_ctx(lc_core):
    lk = mcmc.LikelihoodPlanck()

    chain = mcmc.build_computation_chain(lc_core, lk)

    assert lk._is_lightcone

    ctx = chain.createChainContext()
    lc_core.build_model_data(ctx)
    return ctx


def test_core_coeval_setup(core, likelihood_coeval):
    with pytest.raises(ValueError):  # If simulate is not true, and no datafile given...
        lk = mcmc.Likelihood1DPowerCoeval()
        mcmc.build_computation_chain(core, lk)

    chain = mcmc.build_computation_chain(core, likelihood_coeval)

    assert isinstance(core.chain, LikelihoodComputationChain)
    assert core.initial_conditions_seed is not None

    ctx = chain.build_model_data()

    assert ctx.get("xH_box") is not None
    assert ctx.get("brightness_temp") is not None

    assert not np.all(ctx.get("xH_box") == 0)
    assert not np.all(ctx.get("brightness_temp") == 0)


def test_mcmc(core, likelihood_coeval, default_params, tmpdirec):
    chain = mcmc.run_mcmc(
        core,
        likelihood_coeval,
        model_name="TEST",
        continue_sampling=False,
        datadir=tmpdirec.strpath,
        params=default_params,
        walkersRatio=2,
        burninIterations=0,
        sampleIterations=2,
        threadCount=1,
    )

    samples_from_chain = mcmc.get_samples(chain)
    samples_from_file = mcmc.get_samples(os.path.join(tmpdirec.strpath, "TEST"))

    # make sure reading from file is the same as the chain.
    assert samples_from_chain.iteration == samples_from_file.iteration
    assert np.all(samples_from_file.accepted == samples_from_chain.accepted)
    assert np.all(samples_from_file.get_chain() == samples_from_chain.get_chain())

    assert all(
        c in ["HII_EFF_FACTOR", "ION_Tvir_MIN"] for c in samples_from_chain.param_names
    )
    assert samples_from_chain.has_blobs
    assert samples_from_chain.param_guess["HII_EFF_FACTOR"] == 30.0
    assert samples_from_chain.param_guess["ION_Tvir_MIN"] == 4.7


def test_continue_burnin(core, likelihood_coeval, default_params, tmpdirec):
    with pytest.raises(AssertionError):  # needs to be sampled for at least 1 iteration!
        mcmc.run_mcmc(
            core,
            likelihood_coeval,
            model_name="TESTBURNIN",
            continue_sampling=False,
            datadir=tmpdirec.strpath,
            params=default_params,
            walkersRatio=2,
            burninIterations=1,
            sampleIterations=0,
            threadCount=1,
        )

    chain = mcmc.run_mcmc(
        core,
        likelihood_coeval,
        model_name="TESTBURNIN",
        continue_sampling=False,
        datadir=tmpdirec.strpath,
        params=default_params,
        walkersRatio=2,
        burninIterations=1,
        sampleIterations=1,
        threadCount=1,
    )

    # HAVE TO SAVE THE CHAIN TO MEMORY HERE, BECAUSE THE OBJECT ACCESS THE FILE ON EVERY CALL,
    # WHICH MEANS IT CONSTANTLY UPDATES
    chain_b_chain = mcmc.get_samples(chain, burnin=True).get_chain()
    chain_s_chain = mcmc.get_samples(chain).get_chain()

    chain2 = mcmc.run_mcmc(
        core,
        likelihood_coeval,
        model_name="TESTBURNIN",
        continue_sampling=True,
        datadir=tmpdirec.strpath,
        params=default_params,
        walkersRatio=2,
        burninIterations=2,
        sampleIterations=1,
        threadCount=1,
    )

    burnin2 = mcmc.get_samples(chain2, burnin=True)
    chain2_b_chain = burnin2.get_chain()
    chain2_s_chain = mcmc.get_samples(chain).get_chain()

    assert likelihood_coeval._simulate is False  # because we're continuing sampling
    assert burnin2.iteration == 2
    assert np.all(
        chain2_b_chain[:1] == chain_b_chain
    )  # first 5 iteration should be unchanged

    # The actual samples *should* have been deleted, because they have different burnin times.
    assert not np.all(chain_s_chain == chain2_s_chain)

    chain3 = mcmc.run_mcmc(
        core,
        likelihood_coeval,
        model_name="TESTBURNIN",
        continue_sampling=True,
        datadir=tmpdirec.strpath,
        params=default_params,
        walkersRatio=2,
        burninIterations=2,
        sampleIterations=2,
        threadCount=1,
    )

    samples3 = chain3.samples
    assert samples3.iteration == 2

    chain3_b_chain = mcmc.get_samples(chain3, burnin=True).get_chain()
    assert np.all(chain3_b_chain == chain2_b_chain)

    chain3_s_chain = mcmc.get_samples(chain3).get_chain()
    assert np.all(chain2_s_chain == chain3_s_chain[:1])

    with pytest.raises(
        ValueError
    ):  # don't run if we already have all samples, and let the user know!
        mcmc.run_mcmc(
            core,
            likelihood_coeval,
            model_name="TESTBURNIN",
            continue_sampling=True,
            datadir=tmpdirec.strpath,
            params=default_params,
            walkersRatio=2,
            burninIterations=2,
            sampleIterations=2,
            threadCount=1,
        )

    # We set the _simulate back to True to have no side-effects.
    likelihood_coeval._simulate = True


def test_bad_continuation(core, likelihood_coeval, default_params, tmpdirec):
    "check if trying to continue a chain that isn't compatible with previous chain raises an error"

    mcmc.run_mcmc(
        core,
        likelihood_coeval,
        model_name="TESTBURNIN",
        continue_sampling=False,
        datadir=tmpdirec.strpath,
        params=default_params,
        walkersRatio=2,
        burninIterations=0,
        sampleIterations=1,
        threadCount=1,
    )

    with pytest.raises(RuntimeError):
        # core with different redshift
        core = mcmc.CoreCoevalModule(
            redshift=8,
            user_params={"HII_DIM": 35, "DIM": 70},
            cache_mcmc=False,
            cache_init=False,
        )
        mcmc.run_mcmc(
            core,
            likelihood_coeval,
            model_name="TESTBURNIN",
            continue_sampling=True,
            datadir=tmpdirec.strpath,
            params=default_params,
            walkersRatio=2,
            burninIterations=0,
            sampleIterations=2,
            threadCount=1,
        )


def test_init_pos_generator_good(core, likelihood_coeval, tmpdirec):
    params = Params(
        ("HII_EFF_FACTOR", [30.0, 10.0, 50.0, 10.0]), ("ION_Tvir_MIN", [4.7, 2, 8, 2])
    )

    chain = mcmc.build_computation_chain(core, likelihood_coeval, params=params)

    sampler = CosmoHammerSampler(
        continue_sampling=False,
        likelihoodComputationChain=chain,
        storageUtil=HDFStorageUtil(os.path.join(tmpdirec.strpath, "POSGENERATORTEST")),
        filePrefix=os.path.join(tmpdirec.strpath, "POSGENERATORTEST"),
        reuseBurnin=False,
        burninIterations=0,
        sampleIterations=1,
        walkersRatio=50,
    )

    pos = sampler.createInitPos()

    assert all(chain.isValid(p) for p in pos)


def test_init_pos_generator_bad(core, likelihood_coeval, tmpdirec):
    params = Params(
        ("HII_EFF_FACTOR", [30.0, 29.0, 31.0, 100.0]),
        ("ION_Tvir_MIN", [4.7, 4.6, 4.8, 20]),
    )

    chain = mcmc.build_computation_chain(core, likelihood_coeval, params=params)

    sampler = CosmoHammerSampler(
        continue_sampling=False,
        likelihoodComputationChain=chain,
        storageUtil=HDFStorageUtil(os.path.join(tmpdirec.strpath, "POSGENERATORTEST")),
        filePrefix=os.path.join(tmpdirec.strpath, "POSGENERATORTEST"),
        reuseBurnin=False,
        burninIterations=0,
        sampleIterations=1,
        walkersRatio=50,
    )

    with pytest.raises(ValueError):
        sampler.createInitPos()


def test_lightcone_core(lc_core, lc_core_ctx):
    lk = mcmc.Likelihood1DPowerLightcone(simulate=True)

    mcmc.build_computation_chain(lc_core, lk, setup=False)
    lk.setup()

    assert lc_core_ctx.contains("lightcone")
    assert isinstance(lc_core_ctx.get("lightcone"), LightCone)

    model = lk.reduce_data(lc_core_ctx)
    lk.store(model, lc_core_ctx.getData())

    assert all(k + "_0" in lc_core_ctx.getData() for k in model[0])


def test_planck(lc_core, lc_core_ctx):
    lk = mcmc.LikelihoodPlanck()

    with pytest.raises(mcmc.NotAChain):
        assert lk._is_lightcone

    mcmc.build_computation_chain(lc_core, lk, setup=False)
    lk.setup()

    model = lk.reduce_data(lc_core_ctx)
    assert "tau" in model


def test_neutral_fraction(lc_core, lc_core_ctx):
    with pytest.raises(ValueError):
        mcmc.build_computation_chain(
            mcmc.CoreLuminosityFunction(
                redshift=7, sigma=1
            ),  # Bad core for neutral fraction
            mcmc.LikelihoodNeutralFraction(),
        )

    lk = mcmc.LikelihoodNeutralFraction()

    mcmc.build_computation_chain(lc_core, lk, setup=False)
    lk.setup()

    assert lc_core in lk.lightcone_modules
    assert len(lk.coeval_modules) == 0
    assert lk._require_spline

    model = lk.reduce_data(lc_core_ctx)

    assert "xHI" in model


def test_greig(lc_core, lc_core_ctx):
    lk = mcmc.LikelihoodGreig()

    mcmc.build_computation_chain(lc_core, lk, setup=False)
    lk.setup()

    assert lc_core in lk.lightcone_modules
    assert len(lk.coeval_modules) == 0
    assert lk._require_spline

    model = lk.reduce_data(lc_core_ctx)

    assert "xHI" in model


def test_global_signal(lc_core, lc_core_ctx):
    lk = mcmc.LikelihoodGlobalSignal(simulate=True)

    mcmc.build_computation_chain(lc_core, lk, setup=False)
    lk.setup()

    model = lk.reduce_data(lc_core_ctx)

    assert "frequencies" in model


def test_edges_timing_only(lc_core, lc_core_ctx):
    lk = mcmc.LikelihoodEDGES(simulate=True)

    mcmc.build_computation_chain(lc_core, lk, setup=False)
    lk.setup()

    model = lk.reduce_data(lc_core_ctx)

    assert "freq_tb_min" in model


def test_edges(lc_core, lc_core_ctx):
    lk = mcmc.LikelihoodEDGES(simulate=True, use_width=True)

    mcmc.build_computation_chain(lc_core, lk, setup=False)
    lk.setup()

    model = lk.reduce_data(lc_core_ctx)

    assert "freq_tb_min" in model
    assert "fwhm" in model


def test_load_chain(core, likelihood_coeval, default_params, tmpdirec):
    mcmc.run_mcmc(
        core,
        likelihood_coeval,
        model_name="TESTLOADCHAIN",
        continue_sampling=False,
        datadir=tmpdirec.strpath,
        params=default_params,
        walkersRatio=2,
        burninIterations=0,
        sampleIterations=1,
        threadCount=1,
    )

    lcc = mcmc.load_primitive_chain("TESTLOADCHAIN", direc=tmpdirec.strpath)

    assert lcc.getCoreModules()[0].redshift == core.redshift


def test_wrong_ctx_variable():
    core = mcmc.CoreCoevalModule(
        redshift=6,
        user_params={"HII_DIM": 35, "BOX_LEN": 70},
        ctx_variables=("bad_key", "good key"),
    )
    lk = mcmc.Likelihood1DPowerCoeval(use_data=False)

    chain = mcmc.build_computation_chain(core, lk, setup=False)

    with pytest.raises(ValueError):
        chain.build_model_data()


def test_wrong_lf_paring():
    redshifts = [6, 7, 8, 10]
    with pytest.raises(ValueError):
        cores = [
            mcmc.CoreLuminosityFunction(redshift=z, sigma=0, name="lf")
            for z in redshifts
        ]
        lks = [mcmc.LikelihoodLuminosityFunction(name="lf") for z in redshifts]
        mcmc.build_computation_chain(cores, lks, setup=True)

    cores = [
        mcmc.CoreLuminosityFunction(redshift=z, sigma=0, name="lfz%d" % z)
        for z in redshifts
    ]
    lks = [mcmc.LikelihoodLuminosityFunction(name="lfz%d" % z) for z in redshifts]
    mcmc.build_computation_chain(cores, lks, setup=True)


def test_wrong_lf_redshift():
    with pytest.raises(ValueError):
        cores = [
            mcmc.CoreLuminosityFunction(redshift=9, sigma=0, name="lf"),
        ]
        lks = [
            mcmc.LikelihoodLuminosityFunction(name="lf"),
        ]
        mcmc.build_computation_chain(cores, lks, setup=True)


def test_planckpowerspectra(default_params, tmpdirec):
    global_params = {'Z_HEAT_MAX': 20.0, 'ZPRIME_STEP_FACTOR': 1.1}
    user_params   = {'HII_DIM':128, 'BOX_LEN':250.0}
    flag_options = {'USE_MASS_DEPENDENT_ZETA': True, 'INHOMO_RECO': False, 'PHOTON_CONS': False}
    mcmc.run_mcmc(
        [
            mcmc.CoreLightConeModule(redshift=5.0, user_params=user_params, global_params=global_params, flag_options=flag_options),
            mcmc.CoreCMB(z_extrap_max=global_params['Z_HEAT_MAX']+1),
        ],
        [
            mcmc.LikelihoodPlanckPowerSpectra(name_lkl="Planck_lowl_EE"),
            mcmc.LikelihoodPlanck(),
        ],
        model_name="TESTPLANCK",
        continue_sampling=False,
        datadir=tmpdirec.strpath,
        params=dict(             # Parameter dict as described above.
            F_STAR10 = [-1.3, -3, 0, 1.0],
            ALPHA_STAR = [0.5, -0.5, 1.0, 1.0],
            F_ESC10 = [-1, -3, 0, 1.0],
            ALPHA_ESC = [-0.5, -1.0, 0.5, 1.0],
            M_TURN = [8.69897, 8, 10, 1.0],
            t_STAR = [0.5, 0.01, 1, 0.5],
        ),
        walkersRatio=2,
        burninIterations=0,
        sampleIterations=1,
        threadCount=1,
    )
