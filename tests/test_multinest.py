import py21cmmc as mcmc


def test_multinest():

    model_name = "LuminosityLikelihood"
    redshifts = [6, 7, 8, 10]
    F_STAR10 = [-1.3, -3, 0, 1.0]
    ALPHA_STAR = [0.5, -0.5, 1.0, 1.0]
    M_TURN = [8.69897, 8, 10, 1.0]
    t_STAR = [0.5, 0.01, 1, 0.3]

    mcmc_options = {
        "n_live_points": 10,
        "max_iter": 10,
    }
    mcmc.run_mcmc(
        [
            mcmc.CoreLuminosityFunction(redshift=z, sigma=0, name="lfz%d" % z)
            for z in redshifts
        ],
        [mcmc.LikelihoodLuminosityFunction(name="lfz%d" % z) for z in redshifts],
        model_name=model_name,
        params=dict(
            F_STAR10=F_STAR10, ALPHA_STAR=ALPHA_STAR, M_TURN=M_TURN, t_STAR=t_STAR,
        ),
        use_multinest=True,
        **mcmc_options,
    )

    import pymultinest

    nest = pymultinest.Analyzer(4, outputfiles_basename="./MultiNest/%s" % model_name)
    data = nest.get_data()

    assert data.shape[1] == 6
