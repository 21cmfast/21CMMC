import pytest

import numpy as np
from py21cmfast._utils import ParameterError

import py21cmmc as mcmc
from py21cmmc.cosmoHammer import Params
from py21cmmc.likelihood import LikelihoodBase


def test_ultranest():
    model_name = "LuminosityLikelihood"
    redshifts = [6, 7, 8, 10]
    F_STAR10 = [-1.3, -3, 0, 1.0]
    ALPHA_STAR = [0.5, -0.5, 1.0, 1.0]
    M_TURN = [8.69897, 8, 10, 1.0]
    t_STAR = [0.5, 0.01, 1, 0.3]

    mcmc_options = {
        "min_num_live_points": 10,
        "max_ncalls": 10000,
        "vectorized": True,
        "ndraw_min": 5,
    }
    sampler, result = mcmc.run_mcmc(
        [
            mcmc.CoreLuminosityFunction(redshift=z, sigma=0, name="lfz%d" % z)
            for z in redshifts
        ],
        [mcmc.LikelihoodLuminosityFunction(name="lfz%d" % z) for z in redshifts],
        model_name=model_name,
        params={
            "F_STAR10": F_STAR10,
            "ALPHA_STAR": ALPHA_STAR,
            "M_TURN": M_TURN,
            "t_STAR": t_STAR,
        },
        use_multinest=True,
        continue_sampling=False,
        **mcmc_options,
    )
    sampler.print_results()
    assert result.shape[1] == 6
