import pytest

import numpy as np
from py21cmfast._utils import ParameterError

import py21cmmc as mcmc
from py21cmmc.cosmoHammer import Params
from py21cmmc.likelihood import LikelihoodBase


def test_ultranest_21cmemu():
    model_name = "LuminosityLikelihood"
    redshifts = [6, 7, 8, 10]
    F_STAR10 = [-1.3, -3, 0, 1.0]
    ALPHA_STAR = [0.5, -0.5, 1.0, 1.0]
    M_TURN = [8.69897, 8, 10, 1.0]
    t_STAR = [0.5, 0.01, 1, 0.3]
    L_X = [40, 38, 42, 1]
    NU_X_THRESH = [1000, 100, 1500, 1]
    X_RAY_SPEC_INDEX = [0.1, -1, 3, 1]
    F_ESC10 = [-1, -3, 0, 1.0]
    ALPHA_ESC = [-0.5, -1.0, 0.5, 1.0]

    mcmc_options = {
        "min_num_live_points": 2,
        "max_ncalls": 1000,
        "vectorized": True,
        "ndraw_min": 5,
        "frac_remain": 0.2,
        "Lepsilon": 0.1,
    }
    sampler, result = mcmc.run_mcmc(
        [mcmc.Core21cmEMU()],
        [mcmc.LikelihoodLuminosityFunction(z=z) for z in redshifts],
        model_name=model_name,
        params={
            "F_STAR10": F_STAR10,
            "ALPHA_STAR": ALPHA_STAR,
            "M_TURN": M_TURN,
            "t_STAR": t_STAR,
            "L_X": L_X,
            "NU_X_THRESH": NU_X_THRESH,
            "X_RAY_SPEC_INDEX": X_RAY_SPEC_INDEX,
            "F_ESC10": F_ESC10,
            "ALPHA_ESC": ALPHA_ESC,
        },
        use_ultranest=True,
        continue_sampling=False,
        **mcmc_options,
    )
    assert result["samples"].shape[1] == 9


def test_ultranest_21cmfast():
    model_name = "LuminosityLikelihood"
    redshifts = [7, 8, 10]
    F_STAR10 = [-1.3, -3, 0, 1.0]
    ALPHA_STAR = [0.5, -0.5, 1.0, 1.0]
    M_TURN = [8.69897, 8, 10, 1.0]
    t_STAR = [0.5, 0.01, 1, 0.3]

    mcmc_options = {
        "min_num_live_points": 2,
        "max_ncalls": 500,
        "vectorized": True,
        "ndraw_min": 5,
        "frac_remain": 0.2,
        "Lepsilon": 0.1,
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
        use_ultranest=True,
        continue_sampling=False,
        **mcmc_options,
    )
    assert result["samples"].shape[1] == 4
