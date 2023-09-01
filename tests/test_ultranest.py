import pytest

import numpy as np
from py21cmfast.utils import ParameterError

import py21cmmc as mcmc
from py21cmmc.cosmoHammer import Params
from py21cmmc.likelihood import LikelihoodBase


@pytest.fixture(scope="module")
def astro_params():
    ap = {
        "L_X": [40.0, 38.0, 42.0, 0.05],
        "NU_X_THRESH": [500.0, 200.0, 1500.0, 20.0],
    }
    return Params(*[(k, v) for k, v in ap.items()])


@pytest.fixture(scope="module")
def prior(astro_params):
    class PriorFunction(LikelihoodBase):
        def __init__(self, arg_names, f):
            super().__init__()
            self.arg_names = arg_names
            self.f = f

        def computeLikelihood(self, arg_values):
            return self.f(arg_values)

        def reduce_data(self, ctx):
            params = ctx.getParams()
            arg_values = [v for k, v in params.items() if k in self.arg_names]
            return arg_values

    def f(x):
        if x[-1] <= astro_params[-1][0]:
            return 0.0
        else:
            raise ParameterError()

    return PriorFunction(astro_params.keys, f)


def test_ultranest_samples(astro_params, prior):
    model_name = "Prior"
    mcmc_options = {
        "min_num_live_points": 10,
        "max_ncalls": 10000,
    }

    mcmc.run_mcmc(
        core_modules=[],
        likelihood_modules=[prior],
        model_name=model_name,
        params=astro_params,
        use_ultranest=True,
        continue_sampling=False,
        **mcmc_options,
    )


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
        "ndraw_min": 10,
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

    assert result.shape[1] == 6
