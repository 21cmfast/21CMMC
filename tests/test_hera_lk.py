from py21cmmc import mcmc, core, analyse, likelihood, build_computation_chain
from py21cmmc import Likelihood1DPowerLightconeUpper, LikelihoodPlanck, run_mcmc

def test_hera_lk():
    
    model = [{'k':np.random.rand(10), 'delta':10**np.random.rand(37*2).reshape(2,37)}]
    lk = Likelihood1DPowerLightconeUpper.from_builtin_data('HERA_H1C_IDR3')
    lk.computeLikelihood(model)
    # TODO Add test to check that HERA lk works with 21cmFAST once it's implemented
    
    c21cmemu = core.Core21cmEMU()
    chain = build_computation_chain(
        [c21cmemu],
        [likelihood.Likelihood1DPowerLightconeUpper.from_builtin_data("HERA_H1C_IDR3"),
        ],
        setup = True
    )
    chain({})

