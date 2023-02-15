from py21cmmc import mcmc, core, analyse, likelihood
from py21cmmc.emulator import p21cmEMU
from py21cmmc import Likelihood1DPowerLightconeUpper, LikelihoodPlanck, run_mcmc
import numpy as np

#import inspect
#inspect.getmembers(likelihood.Likelihood1DPowerLightconeUpper, lambda a:not(inspect.isroutine(a)))
c21cmemu = core.Core21cmEMU(emu_path = 'py21cmmc/21cmEMU')
chain = mcmc.run_mcmc(
    [c21cmemu],
    [likelihood.Likelihood1DPowerLightconeUpper.from_builtin_data("HERA_H1C_IDR3_1"), 
     likelihood.LikelihoodPlanck(),
     likelihood.LikelihoodLuminosityFunction(z = 6), 
     likelihood.LikelihoodLuminosityFunction(z = 7), 
     likelihood.LikelihoodLuminosityFunction(z = 8), 
     likelihood.LikelihoodLuminosityFunction(z = 10),
     #likelihood.LikelihoodGlobalSignal(emulate = True) # Not working
     likelihood.LikelihoodNeutralFraction()
    ],
    datadir='mcmc_test',          # Directory for all outputs
    model_name='test',   # Filename of main chain output
    params=dict(F_STAR10 = [-2, -3, 0, 1], ALPHA_STAR = [0.5, -0.5, 1, 1], F_ESC10 = [-2, -3, 0, 1], ALPHA_ESC = [-0.7, -1, 0.5, 1], M_TURN = [9, 8,10,1], 
               t_STAR = [0.05, 0.01, 1, 1], L_X = [40, 38, 42, 1], NU_X_THRESH = [1000, 100, 1500, 1], X_RAY_SPEC_INDEX = [0.1, -1,3,1]
        ),
    walkersRatio=4,          # The number of walkers will be walkersRatio*nparams
    burninIterations=0,      # Number of iterations to save as burnin. Recommended to leave as zero.
    sampleIterations=2,    # Number of iterations to sample, per walker.
    threadCount=1,           # Number of processes to use in MCMC (best as a factor of walkersRatio)
    continue_sampling=False  # Whether to contine sampling from previous run *up to* sampleIterations.
)
