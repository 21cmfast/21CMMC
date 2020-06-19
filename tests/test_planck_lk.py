import os
import shutil

import pytest

import numpy as np

from py21cmmc import mcmc
from py21cmmc import LikelihoodPlanckPowerSpectra, CoreCMB


def test_planck():
    chain = mcmc.run_mcmc(
    CoreCMB(),
    LikelihoodPlanckPowerSpectra(datafolder = ",/baseline/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik/",name_lkl = 'Planck_lowl_EE'),
    params=dict(             # Parameter dict as described above.
        F_STAR10 = [-1.3, -3, 0, 1.0],
        ALPHA_STAR = [0.5, -0.5, 1.0, 1.0],
        F_ESC10 = [-1, -3, 0, 1.0],
        ALPHA_ESC = [-0.5, -1.0, 0.5, 1.0],
        M_TURN = [8.69897, 8, 10, 1.0],
        t_STAR = [0.5, 0.01, 1, 0.5],
    ),
    model_name='TEST',
    walkersRatio=1,
    burninIterations=0,
    sampleIterations=1,
    threadCount=1,
    continue_sampling=False,
)
