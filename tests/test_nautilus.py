import numpy as np

import py21cmmc as mcmc


def test_nautilus_21cmemu():
    """Run a very short Nautilus sampling using the 21cmEMU core.

    This mirrors ``test_ultranest_21cmemu`` but exercises the Nautilus
    sampler integration in :func:`py21cmmc.mcmc.run_mcmc` (``use_nautilus``).
    The run is bounded via ``n_like_max`` so the test stays fast -- we are
    only checking that the Nautilus code path runs end-to-end without
    error and returns a posterior with the right shape, not that it has
    converged.
    """
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
        "n_live": 50,
        "n_like_max": 200,
        "vectorized": True,
        "verbose": False,
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
        use_nautilus=True,
        continue_sampling=False,
        **mcmc_options,
    )
    points, log_w, log_l = result
    assert points.shape[1] == 9
    assert np.all(np.isfinite(log_l))
