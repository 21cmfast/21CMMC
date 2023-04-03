"""High-level functions for running MCMC chains."""
import logging
import numpy as np
import scipy.stats as stats
from cmath import log
from concurrent.futures import ProcessPoolExecutor
from os import mkdir, path
from py21cmfast import yaml
from py21cmfast._utils import ParameterError

from .cosmoHammer import (
    CosmoHammerSampler,
    HDFStorageUtil,
    LikelihoodComputationChain,
    Params,
)

logger = logging.getLogger("21cmFAST")


def build_computation_chain(core_modules, likelihood_modules, params=None, setup=True):
    """
    Build a likelihood computation chain from core and likelihood modules.

    Parameters
    ----------
    core_modules : list
        A list of objects which define the necessary methods to be core modules
        (see :mod:`~py21cmmc.core`).
    likelihood_modules : list
        A list of objects which define the necessary methods to be likelihood modules (see
        :mod:`~py21cmmc.likelihood`)
    params : :class:`~py21cmmc.cosmoHammer.Params`, optional
        If provided, parameters which will be sampled by the chain.

    Returns
    -------
    chain : :class:`~py21cmmc.cosmoHammer.LikelihoodComputationChain`
    """
    if not hasattr(core_modules, "__len__"):
        core_modules = [core_modules]

    if not hasattr(likelihood_modules, "__len__"):
        likelihood_modules = [likelihood_modules]

    chain = LikelihoodComputationChain(params)

    for cm in core_modules:
        chain.addCoreModule(cm)

    for lk in likelihood_modules:
        chain.addLikelihoodModule(lk)

    if setup:
        chain.setup()
    return chain


def run_mcmc(
    core_modules,
    likelihood_modules,
    params,
    datadir=".",
    model_name="21CMMC",
    continue_sampling=True,
    reuse_burnin=True,
    log_level_21CMMC=None,
    sampler_cls=CosmoHammerSampler,
    use_multinest=False,
    use_zeus=False,
    use_ultranest=False,
    **mcmc_options,
) -> CosmoHammerSampler:
    r"""Run an MCMC chain.

    Parameters
    ----------
    core_modules : list
        A list of objects which define the necessary methods to be core modules (see
        :mod:`~py21cmmc.core`).
    likelihood_modules : list
        A list of objects which define the necessary methods to be likelihood modules
        (see :mod:`~py21cmmc.likelihood`)
    params : dict
        Parameters which will be sampled by the chain. Each entry's key specifies the
        name of the parameter, and its value is an iterable `(val, min, max, width)`,
        with `val` the initial guess, `min` and `max` the hard boundaries on the
        parameter's value, and `width` determining the size of the initial ball of
        walker positions for the parameter.
    datadir : str, optional
        Directory to which MCMC info will be written (eg. logs and chain files)
    model_name : str, optional
        Name of the model, which determines filenames of outputs.
    continue_sampling : bool, optional
        If an output chain file can be found that matches these inputs, sampling can be
        continued from its last iteration, up to the number of iterations specified. If
        set to `False`, any output file which matches these parameters will have its
        samples over-written.
    reuse_burnin : bool, optional
        If a pre-computed chain file is found, and `continue_sampling=False`, setting
        `reuse_burnin` will salvage the burnin part of the chain for re-use, but
        re-compute the samples themselves.
    log_level_21CMMC : (int or str, optional)
        The logging level of the cosmoHammer log file.
    use_multinest : bool, optional
        If true, use the MultiNest sampler instead.
    use_zeus : bool, optional
        If true, use the zeus sampler instead.
    use_ultranest : bool, optional
        If true, use the UltraNest sampler instead.

    Other Parameters
    ----------------
    \*\*mcmc_options:
        All other parameters are passed directly to
        :class:`~py21cmmc.cosmoHammer.CosmoHammerSampler`. These include important
        options such as ``walkersRatio`` (the number of walkers is
        ``walkersRatio*nparams``), ``sampleIterations``, ``burninIterations``, ``pool``,
        ``log_level_stream`` and ``threadCount``.
        If use_multinest, parameters required by MultiNest as shown below should be
        provided here.
    n_live_points : int, optional
        number of live points
    importance_nested_sampling : bool, optional
        If True, Multinest will use Importance Nested Sampling (INS).
    sampling_efficiency : float, optional
        defines the sampling efficiency. 0.8 and 0.3 are recommended for parameter
        estimation & evidence evalutation
    evidence_tolerance : float, optional
        A value of 0.5 should give good enough accuracy.
    max_iter : int, optional
        maximum number of iterations. 0 is unlimited.
    multimodal : bool, optional
        whether or not to detect multi mode
    write_output : bool, optional
        write output files? This is required for analysis.

        If use_zeus, parameters required by zeus as shown below should be
        provided here.
    nsteps : int
        number of steps per iteration (Default is 100)
    ndim : int
        number of dimensions to sample over (Default is number of supplied parameters)
    nwalkers : int
        number of walkers (Default is 2*ndim)
    tolerance : float, optional
        Tuning optimization tolerance (Default is 0.05).
    patience : int, optional
        Number of tuning steps to wait to make sure that tuning is done (Default is 5).
    maxsteps : int, optional
        Number of maximum stepping-out steps (Default is 10^4).
    mu : float, optional
        Scale factor (Default value is 1.0), this will be tuned if tune=True.
    maxiter : int, optional
        Number of maximum Expansions/Contractions (Default is 10^4).
    pool : bool, optional
        External pool of workers to distribute workload to multiple CPUs (default is None).
    vectorize : bool, optional
        If true (default is False), logprob_fn receives not just one point but an array of points, and returns an array of log-probabilities.
    blobs_dtype : list, optional
        List containing names and dtypes of blobs metadata e.g. [("log_prior", float), ("mean", float)].
        It's useful when you want to save multiple species of metadata. Default is None.
    verbose : bool, optional
        If True (default) print log statements.
    check_walkers : bool, optional
        If True (default) then check that nwalkers >= 2*ndim and even.
    shuffle_ensemble : bool, optional
        If True (default) then shuffle the ensemble of walkers in every iteration before splitting it.
    light_mode : bool, optional
        If True (default is False) then no expansions are performed after the tuning phase.
        This can significantly reduce the number of log likelihood evaluations but works best in target distributions that are apprroximately Gaussian.

        If use_ultranest, parameters required by UltraNest as shown below should be
        provided here.
    log_dir: str
        where to store output files
    resume: 'resume', 'resume-similar', 'overwrite' or 'subfolder'
        If 'overwrite', overwrite previous data.
        If 'subfolder', create a fresh subdirectory in log_dir.
        If 'resume' or True, continue previous run if available.
        Only works when dimensionality, transform or likelihood are consistent.
        If 'resume-similar', continue previous run if available.
        Only works when dimensionality and transform are consistent.
        If a likelihood difference is detected, the existing likelihoods
        are updated until the live point order differs.
        Otherwise, behaves like resume.
    run_num: int or None
        If resume=='subfolder', this is the subfolder number.
        Automatically increments if set to None.
    num_test_samples: int
        test transform and likelihood with this number of
        random points for errors first. Useful to catch bugs.
    vectorized: bool
        If true, loglike and transform function can receive arrays
        of points.
    draw_multiple: bool
        If efficiency goes down, dynamically draw more points
        from the region between `ndraw_min` and `ndraw_max`.
        If set to False, few points are sampled at once.
    ndraw_min: int
        Minimum number of points to simultaneously propose.
        Increase this if your likelihood makes vectorization very cheap.
    ndraw_max: int
        Maximum number of points to simultaneously propose.
        Increase this if your likelihood makes vectorization very cheap.
        Memory allocation may be slow for extremely high values.
    num_bootstraps: int
        number of logZ estimators and MLFriends region
        bootstrap rounds.
    warmstart_max_tau: float
        Maximum disorder to accept when resume='resume-similar';
        Live points are reused as long as the live point order
        is below this normalised Kendall tau distance.
        Values from 0 (highly conservative) to 1 (extremely negligent).

    update_interval_volume_fraction: float
        Update region when the volume shrunk by this amount.
    log_interval: int
        Update stdout status line every log_interval iterations.
    show_status: bool
        Show integration progress as a status line.
        If no output desired, set to False.
    dlogz: float
        Target evidence uncertainty. This is the std
        between bootstrapped logz integrators.
    dKL: float
        Target posterior uncertainty. This is the
        Kullback-Leibler divergence in nat between bootstrapped integrators.
    frac_remain: float
        Integrate until this fraction of the integral is left in the remainder.
        Set to a low number (1e-2 ... 1e-5) to make sure peaks are discovered.
        Set to a higher number (0.5) if you know the posterior is simple.
    Lepsilon: float
        Terminate when live point likelihoods are all the same,
        within Lepsilon tolerance. Increase this when your likelihood
        function is inaccurate, to avoid unnecessary search.
    min_ess: int
        Target number of effective posterior samples.
    max_iters: int
        maximum number of integration iterations.
    max_ncalls: int
        stop after this many likelihood evaluations.
    max_num_improvement_loops: int
        run() tries to assess iteratively where more samples are needed.
        This number limits the number of improvement loops.
    min_num_live_points: int
        minimum number of live points throughout the run
    cluster_num_live_points: int
        require at least this many live points per detected cluster
    insertion_test_zscore_threshold: float
        z-score used as a threshold for the insertion order test.
        Set to infinity to disable.
    insertion_test_window: int
        Number of iterations after which the insertion order test is reset.
    region_class: `MLFriends` or `RobustEllipsoidRegion` or `SimpleRegion`
        Whether to use MLFriends+ellipsoidal+tellipsoidal region (better for multi-modal problems)
        or just ellipsoidal sampling (faster for high-dimensional, gaussian-like problems)
        or a axis-aligned ellipsoid (fastest, to be combined with slice sampling).

    Returns
    -------
    sampler : :class:`~py21cmmc.cosmoHammer.CosmoHammerSampler` instance.
        The sampler object, from which the chain itself may be accessed (via the
        ``samples`` attribute). If use_multinest, return multinest sampler.
        If use_zeus, return zeus sampler.
    """
    file_prefix = path.join(datadir, model_name)

    # check that only one sampler is specified
    if sum([use_multinest, use_ultranest, use_zeus]) > 1:
        raise ValueError("You cannot use more than one sampler at the time!")

    if use_multinest:
        n_live_points = mcmc_options.get("n_live_points", 100)
        importance_nested_sampling = mcmc_options.get(
            "importance_nested_sampling", True
        )
        sampling_efficiency = mcmc_options.get("sampling_efficiency", 0.8)
        evidence_tolerance = mcmc_options.get("evidence_tolerance", 0.5)
        max_iter = mcmc_options.get("max_iter", 50)
        multimodal = mcmc_options.get("multimodal", True)
        write_output = mcmc_options.get("write_output", True)
        datadir = datadir + "/MultiNest/"
        try:
            from pymultinest import run
        except ImportError:
            raise ImportError("You need to install pymultinest to use this function!")
    try:
        mkdir(datadir)
    except FileExistsError:
        pass

    if use_zeus:
        ndim = mcmc_options.get(
            "ndim", 2
        )  # Number of parameters/dimensions (e.g. m and c)
        nwalkers = mcmc_options.get(
            "nwalkers", 10
        )  # Number of walkers to use. It should be at least twice the number of dimensions.
        nsteps = mcmc_options.get("nsteps", 100)  # Number of steps/iterations.
        # set up parameters
        params = Params(*[(k, v) for k, v in params.items()])
        # initial positions of the walkers
        start = np.asarray(
            [
                stats.truncnorm.rvs(
                    (params[i][1] - params[i][0]) / params[i][-1],
                    (params[i][2] - params[i][0]) / params[i][-1],
                    loc=params[i][0],
                    scale=params[i][-1],
                    size=nwalkers,
                )
                for i in range(ndim)
            ]
        ).T
        tolerance = mcmc_options.get("tolerance", 0.05)
        patience = mcmc_options.get("patience", 5)
        maxsteps = mcmc_options.get("maxsteps", 1e4)
        mu = mcmc_options.get("mu", 1.0)
        maxiter = mcmc_options.get("maxiter", 1e4)
        pool = mcmc_options.get("pool", None)
        vectorize = mcmc_options.get("vectorize", False)
        blobs_dtype = mcmc_options.get("blobs_dtype", None)
        verbose = mcmc_options.get("vectorize", True)
        check_walkers = mcmc_options.get("check_walkers", True)
        shuffle_ensemble = mcmc_options.get("shuffle_ensemble", True)
        light_mode = mcmc_options.get("light_mode", False)
        print(start.shape)
        try:
            import zeus
        except ImportError:
            raise ImportError("You need to install zeus to use this function!")

    if use_ultranest:
        try:
            import ultranest
        except ImportError:
            raise ImportError("You need to install ultranest to use this function!")

        log_dir = mcmc_options.get("log_dir", None)
        resume = mcmc_options.get("resume", "subfolder")
        run_num = mcmc_options.get("run_num", None)
        num_test_samples = mcmc_options.get("num_test_samples", 2)
        vectorized = mcmc_options.get("vectorized", False)
        draw_multiple = mcmc_options.get("draw_multiple", True)
        ndraw_min = mcmc_options.get("ndraw_min", 128)
        ndraw_max = mcmc_options.get("ndraw_max", 65536)
        num_bootstraps = mcmc_options.get("num_bootstraps", 30)
        warmstart_max_tau = mcmc_options.get("warmstart_max_tau", -1)

        update_interval_volume_fraction = mcmc_options.get(
            "update_interval_volume_fraction", 0.8
        )
        log_interval = mcmc_options.get("log_interval", None)
        show_status = mcmc_options.get("show_status", True)
        dlogz = mcmc_options.get("dlogz", 0.5)
        dKL = mcmc_options.get("dKL", 0.5)
        frac_remain = mcmc_options.get("frac_remain", 0.01)
        Lepsilon = mcmc_options.get("Lepsilon", 0.001)
        min_ess = mcmc_options.get("min_ess", 400)
        max_iters = mcmc_options.get("max_iters", None)
        max_ncalls = mcmc_options.get("max_ncalls", None)
        max_num_improvement_loops = mcmc_options.get("max_num_improvement_loops", -1)
        min_num_live_points = mcmc_options.get("min_num_live_points", 400)
        cluster_num_live_points = mcmc_options.get("cluster_num_live_points", 40)
        insertion_test_zscore_threshold = mcmc_options.get(
            "insertion_test_zscore_threshold", 4
        )
        insertion_test_window = mcmc_options.get("insertion_test_window", 10)
        region_class = mcmc_options.get("region_class", ultranest.mlfriends.MLFriends)

    # Setup parameters.
    if not isinstance(params, Params):
        params = Params(*[(k, v) for k, v in params.items()])

    chain = build_computation_chain(
        core_modules, likelihood_modules, params, setup=False
    )

    if continue_sampling and not (use_multinest or use_zeus or use_ultranest):
        try:
            with open(file_prefix + ".LCC.yml", "r") as f:
                old_chain = yaml.load(f)

            if old_chain != chain:
                raise RuntimeError(
                    "Attempting to continue chain, but chain parameters are different. "
                    + "Check your parameters against {file_prefix}.LCC.yml".format(
                        file_prefix=file_prefix
                    )
                )

        except FileNotFoundError:
            pass

        # We need to ensure that simulate=False if trying to continue sampling.
        for lk in chain.getLikelihoodModules():
            if hasattr(lk, "_simulate") and lk._simulate:
                logger.warning(
                    f"Likelihood {lk} was defined to re-simulate data/noise, but this is incompatible with"
                    "`continue_sampling`. Setting simulate=False and continuing..."
                )
                lk._simulate = False

    # Write out the parameters *before* setup.
    # TODO: not sure if this is the best idea -- should it be after setup()?
    try:
        with open(file_prefix + ".LCC.yml", "w") as f:
            yaml.dump(chain, f)
    except Exception as e:
        logger.warning(
            "Attempt to write out YAML file containing LikelihoodComputationChain failed. "
            "Boldly continuing..."
        )
        print(e)

    chain.setup()

    # Set logging levels
    if log_level_21CMMC is not None:
        logging.getLogger("21CMMC").setLevel(log_level_21CMMC)

    if use_multinest:

        def likelihood(p, ndim, nparams):
            try:
                return chain.computeLikelihoods(
                    chain.build_model_data(
                        Params(*[(k, v) for k, v in zip(params.keys, p)])
                    )
                )
            except ParameterError:
                return -np.inf

        def prior(p, ndim, nparams):
            for i in range(ndim):
                p[i] = params[i][1] + p[i] * (params[i][2] - params[i][1])

        try:
            sampler = run(
                likelihood,
                prior,
                n_dims=len(params.keys),
                n_params=len(params.keys),
                n_live_points=n_live_points,
                resume=continue_sampling,
                write_output=write_output,
                outputfiles_basename=datadir + model_name,
                max_iter=max_iter,
                importance_nested_sampling=importance_nested_sampling,
                multimodal=multimodal,
                evidence_tolerance=evidence_tolerance,
                sampling_efficiency=sampling_efficiency,
                init_MPI=False,
            )
            return 1

        except OSError:  # pragma: nocover
            raise ImportError(
                "You also need to build MultiNest library. See https://johannesbuchner.github.io/PyMultiNest/install.html#id4 for more information."
            )

    elif use_zeus:

        def prior(p):
            for i, value in enumerate(p):
                if (value > params[i][2]) or (value < params[i][1]):
                    return p, True
            return p, False

        def likelihood(p):
            print(params)
            try:
                return chain.computeLikelihoods(
                    chain.build_model_data(
                        Params(*[(k, v) for k, v in zip(params.keys, p)])
                    )
                )
            except ParameterError:
                return -np.inf

        def posterior(p):
            # pass point into prior to check if in bounds
            p, inf = prior(p)
            if inf:
                return -np.inf
            # if in bounds, evaluate likelihood
            log_prob = likelihood(p)
            return log_prob

        sampler = zeus.EnsembleSampler(
            nwalkers,
            ndim,
            posterior,
            tolerance=tolerance,
            patience=patience,
            maxsteps=maxsteps,
            mu=mu,
            maxiter=maxiter,
            pool=pool,
            vectorize=vectorize,
            blobs_dtype=blobs_dtype,
            verbose=verbose,
            check_walkers=check_walkers,
            shuffle_ensemble=shuffle_ensemble,
            light_mode=light_mode,
        )  # Initialise the sampler
        sampler.run_mcmc(start, nsteps)  # Run sampling
        return sampler

    elif use_ultranest:

        def likelihood(p):
            if vectorized:
                return chain.computeLikelihoods(
                    chain.build_model_data(
                        Params(*[(k, v) for k, v in zip(params.keys, p.T)])
                    )
                )
            else:
                try:
                    return chain.computeLikelihoods(
                        chain.build_model_data(
                            Params(*[(k, v) for k, v in zip(params.keys, p)])
                        )
                    )
                except ParameterError:
                    return -np.inf

        def prior(p):
            t = np.empty(p.shape, dtype=p.dtype)
            for i in range(p.shape[-1]):
                if vectorized:
                    t[:, i] = params[i][1] + p[:, i] * (params[i][2] - params[i][1])

                else:
                    t[i] = params[i][1] + p[i] * (params[i][2] - params[i][1])
            return t

        sampler = ultranest.ReactiveNestedSampler(
            params.keys,
            loglike=likelihood,
            transform=prior,
            resume=resume,
            run_num=run_num,
            log_dir=log_dir,
            num_test_samples=num_test_samples,
            draw_multiple=draw_multiple,
            num_bootstraps=num_bootstraps,
            vectorized=vectorized,
            ndraw_min=ndraw_min,
            ndraw_max=ndraw_max,
            warmstart_max_tau=warmstart_max_tau,
        )
        result = sampler.run(
            update_interval_volume_fraction=update_interval_volume_fraction,
            log_interval=log_interval,
            show_status=show_status,
            dlogz=dlogz,
            dKL=dKL,
            frac_remain=frac_remain,
            Lepsilon=Lepsilon,
            min_ess=min_ess,
            max_iters=max_iters,
            max_ncalls=max_ncalls,
            max_num_improvement_loops=max_num_improvement_loops,
            min_num_live_points=min_num_live_points,
            cluster_num_live_points=cluster_num_live_points,
            insertion_test_window=insertion_test_window,
            insertion_test_zscore_threshold=insertion_test_zscore_threshold,
            region_class=region_class,
        )
        return sampler, result

    else:
        pool = mcmc_options.pop(
            "pool",
            ProcessPoolExecutor(max_workers=mcmc_options.get("threadCount", 1)),
        )
        sampler = sampler_cls(
            continue_sampling=continue_sampling,
            likelihoodComputationChain=chain,
            storageUtil=HDFStorageUtil(file_prefix),
            filePrefix=file_prefix,
            reuseBurnin=reuse_burnin,
            pool=pool,
            **mcmc_options,
        )

        # The sampler writes to file, so no need to save anything ourselves.
        sampler.startSampling()

        return sampler
