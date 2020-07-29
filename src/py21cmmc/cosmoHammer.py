"""A set of extensions to the basic ``CosmoHammer`` package."""
import emcee
import gc
import h5py
import logging
import numpy as np
import os
import time
import warnings
from cosmoHammer import CosmoHammerSampler as _CosmoHammerSampler
from cosmoHammer import getLogger
from cosmoHammer import util as _util
from cosmoHammer.ChainContext import ChainContext
from cosmoHammer.LikelihoodComputationChain import LikelihoodComputationChain as _Chain
from py21cmfast.wrapper import ParameterError

from py21cmmc.ensemble import EnsembleSampler

logger = getLogger()


class HDFStorage:
    """A HDF Storage utility, based on the HDFBackend from emcee v3.0.0."""

    def __init__(self, filename, name):
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDFBackend")
        self.filename = filename
        self.name = name

    @property
    def initialized(self):
        """Whether the file object has been initialized."""
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False

    def open(self, mode="r"):  # noqa
        """Open the backend file."""
        return h5py.File(self.filename, mode)

    def reset(self, nwalkers, params):
        """Clear the state of the chain and empty the backend.

        Parameters
        ----------
        nwalkers : int
            The size of the ensemble
        params : :class:`~py21cmmc.cosmoHammer.util.Params` instance
            The parameter input
        """
        if os.path.exists(self.filename):
            mode = "a"
        else:
            mode = "w"

        ndim = len(params.keys)

        with self.open(mode) as f:
            if self.name in f:
                del f[self.name]

            g = f.create_group(self.name)
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["has_blobs"] = False
            g.attrs["iteration"] = 0

            g.create_dataset(
                "guess",
                data=np.array(
                    [tuple(v[0] for v in params.values)],
                    dtype=[(k, np.float64) for k in params.keys],
                ),
            )
            g.create_dataset(
                "accepted", (0, nwalkers), maxshape=(None, nwalkers), dtype=np.int
            )
            g.create_dataset(
                "chain",
                (0, nwalkers, ndim),
                maxshape=(None, nwalkers, ndim),
                dtype=np.float64,
            )
            g.create_dataset(
                "log_prob", (0, nwalkers), maxshape=(None, nwalkers), dtype=np.float64
            )

            g.create_dataset(
                "trials",
                (0, nwalkers, ndim),
                maxshape=(None, nwalkers, ndim),
                dtype=np.float64,
            )
            g.create_dataset(
                "trial_log_prob",
                (0, nwalkers),
                maxshape=(None, nwalkers),
                dtype=np.float64,
            )

    @property
    def param_names(self):
        """The parameter names."""
        if not self.initialized:
            raise ValueError(
                "storage needs to be initialized to access parameter names"
            )

        with self.open() as f:
            return f[self.name]["guess"].dtype.names

    @property
    def param_guess(self):
        """An initial guess for the parameters."""
        if not self.initialized:
            raise ValueError(
                "storage needs to be initialized to access parameter guess"
            )

        with self.open() as f:
            return f[self.name]["guess"][...]

    @property
    def blob_names(self):
        """Names for each of the arbitrary blobs."""
        if not self.has_blobs:
            return None

        empty_blobs = self.get_blobs(discard=self.iteration)
        return empty_blobs.dtype.names

    @property
    def has_blobs(self):
        """Whether this file has blobs in it."""
        with self.open() as f:
            return f[self.name].attrs["has_blobs"]

    def get_value(self, name, flat=False, thin=1, discard=0):
        """Get a particular kind of entry from the backend file."""
        if not self.initialized:
            raise AttributeError("Cannot get values from uninitialized storage.")

        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError("No iterations performed for this run.")

            if name == "blobs" and not g.attrs["has_blobs"]:
                return None

            v = g[name][discard + thin - 1 : self.iteration : thin]
            if flat:
                s = list(v.shape[1:])
                s[0] = np.prod(v.shape[:2])
                return v.reshape(s)

            return v

    @property
    def size(self):
        """The length of the chain."""
        with self.open() as f:
            g = f[self.name]
            return g["chain"].shape[0]

    @property
    def shape(self):
        """Tuple of (nwalkers, ndim)."""
        with self.open() as f:
            g = f[self.name]
            return g.attrs["nwalkers"], g.attrs["ndim"]

    @property
    def iteration(self):
        """The iteration the chain is currently at."""
        with self.open() as f:
            return f[self.name].attrs["iteration"]

    @property
    def accepted_array(self):
        """An array of bools representing whether parameter proposals were accepted."""
        with self.open() as f:
            return f[self.name]["accepted"][...]

    @property
    def accepted(self):
        """Number of acceptances for each walker."""
        return np.sum(self.accepted_array, axis=0)

    @property
    def random_state(self):
        """The defining random state of the process."""
        with self.open() as f:
            elements = [
                v
                for k, v in sorted(f[self.name].attrs.items())
                if k.startswith("random_state_")
            ]

        return elements if len(elements) else None

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples.

        Parameters
        ----------
        ngrow : int
            The number of steps to grow the chain.
        blobs : dict or None
            A dictionary of extra data, or None
        """
        self._check_blobs(blobs)

        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["chain"].resize(ntot, axis=0)
            g["trials"].resize(ntot, axis=0)
            g["accepted"].resize(ntot, axis=0)
            g["log_prob"].resize(ntot, axis=0)
            g["trial_log_prob"].resize(ntot, axis=0)

            if blobs:
                has_blobs = g.attrs["has_blobs"]
                if not has_blobs:
                    nwalkers = g.attrs["nwalkers"]
                    blobs_dtype = []
                    for k, v in blobs.items():
                        shape = np.atleast_1d(v).shape
                        if len(shape) == 1:
                            shape = shape[0]
                        blobs_dtype += [(k, (np.atleast_1d(v).dtype, shape))]

                    g.create_dataset(
                        "blobs",
                        (ntot, nwalkers),
                        maxshape=(None, nwalkers),
                        dtype=blobs_dtype,
                    )
                else:
                    g["blobs"].resize(ntot, axis=0)

                g.attrs["has_blobs"] = True

    def save_step(
        self, coords, log_prob, blobs, truepos, trueprob, accepted, random_state
    ):
        """Save a step to the file.

        Parameters
        ----------
        coords : ndarray
            The coordinates of the walkers in the ensemble.
        log_prob : ndarray
            The log probability for each walker.
        blobs : ndarray or None
            The blobs for each walker or ``None`` if there are no blobs.
        accepted : ndarray
            An array of boolean flags indicating whether or not the proposal for each
            walker was accepted.
        random_state :
            The current state of the random number generator.
        """
        self._check(coords, log_prob, blobs, accepted)

        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]

            g["chain"][iteration, :, :] = coords
            g["log_prob"][iteration, :] = log_prob
            g["trials"][iteration, :, :] = truepos
            g["trial_log_prob"][iteration, :] = trueprob

            # i.e. blobs is a list of dicts, and if the first dict is non-empty...
            if blobs[0]:
                blobs = np.array(
                    [tuple(b[name] for name in g["blobs"].dtype.names) for b in blobs],
                    dtype=g["blobs"].dtype,
                )
                # Blobs must be a dict
                g["blobs"][iteration, ...] = blobs

            g["accepted"][iteration, :] = accepted

            for i, v in enumerate(random_state):
                g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1

    def _check_blobs(self, blobs):
        if self.has_blobs and not blobs:
            raise ValueError("inconsistent use of blobs")
        if self.iteration > 0 and blobs and not self.has_blobs:
            raise ValueError("inconsistent use of blobs")

    def get_chain(self, **kwargs):
        r"""
        Get the stored chain of MCMC samples.

        Parameters
        ----------
        \*\*kwargs :
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns
        -------
        array[..., nwalkers, ndim]:
            The MCMC samples.
        """
        return self.get_value("chain", **kwargs)

    def get_blobs(self, **kwargs):
        r"""
        Get the chain of blobs for each sample in the chain.

        Parameters
        ----------
        \*\*kwargs :
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)


        Returns
        -------
        array[..., nwalkers]:
            The chain of blobs.
        """
        return self.get_value("blobs", **kwargs)

    def get_log_prob(self, **kwargs):
        """
        Get the chain of log probabilities evaluated at the MCMC samples.

        Parameters
        ----------
        kwargs:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns
        -------
        array[..., nwalkers]:
            The chain of log probabilities.
        """
        return self.get_value("log_prob", **kwargs)

    def get_trialled_log_prob(self, **kwargs):
        """
        Get the chain of log probabilities evaluated as *trials* of the MCMC.

        .. note:: these do not correspond to the chain, but instead correspond to the
                  trialled parameters. Check the :attr:`accepted` property to check if
                  each trial was accepted.

        Parameters
        ----------
        kwargs :
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns
        -------
        array[..., nwalkers]:
            The chain of log probabilities.
        """
        return self.get_value("trial_log_prob", **kwargs)

    def get_trials(self, **kwargs):
        r"""
        Get the stored chain of trials.

        Note these do not corresond to the chain, but instead correspond to the
        trialled parameters. Check the :attr:`accepted` property to check if
        each trial was accepted.

        Parameters
        ----------
        \*\*kwargs :
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns
        -------
        array[..., nwalkers, ndim]:
            The MCMC samples.
        """
        return self.get_value("trials", **kwargs)

    def get_last_sample(self):
        """
        Access the most recent sample in the chain.

        Returns
        -------
        coords : ndarray
            A list of the current positions of the walkers in the parameter space.
            The shape of this object will be ``(nwalkers, dim)``.
        log_prob : list
            The list of log posterior probabilities for the walkers at positions given by
            ``coords`` . The shape of this object is ``(nwalkers,)``.
        rstate :
            The current state of the random number generator.
        blobs : dict, optional
            The metadata "blobs" associated with the current position. The value is only
            returned if blobs have been saved during sampling.
        """
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        it = self.iteration
        last = [
            self.get_chain(discard=it - 1)[0],
            self.get_log_prob(discard=it - 1)[0],
            self.random_state,
        ]
        blob = self.get_blobs(discard=it - 1)

        if blob is not None:
            last.append(blob[0])
        else:
            last.append(None)

        return tuple(last)

    def _check(self, coords, log_prob, blobs, accepted):

        self._check_blobs(blobs[0])
        nwalkers, ndim = self.shape

        if coords.shape != (nwalkers, ndim):
            raise ValueError(
                "invalid coordinate dimensions; expected {0}".format((nwalkers, ndim))
            )
        if log_prob.shape != (nwalkers,):
            raise ValueError(
                "invalid log probability size; expected {0}".format(nwalkers)
            )
        if blobs and len(blobs) != nwalkers:
            raise ValueError("invalid blobs size; expected {0}".format(nwalkers))
        if accepted.shape != (nwalkers,):
            raise ValueError("invalid acceptance size; expected {0}".format(nwalkers))


class HDFStorageUtil:
    """Storage class for MCMC runs."""

    def __init__(self, file_prefix, chain_number=0):
        self.file_prefix = file_prefix
        self.burnin_storage = HDFStorage(file_prefix + ".h5", name="burnin")
        self.sample_storage = HDFStorage(
            file_prefix + ".h5", name="sample_%s" % chain_number
        )

    def reset(self, nwalkers, params, burnin=True, samples=True):
        """Reset the storage so that it is empty."""
        if burnin:
            self.burnin_storage.reset(nwalkers, params=params)
        if samples:
            self.sample_storage.reset(nwalkers, params=params)

    @property
    def burnin_initialized(self):
        """Whether burnin has been initialized."""
        return self.burnin_storage.initialized

    @property
    def samples_initialized(self):
        """Whether sample storage has been initialized."""
        return self.sample_storage.initialized

    def persistValues(
        self, pos, prob, data, truepos, trueprob, accepted, random_state, burnin=False
    ):
        """Save a set of values to the storage."""
        st = self.burnin_storage if burnin else self.sample_storage
        st.save_step(pos, prob, data, truepos, trueprob, accepted, random_state)

    def close(self):
        """No-op."""
        pass


class Params(_util.Params):
    """Params class with added equality."""

    def items(self):
        """Iterate through the params like a dict."""
        for k, v in zip(self.keys, self.values):
            yield k, v

    def __eq__(self, other):
        """Test equality of two instances."""
        if self.__class__.__name__ != other.__class__.__name__:
            return False

        for i, (k, v) in enumerate(self.items()):
            if k not in other.keys:
                return False
            for j, val in enumerate(v):
                if val != other.values[i][j]:
                    return False

        return True


class LikelihoodComputationChain(_Chain):
    """Feature-laden replacement of :class:`cosmoHammer.LikelihoodComputationChain`."""

    def __init__(self, params, *args, **kwargs):
        self.params = params
        self._setup = False  # flag to say if this chain has been setup yet.

        super().__init__(
            min=params[:, 1] if params is not None else None,
            max=params[:, 2] if params is not None else None,
        )

    def build_model_data(self, p=None):
        """
        For a given set of parameters, generate model data for the entire core chain.

        Parameters
        ----------
        p : list or Params object, optional
            The parameters at which to evaluate the model data.

        Returns
        -------
        ctx : dict-like
            A filled context object containing all model quantities generated by core modules in
            the chain.

        Notes
        -----
        The data generated by this method should ideally be *deterministic*, so that input
        parameters map uniquely to
        output data. All data necessary to fully evaluate probabilities of mock data from the
        model data should be
        determined in this method (including model uncertainties, if applicable).
        """
        ctx = self.createChainContext(p)

        for core in self.getCoreModules():
            core.build_model_data(ctx)

        return ctx

    def simulate_mock(self, p=None):
        """
        For a given set of parameters, generate mock data for the entire core chain.

        Parameters
        ----------
        p : list or Params object, optional
            The parameters at which to evaluate the model data. Default is to use default parameters
            of the cores.

        Returns
        -------
        ctx : dict-like
            A filled context object containing all model quantities generated by core modules in the
            chain.

        Notes
        -----
        The data generated by this method are *not* deterministic in general, i.e. they contain the
        randomness that is being constrained by the MCMC process. It is mock data in the sense that
        it should be a representation of the forward-model of the MCMC.
        """
        logger.debug("Simulating Mock Data... ")
        ctx = self.createChainContext(p)

        for core in self.getCoreModules():
            core.simulate_mock(ctx)

        logger.debug("   finished simulating mock data.")
        return ctx

    def addLikelihoodModule(self, module):
        """
        Add a module to the likelihood module list.

        Parameters
        ----------
        module : callable
            The callable module to add for the likelihood computation.
        """
        self.getLikelihoodModules().append(module)
        module._LikelihoodComputationChain = self

    def addCoreModule(self, module):
        """
        Add a module to the likelihood module list.

        Parameters
        ----------
        module : callable
            The callable module to add for the computation of the data.
        """
        self.getCoreModules().append(module)
        module._LikelihoodComputationChain = self

    def invokeCoreModule(self, coremodule, ctx):
        """Call a particular coremodule, filling up given context.

        Parameters
        ----------
        coremodule :
            An object with the Core API.
        ctx : :class:`cosmoHammer.utils.Context` object
            The context "dict" to fill up.
        """
        # Ensure that the chain is setup before invoking anything.
        if not self._setup:
            self.setup()

        logger.debug(f"Invoking {coremodule.__class__.__name__}...")
        coremodule(ctx)
        logger.debug("... finished.")

        coremodule.prepare_storage(
            ctx, ctx.getData()
        )  # This adds the ability to store stuff.

    def invokeLikelihoodModule(self, module, ctx):
        """Invoke a given likelihood module."""
        # Ensure that the chain is setup before invoking anything.
        if not self._setup:
            self.setup()

        logger.debug(f"Reducing data for {module.__class__.__name__}...")
        model = module.reduce_data(ctx)
        logger.debug("... done reducing data")

        if hasattr(module, "store"):
            logger.debug(f"Storing blobs for {module.__class__.__name__}...")
            module.store(model, ctx.getData())
            logger.debug("... done storing blobs.")

        logger.debug(f"Computing Likelihood for {module.__class__.__name__}...")
        lnl = module.computeLikelihood(model)
        logger.debug(f"... done computing likelihood (lnl = {lnl:.3e}")
        return lnl

    def __call__(self, p):
        """Call the full likelihood chain."""
        # Need to do Garbage Collection explicitly to kill the circular
        # refs that are in the this chain.
        gc.collect(2)

        try:
            return super().__call__(p)
        except ParameterError:
            return -np.inf, []

    def createChainContext(self, p=None):
        """Returns a new instance of a chain context."""
        if p is None:
            p = {}

        try:
            p = Params(*zip(self.params.keys, p))
        except Exception:
            # no params or params has no keys
            pass
        return ChainContext(self, p)

    def setup(self):
        """Run the setup of all cores and likelihoods."""
        if not self._setup:
            for cModule in self.getCoreModules():
                if hasattr(cModule, "setup"):
                    cModule.setup()
                    cModule._is_setup = True

            for cModule in self.getLikelihoodModules():
                if hasattr(cModule, "setup"):
                    cModule.setup()
                    cModule._is_setup = True

            self._setup = True
        else:
            warnings.warn(
                "Attempting to setup LikelihoodComputationChain when it is already setup! "
                "Ignoring..."
            )

    def __eq__(self, other):
        """Check equality with another :class:`LikelihoodComputationChain`."""
        if self.__class__.__name__ != other.__class__.__name__:
            return False

        if self.params != other.params:
            return False

        for thisc, thatc in zip(self.getCoreModules(), other.getCoreModules()):
            if thisc != thatc:
                return False

        return all(
            thisc == thatc
            for thisc, thatc in zip(
                self.getLikelihoodModules(), other.getLikelihoodModules()
            )
        )


class CosmoHammerSampler(_CosmoHammerSampler):
    """Upgraded :class:`cosmoHammer.CosmoHammerSampler` with the ability to continue sampling."""

    def __init__(
        self,
        likelihoodComputationChain,
        continue_sampling=False,
        log_level_stream=logging.ERROR,
        max_init_attempts=100,
        *args,
        **kwargs,
    ):
        self.continue_sampling = continue_sampling
        self._log_level_stream = log_level_stream

        super().__init__(
            params=likelihoodComputationChain.params,
            likelihoodComputationChain=likelihoodComputationChain,
            *args,
            **kwargs,
        )

        self.max_init_attempts = max_init_attempts
        if not self.reuseBurnin:
            self.storageUtil.reset(self.nwalkers, self.params)

        if not continue_sampling:
            self.storageUtil.reset(self.nwalkers, self.params, burnin=False)

        if not self.storageUtil.burnin_initialized:
            self.storageUtil.reset(
                self.nwalkers, self.params, burnin=True, samples=False
            )

        if not self.storageUtil.samples_initialized:
            self.storageUtil.reset(
                self.nwalkers, self.params, burnin=False, samples=True
            )
            ""
        if self.storageUtil.burnin_storage.iteration >= self.burninIterations:
            self.log("all burnin iterations already completed")

        if (
            self.storageUtil.sample_storage.iteration > 0
            and self.storageUtil.burnin_storage.iteration < self.burninIterations
        ):
            self.log(
                "resetting sample iterations because more burnin iterations requested."
            )
            self.storageUtil.reset(self.nwalkers, self.params, burnin=False)

        if self.storageUtil.sample_storage.iteration >= self.sampleIterations:
            raise ValueError(
                "All Samples have already been completed. Try with continue_sampling=False."
            )

    def _configureLogging(self, filename, logLevel):
        super()._configureLogging(filename, logLevel)

        logger = getLogger()
        logger.setLevel(logLevel)
        ch = logging.StreamHandler()
        ch.setLevel(self._log_level_stream)
        # create formatter and add it to the handlers
        formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    def startSampling(self):
        """Launch the sampling."""
        try:
            if self.isMaster():
                self.log(self.__str__())

            prob = None
            rstate = None
            datas = None
            pos = None
            if self.storageUtil.burnin_storage.iteration < self.burninIterations:
                if self.burninIterations:
                    if self.storageUtil.burnin_storage.iteration:
                        pos, prob, rstate, datas = self.loadBurnin()

                    if (
                        self.storageUtil.burnin_storage.iteration
                        < self.burninIterations
                    ):
                        pos, prob, rstate, datas = self.startSampleBurnin(
                            pos, prob, rstate, datas
                        )
                else:
                    pos = self.createInitPos()
            else:
                if self.storageUtil.sample_storage.iteration:
                    pos, prob, rstate, datas = self.loadSamples()

                else:
                    pos = self.createInitPos()

            # Starting from the final position in the burn-in chain, start sampling.
            self.log("start sampling after burn in")
            start = time.time()
            self.sample(pos, prob, rstate, datas)
            end = time.time()
            self.log("sampling done! Took: " + str(round(end - start, 4)) + "s")

            # Print out the mean acceptance fraction. In general, acceptance_fraction
            # has an entry for each walker
            self.log(
                "Mean acceptance fraction:"
                + str(round(np.mean(self._sampler.acceptance_fraction), 4))
            )
        finally:
            if self._sampler.pool is not None:
                try:
                    self._sampler.pool.close()
                except AttributeError:
                    pass
                try:
                    self.storageUtil.close()
                except AttributeError:
                    pass

    def createEmceeSampler(self, lnpostfn, **kwargs):
        """Create the emcee sampler."""
        if self.isMaster():
            self.log("Using emcee " + str(emcee.__version__))
        return EnsembleSampler(
            pmin=self.likelihoodComputationChain.min,
            pmax=self.likelihoodComputationChain.max,
            nwalkers=self.nwalkers,
            dim=self.paramCount,
            lnpostfn=lnpostfn,
            threads=self.threadCount,
            **kwargs,
        )

    def _load(self, burnin=False):
        stg = (
            self.storageUtil.burnin_storage
            if burnin
            else self.storageUtil.sample_storage
        )

        self.log(
            "reusing previous %s: %s iterations"
            % ("burnin" if burnin else "samples", stg.iteration)
        )
        pos, prob, rstate, data = stg.get_last_sample()
        if data is not None:
            data = [{k: d[k] for k in d.dtype.names} for d in data]
        return pos, prob, rstate, data

    def loadBurnin(self):
        """Load the burn in from the file system."""
        return self._load(burnin=True)

    def loadSamples(self):
        """Load the samples from the file system."""
        return self._load(burnin=False)

    def startSampleBurnin(self, pos=None, prob=None, rstate=None, data=None):
        """Run the sampler for the burn in."""
        if self.storageUtil.burnin_storage.iteration:
            self.log("continue burn in")
        else:
            self.log("start burn in")
        start = time.time()

        if pos is None:
            pos = self.createInitPos()
        pos, prob, rstate, data = self.sampleBurnin(pos, prob, rstate, data)
        end = time.time()
        self.log("burn in sampling done! Took: " + str(round(end - start, 4)) + "s")
        self.log(
            "Mean acceptance fraction for burn in:"
            + str(round(np.mean(self._sampler.acceptance_fraction), 4))
        )

        self.resetSampler()

        return pos, prob, rstate, data

    def _sample(self, p0, prob=None, rstate=None, datas=None, burnin=False):
        """Run the emcee sampler."""
        stg = (
            self.storageUtil.burnin_storage
            if burnin
            else self.storageUtil.sample_storage
        )
        niter = self.burninIterations if burnin else self.sampleIterations

        _lastprob = prob if prob is None else [0] * len(p0)

        # Set to None in case iterations is zero.
        pos = None

        for pos, prob, rstate, realpos, realprob, datas in self._sampler.sample(
            p0,
            iterations=niter - stg.iteration,
            lnprob0=prob,
            rstate0=rstate,
            blobs0=datas,
        ):
            if self.isMaster():
                # Need to grow the storage first
                if not stg.iteration:
                    stg.grow(niter - stg.iteration, datas[0])

                # If we are continuing sampling, we need to grow it more.
                if stg.size < niter:
                    stg.grow(niter - stg.size, datas[0])

                self.storageUtil.persistValues(
                    pos,
                    prob,
                    datas,
                    truepos=realpos,
                    trueprob=realprob,
                    accepted=prob != _lastprob,
                    random_state=rstate,
                    burnin=burnin,
                )
                if stg.iteration % 10 == 0:
                    self.log("Iteration finished:" + str(stg.iteration))

                _lastprob = 1 * prob

                if self.stopCriteriaStrategy.hasFinished():
                    break

        return pos, prob, rstate, datas

    def sampleBurnin(self, p0, prob=None, rstate=None, datas=None):
        """Run burnin samples."""
        return self._sample(p0, prob, rstate, datas, burnin=True)

    def sample(self, burninPos, burninProb=None, burninRstate=None, datas=None):
        """Run emcee sampling."""
        return self._sample(burninPos, burninProb, burninRstate, datas)

    @property
    def samples(self):
        """The samples that have been generated."""
        if not self.storageUtil.sample_storage.initialized:
            raise ValueError("Cannot access samples before sampling.")
        else:
            return self.storageUtil.sample_storage

    def createInitPos(self):
        """Create initial positions."""
        i = 0
        pos = []

        while len(pos) < self.nwalkers and i < self.max_init_attempts:
            tmp_pos = self.initPositionGenerator.generate()

            for tmp_p in tmp_pos:
                if self.likelihoodComputationChain.isValid(tmp_p):
                    pos.append(tmp_p)

            i += 1

        if i == self.max_init_attempts:
            raise ValueError(
                "No suitable initial positions for the walkers could be obtained in {"
                "max_attempts} attemps".format(max_attempts=self.max_init_attempts)
            )

        return pos[: self.nwalkers]
