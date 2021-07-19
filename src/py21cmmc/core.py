"""Module providing Core Modules for cosmoHammer.

This is the basis of the plugin system for :mod:`py21cmmc`.

TODO: Add description of the API of cores (and how to define new ones).
"""
import copy
import h5py
import inspect
import logging
import numpy as np
import py21cmfast as p21
import warnings
from hashlib import md5
from os import path

from . import _utils as ut

logger = logging.getLogger("21cmFAST")


class NotSetupError(AttributeError):
    """Exception for when a Core has not yet been setup."""

    def __init__(self):
        default_message = (
            "setup() must have been called on the chain to use this method/attribute!"
        )
        super().__init__(default_message)


class NotAChain(AttributeError):
    """Exception when method is called outside a :class:`LikelihoodComputationChain`."""

    def __init__(self):
        default_message = (
            "this Core or Likelihood must be part of a LikelihoodComputationChain "
            "to enable this method/attribute!"
        )
        super().__init__(default_message)


class AlreadySetupError(Exception):
    """Exception to be raised if trying to setup a core twice."""

    pass


class ModuleBase:
    """Base module for both Cores and Likelihoods."""

    # extra attributes (in addition to those passed to init) that define equality
    _extra_defining_attributes = ()

    # attributes to ignore (from those passed to init) for determining equality
    _ignore_attributes = ()

    # Cores that need to be loaded if this core is loaded. Sub-tuples in the list indicate
    # "or" relationship.
    required_cores = ()

    def __init__(self):
        self._is_setup = False

    def _check_required_cores(self):
        for rc in self.required_cores:
            # Ensure the required_core is a tuple -- we check that at least *one*
            # of the cores in the tuple is in the _cores.
            if not hasattr(rc, "__len__"):
                rc = (rc,)

            if not any(any(isinstance(m, r) for r in rc) for m in self._cores):
                raise ValueError(
                    "%s needs the %s to be loaded."
                    % (self.__class__.__name__, rc.__class__.__name__)
                )

    @property
    def chain(self):
        """Reference to the :class:`~LikelihoodComputationChain` containing this core."""
        try:
            return self._LikelihoodComputationChain
        except AttributeError:
            raise NotAChain

    @property
    def parameter_names(self):
        """Names of the parameters of the full chain."""
        return getattr(self.chain.params, "keys", [])

    def __eq__(self, other):
        """Compare this to another object for equality."""
        if self.__class__.__name__ != other.__class__.__name__:
            return False

        args = []
        for cls in self.__class__.mro():
            args += inspect.getfullargspec(cls.__init__).args

        args = tuple(set(args))

        for arg in args + self._extra_defining_attributes:

            if arg == "self" or arg in self._ignore_attributes:
                continue

            try:
                if hasattr(self, "_" + arg):
                    if getattr(self, "_" + arg) != getattr(other, "_" + arg):
                        return False
                elif hasattr(self, arg):
                    if getattr(self, arg) != getattr(other, arg):
                        return False
                else:
                    logger.warning(f"parameter {arg} not found in instance")

            except ValueError:
                logger.warning(
                    f"parameter {arg} has type which does not allow for comparison"
                )

        return True

    @property
    def _cores(self):
        """List of all loaded cores."""
        return self.chain.getCoreModules()

    @property
    def _rq_cores(self):
        """List of all loaded cores that are in the requirements, in order of the requirements."""
        req = ut.flatten(self.required_cores)
        return tuple(core for core in self._cores for r in req if isinstance(core, r))

    @property
    def core_primary(self):
        """The first core that appears in the requirements."""
        return self._rq_cores[0] if self._rq_cores else self._cores[0]

    def setup(self):
        """Perform any post-init setup of the object."""
        self._check_required_cores()


class CoreBase(ModuleBase):
    """Base module for all cores."""

    def __init__(self, store=None):
        super().__init__()
        self.store = store or {}

    def _check_required_cores(self):
        for core in self._cores:
            for rc in self.required_cores:
                if core.__class__.__name__ == rc.__class__.__name__:
                    break
                if core.__class__.__name__ == self.__class__.__name__:
                    raise ValueError(
                        "{this} requires {that} to be loaded.".format(
                            this=self.__class__.__name__, that=rc.__class__.__name__
                        )
                    )

    def prepare_storage(self, ctx, storage):
        """Add variables to dict which cosmoHammer will automatically store with the chain."""
        for name, storage_function in self.store.items():
            try:
                storage[name] = storage_function(ctx)
            except Exception:
                logger.error(
                    "Exception while trying to evaluate storage function %s" % name
                )
                raise

    def build_model_data(self, ctx):
        """
        Construct model data and place it in the context.

        The data generated by this method should ideally be *deterministic*, so that
        input parameters (which are inherently contained in the `ctx` object) map
        uniquely to output data. The addition of stochasticity in order to produce mock
        data is done in the :meth:`~convert_model_to_mock` method. All data necessary to
        fully evaluate probabilities of mock data from the model data should be
        determined in this method (including model uncertainties, if applicable).

        Parameters
        ----------
        ctx : dict-like
            The context, from which parameters are accessed.

        Returns
        -------
        dct : dict
            A dictionary of data which was simulated.
        """
        pass

    def convert_model_to_mock(self, ctx):
        """
        Generate random mock data.

        Given a context object containing data from :meth:`~build_model_data`, generate
        random mock data, which should represent an exact forward-model of the process
        under investigation.

        Parameters
        ----------
        ctx : dict-like
            The context, from which parameters and other simulated model data can be accessed.
        """
        pass

    def simulate_mock(self, ctx):
        """Generate all mock data and add it to the context."""
        self.build_model_data(ctx)
        self.convert_model_to_mock(ctx)

    def __call__(self, ctx):
        """Call the class.

        By default, it will just build model data, with no stochasticity.
        """
        self.build_model_data(ctx)


class CoreCoevalModule(CoreBase):
    """A Core Module which evaluates coeval cubes at given redshift.

    On each iteration, this module will add to the context:

    * ``init``: an :class:`~py21cmmc._21cmfast.wrapper.InitialConditions` instance
    * ``perturb``: a :class:`~py21cmmc._21cmfast.wrapper.PerturbedField` instance
    * ``xHI``: an :class:`~py21cmmc._21cmfast.wrapper.IonizedBox` instance
    * ``brightness_temp``: a :class:`~py21cmmc._21cmfast.wrapper.BrightnessTemp` instance

    Parameters
    ----------
    redshift : float or array_like
         The redshift(s) at which to evaluate the coeval cubes.
    user_params : dict or :class:`~py21cmfast.UserParams`
        Parameters affecting the overall dimensions of the cubes.
    flag_options : dict or :class:`~py21cmfast.FlagOptions`
        Options affecting choices for how the reionization is calculated.
    astro_params : dict or :class:`~py21cmfast.AstroParams`
        Astrophysical parameters of reionization.

        .. note:: None of the parameters provided here affect the *MCMC* as such; they
              merely provide a background model on which the MCMC will be performed.
              Thus for example, passing ``HII_EFF_FACTOR=30`` in ``astro_params`` here
              will be over-written per-iteration if ``HII_EFF_FACTOR`` is also passed as
              a ``parameter`` to an MCMC routine using this core module.

    cosmo_params : dict or :class:`~py21cmfast.CosmoParams`
        Cosmological parameters of the simulations. Like ``astro_params``, these
        are the *fiducial* parameters, but may be updated during an MCMC.
    regenerate : bool, optional
        Whether to force regeneration of simulations, even if matching cached data is found.
    do_spin_temp: bool, optional
        Whether to use spin temperature in the calculation, or assume the saturated limit.
    z_step_factor: float, optional
        How large the logarithmic steps between redshift are (if required).
    z_heat_max: float, optional
        Controls the global `Z_HEAT_MAX` parameter, which specifies the maximum redshift
        up to which heating sources are required to specify the ionization field. Beyond
        this, the ionization field is specified directly from the perturbed density field.
    ctx_variables : list of str, optional
            A list of strings. The strings must correspond to any (pickleable) member of
            :class:`py21cmfast.Coeval`. These will be stored in the context on every iteration. Omitting as many as
            possible is useful in that it reduces the memory that needs to be transmitted to each process. Furthermore,
            in-built pickling has a restriction that arrays cannot be larger than 4GiB, which can be
            easily over-run. Some typical options are:
            * "brightness_temp"
            * "xH_box"
            * "density"
            * "velocity"
            * "Gamma12_box"
    initial_conditions_seed : int, optional
        If not `change_seeds_every_iter`, then this will define the random seed on which
        the initial conditions for _all_ iterations is based. By default, a seed will be
        chosen at random, _unless_ initial conditions exist in cache that match the
        parameters of this instance (and ``regenerate`` is False). In this case, the
        seed of the existing box will be adopted.

    Other Parameters
    ----------------
    store :  dict, optional
        The (derived) quantities/blobs to store in the MCMC chain, default empty. See
        Notes below for details.
    cache_dir : str, optional
        The directory in which to search for the boxes and write them. By default, this
        is the directory given by ``direc`` in the configuration file,
        ``~/.21CMMC/config.yml``. Note that for *reading* data, while the specified
        ``direc`` is searched first, the default directory will *also* be searched if no
        appropriate data is found in ``direc``.
    cache_ionize : bool, optional
        Whether to cache ionization data sets (done before parameter retention step).
        Default False.


    Notes
    -----
    The ``store`` keyword is a dictionary, where each key specifies the name of the
    resulting data entry in the samples object, and the value is a callable which
    receives the ``context``, and returns a value from it.

    .. note:: the ``store`` callable is saved to the core instance, which must be
              pickleable in order to use multiprocessing. Thus it is generally unwise
              to use a ``lambda`` function as the callable.

    This means that the context can be inspected and arbitrarily summarised before
    storage. In particular, this allows for taking slices of arrays and saving them. One
    thing to note is that the context is dictionary-like, but is not a dictionary. The
    elements of the context are only available by using the ``get`` method, rather than
    directly subscripting the object like a normal dictionary.

    .. warning:: Only scalars and arrays are supported for storage in the chain itself.
    """

    _ignore_attributes = ["keep_data_in_memory"]

    def __init__(
        self,
        redshift,
        user_params=None,
        flag_options=None,
        astro_params=None,
        cosmo_params=None,
        regenerate=True,
        change_seed_every_iter=False,
        ctx_variables=("brightness_temp", "xH_box"),
        initial_conditions_seed=None,
        global_params=None,
        **io_options,
    ):
        super().__init__(io_options.get("store", None))

        self.redshift = redshift
        if not hasattr(self.redshift, "__len__"):
            self.redshift = [self.redshift]

        self.user_params = p21.UserParams(user_params)
        self.flag_options = p21.FlagOptions(flag_options)
        self.astro_params = p21.AstroParams(astro_params)
        self.cosmo_params = p21.CosmoParams(cosmo_params)
        self.change_seed_every_iter = change_seed_every_iter
        self.initial_conditions_seed = initial_conditions_seed

        self.regenerate = regenerate
        self.ctx_variables = ctx_variables

        self.global_params = global_params or {}

        self.io_options = {
            "store": {},  # (derived) quantities to store in the MCMC chain.
            "cache_dir": None,  # where full data sets will be written/read from.
            "cache_mcmc": False,  # whether to cache ionization data sets
            # (done before parameter retention step)
        }

        self.io_options.update(io_options)

        if self.initial_conditions_seed and self.change_seed_every_iter:
            logger.warning(
                "Attempting to set initial conditions seed while desiring to change seeds every "
                "iteration. Unsetting initial conditions seed."
            )
            self.initial_conditions_seed = None

    def setup(self):
        """
        Perform setup of the core.

        Notes
        -----
        This method is called automatically by its parent
        :class:`~LikelihoodComputationChain`, and should not be invoked directly.
        """
        super().setup()

        # If the chain has different parameter truths, we want to use those for our defaults.
        self.astro_params, self.cosmo_params = self._update_params(
            self.chain.createChainContext().getParams()
        )

        # Here we save to disk the full default realization.
        # The init and perturb boxes here can usually be re-used, and the box serves
        # as a nice thing to compare to after MCMC.
        # If modifying cosmo, we don't want to do this, because we'll create them
        # on the fly on every iteration.
        if (
            all(p not in self.cosmo_params.self.keys() for p in self.parameter_names)
            and not self.change_seed_every_iter
        ):
            logger.info("Initializing default boxes for the entire chain.")
            coeval = p21.run_coeval(
                redshift=self.redshift,
                user_params=self.user_params,
                cosmo_params=self.cosmo_params,
                astro_params=self.astro_params,
                flag_options=self.flag_options,
                write=True,
                regenerate=self.regenerate,
                direc=self.io_options["cache_dir"],
                random_seed=self.initial_conditions_seed,
                **self.global_params,
            )

            # update the seed
            self.initial_conditions_seed = coeval[0].random_seed

            logger.info("Initialization done.")

    def build_model_data(self, ctx):
        """Compute all data defined by this core and add it to the context."""
        # Update parameters
        logger.debug(f"Updating parameters: {ctx.getParams()}")
        astro_params, cosmo_params = self._update_params(ctx.getParams())
        logger.debug(f"AstroParams: {astro_params}")
        logger.debug(f"CosmoParams: {cosmo_params}")

        # Call C-code
        coeval = p21.run_coeval(
            redshift=self.redshift,
            astro_params=astro_params,
            cosmo_params=cosmo_params,
            flag_options=self.flag_options,
            user_params=self.user_params,
            regenerate=False,
            random_seed=self.initial_conditions_seed,
            write=self.io_options["cache_mcmc"],
            direc=self.io_options["cache_dir"],
            **self.global_params,
        )

        logger.debug(f"Adding {self.ctx_variables} to context data")
        for key in self.ctx_variables:
            try:
                ctx.add(key, [getattr(c, key) for c in coeval])
            except AttributeError:
                raise ValueError(f"ctx_variable {key} not an attribute of Coeval")

    def _update_params(self, params):
        """
        Update all the parameter structures which get passed to the driver, for one iteration.

        Parameters
        ----------
        params :
            Parameter object from cosmoHammer
        """
        ap_dict = copy.copy(self.astro_params.self)
        cp_dict = copy.copy(self.cosmo_params.self)

        ap_dict.update(
            **{
                k: getattr(params, k)
                for k, v in params.items()
                if k in self.astro_params.defining_dict
            }
        )
        cp_dict.update(
            **{
                k: getattr(params, k)
                for k, v in params.items()
                if k in self.cosmo_params.defining_dict
            }
        )

        return p21.AstroParams(**ap_dict), p21.CosmoParams(**cp_dict)


class CoreLightConeModule(CoreCoevalModule):
    """
    Core module for evaluating lightcone simulations.

    See :class:`~CoreCoevalModule` for info on all parameters, which are identical to
    this class, with the exception of `redshift`, which in this case must be a scalar.

    This module will add the following quantities to the context:

    * ``lightcone``: a :class:`~py21cmfast.LightCone` instance.
    """

    def __init__(self, *, max_redshift=None, **kwargs):
        if "ctx_variables" in kwargs:
            warnings.warn(
                "ctx_variables does not apply to the lightcone module (at least not yet). It will "
                "be ignored."
            )

        super().__init__(**kwargs)
        self.max_redshift = max_redshift

    def setup(self):
        """Setup the chain."""
        # If the chain has different parameter truths, we want to use those for our defaults.
        self.astro_params, self.cosmo_params = self._update_params(
            self.chain.createChainContext().getParams()
        )

        # Here we save to disk the full default realization.
        # The init and perturb boxes here can usually be re-used, and the box serves
        # as a nice thing to compare to after MCMC.
        # If modifying cosmo, we don't want to do this, because we'll create them
        # on the fly on every iteration.
        if (
            all(p not in self.cosmo_params.self.keys() for p in self.parameter_names)
            and not self.change_seed_every_iter
        ):
            logger.info("Initializing default boxes for the entire chain.")
            lightcone_quantities = (
                "brightness_temp",
                "xH_box",
                "temp_kinetic_all_gas",
                "Gamma12_box",
                "density",
                "MFP_box",
                "temp_kinetic_all_gas",
                "dNion_box",
            )
            lightcone = p21.run_lightcone(
                redshift=self.redshift[0],
                max_redshift=self.max_redshift,
                user_params=self.user_params,
                cosmo_params=self.cosmo_params,
                astro_params=self.astro_params,
                flag_options=self.flag_options,
                write=True,
                regenerate=self.regenerate,
                direc=self.io_options["cache_dir"],
                random_seed=self.initial_conditions_seed,
                lightcone_quantities=lightcone_quantities,
                global_quantities=lightcone_quantities,
                **self.global_params,
            )

            # update the seed
            self.initial_conditions_seed = lightcone.random_seed

            logger.info("Initialization done.")

    def build_model_data(self, ctx):
        """Compute all data defined by this core and add it to the context."""
        # Update parameters
        astro_params, cosmo_params = self._update_params(ctx.getParams())

        # TODO: make it a option that users can decide
        lightcone_quantities = (
            "brightness_temp",
            "xH_box",
            "temp_kinetic_all_gas",
            "Gamma12_box",
            "density",
            "MFP_box",
            "temp_kinetic_all_gas",
            "dNion_box",
        )

        # Call C-code
        lightcone = p21.run_lightcone(
            redshift=self.redshift[0],
            max_redshift=self.max_redshift,
            astro_params=astro_params,
            flag_options=self.flag_options,
            cosmo_params=cosmo_params,
            user_params=self.user_params,
            regenerate=False,
            random_seed=self.initial_conditions_seed,
            write=self.io_options["cache_mcmc"],
            direc=self.io_options["cache_dir"],
            lightcone_quantities=lightcone_quantities,
            global_quantities=lightcone_quantities,
            **self.global_params,
        )

        ctx.add("lightcone", lightcone)


class CoreLuminosityFunction(CoreCoevalModule):
    r"""A Core Module that produces model luminosity functions at a range of redshifts.

    Parameters
    ----------
    sigma : float, callable, list of callables, or array_like
        The standard deviation on the luminosity function measurement. If a float,
        it is considered to be the standard deviation for all redshifts and luminosity
        bins. If a 1D array, it is assumed to be a function of luminosity, and must
        have the same length as the output luminosity from
        :func:`py21cmfast.wrapper.compute_luminosity_function`. If a callable,
        assumed to take a single argument -- a UV magitude array -- and return the
        standard deviation (the same for all redshifts). If a list of callables, must
        be the same length as redshift, with each callable having the same signature
        as already described. If a 2D array, must have shape ``(n_redshifts, n_luminosity_bins)``.

    Other Parameters
    ----------------
    \*\*kwargs :
        All other parameters are the same as :class:`CoreCoevalModule`.
    """

    def __init__(self, sigma=None, name="", n_muv_bins=100, **kwargs):
        self._sigma = sigma
        self.name = str(name)
        self.n_muv_bins = n_muv_bins
        super().__init__(**kwargs)

    def setup(self):
        """Run post-init setup."""
        CoreBase.setup(self)

        # If the chain has different parameter truths, we want to use those for our defaults.
        self.astro_params, self.cosmo_params = self._update_params(
            self.chain.createChainContext().getParams()
        )

    def run(self, astro_params, cosmo_params):
        """Return the luminosity function for given parameters."""
        return p21.compute_luminosity_function(
            redshifts=self.redshift,
            astro_params=astro_params,
            flag_options=self.flag_options,
            cosmo_params=cosmo_params,
            user_params=self.user_params,
            nbins=self.n_muv_bins,
        )

    def build_model_data(self, ctx):
        """Compute all data defined by this core and add it to the context."""
        # Update parameters
        astro_params, cosmo_params = self._update_params(ctx.getParams())

        # Call C-code
        Muv, mhalo, lfunc = self.run(astro_params, cosmo_params)

        Muv = [m[~np.isnan(lf)] for lf, m in zip(lfunc, Muv)]
        mhalo = [m[~np.isnan(lf)] for lf, m in zip(lfunc, mhalo)]
        lfunc = [m[~np.isnan(lf)] for lf, m in zip(lfunc, lfunc)]

        ctx.add(
            "luminosity_function" + self.name,
            {"Muv": Muv, "mhalo": mhalo, "lfunc": lfunc},
        )

    @property
    def sigma(self):
        """Either a list of callables, or list/array of arrays. Length n_redshifts."""
        if self._sigma is None:
            return None

        if not hasattr(self._sigma, "__len__") or len(self._sigma) != len(
            self.redshift
        ):
            return [self._sigma] * len(self.redshift)
        else:
            return self._sigma

    def convert_model_to_mock(self, ctx):
        """Update context entries for luminosity function to have randomness."""
        if self.sigma is None:
            raise ValueError("Cannot create a mock with sigma=None!")

        lfunc = ctx.get("luminosity_function" + self.name)["lfunc"]
        muv = ctx.get("luminosity_function" + self.name)["Muv"]

        for i, s in enumerate(self.sigma):  # each redshift
            try:
                lfunc[i] += np.random.normal(loc=0, scale=s(muv), size=len(lfunc[i]))
            except TypeError:

                lfunc[i] += np.random.normal(loc=0, scale=s, size=len(lfunc[i]))


class CoreForest(CoreLightConeModule):
    r"""A Core Module that produces model effective optical depth at a range of redshifts.

    name : str
        The name used to match the likelihood

    observation : str
        The observation that is used to construct the tau_eff statisctic.
        Currently, only bosman_optimistic and bosman_pessimistic are provided for Bosman+18,
        and xqr30 for Bosman+21.

    n_realization : int
        The number of realizations to evaluate the error covariance matrix, default is 150.

    bin_size : float
        The bin size for calculating effective optical depth, default is 50 / hlittle

    mean_flux : float
        The mean flux (usually from observation) used to rescale the modelling results.
        If not provided, the modelled mean flux will be rescaled according to input parameters
        log10_f_rescale and f_rescale_slope.

    even_spacing : bool
        Whether or not to do different realizations by having even space, default is True.

    seed : int
        The randome seed for doing realizations, default if 1993. This is only needed when even_spacing is False.

    Other Parameters
    ----------------
    \*\*kwargs :
        All other parameters are the same as :class:`CoreCoevalModule`.
    """

    def __init__(
        self,
        name="",
        observation="bosman_optimistic",
        n_realization=150,
        bin_size=None,
        mean_flux=None,
        even_spacing=True,
        seed=1993,
        **kwargs,
    ):
        self.name = str(name)
        self.observation = str(observation)
        self.n_realization = n_realization
        self.bin_size = bin_size
        self.mean_flux = mean_flux
        self.even_spacing = even_spacing
        self.seed = seed
        super().__init__(**kwargs)

        if self.bin_size is None:
            self.bin_size = 50 / self.cosmo_params.hlittle

        if (
            self.observation == "bosman_optimistic"
            or self.observation == "bosman_pessimistic"
        ):
            data = np.load(
                path.join(path.dirname(__file__), "data/Forests/Bosman18/data.npz"),
                allow_pickle=True,
            )
            targets = (data["zs"] > self.redshift[0] - 0.1) * (
                data["zs"] <= self.redshift[0] + 0.1
            )
        elif self.observation == "xqr30":
            data = np.load(
                path.join(path.dirname(__file__), "data/Forests/Bosman21/data.npz"),
                allow_pickle=True,
            )
            self.fbias_FGPA = np.load(
                path.join(
                    path.dirname(__file__),
                    "data/Forests/Bosman21/fbias_FGPA/z%s.npy"
                    % str(self.redshift[0]).replace(".", "pt"),
                ),
                allow_pickle=True,
            )
            targets = (data["zs"] > self.redshift[0] - 0.05) * (
                data["zs"] <= self.redshift[0] + 0.05
            )
        else:
            raise NotImplementedError("Use bosman_optimistic or bosman_pessimistic!")
        self.nlos = sum(targets)
        logger.debug(
            "doing %s at z=%.1f, %d los"
            % (self.observation, self.redshift[0], sum(targets))
        )

        if self.even_spacing:
            if self.nlos * self.n_realization > self.user_params.HII_DIM ** 2:
                raise ValueError(
                    "You asked for %d realizations with even_spacing, larger than what the box has (Total los / needed los = %d / %d)! Increase HII_DIM!"
                    % (self.n_realization, self.user_params.HII_DIM ** 2, self.nlos)
                )
        else:
            if self.nlos * self.n_realization > self.user_params.HII_DIM ** 2:
                logger.warning(
                    "You asked for %d realizations with even_spacing, larger than what the box has (Total los / needed los = %d / %d)! Increase HII_DIM otherwise Cosmic Variance can be underestiamted!"
                    % (self.nlos, self.user_params.HII_DIM ** 2, self.nlos)
                )
            if self.nlos > self.user_params.HII_DIM ** 2:
                raise ValueError(
                    "You asked for %d los without even_spacing, larger than what the box has (%d)! Increase HII_DIM!"
                    % (self.nlos, self.user_params.HII_DIM ** 2)
                )

    def setup(self):
        """Run post-init setup."""
        CoreBase.setup(self)

    def tau_GP(self, gamma_bg, delta, temp, redshifts):
        r"""Calculating the lyman-alpha optical depth in each pixel using the fluctuating GP approximation.

        Parameters
        ----------
        gamma_bg : float or array_like
            The background photonionization rate in units of 1e-12 s**-1

        delta : float or array_like
            The underlying overdensity

        temp : float or array_like
            The kinectic temperature of the gas in 1e4 K

        redshifts : float or array_like
            Correspoding redshifts along the los
        """
        gamma_local = np.zeros_like(gamma_bg)
        residual_xHI = np.zeros_like(gamma_bg, dtype=np.float64)

        flag_neutral = gamma_bg == 0
        flag_zerodelta = delta == 0

        if gamma_bg.shape != redshifts.shape:
            redshifts = np.tile(redshifts, (*gamma_bg.shape[:-1], 1))

        delta_ss = (
            2.67e4 * temp ** 0.17 * (1.0 + redshifts) ** -3 * gamma_bg ** (2.0 / 3.0)
        )
        gamma_local[~flag_neutral] = gamma_bg[~flag_neutral] * (
            0.98
            * (
                (1.0 + (delta[~flag_neutral] / delta_ss[~flag_neutral]) ** 1.64)
                ** -2.28
            )
            + 0.02 * (1.0 + (delta[~flag_neutral] / delta_ss[~flag_neutral])) ** -0.84
        )

        Y_He = 0.245
        # TODO: use global_params
        residual_xHI[~flag_zerodelta] = 1 + gamma_local[~flag_zerodelta] * 1.0155e7 / (
            1.0 + 1.0 / (4.0 / Y_He - 3)
        ) * temp[~flag_zerodelta] ** 0.75 / (
            delta[~flag_zerodelta] * (1.0 + redshifts[~flag_zerodelta]) ** 3
        )
        residual_xHI[~flag_zerodelta] = residual_xHI[~flag_zerodelta] - np.sqrt(
            residual_xHI[~flag_zerodelta] ** 2 - 1.0
        )

        return (
            7875.053145028655
            / (
                self.cosmo_params.hlittle
                * np.sqrt(
                    self.cosmo_params.OMm * (1.0 + redshifts) ** 3
                    + self.cosmo_params.OMl
                )
            )
            * delta
            * (1.0 + redshifts) ** 3
            * residual_xHI
        )

    def find_n_rescale(self, tau, mean_fluxave_target):
        """Find the rescaling factor so that the mean transmission equal to observations."""
        # Newton-Raphson method
        x = 1
        Ntry = 0
        while np.abs(np.mean(np.exp(-tau * x)) / mean_fluxave_target - 1) > 1e-2:
            f_x = np.mean(np.exp(-tau * x)) - mean_fluxave_target
            f_prime_x = np.min([-1e-10, np.mean(-tau * np.exp(-tau * x))])
            x -= f_x / f_prime_x
            if x < 0:
                x = 0
            Ntry += 1
            if Ntry > 1e3:
                break
                raise RuntimeError("I've tried too many times...", x, f_x, f_prime_x)
        return x

    def build_model_data(self, ctx):
        """Compute all data defined by this core and add it to the context."""
        astro_params, cosmo_params = self._update_params(ctx.getParams())

        lc = ctx.get("lightcone")
        if not lc:
            raise NotImplementedError("A lightcone core is required!")
        lightcone_redshifts = lc.lightcone_redshifts
        lightcone_distances = lc.lightcone_distances
        total_los = lc.user_params.HII_DIM ** 2

        index_right = np.where(
            lightcone_distances
            > (
                lightcone_distances[
                    np.where(lightcone_redshifts > self.redshift[0])[0][0]
                ]
                + self.bin_size / 2
            )
        )[0][0]
        index_left = np.where(
            lightcone_distances
            > (
                lightcone_distances[
                    np.where(lightcone_redshifts > self.redshift[0])[0][0]
                ]
                - self.bin_size / 2
            )
        )[0][0]
        if index_left == 0:
            # TODO here should give a warning!
            index_right = np.where(
                lightcone_distances > (lightcone_distances[0] + self.bin_size)
            )[0][0]

        # select a few number of the los according to the observation
        tau_eff = np.zeros([self.n_realization, self.nlos])

        fbias = 1
        if self.observation == "xqr30":
            index = np.where(np.asarray(lc.node_redshifts) < self.redshift[0])[0][0]
            filling_factor = lc.global_xH[index] * (
                lc.node_redshifts[index - 1] - self.redshift[0]
            ) + lc.global_xH[index - 1] * (self.redshift[0] - lc.node_redshifts[index])
            ctx.add("filling_factor_%s" % self.name, filling_factor)
            if not self.mean_flux:
                if filling_factor > 0.7:
                    fbias = self.fbias_FGPA[7]
                else:
                    index_left = int(filling_factor * 10)
                    index_right = index_left + 1
                    fbias = self.fbias_FGPA[index_left] * (
                        index_right - filling_factor * 10
                    ) + self.fbias_FGPA[index_right] * (
                        filling_factor * 10 - index_left
                    )
            logger.debug(
                "doing xqr30 at z=%.1f with filling factor of %.2f and fbias of %.2f"
                % (self.redshift[0], filling_factor, fbias)
            )

        if not self.mean_flux:
            if not hasattr(ctx.getParams(), "log10_f_rescale"):
                logger.warning(
                    "missing input hyper parameter, log10_f_rescale, assigning 0!"
                )
                f_rescale = 1
            else:
                f_rescale = 10 ** ctx.getParams().log10_f_rescale

            if not hasattr(ctx.getParams(), "f_rescale_slope"):
                logger.warning(
                    "missing input hyper parameter, f_rescale_slope, assigning 0!"
                )
            else:
                f_rescale += (self.redshift[0] - 5.7) * ctx.getParams().f_rescale_slope

        for jj in range(self.n_realization):
            if self.even_spacing:
                gamma_bg = lc.Gamma12_box[:, :, index_left:index_right].reshape(
                    [total_los, index_right - index_left]
                )[jj :: int(total_los / self.nlos)][: self.nlos]
                delta = (
                    lc.density[:, :, index_left:index_right].reshape(
                        [total_los, index_right - index_left]
                    )[jj :: int(total_los / self.nlos)][: self.nlos]
                    + 1.0
                )
                temp = (
                    lc.temp_kinetic_all_gas[:, :, index_left:index_right].reshape(
                        [total_los, index_right - index_left]
                    )[jj :: int(total_los / self.nlos)][: self.nlos]
                    / 1e4
                )
            else:
                rng = np.random.default_rng(self.seed + jj)
                random_index = rng.choice(
                    self.user_params.HII_DIM ** 2, size=self.nlos, replace=False
                )
                random_ii = (random_index / self.user_params.HII_DIM).astype(int)
                random_jj = random_index % self.user_params.HII_DIM
                gamma_bg = lc.Gamma12_box[random_ii, random_jj, index_left:index_right]
                delta = lc.density[random_ii, random_jj, index_left:index_right] + 1.0
                temp = (
                    lc.temp_kinetic_all_gas[
                        random_ii, random_jj, index_left:index_right
                    ]
                    / 1e4
                )

            tau_lyman_alpha = self.tau_GP(
                gamma_bg, delta, temp, lightcone_redshifts[index_left:index_right]
            )

            if self.mean_flux:
                f_rescale = self.find_n_rescale(tau_lyman_alpha, self.mean_flux)

            tau_eff[jj] = -np.log(
                np.mean(np.exp(-tau_lyman_alpha * f_rescale * fbias), axis=1)
            )
        ctx.add("tau_eff_%s" % self.name, tau_eff)
        self.save(ctx)

    def save(self, ctx):
        """Save outputs and astro_params details."""
        lc = ctx.get("lightcone")
        params = ctx.getParams()
        filename = md5(str(params).replace("\n", "").encode()).hexdigest()
        logger.debug(f"parameters: {params}, filename: {filename}")

        with h5py.File("output/run_%s.hdf5" % filename, "a") as f:
            f.create_dataset(
                "tau_eff_%s" % self.name,
                data=ctx.get("tau_eff_%s" % self.name),
                shape=(self.n_realization, self.nlos),
                dtype="float",
            )
            if f.get("node_redshifts") is None:
                f.create_dataset(
                    "node_redshifts",
                    data=lc.node_redshifts,
                    dtype="float",
                )
                f.create_dataset(
                    "global_xH",
                    data=lc.global_xH,
                    dtype="float",
                )
                f.create_dataset(
                    "global_Gamma12",
                    data=lc.global_Gamma12,
                    dtype="float",
                )
                f.create_dataset(
                    "global_MFP",
                    data=lc.global_MFP,
                    dtype="float",
                )
                f.create_dataset(
                    "global_temp_kinetic_all_gas",
                    data=lc.global_temp_kinetic_all_gas,
                    dtype="float",
                )
                f.create_dataset(
                    "global_dNion",
                    data=lc.global_dNion,
                    dtype="float",
                )
                grp = f.create_group("params")
                for kk, v in getattr(lc, "astro_params").__dict__.items():
                    if v is None:
                        continue
                    else:
                        grp.attrs[kk] = v
                    grp.attrs["log10_f_rescale"] = params.log10_f_rescale
                    grp.attrs["f_rescale_slope"] = params.f_rescale_slope

                f.attrs["random_seed"] = ctx.get("lightcone").random_seed
