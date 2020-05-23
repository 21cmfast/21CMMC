"""Module containing 21CMMC likelihoods."""

import logging
from io import IOBase
from os import path
from os import rename

import numpy as np
from cached_property import cached_property
from powerbox.tools import get_power
from scipy.interpolate import InterpolatedUnivariateSpline

from py21cmfast import wrapper as lib

from . import core

logger = logging.getLogger("21cmFAST")

np.seterr(invalid="ignore", divide="ignore")


def _ensure_iter(a):
    try:
        iter(a)
        return a
    except TypeError:
        return [a]


def _listify(lst):
    if type(lst) == list:
        return lst
    else:
        return [lst]


class LikelihoodBase(core.ModuleBase):
    """Base class for Likelihoods in 21CMMC."""

    def computeLikelihood(self, model):
        """
        Calculate the likelihood of the instance data given the model.

        Parameters
        ----------
        model : dict
            A dictionary containing all model-dependent quantities required to calculate
            the likelihood. Explicitly, matches the output of :meth:`~reduce_data`.

        Returns
        -------
        lnL : float
            The log-posterior of the given model.
        """
        raise NotImplementedError("The Base likelihood should never be used directly!")

    def reduce_data(self, ctx):
        """
        Perform reduction on raw data (from all cores in the chain).

        This should perform whatever reduction operations are required on the real/mock data to obtain the form used
        directly in the likelihood. That is, keep in mind that this method will be called on mock data produced
        by the core modules in this chain to obtain the static data which is used in the likelihood. In this regard
        it is called, if applicable, once at :meth:`setup()`. For efficiency then, this method should reduce the data
        as far as possible so that no un-necessary calculations are required per-iteration.

        Furthermore, it should be a deterministic calculation.

        Parameters
        ----------
        ctx ; dict-like
            A context filled with model data from all cores in the chain. Specifically, this is the context obtained
            by using the __call__ method of each core in sequence.

        Returns
        -------
        model : dict
            A dictionary containing all reduced quantities required for obtaining the likelihood.
        """
        raise NotImplementedError("The Base likelihood should never be used directly!")

    def _expose_core_parameters(self):
        # Try to get the params out of the core module
        for m in self._cores:
            for k in ["user_params", "flag_options", "cosmo_params", "astro_params"]:
                if hasattr(m, k):
                    if hasattr(self, k) and getattr(self, k) != getattr(m, k):
                        raise ValueError(
                            f"Setup has detected incompatible input parameter dicts in "
                            f"specified cores: {k}"
                        )
                    else:
                        setattr(self, k, getattr(m, k))

    def setup(self):
        """Perform post-init setup."""
        super().setup()

        # Expose user, flag, cosmo, astro params to this likelihood if available.
        self._expose_core_parameters()

    def get_fiducial_model(self):
        """Compute and return a model dictionary at the fiducial set of parameters."""
        ctx = self.chain.build_model_data()
        return self.reduce_data(ctx)


class LikelihoodBaseFile(LikelihoodBase):
    """Base class for likelihoods whose data is read from a file.

    Parameters
    ----------
    datafile : str, optional
        The file(s) from which to read the data. Alternatively, the file to which to
        write the data. Data will *only* be written to the file if `simulate` is
        set to True explicitly.
    noisefile : str, optional
        The file(s) from which to read the noise profile. If not given, no noise is
        used.
    simulate : bool, optional
        Whether to perform a simulation to create the (mock) data (i.e. instead of
        reading from file). Default is False, which prevents the mock data from
        potentially overwriting the `datafile`.
    use_data : bool, optional
        Sometimes you may want to construct the likelihood without any data at all.
        Set this to False if this is the case.
    """

    _ignore_attributes = LikelihoodBase._ignore_attributes + ("simulate",)

    def __init__(self, datafile=None, noisefile=None, simulate=False, use_data=True):
        super().__init__()
        self.datafile = datafile
        self.noisefile = noisefile
        self._use_data = use_data

        # We *always* make the datafile and noisefile a list
        if isinstance(self.datafile, str) or isinstance(self.datafile, IOBase):
            self.datafile = [self.datafile]
        if isinstance(self.noisefile, str) or isinstance(self.noisefile, IOBase):
            self.noisefile = [self.noisefile]

        self._simulate = simulate

        self.data = None
        self.noise = None

    def setup(self):
        """Perform post-init setup."""
        super().setup()

        if self._use_data:
            if not self._simulate and not self.datafile:
                raise ValueError(
                    "Either an existing datafile has to be specified, or simulate set to True."
                )

            if self._simulate:
                simctx = self.chain.simulate_mock()

            # Read in or simulate the data and noise.
            self.data = (
                self.reduce_data(simctx) if self._simulate else self._read_data()
            )

            # If we can't/won't simulate noise, and no noisefile is provided, assume no
            # noise is necessary.
            if (hasattr(self, "define_noise") and self._simulate) or self.noisefile:
                self.noise = (
                    self.define_noise(simctx, self.data)
                    if (hasattr(self, "define_noise") and self._simulate)
                    else self._read_noise()
                )

            # Now, if data has been simulated, and a file is provided, write to the file.
            if self.datafile and self._simulate:
                self._write_data()

            if self.noisefile and self._simulate and hasattr(self, "define_noise"):
                self._write_noise()

    def _read_data(self):
        data = []
        for fl in self.datafile:
            if not path.exists(fl):
                raise FileNotFoundError(
                    "Could not find datafile: {fl}. If you meant to simulate data, set simulate=True.".format(
                        fl=fl
                    )
                )
            else:
                data.append(dict(np.load(fl, allow_pickle=True)))

        return data

    def _read_noise(self):
        if self.noisefile:
            noise = []
            for fl in self.noisefile:
                if not path.exists(fl):
                    msg = ""
                    if hasattr(self, "define_noise"):
                        msg = "If you meant to simulate noise, set simulate=True."

                    raise FileNotFoundError(
                        "Could not find noisefile: {fl}. {msg}".format(fl=fl, msg=msg)
                    )

                else:
                    noise.append(dict(np.load(fl, allow_pickle=True)))

            return noise

    def _write_data(self):
        for fl, d in zip(self.datafile, self.data):
            if path.exists(fl):
                logger.warning(
                    "File {fl} already exists. Moving previous version to {fl}.bk".format(
                        fl=fl
                    )
                )
                rename(fl, fl + ".bk")

            np.savez(fl, **d)
            logger.info("Saving data file: {fl}".format(fl=fl))

    def _write_noise(self):
        for fl, d in zip(self.noisefile, self.noise):
            if path.exists(fl):
                logger.warning(
                    "File {fl} already exists. Moving previous version to {fl}.bk".format(
                        fl=fl
                    )
                )
                rename(fl, fl + ".bk")

            np.savez(fl, **d)
            logger.info("Saved noise file: {fl}".format(fl=fl))

    def _check_data_format(self):
        pass

    def _check_noise_format(self):
        pass


class Likelihood1DPowerCoeval(LikelihoodBaseFile):
    r"""
    A Gaussian likelihood for the 1D power spectrum of a coeval cube.

    Requires the :class:`~core.CoreCoevalModule` to be loaded to work, and inherently
    deals with the multiple-redshift cubes which that module produces.


    Parameters
    ----------
    datafile : str, optional
        Path to file containing data. See :class:`LikelihoodBaseFile` for details.
    noisefile : str, optional
        Path to file containing noise data. See :class:`LikelihoodBaseFile` for details.
    n_psbins : int, optional
        The number of bins for the spherically averaged power spectrum. By default
        automatically calculated from the number of cells.
    min_k : float, optional
        The minimum k value at which to compare model and data (units 1/Mpc).
    max_k : float, optional
        The maximum k value at which to compare model and data (units 1/Mpc).
    logk : bool, optional
        Whether the power spectrum bins should be regular in logspace or linear space.
    model_uncertainty : float, optional
        The amount of uncertainty in the modelling, per power spectral bin (as
        fraction of the amplitude).
    error_on_model : bool, optional
        Whether the `model_uncertainty` is applied to the model, or the data.
    ignore_kperp_zero : bool, optional
        Whether to ignore the kperp=0 when generating the power spectrum.
    ignore_kpar_zero : bool, optional
        Whether to ignore the kpar=0 when generating the power spectrum.
    ignore_k_zero : bool, optional
        Whether to ignore the ``|k| = 0`` mode when generating the power spectrum.

    Other Parameters
    ----------------
    \*\*kwargs :
        All other parameters sent to :class:`LikelihoodBaseFile`. In particular,
        ``datafile``, ``noisefile``, ``simulate`` and ``use_data``.

    Notes
    -----
    The ``datafile`` and ``noisefile`` have specific formatting required. Both should be
    `.npz` files. The datafile should have 'k' and 'delta' arrays in it (k-modes in
    1/Mpc and power spectrum respectively) and the noisefile should have 'k' and
    'errs' arrays in it (k-modes and their standard deviations respectively). Note
    that the latter is *almost* the default output of 21cmSense, except that
    21cmSense has k in units of h/Mpc, whereas 21cmFAST/21CMMC use units of 1/Mpc.

    .. warning:: Please ensure that the data/noise is in the correct units for
                 21CMMC, as this class does not automatically convert units!

    To make this more flexible, simply subclass this class, and overwrite the
    :meth:`_read_data` or :meth:`_read_noise` methods, then use that likelihood
    instead of this in your likelihood chain. Both of these functions should return
    dictionaries in which the above entries exist. For example::

        >>> class MyCoevalLikelihood(Likelihood1DPowerCoeval):
        >>>    def _read_data(self):
        >>>        data = np.genfromtxt(self.datafile)
        >>>        return {"k": data[:, 0], "p": data[:, 1]}

    Also note that an extra method, ``define_noise`` may be used to define the noise
    properties dynamically (i.e. without reading it). This method will be called if
    available and simulate=True. It should have the signature
    ``define_noise(self, ctx, model)``, where ``ctx`` is the context with all cores
    having added their data, and ``model`` is the output of the :meth:`simulate`
    method.
    """

    required_cores = (core.CoreCoevalModule,)

    def __init__(
        self,
        n_psbins=None,
        min_k=0.1,
        max_k=1.0,
        logk=True,
        model_uncertainty=0.15,
        error_on_model=True,
        ignore_kperp_zero=True,
        ignore_kpar_zero=False,
        ignore_k_zero=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if (
            self.noisefile
            and self.datafile
            and len(self.datafile) != len(self.noisefile)
        ):
            raise ValueError(
                "If noisefile or datafile are provided, they should have the same number "
                "of files (one for each coeval box)"
            )

        self.n_psbins = n_psbins
        self.min_k = min_k
        self.max_k = max_k
        self.logk = logk
        self.error_on_model = error_on_model
        self.model_uncertainty = model_uncertainty
        self.ignore_k_zero = ignore_k_zero
        self.ignore_kperp_zero = ignore_kperp_zero
        self.ignore_kpar_zero = ignore_kpar_zero

    def _check_data_format(self):
        for i, d in enumerate(self.data):
            if "k" not in d or "delta" not in d:
                raise ValueError(
                    f"datafile #{i+1} of {len(self.datafile)} has the wrong format."
                )

    def _check_noise_format(self):
        for i, n in enumerate(self.noise):
            if "k" not in n or "errs" not in n:
                raise ValueError(
                    f"noisefile #{i+1} of {len(self.noise)} has the wrong format"
                )

    def setup(self):
        """Perform post-init setup."""
        super().setup()

        # Ensure that there is one dataset and noiseset per redshift.
        if len(self.data) != len(self.redshift):
            raise ValueError(
                "There needs to be one dataset (datafile) for each redshift!"
            )

        if self.noise and len(self.noise) != len(self.redshift):
            raise ValueError(
                "There needs to be one dataset (noisefile) for each redshift!"
            )

        # Check if all data is formatted correctly.
        self._check_data_format()
        if self.noise:
            self._check_noise_format()

    @cached_property
    def data_spline(self):
        """Splines of data power spectra."""
        return [
            InterpolatedUnivariateSpline(d["k"], d["delta"], k=1) for d in self.data
        ]

    @cached_property
    def noise_spline(self):
        """Splines of noise power spectra."""
        if self.noise:
            return [
                InterpolatedUnivariateSpline(n["k"], n["errs"], k=1) for n in self.noise
            ]
        else:
            return None

    @staticmethod
    def compute_power(
        brightness_temp,
        L,
        n_psbins,
        log_bins=True,
        ignore_kperp_zero=True,
        ignore_kpar_zero=False,
        ignore_k_zero=False,
    ):
        """Compute power spectrum from coeval box.

        Parameters
        ----------
        brightness_temp : :class:`py21cmfast.BrightnessTemp` instance
            The brightness temperature coeval box.
        L : float
            Size of the coeval cube along a side, in Mpc.
        n_psbins : int
            Number of power spectrum bins to return.
        log_bins : bool, optional
            Whether the bins are regular in log-space.
        ignore_kperp_zero : bool, optional
            Whether to ignore perpendicular ``k=0`` modes when performing spherical average.
        ignore_kpar_zero : bool, optoinal
            Whether to ignore parallel ``k=0`` modes when performing spherical average.
        ignore_k_zero : bool, optional
            Whether to ignore the ``|k|=0`` mode when performing spherical average.

        Returns
        -------
        power : ndarray
            The power spectrum as a function of k
        k : ndarray
            The centres of the k-bins defining the power spectrum.
        """
        # Determine the weighting function required from ignoring k's.
        k_weights = np.ones(brightness_temp.shape, dtype=np.int)
        n = k_weights.shape[0]

        if ignore_kperp_zero:
            k_weights[n // 2, n // 2, :] = 0
        if ignore_kpar_zero:
            k_weights[:, :, n // 2] = 0
        if ignore_k_zero:
            k_weights[n // 2, n // 2, n // 2] = 0

        res = get_power(
            brightness_temp,
            boxlength=L,
            bins=n_psbins,
            bin_ave=False,
            get_variance=False,
            log_bins=log_bins,
            k_weights=k_weights,
        )

        res = list(res)
        k = res[1]
        if log_bins:
            k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
        else:
            k = (k[1:] + k[:-1]) / 2

        res[1] = k
        return res

    @property
    def redshift(self):
        """The redshifts of coeval simulations."""
        return self.core_primary.redshift

    def computeLikelihood(self, model):
        """Compute the likelihood given a model.

        Parameters
        ----------
        model : list of dict
            A list of dictionaries, one for each redshift. Exactly the output of
            :meth:'reduce_data`.
        """
        lnl = 0
        noise = 0
        for i, (m, pd) in enumerate(zip(model, self.data_spline)):
            mask = np.logical_and(m["k"] <= self.max_k, m["k"] >= self.min_k)

            moduncert = (
                self.model_uncertainty * pd(m["k"][mask])
                if not self.error_on_model
                else self.model_uncertainty * m["delta"][mask]
            )

            if self.noise_spline:
                noise = self.noise_spline[i](m["k"][mask])

            # TODO: if moduncert depends on model, not data, then it should appear
            #  as -0.5 log(sigma^2) term below.
            lnl += -0.5 * np.sum(
                (m["delta"][mask] - pd(m["k"][mask])) ** 2
                / (moduncert ** 2 + noise ** 2)
            )
        logger.debug("Likelihood computed: {lnl}".format(lnl=lnl))

        return lnl

    def reduce_data(self, ctx):
        """Reduce the data in the context to a list of models (one for each redshift)."""
        brightness_temp = ctx.get("brightness_temp")
        data = []

        for bt in brightness_temp:
            power, k = self.compute_power(
                bt,
                self.user_params.BOX_LEN,
                self.n_psbins,
                log_bins=self.logk,
                ignore_k_zero=self.ignore_k_zero,
                ignore_kpar_zero=self.ignore_kpar_zero,
                ignore_kperp_zero=self.ignore_kperp_zero,
            )
            data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2)})

        return data

    def store(self, model, storage):
        """Save elements of the model to the storage dict.

        Do not use this method -- it is called by the MCMC routine to save data to the
        storage backend.
        """
        # add the power to the written data
        for i, m in enumerate(model):
            storage.update({k + "_z%s" % self.redshift[i]: v for k, v in m.items()})


class Likelihood1DPowerLightcone(Likelihood1DPowerCoeval):
    """
    A likelihood very similar to :class:`Likelihood1DPowerCoeval`, except for a lightcone.

    Since most of the functionality is the same, please see the other documentation for details.
    """

    required_cores = (core.CoreLightConeModule,)

    def __init__(self, *args, nchunks=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.nchunks = nchunks

    def setup(self):
        """Perform post-init setup."""
        LikelihoodBaseFile.setup(self)

        # Ensure that there is one dataset and noiseset per redshift.
        if len(self.data) != self.nchunks:
            raise ValueError(
                "There needs to be one dataset (datafile) for each chunk!!"
            )

        if self.noise and len(self.noise) != self.nchunks:
            raise ValueError(
                "There needs to be one dataset (noisefile) for each chunk!"
            )

        # Check if all data is formatted correctly.
        self._check_data_format()
        if self.noise:
            self._check_noise_format()

    @staticmethod
    def compute_power(
        box,
        length,
        n_psbins,
        log_bins=True,
        ignore_kperp_zero=True,
        ignore_kpar_zero=False,
        ignore_k_zero=False,
    ):
        """Compute power spectrum from coeval box.

        Parameters
        ----------
        box : :class:`py21cmfast.Lightcone` instance
            The lightcone to take the power spectrum of.
        length : 3-tuple
            Size of the lightcone in its 3 dimensions (X,Y,Z)
        n_psbins : int
            Number of power spectrum bins to return.
        log_bins : bool, optional
            Whether the bins are regular in log-space.
        ignore_kperp_zero : bool, optional
            Whether to ignore perpendicular k=0 modes when performing spherical average.
        ignore_kpar_zero : bool, optional
            Whether to ignore parallel k=0 modes when performing spherical average.
        ignore_k_zero : bool, optional
            Whether to ignore the ``|k|=0`` mode when performing spherical average.

        Returns
        -------
        power : ndarray
            The power spectrum as a function of k
        k : ndarray
            The centres of the k-bins defining the power spectrum.
        """
        # Determine the weighting function required from ignoring k's.
        k_weights = np.ones(box.shape, dtype=np.int)
        n0 = k_weights.shape[0]
        n1 = k_weights.shape[-1]

        if ignore_kperp_zero:
            k_weights[n0 // 2, n0 // 2, :] = 0
        if ignore_kpar_zero:
            k_weights[:, :, n1 // 2] = 0
        if ignore_k_zero:
            k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

        res = get_power(
            box,
            boxlength=length,
            bins=n_psbins,
            bin_ave=False,
            get_variance=False,
            log_bins=log_bins,
            k_weights=k_weights,
        )

        res = list(res)
        k = res[1]
        if log_bins:
            k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
        else:
            k = (k[1:] + k[:-1]) / 2

        res[1] = k
        return res

    def reduce_data(self, ctx):
        """Reduce the data in the context to a list of models (one for each redshift chunk)."""
        brightness_temp = ctx.get("lightcone")
        data = []
        chunk_indices = list(
            range(
                0,
                brightness_temp.n_slices,
                round(brightness_temp.n_slices / self.nchunks),
            )
        )

        if len(chunk_indices) > self.nchunks:
            chunk_indices = chunk_indices[:-1]

        chunk_indices.append(brightness_temp.n_slices)

        for i in range(self.nchunks):
            start = chunk_indices[i]
            end = chunk_indices[i + 1]
            chunklen = (end - start) * brightness_temp.cell_size

            power, k = self.compute_power(
                brightness_temp.brightness_temp[:, :, start:end],
                (self.user_params.BOX_LEN, self.user_params.BOX_LEN, chunklen),
                self.n_psbins,
                log_bins=self.logk,
                ignore_kperp_zero=self.ignore_kperp_zero,
                ignore_kpar_zero=self.ignore_kpar_zero,
                ignore_k_zero=self.ignore_k_zero,
            )
            data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2)})

        return data

    def store(self, model, storage):
        """Store the model into backend storage."""
        # add the power to the written data
        for i, m in enumerate(model):
            storage.update({k + "_%s" % i: v for k, v in m.items()})


class LikelihoodPlanck(LikelihoodBase):
    """
    A likelihood which utilises Planck optical depth data.

    In practice, any optical depth measurement (or mock measurement) may be used, by
    defining the class variables ``tau_mean`` and ``tau_sigma``.
    """

    required_cores = ((core.CoreCoevalModule, core.CoreLightConeModule),)

    # Mean and one sigma errors for the Planck constraints
    # The Planck prior is modelled as a Gaussian: tau = 0.058 \pm 0.012
    # (https://arxiv.org/abs/1605.03507)
    tau_mean = 0.058
    tau_sigma = 0.012

    # Simple linear extrapolation of the redshift range provided by the user, to be
    # able to estimate the optical depth
    n_z_interp = 15

    # Minimum of extrapolation is chosen to 5.9, to correspond to the McGreer et al.
    # prior on the IGM neutral fraction.
    # The maximum is chosen to be z = 18., which is arbitrary.
    z_extrap_min = 5.9
    z_extrap_max = 20.0

    def computeLikelihood(self, model):
        """
        Compute the likelihood.

        This is the likelihood arising from Planck (2016) (https://arxiv.org/abs/1605.03507).

        Parameters
        ----------
        model : list of dicts
            Exactly the output of :meth:`simulate`.

        Returns
        -------
        lnl : float
            The log-likelihood for the given model.
        """
        return -0.5 * ((self.tau_mean - model["tau"]) / self.tau_sigma) ** 2

    @property
    def _is_lightcone(self):
        return isinstance(self.core_primary, core.CoreLightConeModule)

    def reduce_data(self, ctx):
        """Reduce the data in the context to a model.

        Returns
        -------
        dict :
            Only key is "tau", the optical depth to reionization.
        """
        # Extract relevant info from the context.

        if self._is_lightcone:

            lc = ctx.get("lightcone")

            redshifts = lc.node_redshifts
            xHI = lc.global_xH

        else:
            redshifts = self.core_primary.redshift
            xHI = [np.mean(x.xH_box) for x in ctx.get("xHI")]

        if len(redshifts) < 3:
            raise ValueError(
                "You cannot use the Planck prior likelihood with less than 3 redshifts"
            )

        # Order the redshifts in increasing order
        redshifts, xHI = np.sort(np.array([redshifts, xHI]))

        # The linear interpolation/extrapolation function, taking as input the redshift
        # supplied by the user and the corresponding neutral fractions recovered for
        # the specific EoR parameter set
        neutral_frac_func = InterpolatedUnivariateSpline(redshifts, xHI, k=1)

        # Perform extrapolation
        z_extrap = np.linspace(self.z_extrap_min, self.z_extrap_max, self.n_z_interp)
        xHI = neutral_frac_func(z_extrap)

        # Ensure that the neutral fraction does not exceed unity, or go negative
        np.clip(xHI, 0, 1, xHI)

        # Set up the arguments for calculating the estimate of the optical depth.
        # Once again, performed using command line code.
        # TODO: not sure if this works.
        tau_value = lib.compute_tau(
            user_params=self.core_primary.user_params,
            cosmo_params=self.core_primary.cosmo_params,
            redshifts=z_extrap,
            global_xHI=xHI,
        )

        return {"tau": tau_value}


class LikelihoodNeutralFraction(LikelihoodBase):
    """
    A likelihood based on the measured neutral fraction at a range of redshifts.

    The log-likelihood statistic is a simple chi^2 if the model has xHI > threshold,
    and 0 otherwise.
    """

    required_cores = ((core.CoreLightConeModule, core.CoreCoevalModule),)
    threshold = 0.06

    def __init__(self, redshift=5.9, xHI=0.06, xHI_sigma=0.05):
        """
        Neutral fraction likelihood/prior.

        Note that the default parameters are based on McGreer et al. constraints
        Modelled as a flat, unity prior at x_HI <= 0.06, and a one sided Gaussian at
        x_HI > 0.06 (Gaussian of mean 0.06 and one sigma of 0.05).

        Limit on the IGM neutral fraction at z = 5.9, from dark pixels by I. McGreer et al.
        (2015) (http://adsabs.harvard.edu/abs/2015MNRAS.447..499M)

        Parameters
        ----------
        redshift : float or list of floats
            Redshift(s) at which the neutral fraction has been measured.
        xHI : float or list of floats
            Measured values of the neutral fraction, corresponding to `redshift`.
        xHI_sigma : float or list of floats
            One-sided uncertainty of measurements.
        """
        self.redshift = _ensure_iter(redshift)
        self.xHI = _ensure_iter(xHI)
        self.xHI_sigma = _ensure_iter(xHI_sigma)

        # By default, setup as if using coeval boxes.
        self.redshifts = (
            []
        )  # these will become the redshifts of all coeval boxes, if that exists.
        self._use_coeval = True
        self._require_spline = False

    @property
    def lightcone_modules(self):
        """All lightcone core modules that are loaded."""
        return [m for m in self._cores if isinstance(m, core.CoreLightConeModule)]

    @property
    def coeval_modules(self):
        """All coeval core modules that are loaded."""
        return [
            m
            for m in self._cores
            if isinstance(m, core.CoreCoevalModule)
            and not isinstance(m, core.CoreLightConeModule)
        ]

    def setup(self):
        """Perform post-init setup."""
        if not self.lightcone_modules + self.coeval_modules:
            raise ValueError(
                "LikelihoodNeutralFraction needs the CoreLightConeModule *or* "
                "CoreCoevalModule to be loaded."
            )

        if not self.lightcone_modules:
            # Get all unique redshifts from all coeval boxes in cores.
            self.redshifts = list(
                set(sum((x.redshift for x in self.coeval_modules), []))
            )

            for z in self.redshift:
                if z not in self.redshifts and len(self.redshifts) < 3:
                    raise ValueError(
                        "To use LikelihoodNeutralFraction, the core must be a lightcone, "
                        "or coeval with >=3 redshifts, or containing the desired redshift"
                    )
                elif z not in self.redshifts:
                    self._require_spline = True

            self._use_coeval = True

        else:
            self._use_coeval = False
            self._require_spline = True

    def reduce_data(self, ctx):
        """Return a dictionary of model quantities from the context."""
        if self._use_coeval:
            xHI = np.array([np.mean(x) for x in ctx.get("xHI")])
            redshifts = self.redshifts
        else:
            xHI = ctx.get("lightcone").global_xH
            redshifts = ctx.get("lightcone").node_redshifts

        redshifts, xHI = np.sort([redshifts, xHI])
        return {"xHI": xHI, "redshifts": redshifts}

    def computeLikelihood(self, model):
        """Compute the likelihood."""
        lnprob = 0

        if self._require_spline:
            model_spline = InterpolatedUnivariateSpline(
                model["redshifts"], model["xHI"], k=1
            )

        for z, data, sigma in zip(self.redshift, self.xHI, self.xHI_sigma):
            if z in model["redshifts"]:
                lnprob += self.lnprob(
                    model["xHI"][self.redshifts.index(z)], data, sigma
                )
            else:
                lnprob += self.lnprob(model_spline(z), data, sigma)

        return lnprob

    def lnprob(self, model, data, sigma):
        """Compute the log prob given a model, data and error."""
        model = np.clip(model, 0, 1)

        if model > self.threshold:
            return -0.5 * ((data - model) / sigma) ** 2
        else:
            return 0


class LikelihoodGreig(LikelihoodNeutralFraction, LikelihoodBaseFile):
    """Likelihood using QSOs.

    See :class:`LikelihoodNeutralFraction` and :class:`LikelihoodBaseFile` for
    parameter descriptions.
    """

    qso_redshift = 7.0842  # The redshift of the QSO

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Read in data files.
        nf = np.load(
            path.join(path.dirname(__file__), "data", "NeutralFractionsForPDF.npy")
        )
        pdf = np.load(
            path.join(path.dirname(__file__), "data", "NeutralFractionPDF_SmallHII.npy")
        )
        # Normalising the PDF to have a peak probability of unity (consistent with
        # how other priors are treated). Ultimately, this step does not matter
        pdf /= np.amax(pdf)

        # Interpolate the QSO damping wing PDF
        # Interpolate in log space because the pdf must be greater than zero.
        self.spline_qso_damping_pdf = InterpolatedUnivariateSpline(
            nf[pdf > 0], np.log(pdf[pdf > 0])
        )

    def computeLikelihood(self, model):
        """
        Compute the likelihood.

        Constraints on the IGM neutral fraction at z = 7.1 from the IGM damping wing of
        ULASJ1120+0641. Greig et al (2016) (http://arxiv.org/abs/1606.00441).
        """
        redshifts = model["redshifts"]
        ave_nf = model["xHI"]

        if self.qso_redshift in redshifts:
            nf_qso = redshifts.index(self.qso_redshift)

        elif len(redshifts) > 2:

            # Check the redshift range input by the user to determine whether to
            # interpolate or extrapolate the IGM neutral fraction to the QSO redshift
            if self.qso_redshift < np.min(redshifts):
                # The QSO redshift is outside the range set by the user. Need to
                # extrapolate the reionisation history to obtain the neutral fraction at
                # the QSO redshift

                # The linear interpolation/extrapolation function, taking as input the
                # redshift supplied by the user and the corresponding neutral fractions
                # recovered for the specific EoR parameter set
                global_nf_spl = InterpolatedUnivariateSpline(redshifts, ave_nf, k=1)

            else:
                # The QSO redshift is within the range set by the user. Can interpolate
                # the reionisation history to obtain the neutral fraction at the QSO redshift
                global_nf_spl = InterpolatedUnivariateSpline(
                    redshifts, ave_nf, k=2 if len(redshifts) == 3 else 3
                )

            nf_qso = global_nf_spl(self.qso_redshift)
        else:
            raise ValueError(
                "You cannot use the Greig prior likelihood with either less than 3 "
                "redshifts or the redshift being directly evaluated."
            )

        # Ensure that the neutral fraction does not exceed unity, or go negative
        nf_qso = np.clip(nf_qso, 0, 1)

        # un-log the spline.
        qso_prob = np.exp(self.spline_qso_damping_pdf(nf_qso))

        # We work with the log-likelihood, therefore convert the IGM Damping wing PDF to
        # log space
        return np.log(qso_prob)


class LikelihoodGlobalSignal(LikelihoodBaseFile):
    """Chi^2 likelihood of Global Signal, where global signal is in mK as a function of MHz."""

    required_cores = (core.CoreLightConeModule,)

    def reduce_data(self, ctx):
        """Reduce data to model."""
        return {
            "frequencies": 1420.4
            / (np.array(ctx.get("lightcone").node_redshifts) + 1.0),
            "global_signal": ctx.get("lightcone").global_brightness_temp,
        }

    def computeLikelihood(self, model):
        """Compute the likelihood, given the lightcone output from 21cmFAST."""
        model_spline = InterpolatedUnivariateSpline(
            model["frequencies"], model["global_signal"]
        )

        lnl = -0.5 * np.sum(
            (self.data["global_signal"] - model_spline(self.data["frequencies"])) ** 2
            / self.noise["sigma"] ** 2
        )

        return lnl


class LikelihoodLuminosityFunction(LikelihoodBaseFile):
    r"""
    Likelihood based on Chi^2 comparison to luminosity function data.

    Parameters
    ----------
    datafile : str, optional
        Input data should be in a `.npz` file, and contain the arrays:
        * ``Muv``: the brightness magnitude array
        * ``lfunc``: the number density of galaxies at each ``Muv`` bin.
        Each of these arrays can be either 1D or 2D. If 1D, they will be
        interpreted to be arrays over ``Muv``. If 2D, first axis will be
        interpreted to be redshift. If you require each luminosity function
        at different redshifts to have different numbers of Muv bins, you
        should create multiple files, and create a separate core/likelihood
        instance pair for each, pairing them by ``name``.
        A set of default LFs (z=6,7,8,10; Bouwens+15,16; Oesch+17) are provided
        in the folder ``data`` where datafile and noisefile (see below) are named
        LF_lfuncs_z*.npz and LF_sigmas_z*.npz. To use these files, a separate
        core/likelihood instance pair for each redshift is required.
    noisefile : str, optional
        Noise should be a `.npz` file with a single array 'sigma` which gives the
        error at each of the `Muv` bins in the `datafile`. If 1D, it must have the
        same length as ``Muv``. If 2D, must have the same length as the number
        of redshifts as the first dimension.
    mag_brightest : float, optional
        Brightest magnitude when calculating the likelihood. Default is -20.
    name : str, optional
        A name for the instance. This is used to pair it with a particular core
        instance.
    """

    required_cores = (core.CoreLuminosityFunction,)

    def __init__(self, *args, name="", mag_brightest=-20.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.datafile is not None and len(self.datafile) != 1:
            raise ValueError(
                "can only pass a single datafile to LikelihoodLuminosityFunction!"
            )
        if self.noisefile is not None and len(self.noisefile) != 1:
            raise ValueError(
                "can only pass a single noisefile to LikelihoodLuminosityFunction!"
            )

        self.name = str(name)
        self.mag_brightest = mag_brightest

    def setup(self):
        """Setup instance."""
        if not self._simulate:
            if self.datafile is None:
                if len(self.redshifts) != 1:
                    raise ValueError(
                        "to use the provided LFs, a separate core/likelihood instance pair for each redshift is required!"
                    )
                if self.redshifts[0] not in [6, 7, 8, 10]:
                    raise ValueError(
                        "only LFs at z=6,7,8 and 10 are provided! use your own LF :)"
                    )
                self.datafile = [
                    path.join(
                        path.dirname(__file__),
                        "data",
                        "LF_lfuncs_z%d.npz" % self.redshifts[0],
                    )
                ]
            if self.noisefile is None:
                self.noisefile = [
                    path.join(
                        path.dirname(__file__),
                        "data",
                        "LF_sigmas_z%d.npz" % self.redshifts[0],
                    )
                ]
        super().setup()

        # We only allow one datafile, so get the data out of it
        # to make it easier to work with.
        self.data = self.data[0]
        if self.noise is not None:
            self.noise = self.noise[0]

    def _read_data(self):
        data = super()._read_data()

        # data only has one entry (only one file allowed)
        # if that entry has values with one dimension, add a first
        # dimension as a redshift dimension of size 1.
        for key, val in data[0].items():
            if not hasattr(val[0], "__len__"):
                data[0][key] = np.atleast_2d(val)

        return data

    def _read_noise(self):
        noise = super()._read_noise()

        # data only has one entry (only one file allowed)
        # if that entry has values with one dimension, add a first
        # dimension as a redshift dimension of size 1.
        for key, val in noise[0].items():
            if not hasattr(val[0], "__len__"):
                noise[0][key] = np.atleast_2d(val)

        return noise

    @cached_property
    def paired_core(self):
        """The luminosity function core that is paired with this likelihood."""
        paired = []
        for c in self._cores:
            if isinstance(c, core.CoreLuminosityFunction) and c.name == self.name:
                paired.append(c)
        if len(paired) > 1:
            raise ValueError(
                "You've got more than one CoreLuminosityFunction with the same name -- they will overwrite each other!"
            )
        return paired[0]

    @property
    def redshifts(self):
        """Redshifts at which luminosity function is defined."""
        return self.paired_core.redshift

    def reduce_data(self, ctx):
        """Reduce simulated model data."""
        lfunc = ctx.get("luminosity_function" + self.name)
        if not self._is_setup:
            # During setup, return a list, so that it can be matched with the
            # list of length one of datafile to be written.
            return [lfunc]
        else:
            return lfunc

    def computeLikelihood(self, model):
        """Compute the likelihood."""
        lnl = 0
        for i, z in enumerate(self.redshifts):
            model_spline = InterpolatedUnivariateSpline(
                model["Muv"][i][::-1], model["lfunc"][i][::-1]
            )
            lnl += -0.5 * np.sum(
                (
                    (self.data["lfunc"][i] - 10 ** model_spline(self.data["Muv"][i]))
                    ** 2
                    / self.noise["sigma"][i] ** 2
                )[self.data["Muv"][i] > self.mag_brightest]
            )
        return lnl

    def define_noise(self, ctx, model):
        """Define noise properties."""
        sig = self.paired_core.sigma

        if callable(sig[0]):
            return [{"sigma": [s(m["Muv"]) for s, m in zip(sig, model)]}]
        else:
            return [{"sigma": sig}]


class LikelihoodEDGES(LikelihoodBaseFile):
    """
    A likelihood based on chi^2 comparison to Global Signal of EDGES timing and fwhm.

    This is the likelihood arising from Bowman et al. (2018), which reports an absorption feature
    in the 21-cm brightness temperature spectra

    Parameters
    ----------
    use_width : bool
        whether to use the fwhm in the likelihood, by default it's False
    """

    freq_edges = 78.0
    freq_err_edges = 1.0
    fwhm_edges = 19.0
    fwhm_err_upp_edges = 4.0
    fwhm_err_low_edges = 2.0

    required_cores = (core.CoreLightConeModule,)

    def __init__(self, use_width=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_width = use_width

    def reduce_data(self, ctx):
        """Reduce data to model."""
        frequencies = 1420.4 / (np.array(ctx.get("lightcone").node_redshifts) + 1)
        global_signal = ctx.get("lightcone").global_brightness_temp
        global_signal_interp = InterpolatedUnivariateSpline(
            frequencies, global_signal, k=4
        )
        cr_pts = global_signal_interp.derivative().roots()
        cr_vals = global_signal_interp(cr_pts)
        results = {}
        if len(cr_vals) == 0:
            # there is no solution -- global signal never reaches a minimum
            results["freq_tb_min"] = None
            results["fwhm"] = None
        else:
            freq_tb_min = cr_pts[np.argmin(cr_vals)]
            results["freq_tb_min"] = freq_tb_min
            if not self.use_width:
                results["fwhm"] = None
            # calculating the frequencies when global signal = the fwhm
            freqs_hm = InterpolatedUnivariateSpline(
                frequencies,
                global_signal - global_signal_interp(freq_tb_min) * 0.5,
                k=3,
            ).roots()
            if len(freqs_hm) == 2:
                # there are two of them, one is the lower bound of the fwhm and the
                # other one is the upper
                freq_r = freqs_hm[1]
                freq_l = freqs_hm[0]
            elif len(freqs_hm) == 1:
                # therea are only one of them
                if freqs_hm[0] > freq_tb_min:
                    # it's larger than the frequency of the minimum, so it's the upper
                    # bound of the fwhm then use the boundary to be the lower one
                    freq_r = freqs_hm[0]
                    freq_l = frequencies[0]
                else:
                    # it's smaller than the frequency of the minimum, so it's the lower
                    # bound of the fwhm then use the boundary to be the upper one
                    freq_l = freqs_hm[0]
                    freq_r = frequencies[-1]
            elif len(freqs_hm) > 2:
                # therea are more two of them, need to find the closest two
                freq_1 = freqs_hm[np.argmin(np.abs(freqs_hm - freq_tb_min))]
                if freq_1 < freq_tb_min:
                    # the closest one is smaller than the frequency of the minimum, so
                    # it's the lower bound of the fwhm
                    freq_l = freq_1
                    freq_rs = freqs_hm[freqs_hm > freq_tb_min]
                    # find the rest which are larger than the frequency of the minimum
                    if len(freq_rs) > 0:
                        # the smallest should be the upper bound of the fwhm
                        freq_r = freq_rs[0]
                    else:
                        # if none, use the boundary
                        freq_r = frequencies[-1]
                else:
                    # the closest one is larger than the frequency of the minimum, so
                    # it's the upper bound of the fwhm
                    freq_r = freq_1
                    freq_ls = freqs_hm[freqs_hm < freq_tb_min]
                    # find the rest which are smaller than the frequency of the minimum
                    if len(freq_ls) > 0:
                        # the largest should be the lower bound of the fwhm
                        freq_l = freq_ls[-1]
                    else:
                        # if none, use the boundary
                        freq_l = frequencies[0]
            if len(freqs_hm) == 0:
                results["fwhm"] = None
            else:
                results["fwhm"] = freq_r - freq_l
        return results

    def computeLikelihood(self, model):
        """
        Compute the likelihood, given the lightcone output from 21cmFAST.

        Parameters
        ----------
        model : list of dicts
            Exactly the output of :meth:`simulate`.

        Returns
        -------
        lnl : float
            The log-likelihood for the given model.
        """
        if model["freq_tb_min"] is None:
            return -np.inf

        if self.use_width:
            if model["fwhm"] is None:
                return -np.inf
            else:
                # asymmetric uncertainty follows approximation in Barlow04, Sec 3.6
                denominator = self.fwhm_err_upp_edges * self.fwhm_err_low_edges + (
                    self.fwhm_err_upp_edges - self.fwhm_err_low_edges
                ) * (model["fwhm"] - self.fwhm_edges)
                if denominator <= 0:
                    return -np.inf
                else:
                    return (
                        -0.5
                        * np.square(
                            (model["freq_tb_min"] - self.freq_edges)
                            / self.freq_err_edges
                        )
                        + -0.5
                        * np.square(model["fwhm"] - self.fwhm_edges)
                        / denominator
                    )
        else:
            return -0.5 * np.square(
                (model["freq_tb_min"] - self.freq_edges) / self.freq_err_edges
            )
