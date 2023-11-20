"""Module containing 21CMMC likelihoods."""
import logging
import numpy as np
from cached_property import cached_property
from io import IOBase
from os import path, rename
from pathlib import Path
from powerbox.tools import get_power
from py21cmfast import wrapper as lib
from scipy.interpolate import (
    InterpolatedUnivariateSpline,
    RectBivariateSpline,
    interp1d,
)
from scipy.special import erf

from . import core

loaded_cliks = {}
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
    emulate : bool, optional
        Default is False
    use_data : bool, optional
        Sometimes you may want to construct the likelihood without any data at all.
        Set this to False if this is the case.
    """

    _ignore_attributes = LikelihoodBase._ignore_attributes + ("simulate",)

    def __init__(
        self,
        datafile=None,
        noisefile=None,
        simulate=False,
        emulate=False,
        use_data=True,
    ):
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
        self._emulate = emulate

        self.data = None
        self.noise = None

    def setup(self):
        """Perform post-init setup."""
        super().setup()

        if self._use_data:
            if not self._simulate and not self.datafile and not self._emulate:
                raise ValueError(
                    "Either an existing datafile has to be specified, or simulate or emulate set to True."
                )

            if self._simulate or self._emulate:
                simctx = self.chain.simulate_mock()

            # Read in or simulate the data and noise.
            self.data = (
                self.reduce_data(simctx)
                if (self._simulate or self._emulate)
                else self._read_data()
            )

            # If we can't/won't simulate noise, and no noisefile is provided, assume no
            # noise is necessary.
            # If emulate is True, this part includes the avg emulator error.
            if (hasattr(self, "define_noise") and self._simulate) or self.noisefile:
                self.noise = (
                    self.define_noise(simctx, self.data)
                    if (hasattr(self, "define_noise") and self._simulate)
                    else self._read_noise()
                )

            # Now, if data has been simulated, and a file is provided, write to the file.
            if self.datafile and (self._simulate or self._emulate):
                self._write_data()

            if (
                self.noisefile
                and (self._simulate or self._emulate)
                and hasattr(self, "define_noise")
            ):
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
                    try:
                        noise.append(dict(np.load(fl, allow_pickle=True)))
                    except ValueError:
                        # TODO: this one is for reading the error covariance matrix, need to better deal with it
                        noise.append(np.load(fl, allow_pickle=True))

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

    required_cores = ((core.CoreCoevalModule, core.Core21cmEMU),)

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
        name="",
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
        self.name = name

    def _check_data_format(self):
        for i, d in enumerate(self.data):
            if ("k" not in d and "kwfband8" not in d) or (
                "delta" not in d and "band8" not in d
            ):
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
        k_weights = np.ones(brightness_temp.shape, dtype=int)
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
        if isinstance(self.paired_core, core.Core21cmEMU):
            return self.data[0]["z_bands"]
        else:
            return self.core_primary.redshift

    def computeLikelihood(self, model):
        """Compute the likelihood given a model.

        Parameters
        ----------
        model : list of dict
            A list of dictionaries, one for each redshift. Exactly the output of
            :meth:'reduce_data`.
        """
        if isinstance(self.paired_core, core.Core21cmEMU):
            N = len(model)
            if N > 1:
                lnl = np.zeros(N)
            else:
                lnl = 0
            hera_data = self.data[0]
            all_band_keys = []
            for z in self.redshift:
                for key in list(hera_data.keys()):
                    if "band" in key and "wf" not in key and "k" not in key and str(int(np.round(z))) in key:
                        all_band_keys.append(key)
            for i in range(N):
                for j, (band, band_key) in enumerate(zip(self.redshift, all_band_keys)):
                    nfields = hera_data[band_key].shape[0]
                    for field in range(nfields):
                        PS_limit_ks = hera_data[band_key][field, :, 0]
                        PS_limit_ks = PS_limit_ks[~np.isnan(PS_limit_ks)]
                        Nkbins = len(PS_limit_ks)
                        PS_limit_vals = hera_data[band_key][field, :Nkbins, 1]
                        PS_limit_vars = hera_data[band_key][field, :Nkbins, 2]

                        kwf_limit_vals = hera_data["kwf" + band_key]
                        Nkwfbins = len(kwf_limit_vals)
                        PS_limit_wfcs = hera_data["wf" + band_key][field, :Nkbins, :]

                        PS_limit_wfcs = PS_limit_wfcs.reshape([Nkbins, Nkwfbins])

                        ModelPS_val = model[i][j]["delta"][:Nkwfbins]

                        ModelPS_val_afterWF = np.dot(PS_limit_wfcs, ModelPS_val)
                        # Include emulator error term if present
                        if "delta_err" in model[i][j].keys():
                            ModelPS_val_1sigma_upper_afterWF = np.dot(
                                PS_limit_wfcs,
                                ModelPS_val + model[i][j]["delta_err"][:Nkwfbins],
                            )
                            ModelPS_val_1sigma_lower_afterWF = np.dot(
                                PS_limit_wfcs,
                                ModelPS_val - model[i][j]["delta_err"][:Nkwfbins],
                            )
                            # The upper and lower errors are very similar usually, so we can just take the mean and use that.
                            mean_err = np.mean(
                                [
                                    ModelPS_val_1sigma_upper_afterWF
                                    - ModelPS_val_afterWF,
                                    ModelPS_val_afterWF
                                    - ModelPS_val_1sigma_lower_afterWF,
                                ],
                                axis=0,
                            )
                            error_val = np.sqrt(
                                PS_limit_vars
                                + (0.2 * ModelPS_val_afterWF) ** 2
                                + (mean_err) ** 2
                            )
                        else:
                            error_val = np.sqrt(
                                PS_limit_vars + (0.2 * ModelPS_val_afterWF) ** 2
                            )
                        if N > 1:
                            lnl[i] += -0.5 * np.sum(
                                (ModelPS_val_afterWF - PS_limit_vals) ** 2
                                / (error_val**2)
                            )

                        else:
                            lnl += -0.5 * np.sum(
                                (ModelPS_val_afterWF - PS_limit_vals) ** 2
                                / (error_val**2)
                            )
        else:
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
                    / (moduncert**2 + noise**2)
                )
        logger.debug("Likelihood computed: {lnl}".format(lnl=lnl))

        return lnl

    def reduce_data(self, ctx):
        """Reduce the data in the context to a list of models (one for each redshift)."""
        data = []
        if isinstance(self.paired_core, core.Core21cmEMU):
            # Interpolate the data onto the HERA bands and ks
            if len(ctx.get("PS").shape) > 2:
                N = ctx.get("PS").shape[0]
            else:
                N = 1
            for j in range(N):
                tmp_data = []
                for i in range(self.redshift.shape[0]):
                    interp_ks = self.k[i]
                    tmp_data.append(
                        {
                            "k": interp_ks,
                            "delta": RectBivariateSpline(
                                ctx.get("PS_redshifts"),
                                ctx.get("k"),
                                ctx.get("PS")[j, ...] if N > 1 else ctx.get("PS"),
                            )(self.redshift[i], interp_ks)[0],
                            "delta_err": RectBivariateSpline(
                                ctx.get("PS_redshifts"),
                                ctx.get("k"),
                                ctx.get("PS_err"),
                            )(self.redshift[i], interp_ks)[0],
                        }
                    )
                data.append(tmp_data)

        else:
            brightness_temp = ctx.get("brightness_temp")

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
                data.append({"k": k, "delta": power * k**3 / (2 * np.pi**2)})

        return data

    def store(self, model, storage):
        """Save elements of the model to the storage dict.

        Do not use this method -- it is called by the MCMC routine to save data to the
        storage backend.
        """
        # add the power to the written data
        for i, m in enumerate(model):
            storage.update({k + "_z%s" % self.redshift[i]: v for k, v in m.items()})

    @cached_property
    def paired_core(self):
        """The PS core that is paired with this likelihood."""
        paired = []
        for c in self._cores:
            if isinstance(c, core.Core21cmEMU) and c.name == self.name:
                paired.append(c)
            else:
                if isinstance(c, core.CoreCoevalModule) or isinstance(
                    c, core.CoreCoevalModule
                ):
                    paired.append(c)
        if len(paired) > 1:
            raise ValueError(
                "You've got more than one CoreCoevalModule / Core21cmEMU with the same name -- they will overwrite each other!"
            )
        return paired[0]


class Likelihood1DPowerLightcone(Likelihood1DPowerCoeval):
    """
    A likelihood very similar to :class:`Likelihood1DPowerCoeval`, except for a lightcone.

    This likelihood is vectorized i.e., it accepts an array of ``astro_params``,
    but only when used with ``Core21cmEMU``.

    Since most of the functionality is the same, please see the other documentation for details.
    """

    required_cores = ((core.CoreLightConeModule, core.Core21cmEMU),)

    def __init__(self, *args, datafile="", nchunks=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.nchunks = nchunks
        self.datafile = [datafile] if isinstance(datafile, (str, Path)) else datafile

    @classmethod
    def from_builtin_data(cls, datafile="", **kwargs):
        """Create the class loading in built-in data."""
        datafile = path.join(path.dirname(__file__), "data", datafile + ".npz")

        return cls(datafile=datafile, **kwargs)

    def setup(self):
        """Perform post-init setup."""
        LikelihoodBaseFile.setup(self)
        if isinstance(self.paired_core, core.Core21cmEMU):
            all_keys = np.array(list(self.data[0].keys()))
            m = ["kwf" in i for i in all_keys]
            all_kwfs_keys = all_keys[m]
            self.k = [self.data[0][j] for j in all_kwfs_keys]
            self.k_len = max(len(i) for i in self.k)
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
        k_weights = np.ones(box.shape, dtype=int)
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
        data = []
        if isinstance(self.paired_core, core.Core21cmEMU):
            # Interpolate the data onto the HERA bands and ks
            if len(ctx.get("PS").shape) > 2:
                N = ctx.get("PS").shape[0]
            else:
                N = 1

            for j in range(N):
                tmp_data = []
                for i in range(self.redshift.shape[0]):
                    interp_ks = self.k[i]
                    tmp_data.append(
                        {
                            "k": interp_ks,
                            "delta": RectBivariateSpline(
                                ctx.get("PS_redshifts"),
                                ctx.get("k"),
                                ctx.get("PS")[j, ...] if N > 1 else ctx.get("PS"),
                            )(self.redshift[i], interp_ks)[0],
                            "delta_err": RectBivariateSpline(
                                ctx.get("PS_redshifts"),
                                ctx.get("k"),
                                ctx.get("PS_err"),
                            )(self.redshift[i], interp_ks)[0],
                        }
                    )
                data.append(tmp_data)

        else:
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
                data.append({"k": k, "delta": power * k**3 / (2 * np.pi**2)})

        return data

    def store(self, model, storage):
        """Store the model into backend storage."""
        # add the power to the written data
        for i, m in enumerate(model):
            if isinstance(self.paired_core, core.Core21cmEMU):
                if isinstance(m, list):
                    for j, n in enumerate(m):
                        storage.update({k + "_%s" % j: v for k, v in n.items()})
            else:
                storage.update({k + "_%s" % i: v for k, v in m.items()})

    @cached_property
    def paired_core(self):
        """The PS core that is paired with this likelihood."""
        paired = []
        for c in self._cores:
            if (isinstance(c, core.Core21cmEMU) and c.name == self.name) or (
                isinstance(c, core.CoreLightConeModule) and c.name == self.name
            ):
                paired.append(c)
        if len(paired) > 1:
            raise ValueError(
                "You've got more than one CoreCoevalModule / Core21cmEMU with the same name -- they will overwrite each other!"
            )

        return paired[0]


class LikelihoodPlanckPowerSpectra(LikelihoodBase):
    r"""A likelihood template to use Planck power spectrum.

    It makes use of the clik code developed by the planck collaboration.
    Go to the planck legacy archive website to download the code and data and get information: https://pla.esac.esa.int
    You should download the "baseline" data package.
    This likelihood currently supports Planck_lensing, Planck_highl_TTTEEE and Planck_lowl_EE data. Could easily be extended if required.

    Parameters
    ----------
    name_lkl : str
        the planck likelihood to compute. choice: Planck_lensing, Planck_highl_TTTEEE, Planck_lowl_EE
    """

    required_cores = ((core.CoreLightConeModule, core.CoreCMB),)

    def __init__(
        self,
        *args,
        name_lkl=None,
        A_planck_prior_center=1,
        A_planck_prior_variance=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name_lkl
        self.A_planck_prior_center = A_planck_prior_center
        self.A_planck_prior_variance = A_planck_prior_variance

        if "Planck_lensing" in self.name:
            self.lensing = True
        else:
            self.lensing = False
        if "Planck_highl_TTTEEE" in self.name:
            self.TTTEEE = True
        else:
            self.TTTEEE = False
        if "Planck_lowl_EE" in self.name:
            self.EE = True
        else:
            self.EE = False
        if not self.TTTEEE and not self.EE and not self.lensing:
            raise AttributeError(
                "I did not understand name %s" % (self.name)
                + " please choose between "
                + "Planck_lensing, Planck_highl_TTTEEE, Planck_lowl_EE"
            )

        self.initialize = True
        if self.initialize:
            self.initialize_clik_and_class(self.name)

    def reduce_data(self, ctx):
        """Get the CMB power spectra and the nuisance parameter A_planck from the coreCMB."""
        cl = ctx.get("cl_cmb")
        A_planck = ctx.get("A_planck", 1)
        data = {"cl_cmb": cl, "A_planck_cmb": A_planck}

        return data

    def computeLikelihood(self, model):
        """
        Compute the likelihood.

        This is the likelihood arising from Planck Lite (2018).

        Parameters
        ----------
        cosmo : contains cosmological observables computed with CLASS
            Exactly the output of CLASS

        Returns
        -------
        lnl : float
            The log-likelihood for the given model.
        """
        cl = model["cl_cmb"]

        if self.lensing:
            # these my_clik * are global variables defined in initialize_clik_and_class.
            my_clik = my_clik_lensing
            my_l_max = my_l_max_lensing
        if self.TTTEEE:
            my_clik = my_clik_TTTEEE
            my_l_max = my_l_max_TTTEEE
        if self.EE:
            my_clik = my_clik_EE
            my_l_max = my_l_max_EE
        if self.lensing:
            try:
                length = len(my_clik.get_lmax())
                tot = np.zeros(
                    np.sum(my_clik.get_lmax())
                    + length
                    + len(my_clik.get_extra_parameter_names())
                )
            # following 3 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                length = 2
                tot = np.zeros(
                    2 * my_l_max + length + len(my_clik.get_extra_parameter_names())
                )
        else:
            length = len(my_clik.get_has_cl())
            tot = np.zeros(
                np.sum(my_clik.get_lmax())
                + length
                + len(my_clik.get_extra_parameter_names())
            )
        #
        # fill with Cl's
        index = 0
        if not self.lensing:
            for i in range(length):
                if my_clik.get_lmax()[i] > -1:
                    for j in range(my_clik.get_lmax()[i] + 1):
                        if i == 0:
                            tot[index + j] = cl["tt"][j]
                        elif i == 1:
                            tot[index + j] = cl["ee"][j]
                        elif i == 2:
                            tot[index + j] = cl["bb"][j]
                        elif i == 3:
                            tot[index + j] = cl["te"][j]
                        elif i == 4:
                            tot[index + j] = 0  # cl['tb'][j] class does not compute tb
                        elif i == 5:
                            tot[index + j] = 0  # cl['eb'][j] class does not compute eb

                    index += my_clik.get_lmax()[i] + 1

        else:
            try:
                for i in range(length):
                    if my_clik.get_lmax()[i] > -1:
                        for j in range(my_clik.get_lmax()[i] + 1):
                            if i == 0:
                                tot[index + j] = cl["pp"][j]
                            elif i == 1:
                                tot[index + j] = cl["tt"][j]
                            elif i == 2:
                                tot[index + j] = cl["ee"][j]
                            elif i == 3:
                                tot[index + j] = cl["bb"][j]
                            elif i == 4:
                                tot[index + j] = cl["te"][j]
                            elif i == 5:
                                tot[
                                    index + j
                                ] = 0  # cl['tb'][j] class does not compute tb
                            elif i == 6:
                                tot[
                                    index + j
                                ] = 0  # cl['eb'][j] class does not compute eb

                        index += my_clik.get_lmax()[i] + 1

            # following 8 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                for i in range(length):
                    for j in range(my_l_max):
                        if i == 0:
                            tot[index + j] = cl["pp"][j]
                        if i == 1:
                            tot[index + j] = cl["tt"][j]
                    index += my_l_max + 1
        # fill with nuisance parameters
        A_planck = model["A_planck_cmb"]
        tot[index] = A_planck
        index += 1

        lkl = my_clik(tot)[0]  # -loglike

        # add nuisance parameter A_planck
        lkl += (
            -0.5
            * ((A_planck - self.A_planck_prior_center) / self.A_planck_prior_variance)
            ** 2
        )

        return lkl

    def initialize_clik_and_class(self, name=None):
        """Initialize clik and class."""
        global my_clik_TTTEEE, my_clik_lensing, my_clik_EE, my_l_max_lensing, my_l_max_EE, my_l_max_TTTEEE
        self.initialize = False

        try:
            import clik

        except ModuleNotFoundError:
            raise ImportError(
                "You must first activate the binaries from the Clik "
                + "distribution. Please run : \n "
                + "]$ source /path/to/clik/bin/clik_profile.sh \n "
                + "and try again."
            )

        my_path = path.join(
            "%s/.ccode/baseline/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik"
            % path.expanduser("~")
        )
        if not path.isdir(my_path):
            import tarfile
            from astropy.utils.data import download_file

            tarfile.open(
                download_file(
                    "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz",
                )
            ).extractall(path.expanduser("~/.ccode"))

        try:
            if self.lensing:
                my_clik_lensing = clik.clik_lensing(my_path)
                try:
                    my_l_max_lensing = max(my_clik_lensing.get_lmax())
                # following 2 lines for compatibility with lensing likelihoods of 2013 and before
                # (then, clik.get_lmax() just returns an integer for lensing likelihoods;
                # this behavior was for clik versions < 10)
                except:
                    my_l_max_lensing = my_clik_lensing.get_lmax()
            elif self.TTTEEE:
                my_clik_TTTEEE = clik.clik(my_path)
                my_l_max_TTTEEE = max(my_clik_TTTEEE.get_lmax())
            elif self.EE:
                my_clik_EE = clik.clik(my_path)
                my_l_max_EE = max(my_clik_EE.get_lmax())
            else:
                raise AttributeError(
                    f"I did not understand name {name}"
                    + "please choose between"
                    + "Planck_lensing, Planck_highl_TTTEEE, Planck_lowl_EE"
                )
        except AttributeError:
            raise AttributeError(
                "The path to the .clik file for the likelihood "
                "%s was not found where indicated:\n%s\n" % (name, my_path)
                + " Note that the default path to search for it is"
                " one directory above the path['clik'] field. You"
                " can change this behaviour in all the "
                "Planck_something.data, to reflect your local configuration, "
                "or alternatively, move your .clik files to this place."
            )


class LikelihoodPlanck(LikelihoodBase):
    """
    A likelihood which utilises Planck optical depth data.

    In practice, any optical depth measurement (or mock measurement) may be used, by
    defining the class variables ``tau_mean`` and ``tau_sigma``.

    This likelihood is vectorized i.e., it accepts an array of ``astro_params``.

    Parameters
    ----------
    tau_mean : float
        Mean for the optical depth constraint.
        By default, it is 0.0569 from Planck 2018 (2006.16828)
    tau_sigma_u : float
        One sigma errors for the optical depth constraint.
        By default, it is 0.0081 from Planck 2018 (2006.16828)
    tau_sigma_l : float
        One sigma errors for the optical depth constraint.
        By default, it is 0.0066 from Planck 2018 (2006.16828)
    """

    required_cores = (
        (core.CoreCoevalModule, core.CoreLightConeModule, core.Core21cmEMU),
    )

    def __init__(
        self,
        *args,
        tau_mean=0.0569,
        tau_sigma_u=0.0073,
        tau_sigma_l=0.0066,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Mean and one sigma errors for the Planck constraints
        # Cosmology from Planck 2018(https://arxiv.org/abs/2006.16828)
        self.tau_mean = tau_mean
        self.tau_sigma_u = tau_sigma_u
        self.tau_sigma_l = tau_sigma_l

        # Simple linear extrapolation of the redshift range provided by the user, to be
        # able to estimate the optical depth
        self.n_z_interp = 25
        self.z_extrap_min = 5.0
        self.z_extrap_max = 30.0

    def computeLikelihood(self, model):
        """
        Compute the likelihood.

        This is the likelihood arising from Planck (2016)
        (https://arxiv.org/abs/1605.03507).

        Parameters
        ----------
        model : list of dicts
            Exactly the output of :meth:`simulate`.

        Returns
        -------
        lnl : float
            The log-likelihood for the given model.
        """
        tau_sigma_u = np.sqrt(self.tau_sigma_u**2 + 0.5 * model["tau_err"] ** 2)
        tau_sigma_l = np.sqrt(self.tau_sigma_l**2 + 0.5 * model["tau_err"] ** 2)

        lnl = (
            -0.5
            * np.square(self.tau_mean - model["tau"])
            / (
                tau_sigma_u * tau_sigma_l
                + (tau_sigma_u - tau_sigma_l) * (model["tau"] - self.tau_mean)
            )
        )
        logger.debug("Planck Likelihood computed: {lnl}".format(lnl=lnl))
        return lnl

    @property
    def _is_lightcone(self):
        return isinstance(self.core_primary, core.CoreLightConeModule)

    @property
    def _is_emu(self):
        return isinstance(self.core_primary, core.Core21cmEMU)

    def reduce_data(self, ctx):
        """Reduce the data in the context to a model.

        Returns
        -------
        dict :
            Only key is "tau", the optical depth to reionization.
            If the emulator is used, 'tau_err' is the emulator error on tau calculated using compute_tau.
        """
        # Extract relevant info from the context.
        if self._is_emu:
            tau_err = ctx.get("tau_err")
            tau_value = ctx.get("tau")

        elif self._is_lightcone:
            lc = ctx.get("lightcone")

            redshifts = lc.node_redshifts
            xHI = lc.global_xHI
            tau_err = 0.0

        else:
            redshifts = self.core_primary.redshift
            xHI = [np.mean(x.xH_box) for x in ctx.get("xHI")]
            tau_err = 0.0

        if not self._is_emu:
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
            z_extrap = np.linspace(
                self.z_extrap_min, self.z_extrap_max, self.n_z_interp
            )
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

        return {"tau": tau_value, "tau_err": tau_err}


class LikelihoodNeutralFraction(LikelihoodBase):
    """
    A likelihood based on the measured neutral fraction at a range of redshifts.

    This likelihood is vectorized i.e., it accepts an array of ``astro_params``.

    The log-likelihood statistic is a simple chi^2 if the model has xHI > threshold,
    and 0 otherwise.
    """

    required_cores = (
        (
            core.CoreLightConeModule,
            core.CoreCoevalModule,
            core.CoreCMB,
            core.Core21cmEMU,
        ),
    )
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
        self._use_tanh = False

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

    @property
    def emu_modules(self):
        """All emulator core modules that are loaded."""
        return [m for m in self._cores if isinstance(m, core.Core21cmEMU)]

    @property
    def cmb_modules(self):
        """All CMB core modules that are loaded."""
        return [m for m in self._cores if (isinstance(m, core.CoreCMB))]

    def setup(self):
        """Perform post-init setup."""
        if (
            not self.lightcone_modules
            + self.coeval_modules
            + self.cmb_modules
            + self.emu_modules
        ):
            raise ValueError(
                "LikelihoodNeutralFraction needs the CoreLightConeModule *or* "
                "CoreCoevalModule *or* CoreCMB to be loaded."
            )

        if not self.lightcone_modules:
            if self.cmb_modules:
                self._use_tanh = True
                self._use_coeval = False
                self._require_spline = True
            elif self.emu_modules:
                self._use_tanh = True
                self._use_coeval = False
                self._require_spline = True
            else:
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
        err = 0
        if self._use_coeval:
            xHI = np.array([np.mean(x) for x in ctx.get("xHI")])
            redshifts = self.redshifts
        else:
            if self._use_tanh:
                # TODO just a temperory fix to get xHI from x_e
                # 1.0818709330934035 = 1+0.25*YHe/(1-YHe) with YHe coming from BBN
                # I think there is a way to fix YHe using user defined value
                xHI = ctx.get("xHI")
                redshifts = ctx.get("zs")
                if redshifts is None:
                    redshifts = ctx.get("redshifts")
                    err = ctx.get("xHI_err")
            else:
                xHI = ctx.get("lightcone").global_xHI
                redshifts = ctx.get("lightcone").node_redshifts
        if xHI.ndim == 1:
            redshifts, xHI = np.sort([redshifts, xHI])

        return {"xHI": xHI, "redshifts": redshifts, "err": err}

    def computeLikelihood(self, model):
        """Compute the likelihood."""
        n = model["xHI"].shape[0]
        xHI = np.atleast_2d(model["xHI"])
        lnprob = np.zeros(n)
        for i in range(n):
            if self._require_spline:
                model_spline = InterpolatedUnivariateSpline(
                    model["redshifts"], xHI[i, :], k=1
                )
                if np.sum(model["err"]) > 0:
                    err_spline = InterpolatedUnivariateSpline(
                        model["redshifts"], model["err"], k=1
                    )

            for z, data, sigma in zip(self.redshift, self.xHI, self.xHI_sigma):
                if np.sum(model["err"]) > 0:
                    sigma_t = np.sqrt(sigma**2 + err_spline(z) ** 2)
                else:
                    sigma_t = sigma
                if z in model["redshifts"]:
                    lnprob[i] += self.lnprob(
                        xHI[model["redshifts"].index(z)], data, sigma_t
                    )
                else:
                    lnprob[i] += self.lnprob(model_spline(z), data, sigma_t)

        logger.debug("Neutral fraction Likelihood computed: {lnl}".format(lnl=lnprob))
        return lnprob

    def lnprob(self, model, data, sigma):
        """Compute the log prob given a model, data and error."""
        model = np.clip(model, 0, 1)

        if model > self.threshold:
            return -0.5 * ((data - model) / sigma) ** 2
        else:
            return 0


class LikelihoodNeutralFractionTwoSided(LikelihoodNeutralFraction):
    """
    A likelihood based on the measured neutral fraction at a range of redshifts.

    This likelihood is vectorized i.e., it accepts an array of ``astro_params``.
    See ``LikelihoodNeutralFraction`` for more information.

    The log-likelihood statistic is a simple chi^2.
    """

    required_cores = (
        (
            core.CoreLightConeModule,
            core.CoreCoevalModule,
            core.CoreCMB,
            core.Core21cmEMU,
        ),
    )
    threshold = 0.06

    def __init__(self, redshift=5.9, xHI=0.06, xHI_sigma=0.05):
        """
        Neutral fraction likelihood/prior.

        Parameters
        ----------
        redshift : float or list of floats
            Redshift(s) at which the neutral fraction has been measured.
        xHI : float or list of floats
            Measured values of the neutral fraction, corresponding to `redshift`.
        xHI_sigma : float or list of floats
            Two-sided uncertainty of measurements.
        """
        super().__init__(redshift=redshift, xHI=xHI, xHI_sigma=xHI_sigma)

    def lnprob(self, model, data, sigma):
        """Compute the log prob given a model, data and error."""
        model = np.clip(model, 0, 1)

        return -0.5 * ((data - model) / sigma) ** 2


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

    required_cores = ((core.CoreLightConeModule, core.Core21cmEMU),)

    @property
    def _is_emu(self):
        return isinstance(self.core_primary, core.Core21cmEMU)

    def reduce_data(self, ctx):
        """Reduce data to model."""
        if self._is_emu:
            return {
                "frequencies": 1420.4 / (self._cores[0].redshift + 1.0),
                "global_signal": ctx.get("brightness_temp"),
            }
        else:
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

    This likelihood is vectorized i.e., it accepts an array of ``astro_params``.

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

    required_cores = ((core.CoreLuminosityFunction, core.Core21cmEMU),)

    def __init__(self, *args, name="", mag_brightest=-20.0, z=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.datafile is not None and len(self.datafile) != 1:
            raise ValueError(
                "can only pass a single datafile to LikelihoodLuminosityFunction!"
            )
        if self.noisefile is not None and len(self.noisefile) != 1:
            raise ValueError(
                "can only pass a single noisefile to LikelihoodLuminosityFunction!"
            )
        # This argument is for the emulator to know which z bin this likelihood is for.
        self.z = z
        self.name = str(name)
        self.mag_brightest = mag_brightest

    def setup(self):
        """Setup instance."""
        if isinstance(self.paired_core, core.Core21cmEMU):
            if self.z is None:
                raise ValueError(
                    "Must specify which z bin out of [6, 7, 8, 10] the likelihood is comparing to the data."
                )
            self.zidx = np.argmin(abs(np.array([6, 7, 8, 10]) - int(self.z)))

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
            if (isinstance(c, core.CoreLuminosityFunction) and c.name == self.name) or (
                isinstance(c, core.Core21cmEMU) and c.name == self.name
            ):
                paired.append(c)
        if len(paired) > 1:
            raise ValueError(
                "You've got more than one CoreLuminosityFunction / Core21cmEMU with the same name -- they will overwrite each other!"
            )
        return paired[0]

    @property
    def redshifts(self):
        """Redshifts at which luminosity function is defined."""
        if isinstance(self.paired_core, core.Core21cmEMU):
            if hasattr(self.z, "__len__"):
                return self.z
            else:
                return np.array([self.z])
        else:
            return self.paired_core.redshift

    def reduce_data(self, ctx):
        """Reduce simulated model data."""
        if isinstance(self.paired_core, core.Core21cmEMU):
            final_data = {}
            shape = ctx.get("UVLFs").shape
            if len(shape) == 3:
                final_data["lfunc"] = ctx.get("UVLFs")[:, self.zidx, :].reshape(
                    (shape[0], 1, shape[-1])
                )
            else:
                final_data["lfunc"] = ctx.get("UVLFs")[self.zidx, :].reshape([1, -1])[
                    np.newaxis, ...
                ]
            # Add two dimensions for nparams
            N = final_data["lfunc"].shape[0]
            final_data["Muv"] = np.array(
                list(ctx.get("Muv").reshape([1, -1])[np.newaxis, ...]) * N
            )
            # TODO check if error is Gaussian and incorporate it properly
            return final_data
        else:
            lfunc = ctx.get("luminosity_function" + self.name)
            if not self._is_setup:
                # During setup, return a list, so that it can be matched with the
                # list of length one of datafile to be written.
                return [lfunc]
            else:
                return lfunc

    def computeLikelihood(self, model):
        """Compute the likelihood."""
        N = model["lfunc"].shape[0]
        lnl = np.zeros(N)

        if len(self.data["lfunc"].shape) == 3:
            data = {"lfunc": self.data["lfunc"][0], "Muv": self.data["Muv"][0]}
        else:
            data = self.data
        for n in range(N):
            for i, z in enumerate(self.redshifts):
                if len(model["Muv"].shape) == 3:
                    if model["Muv"][n][i][0] > model["Muv"][n][i][1]:
                        muv = model["Muv"][n][i][::-1]
                        lfunc = model["lfunc"][n, i][::-1]
                    else:
                        muv = model["Muv"][n][i]
                        lfunc = model["lfunc"][n, i]
                else:
                    muv = model["Muv"][n][i]
                    lfunc = model["lfunc"][n, i]

                mask = ~np.isnan(lfunc)
                model_spline = InterpolatedUnivariateSpline(muv[mask], lfunc[mask])

                total_err = self.noise["sigma"][i] ** 2

                lnl[n] += -0.5 * np.sum(
                    (
                        (data["lfunc"][i] - 10 ** model_spline(data["Muv"][i])) ** 2
                        / total_err
                    )[data["Muv"][i] > self.mag_brightest]
                )
        logger.debug("UV LF Likelihood computed: {lnl}".format(lnl=lnl))
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


class LikelihoodForest(LikelihoodBaseFile):
    """
    A likelihood based on chi^2 comparison to measured CDF of Lyman-alpha forest effective optical depth.

    This is the likelihood arising from Bosman et al. (2018), which reports new constraints on Lyman-alpha
    opacity with a sample of 62 quasars at z>5.7, or D'Odorico et al. in prep., which includes new samples
    from the XQR-30 survey.

    Parameters
    ----------
    name : str
        The name used to match the core

    observation : str
        The observation that is used to construct the tau_eff statisctic.

    """

    required_cores = (core.CoreForest,)

    def __init__(self, name="", observation="bosman_optimistic", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(name)
        self.observation = str(observation)
        self.n_realization = 150

    def setup(self):
        """Setup instance."""
        if len(self.redshifts) != 1:
            raise ValueError(
                "to use the provided forests, a separate core/likelihood instance pair for each redshift is required!"
            )
        if "bosman" in self.observation:
            if self.redshifts[0] not in [5.0, 5.2, 5.4, 5.6, 5.8, 6.0]:
                raise ValueError(
                    "only forests at z=5.0, 5.2, 5.4, 5.6, 5.8, 6.0 are provided for bosman!"
                )
            self.tau_range = [0, 8]
            self.hist_bin_width = 0.1
            self.hist_bin_size = int(
                (self.tau_range[1] - self.tau_range[0]) / self.hist_bin_width
            )

            self.datafile = [
                path.join(path.dirname(__file__), "data/Forests/Bosman18/data.npz")
            ]
            self.noisefile = [
                path.join(
                    path.dirname(__file__),
                    "data/Forests/Bosman18/PDF_ErrorCovarianceMatrix_GP/z%s.npy"
                    % str(self.redshifts[0]).replace(".", "pt"),
                )
            ]

        else:
            raise NotImplementedError("Use bosman_optimistic or bosman_pessimistic!")

        super().setup()

    def _read_data(self):
        data = super()._read_data()[0]
        targets = np.where(
            (data["zs"] > (self.redshifts[0] - 0.1))
            * (data["zs"] <= (self.redshifts[0] + 0.1))
        )[0]
        pdfs = np.zeros([2, self.hist_bin_size])

        pdfs[0] = np.histogram(
            data["tau_lower"][targets], range=self.tau_range, bins=self.hist_bin_size
        )[0]

        pdfs[1] = np.histogram(
            data["tau_upper"][targets], range=self.tau_range, bins=self.hist_bin_size
        )[0]
        return pdfs

    def _read_noise(self):
        # read the ECM due to the GP approximation
        ErrorCovarianceMatrix_GP = super()._read_noise()[0]
        return ErrorCovarianceMatrix_GP

    @cached_property
    def paired_core(self):
        """The forest core that is paired with this likelihood."""
        paired = []
        for c in self._cores:
            if isinstance(c, core.CoreForest) and c.name == self.name:
                paired.append(c)
        if len(paired) > 1:
            raise ValueError(
                "You've got more than one CoreForest with the same name -- they will overwrite each other!"
            )
        return paired[0]

    @property
    def redshifts(self):
        """Redshifts at which forest is defined."""
        return self.paired_core.redshift

    @property
    def _is_lightcone(self):
        return isinstance(self.core_primary, core.CoreLightConeModule)

    def reduce_data(self, ctx):
        """Reduce data to model."""
        if not self._is_lightcone:
            raise NotImplementedError(
                "The Forest can only work with lightcone at the moment"
            )

        tau_eff = ctx.get("tau_eff_%s" % self.name)
        # use the same binning as the obs

        n_realization = tau_eff.shape[0]
        pdfs = np.zeros([n_realization, self.hist_bin_size])
        for jj in range(n_realization):
            pdfs[jj] = np.histogram(
                tau_eff[jj], range=self.tau_range, bins=self.hist_bin_size
            )[0]

        ecm_cosmic = np.cov(pdfs.T)
        self.noise = (
            self.noise + ecm_cosmic + np.diag(np.ones(self.hist_bin_size) * 1e-5)
        )

        return np.mean(pdfs, axis=0)

    def computeLikelihood(self, model):
        """
        Compute the likelihood, given the lightcone output from 21cmFAST.

        Parameters
        ----------
        model : list of pdfs
            Exactly the output of :meth:`simulate`.

        Returns
        -------
        lnl : float
            The log-likelihood for the given model.
        """
        det = np.linalg.det(self.noise)
        if det == 0:
            logger.warning(
                "Determinant is zero for this error covariance matrix, return -inf for lnl"
            )
            return -np.inf

        diff = model - self.data[0]
        for ii in np.where(self.data[0] != self.data[1])[0]:
            if model[ii] < self.data[0][ii]:
                diff[ii] = min(0, model[ii] - self.data[1][ii])
        diff = diff.reshape([1, -1])

        lnl = (
            -0.5 * np.linalg.multi_dot([diff, np.linalg.inv(self.noise), diff.T])[0, 0]
        )
        if det < 0:
            logger.warning(
                "Determinant (%f) is negative for this error covariance matrix, lnl=%f, return -inf for lnl"
                % (det, lnl)
            )
            return -np.inf
        return lnl


class Likelihood1DPowerLightconeUpper(Likelihood1DPowerLightcone):
    r"""
    Likelihood based on Chi^2 comparison of a 21 cm PS model to HERA H1C upper limit data.

    Parameters
    ----------
    datafile : str, optional
        Input data should be in a `.npz` file, and contain the following fields:
        * ``z_bands``: the redshift of the observation bands
        * ``bandx``: the power spectrum upper limits at redshift x
        * ``wfbandx``: the window function for band x
        * ``kwfbandx``: the k values [/Mpc] of the window function for band x
    """

    def __init__(
        self,
        datafile="",
        data=None,
        name="",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name

        self.datafile = [datafile] if isinstance(datafile, (str, Path)) else datafile

    @classmethod
    def from_builtin_data(cls, datafile="", **kwargs):
        """Create the class loading in built-in data."""
        datafile = path.join(path.dirname(__file__), "data", datafile + ".npz")

        return cls(datafile=datafile, **kwargs)

    def setup(self):
        """Setup the object."""
        super().setup()
        self.redshifts = self.data[0]["z_bands"]
        all_keys = np.array(list(self.data[0].keys()))
        m = ["kwf" in i for i in all_keys]
        all_kwfs_keys = all_keys[m]
        self.k = [self.data[0][j] for j in all_kwfs_keys]
        self.k_len = max(len(i) for i in self.k)

    def reduce_data(self, ctx):
        """Get the computed core data in nice form."""
        # Interpolate the data onto the HERA bands and ks
        if len(ctx.get("PS").shape) > 2:
            final_PS = np.zeros(
                (ctx.get("PS").shape[0], len(self.redshifts), self.k_len)
            )
            for j in range(ctx.get("PS").shape[0]):
                for i in range(self.redshifts.shape[0]):
                    interp_ks = self.k[i]
                    final_PS[j, i, : len(interp_ks)] = RectBivariateSpline(
                        ctx.get("PS_redshifts"), ctx.get("k"), ctx.get("PS")[j, ...]
                    )(self.redshifts[i], interp_ks)
            final_data = {
                "k": self.k,
                "delta": final_PS,
            }
        else:
            final_PS = np.zeros((1, len(self.redshifts), self.k_len))
            for i in range(self.redshifts.shape[0]):
                interp_ks = self.k[i]
                final_PS[0, i, : len(interp_ks)] = RectBivariateSpline(
                    ctx.get("PS_redshifts"), ctx.get("k"), ctx.get("PS")
                )(self.redshifts[i], interp_ks)
            final_data = {
                "k": self.k,
                "delta": final_PS,
            }

        # If we are using the emualtor, include the error.
        try:
            final_PS_err = np.zeros((len(self.redshifts), self.k_len))
            for i in range(self.redshifts.shape[0]):
                interp_ks = self.k[i]
                final_PS_err[i, : len(interp_ks)] = RectBivariateSpline(
                    ctx.get("PS_redshifts"), ctx.get("k"), ctx.get("PS_err")
                )(self.redshifts[i], interp_ks)
            final_data["delta_err"] = final_PS_err
        except:
            pass
        return [final_data]

    def computeLikelihood(self, model):
        """
        Compute the likelihood given 1D power spectrum values at the same k-bins as the data.

        Parameters
        ----------
        model : np.ndarray
            1D Power spectrum in log10(mK^2) and its k bin values in /Mpc
            If there is a window function available, then input the PS after the WF has been applied.

        Returns
        -------
        lnl : float
            Log likelihood for the provided model.
            For H1C: Data shape = 5 fields, 37 kbins, 4 = [kval, power, variance],
            2 (band1=10 band2=8)
        """
        N = model[0]["delta"].shape[0]
        lnl = np.zeros(N)
        hera_data = self.data[0]
        for i in range(N):
            for band in self.redshifts:
                for field in range(hera_data["band" + str(round(band))].shape[0]):
                    PS_limit_ks = hera_data["band" + str(round(band))][field, :, 0]
                    PS_limit_ks = PS_limit_ks[~np.isnan(PS_limit_ks)]
                    Nkbins = len(PS_limit_ks)
                    PS_limit_vals = hera_data["band" + str(round(band))][
                        field, :Nkbins, 1
                    ]
                    PS_limit_vars = hera_data["band" + str(round(band))][
                        field, :Nkbins, 2
                    ]

                    kwf_limit_vals = hera_data["kwfband" + str(round(band))]
                    Nkwfbins = len(kwf_limit_vals)
                    PS_limit_wfcs = hera_data["wfband" + str(round(band))][
                        field, :Nkbins, :
                    ]

                    PS_limit_wfcs = PS_limit_wfcs.reshape([Nkbins, Nkwfbins])

                    model_zs = self.redshifts
                    zbin = np.argmin(abs(band - model_zs))
                    ModelPS_val = model[0]["delta"][i, zbin, :Nkwfbins]

                    ModelPS_val_afterWF = np.dot(PS_limit_wfcs, ModelPS_val)
                    # Include emulator error term if present
                    if "delta_err" in model[0].keys():
                        ModelPS_val_1sigma_upper_afterWF = np.dot(
                            PS_limit_wfcs,
                            ModelPS_val + model[0]["delta_err"][zbin, :Nkwfbins],
                        )
                        ModelPS_val_1sigma_lower_afterWF = np.dot(
                            PS_limit_wfcs,
                            ModelPS_val - model[0]["delta_err"][zbin, :Nkwfbins],
                        )
                        # The upper and lower errors are very similar usually, so we can just take the mean and use that.
                        mean_err = np.mean(
                            [
                                ModelPS_val_1sigma_upper_afterWF - ModelPS_val_afterWF,
                                ModelPS_val_afterWF - ModelPS_val_1sigma_lower_afterWF,
                            ],
                            axis=0,
                        )

                        error_val = np.sqrt(
                            PS_limit_vars
                            + (0.2 * ModelPS_val_afterWF) ** 2
                            + (mean_err) ** 2
                        )
                    else:
                        error_val = np.sqrt(
                            PS_limit_vars + (0.2 * ModelPS_val_afterWF) ** 2
                        )

                    likelihood = 0.5 + 0.5 * erf(
                        (PS_limit_vals - ModelPS_val_afterWF) / (np.sqrt(2) * error_val)
                    )  # another way to write likelihood for 1-side Gaussian
                    likelihood[likelihood <= 0.0] = 1e-50
                    lnl[i] += np.nansum(np.log(likelihood))
                    logger.debug(
                        "HERA PS upper Likelihood computed: {lnl}".format(
                            lnl=np.nansum(np.log(likelihood))
                        )
                    )
        logger.debug("Total HERA PS upper Likelihood computed: {lnl}".format(lnl=lnl))
        return lnl

    @cached_property
    def paired_core(self):
        """The 21cmEMU core that is paired with this likelihood."""
        paired = []
        for c in self._cores:
            if isinstance(c, core.Core21cmEMU) and c.name == self.name:
                paired.append(c)
        if len(paired) > 1:
            raise ValueError(
                "You've got more than one 21cmEMU with the same name -- they will overwrite each other!"
            )
        return paired[0]
