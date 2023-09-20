"""Functions to analyse the results of MCMC chains.

Also enables more transparent input/output of chains.
"""
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
from pathlib import Path
from py21cmfast import yaml

from .cosmoHammer import CosmoHammerSampler, HDFStorage


def get_samples(chain, indx=0, burnin=False):
    """
    Extract sample storage object from a chain.

    Parameters
    ----------
    chain : :class:`~py21cmmc.cosmoHammer.CosmoHammerSampler` or str
        Either a :class:`~py21cmmc.cosmoHammer.CosmoHammerSampler`, which is the output
        of the :func:`~py21cmmc.mcmc.run_mcmc` function,
        or a path to an output HDF5 file containing the chain.
    indx : int, optional
        This is used only if `chain` is a string. It gives the index of the samples in
        the HDF file. Usually this is zero.
    burnin : bool
        Whether to return the burnin samples, rather than the actual run samples.

    Returns
    -------
    store : a :class:`~py21cmmc.cosmoHammer.HDFStore` object.
    """
    if isinstance(chain, CosmoHammerSampler):
        return (
            chain.storageUtil.sample_storage
            if not burnin
            else chain.storageUtil.burnin_storage
        )
    elif isinstance(chain, str):
        if not chain.endswith(".h5"):
            chain += ".h5"
    elif isinstance(chain, Path):
        if chain.suffix != ".h5":
            chain = chain.with_suffix(".h5")
    else:
        raise AttributeError(
            "chain must either be a CosmoHammerSampler instance, str or Path"
        )

    return HDFStorage(chain, name="burnin" if burnin else "sample_%s" % indx)


def load_primitive_chain(modelname, direc="."):
    """
    Load a chain file produced by :func:`~py21cmmc.mcmc.run_mcmc` to be interactively useable.

    Parameters
    ----------
    modelname : str
        Model name of the MCMC run.
    direc : str
        Directory in which data was stored.

    Returns
    -------
    chain : :class:`~py21cmmc.cosmoHammer.LikelihoodComputationChain`
        The fully set-up chain, with no computed samples.
    """
    with open(join(direc, modelname + ".LCC.yml")) as f:
        chain = yaml.load(f)

    chain.setup()
    return chain


def corner_plot(
    samples, include_lnl=True, show_guess=True, start_iter=0, thin=1, **kwargs
):
    """
    Make a corner plot given samples.

    Parameters
    ----------
    samples: :class:`py21cmmc.cosmoHammer.HDFStorage` instance
        The ``samples`` attribute of a sampler (i.e. the return value of
        :func:`~.mcmc.run_mcmc`), or equivalently, the return value of :func:`~get_samples`.
    include_lnl: bool, optional
        Whether to plot the log-likelihood as if it were a parameter.
    show_guess: bool, optional
        Whether to show the initial guess as "truths" in the corner plot.
    start_iter: int, optional
        The first iteration to include in the plotted samples.
    thin: int, optional
        Use only every "thin" sample to plot.
    kwargs:
        All kwargs are passed directly to the `corner` function from the `corner` package.

    Returns
    -------
    fig:
        Matlotlib figure object.
    """
    try:
        from corner import corner
    except ImportError:
        raise ImportError(
            "Please install the corner package to use this function (``pip install corner``)"
        )

    chain = samples.get_chain(discard=start_iter, thin=thin)
    lnprob = samples.get_log_prob(discard=start_iter, thin=thin)
    niter, mwalkers, nparams = chain.shape

    if show_guess:
        guess = list(samples.param_guess[0])

    labels = list(samples.param_names)

    if include_lnl:
        plotchain = np.vstack((chain.T, np.atleast_3d(lnprob).T)).T.reshape(
            (-1, nparams + 1)
        )
        if show_guess:
            guess += [None]
        labels += ["lnL"]
    else:
        plotchain = chain.reshape((-1, nparams))

    fig = corner(
        plotchain,
        labels=labels,
        truths=guess if show_guess else None,
        smooth=kwargs.get("smooth", 0.75),
        smooth1d=kwargs.get("smooth1d", 1.0),
        show_titles=True,
        quantiles=kwargs.get("quantiles", [0.16, 0.5, 0.84]),
    )

    return fig


def trace_plot(
    samples, include_lnl=True, show_guess=True, start_iter=0, thin=1, colored=False
):
    """
    Make a trace plot given samples.

    Parameters
    ----------
    samples: :class:`py21cmmc.cosmoHammer.HDFStorage`
        The ``samples`` attribute of a sampler (i.e. the return value of
        :func:`~py21cmmc.mcmc.run_mcmc`), or equivalently,
        the return value of :func:`~get_samples`.
    include_lnl: bool, optional
        Whether to plot the log-likelihood as if it were a parameter.
    show_guess: bool, optional
        Whether to show the initial guess as "truths" in the corner plot.
    start_iter: int, optional
        The first iteration to include in the plotted samples.
    thin: int, optional
        Use only every "thin" sample to plot.
    colored: bool, optional
        Whether to use a color-cycle to color each walker. Otherwise each trace is black.

    Returns
    -------
    fig, ax:
        Matlotlib figure and axis objects.
    """
    nwalkers, nparams = samples.shape
    if include_lnl:
        nparams += 1

    chain = samples.get_chain(thin=thin, discard=start_iter)
    lnprob = samples.get_log_prob(thin=thin, discard=start_iter)

    fig, ax = plt.subplots(
        nparams,
        1,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
        figsize=(8, 3 * nparams),
    )

    for i in range(nwalkers):
        for j, param in enumerate(samples.param_names):
            ax[j].plot(
                chain[:, i, j],
                color="C%s" % (i % 8) if colored else "k",
                alpha=0.75,
                lw=1,
            )
            ax[j].set_ylabel(param)
            if show_guess and not i:
                ax[j].axhline(samples.param_guess[param][0], color="C0", lw=3)

        if include_lnl:
            ax[-1].plot(
                lnprob[:, i],
                color="C%s" % (i % 8) if colored else "k",
                alpha=0.75,
                lw=1,
            )
            ax[-1].set_ylabel("lnL")

    return fig, ax
