"""Tests of LikelihoodPspec (pspec_likelihood) and of the units of the HERA data."""

from os import path

import numpy as np
import pytest
from astropy import units as un

from py21cmmc import LikelihoodPspec, build_computation_chain, core, systematics

# pspec_likelihood is optional (see the "likelihoods" extra in pyproject.toml).
pytest.importorskip("pspec_likelihood")


def test_pspec_lk():
    """Basic smoke test for the pspec_likelihood-based likelihood.

    Uses the same built-in HERA H1C IDR3 upper-limit data as
    ``test_hera_lk.py``, but computed through :class:`LikelihoodPspec`
    (backed by ``pspec_likelihood``) instead of
    :class:`~py21cmmc.likelihood.Likelihood1DPowerLightconeUpper`.
    """
    lk = LikelihoodPspec.from_builtin_data("HERA_H1C_IDR3")

    c21cmemu = core.Core21cmEMU()
    chain = build_computation_chain(
        [c21cmemu],
        [lk],
        setup=True,
    )
    chain({})


def test_hera_h1c_data_in_inv_mpc():
    """The built-in HERA H1C IDR3 data have k in 1/Mpc.

    HERA Collaboration 2023 quote their best 2 sigma limits as 457 mK^2 at
    k = 0.34 h/Mpc (z = 7.9) and 3496 mK^2 at k = 0.36 h/Mpc (z = 10.4).
    """
    data = np.load(path.join(path.dirname(core.__file__), "data", "HERA_H1C_IDR3.npz"))
    for band, limit, k in [("band8", 460, 0.228), ("band10", 3496, 0.241)]:
        x = data[band]
        ok = np.isfinite(x[..., 0]) & (x[..., 2] > 0)
        ul = np.where(ok, x[..., 1] + 2 * np.sqrt(np.where(ok, x[..., 2], 0)), np.inf)
        best = np.unravel_index(np.argmin(ul), ul.shape)
        assert np.isclose(ul[best], limit, rtol=1e-3)
        assert np.isclose(x[best][0], k, rtol=1e-2)


class FakeContext(dict):
    """A minimal chain context holding the model outputs."""

    def get(self, key):
        """Get a model output."""
        return self[key]


def test_pspec_theory_k_in_inv_mpc():
    """The model PS is evaluated at the k of the data file, both in 1/Mpc."""
    lk = LikelihoodPspec.from_builtin_data("HERA_H1C_IDR3")
    build_computation_chain([core.Core21cmEMU(emulator_name="acg")], [lk], setup=True)

    # A model with Delta^2(k) = k, so that the model evaluated in each bin is its k.
    k = np.linspace(0.01, 3.0, 1000)
    ctx = FakeContext(
        PS=np.tile(k, (1, len(lk.redshifts), 1)),
        k=k,
        PS_redshifts=np.array(lk.redshifts),
    )
    for dmi in lk.reduce_data(ctx)[0]:
        kwf = lk.k[list(lk.redshifts).index(dmi.redshift)]
        model = dmi.discretize_theory(dmi.redshift, [])
        assert np.allclose(model.to_value(un.mK**2), kwf)


def test_pspec_systematics():
    """systematics=Flat(np.inf) gives the HERA upper-limit likelihood."""
    k = np.linspace(0.01, 3.0, 1000)
    lnl = {}
    for name, kwargs in [
        ("ul", {"lk": "upper_limit"}),
        ("flat_inf", {"systematics": systematics.Flat(np.inf)}),
        ("halfnormal", {"systematics": systematics.HalfNormal(1.0)}),
    ]:
        lk = LikelihoodPspec.from_builtin_data("HERA_H1C_IDR3", **kwargs)
        build_computation_chain(
            [core.Core21cmEMU(emulator_name="acg")], [lk], setup=True
        )
        ps = np.stack([np.tile(100 * k, (len(lk.redshifts), 1)) * a for a in [1, 30]])
        ctx = FakeContext(PS=ps, k=k, PS_redshifts=np.array(lk.redshifts))
        lnl[name] = lk.computeLikelihood(lk.reduce_data(ctx))

    assert np.allclose(lnl["flat_inf"], lnl["ul"])
    assert np.all(np.isfinite(lnl["halfnormal"]))
    assert not np.allclose(lnl["halfnormal"], lnl["ul"])
