"""Checks that an observable type keeps the contract every observable shares.

A new observable type is tested by passing an example of it, with a covariance and
at least two entries in some group, to :func:`assert_observable_contract`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import attrs
import numpy as np
from astropy import units as u

from .observables import Observable, _groups


def assert_observable_contract(obs: Observable, tmp_path: Path) -> None:
    """Check that ``obs`` keeps the observable contract; raise AssertionError if not.

    ``obs`` needs a covariance, and some group with at least two entries along the
    interpolation axis. ``tmp_path`` is a directory for a temporary HDF5 file.
    """
    assert obs.covariance is not None, "The example needs a covariance."
    _check_units_and_shapes(obs)
    _check_at_own_coords_is_identity(obs)
    _check_at_a_subset_matches_mask(obs)
    _check_interpolation_is_linear(obs)
    _check_units_convert(obs)
    _check_hdf5_round_trip(obs, tmp_path / "observable.h5")
    _check_plot_labels(obs)


def _assert_same(a: Observable, b: Observable) -> None:
    """Assert that two observables have the same type, coordinates and values."""
    assert type(a) is type(b)
    assert list(a.coords) == list(b.coords)
    for name in a.coords:
        assert a.coords[name].unit == b.coords[name].unit
        np.testing.assert_allclose(a.coords[name].value, b.coords[name].value)
    assert a.values.unit == b.values.unit
    np.testing.assert_allclose(a.values.value, b.values.value, atol=0, rtol=1e-12)
    assert (a.covariance is None) == (b.covariance is None)
    if a.covariance is not None:
        np.testing.assert_allclose(
            a.covariance.value, b.covariance.value, rtol=1e-12, atol=0
        )


def _check_units_and_shapes(obs: Observable) -> None:
    """Check the units and sizes of the values, coordinates and covariance."""
    n = len(obs)
    assert obs.values.shape == (n,)
    assert obs.values.unit == obs.unit
    for name, unit in obs.axes.items():
        assert obs.coords[name].shape == (n,)
        assert obs.coords[name].unit == unit
    assert obs.covariance.shape == (n, n)
    assert obs.covariance.unit == obs.unit**2
    np.testing.assert_allclose(obs.sigma**2, np.diag(obs.covariance))


def _check_at_own_coords_is_identity(obs: Observable) -> None:
    """Check that interpolating onto its own coordinates changes nothing."""
    _assert_same(obs.at(obs.coords), obs)


def _check_at_a_subset_matches_mask(obs: Observable) -> None:
    """Check that interpolating onto some of its points equals masking the others."""
    drop = np.arange(len(obs)) % 2 == 1
    keep = ~drop
    _assert_same(obs.at({k: v[keep] for k, v in obs.coords.items()}), obs.mask(drop))


def _check_interpolation_is_linear(obs: Observable) -> None:
    """Check that halfway between neighbours, value and variance are of their mean.

    Halfway is in the coordinate, or in its logarithm for a type that interpolates
    logarithmically.
    """
    *group_axes, along = obs.axes
    a, b = _neighbours(obs, group_axes, along)
    x = obs.coords[along][[a, b]]
    mid = np.sqrt(x[0] * x[1]) if obs.log_interp else (x[0] + x[1]) / 2
    coords = {g: obs.coords[g][[a]] for g in group_axes}
    coords[along] = u.Quantity([mid])
    out = obs.at(coords)
    np.testing.assert_allclose(out.values, (obs.values[a] + obs.values[b]) / 2)
    c = obs.covariance
    np.testing.assert_allclose(
        out.covariance[0, 0], (c[a, a] + c[b, b] + 2 * c[a, b]) / 4
    )


def _neighbours(obs: Observable, group_axes: list[str], along: str) -> tuple[int, int]:
    """Return two entries of one group that are adjacent along the last axis."""
    for idx in _groups(obs.coords, group_axes, along).values():
        if len(idx) > 1:
            return int(idx[0]), int(idx[1])
    raise AssertionError("The example needs a group with at least two entries.")


def _check_units_convert(obs: Observable) -> None:
    """Check that the observable given in other, compatible units is unchanged."""
    other = type(obs)(
        coords={k: v.to(u.Unit(1e-3 * v.unit)) for k, v in obs.coords.items()},
        values=obs.values.to(u.Unit(1e3 * obs.unit)),
        covariance=obs.covariance.to(u.Unit(1e6 * obs.unit**2)),
    )
    _assert_same(other, obs)


def _check_hdf5_round_trip(obs: Observable, path: Path) -> None:
    """Check that writing to HDF5 and reading back, by either type, loses nothing."""
    obs.to_hdf5(path)
    _assert_same(Observable.from_hdf5(path), obs)
    _assert_same(type(obs).from_hdf5(path), obs)
    no_cov = attrs.evolve(obs, covariance=None)
    no_cov.to_hdf5(path)
    _assert_same(Observable.from_hdf5(path), no_cov)


def _check_plot_labels(obs: Observable) -> None:
    """Check that the plot labels its axes with the quantities and their units.

    The observable plots with or without a covariance (as a model may have none).
    Plotting needs matplotlib, so this check is skipped without it.
    """
    if importlib.util.find_spec("matplotlib") is None:
        return
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    for example in (obs, attrs.evolve(obs, covariance=None)):
        ax = example.plot()
        try:
            along = list(obs.axes)[-1]
            assert ax.get_xlabel() == f"{along} [{obs.axes[along]:latex_inline}]"
            assert ax.get_ylabel() == f"{obs.label} [{obs.unit:latex_inline}]"
        finally:
            plt.close(ax.figure)
