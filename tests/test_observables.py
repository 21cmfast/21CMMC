"""Tests of the observable base type and the Observation container."""

from operator import setitem
from typing import ClassVar

import attrs
import h5py
import numpy as np
import pytest
from astropy import units as u

from py21cmmc.observables import Observable, Observation, _groups
from py21cmmc.testing import assert_observable_contract


@attrs.frozen(eq=False, kw_only=True)
class ToySignal(Observable):
    """A signal against frequency, like a global 21-cm signal."""

    axes: ClassVar = {"nu": u.MHz}
    unit: ClassVar = u.mK
    quantity: ClassVar = "T21"


@attrs.frozen(eq=False, kw_only=True)
class ToyBandedPower(Observable):
    """A power spectrum in redshift bands, with different k bins in each band."""

    axes: ClassVar = {"z": u.dimensionless_unscaled, "k": u.Mpc**-1}
    unit: ClassVar = u.mK**2
    quantity: ClassVar = "delta2"
    log_interp: ClassVar = True


def random_covariance(n, rng, scale=1.0):
    a = rng.standard_normal((n, n))
    return scale * (a @ a.T / n + np.eye(n))


def toy_signal(rng):
    nu = np.linspace(50, 200, 16) * u.MHz
    return ToySignal(
        coords={"nu": nu},
        values=-100 * np.exp(-(((nu.value - 78) / 10) ** 2)) * u.mK,
        covariance=random_covariance(16, rng, 4.0) * u.mK**2,
    )


def toy_banded_power(rng):
    # Ragged: 6 bins in the first band, 4 in the second, not on a shared grid.
    k = np.concatenate([np.geomspace(0.1, 1.0, 6), np.geomspace(0.15, 0.8, 4)])
    z = np.repeat([7.9, 10.4], [6, 4])
    return ToyBandedPower(
        coords={"z": z * u.one, "k": k / u.Mpc},
        values=(100 * k**0.5 + z) * u.mK**2,
        covariance=random_covariance(10, rng, 25.0) * u.mK**4,
    )


@pytest.mark.parametrize("make", [toy_signal, toy_banded_power])
def test_toy_types_keep_the_observable_contract(make, tmp_path):
    assert_observable_contract(make(np.random.default_rng(0)), tmp_path)


@pytest.mark.parametrize(
    ("make", "f"),
    [
        # Interpolation is linear in the coordinate, or in its log for log_interp, so a
        # function linear in that variable is reproduced exactly between the points.
        (toy_signal, lambda c: (3.0 * c["nu"].value - 7.0) * u.mK),
        (
            toy_banded_power,
            lambda c: (2.0 * np.log(c["k"].value) + c["z"].value) * u.mK**2,
        ),
    ],
)
def test_interpolation_is_exact_for_a_linear_function(make, f):
    obs = make(np.random.default_rng(1))
    obs = attrs.evolve(obs, values=f(obs.coords))
    *groups, along = obs.axes
    rng = np.random.default_rng(2)
    targets = []
    for key, idx in _groups(obs.coords, groups, along).items():
        x = obs.coords[along][idx]
        new = rng.uniform(x.value.min(), x.value.max(), 5) * x.unit
        targets.append(
            {
                **{
                    g: np.full(5, v) * obs.axes[g]
                    for g, v in zip(groups, key, strict=True)
                },
                along: new,
            }
        )
    coords = {a: np.concatenate([t[a] for t in targets]) for a in obs.axes}
    np.testing.assert_allclose(obs.at(coords).values, f(coords), rtol=1e-12)


def test_interpolated_covariance_matches_monte_carlo_draws():
    """Interpolating each draw of the values reproduces the propagated covariance."""
    rng = np.random.default_rng(3)
    obs = toy_signal(rng)
    target = {"nu": np.array([60.0, 77.5, 130.3, 199.0]) * u.MHz}
    n = 2000
    draws = rng.multivariate_normal(obs.values.value, obs.covariance.value, n)
    bare = attrs.evolve(obs, covariance=None)
    interpolated = np.array(
        [attrs.evolve(bare, values=d * u.mK).at(target).values.value for d in draws]
    )
    expected = obs.at(target).covariance.value
    # A sample covariance entry has standard error at most sqrt(2 C_ii C_jj / n).
    se = np.sqrt(2 * np.outer(np.diag(expected), np.diag(expected)) / n)
    np.testing.assert_array_less(np.abs(np.cov(interpolated.T) - expected), 5 * se)


@pytest.mark.parametrize(
    ("coords", "match"),
    [
        ({"nu": [40.0] * u.MHz}, "no extrapolation"),
        ({"nu": [210.0] * u.MHz}, "no extrapolation"),
        ({"nu": [100.0] * u.s}, "cannot be converted"),
        ({"freq": [100.0] * u.MHz}, "axes"),
    ],
)
def test_bad_target_coordinates_are_rejected(coords, match):
    with pytest.raises((ValueError, u.UnitConversionError), match=match):
        toy_signal(np.random.default_rng(4)).at(coords)


def test_a_missing_group_is_not_interpolated_from_another():
    obs = toy_banded_power(np.random.default_rng(5))
    with pytest.raises(ValueError, match="No entries at"):
        obs.at({"z": [9.0] * u.one, "k": [0.3] / u.Mpc})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"values": [1.0, 2.0] * u.s}, "cannot be converted"),
        ({"values": [1.0, 2.0]}, "must be a quantity"),
        ({"values": [1.0, 2.0, 3.0] * u.mK}, "3 values"),
        ({"covariance": np.eye(3) * u.mK**2}, "shape"),
        ({"covariance": [[1.0, 0.5], [0.0, 1.0]] * u.mK**2}, "symmetric"),
        # Asymmetric at the scale of its own entries, however small they are.
        ({"covariance": [[1e-12, 5e-13], [0.0, 1e-12]] * u.mK**2}, "symmetric"),
    ],
)
def test_inconsistent_observables_are_rejected(kwargs, match):
    args = {"coords": {"nu": [1.0, 2.0] * u.MHz}, "values": [1.0, 2.0] * u.mK}
    with pytest.raises((ValueError, TypeError, u.UnitConversionError), match=match):
        ToySignal(**{**args, **kwargs})


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (
            lambda: Observable(coords={"x": [1.0] * u.one}, values=[1.0] * u.one),
            "has no axes",
        ),
        (lambda: type("NoAxes", (Observable,), {}), "declares no axes"),
        (
            lambda: ToyBandedPower(
                coords={"z": [8.0, 8.0] * u.one, "k": [0.1] / u.Mpc},
                values=[1.0, 2.0] * u.mK**2,
            ),
            "different lengths",
        ),
        (lambda: toy_signal(np.random.default_rng(10)).mask([True]), "one entry per"),
    ],
)
def test_malformed_types_coordinates_and_masks_are_rejected(make, match):
    with pytest.raises((TypeError, ValueError), match=match):
        make()


@pytest.mark.parametrize(
    ("example", "match"),
    [
        (
            attrs.evolve(toy_signal(np.random.default_rng(11)), covariance=None),
            "covariance",
        ),
        (
            ToyBandedPower(
                coords={"z": [7.9, 10.4] * u.one, "k": [0.1, 0.2] / u.Mpc},
                values=[1.0, 2.0] * u.mK**2,
                covariance=np.eye(2) * u.mK**4,
            ),
            "at least two entries",
        ),
    ],
)
def test_the_contract_check_refuses_an_example_it_cannot_use(example, match, tmp_path):
    with pytest.raises(AssertionError, match=match):
        assert_observable_contract(example, tmp_path)


def test_a_file_is_only_read_as_an_imported_type(tmp_path):
    """The stored type name selects among imported types; it never imports code."""
    path = tmp_path / "obs.h5"
    toy_signal(np.random.default_rng(6)).to_hdf5(path)
    with h5py.File(path, "a") as f:
        f.attrs["observable"] = "os.system"
    with pytest.raises(TypeError, match="not an imported observable"):
        Observable.from_hdf5(path)

    toy_signal(np.random.default_rng(6)).to_hdf5(path)
    with pytest.raises(TypeError, match="not a"):
        ToyBandedPower.from_hdf5(path)


def make_observation(rng):
    return Observation(
        value=toy_banded_power(rng),
        citation="Doe et al. (2026), ApJ 1, 1",
        meta={"doi": "10.0000/example", "version": 2, "h": 0.6766},
        instrument={
            "window": rng.uniform(size=(10, 30)) * u.one,
            "band_edges": [7.5, 8.3, 10.0, 10.8] * u.one,
        },
    )


def test_an_observation_round_trips_through_hdf5(tmp_path):
    obs = make_observation(np.random.default_rng(7))
    obs.to_hdf5(tmp_path / "obs.h5")
    back = Observation.read(tmp_path / "obs.h5")

    assert type(back.value) is ToyBandedPower
    np.testing.assert_array_equal(back.covariance, obs.covariance)
    assert back.citation == obs.citation
    assert back.meta == obs.meta
    assert {type(v) for v in back.meta.values()} == {str, int, float}
    assert set(back.instrument) == set(obs.instrument)
    for k, v in obs.instrument.items():
        np.testing.assert_array_equal(back.instrument[k], v)
        assert back.instrument[k].unit == v.unit


def test_an_observation_is_read_by_any_reader_function(tmp_path):
    expected = make_observation(np.random.default_rng(8))
    path = tmp_path / "data.txt"
    path.write_text("not hdf5")

    def reader(p):
        assert p == path
        return expected

    assert Observation.read(path, reader=reader) is expected
    with pytest.raises(TypeError, match="not an Observation"):
        Observation.read(path, reader=lambda p: expected.value)


def test_observation_metadata_must_be_plain_values():
    with pytest.raises(TypeError, match="strings or numbers"):
        Observation(
            value=toy_signal(np.random.default_rng(9)),
            citation="x",
            meta={"labels": ["a", "b"]},
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda o: setitem(o.value.values, 0, 1.0 * u.mK**2),
        lambda o: setitem(o.value.coords["k"], 0, 1.0 / u.Mpc),
        lambda o: setitem(o.value.coords, "k", [1.0] / u.Mpc),
        lambda o: setitem(o.value.covariance, (0, 0), 1.0 * u.mK**4),
        lambda o: setitem(o.meta, "doi", "other"),
        lambda o: setitem(o.instrument["window"], (0, 0), 1.0),
    ],
)
def test_observations_cannot_be_changed_after_construction(change):
    obs = make_observation(np.random.default_rng(12))
    before = obs.value.values.copy(), obs.instrument["window"].copy()
    with pytest.raises((TypeError, ValueError)):
        change(obs)
    np.testing.assert_array_equal(obs.value.values, before[0])
    np.testing.assert_array_equal(obs.instrument["window"], before[1])


def test_an_observable_does_not_share_the_callers_arrays():
    values = np.arange(3.0) * u.mK
    obs = ToySignal(coords={"nu": [1.0, 2.0, 3.0] * u.MHz}, values=values)
    values[0] = 99 * u.mK
    assert obs.values[0] == 0 * u.mK


def test_a_populated_group_is_not_overwritten(tmp_path):
    rng = np.random.default_rng(13)
    first, second = toy_signal(rng), toy_banded_power(rng)
    with h5py.File(tmp_path / "obs.h5", "w") as f:
        first.to_hdf5(f.create_group("obs"))
        with pytest.raises(ValueError, match="not empty"):
            second.to_hdf5(f["obs"])
        back = Observable.from_hdf5(f["obs"])
    assert type(back) is ToySignal
    np.testing.assert_array_equal(back.values, first.values)


def test_plot_options_override_the_defaults():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    ax = toy_banded_power(np.random.default_rng(14)).plot(label="data", yerr=None)
    assert [c.get_label() for c in ax.containers] == ["data", "data"]
    plt.close(ax.figure)


def test_the_contract_check_skips_plotting_without_matplotlib(monkeypatch, tmp_path):
    import importlib.util

    find_spec = importlib.util.find_spec

    def hide_matplotlib(name, *args):
        return None if name == "matplotlib" else find_spec(name, *args)

    def fail(*args, **kwargs):
        raise AssertionError("plot was called")

    monkeypatch.setattr(importlib.util, "find_spec", hide_matplotlib)
    monkeypatch.setattr(ToySignal, "plot", fail)
    assert_observable_contract(toy_signal(np.random.default_rng(15)), tmp_path)
