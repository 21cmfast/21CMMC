"""Observables: the shared type for models, data and mocks, and the Observation.

Every observable type has the same interface, whatever it measures:

- ``.coords``: its named coordinate axes, each a :class:`~astropy.units.Quantity`;
- ``.values``: the measured or modelled values, with units;
- ``.covariance`` (or just ``.sigma``): the covariance of the values, if known;
- ``.at(coords)``: the observable interpolated onto other coordinates;
- ``.mask(where)``: the observable without the masked entries;
- ``.to_hdf5()`` / ``.from_hdf5()`` and ``.plot()``.

Values are stored flat, in one documented order: entry ``i`` of ``values`` sits at
``{axis: coords[axis][i]}``. This holds for regular grids and ragged data alike, such
as power spectra with different ``k`` bins in each redshift band.

A new type subclasses :class:`Observable` and declares its axes and units:

.. code-block:: python

    @attrs.frozen(eq=False)
    class GlobalSignal(Observable):
        axes: ClassVar = {"nu": u.MHz}
        unit: ClassVar = u.mK
        quantity: ClassVar = "T21"

The last axis is the one interpolated along; the axes before it label groups, such as
redshift bands, that interpolation never mixes.

An :class:`Observation` is a measured observable with its citation, metadata and
instrument description (window functions, chunk edges). It is stored and read with
HDF5, never with pickle.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

import attrs
import h5py
import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_TYPES: dict[str, type[Observable]] = {}


def _type_name(cls: type) -> str:
    """Return the name under which an observable type is stored in a file."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _as_dict(coords: Mapping[str, Any]) -> dict[str, Any]:
    """Copy ``coords`` into a plain dict."""
    return dict(coords)


@attrs.frozen(eq=False, kw_only=True)
class Observable:
    """The base observable type: values on named coordinate axes, with units.

    Subclasses declare the class attributes ``axes`` (axis names and their units,
    in order, the last being the one interpolated along), ``unit`` (the unit of the
    values) and ``quantity`` (the name of what is measured, such as ``"T21"``).

    Parameters
    ----------
    coords
        Each axis's coordinate of every entry: 1D quantities of the same length as
        ``values``, convertible to the axis units.
    values
        The values, a 1D quantity convertible to ``unit``.
    covariance
        The covariance of the values, an ``(n, n)`` quantity in ``unit**2``, or
        ``None`` if not known.
    """

    axes: ClassVar[Mapping[str, u.UnitBase]] = {}
    unit: ClassVar[u.UnitBase] = u.dimensionless_unscaled
    quantity: ClassVar[str] = "value"
    log_interp: ClassVar[bool] = False

    coords: Mapping[str, u.Quantity] = attrs.field(converter=_as_dict)
    values: u.Quantity = attrs.field()
    covariance: u.Quantity | None = attrs.field(default=None)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register the type, so that :meth:`from_hdf5` can find it by name."""
        super().__init_subclass__(**kwargs)
        _TYPES[_type_name(cls)] = cls

    def __attrs_post_init__(self) -> None:
        """Convert every array to its declared unit, and check their shapes."""
        if not self.axes:
            raise TypeError(
                f"{type(self).__name__} declares no axes; subclass Observable and set "
                "`axes`, `unit` and `quantity`."
            )
        values = _to(self.values, self.unit, "values").ravel()
        n = values.size
        coords = self._convert_coords(self.coords)
        if len(next(iter(coords.values()))) != n:
            raise ValueError(
                f"There are {n} values, but coordinates for "
                f"{len(next(iter(coords.values())))} entries."
            )
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "values", values)
        if self.covariance is not None:
            cov = _to(self.covariance, self.unit**2, "covariance")
            if cov.shape != (n, n):
                raise ValueError(
                    f"The covariance must have shape ({n}, {n}), got {cov.shape}."
                )
            if not np.allclose(cov.value, cov.value.T):
                raise ValueError("The covariance must be symmetric.")
            object.__setattr__(self, "covariance", cov)

    @classmethod
    def _convert_coords(cls, coords: Mapping[str, Any]) -> dict[str, u.Quantity]:
        """Return ``coords`` as 1D quantities in the axis units, of equal lengths."""
        if set(coords) != set(cls.axes):
            raise ValueError(
                f"{cls.__name__} has axes {list(cls.axes)}, got coordinates for "
                f"{list(coords)}."
            )
        out = {
            name: _to(coords[name], unit, f"coordinate {name!r}").ravel()
            for name, unit in cls.axes.items()
        }
        lengths = {name: c.size for name, c in out.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"The coordinates have different lengths: {lengths}.")
        return out

    def __len__(self) -> int:
        """Return the number of entries."""
        return self.values.size

    @property
    def sigma(self) -> u.Quantity | None:
        """The standard deviation of each value, or ``None`` with no covariance."""
        if self.covariance is None:
            return None
        return np.sqrt(np.diag(self.covariance))

    def mask(self, where: np.ndarray) -> Self:
        """Return the observable without the entries where ``where`` is true.

        As with :mod:`numpy.ma`, true marks an entry to remove, so
        ``ps.mask(ps.coords["k"] > 1 / u.Mpc)`` keeps ``k <= 1 / Mpc``.
        """
        where = np.asarray(where, dtype=bool)
        if where.shape != (len(self),):
            raise ValueError(
                f"The mask must have one entry per value ({len(self)}), got shape "
                f"{where.shape}."
            )
        keep = ~where
        cov = None if self.covariance is None else self.covariance[np.ix_(keep, keep)]
        return attrs.evolve(
            self,
            coords={k: v[keep] for k, v in self.coords.items()},
            values=self.values[keep],
            covariance=cov,
        )

    def at(self, coords: Mapping[str, u.Quantity]) -> Self:
        """Return the observable at other coordinates, interpolating linearly.

        Interpolation runs along the last axis, within each group of entries that
        share their other coordinates; every requested group must exist. A point
        outside a group's range is an error: there is no extrapolation. The
        covariance is carried through exactly: interpolation is a linear map ``W``
        of the values, so the new covariance is ``W C W^T``.
        """
        target = self._convert_coords(coords)
        weights = self._interpolation_weights(target)
        cov = None
        if self.covariance is not None:
            cov = weights @ self.covariance @ weights.T
        return attrs.evolve(
            self, coords=target, values=weights @ self.values, covariance=cov
        )

    def _interpolation_weights(self, target: Mapping[str, u.Quantity]) -> np.ndarray:
        """Return the matrix ``W`` mapping these values onto ``target`` coordinates."""
        *group_axes, along = self.axes
        groups: dict[tuple[float, ...], np.ndarray] = {}
        keys = self._group_keys(self.coords, group_axes)
        for key in dict.fromkeys(keys):
            idx = np.flatnonzero([k == key for k in keys])
            groups[key] = idx[np.argsort(self.coords[along].value[idx])]

        def scale(x: np.ndarray) -> np.ndarray:
            """Return the variable interpolated in: the coordinate, or its logarithm."""
            return np.log(x) if self.log_interp else x

        x_new = scale(target[along].value)
        weights = np.zeros((len(x_new), len(self)))
        for j, key in enumerate(self._group_keys(target, group_axes)):
            match = [g for g in groups if np.allclose(g, key, rtol=1e-10, atol=0)]
            if not match:
                raise ValueError(
                    f"No entries at {dict(zip(group_axes, key, strict=True))} to "
                    "interpolate from."
                )
            idx = groups[match[0]]
            x = scale(self.coords[along].value[idx])
            close = np.isclose(x, x_new[j], rtol=1e-10, atol=0)
            if close.any():
                weights[j, idx[np.argmax(close)]] = 1.0
                continue
            if not x[0] < x_new[j] < x[-1]:
                raise ValueError(
                    f"{along} = {target[along][j]} is outside the range of the data "
                    f"({self.coords[along][idx[0]]} to {self.coords[along][idx[-1]]}); "
                    "there is no extrapolation."
                )
            hi = np.searchsorted(x, x_new[j])
            t = (x_new[j] - x[hi - 1]) / (x[hi] - x[hi - 1])
            weights[j, idx[hi - 1]] = 1 - t
            weights[j, idx[hi]] = t
        return weights

    @staticmethod
    def _group_keys(
        coords: Mapping[str, u.Quantity], group_axes: list[str]
    ) -> list[tuple[float, ...]]:
        """Return each entry's group: its coordinates on every axis but the last."""
        n = len(next(iter(coords.values())))
        return [tuple(float(coords[a].value[i]) for a in group_axes) for i in range(n)]

    def to_hdf5(self, target: str | Path | h5py.Group) -> None:
        """Write the observable to an HDF5 file, or into a group of an open file."""
        with _group(target, "w") as g:
            g.attrs["observable"] = _type_name(type(self))
            g.attrs["axes"] = list(self.axes)
            _write_quantity(g, "values", self.values)
            coords = g.create_group("coords")
            for name, c in self.coords.items():
                _write_quantity(coords, name, c)
            if self.covariance is not None:
                _write_quantity(g, "covariance", self.covariance)

    @classmethod
    def from_hdf5(cls, source: str | Path | h5py.Group) -> Observable:
        """Read an observable written by :meth:`to_hdf5`.

        Called on :class:`Observable`, it returns whichever type was written, as long
        as that type has been imported. Called on a type, it checks that the file
        holds that type.
        """
        with _group(source, "r") as g:
            name = g.attrs["observable"]
            if name not in _TYPES:
                raise TypeError(
                    f"The file holds a {name}, which is not an imported observable "
                    "type; import the module that defines it first."
                )
            kind = _TYPES[name]
            if not issubclass(kind, cls):
                raise TypeError(f"The file holds a {name}, not a {_type_name(cls)}.")
            coords = {k: _read_quantity(g["coords"], k) for k in g.attrs["axes"]}
            cov = _read_quantity(g, "covariance") if "covariance" in g else None
            return kind(
                coords=coords, values=_read_quantity(g, "values"), covariance=cov
            )

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the values against the last axis, one line per group.

        With a covariance, each value gets an error bar of one standard deviation.
        Keyword arguments go to :meth:`matplotlib.axes.Axes.errorbar`.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        *group_axes, along = self.axes
        keys = self._group_keys(self.coords, group_axes)
        for key in dict.fromkeys(keys):
            idx = np.flatnonzero([k == key for k in keys])
            idx = idx[np.argsort(self.coords[along].value[idx])]
            label = ", ".join(
                f"{a} = {v:g} {self.axes[a]:latex_inline}"
                for a, v in zip(group_axes, key, strict=True)
            )
            sigma = None if self.sigma is None else self.sigma.value[idx]
            ax.errorbar(
                self.coords[along].value[idx],
                self.values.value[idx],
                yerr=sigma,
                label=label or None,
                **kwargs,
            )
        ax.set_xlabel(f"{along} [{self.axes[along]:latex_inline}]")
        ax.set_ylabel(f"{self.quantity} [{self.unit:latex_inline}]")
        if group_axes:
            ax.legend()
        return ax


def _as_meta(meta: Mapping[str, Any]) -> dict[str, str | int | float]:
    """Check that metadata values are strings or numbers, and copy them."""
    bad = [k for k, v in meta.items() if not isinstance(v, str | int | float)]
    if bad:
        raise TypeError(f"Metadata values must be strings or numbers; {bad} are not.")
    return dict(meta)


@attrs.frozen(eq=False, kw_only=True)
class Observation[T: Observable]:
    """A measured observable, with its citation, metadata and instrument.

    Parameters
    ----------
    value
        The measurement. Its covariance is the measurement covariance.
    citation
        How to cite the measurement.
    meta
        Further metadata, such as a DOI, a version or redshift labels, as strings or
        numbers.
    instrument
        Arrays describing the instrument, such as window functions or chunk edges, by
        name.
    """

    value: T = attrs.field(validator=attrs.validators.instance_of(Observable))
    citation: str = attrs.field(validator=attrs.validators.instance_of(str))
    meta: Mapping[str, str | int | float] = attrs.field(
        factory=dict, converter=_as_meta
    )
    instrument: Mapping[str, u.Quantity] = attrs.field(factory=dict, converter=_as_dict)

    @property
    def covariance(self) -> u.Quantity | None:
        """The measurement covariance."""
        return self.value.covariance

    def to_hdf5(self, target: str | Path | h5py.Group) -> None:
        """Write the observation to an HDF5 file, or into a group of an open file."""
        with _group(target, "w") as g:
            g.attrs["citation"] = self.citation
            self.value.to_hdf5(g.create_group("value"))
            meta = g.create_group("meta")
            for k, v in self.meta.items():
                meta.attrs[k] = v
            instrument = g.create_group("instrument")
            for k, v in self.instrument.items():
                _write_quantity(instrument, k, u.Quantity(v))

    @classmethod
    def from_hdf5(cls, source: str | Path | h5py.Group) -> Observation:
        """Read an observation written by :meth:`to_hdf5`."""
        with _group(source, "r") as g:
            return cls(
                value=Observable.from_hdf5(g["value"]),
                citation=str(g.attrs["citation"]),
                meta={k: _from_attr(v) for k, v in g["meta"].attrs.items()},
                instrument={
                    k: _read_quantity(g["instrument"], k) for k in g["instrument"]
                },
            )

    @classmethod
    def read(
        cls,
        path: str | Path,
        reader: Callable[[Path], Observation] | None = None,
    ) -> Observation:
        """Read an observation from ``path``.

        ``reader`` is any function from a path to an :class:`Observation`, for data
        in another format; by default the file is read with :meth:`from_hdf5`.
        """
        path = Path(path)
        obs = cls.from_hdf5(path) if reader is None else reader(path)
        if not isinstance(obs, Observation):
            raise TypeError(
                f"The reader returned a {type(obs).__name__}, not an Observation."
            )
        return obs


def _to(q: Any, unit: u.UnitBase, what: str) -> u.Quantity:
    """Return ``q`` as a float quantity in ``unit``, or raise naming ``what``."""
    if not isinstance(q, u.Quantity):
        raise TypeError(f"The {what} must be a quantity in {unit}, got {type(q)}.")
    try:
        return q.to(unit).astype(float)
    except u.UnitConversionError:
        raise u.UnitConversionError(
            f"The {what} are in {q.unit}, which cannot be converted to {unit}."
        ) from None


def _write_quantity(g: h5py.Group, name: str, q: u.Quantity) -> None:
    """Write a quantity as a dataset with its unit as an attribute."""
    ds = g.create_dataset(name, data=q.value)
    ds.attrs["unit"] = q.unit.to_string()


def _read_quantity(g: h5py.Group, name: str) -> u.Quantity:
    """Read a quantity written by :func:`_write_quantity`."""
    ds = g[name]
    return u.Quantity(ds[()], u.Unit(ds.attrs["unit"]))


def _from_attr(v: Any) -> str | int | float:
    """Turn an HDF5 attribute back into a plain string or number."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


@contextmanager
def _group(target: str | Path | h5py.Group, mode: str) -> Iterator[h5py.Group]:
    """Open ``target`` as an HDF5 group: a file path is opened, a group used as is."""
    if isinstance(target, h5py.Group):
        yield target
        return
    with h5py.File(target, mode) as f:
        yield f
