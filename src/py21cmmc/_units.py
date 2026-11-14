"""Unit handling shared by the observable types."""

from __future__ import annotations

from typing import Any

from astropy import units as u


def to_unit(q: Any, unit: u.UnitBase, what: str) -> u.Quantity:
    """Return a read-only copy of the quantity ``q``, converted to ``unit``.

    Raises
    ------
    TypeError
        If ``q`` is not a quantity.
    astropy.units.UnitConversionError
        If its unit cannot be converted to ``unit``. ``what`` names it in the message.
    """
    if not isinstance(q, u.Quantity):
        raise TypeError(f"The {what} must be a quantity in {unit}, got {type(q)}.")
    try:
        out = u.Quantity(q, unit, copy=True)
    except u.UnitConversionError:
        raise u.UnitConversionError(
            f"The {what} are in {q.unit}, which cannot be converted to {unit}."
        ) from None
    out.setflags(write=False)
    return out
