"""Utility functions used throughout 21CMMC."""

try:
    from collections.abc import Iterable  # Python <= 3.9

except ImportError:
    from collections.abc import Iterable  # Python > 3.9


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
