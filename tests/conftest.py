"""Test configuration: the fast and nightly test tiers.

The fast tier runs on every push and pull request, without 21cmFAST installed. Mark a
test ``nightly`` when it needs 21cmFAST or a full simulation; those run in the
scheduled Nightly workflow (``pytest -m nightly``) and are skipped otherwise.
"""

import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip nightly tests unless they were selected with ``-m nightly``."""
    if "nightly" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="nightly tier: run with `pytest -m nightly`")
    for item in items:
        if "nightly" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def py21cmfast():
    """The ``py21cmfast`` module, or skip the test if 21cmFAST is not installed."""
    if importlib.util.find_spec("py21cmfast") is None:
        pytest.skip("21cmFAST is not installed")
    import py21cmfast

    return py21cmfast
