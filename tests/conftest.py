"""Shared pytest fixtures for the test suite."""

from pathlib import Path

import pytest

# On macOS, py21cmfast's compiled extension links against an OpenMP runtime
# (e.g. Homebrew's libomp). If it initializes OpenMP before torch (imported
# indirectly via py21cmemu) gets a chance to initialize its own bundled
# runtime, the two can conflict and segfault. Importing torch first avoids
# this, so it must stay above the py21cmfast import below.
import torch  # noqa: F401
from py21cmfast import global_params


@pytest.fixture(scope="session")
def tmpdirec(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def cache(tmpdirec) -> Path:
    pth = tmpdirec / "cache"
    pth.mkdir()
    return pth


@pytest.fixture(autouse=True)
def setup_package():
    txt = "".join(a.decode() for a in global_params.external_table_path)
    txt.replace(r"\x00", "")
    print("External table path: ", txt)
