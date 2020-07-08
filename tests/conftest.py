import pytest

from pathlib import Path
from py21cmfast import global_params


@pytest.fixture(scope="session")
def tmpdirec(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def cache(tmpdirec) -> Path:
    pth = tmpdirec / "cache"
    pth.mkdir()
    return pth


@pytest.fixture(scope="function", autouse=True)
def setup_package():
    txt = "".join(a.decode() for a in global_params.external_table_path)
    txt.replace(r"\x00", "")
    print("External table path: ", txt)
