import os

import pytest

from py21cmfast import global_params


@pytest.fixture(scope="session")
def tmpdirec(tmpdir_factory):
    return tmpdir_factory.mktemp("data")


@pytest.fixture(scope="function", autouse=True)
def setup_package():
    txt = "".join(a.decode() for a in global_params.external_table_path)
    txt.replace(r"\x00", "")
    print("External table path: ", txt)
