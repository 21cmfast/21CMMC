"""Smoke tests for the package skeleton."""

import importlib
import sys

import pytest


def test_import_does_not_need_21cmfast(monkeypatch):
    # The fast tier runs without 21cmFAST, so py21cmmc must import without it.
    # A None entry in sys.modules makes any `import py21cmfast` raise ImportError.
    monkeypatch.setitem(sys.modules, "py21cmfast", None)
    monkeypatch.delitem(sys.modules, "py21cmmc", raising=False)
    importlib.import_module("py21cmmc")


@pytest.mark.nightly
def test_21cmfast_input_api(py21cmfast):
    # Canary for the 21cmFAST input API that our parameter routing relies on
    # (see https://github.com/21cmfast/21CMMC/issues/97).
    inputs = py21cmfast.InputParameters.from_template(
        ["simple", "size-small"], random_seed=1
    )
    new = inputs.evolve_input_structs(F_STAR10_ACG=-1.0)
    assert new.astro_params.F_STAR10_ACG == -1.0

    with pytest.raises(TypeError):
        inputs.evolve_input_structs(NOT_A_PARAMETER=1)

    # N_THREADS is part of SimulationOptions, so it enters the IC cache hash.
    assert hasattr(inputs.simulation_options, "N_THREADS")
