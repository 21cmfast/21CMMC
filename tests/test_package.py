"""Smoke tests for the package skeleton."""

import subprocess
import sys

import pytest

import py21cmmc


def test_version():
    assert py21cmmc.__version__ != "unknown"


def test_import_does_not_need_21cmfast():
    # The fast tier runs without 21cmFAST, so importing py21cmmc must not import it.
    code = "import sys, py21cmmc; sys.exit('py21cmfast' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


@pytest.mark.nightly
def test_21cmfast_input_api(py21cmfast):
    # Canary for the 21cmFAST input API that v2's parameter routing relies on
    # (checked in M0.4 against 4.3.1).
    inputs = py21cmfast.InputParameters.from_template(
        ["simple", "size-small"], random_seed=1
    )
    new = inputs.evolve_input_structs(F_STAR10_ACG=-1.0)
    assert new.astro_params.F_STAR10_ACG == -1.0

    with pytest.raises(TypeError):
        inputs.evolve_input_structs(NOT_A_PARAMETER=1)

    # N_THREADS is part of SimulationOptions, so it enters the IC cache hash.
    assert hasattr(inputs.simulation_options, "N_THREADS")
