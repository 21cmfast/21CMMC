"""M0.4: check the 21cmFAST API surface that the v2 design relies on.

Each check prints OK or FAIL with what it found. Run it against a new 21cmFAST
release before bumping the pin:

    python check_api.py            # quick checks only (~1 s)
    python check_api.py --run      # also runs a tiny global evolution and lightcone (~5 s)
"""

import argparse
import inspect
import tempfile
import warnings

import py21cmfast as p21
from py21cmfast.io.caching import OutputCache
from py21cmfast.wrapper import exceptions as ex

TEMPLATE = ["simple", "size-small"]
FAILED = []


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def base():
    return p21.InputParameters.from_template(TEMPLATE, random_seed=1)


def raises(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return e
    return None


def main(run):
    print("21cmFAST", p21.__version__)
    inp = base()

    # Public entry points.
    names = [
        "InputParameters",
        "run_global_evolution",
        "generate_lightcone",
        "run_lightcone",
        "run_coeval",
        "RectilinearLightconer",
        "OutputCache",
        "CacheConfig",
    ]
    missing = [n for n in names if not hasattr(p21, n)]
    check("public entry points", not missing, f"missing {missing}" if missing else "")
    sig = inspect.signature(p21.run_global_evolution)
    check(
        "run_global_evolution(inputs, source_model, progressbar, overdensity_z0)",
        list(sig.parameters)
        == ["inputs", "source_model", "progressbar", "overdensity_z0"],
        str(sig),
    )

    # Parameter routing.
    e = raises(lambda: inp.evolve_input_structs(NOT_A_PARAM=1))
    check("unknown name raises TypeError", isinstance(e, TypeError), repr(e))
    new = inp.evolve_input_structs(F_STAR10_ACG=-1.0)
    check(
        "new name F_STAR10_ACG reaches C",
        abs(new.astro_params.cdict["F_STAR10_ACG"] - 0.1) < 1e-12,
    )
    e = raises(lambda: inp.evolve_input_structs(F_STAR10=-1.0))
    check(
        "deprecated name F_STAR10 is refused by evolve (ValueError)",
        isinstance(e, ValueError),
        repr(e)[:100],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = p21.InputParameters.from_template(TEMPLATE, random_seed=1, F_STAR10=-1.0)
    e = raises(lambda: old.evolve_input_structs(F_STAR10_ACG=-0.5))
    check(
        "a base built with an old name then refuses the new name (ValueError)",
        isinstance(e, ValueError),
        repr(e)[:100],
    )

    # Dependent defaults are frozen by evolve, but follow on a fresh build.
    fresh = p21.InputParameters.from_template(
        TEMPLATE, random_seed=1, F_STAR10_ACG=-1.0
    )
    frozen = new.astro_params.F_STAR7_MCG
    check(
        "F_STAR7_MCG does not follow F_STAR10_ACG under evolve",
        frozen != fresh.astro_params.F_STAR7_MCG,
        f"evolve {frozen} vs fresh {fresh.astro_params.F_STAR7_MCG}",
    )

    # Invalid values: plain ValueError, not ParameterError.
    e = raises(lambda: inp.evolve_input_structs(F_STAR10_ACG=0.5))
    check(
        "out-of-range value raises ValueError (not ParameterError)",
        isinstance(e, ValueError) and not isinstance(e, ex.ParameterError),
        repr(e)[:80],
    )

    # Exceptions used by the failure policy.
    param = {
        "GSLError",
        "PhotonConsError",
        "TableGenerationError",
        "TableEvaluationError",
        "InfinityorNaNError",
        "MassDepZetaError",
    }
    fatal = {"FileIOError", "ArgumentValueError", "MemoryAllocError"}
    ok = all(issubclass(getattr(ex, n), ex.ParameterError) for n in param) and all(
        issubclass(getattr(ex, n), ex.FatalCError) for n in fatal
    )
    check("exception hierarchy (ParameterError / FatalCError)", ok)

    # N_THREADS: part of the IC (matter_cosmo) cache hash.
    h0 = OutputCache._get_hashes(inp)["matter_cosmo"]
    h1 = OutputCache._get_hashes(inp.evolve_input_structs(N_THREADS=4))["matter_cosmo"]
    check("N_THREADS changes the IC cache hash", h0 != h1)

    if not run:
        return

    small = inp.evolve_input_structs(HII_DIM=25, DIM=50, BOX_LEN=100, N_THREADS=2)
    g = p21.run_global_evolution(small)
    check("run_global_evolution runs", "neutral_fraction" in g.quantities)

    want = (
        "brightness_temp",
        "neutral_fraction",
        "ionisation_rate_G12",
        "kinetic_temperature",
        "density",
    )
    lc = p21.RectilinearLightconer.between_redshifts(
        min_redshift=7.0,
        max_redshift=9.0,
        resolution=small.simulation_options.cell_size,
        quantities=want,
    )
    small = small.with_logspaced_redshifts(zmin=6.5, zmax=10.0, zstep_factor=1.1)
    cache = p21.OutputCache(tempfile.mkdtemp())
    out = p21.run_lightcone(lightconer=lc, inputs=small, cache=cache, write=False)
    extra = sorted(set(out.lightcones) - set(want))
    check(
        "lightcone returns the requested quantities",
        set(want) <= set(out.lightcones),
        f"extra: {extra}",
    )
    bad = p21.RectilinearLightconer.between_redshifts(
        min_redshift=7.0,
        max_redshift=9.0,
        resolution=small.simulation_options.cell_size,
        quantities=("xH_box",),
    )
    e = raises(
        lambda: p21.run_lightcone(
            lightconer=bad, inputs=small, cache=cache, write=False
        )
    )
    check("v3 quantity name xH_box is refused (ValueError)", isinstance(e, ValueError))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    main(ap.parse_args().run)
    print("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
    raise SystemExit(bool(FAILED))
