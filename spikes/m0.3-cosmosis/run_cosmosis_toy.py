"""Run the toy posterior through CosmoSIS, built purely from Python (no ini file).

This is the A-prime path from the v2 plan: our posterior is a single function
returning ``(lnL, derived)``, wrapped by ``LikelihoodPipeline.from_likelihood_function``.

Examples
--------
    python run_cosmosis_toy.py --sampler nautilus --out runs/naut
    mpirun -n 4 python -m mpi4py run_cosmosis_toy.py --sampler multinest --out runs/mn --resume

Launch MPI runs with ``python -m mpi4py``: a plain launch hangs if the master rank
dies (e.g. on a corrupt checkpoint), because the workers never get told to stop.
"""

import argparse
import os
import time

import toy
from cosmosis import LikelihoodPipeline, MPIPool, run_cosmosis


def build_pipeline():
    ranges = [(toy.LOW, float(m), toy.HIGH) for m in toy.MEAN]
    return LikelihoodPipeline.from_likelihood_function(
        toy.log_like_with_derived,
        ranges,
        derived=["sum", f"vec#{toy.VEC_LEN}"],
        debug=True,
    )


def build_ini(sampler, out, resume, n_live):
    os.makedirs(out, exist_ok=True)
    ini = {
        "runtime": {"sampler": sampler, "verbosity": "quiet", "resume": str(resume)},
        "output": {"filename": os.path.join(out, "chain.txt"), "format": "text"},
        "nautilus": {
            "n_live": str(n_live),
            "resume": "T" if resume else "F",
            "seed": "42",
        },
        "multinest": {
            "live_points": str(n_live),
            "max_iterations": "100000",
            "multinest_outfile_root": os.path.join(out, "mn_"),
            "resume": "T" if resume else "F",
            "random_seed": "42",
            "feedback": "F",
            # Importance nested sampling: CosmoSIS default is on. MN_INS=F turns it off.
            "ins": os.environ.get("MN_INS", "T"),
        },
    }
    return ini


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=["nautilus", "multinest"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-live", type=int, default=400)
    args = ap.parse_args()

    ini = build_ini(args.sampler, args.out, args.resume, args.n_live)
    pipeline = build_pipeline()

    t0 = time.perf_counter()
    try:
        from mpi4py import MPI

        use_mpi = MPI.COMM_WORLD.Get_size() > 1
    except ImportError:
        use_mpi = False

    if use_mpi:
        with MPIPool() as pool:
            run_cosmosis(ini=ini, pool=pool, pipeline=pipeline)
            if pool.is_master():
                print(f"WALL {time.perf_counter() - t0:.2f} s")
    else:
        run_cosmosis(ini=ini, pipeline=pipeline)
        print(f"WALL {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
