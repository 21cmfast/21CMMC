"""Does an exception on the master rank leave CosmoSIS MPIPool workers hanging?

Mirrors run_cosmosis: the master drives the sampler, workers sit in pool.wait().
Run via hang_test.sh, which compares a plain launch with `python -m mpi4py`.
"""

from cosmosis import MPIPool

with MPIPool() as pool:
    if pool.is_master():
        pool.map(abs, range(8))  # workers are alive and serving
        raise RuntimeError("simulated master failure (e.g. corrupt checkpoint)")
    else:
        pool.wait()
