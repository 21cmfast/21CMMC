"""Overhead of driving samplers through CosmoSIS, versus calling nautilus directly.

(a) per-call: raw ``log_like_with_derived`` vs ``LikelihoodPipeline.run_results``;
(b) end-to-end, serial: direct nautilus (with blobs) vs CosmoSIS+nautilus,
    same live points and seed, cheap likelihood so overhead dominates.

Usage: python bench_overhead.py OUTDIR
"""

import os
import sys
import time

import numpy as np
import run_cosmosis_toy as rc
import toy
from nautilus import Sampler


def per_call(n=20000):
    rng = np.random.default_rng(1)
    pts = rng.uniform(toy.LOW, toy.HIGH, size=(n, toy.NDIM))
    t = time.perf_counter()
    for p in pts:
        toy.log_like_with_derived(p)
    raw = (time.perf_counter() - t) / n

    pipe = rc.build_pipeline()
    t = time.perf_counter()
    for p in pts:
        pipe.run_results(p)
    cs = (time.perf_counter() - t) / n
    return raw, cs


def direct_nautilus(out, n_live):
    def prior(u):
        return toy.LOW + (toy.HIGH - toy.LOW) * u

    def like(x):
        lnl, d = toy.log_like_with_derived(x)
        return lnl, d["sum"], d["vec"]

    blobs = [("sum", float), ("vec", float, toy.VEC_LEN)]
    t = time.perf_counter()
    s = Sampler(
        prior,
        like,
        n_dim=toy.NDIM,
        n_live=n_live,
        seed=42,
        blobs_dtype=blobs,
        filepath=os.path.join(out, "direct.hdf5"),
    )
    s.run(verbose=False)
    wall = time.perf_counter() - t
    return wall, s.n_like, s.log_z


def cosmosis_nautilus(out, n_live):
    from cosmosis import run_cosmosis

    ini = rc.build_ini("nautilus", os.path.join(out, "cs"), False, n_live)
    t = time.perf_counter()
    run_cosmosis(ini=ini, pipeline=rc.build_pipeline())
    wall = time.perf_counter() - t
    with open(os.path.join(out, "cs", "chain.txt")) as f:
        ev = [ln for ln in f if ln.startswith("#evaluations=")]
    return wall, int(ev[0].split("=")[1])


def main(out):
    os.makedirs(out, exist_ok=True)
    raw, cs = per_call()
    print(
        f"(a) per call: raw {raw * 1e6:.1f} us, through CosmoSIS pipeline "
        f"{cs * 1e6:.1f} us  -> overhead {1e6 * (cs - raw):.1f} us/eval"
    )

    wd, nd, lz = direct_nautilus(out, 400)
    wc, nc = cosmosis_nautilus(out, 400)
    print(
        f"(b) direct nautilus : {wd:.1f} s for {nd} evals ({1e3 * wd / nd:.2f} ms/eval), log_z {lz:.3f}"
    )
    print(
        f"    CosmoSIS+nautilus: {wc:.1f} s for {nc} evals ({1e3 * wc / nc:.2f} ms/eval)"
    )
    print(f"    difference: {1e3 * (wc / nc - wd / nd):+.2f} ms/eval")


if __name__ == "__main__":
    main(sys.argv[1])
