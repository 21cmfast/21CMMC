# M0.3 spike: can CosmoSIS drive 21cmMC v2's samplers?

Tracks [21cmfast/21CMMC#96](https://github.com/21cmfast/21CMMC/issues/96). Throwaway code: it tests option A′ from the v2 architecture plan, where our posterior is one Python function returning `(lnL, derived)` and CosmoSIS runs the sampler around it via `LikelihoodPipeline.from_likelihood_function`.

**Verdict: conditional go.** All four checks pass or have a cheap fix, but two failure modes need fixing before long HPC runs (see "Must fix").

## Results (2026-10-23, Linux container, 4 cores, OpenMPI 4.1.6, CosmoSIS 3.26, nautilus 1.0.6)

| Check | Result |
| --- | --- |
| 1. Derived values under MPI | **Pass.** With `mpirun -n 4`, nautilus and MultiNest both write a scalar and an 8-element array (`derived/vec#8`) on every row, matching the inputs exactly. Moments and ln Z match the analytic toy. |
| 2. Resume after a kill | **nautilus: pass, with a serious caveat.** Resuming from an intact checkpoint finishes correctly (the result is identical to an uninterrupted run with the same seed; it needed 6,100 more of 16,600 evaluations). But **3 of 13 SIGTERM kills left a corrupt checkpoint**, so the run could not resume at all. **MultiNest: posteriors and derived values resume fine, but the default importance-nested-sampling evidence is biased after a resume** (+0.4 to +1.05 in ln Z, 4 of 4 trials; it is exact when uninterrupted). With `ins=F` resumed runs agree with uninterrupted ones. |
| 3. Overhead per evaluation | **Pass.** 0.18 ms per call through CosmoSIS's pipeline (vs 4.5 µs raw); +0.10 ms per evaluation end to end vs calling nautilus directly. Negligible next to 21cmFAST. |
| 4. Clean install | **conda: pass** (`micromamba create -c conda-forge cosmosis`, 71 s, binaries, MPI MultiNest included). **pip: conditional.** No wheels on PyPI, so `pip install cosmosis` fails without gfortran. With gfortran it takes ~77 s, but MPI MultiNest/PolyChord are only built if `MPIFC=mpif90` is set at install time. It also pulls in heavy dependencies (pandas, scikit-learn, seaborn, pytest, …). **Not yet tested: macOS and a real HPC system.** |

## Must fix before long HPC runs

1. **nautilus checkpoints are not written atomically.** `Sampler.write(overwrite=True)` deletes the old file and then writes a new one (`unlink`, then `h5py.File(..., 'x')`), and `write_shell_update` edits the file in place (`'r+'`). A kill mid-write loses the whole run. With this cheap toy a kill hit the write about 1 time in 4; with minutes-long 21cmFAST evaluations it will be rarer, but over many walltime kills it is a matter of when. Fix: write to a temporary file and `os.replace` it into place (an upstream PR to nautilus); until then, keep a copy of the last good checkpoint in our driver.
2. **A failed resume hangs the job.** When the master rank dies (e.g. on the corrupt checkpoint), a plain `mpirun python …` launch leaves the workers alive until walltime. Launching with `python -m mpi4py …` aborts all ranks within seconds (tested). Our sbatch templates and docs must use it.

## Should document

- Prefer conda for installs. For pip, document gfortran plus `MPIFC=mpif90`.
- With MultiNest, don't trust ln Z from a resumed run with `ins=T`; use nautilus for evidence, or `ins=F`.
- CosmoSIS as a *core* dependency means `pip install 21cmMC` needs a Fortran compiler. That is a decision for the team (R18).

## Files

| File | What it does |
| --- | --- |
| `toy.py` | Analytic toy: 5D correlated Gaussian in a uniform box, plus derived values (scalar + 8-vector). Set `TOY_SLEEP` to make evaluations slow |
| `run_cosmosis_toy.py` | Runs the toy through CosmoSIS from Python only (no ini file). `--sampler nautilus\|multinest`, `--resume`. `MN_INS=F` turns MultiNest INS off |
| `check_output.py` | Checks a chain: derived values on every row, moments, ln Z |
| `resume_test.sh` | Kill (SIGTERM) mid-run under MPI, then resume and check |
| `kill_trials.sh` | Repeated kills at different times; reports whether the nautilus checkpoint survives |
| `bench_overhead.py` | Per-call and end-to-end overhead vs direct nautilus |
| `hang_repro.py`, `hang_test.sh` | Master-rank failure under MPIPool, plain vs `python -m mpi4py` |

## Reproduce

```bash
# conda (recommended)
micromamba create -p ./env -c conda-forge python=3.12 cosmosis nautilus-sampler mpi4py openmpi
# or pip (needs gfortran and an MPI compiler wrapper)
MPIFC=mpif90 pip install cosmosis nautilus-sampler mpi4py

python run_cosmosis_toy.py --sampler nautilus --out runs/naut && python check_output.py runs/naut/chain.txt
mpirun -n 4 python -m mpi4py run_cosmosis_toy.py --sampler multinest --out runs/mn && python check_output.py runs/mn/chain.txt
PYTHON=$(which python) TOY_SLEEP=0.02 ./resume_test.sh nautilus runs/naut_resume 45
python bench_overhead.py runs/bench
```

On an HPC system, run the same commands through `srun` from an sbatch script, and kill the first resume-test run with `scancel`.
