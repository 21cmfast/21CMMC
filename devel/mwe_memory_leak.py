import os
import tracemalloc

import psutil

import py21cmmc
from py21cmmc import mcmc

print(py21cmmc.__version__)

location = "cosmohammer_data"
model_name = "SimpleTest"
PROCESS = psutil.Process(os.getpid())
oldmem = 0


# Stuff to track memory usage.
tracemalloc.start()
snapshot = tracemalloc.take_snapshot()


def trace_print():
    global snapshot
    global oldmem

    snapshot2 = tracemalloc.take_snapshot()
    snapshot2 = snapshot2.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
            tracemalloc.Filter(False, tracemalloc.__file__),
        )
    )

    if snapshot is not None:
        thismem = PROCESS.memory_info().rss / 1024 ** 2
        diff = thismem - oldmem
        print(
            "===================== Begin Trace (TOTAL MEM={:1.4e} MB... [{:+1.4e} MB]):".format(
                thismem, diff
            )
        )
        top_stats = snapshot2.compare_to(snapshot, "lineno", cumulative=True)
        for stat in top_stats[:4]:
            print(stat)
        print("End Trace ===========================================")
        print()
        oldmem = thismem

    snapshot = snapshot2


class MyPrinterCore(py21cmmc.CoreCoevalModule):
    def build_model_data(self, ctx):
        trace_print()
        super().build_model_data(ctx)


if __name__ == "__main__":
    core = MyPrinterCore(
        redshift=[7, 8, 9],
        user_params={"HII_DIM": 50, "BOX_LEN": 125.0},
        regenerate=True,
    )

    datafiles = [
        location + "/" + model_name + "_mcmc_data_%s.npz" % z for z in core.redshift
    ]

    likelihood = py21cmmc.likelihood.Likelihood1DPowerCoeval(
        # All likelihood modules are prefixed by Likelihood*
        datafile=datafiles,  # All likelihoods have this, which specifies where to write/read data
        noisefile=None,  # All likelihoods have this, specifying where to find noise profiles.
        logk=False,  # Should the power spectrum bins be log-spaced?
        min_k=0.1,  # Minimum k to use for likelihood
        max_k=1.0,  # Maximum ""
        simulate=True,
    )

    chain = mcmc.run_mcmc(
        core,
        likelihood,
        datadir=location,  # Directory for all outputs
        model_name=model_name,  # Filename of main chain output
        params={  # Parameter dict as described above.
            "HII_EFF_FACTOR": [30.0, 10.0, 50.0, 3.0],
            "ION_Tvir_MIN": [4.7, 4, 6, 0.1],
        },
        walkersRatio=2,  # The number of walkers will be walkersRatio*nparams
        burninIterations=0,  # Number of iterations to save as burnin. Recommended to leave as zero.
        sampleIterations=3,  # Number of iterations to sample, per walker.
        threadCount=1,  # Number of processes to use in MCMC (best as a factor of walkersRatio)
        continue_sampling=False,
    )
