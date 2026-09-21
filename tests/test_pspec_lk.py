from py21cmmc import LikelihoodPspec, build_computation_chain, core


def test_pspec_lk():
    """Basic smoke test for the pspec_likelihood-based likelihood.

    Uses the same built-in HERA H1C IDR3 upper-limit data as
    ``test_hera_lk.py``, but computed through :class:`LikelihoodPspec`
    (backed by ``pspec_likelihood``) instead of
    :class:`~py21cmmc.likelihood.Likelihood1DPowerLightconeUpper`.
    """
    lk = LikelihoodPspec.from_builtin_data("HERA_H1C_IDR3")

    c21cmemu = core.Core21cmEMU()
    chain = build_computation_chain(
        [c21cmemu],
        [lk],
        setup=True,
    )
    chain({})
