"""Tests of the Planck optical depth likelihood."""

from py21cmmc import LikelihoodPlanck


def test_planck_tau_defaults():
    """Default tau constraint is 0.0569 +0.0081 -0.0066 (Qin+20, 2006.16828)."""
    lk = LikelihoodPlanck()
    assert lk.tau_mean == 0.0569
    assert lk.tau_sigma_u == 0.0081
    assert lk.tau_sigma_l == 0.0066
