import pytest

import numpy as np
from scipy.integrate import quad
from scipy.stats import expon, halfnorm, norm, uniform

from py21cmmc.systematics import Exponential, Flat, HalfNormal

PRIORS = [
    (Flat(3.0), lambda w: uniform(0, w)),
    (HalfNormal(2.0), lambda w: halfnorm(scale=w)),
    (Exponential(1.5), lambda w: expon(scale=w)),
    (HalfNormal(2.0, k_slope=2.0), lambda w: halfnorm(scale=w)),
]


@pytest.mark.parametrize("systematics, prior", PRIORS)
def test_lnlike_is_marginal(systematics, prior):
    """lnlike = log of the integral of N(d | m + s, sigma) p(s) over s."""
    sigma, k, m = 2.0, 0.5, 1.0
    p = prior(systematics.width(sigma, k))
    for d in [-10.0, 0.0, 3.0, 20.0]:
        like = quad(
            lambda s: norm.pdf(d, m + s, sigma) * p.pdf(s), 0, p.ppf(1 - 1e-12)
        )[0]
        assert np.isclose(systematics.lnlike(d, m, sigma, k), np.log(like))


def test_flat_inf_is_hera_likelihood():
    d = np.array([-100.0, -3.0, 0.0, 5.0])
    assert np.allclose(Flat(np.inf).lnlike(d, 1.0, 2.0, 0.3), norm.logcdf(d - 1, 0, 2))


@pytest.mark.parametrize("systematics, prior", PRIORS)
def test_sample(systematics, prior):
    rng = np.random.default_rng(1)
    k = np.full(1000, 0.4)
    s = np.concatenate([systematics.sample(np.ones_like(k), k, rng) for _ in range(20)])
    assert np.isclose(s.mean(), prior(systematics.width(1, 0.4)).mean(), rtol=0.05)


def test_width():
    sys = HalfNormal(scale=2.0, k_slope=1.0, k0=0.2)
    assert sys.width(3.0, 0.2) == 6.0
    assert sys.width(3.0, 0.4) == 3.0
    with pytest.raises(ValueError):
        Flat(np.inf).sample(np.ones(3), np.ones(3), np.random.default_rng())
