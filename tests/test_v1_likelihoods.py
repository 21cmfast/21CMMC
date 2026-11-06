"""The generic likelihoods reproduce the likelihoods of 21cmMC v1.

Each ``v1_*`` function transcribes the log-likelihood of the v1 class it is named
after, as a function of the model value. A v2 likelihood reproduces it when the two
differ by the same constant at every model, which changes nothing in a posterior.
The tutorial "Likelihoods from 21cmMC v1" shows the same comparisons, with plots.
"""

import attrs
import numpy as np
import pytest
from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline
from yabf.likelihoods import AsymmetricGaussian, Gaussian, TabulatedPDF

RNG = np.random.default_rng(42)
NU = np.linspace(50.0, 100.0, 21)
SIGMA = RNG.uniform(0.01, 0.05, NU.size)
GLOBAL_DATA = -0.2 * np.exp(-0.5 * ((NU - 78) / 6) ** 2) + SIGMA * RNG.normal(
    size=NU.size
)
T_ARCADE = 1.19 * (1.42 / (1 + 15.0)) ** -2.62 * (1 + 15.0)  # in K, at z = 15


def trough(depth: float) -> np.ndarray:
    """Return a Gaussian absorption trough of the given depth, in K, at NU."""
    return -depth * np.exp(-0.5 * ((NU - 78) / 6) ** 2)


def v1_neutral_fraction(x_hi: float) -> float:
    """Return the log-likelihood of ``LikelihoodNeutralFraction``, one-sided."""
    return -0.5 * ((0.06 - x_hi) / 0.05) ** 2 if x_hi > 0.06 else 0.0


def v1_neutral_fraction_two_sided(x_hi: float) -> float:
    """Return the log-likelihood of ``LikelihoodNeutralFractionTwoSided``."""
    return -0.5 * ((0.06 - x_hi) / 0.05) ** 2


def v1_arcade(t_radio: float) -> float:
    """Return the log-likelihood of ``LikelihoodArcade``, one-sided."""
    chi2 = ((T_ARCADE - t_radio) / (0.2 * T_ARCADE)) ** 2
    return -0.5 * chi2 * np.heaviside(t_radio - T_ARCADE, 0)


def v1_edges_timing(nu_min: float) -> float:
    """Return the log-likelihood of ``LikelihoodEDGES``, without the width."""
    return -0.5 * ((nu_min - 78.0) / 1.0) ** 2


def v1_global_signal(depth: float) -> float:
    """Return the log-likelihood of ``LikelihoodGlobalSignal``."""
    return -0.5 * np.sum((GLOBAL_DATA - trough(depth)) ** 2 / SIGMA**2)


@pytest.mark.parametrize(
    ("v1", "likelihood", "data", "models", "as_model"),
    [
        (
            v1_neutral_fraction,
            AsymmetricGaussian(sigma_lower=0.05, sigma_upper=1e4),
            [0.06],
            np.linspace(0, 1, 101),
            np.atleast_1d,
        ),
        (
            v1_neutral_fraction_two_sided,
            Gaussian.from_sigma([0.05]),
            [0.06],
            np.linspace(0, 1, 101),
            np.atleast_1d,
        ),
        (
            v1_arcade,
            AsymmetricGaussian(sigma_lower=0.2 * T_ARCADE, sigma_upper=1e6 * T_ARCADE),
            [T_ARCADE],
            np.linspace(0.5, 3, 101) * T_ARCADE,
            np.atleast_1d,
        ),
        (
            v1_edges_timing,
            Gaussian.from_sigma([1.0]),
            [78.0],
            np.linspace(60, 100, 101),
            np.atleast_1d,
        ),
        (
            v1_global_signal,
            Gaussian.from_sigma(SIGMA),
            GLOBAL_DATA,
            np.linspace(0, 0.5, 101),
            trough,
        ),
    ],
    ids=["neutral_fraction", "two_sided", "arcade", "edges_timing", "global_signal"],
)
def test_likelihoods_agree_up_to_a_constant(v1, likelihood, data, models, as_model):
    """Check that v2 differs from v1 by the same constant at every model."""
    v1_values = np.array([v1(m) for m in models])
    v2_values = np.array([likelihood.loglike(as_model(m), data) for m in models])
    assert np.ptp(v2_values - v1_values) < 1e-8


def test_flagged_data_reproduce_the_bright_end_cut_of_the_luminosity_function():
    """Check that flagging data with NaN reproduces the bright-end cut.

    ``LikelihoodLuminosityFunction`` left out the bins brighter than ``mag_brightest``.
    """
    muv = np.linspace(-22, -14, 17)
    sigma = RNG.uniform(1e-5, 1e-4, muv.size)
    data = RNG.uniform(1e-5, 1e-2, muv.size)
    faint = muv > -20
    flagged = np.where(faint, data, np.nan)
    likelihood = Gaussian.from_sigma(sigma)

    models = RNG.uniform(1e-5, 1e-2, (50, muv.size))
    v1 = [-0.5 * np.sum(((data - m) ** 2 / sigma**2)[faint]) for m in models]
    v2 = [likelihood.loglike(m, flagged) for m in models]
    assert np.ptp(np.subtract(v2, v1)) < 1e-8


@pytest.mark.parametrize(
    ("mean", "upper", "lower"),
    [(0.0569, 0.0081, 0.0066), (19.0, 4.0, 2.0)],
    ids=["planck_tau", "edges_width"],
)
def test_asymmetric_errors_agree_with_v1_at_the_mean_and_one_sigma(mean, upper, lower):
    """Check that the split normal agrees with v1 at the mean and one sigma.

    ``LikelihoodPlanck`` and the width of ``LikelihoodEDGES`` use Barlow's
    variable-width Gaussian. The split normal, with the measurement's upper error as
    its width for data below the model, agrees with it at the mean and one standard
    deviation either side of it.
    """

    def v1(model: float) -> float:
        """Return Barlow's variable-width Gaussian."""
        width = upper * lower + (upper - lower) * (model - mean)
        return -0.5 * (model - mean) ** 2 / width

    likelihood = AsymmetricGaussian(sigma_lower=upper, sigma_upper=lower)
    at_mean = likelihood.loglike([mean], [mean])
    for model in (mean - lower, mean + upper):
        v2 = likelihood.loglike([model], [mean]) - at_mean
        np.testing.assert_allclose(v2, v1(model), rtol=1e-12)
        np.testing.assert_allclose(v2, -0.5, rtol=1e-12)


def test_a_tabulated_density_of_the_model_reproduces_the_greig_likelihood():
    """Check that a tabulated density of the residual reproduces ``LikelihoodGreig``.

    ``LikelihoodGreig`` used a tabulated density of the neutral fraction itself.
    Tabulating it as the density of the residual, with the data at zero, gives the
    same likelihood up to the interpolation, here on a fine grid.
    """
    x = np.linspace(0, 1, 1000)
    pdf = stats.beta(3, 5).pdf(x)
    positive = pdf > 0
    spline = InterpolatedUnivariateSpline(x[positive], np.log(pdf[positive]))
    likelihood = TabulatedPDF(x=-x[::-1], pdf=pdf[::-1])

    models = np.linspace(0.02, 0.9, 89)
    v1 = spline(models)
    v2 = np.array([likelihood.loglike([m], [0.0]) for m in models])
    assert np.ptp(v2 - v1) < 1e-4


@attrs.frozen
class PowerSpectrum:
    """A power-law model power spectrum with a 15 % uncertainty of its own."""

    amplitude: float
    k: np.ndarray = attrs.field(factory=lambda: np.geomspace(0.1, 1, 8))

    @property
    def values(self) -> np.ndarray:
        """Return the power spectrum at k."""
        return self.amplitude * 100 * (self.k / 0.3) ** 0.5

    @property
    def covariance(self) -> np.ndarray:
        """Return the covariance of the model's uncertainty."""
        return np.diag((0.15 * self.values) ** 2)


def test_a_model_covariance_adds_the_determinant_that_v1_left_out():
    """Check that a model covariance adds the log-determinant that v1 left out.

    ``Likelihood1DPowerCoeval`` added a fractional model error to the noise but
    computed only the chi-squared. The v2 Gaussian with that model covariance differs
    from it by exactly the log-determinant, which depends on the model.
    """
    noise = 0.1 * PowerSpectrum(1.0).values
    likelihood = Gaussian(np.diag(noise**2))
    data = likelihood.simulate(PowerSpectrum(1.0), RNG)

    for amplitude in RNG.uniform(0.5, 2, 20):
        model = PowerSpectrum(amplitude)
        variance = noise**2 + (0.15 * model.values) ** 2
        v1 = -0.5 * np.sum((model.values - data) ** 2 / variance)
        determinant = -0.5 * np.sum(np.log(2 * np.pi * variance))
        np.testing.assert_allclose(likelihood.loglike(model, data), v1 + determinant)
