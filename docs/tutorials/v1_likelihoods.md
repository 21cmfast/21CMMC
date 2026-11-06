---
file_format: mystnb
kernelspec:
  name: python3
---

# Likelihoods from 21cmMC v1

In 21cmMC v1, each likelihood class combined three things: a summary of the
simulation, a data set, and a model of the noise. In v2 these are separate. The
simulation's stages compute the observable, an `Observation` holds the data, and a
generic likelihood from {mod}`yabf.likelihoods` models the noise, with two methods:
`loglike(model, data)` and `simulate(model, rng)`, which draws mock data from the
same distribution.

This page takes the v1 likelihoods one at a time. Each section writes the v1
log-likelihood as a function of the model value, gives the v2 likelihood that
replaces it, and compares the two. A difference that is the same for every model
changes nothing in a posterior, so the curves are compared after subtracting their
maxima.

```{code-cell} python
from collections.abc import Sequence
from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from yabf.likelihoods import AsymmetricGaussian, Gaussian, Likelihood, TabulatedPDF

DATA = Path("data")
rng = np.random.default_rng(5)


def loglike(likelihood: Likelihood, models: Sequence, data: object) -> np.ndarray:
    """Return the log-likelihood of ``data`` at each of ``models``."""
    return np.array([likelihood.loglike(np.atleast_1d(m), data) for m in models])


def compare(ax: plt.Axes, x: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> None:
    """Plot the v1 and v2 log-likelihoods against ``x``, each relative to its peak."""
    ax.plot(x, v1 - np.max(v1), lw=3, alpha=0.5, label="v1")
    ax.plot(x, v2 - np.max(v2), "k--", label="v2")
    ax.set_ylabel(r"$\ln L - \ln L_{\rm max}$")
    ax.legend()
```

## Gaussian errors

`LikelihoodLuminosityFunction`, `LikelihoodGlobalSignal`,
`LikelihoodNeutralFractionTwoSided` and the timing term of `LikelihoodEDGES` all
compute a chi-squared, $-\frac12 \sum_i (d_i - m_i)^2 / \sigma_i^2$. In v2 each is a
{class}`~yabf.likelihoods.Gaussian`, which adds the normalisation
$-\frac12 \sum_i \ln(2\pi\sigma_i^2)$: a constant, since the errors do not depend on
the model.

Take the UV luminosity function at $z = 6$ that v1 used by default. v1 left out
magnitudes brighter than $M_{\rm UV} = -20$ (`mag_brightest`); in v2, a NaN in the
data marks a value to leave out.

```{code-cell} python
lf = np.load(DATA / "LF_lfuncs_z6.npz")
muv, phi = lf["Muv"], lf["lfunc"]
sigma = np.load(DATA / "LF_sigmas_z6.npz")["sigma"]
faint = muv > -20


def schechter(muv: np.ndarray, alpha: float) -> np.ndarray:
    """Return a Schechter luminosity function, in Mpc^-3 mag^-1."""
    phi_star, m_star = 5e-4, -20.94
    x = 10 ** (-0.4 * (muv - m_star))
    return 0.4 * np.log(10) * phi_star * x ** (alpha + 1) * np.exp(-x)


def v1_luminosity_function(alpha: float) -> float:
    """Return the v1 log-likelihood: a chi-squared over the faint bins."""
    chi2 = (phi - schechter(muv, alpha)) ** 2 / sigma**2
    return -0.5 * np.sum(chi2[faint])


uvlf = Gaussian.from_sigma(sigma)
data = np.where(faint, phi, np.nan)

alphas = np.linspace(-2.2, -1.6, 121)
v1 = np.array([v1_luminosity_function(a) for a in alphas])
v2 = np.array([uvlf.loglike(schechter(muv, a), data) for a in alphas])

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.5))
left.errorbar(muv[faint], phi[faint], sigma[faint], fmt="o", label="used")
left.errorbar(
    muv[~faint], phi[~faint], sigma[~faint], fmt="o", mfc="w", label="left out"
)
left.plot(muv, schechter(muv, -1.9), "k")
left.set_yscale("log")
left.set_xlabel(r"$M_{\rm UV}$")
left.set_ylabel(r"$\phi$ [Mpc$^{-3}$ mag$^{-1}$]")
left.legend()
compare(right, alphas, v1, v2)
right.set_xlabel(r"faint-end slope $\alpha$")
print("largest difference, after the constant:", np.ptp(v2 - v1))
```

The two agree to rounding error. Here the model is evaluated at the data's
magnitudes directly; v1 interpolated it there with a cubic spline in
$\log_{10}\phi$, and in v2 moving a model to the data's coordinates is the job of the
observable.

## One-sided limits

`LikelihoodNeutralFraction` uses the limit on the neutral fraction at $z = 5.9$
from dark pixels (McGreer et al. 2015): no penalty when $x_{\rm HI} \le 0.06$, and a
half-Gaussian of width 0.05 above it. `LikelihoodArcade` is the same for the radio
background, with a width of 20 % of the ARCADE excess.

This is not a probability density of the data: it is flat for every measurement
above the model, so it cannot be normalised, and mock data cannot be drawn from it.
The {class}`~yabf.likelihoods.AsymmetricGaussian` (a split normal) is the closest
proper likelihood. It has one width for data below the model, and another above.
The v1 penalty applies when the data, 0.06, are below the model, so that is
`sigma_lower = 0.05`; a very wide `sigma_upper` makes the other side flat:

```{code-cell} python
def v1_neutral_fraction(x_hi: float) -> float:
    """Return the v1 log-likelihood of the McGreer et al. limit."""
    return -0.5 * ((0.06 - x_hi) / 0.05) ** 2 if x_hi > 0.06 else 0.0


mcgreer = AsymmetricGaussian(sigma_lower=0.05, sigma_upper=1e4)

x_hi = np.linspace(0, 0.4, 201)
v1 = np.array([v1_neutral_fraction(x) for x in x_hi])
v2 = loglike(mcgreer, x_hi, [0.06])

fig, ax = plt.subplots(figsize=(5, 3.5))
compare(ax, x_hi, v1, v2)
ax.set_xlabel(r"model $x_{\rm HI}(z=5.9)$")
print("largest difference, after the constant:", np.ptp(v2 - v1))
```

Over every neutral fraction a model can have, they agree to $10^{-10}$. The cost of
the wide side shows only in mocks, which scatter far above the model. For mocks that
resemble the measurement, give `sigma_upper` a realistic width.

## Asymmetric errors

`LikelihoodPlanck` uses the optical depth $\tau = 0.0569^{+0.0081}_{-0.0066}$ by
default (arXiv:2006.16828), and the width term of `LikelihoodEDGES` an asymmetric error on the
width of the trough. Both use a "variable-width Gaussian" (Barlow 2004),
$-\frac12 (\tau - \bar\tau)^2 / [\sigma_+\sigma_- + (\sigma_+ - \sigma_-)(\tau -
\bar\tau)]$.

In v2, asymmetric errors are an `AsymmetricGaussian`. Mind the names: a model
*above* the measured value means data *below* the model, so the measurement's upper
error $\sigma_+$ is `sigma_lower`.

```{code-cell} python
tau_mean, sigma_plus, sigma_minus = 0.0569, 0.0081, 0.0066


def v1_planck(tau: float) -> float:
    """Return the v1 log-likelihood of tau: a variable-width Gaussian."""
    width = sigma_plus * sigma_minus + (sigma_plus - sigma_minus) * (tau - tau_mean)
    return -0.5 * (tau - tau_mean) ** 2 / width


planck = AsymmetricGaussian(sigma_lower=sigma_plus, sigma_upper=sigma_minus)

tau = np.linspace(0.025, 0.1, 301)
v1 = np.array([v1_planck(t) for t in tau])
v2 = loglike(planck, tau, [tau_mean])

fig, ax = plt.subplots(figsize=(5, 3.5))
compare(ax, tau, v1, v2)
for t in (tau_mean - sigma_minus, tau_mean, tau_mean + sigma_plus):
    ax.axvline(t, color="grey", lw=0.5)
ax.set_ylim(-15, 1)
ax.set_xlabel(r"model $\tau$");
```

The two agree at the measured value and one standard deviation either side of it
(the grey lines), but not beyond. The v1 form falls off more slowly above the mean,
and faster below it: its width shrinks to zero at
$\tau = \bar\tau - \sigma_+\sigma_-/(\sigma_+ - \sigma_-)$, where the
log-likelihood goes to $-\infty$:

```{code-cell} python
tau_mean - sigma_plus * sigma_minus / (sigma_plus - sigma_minus)
```

Below that, the v1 log-likelihood is positive and grows: a model with
$\tau < 0.021$ scores better than one at the measured value. The split normal is a
normalised distribution of the data everywhere, and can simulate mocks.

## A tabulated distribution

`LikelihoodGreig` uses the distribution of the neutral fraction at $z = 7.1$ inferred
from the damping wing of the quasar ULAS J1120+0641 (Greig et al. 2017, arXiv:1606.00441). v1
interpolated its logarithm with a spline.

A {class}`~yabf.likelihoods.TabulatedPDF` tabulates the distribution of the residual,
data minus model. Here the measurement is the distribution of the model value itself,
so the data are zero, and the table is that of $-x_{\rm HI}$. The published table has
one value of $-10^{-18}$, rounding error, which v1 left out; here it is set to zero.

```{code-cell} python
nf = np.load(DATA / "NeutralFractionsForPDF.npy")
pdf = np.load(DATA / "NeutralFractionPDF_SmallHII.npy")
positive = pdf > 0
v1_spline = InterpolatedUnivariateSpline(nf[positive], np.log(pdf[positive]))
greig = TabulatedPDF(x=-nf[::-1], pdf=np.clip(pdf, 0, None)[::-1])

x_hi = np.linspace(0.02, 0.98, 241)
v1 = v1_spline(x_hi)
v2 = loglike(greig, x_hi, [0.0])

fig, ax = plt.subplots(figsize=(5, 3.5))
compare(ax, x_hi, v1, v2)
ax.set_xlabel(r"model $x_{\rm HI}(z=7.1)$")
print("largest difference, after the constant:", np.ptp(v2 - v1))
```

They differ by less than $10^{-3}$, from interpolating the density linearly rather
than its logarithm with a spline.

## Errors that depend on the model

`Likelihood1DPowerCoeval` and `Likelihood1DPowerLightcone` add a fractional error on
the model power spectrum, 15 % by default, to the noise:
$\sigma_i^2 = \sigma_{{\rm noise},i}^2 + (0.15\,\Delta^2_i)^2$. They compute only the
chi-squared, though. When the variance depends on the model, a Gaussian also has the
term $-\frac12 \sum_i \ln(2\pi\sigma_i^2)$, which changes with the model; v1 left it
out.

In v2, the model's own uncertainty travels with the model, as its `covariance`, and
the {class}`~yabf.likelihoods.Gaussian` adds it to the noise, and keeps the
determinant. Here is a power law in $k$, with 10 % noise:

```{code-cell} python
k = np.geomspace(0.1, 1.0, 8)  # in Mpc^-1


@attrs.frozen
class PowerSpectrum:
    """A model power spectrum, in mK^2, with a 15 % uncertainty of its own."""

    amplitude: float

    @property
    def values(self) -> np.ndarray:
        """Return the power spectrum at k."""
        return self.amplitude * 100 * (k / 0.3) ** 0.5

    @property
    def covariance(self) -> np.ndarray:
        """Return the covariance of the model's uncertainty."""
        return np.diag((0.15 * self.values) ** 2)


noise = 0.1 * PowerSpectrum(1.0).values
power = Gaussian(np.diag(noise**2))


def v1_power(amplitude: float, data: np.ndarray) -> float:
    """Return the v1 log-likelihood: the chi-squared alone."""
    model = PowerSpectrum(amplitude).values
    return -0.5 * np.sum((model - data) ** 2 / ((0.15 * model) ** 2 + noise**2))
```

The difference between the two is exactly the missing term:

```{code-cell} python
data = power.simulate(PowerSpectrum(1.0), rng)
amplitudes = np.linspace(0.6, 1.6, 101)
v1 = np.array([v1_power(a, data) for a in amplitudes])
v2 = np.array([power.loglike(PowerSpectrum(a), data) for a in amplitudes])
variance = [noise**2 + (0.15 * PowerSpectrum(a).values) ** 2 for a in amplitudes]
missing = np.array([-0.5 * np.sum(np.log(2 * np.pi * v)) for v in variance])
print("largest difference from the missing term:", np.max(np.abs(v2 - v1 - missing)))
```

The missing term penalises large models, whose errors are large. Without it, the
fit is drawn towards them. Over many mocks drawn at an amplitude of one, the
maximum-likelihood amplitude of v1 is biased high; that of v2 is not, within the
statistical error:

```{code-cell} python
best = {"v1": [], "v2": []}
for _ in range(300):
    data = power.simulate(PowerSpectrum(1.0), rng)
    v1 = [v1_power(a, data) for a in amplitudes]
    v2 = [power.loglike(PowerSpectrum(a), data) for a in amplitudes]
    best["v1"].append(amplitudes[np.argmax(v1)])
    best["v2"].append(amplitudes[np.argmax(v2)])

for name, values in best.items():
    error = np.std(values) / np.sqrt(len(values))
    print(f"{name}: mean best amplitude {np.mean(values):.3f} ± {error:.3f}")
```

## Not yet covered

Some v1 likelihoods need observables and data sets that v2 does not have yet; this
page will cover them as they arrive:

- the HERA upper limits on the power spectrum (`Likelihood1DPowerLightconeUpper`,
  and the emulator branch of the power-spectrum likelihoods), which need the
  window functions and {class}`~yabf.likelihoods.UpperLimit`;
- the Lyman-α forest (`LikelihoodForest`) and `LikelihoodPspec`;
- the Planck CMB power spectra (`LikelihoodPlanckPowerSpectra`).
