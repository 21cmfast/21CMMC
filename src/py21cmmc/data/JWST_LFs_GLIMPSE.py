"""GLIMPSE faint-end UV LFs (Chemerynska et al. 2026) as LikelihoodLuminosityFunction data files.

Source: Chemerynska et al. 2026, MNRAS 546, staf2267, "The first GLIMPSE of the faint galaxy population at
Cosmic Dawn with JWST: The evolution of the ultraviolet luminosity function across z~9-15"
(arXiv:2509.24881), Table 1. JWST/NIRCam imaging of the lensing cluster Abell S1063 (GLIMPSE), 105
photometric candidates at 9 < z < 15. M_UV are intrinsic (de-lensed), 1-mag bins. The errors include
Poisson (Gehrels 1986), cosmic variance (Trapp & Furlanetto 2020), magnification and photo-z uncertainties.
Cosmology of the paper: flat LCDM, H0 = 70, Omega_M = 0.3 (as Donnan+24 and the HST LFs; not converted).

The paper gives log10(phi / mag^-1 Mpc^-3) with asymmetric errors. As for the Donnan+24 files,
the likelihood gets phi in linear units and sigma = mean of the upper and lower linear errors.
Redshift of each bin: the label the paper uses (z = 9, 10, 11, 13).

Only the paper's OWN binned measurements are used (Table 1, GLIMPSE sample only), never its fitted LFs:
its double power-law fits (Sec. 6.1) also include the bright-end points of Donnan et al. (2024) and Adams
et al. (2024), and Donnan+24 is a separate data set of our likelihood. The M_UV = -12.5 bins (z = 9, 10)
are not written: "To ensure the robustness of our faint-end fit, we exclude the lowest luminosity bin
(-12.5 mag) due to its limited completeness (~10%)" (Sec. 6.1). They stay in TABLE1 for reference.

Run from this folder: python JWST_LFs_GLIMPSE.py   (writes Chemerynska26_LF_{lfuncs,sigmas}_z{9,10,11,13}.npz)
"""

import numpy as np

# Table 1 of Chemerynska+26: M_UV, N_obj, log10 phi, +err, -err (dex)
TABLE1 = {
    9: [  # 8.5 < z < 9.5
        (-18.5, 2, -3.48, 0.34, 0.60),
        (-17.5, 12, -2.68, 0.21, 0.35),
        (-16.5, 28, -2.26, 0.18, 0.19),
        (-15.5, 6, -2.34, 0.24, 0.31),
        (-14.5, 2, -1.99, 0.32, 0.59),
        (-13.5, 1, -1.41, 0.39, 1.00),
        (-12.5, 2, -0.03, 0.31, 0.58),
    ],
    10: [  # 9.5 < z < 10.5
        (-18.5, 1, -3.74, 0.40, 1.01),
        (-17.5, 5, -3.01, 0.26, 0.35),
        (-16.5, 5, -2.93, 0.25, 0.34),
        (-15.5, 4, -2.29, 0.27, 0.38),
        (-13.5, 1, -1.22, 0.38, 1.00),
        (-12.5, 1, -0.12, 0.38, 1.00),
    ],
    11: [  # 10.5 < z < 11.5
        (-17.5, 4, -3.06, 0.29, 0.40),
        (-16.5, 6, -2.58, 0.25, 0.32),
        (-15.5, 2, -1.93, 0.34, 0.29),
    ],
    13: [  # 11.5 < z < 15
        (-18.5, 1, -4.16, 0.44, 1.02),
        (-17.5, 4, -3.53, 0.31, 0.41),
        (-16.5, 5, -3.00, 0.29, 0.37),
        (-15.5, 1, -3.01, 0.40, 1.01),
    ],
}


# Bins left out of the data files, as in the paper's own LF fits (completeness ~10%)
EXCLUDED_MUV = [-12.5]


def to_linear(rows):
    """M_UV, phi and the upper and lower errors of phi, in linear units."""
    muv, _n, logphi, up, lo = (
        np.array(c, dtype=float) for c in zip(*rows, strict=False)
    )
    phi = 10**logphi
    sigma_upper = 10 ** (logphi + up) - phi
    sigma_lower = phi - 10 ** (logphi - lo)
    return muv, phi, sigma_upper, sigma_lower


if __name__ == "__main__":
    for z, rows in TABLE1.items():
        rows = [r for r in rows if not any(abs(r[0] - m) < 1e-6 for m in EXCLUDED_MUV)]
        muv, phi, su, sl = to_linear(rows)
        np.savez(f"Chemerynska26_LF_lfuncs_z{z}", Muv=muv, lfunc=phi)
        np.savez(f"Chemerynska26_LF_sigmas_z{z}", sigma=(su + sl) / 2.0)
        print(f"z = {z}:")  # noqa: T201
        for m, p, a, b in zip(muv, phi, su, sl, strict=False):
            print(  # noqa: T201
                f"   M_UV = {m:6.1f}   phi = {p:.3e}   +{a:.3e} -{b:.3e}   sigma = {(a + b) / 2:.3e}"
            )
