
Changelog
=========

v1.0.0 [2026-10-24]
-------------------
The last release of the v1 series, for 21cmFAST < 4. From here on, v1 gets bug fixes
only, on the ``v1.x`` branch; new development goes into v2 on ``main``.

- Fix: ``LikelihoodPlanck`` default upper tau error is 0.0081 (Qin+20), as in its
  docstring (it was 0.0073).
- Fix: ``LikelihoodPspec`` reads the data k in 1/Mpc, like all other likelihoods and
  data files (it treated them as h/Mpc and evaluated the model at k*h). k are in 1/Mpc
  everywhere in 21CMMC.
- ``LikelihoodLuminosityFunction`` with ``Core21cmEMU`` interpolates the emulated UV LF
  linearly in z to the redshift of the data (``interpolate_z``), instead of using the
  emulated z closest to int(z). No extrapolation outside the emulated redshifts.
- ``LikelihoodLuminosityFunction(telescope="JWST", jwst_lf=...)`` chooses the built-in
  JWST LF: Donnan+24 (default) or GLIMPSE Chemerynska+26 (new data files).
- New ``py21cmmc.systematics`` priors on positive systematic power (flat, half-normal,
  exponential), marginalised in each bin by ``LikelihoodPspec(systematics=...)``.

v1.0.0dev
---------
- More fleshed-out interface to cosmoHammer, with base classes abstracting some common
  patterns.
- New likelihoods and cores that are able to work on any data from the ``21cmFAST`` pipeline.
- Better logging
- Better exception handling
- pip-installable
- Documentation
- Pipenv support
- Full code formatting applied
- New likelihood to use Bowman+18 EDGES measurement (2003.04442, 2009.11493)
- New likelihood to use Bosman+18 Lyman-alpha forest data (2101.09033)
- New likelihood to use Planck18 EE PS (2006.16828)
- Update default LikelihoodPlanck to use Planck18 tau_e derived in 2006.16828
- Update CoreLuminosityFunction to work with models USE_MINI_HALOS
