
Changelog
=========

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
