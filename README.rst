======
21cmMC
======

.. start-badges
.. image:: https://github.com/21cmfast/21CMMC/actions/workflows/test_suite.yaml/badge.svg
    :target: https://github.com/21cmfast/21CMMC/actions/workflows/test_suite.yaml
.. image:: https://codecov.io/gh/21cmfast/21CMMC/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/21cmfast/21CMMC
.. image:: https://readthedocs.org/projects/21cmmc/badge/?version=latest
    :target: https://21cmmc.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. end-badges

Bayesian inference for 21cmFAST and the 21-cm signal.

**This branch is 21cmMC v2, under development.** It is a rewrite: a typed pipeline of
simulator, observable and likelihood stages on top of
`yabf <https://github.com/21cmfast/yabf>`_ v3, with CosmoSIS driving the samplers,
21cmFAST v4 support and full HPC support. Nothing here is usable yet. Progress is
tracked in the `21cmMC v2 project <https://github.com/orgs/21cmfast/projects/15>`_.

Looking for 21cmMC v1?
======================

v1 (for 21cmFAST < 4) is frozen and gets bug fixes only:

* install it with ``pip install "21cmMC<2"``;
* its code is on the `v1.x branch <https://github.com/21cmfast/21CMMC/tree/v1.x>`_;
* its documentation is at https://21cmmc.readthedocs.io/en/stable/.

* Free software: MIT license
