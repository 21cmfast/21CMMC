Installation
============

``21CMMC`` is a pure-Python MCMC framework built around ``21cmFAST``, and currently
depends on it directly. ``21cmFAST`` has some non-python (compiled) dependencies,
so you must ensure these are installed *before* attempting to install ``21CMMC``.
See https://21cmfast.readthedocs.io/en/latest/installation.html for details. Note
that the current codebase does not yet support ``21cmFAST`` v4, so it is pinned to
``21cmFAST<4.0.0`` (support for v4 -- and making ``21cmFAST`` an optional dependency
-- is planned for a future PR).

To use the MultiNest sampler, you will also need to install the compiled ``multinest``
library and its Python interface ``pymultinest``. These are most easily installed via
``conda``, since they include non-python compiled components.

For Users
---------

.. note:: ``conda`` users may want to pre-install the following packages before running
          the below installation commands::

            conda install numpy scipy click pyyaml cffi astropy h5py 21cmfast

If you are confident that the non-python dependencies (``21cmFAST`` and, if desired,
``multinest``) are installed, you can simply install ``21CMMC`` in the usual fashion::

    pip install 21CMMC

or, using `uv <https://docs.astral.sh/uv/>`_::

    uv pip install 21CMMC

If you would also like to install the samplers (``pymultinest``, ``ultranest``,
``zeus-mcmc``), use the relevant extra::

    pip install "21CMMC[samplers]"

For Developers
--------------
If you are developing ``21CMMC``, we recommend using `uv <https://docs.astral.sh/uv/>`_
to manage your environment. After cloning the repository::

    uv sync --extra dev

This creates a ``.venv`` with all testing, documentation and sampler dependencies
installed, along with the pre-commit hooks configured via ``ruff``.

Because ``multinest`` and ``21cmFAST`` have non-python (compiled) dependencies, we
recommend using ``conda``/``mamba`` to install those pieces (and any of their system
dependencies) before installing ``21CMMC`` itself, e.g.::

    conda env create -f ci/test-env.yml
    conda activate test-suite
    pip install -e ".[dev]"

This mirrors the environment used in continuous integration (see
`ci/test-env.yml <ci/test-env.yml>`_).
