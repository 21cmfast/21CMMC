Installation
============

As may be expected, ``21CMMC`` depends on ``21cmFAST``, and this has some non-python
dependencies. Thus, you must ensure that these dependencies (usually system-wide ones)
are installed *before* attempting to install ``21CMMC``. See
https://21cmfast.readthedocs.org/en/latest/installation for details on these dependencies.

Then follow the instructions below, depending on whether you are a user or developer.

For Users
---------

.. note:: ``conda`` users may want to pre-install the following packages before running
          the below installation commands::

            conda install numpy scipy click pyyaml cffi astropy h5py


If you are confident that the non-python dependencies are installed, you can simply
install ``21CMMC`` in the usual fashion::

    pip install 21CMMC

Note that if ``21cmFAST`` is not installed, it will be installed automatically. There
are several environment variables which control the compilation of ``21cmFAST``, and these
can be set during this call. See the above installation docs for details.

For Developers
--------------
If you are developing `21CMMC`, we highly recommend using `conda` to manage your
environment, and setting up an isolated environment. If this is the case, setting up
a full environment (with all testing and documentation dependencies) should be as easy
as (from top-level dir)::

    conda env create -f environment_dev.yml

Otherwise, if you are using `pip`::

    pip install -r requirements_dev.txt
    pip install -e .

And if you would like to also compile documentation::

    pip install -r docs/requirements.txt
