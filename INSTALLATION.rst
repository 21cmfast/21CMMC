Installation
============

.. note:: These installation instructions concern a *user* install. For intructions on
          how to install for development purposes, see :doc:`contributing <contributing>`.

As may be expected, `21CMMC` depends on `21cmFAST`, and this has some non-python
dependencies. Thus, you must ensure that these dependencies (usually system-wide ones)
are installed *before* attempting to install `21CMMC`. See
https://21cmfast.readthedocs.org/en/latest/installation for details on these dependencies.

If you are confident that the non-python dependencies are installed, you can simply
install `21CMMC` in the usual fashion::

    $ pip install 21CMMC

Note that if `21cmFAST` is not installed, it will be installed automatically. There
are several environment variables which control the compilation of `21cmFAST`, and these
can be set during this call. See the above installation docs for details.