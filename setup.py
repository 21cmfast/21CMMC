#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ).read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# ======================================================================================================================

setup(
    name="py21cmmc",
    version=find_version("src", "py21cmmc", "__init__.py"),
    license="MIT license",
    description="An extensible MCMC framework for 21cmFAST",
    long_description="%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.rst")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    author="Brad Greig",
    author_email="greigb@unimelb.edu.au",
    url="https://github.com/21cmFAST/21CMMC",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"py21cmmc": "data/*"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords=["Epoch of Reionization", "Cosmology"],
    install_requires=[
        "click",
        # 'tqdm',
        "numpy",
        "cosmoHammer",
        "scipy",
        "matplotlib>=2.1",
        "emcee<3",
        "powerbox>=0.5.7",
        "cached_property",
        "py21cmfast @ https://github.com/21cmFAST/21cmFAST",  # TODO: publish to pypi
    ],
    # entry_points={
    #     'console_scripts': [
    #         '21CMMC = py21cmmc.cli:main',
    #     ]
    # },
)
