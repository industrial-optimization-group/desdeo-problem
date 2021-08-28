# <img alt="DESDEO" src="https://github.com/industrial-optimization-group/DESDEO/blob/migrate-to-new/assets/desdeo_logo.png" height="80">

[![PyPI version](https://badge.fury.io/py/desdeo.svg)](https://badge.fury.io/py/desdeo)
[![Documentation Status](https://readthedocs.org/projects/desdeo/badge/?version=latest)](https://desdeo.readthedocs.io/en/latest/?badge=latest)

# DESDEO

## About

DESDEO is an open source framework for interactive multiobjective
optimization methods. DESDEO contains implementations of some interactive
methods and modules that can be utilized to implement further methods.

The mission of DESDEO is to increase awarenss of the benefits of interactive
methods make interactive methods more easily available and applicable. Thanks
to the open architecture, interactive methods are easier to be utilized and
further developed. The framework consists of reusable components that can be
utilized for implementing new methods or modifying the existing methods. The
framework is released under a permissive open source license.

This repository contains the main DESDEO module aggregating together all of
the modules in the DESDEO framework.

## Research and website

### Research

The [Multiobjective Optimization Group](http://www.mit.jyu.fi/optgroup/)
residing at the University of Jyväskylä is the main force behind the DESDEO
framework. The research group develops theory, methodology and open-source
computer implementations for solving real-world decision-making problems.
Most of the research concentrates on multiobjective optimization (MO) in
which multiple conflicting objectives are optimized simultaneously and a
decision maker (DM) is supported in finding a preferred compromise.

### Website

To learn more about DESDEO and the Multiobjective Optimization Research
group, visit the official [homepage](https://desdeo.it.jyu.fi).

## Installation

### Requirements

The packages belonging to the DESDEO framework require Python 3.7 or newer.

### Using DESDEO as a software library

The DESDEO package can be found on [PyPI](https://pypi.org/project/desdeo/), and can be installed by invoking pip:

`pip install desdeo`

### For development (on \*nix systems)

Requires [poetry](https://python-poetry.org/). See `pyproject.toml` for Python package requirements.

1. `git clone https://github.com/industrial-optimization-group/DESDEO`
2. `mkdir desdeo`
3. `cd desdeo`
4. `poetry init`
5. `poetry install`

**NOTE**: This repository does not contain any code implementing the different features in DESDEO. Instead, this
repository contains the main documentation of the framework, and is used to build and define the DESDEO framework
software package combining all of the Python subpackages in the framework.

## Documentation

The DESDEO framework's documentation can be found [here](https://desdeo.readthedocs.io/en/latest/)

## Contributing

Contributions to the DESDEO framework and its different modules is warmly welcome! See the documentation's [contributing](https://desdeo.readthedocs.io/en/latest/contributing.html) for further details.

## Legacy code
The old version of DESDEO, which is no longer maintained, can be found [here](https://github.com/industrial-optimization-group/DESDEO/tree/legacy)
alongside with its [documentation](https://desdeo.readthedocs.io/en/legacy/). The support for this version of DESDEO ceased
June 2020, and is no longer supported.
