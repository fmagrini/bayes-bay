# BayesBay

[![PyPI version](https://img.shields.io/pypi/v/bayesbay?logo=pypi&style=flat-square&color=cae9ff&labelColor=f8f9fa)](https://pypi.org/project/bayesbay/)
[![Documentation Status](https://img.shields.io/readthedocs/bayes-bay?logo=readthedocs&style=flat-square&color=fed9b7&labelColor=f8f9fa&logoColor=eaac8b)](https://bayes-bay.readthedocs.io/en/latest/?badge=latest)

BayesBay is a user-friendly Python package designed for **generalised trans-dimensional and hierarchical Bayesian inference**. 
Optimised computationally through Cython, our library offers multi-processing capabilities and runs seamlessly on both standard computers and computer clusters. 

Distinguishing itself from existing packages, BayesBay provides **high-level functionalities for defining complex parameterizations**. 
These include prior probabilities that can be specified by uniform, Gaussian, or custom density functions and may vary depending on the spatial position in a 
hypothetical discretization. 

By default, BayesBay employs **reversible-jump Markov chain Monte Carlo** (MCMC) for sampling the posterior probability. 
It also offers options for **parallel tempering** or **simulated annealing**, while its low-level features enable the effortless implementation of arbitrary sampling criteria. 
Utilising object-oriented programming principles, BayesBay ensures that each component of an inference problem --- such as observed data, forward function(s), and parameterization --- 
is a self-contained unit. This design facilitates the integration of various forward solvers and data sets, promoting the simultaneous use of multiple data types in the 
considered inverse problem.

## Development tips

- To set up development environment:

    ```console
    $ mamba env create -f envs/environment_dev.yml
    ```

    or 

    ```console
    $ python -m venv bayesbay_dev
    $ source bayesbay_dev/bin/activate
    $ pip install -r envs/requirements_dev.txt
    ```

- To install the package:

    ```console
    $ python -m pip install .
    ```

- Look at [noxfile.py](noxfile.py) for building, testing, formatting and linting.
