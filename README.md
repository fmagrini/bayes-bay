# BayesBay

[![PyPI version](https://img.shields.io/pypi/v/bayesbay?logo=pypi&style=flat-square&color=cae9ff&labelColor=f8f9fa)](https://pypi.org/project/bayesbay/)
[![Documentation Status](https://img.shields.io/readthedocs/bayes-bay?logo=readthedocs&style=flat-square&color=fed9b7&labelColor=f8f9fa&logoColor=eaac8b)](https://bayes-bay.readthedocs.io/en/latest/?badge=latest)

Trans-dimensional McMC sampling implemented in Python and Cython.

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
