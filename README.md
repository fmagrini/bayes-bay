# BayesBay

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
