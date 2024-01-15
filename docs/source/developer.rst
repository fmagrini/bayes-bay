Developer Notes
===============

This package is still under active development. We are still adding features to it, but 
contributions are welcomed especially if it is bug fixes or aligns with our development
plans:

- Simulated annealing and simulated tempering techniques
- 2D and 3D Voronoi nuclei parameterization utilities
- Expanding our scientific examples set by running BayesBay against your own problem

Local setup
-----------

1. To set up the development environment

   .. code-block:: console

        $ mamba env create -f envs/environment_dev.yml

   Or

   .. code-block:: console

        $ python -m venv bayesbay_dev
        $ source bayesbay_dev/bin/activate
        $ pip install -r envs/requirements_dev.txt

2. To install the package in editable mode

   .. code-block:: console

        $ python -m pip install -e .

3. Look at ``noxfile.py`` for details on building, testing, formatting and linting.
