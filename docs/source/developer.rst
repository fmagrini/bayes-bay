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

4. Pre commit is also recommended, so that formatting and notebook executation in the
   documentation will be done automatically after you make a commit. 
   Set it up by creating a new file ``.git/hooks/pre-commit`` as following:

   .. code-block:: console

     #!/bin/sh

     black src/bayesbay

     if git diff --cached --name-status | grep -q 'docs/source/tutorials/*.ipynb'; then
          cd docs
          make html
     fi
