============
Installation
============

Install BayesBay
-------------------

**Step 1**: (*Optional*) Set up a virtual environment.

We strongly recommend installing BayesBay within a 
`virtual environment <https://docs.python.org/3/tutorial/venv.html>`_. 
This ensures that BayesBay can install the various modules that it needs without the 
risk of breaking anything else on your system. There are a number of tools that can 
facilitate this, including `venv`, `virtualenv`, `conda` and `mamba`.

.. dropdown:: Expand for how to manage virtual environments
  :icon: package

  .. tab-set::

    .. tab-item:: venv

      Ensure you have `python>=3.7`. Then, you can create a new virtual environment by 
      running the command:

      .. code-block:: console

        $ python -m venv <path-to-new-env>/bb_env

      where :code:`<path-to-new-env>` is your prefered location for storing information 
      about this environment, and :code:`<env-name>` is your preferred name for the 
      virtual environmment. For example,

      .. code-block:: console

        $ python -m venv ~/my_envs/bb_env 

      will create a virtual environment named :code:`bb_env` and store everything 
      within a sub-directory of your home-space named :code:`my_envs`.

      To 'activate' or 'switch on' the virtual environment, run the command
    
      .. code-block:: console

        $ source <path-to-new-env>/<env-name>/bin/activate

      At this point you effectively have a 'clean' Python installation. You can now 
      install and use BayesBay, following the instructions at step 2. When you are 
      finished, you can run the command
      
      .. code-block:: console

        $ deactivate

      and your system will return to its default state. If you want to use BayesBay again, 
      simply re-run the 'activate' step above; you do not need to repeat the 
      installation process. Alternatively, you can remove BayesBay and the virtual 
      environment from your system by running

      .. code-block:: console

        $ rm -rf <path-to-new-env>/<env-name>

    .. tab-item:: virtualenv

      You can create a new virtual environment (using Python version 3.10) by running 
      the command

      .. code-block:: console

        $ virtualenv <path-to-new-env>/<env-name> -p=3.10
      
      where :code:`<path-to-new-env>` is your prefered location for storing information 
      about this environment, and :code:`<env-name>` is your preferred name for the 
      virtual environmment. For example,

      .. code-block:: console

        $ virtualenv ~/my_envs/bb_env -p=3.10

      will create a virtual environment named :code:`bb_env` and store everything 
      within a sub-directory of your home-space named :code:`my_envs`.

      To 'activate' or 'switch on' the virtual environment, run the command

      .. code-block:: console

        $ source <path-to-new-env>/<env-name>/bin/activate

      At this point you effectively have a 'clean' Python installation. You can now 
      install and use BayesBay, following the instructions at step 2. When you are 
      finished, you can run the command

      .. code-block:: console

        $ deactivate

      and your system will return to its default state. If you want to use BayesBay again, 
      simply re-run the 'activate' step above; you do not need to repeat the 
      installation process. Alternatively, you can remove BayesBay and the virtual 
      environment from your system by running

      .. code-block:: console

        $ rm -rf <path-to-new-env>/<env-name>

    .. tab-item::  conda / mamba

      You can create a new virtual environment (using Python version 3.10) by running 
      the command

      .. code-block:: console

        $ conda create -n <env-name> python=3.10

      where :code:`<env-name>` is your preferred name for the virtual environmment. 
      For example,

      .. code-block:: console

        $ conda create -n bb_env python=3.10

      will create a virtual environment named :code:`bb_env`.
      
      To 'activate' or 'switch on' the virtual environment, run the command

      .. code-block:: console

        $ conda activate <env-name>

      At this point you effectively have a 'clean' Python installation. You can now 
      install and use BayesBay, following the instructions at step 2. When you are 
      finished, you can run the command
      
      .. code-block:: console

        $ conda deactivate

      and your system will return to its default state. If you want to use BayesBay again, 
      simply re-run the 'activate' step above; you do not need to repeat the 
      installation process. Alternatively, you can remove BayesBay and the virtual 
      environment from your system by running
      
      .. code-block:: console

        $ conda env remove -n <env-name>



**Step 2**: Install BayesBay

.. tab-set::

  .. tab-item:: pip

    BayesBay is available on `PyPI <https://pypi.org/project/bayesbay/>`_, so for most users
    installation is as simple as:

    .. code-block:: console

      $ pip install bayesbay

  .. tab-item:: From source

    You can build BayesBay from source. You are most likely to want to do this if you 
    want to work in 'developer mode', and make changes to BayesBay's source code.

    .. code-block:: console

      $ git clone https://github.com/fmagrini/bayes-bridge
      $ cd bayes-bridge
      $ pip install -e .
    
    The :code:`-e` flag ensures that the module is installed in editable mode; you can 
    omit this if you do not intend to make any changes.
