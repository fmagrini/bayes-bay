List of functions and classes (API)
===================================

.. automodule:: bayesbridge

Welcome to the API references for BayesBridge, your go-to resource for detailed 
information on the package's functions and classes. If you're seeking specifics on how 
to use a particular item, you're in the right place. 

If you're new to BayesBridge, we recommend :ref:`our tutorial page <basic-usage>` for easy-to-follow usage 
guides.


Low level inversion API
-----------------------


.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbridge.BaseBayesianInversion
    bayesbridge.BaseMarkovChain

High level inversion API
------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbridge.BayesianInversion
    bayesbridge.MarkovChain
    bayesbridge.Target
    bayesbridge.LogLikelihood

Data structure for states
-------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbridge.State
    bayesbridge.ParameterSpaceState
    bayesbridge.DataNoiseState

Parameterizations
^^^^^^^^^^^^^^^^^

.. toggle::

    .. autosummary::
        :toctree: generated/
        :nosignatures:

        bayesbridge.parameterization.Parameterization
        bayesbridge.parameterization.ParameterSpace

Parameters
^^^^^^^^^^

.. toggle::

    .. autosummary::
        :toctree: generated/
        :nosignatures:

        bayesbridge.parameters.Parameter
        bayesbridge.parameters.UniformParameter
        bayesbridge.parameters.GaussianParameter
        bayesbridge.parameters.CustomParameter

Discretizations
^^^^^^^^^^^^^^^

.. toggle::

    .. autosummary::
        :toctree: generated/
        :nosignatures:

        bayesbridge.discretization.Discretization
        bayesbridge.discretization.Voronoi
        bayesbridge.discretization.Voronoi1D

Perturbations
^^^^^^^^^^^^^

.. toggle::

    .. autosummary::
        :toctree: generated/
        :nosignatures:

        bayesbridge.perturbations.Perturbation
        bayesbridge.perturbations.ParamPerturbation
        bayesbridge.perturbations.BirthPerturbation
        bayesbridge.perturbations.DeathPerturbation
        bayesbridge.perturbations.NoisePerturbation

Samplers
^^^^^^^^

.. toggle::

    .. autosummary::
        :toctree: generated/
        :nosignatures:

        bayesbridge.samplers.Sampler
        bayesbridge.samplers.VanillaSampler
        bayesbridge.samplers.ParallelTempering
        bayesbridge.samplers.SimulatedAnnealing
