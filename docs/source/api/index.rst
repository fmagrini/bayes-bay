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
    bayesbridge.State
    bayesbridge.Target
    bayesbridge.LogLikelihood


Parameterizations
^^^^^^^^^^^^^^^^^

.. toggle::

    .. autosummary::
        :toctree: generated/
        :nosignatures:

        bayesbridge.Parameterization
        bayesbridge.Voronoi1D


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


Perturbations
^^^^^^^^^^^^^

.. toggle::

    .. autosummary::
        :toctree: generated/
        :nosignatures:

        bayesbridge.perturbations.Perturbation
        bayesbridge.perturbations.ParamPerturbation
        bayesbridge.perturbations.Voronoi1DPerturbation
        bayesbridge.perturbations.BirthPerturbation1D
        bayesbridge.perturbations.BirthFromPrior1D
        bayesbridge.perturbations.BirthFromNeighbour1D
        bayesbridge.perturbations.DeathPerturbation1D
        bayesbridge.perturbations.DeathFromPrior1D
        bayesbridge.perturbations.DeathFromNeighbour1D


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
