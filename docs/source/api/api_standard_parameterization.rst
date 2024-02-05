Parameterization
================

In BayesBay, all free parameters of an inference problem (see :doc:`api_standard_parameters`) should be encapsulated within one or more instances of ``ParameterSpace`` (or, alternatively, of :doc:`api_standard_discretization`), which are used to define a ``Parameterization``. ``ParameterSpace`` serves as a specialized container that not only groups an arbitrary number of free parameters but also (i) determines their dimensionality, and (ii) specifies the perturbation functions used to propose new model parameters from the current ones at each Markov chain step.

Compared to ``ParameterSpace``, the ``Parameterization`` object is simpler and primarily designed to aggregate all model parameters from every specified instance of ``ParameterSpace`` and ``Discretization``.


.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.parameterization.Parameterization
    bayesbay.parameterization.ParameterSpace
