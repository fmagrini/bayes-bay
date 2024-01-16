Parameterization
================

In BayesBay, a user can define several instances of ``ParameterSpace`` that
host different groups of parameters. 

All the parameter space(s) will be managed by a single ``Parameterization`` object.

For parameters that are associated with (unknown) spatial positions, see 
:doc:`api_standard_discretization`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.parameterization.Parameterization
    bayesbay.parameterization.ParameterSpace
