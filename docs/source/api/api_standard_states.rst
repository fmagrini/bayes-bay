States
======

A state in BayesBridge is the data structure that holds values associated with a phase
in the inference (such as the initial status or at a certain iteration). It can 
include the following, depending on what are being inverted in the current problem:

- Values of discretizations
- Values of parameters
- Values of noise standard deviation (and correlation)
- (optional) Cached computing results, if the user chooses to attach them to a certain 
  model via :meth:`bayesbridge.State.store_cache`

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbridge.State
    bayesbridge.ParameterSpaceState
    bayesbridge.DataNoiseState
