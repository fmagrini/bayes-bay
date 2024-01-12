Samplers
========

Samplers manage the iterations in a BayesBridge. It allows users to customize the 
initialization, to add callback functions, or to manipulate chains in between 
iterations.

One can customize the sampling by adding their own functions on top of existing 
samplers defined by us below; they could also write their own subclass of
:class:`bayesbridge.samplers.Sampler`, depending on their own preference.

By default, :meth:`bayesbridge.BayesianInversion` runs the sampling using the 
:class:`bayesbridge.samplers.VanillaSampler`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbridge.samplers.Sampler
    bayesbridge.samplers.VanillaSampler
    bayesbridge.samplers.ParallelTempering
    bayesbridge.samplers.SimulatedAnnealing
