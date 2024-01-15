Samplers
========

Samplers manage the iterations in a BayesBay. It allows users to customize the 
initialization, to add callback functions, or to manipulate chains in between 
iterations.

One can customize the sampling by adding their own functions on top of existing 
samplers defined by us below; they could also write their own subclass of
:class:`bayesbay.samplers.Sampler`, depending on their own preference.

By default, :meth:`bayesbay.BayesianInversion` runs the sampling using the 
:class:`bayesbay.samplers.VanillaSampler`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.samplers.Sampler
    bayesbay.samplers.VanillaSampler
    bayesbay.samplers.ParallelTempering
    bayesbay.samplers.SimulatedAnnealing
