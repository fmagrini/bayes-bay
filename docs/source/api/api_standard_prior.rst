Prior
=====

This module facilitates the definition of prior probabilities for model 
parameters within Bayesian inversion problems. It offers out-of-the-box 
support for free parameters described by uniform and normal probability 
distributions, along with high-level functionalities for custom prior 
definitions. When used in conjunction with a discretized spatial domain 
(see :doc:`api_standard_discretization`), these prior distributions can 
also be tailored to vary as a function of position within the domain.


.. mermaid::

   graph TD;
       Prior-->UniformPrior;
       Prior-->GaussianPrior;
       Prior-->CustomPrior;

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.prior.Prior
    bayesbay.prior.UniformPrior
    bayesbay.prior.GaussianPrior
    bayesbay.prior.CustomPrior
