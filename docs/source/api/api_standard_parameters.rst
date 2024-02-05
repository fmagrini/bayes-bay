Parameters
==========

This module facilitates the definition of prior probabilities for model parameters within Bayesian inversion problems. It offers out-of-the-box support for parameters described by uniform and normal probability distributions, along with high-level functionalities for custom prior definitions. When used in conjunction with a discretized spatial domain (see :doc:`api_standard_discretization`), these prior distributions can also be tailored to vary as a function of position within the domain.


.. mermaid::

   graph TD;
       Parameter-->UniformParameter;
       Parameter-->GaussianParameter;
       Parameter-->CustomParameter;

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.parameters.Parameter
    bayesbay.parameters.UniformParameter
    bayesbay.parameters.GaussianParameter
    bayesbay.parameters.CustomParameter
