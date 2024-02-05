Standard API --- Overview
=========================

.. automodule:: bayesbay.parameterization

BayesBay's standard API offers comprehensive utilities for defining Bayesian inference problems and collecting samples from posterior distributions via Markov chain Monte Carlo (MCMC). The typical workflow includes the following steps:

#. Define the prior probability for the parameters to be inferred from the data (see :doc:`api_standard_parameters`)
#. Link the free parameters to what we call a :class:`ParameterSpace` or to a discretized spatial domain (see :doc:`api_standard_discretization`)
#. Create a :class:`Parameterization` that encompasses all instances of  of :class:`ParameterSpace` or :class:`Discretization`
#. Define a :class:`Target` to store information about the observed data and its associated noise, used for comparison with model predictions (see :doc:`api_standard_target`)
#. Configure an instance of :class:`BayesianInversion` using the previously defined objects, enabling the sampling of the posterior through one or multiple Markov chains executed in parallel (detailed in :doc:`api_standard_inference`)
#. (optional) Use a :class:`Sampler` (see :doc:`api_standard_samplers`) to customize specific sampling aspects, such as Markov chains behavior at initialization or after each iteration.
#. Run the inversion (:meth:`bayesbay.BayesianInversion.run`).

