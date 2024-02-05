Standard API --- Overview
=========================

.. automodule:: bayesbay.parameterization

BayesBay's standard API offers comprehensive utilities for defining Bayesian inference problems and collecting samples from posterior distributions via Markov chain Monte Carlo (MCMC). The typical workflow includes the following steps:

#. Define the prior probability for the parameters to be inferred from the data (see :doc:`api_standard_parameters`)
#. Link the free parameters to what we call a ``ParameterSpace`` or to a discretized spatial domain (see :doc:`api_standard_discretization`)
#. Create a ``Parameterization`` that encompasses all instances of ``ParameterSpace`` or ``Discretization``
#. Define one or more instances of ``Target`` to store information about the observed data and their associated noise. Use such instances, along with forward functions that enable data predictions from the considered model parameters, to define a ``LogLikelihood`` instance (see :doc:`api_standard_target`).
#. Configure an instance of ``BayesianInversion`` using the previously defined objects, enabling the sampling of the posterior through one or multiple Markov chains executed in parallel (detailed in :doc:`api_standard_inference`)
#. Optionally, use a ``Sampler`` (see :doc:`api_standard_samplers`) to customize specific sampling aspects, such as Markov chains behavior at initialization or after each iteration.
#. Run the inversion (:meth:`bayesbay.BayesianInversion.run`).

