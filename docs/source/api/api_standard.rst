Standard API --- Overview
=========================

.. automodule:: bayesbay.parameterization

BayesBay standard API provides high-level tools for defining Bayesian inference problems and collecting samples from posterior distributions using Markov chain Monte Carlo (MCMC). The typical workflow includes the following steps:

#. Define the prior probability for the parameters to be inferred from the data (see :doc:`api_standard_prior`).
#. Link the free parameters to one or more instances of what we call a :class:`ParameterSpace <bayesbay.parameterization.ParameterSpace>` or to a discretized spatial domain (see :doc:`api_standard_discretization`).
#. Create a :class:`Parameterization <bayesbay.parameterization.Parameterization>` that groups all instances of ``ParameterSpace`` or ``Discretization``.
#. Define one or more instances of :class:`Target <bayesbay.likelihood.Target>` to store information about the observed data and their associated noise. Use such instances, along with forward functions that enable data predictions from the considered model parameters, to define a :class:`LogLikelihood <bayesbay.likelihood.LogLikelihood>` instance (see also :doc:`api_standard_target`). Alternatively, initialize ``LogLikelihood`` using your own log-likelihood function.
#. Initialize ``BayesianInversion`` using the previously defined ``Parameterization`` and ``LogLikelihood``, and sample the posterior through one or multiple Markov chains executed in parallel (detailed in :doc:`api_standard_inference`).
#. Optionally, use a ``Sampler`` (see :doc:`api_standard_samplers`) to customize specific sampling aspects, such as Markov chains behavior at initialization or after each iteration.

