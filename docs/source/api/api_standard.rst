OVERVIEW
========

.. automodule:: bayesbay.parameterization

BayesBay's standard API includes full-fledged utilities to define a Bayesian inference problem and to collect samples from a posterior distribution using Markov chain Monte Carlo (MCMC). Typically, this involves the following steps:

#. Define the prior probability for the unknown parameters to be inferred from the data (see :doc:`api_standard_parameters`)
#. Link the free parameters to what we call a :class:`ParameterSpace` or to a discretized spatial domain (see :doc:`api_standard_discretization`)
#. Define a :class:`Parameterization`, embedding all the defined instances of :class:`ParameterSpace` or `:class:`Discretization`
#. Define a :class:`Target`, storing information about the observed data and noise (see :doc:`api_standard_target`)
#. Use the above objects to define an instance of :class:`BayesianInversion`, that allows to sample the posterior in parallel through one or multiple Markov chains (see :doc:`api_standard_inference`)
#. (optional) Define a :class:`Sampler` (see :doc:`api_standard_samplers`). This allows for customizing specific aspects of the Bayesian sampling, such as the behaviour of a Markov chain at the initialization or at the end of each iteration.
#. Run the inversion (:meth:`bayesbay.BayesianInversion.run`)

