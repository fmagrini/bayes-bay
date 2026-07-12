Low-Level API --- Overview
==========================

BayesBay offers a low-level API for a highly customizable experience. Instead of creating one or more instances of ``ParameterSpace`` or ``Discretization`` to define a ``Parameterization`` as in the standard API (see :doc:`api_standard`), our low-level API allows users to write their own perturbation functions and define a Markov chain state via an arbitrary Python object. More specifically, the steps involved in a typical Bayesian inference defined through our low-level API involve the following:

#. Define an arbitrary log-likelihood function
#. Define arbitrary perturbation functions to propose a Markov chain state from the current one
#. Generate your initial Markov chain states
#. Define an instance of :class:`bayesbay.BaseBayesianInversion` using the above objects
#. Run the inversion (:meth:`bayesbay.BaseBayesianInversion.run`)

When a well-formed proposal is deliberately outside the support of the model,
raise :class:`bayesbay.exceptions.InvalidProposalException`. BayesBay records one
rejected self-transition, preserving the proposal probability and detailed
balance. Unexpected forward errors reject a proposal by default; pass
``on_forward_error="raise"`` to ``BaseBayesianInversion`` when debugging to
propagate them instead.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.exceptions.InvalidProposalException
