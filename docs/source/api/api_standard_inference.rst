Inference
=========


BayesBay's standard API utilizes reversible-jump Markov chain Monte Carlo (MCMC) for sampling the posterior probability. This is achieved through the ``BayesianInversion`` and ``MarkovChain`` classes --- which are subclasses of :class:`bayesbay.BaseBayesianInversion` and :class:`bayesbay.BaseMarkovChain`, available in the low-level API. A ``BayesianInversion`` instance serves as a bridge between the parameterization of the inference problem and its operational facets, such as the parallel execution of a specified number of Markov chains to gather posterior samples. Upon defining a ``BayesianInversion`` instance, one or multiple ``MarkovChain`` instances are automatically generated; these encapsulate all parameters and hyper-parameters related to a given state of the inference problem, and update such information at each Markov chain step. All numerical values associated with a state is stored in an instance of :class:`bayesbay.State`, and can be accessed via a subclass of :class:`bayesbay.samplers.Sampler` (see :doc:`api_standard_samplers`). This enables the modification of the sampling criteria at any point of the inference process.

.. mermaid::

    graph TD;
        BaseBayesianInversion-->BayesianInversion;
        BaseMarkovChain-->MarkovChain;

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.BayesianInversion
    bayesbay.MarkovChain
    
