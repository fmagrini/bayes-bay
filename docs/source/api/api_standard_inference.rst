Inference
=========


BayesBay utilizes reversible-jump Markov chain Monte Carlo (MCMC) for sampling the posterior probability. This is achieved through the ``BayesianInversion`` and ``MarkovChain`` classes (which are subclasses of :class:`bayesbay.BaseBayesianInversion` and :class:`bayesbay.BaseMarkovChain`). A ``BayesianInversion`` instance serves as a bridge between the parameterization of the inference problem and its operational facets, such as the parallel execution of a specified number of Markov chains to gather posterior samples. Upon defining a ``BayesianInversion`` instance, one or multiple ``MarkovChain`` instances are automatically generated. Via a :class:`bayesbay.State`, these encapsulate all information related to a given state of the inference problem, which can be accessed via a subclass of :class:`bayesbay.samplers.Sampler` (see :doc:`api_standard_samplers`).

.. mermaid::

    graph TD;
        BaseBayesianInversion-->BayesianInversion;
        BaseMarkovChain-->MarkovChain;

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.BayesianInversion
    bayesbay.MarkovChain
    
