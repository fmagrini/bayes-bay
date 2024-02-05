Inference
=========

Our low-level API enables users to define a Bayesian inference problem via ``BaseBayesianInversion``. Internally, this process creates one or more instances of ``BaseMarkovChain``, which encapsulate the state of the inference problem and sample the posterior for an arbitrary number of iterations.


.. mermaid::

    graph TD;
        BaseBayesianInversion-->BayesianInversion;
        BaseMarkovChain-->MarkovChain;

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.BaseBayesianInversion
    bayesbay.BaseMarkovChain
