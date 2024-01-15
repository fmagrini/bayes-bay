Inference
=========

This page contains the high level ``BayesianInversion``, the instances of which keep a 
list of ``MarkovChain`` as field.

It's worth noting that the classes below are subclasses of the ones used in our 
low-level API, as illustrated below:

.. mermaid::

    graph TD;
        BaseBayesianInversion-->BayesianInversion;
        BaseMarkovChain-->MarkovChain;

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.BayesianInversion
    bayesbay.MarkovChain
