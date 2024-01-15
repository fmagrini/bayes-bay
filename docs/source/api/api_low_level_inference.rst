Inference
=========

This page contains the low-level ``BaseBayesianInversion``, the instances of which 
keep a list of ``BaseMarkovChain`` as field.

It's worth noting that the classes in the standard API are subclasses of them as 
shown below:

.. mermaid::

    graph TD;
        BaseBayesianInversion-->BayesianInversion;
        BaseMarkovChain-->MarkovChain;

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.BaseBayesianInversion
    bayesbay.BaseMarkovChain
