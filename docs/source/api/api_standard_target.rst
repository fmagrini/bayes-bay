Data and Likelihood
===================

In BayesBay, observed data should be embedded in a ``Target``. When paired with a forward function that enables data predictions from the considered model parameters, the ``Target`` instance can be utilized to define an instance of ``LogLikelihood``. Importantly, ``Target`` facilitates treating the noise properties of the associated data as unknown, enabling hierarchical inversions.

An instance of ``LogLikelihood`` can encapsulate multiple ``Target`` instances, each associated with a different forward function. This capability facilitates the joint inversion of various types of data sets.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.likelihood.Target
    bayesbay.likelihood.LogLikelihood
