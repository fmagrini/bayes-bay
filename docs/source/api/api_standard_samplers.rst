Samplers
========

This module facilitates the definition of arbitrary Bayesian sampling criteria. In BayesBay, the specific actions undertaken by a Markov chain at each iteration are handled by a ``Sampler`` instance. These actions encompass, for example, state initializations, management of Markov chain temperature, adjustment of the standard deviation of the Gaussians used to perturb the free parameters under inference, etc.

A ``Sampler`` instance allows the user to insert callback functions at either the beginning or the end of a Markov chain iteration, and even to customize complex sampling behaviors such as those involving interactions between different Markov chains that are run in parallel. (This capability is leveraged in our implementation of parallel tempering.) Furthermore, through the insertion of callback functions, users can augment our predefined classes (listed below) with their own, integrating custom functionalities seamlessly.

.. mermaid::

   graph TD;
       Sampler-->VanillaSampler;
       Sampler-->ParallelTempering;
       Sampler-->SimulatedAnnealing;

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.samplers.Sampler
    bayesbay.samplers.VanillaSampler
    bayesbay.samplers.ParallelTempering
    bayesbay.samplers.SimulatedAnnealing

To use a specific ``Sampler`` for sampling a posterior distribution, initialise it and pass it to the appropriate argument in :meth:`BayesBay.BayesianInversion.run`. Most examples in this documentation make use of ``VanillaSampler``, which is the default. Examples using ``SimulatedAnnealing`` include:

* :doc:`Gaussian Mixture Modelling: Part II <../tutorials/12_transd_gaussian_mixture>`