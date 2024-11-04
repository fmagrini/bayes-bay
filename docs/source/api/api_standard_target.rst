Data and Likelihood
===================

In BayesBay, the observed data should be embedded in a ``Target``. When paired with a forward function that enables data predictions from the considered model parameters, a ``Target`` can be utilized to define an instance of ``LogLikelihood``. Importantly, ``Target`` facilitates treating the noise properties of the associated data as unknown, enabling hierarchical inversions.

An instance of ``LogLikelihood`` can encapsulate multiple ``Target`` instances, each associated with a distinct forward function. This enables joint inversions of various types of data sets.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.likelihood.Target
    bayesbay.likelihood.LogLikelihood

All examples in this documentation make use of one or more ``Target`` instances (and associated forward functions) to initialize ``LogLikelihood``. For usage tips, refer to our :doc:`Quickstart Tutorial <../tutorials/00_quickstart>`. Examples in this documentation employing hierarchical sampling, where the data noise is treated as unknown, include:

* :doc:`Polynomial Fitting: Part II <../tutorials/02_hierarchical_polyfit>`
* :doc:`Gaussian Mixture Modelling: Part I <../tutorials/11_gaussian_mixture>` and :doc:`Part II <../tutorials/12_transd_gaussian_mixture>`
* :doc:`Inversion of Surface-Wave Dispersion Curves: Part I <../tutorials/21_rayleigh>` and :doc:`Part II <../tutorials/22_rayleigh_love>`
* :doc:`Surface-Wave Tomography <../tutorials/31_sw_tomography>`
* :doc:`Partition Modelling: Part I <../tutorials/41_simple_partition_mod>` and :doc:`Part II <../tutorials/42_transd_partition_mod>`
