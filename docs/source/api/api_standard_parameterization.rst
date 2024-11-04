Parameterization
================

In BayesBay, all free parameters of an inference problem (each characterized by a prior probability, see :doc:`api_standard_prior`) should be encapsulated within one or more instances of ``ParameterSpace`` (or, alternatively, of :doc:`api_standard_discretization`). ``ParameterSpace`` serves as a specialized container that not only groups an arbitrary number of free parameters but also (i) determines their dimensionality, and (ii) specifies the perturbation functions used to propose new model parameters from the current ones at each Markov chain iteration. 

One or more instances of ``ParameterSpace`` allow for defining a ``Parameterization``. Compared to ``ParameterSpace``, the ``Parameterization`` object is simpler and primarily designed to aggregate all model parameters from every specified instance of ``ParameterSpace`` and ``Discretization``.


.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.parameterization.Parameterization
    bayesbay.parameterization.ParameterSpace

All examples in this documentation involve the use of ``Parameterization``. Examples using ``ParameterSpace`` include:

* :doc:`Quickstart Tutorial <../tutorials/00_quickstart>`
* :doc:`Polynomial Fitting: Part I <../tutorials/01_polyfit>` and :doc:`Part II <../tutorials/02_hierarchical_polyfit>`
* :doc:`Gaussian Mixture Modelling: Part I <../tutorials/11_gaussian_mixture>` and :doc:`Part II <../tutorials/12_transd_gaussian_mixture>`

Examples of trans-dimensional parameterizations can be found in:

* :doc:`Gaussian Mixture Modelling: Part II <../tutorials/12_transd_gaussian_mixture>`
* :doc:`Inversion of Surface-Wave Dispersion Curves: Part I <../tutorials/21_rayleigh>` and :doc:`Part II <../tutorials/22_rayleigh_love>`
* :doc:`Surface-Wave Tomography <../tutorials/31_sw_tomography>`
* :doc:`Partition Modelling: Part II <../tutorials/42_transd_partition_mod>`
