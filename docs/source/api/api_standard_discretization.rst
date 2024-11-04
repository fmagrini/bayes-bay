Discretization
==============

The ``Discretization`` class can be seen as a special type of ``ParameterSpace``, designed for applications that involve spatial domains (see :doc:`api_standard_parameterization`). Each dimension within a ``Discretization`` instance corresponds to a discretized element of the spatial domain, such as a pixel in a 2D space. Consequently, all free parameters associated with a ``Discretization`` instance are intrinsically associated with spatial domain elements. This allows for the definition of prior probabilities that are functions of position within the domain (see :doc:`api_standard_prior`).


.. mermaid::

   graph TD;
       ParameterSpace-->Discretization;
       Prior-->Discretization;
       Discretization-->Voronoi;
       Voronoi-->Voronoi1D;
       Voronoi-->Voronoi2D

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.discretization.Discretization
    bayesbay.discretization.Voronoi
    bayesbay.discretization.Voronoi1D
    bayesbay.discretization.Voronoi2D

Examples in this documentation using :class:`Voronoi1D <bayesbay.discretization.Voronoi1D>` include:

* :doc:`Inversion of Surface-Wave Dispersion Curves: Part I <../tutorials/21_rayleigh>` and :doc:`Part II <../tutorials/22_rayleigh_love>`
* :doc:`Partition Modelling: Part I <../tutorials/41_simple_partition_mod>` and :doc:`Part II <../tutorials/42_transd_partition_mod>`

Examples using :class:`Voronoi2D <bayesbay.discretization.Voronoi2D>` include:

* :doc:`Surface-Wave Tomography <../tutorials/31_sw_tomography>`
