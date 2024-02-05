Discretization
==============

The ``Discretization`` class can be seen as a special type of ``ParameterSpace``, designed for applications that involve spatial domains (see :doc:`api_standard_parameterization`). Each dimension within a ``Discretization`` instance corresponds to a discretized element of the spatial domain, such as a pixel in a 2D space. Consequently, all free parameters associated with a ``Discretization`` instance are intrinsically associated with spatial domain elements. This allows for the definition of prior probabilities that are functions of position within the domain (see :doc:`api_standard_parameters`), enabling sophisticated parameterizations.


.. mermaid::

   graph TD;
       ParameterSpace-->Discretization;
       Parameter-->Discretization;
       Discretization-->Voronoi;
       Voronoi-->Voronoi1D;

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.discretization.Discretization
    bayesbay.discretization.Voronoi
    bayesbay.discretization.Voronoi1D
