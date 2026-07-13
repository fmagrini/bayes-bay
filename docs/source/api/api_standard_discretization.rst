Discretization
==============

The ``Discretization`` class can be seen as a special type of ``ParameterSpace``, designed for applications that involve spatial domains (see :doc:`api_standard_parameterization`). Each dimension within a ``Discretization`` instance corresponds to a discretized element of the spatial domain, such as a pixel in a 2D space. Consequently, all free parameters associated with a ``Discretization`` instance are intrinsically associated with spatial domain elements. This allows for the definition of prior probabilities that are functions of position within the domain (see :doc:`api_standard_prior`).


.. mermaid::

   graph TD;
       ParameterSpace-->Discretization;
       Prior-->Discretization;
       Discretization-->Voronoi;
       Voronoi-->Voronoi1D;
       Voronoi-->Voronoi2D;
       Voronoi-->Voronoi2DSphere

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.discretization.Discretization
    bayesbay.discretization.Voronoi
    bayesbay.discretization.Voronoi1D
    bayesbay.discretization.Voronoi2D
    bayesbay.discretization.Voronoi2DSphere

Examples in this documentation using :class:`Voronoi1D <bayesbay.discretization.Voronoi1D>` include:

* :doc:`Inversion of Surface-Wave Dispersion Curves: Part I <../tutorials/21_rayleigh>` and :doc:`Part II <../tutorials/22_rayleigh_love>`
* :doc:`Partition Modelling: Part I <../tutorials/41_simple_partition_mod>` and :doc:`Part II <../tutorials/42_transd_partition_mod>`

Examples using :class:`Voronoi2D <bayesbay.discretization.Voronoi2D>` include:

* :doc:`Surface-Wave Tomography <../tutorials/31_sw_tomography>`

Examples using :class:`Voronoi2DSphere <bayesbay.discretization.Voronoi2DSphere>` include:

* :doc:`Surface-Wave Tomography on the Sphere <../tutorials/32_sw_tomography_sphere>`

Plotting two-dimensional tessellations
---------------------------------------

The cell boundaries of :class:`Voronoi2D <bayesbay.discretization.Voronoi2D>`
are straight segments in Cartesian coordinates. They are drawn exactly, so no
plotting-resolution or boundary-spacing argument is needed.

The great-circle boundaries of
:class:`Voronoi2DSphere <bayesbay.discretization.Voronoi2DSphere>` are curved
and must be sampled when rendered. BayesBay handles this internally with a
fixed one-degree maximum angular spacing. This is sufficiently fine for normal
and publication-scale figures and does not change the tessellation itself.

The three-dimensional plotting method uses
``surface_spacing_deg`` for the angular spacing of the coloured sphere mesh.
Smaller values make colour transitions less pixelated, but the number of mesh
patches grows approximately with the inverse square of the spacing. For
example, halving ``surface_spacing_deg`` creates roughly four times as many
surface patches. These BayesBay spacing arguments are unrelated to Cartopy's
``resolution`` argument, which selects the detail level of geographic datasets
such as coastlines.
