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

Coordinate convention
~~~~~~~~~~~~~~~~~~~~~

Spherical coordinates are given in degrees and ordered as
``(longitude, latitude)``. Valid latitudes satisfy
``-90 <= latitude <= 90``. Longitude is periodic, so values that differ by 360
degrees represent the same meridian.

By default, :class:`Voronoi2DSphere
<bayesbay.discretization.Voronoi2DSphere>` stores site longitudes in
``-180 <= longitude < 180``. Its ``lon_shift`` constructor argument changes
this interval to ``lon_shift - 180 <= longitude < lon_shift + 180`` when a
different map seam is needed; it does not change the spherical geometry.

The static ``plot_tessellation_3d`` method accepts any finite site longitude
because the three-dimensional geometry has no map seam. Its ``center_lonlat``
argument sets the initial camera direction using the same
``(longitude, latitude)`` order. The camera longitude is normalized to
``-180 <= longitude < 180``; for example, 190 degrees is treated as -170
degrees. ``center_lonlat=(12.5, 42.5)`` centres the initial view on Italy.

Rendering resolution
~~~~~~~~~~~~~~~~~~~~

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
surface patches.

In Cartopy, ``resolution`` instead selects the detail level of a geographic
dataset, such as the Natural Earth coastlines used by the example. It does not
control the Voronoi boundaries or the coloured surface mesh.
