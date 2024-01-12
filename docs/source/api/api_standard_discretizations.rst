Discretizations
===============

Discretizations are special parameter spaces that are also parameters. They have values
that represent spatial discretizations, and hold a list of parameters that are 
dependent on these values.

Here is a class inheritance diagram of related classes we have in BayesBridge so far:

.. mermaid::

   graph TD;
       ParameterSpace-->Discretization;
       Parameter-->Discretization;
       Discretization-->Voronoi;
       Voronoi-->Voronoi1D;

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbridge.discretization.Discretization
    bayesbridge.discretization.Voronoi
    bayesbridge.discretization.Voronoi1D
