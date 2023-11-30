.. Bayes Bridge documentation master file, created by
   sphinx-quickstart on Wed Nov  8 18:58:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Bayes Bridge's documentation!
========================================

Welcome to the BayesBridge documentation. BayesBridge is a Python package that provides 
tools for trans-dimensional MCMC sampling, with a particular emphasis on geophysical 
inversion and other fields.

Key features:

- Fixed and trans-dimensional Bayesian MCMC sampling.
- 1D Voronoi cell parameterization utilities for inversion.
- Hierarchical sampling that includes data noise as additional unknowns.
- Parallel tempering to improve chain mixing in complex problems.

Explore our documentation to understand how BayesBridge can assist in your inversion 
experiments.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/vanilla

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   developer
