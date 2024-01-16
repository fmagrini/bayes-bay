.. BayesBay documentation master file, created by
   sphinx-quickstart on Wed Nov  8 18:58:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BayesBay's documentation!
========================================

Welcome to the BayesBay documentation. BayesBay is a Python package that provides 
tools for trans-dimensional MCMC sampling, with a particular emphasis on geophysical 
inversion and other fields.

Key features:

- Fixed and trans-dimensional Bayesian MCMC sampling.
- 1D Voronoi cell parameterization utilities for inversion.
- Hierarchical sampling that includes data noise as additional unknowns.
- Parallel tempering to improve chain mixing in complex problems.

Explore our documentation to understand how BayesBay can assist in your inversion 
experiments.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/0_quickstart
   tutorials/1_hierarchical

.. toctree::
   :maxdepth: 1
   :caption: Standard API

   api/api_standard
   api/api_standard_inference
   api/api_standard_parameters
   api/api_standard_parameterization
   api/api_standard_discretization
   api/api_standard_states
   api/api_standard_target
   api/api_standard_samplers
   api/api_standard_perturbations

.. toctree::
   :maxdepth: 1
   :caption: Low-level API

   api/api_low_level
   api/api_low_level_inference

.. toctree::
   :maxdepth: 1
   :caption: Development

   developer
   changelog
   licence
