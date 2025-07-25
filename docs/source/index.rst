.. BayesBay documentation master file, created by
   sphinx-quickstart on Wed Nov  8 18:58:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BayesBay's documentation!
========================================

BayesBay is a Python package providing a versatile framework for **trans-dimensional and hierarchical Markov Chain Monte Carlo (MCMC) sampling**. It leverages object-oriented programming principles to facilitate the definition of Bayesian sampling problems across a range of applications. This includes joint inversions of multiple data sets with different forward functions and unknown noise properties, as well as complex parameterizations involving multiple parameters with unknown dimensionality and/or spatially varying priors.

.. admonition:: KEY FEATURES
   :class: note

   * :bold-underline:`Modular Architecture` Each component of the inversion (e.g., parameterization, data noise, forward functions) is treated as a self-contained unit, allowing solution of a wide range of inverse problems.
   * :bold-underline:`Trans-dimensional` The dimensionality of the inverse problem can be treated as unknown.
   * :bold-underline:`Hierarchical` When unknown, data errors can be treated as free hyperparameters.
   * :bold-underline:`Joint Inversion Support` High-level features facilitate integration of multiple data sets, enabling seamless joint inversions.
   * :bold-underline:`Flexible Parameterizations` BayesBay streamlines the setup of complex prior probabilities, allowing users to incorporate detailed knowledge of the inverse problem.
   * :bold-underline:`Discretization Support` Includes high-level features for implementing trans-dimensional Voronoi tessellations.
   * :bold-underline:`Multi-Processing Capabilities` Multiple Markov chains can be distributed across CPUs for parallel execution.
   * :bold-underline:`User-Friendly Sampling` Settings for burn-in period, model save intervals, number of chains, and CPU allocation can be configured in a single line of code.
   * :bold-underline:`Advanced Sampling Techniques` Built-in support for parallel tempering and simulated annealing for sampling complex posterior distributions.
  


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   tutorials/00_quickstart
   cite
  
.. toctree::
   :maxdepth: 1
   :caption: Standard API

   api/api_standard
   api/api_standard_inference
   api/api_standard_prior
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
   :caption: Examples

   tutorials/polyfit
   tutorials/gmm
   tutorials/partition_modeling
   tutorials/sw
   tutorials/sw_rf
   tutorials/tomography

.. toctree::
   :maxdepth: 1
   :caption: Development

   developer
   changelog
   licence
