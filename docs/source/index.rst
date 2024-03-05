.. BayesBay documentation master file, created by
   sphinx-quickstart on Wed Nov  8 18:58:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BayesBay's documentation!
========================================

BayesBay is a user-friendly Python package designed for **generalised trans-dimensional and hierarchical Bayesian inference**. 
Optimised computationally through Cython, our library offers multi-processing capabilities and runs seamlessly on both standard computers and computer clusters. 

Distinguishing itself from existing packages, BayesBay provides **high-level functionalities for defining complex parameterizations**. 
These include prior probabilities that can be specified by uniform, Gaussian, or custom density functions and may vary depending on the spatial position in a 
hypothetical discretization. 

By default, BayesBay employs **reversible-jump Markov chain Monte Carlo** (MCMC) for sampling the posterior probability. 
It also offers options for **parallel tempering** or **simulated annealing**, while its low-level features enable the effortless implementation of arbitrary sampling criteria. 
Utilising object-oriented programming principles, BayesBay ensures that each component of an inference problem --- such as observed data, forward function(s), and parameterization --- 
is a self-contained unit. This design facilitates the integration of various forward solvers and data sets, promoting the simultaneous use of multiple data types in the 
considered inverse problem.

.. admonition:: KEY FEATURES
   :class: note

   * :bold-underline:`Flexible Parameterizations` The free parameters in the inverse problem can be defined by uniform, Gaussian, or custom prior probabilities. In the case of spatially discretized problems, these may or may not be dependent on position.
   * :bold-underline:`Trans-dimensional` The dimensionality of the inverse problem can be treated as unknown, i.e., as a free parameter to be inferred from the observations.
   * :bold-underline:`Hierarchical` When unknown, data errors can be treated as free hyper-parameters. Ideal for data fusion, this approach allows the data itself to drive the characteristics of the noise properties and avoids arbitrary weights for different observables in the evaluation of the likelihood.
   * :bold-underline:`Multi-Processing Capabilities` Multiple Markov chains can be efficiently distributed across different CPUs for parallel execution.
   * :bold-underline:`Flexible Sampling Criteria` Besides providing ready-to-use functionalities such as parallel tempering or simulated annealing to sample complex posterior probabilities, the low-level features of BayesBay enable the definition of arbitrary sampling criteria.
   * :bold-underline:`Discretization Support` Particularly relevant to geoscientific inverse problems, BayesBay currently supports (trans-dimensional) spatial discretization through 1-D Voronoi tessellation. Development of 2-D and 3-D tessellations is actively underway.
   * :bold-underline:`Joint Inversion Support` Designed to facilitate joint inversions of multiple data sets, BayesBay provides high-level functionalities for data fusion.


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation

  
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
   tutorials/tomography

.. toctree::
   :maxdepth: 1
   :caption: Development

   developer
   changelog
   licence
