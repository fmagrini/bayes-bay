Standard API
============

.. automodule:: bayesbridge.parameterization

BayesBridge's standard API includes full-fledged utilities to define a 
trans-dimensional or fixed-dimensional McMC sampler and to run it.

Typically, this invovles the following steps:

#. Define your parameters (:doc:`api_standard_parameters`)
#. Define your parameter space(s), which can be a discretization (
   :doc:`api_standard_discretizations`), or a parameter space (without discretization
   to be inverted :class:`ParameterSpace`)
#. Define your parameterization with all the parameter space(s) above
   (:class:`Parameterization`)
#. Define your data target(s) (:doc:`api_standard_target`)
#. Define your inversion with all the above objects (:doc:`api_standard_inference`)
#. (optional) Define and customize your own sampler (:doc:`api_standard_samplers`), if 
   you'd like to customize chain initialization, or to add callback functions on the 
   end of each iteration or each batch of iterations
#. Run the inversion (:meth:`bayesbridge.BayesianInversion.run`)

If you are new to BayesBridge, we recommend :ref:`our tutorial page <basic-usage>` to 
get started.

.. toctree::
    :maxdepth: 1

    api_standard_inference
    api_standard_parameters
    api_standard_parameterization
    api_standard_discretizations
    api_standard_states
    api_standard_target
    api_standard_samplers
    api_standard_perturbations