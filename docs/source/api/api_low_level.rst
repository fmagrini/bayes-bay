Low level API
=============

Optionally, users can use BayesBay's low-level API for a highly customisable 
experience. 

Instead of configuring parameters and parameterization as in the standard API (see
:doc:`api_standard`), here users write their own perturbation functions and initialize
the states on their own. More specifically, here are steps involved in a typical 
inference run with the low-level API:

#. Define your own log likelihood function
#. Define your own perturbation functions
#. Generate your initial walker states
#. Pass the above to the inference runner (:class:`bayesbay.BaseBayesianInversion`)
#. Run the inversion (:meth:`bayesbay.BayesianInversion.run`)

Again, we recommend :ref:`our tutorial page <quickstart>` to get started if you are
new to BayesBay.

.. toctree::
    :maxdepth: 1

    api_low_level_inference
