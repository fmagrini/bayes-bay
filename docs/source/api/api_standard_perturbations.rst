Perturbations
=============

Perturbation functions are generated (under the hood) as instances of the below 
classes, in the initialization stage of 
:class:`bayesbridge.parameterization.Parameterization` and 
:class:`bayesbridge.LogLikelihood`.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbridge.perturbations.Perturbation
    bayesbridge.perturbations.ParamPerturbation
    bayesbridge.perturbations.BirthPerturbation
    bayesbridge.perturbations.DeathPerturbation
    bayesbridge.perturbations.NoisePerturbation
