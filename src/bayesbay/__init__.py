import warnings

from ._state import DataNoiseState, ParameterSpaceState, State
from .likelihood._target import Target as _Target
from .likelihood._log_likelihood import LogLikelihood as _LogLikelihood
from ._markov_chain import MarkovChain, BaseMarkovChain
from ._bayes_inversion import BayesianInversion, BaseBayesianInversion

from . import (
    samplers,
    prior,
    likelihood,
    perturbations,
    parameterization,
    discretization,
)

from ._version import __version__


def Target(*args, **kwargs):
    warnings.warn(
        (
            "The 'Target' class has been moved to the 'likelihood' module. Please use "
            "'from bayesbay.likelihood import Target' instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return _Target(*args, **kwargs)


def LogLikelihood(*args, **kwargs):
    warnings.warn(
        (
            "The 'LogLikelihood' class has been moved to the 'likelihood' module. "
            "Please use 'from bayesbay.likelihood import LogLikelihood' instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return _LogLikelihood(*args, **kwargs)


__all__ = [
    "DataNoiseState",
    "ParameterSpaceState",
    "State",
    "Target",
    "LogLikelihood",
    "MarkovChain",
    "BaseMarkovChain",
    "BayesianInversion",
    "BaseBayesianInversion",
]
