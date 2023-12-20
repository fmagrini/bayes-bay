from typing import List, Callable, Tuple, Dict
from numbers import Number
import random

from .._state import State
from ..parameters import Parameter
from ..perturbations import ParamPerturbation, BirthFromPrior1D, DeathFromPrior1D


class ParameterSpace:
    """Utility class to parameterize the Bayesian inference problem"""
    def __init__(
        self, 
        name: str,
        n_dimensions: int = None, 
        n_dimensions_min: int = 1, 
        n_dimensions_max: int = 10, 
        n_dimensions_init_range: Number = 0.3, 
        parameters: List[Parameter] = None, 
    ):
        self.name = name
        self._trans_d = n_dimensions is None
        self._n_dimensions = n_dimensions
        self._n_dimensions_min = n_dimensions_min
        self._n_dimensions_max = n_dimensions_max
        self._n_dimensions_init_range = n_dimensions_init_range
        self._parameters = dict()
        if parameters is not None:
            for param in parameters:
                self._parameters[param.name] = param
        self._init_perturbation_funcs()
        self._init_log_prior_ratio_funcs()
    
    @property
    def trans_d(self) -> bool:
        """indicates whether the current configuration allows dimensionality change
        """
        return self._trans_d
    
    @property
    def parameters(self) -> Dict[str, Parameter]:
        """all the unknown parameters under this dimensionality setting"""
        return self._parameters
    
    @property
    def perturbation_functions(self) -> List[Callable[[State], Tuple[State, Number]]]:
        """a list of perturbation functions allowed in the current dimensionality
        configurations, each of which takes in a state :class:`State` and returns a new
        state and a log proposal ratio value
        """
        return self._perturbation_funcs
    
    @property
    def log_prior_ratio_functions(self) -> List[Callable[[State, State], Number]]:
        """a list of log prior ratio functions corresponding to each of the
        :meth:`perturbation_functions`
        """
        return self._log_prior_ratio_funcs
    
    def initialize(self, state: State):
        """initializes the parameterization (if it's trans dimensional) and the
        parameter values

        Returns
        -------
        State
            an initial model state
        """
        # initialize number of dimensions
        if not self.trans_d:
            n_dimensions = self._n_dimensions
        else:
            init_range = self._n_dimensions_init_range
            n_dims_min = self._n_dimensions_min
            n_dims_max = self._n_dimensions_max
            init_max = int((n_dims_max - n_dims_min) * init_range + n_dims_min)
            n_dimensions = random.randint(n_dims_min, init_max)
        # initialize parameter values
        parameter_vals = dict()
        for name, param in self.parameters.items():
            parameter_vals[name] = param.initialize()
        raise NotImplementedError("Need to decide what State looks like")
    
    def _init_perturbation_funcs(self):
        self._perturbation_funcs = []
        for name, param in self.parameters.items():
            self._perturbation_funcs.append(ParamPerturbation(name, param))
        if self.trans_d:
            self._perturbation_funcs.append(
                BirthFromPrior1D(
                    parameters=self.parameters, 
                    n_dimensions_max=self._n_dimensions_max, 
                )
            )
            self._perturbation_funcs.append(
                DeathFromPrior1D(
                    parameters=self.parameters, 
                    n_dimensions_min=self._n_dimensions_min, 
                )
            )
    
    def _init_log_prior_ratio_funcs(self):
        self._log_prior_ratio_funcs = [
            func.log_prior_ratio for func in self.perturbation_functions
        ]
    