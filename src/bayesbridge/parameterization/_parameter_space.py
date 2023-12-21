from typing import List, Callable, Tuple, Dict
from numbers import Number
import random
import math
import numpy as np

from .._state import State, ParameterSpaceState
from ..exceptions import DimensionalityException
from ..parameters import Parameter
from ..perturbations._param_values import ParamPerturbation
from ..perturbations._birth_death import BirthPerturbation, DeathPerturbation
from .._utils_1d import delete


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
        self._name = name
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
    
    @property
    def name(self) -> str:
        """name of the current parameter space"""
        return self._name
    
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
    
    def initialize(self) -> ParameterSpaceState:
        """initializes the parameter space including its parameter values

        Returns
        -------
        ParameterSpaceState
            an initial parameter space state
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
            parameter_vals[name] = param.initialize(np.empty(n_dimensions))
        return ParameterSpaceState(n_dimensions, parameter_vals)
    
    def birth(self, ps_state: ParameterSpaceState) -> Tuple[ParameterSpaceState, float]:
        n_dims = ps_state.n_dimensions
        if n_dims == self._n_dimensions_max:
            raise DimensionalityException("Birth")
        new_param_values = dict()
        for param_name, param_vals in ps_state.param_values.items():
            new_param_values[param_name] = np.append(
                param_vals, 
                self.parameters[param_name].initialize()
            )
        new_state = ParameterSpaceState(n_dims+1, new_param_values)
        prob_ratio = - math.log(n_dims + 1)
        return new_state, prob_ratio
    
    def death(self, ps_state: ParameterSpaceState) -> Tuple[ParameterSpaceState, float]:
        n_dims = ps_state.n_dimensions
        if n_dims == self._n_dimensions_min:
            raise DimensionalityException("Death")
        i_to_remove = random.randint(0, n_dims-1)
        new_param_values = dict()
        for param_name, param_vals in ps_state.param_values.items():
            new_param_values[param_name] = delete(param_vals, i_to_remove)
        new_state = ParameterSpaceState(n_dims-1, new_param_values)
        prob_ratio = math.log(n_dims)
        return new_state, prob_ratio
    
    def _init_perturbation_funcs(self):
        self._perturbation_funcs = [
            ParamPerturbation(self.name, self.parameters)
        ]
        if self.trans_d:
            self._perturbation_funcs.append(BirthPerturbation(self))
            self._perturbation_funcs.append(DeathPerturbation(self))
