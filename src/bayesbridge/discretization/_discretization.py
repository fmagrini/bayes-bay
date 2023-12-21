from abc import abstractmethod
from typing import Union, List, Tuple
from numbers import Number
import numpy as np

from ..parameters._parameters import Parameter
from ..parameterization._parameter_space import ParameterSpace
from .._state import ParameterSpaceState
from ..perturbations._param_values import ParamPerturbation


class Discretization(Parameter, ParameterSpace):
    
    def __init__(
        self,
        name: str,
        spatial_dimensions: Number,
        perturb_std: Union[Number, np.ndarray],
        n_dimensions: int = None, 
        n_dimensions_min: int = 1, 
        n_dimensions_max: int = 10, 
        n_dimensions_init_range: Number = 0.3,
        parameters: List[Parameter] = None, 
        birth_from: str = "prior",
        **kwargs
    ):
        Parameter.__init__(
            self, 
            name=name,
            perturb_std=perturb_std,
            spatial_dimensions=spatial_dimensions,        
            **kwargs
        )
        ParameterSpace.__init__(
            self, 
            name=name,
            n_dimensions=n_dimensions,
            n_dimensions_min=n_dimensions_min,
            n_dimensions_max=n_dimensions_max,
            n_dimensions_init_range=n_dimensions_init_range,
            parameters=parameters
        )
        self.spatial_dimensions = spatial_dimensions
        self.perturb_std = perturb_std
        self.birth_from = birth_from        
    
    @abstractmethod
    def initialize(self, *args) -> ParameterSpaceState:
        """initializes the values of this discretization including its paramter values

        Returns
        -------
        ParameterSpaceState
            an initial parameter space state
        """
        raise NotImplementedError

    def _init_perturbation_funcs(self):
        ParameterSpace._init_perturbation_funcs(self)
        self._perturbation_funcs.append(ParamPerturbation(self.name, [self]))
        
    @abstractmethod
    def birth(
        self, param_space_state: ParameterSpaceState
    ) -> Tuple[ParameterSpaceState, Number]:
        raise NotImplementedError
    
    @abstractmethod
    def death(
        self, param_space_state: ParameterSpaceState
    ) -> Tuple[ParameterSpaceState, Number]:
        raise NotImplementedError

    @abstractmethod
    def perturb_value(
        self, param_space_state: ParameterSpaceState
    ) -> Tuple[ParameterSpaceState, Number]:
        raise NotImplementedError
    
    @abstractmethod
    def log_prior(self, value: Number, *args) -> Number:
        raise NotImplementedError

    def get_perturb_std(self, *args) -> Number:
        return self.perturb_std
