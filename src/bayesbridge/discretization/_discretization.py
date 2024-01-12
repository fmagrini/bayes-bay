from abc import abstractmethod
from typing import Union, List, Tuple
from numbers import Number
import numpy as np

from ..parameters._parameters import Parameter
from ..parameterization._parameter_space import ParameterSpace
from .._state import ParameterSpaceState


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
        self._additional_kwargs = kwargs
    
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
        raise NotImplementedError
        
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
        self, param_space_state: ParameterSpaceState, isite: int
    ) -> Tuple[ParameterSpaceState, Number]:
        raise NotImplementedError
    
    @abstractmethod
    def log_prior(self, param_space_state: ParameterSpaceState) -> Number:
        raise NotImplementedError

    def get_perturb_std(self, *args) -> Number:
        return self.perturb_std

    def _repr_dict(self) -> dict:      # to be called by ParameterSpace.__repr__
        attr_to_show = ParameterSpace._repr_dict(self)
        attr_to_show["spatial_dimensions"] = self.spatial_dimensions
        attr_to_show["perturb_std"] = self.perturb_std
        if self.trans_d:
            attr_to_show["birth_from"] = self.birth_from
        attr_to_show.update(self._additional_kwargs)
        return attr_to_show
    
    def __repr__(self) -> str:
        return ParameterSpace.__repr__(self)
