from typing import List, Union
from numbers import Number
from pprint import pformat

from ._parameter_space import ParameterSpace
from ..perturbations import Perturbation
from .._state import State


class Parameterization:
    """Parameterization setting that consists of one or more :class:`ParameterSpace`
    instances
    
    Parameters
    ----------
    parameter_space : Union[ParameterSpace, List[ParameterSpace]]
        one or more :class:`ParameterSpace` instance(s)
    """
    def __init__(self, parameter_space: Union[ParameterSpace, List[ParameterSpace]]):
        parameter_spaces = parameter_space if isinstance(parameter_space, list) else [parameter_space]
        self._check_duplicate_param_space_names(parameter_spaces)
        self._check_duplicate_param_names(parameter_spaces)
        
        self.parameter_spaces = {ps.name: ps for ps in parameter_spaces}
        self._init_perturbation_funcs()
    
    def _check_duplicate_param_space_names(self, parameter_spaces: List[ParameterSpace]):
        all_param_space_names = [ps.name for ps in parameter_spaces]
        if len(all_param_space_names) != len(set(all_param_space_names)):
            raise ValueError("duplicate parameter space names found")
    
    def _check_duplicate_param_names(self, parameter_spaces: List[ParameterSpace]):
        for ps in parameter_spaces:
            param_names = list(ps.parameters.keys())
            if len(param_names) != len(set(param_names)):
                raise ValueError(
                    f"duplicate parameter names found in ParameterSpace {ps.name}"
                )

    def _init_perturbation_funcs(self):
        self._perturbation_funcs = []
        self._perturbation_weights = []
        for ps in self.parameter_spaces.values():
            self._perturbation_funcs.extend(ps.perturbation_functions)
            self._perturbation_weights.extend(ps.perturbation_weights)
    
    def initialize(self) -> State:
        """initializes the parameter space(s) constituting the parameterization
        
        Returns
        -------
        State
            Numerical values corresponding to the free parameters in the
            parameter space(s) constituting the parameterization
        """
        param_values = dict()
        for ps_name, ps in self.parameter_spaces.items():
            param_values[ps_name] = ps.initialize()
        return State(param_values)
    
    @property
    def perturbation_functions(self) -> List[Perturbation]:
        """the list of perturbation functions allowed to perturb the
        parameterization
        """
        return self._perturbation_funcs
    
    @property
    def perturbation_weights(self) -> List[Number]:
        """the list of perturbation function weights that determines the probability of
        which each corresponding perturbation function is to be chosen"""
        return self._perturbation_weights
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(parameter_spaces="
            f"{list(self.parameter_spaces.values())})"
        )
    
    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{pformat(list(self.parameter_spaces.values()))})"
        )
