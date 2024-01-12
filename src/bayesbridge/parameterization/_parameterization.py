from typing import List, Union

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
        if not isinstance(parameter_space, list):
            parameter_space = [parameter_space]
        self.parameter_spaces = parameter_space
        self._check_duplicate_param_space_names()
        self._check_duplicate_param_names()
        self._init_perturbation_funcs()
    
    def _check_duplicate_param_space_names(self):
        all_param_space_names = [ps.name for ps in self.parameter_spaces]
        if len(all_param_space_names) != len(set(all_param_space_names)):
            raise ValueError("duplicate parameter space names found")
    
    def _check_duplicate_param_names(self):
        all_param_names = []
        for ps in self.parameter_spaces:
            param_names = list(ps.parameters.keys())
            all_param_names.extend(param_names)
        if len(all_param_names) != len(set(all_param_names)):
            raise ValueError(
                "duplicate parameter names found, or one parameter has been assigned "
                "to different parameter spaces"
            )
    
    def _init_perturbation_funcs(self):
        self._perturbation_funcs = []
        for ps in self.parameter_spaces:
            self._perturbation_funcs.extend(ps.perturbation_functions)
    
    def initialize(self) -> State:
        """initializes the parameter space(s) constituting the parameterization
        
        Returns
        -------
        State
            Numerical values corresponding to the free parameters in the
            parameter space(s) constituting the parameterization
        """
        param_values = dict()
        for ps in self.parameter_spaces:
            param_values[ps.name] = ps.initialize()
        return State(param_values)
    
    @property
    def perturbation_functions(self) -> List[Perturbation]:
        """the list of perturbation functions allowed to perturb the
        parameterization
        """
        return self._perturbation_funcs
