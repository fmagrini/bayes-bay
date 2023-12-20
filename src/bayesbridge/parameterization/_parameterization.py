from typing import List, Union

from ._parameter_space import ParameterSpace
from ..perturbations import Perturbation


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
        self.parameter_space = parameter_space
        self._perturbation_funcs = []
        for ps in self.parameter_space:
            self._perturbation_funcs.extend(ps.perturbation_functions)
    
    @property
    def perturbation_functions(self) -> List[Perturbation]:
        return self._perturbation_funcs
