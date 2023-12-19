from typing import List, Union

from ._dimensionality import Dimensionality
from ..perturbations import Perturbation


class Parameterization:
    def __init__(self, dimensionalities: Union[Dimensionality, List[Dimensionality]]):
        if not isinstance(dimensionalities, list):
            dimensionalities = [dimensionalities]
        self.dimensionalities = dimensionalities
        self._perturbation_funcs = []
        for dim in self.dimensionalities:
            self._perturbation_funcs.extend(dim.perturbation_functions)
    
    @property
    def perturbation_functions(self) -> List[Perturbation]:
        return self._perturbation_funcs
