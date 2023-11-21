from numbers import Number
from typing import Tuple

from .._state import State
from .._parameterizations import Parameterization
from ._base_perturbation import Perturbation


class NoiseStdPerturbation(Perturbation):           # TODO
    def __init__(self, parameterization: Parameterization):
        super().__init__(parameterization)
    
    def perturb(self, model: State) -> Tuple[State, Number]:
        return super().perturb(model)


class NoiseCorrPerturbation(Perturbation):          # TODO
    def __init__(self, parameterization: Parameterization):
        super().__init__(parameterization)
    
    def perturb(self, model: State) -> Tuple[State, Number]:
        return super().perturb(model)    
