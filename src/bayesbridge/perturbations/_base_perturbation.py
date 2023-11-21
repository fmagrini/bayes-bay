from abc import abstractmethod
from typing import Tuple
from numbers import Number

from .._parameterizations import Parameterization
from .._state import State


class Perturbation:
    def __init__(self, parameterization: Parameterization):
        self.parameterization = parameterization

    @abstractmethod
    def perturb(self, model: State) -> Tuple[State, Number]:
        raise NotImplementedError
    
    def __run__(self, model: State) -> Tuple[State, Number]:
        return self.perturb(model)

    @property
    def type(self) -> str:
        return self.__class__.__name__
