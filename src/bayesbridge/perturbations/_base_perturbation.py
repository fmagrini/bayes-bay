from abc import abstractmethod
from typing import Tuple
from numbers import Number

from .._state import State


class Perturbation:
    @abstractmethod
    def perturb(self, model: State) -> Tuple[State, Number]:
        raise NotImplementedError

    def __call__(self, model: State) -> Tuple[State, Number]:
        return self.perturb(model)

    @abstractmethod
    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        raise NotImplementedError

    @property
    def type(self) -> str:
        return self.__class__.__name__

    @property
    def __name__(self) -> str:
        return self.type
