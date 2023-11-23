from numbers import Number
from typing import Tuple
import random

from .._state import State
from .._parameterizations import Parameterization
from ._base_perturbation import Perturbation


class NoisePerturbation(Perturbation):  # TODO
    def __init__(
        self, 
        std_min: Number, 
        std_max: Number, 
        std_perturb_std: Number, 
        correlation_min: Number = None, 
        correlation_max: Number = None, 
        correlation_perturb_std: Number = None, 
    ):
        self._std_min = std_min
        self._std_max = std_max
        self._std_perturb_std = std_perturb_std
        self._correlation_min = correlation_min
        self._correlation_max = correlation_max
        self._correlation_perturb_std = correlation_perturb_std

    def perturb(self, model: State) -> Tuple[State, Number]:
        to_be_perturbed = random.choice(["std", "correlation"])
        vmin = getattr(self, f"_{to_be_perturbed}_min")
        vmax = getattr(self, f"_{to_be_perturbed}_min")
        std = getattr(self, f"_{to_be_perturbed}_perturb_std")
        old_value = getattr(model, "noise_{to_be_perturbed}")
        while True:
            random_deviate = random.normalvariate(0, std)
            new_value = old_value + random_deviate
            if new_value < vmin or new_value > vmax:
                continue
        new_model = model.clone()
        setattr(self, f"noise_{to_be_perturbed}", new_value)
        return new_model, 0
        
    def prior_ratio(self, old_model: State, new_model: State) -> Number:
        return super().prior_ratio(old_model, new_model)
