from numbers import Number
from typing import Tuple
import random

from .._state import State
from ._base_perturbation import Perturbation


class NoisePerturbation(Perturbation):
    def __init__(
        self,
        target_name: str, 
        std_min: Number,
        std_max: Number,
        std_perturb_std: Number,
        correlation_min: Number = None,
        correlation_max: Number = None,
        correlation_perturb_std: Number = None,
    ):
        self.target_name = target_name
        self._std_min = std_min
        self._std_max = std_max
        self._std_perturb_std = std_perturb_std
        self._correlation_min = correlation_min
        self._correlation_max = correlation_max
        self._correlation_perturb_std = correlation_perturb_std

    def perturb(self, model: State) -> Tuple[State, Number]:
        if self._correlation_min is not None:
            to_be_perturbed = random.choice(["std", "correlation"])
        else:
            to_be_perturbed = "std"
        vmin = getattr(self, f"_{to_be_perturbed}_min")
        vmax = getattr(self, f"_{to_be_perturbed}_max")
        std = getattr(self, f"_{to_be_perturbed}_perturb_std")
        hyper_param_key = (self.target_name, f"noise_{to_be_perturbed}")
        old_value = model.get_param_values(hyper_param_key)
        while True:
            random_deviate = random.normalvariate(0, std)
            new_value = old_value + random_deviate
            if new_value < vmin or new_value > vmax:
                continue
            break
        new_model = model.clone()
        new_model.set_param_values(hyper_param_key, new_value)
        return new_model, 0

    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        return 0
