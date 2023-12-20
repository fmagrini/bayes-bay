from numbers import Number
from typing import Tuple
import random

from .._state import State, DataNoise
from ._base_perturbation import Perturbation


class NoisePerturbation(Perturbation):
    """Perturbation by changing the noise level estimation

    Parameters
    ----------
    target_name : str
        name of the data target associated with this noise perturbation
    std_min : Number
        minimum bound of the standard deviation for data noise
    std_max : Number
        maximum bound of the standard deviation for data noise
    std_perturb_std : Number
        perturbation standard deviation of the standard deviation for data noise
    correlation_min : Number, optional
        minimum bound of the correlation for data noise, by default None
    correlation_max : Number, optional
        maximum bound of the correlation for data noise, by default None
    correlation_perturb_std : Number, optional
        perturbation standard deviation of the correlation for data noise, by 
        default None
    """
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

    def perturb(self, state: State) -> Tuple[State, Number]:
        """propose a new state that changes the data noise estimation from the given
        state and calcualtes its associated acceptance criteria excluding log 
        likelihood ratio

        Parameters
        ----------
        state : State
            the given current state

        Returns
        -------
        Tuple[State, Number]
            proposed new state and the partial acceptance criteria excluding log
            likelihood ratio for this perturbation
        """
        if self._correlation_min is not None:
            to_be_perturbed = random.choice(["std", "correlation"])
        else:
            to_be_perturbed = "std"
        vmin = getattr(self, f"_{to_be_perturbed}_min")
        vmax = getattr(self, f"_{to_be_perturbed}_max")
        std = getattr(self, f"_{to_be_perturbed}_perturb_std")
        old_noise = state.get_param_values(self.target_name)
        old_value_std = getattr(old_noise, "std")
        old_value_corr = getattr(old_noise, "correlation")
        old_value = old_value_std if to_be_perturbed == "std" else old_value_corr
        while True:
            random_deviate = random.normalvariate(0, std)
            new_value = old_value + random_deviate
            if new_value < vmin or new_value > vmax:
                continue
            break
        new_state = state.copy()
        new_value_std = new_value if to_be_perturbed == "std" else old_value_std
        new_value_corr = old_value_corr if to_be_perturbed == "std" else new_value
        new_noise = DataNoise(std=new_value_std, correlation=new_value_corr)
        new_state.set_param_values(self.target_name, new_noise)
        return new_state, 0

    @property
    def __name__(self) -> str:
        return f"{self.type}({self.target_name})"
