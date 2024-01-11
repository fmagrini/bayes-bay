from numbers import Number
from typing import Tuple, List
import random

from .._state import State, DataNoiseState
from .._target import Target
from ._base_perturbation import Perturbation


class NoisePerturbation(Perturbation):
    """Perturbation by changing the noise level estimation

    Parameters
    ----------
    targets : List[Target]
    """
    def __init__(
        self,
        targets: List[Target]
    ):
        self.targets = targets

    def perturb(self, state: State) -> Tuple[State, Number]:
        """propose a new state that changes the data noise estimation from the given
        state and calcualtes its associated acceptance probability excluding log 
        likelihood ratio

        Parameters
        ----------
        state : State
            the given current state

        Returns
        -------
        Tuple[State, Number]
            proposed new state and the partial acceptance probability (equal to
            zero, in this case) excluding log likelihood ratio for this perturbation
        """
        new_data_noise_all = dict()
        for target in self.targets:
            old_data_noise_state = state[target.name]
            new_data_noise_state = self._perturb_target(target, old_data_noise_state)
            new_data_noise_all[target.name] = new_data_noise_state
        new_state = state.copy()
        new_state.param_values.update(new_data_noise_all)
        return new_state, 0
    
    def _perturb_target(
        self, target: Target, data_noise_state: DataNoiseState
    ) -> DataNoiseState:
        to_be_perturbed = ["std"]
        if target.noise_is_correlated:
            to_be_perturbed.append("correlation")
        new_values = {"correlation": None}
        for p in to_be_perturbed:
            vmin = getattr(target, f"{p}_min")
            vmax = getattr(target, f"{p}_max")
            perturb_std = getattr(target, f"{p}_perturb_std")
            old_value = getattr(data_noise_state, p)
            while True:
                random_deviate = random.normalvariate(0, perturb_std)
                new_value = old_value + random_deviate
                if new_value < vmin or new_value > vmax:
                    continue
                break
            new_values[p] = new_value
        return DataNoiseState(**new_values)

    @property
    def __name__(self) -> str:
        """Identifier for the perturbation"""
        target_names = [t.name for t in self.targets]
        target_names_str = (
            str(target_names) if len(target_names) > 1 else str(target_names[0])
        )
        return f"{self.type}({target_names_str})"
