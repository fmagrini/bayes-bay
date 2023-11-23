from typing import Callable, Tuple
from numbers import Number
import numpy as np

from ._utils_bayes import inverse_covariance
from .perturbations._data_noise import NoisePerturbation
from ._state import State


class Target:
    def __init__(
        self,
        name,
        dobs,
        covariance_mat_inv=None,
        noise_is_correlated=False,
        std_min=0.01,
        std_max=1,
        std_perturb_std=0.1,
        correlation_min=0.01,
        correlation_max=1,
        correlation_perturb_std=0.1,
    ):
        self.name = name
        self.dobs = np.array(dobs)
        self.noise_is_correlated = noise_is_correlated
        self.std = None
        self.correlation = None
        if covariance_mat_inv is None:
            self._perturbation_func = NoisePerturbation(
                std_min=std_min, 
                std_max=std_max, 
                std_perturb_std=std_perturb_std, 
                correlation_min=correlation_min if noise_is_correlated else None, 
                correlation_max=correlation_max if noise_is_correlated else None, 
                correlation_perturb_std=correlation_perturb_std if noise_is_correlated else None, 
            )
        else:
            self._perturbation_func = None
            if np.isscalar(covariance_mat_inv):
                self.covariance_mat_inv = covariance_mat_inv
            else:
                self.covariance_mat_inv = np.array(covariance_mat_inv)

    @property
    def perturbation_function(self) -> Callable[[State], Tuple[State, Number]]:
        return self._perturbation_func

    @property
    def is_hierarchical(self):
        return self.perturbation_function is not None

    # def perturb_covariance(self):
    #     to_be_perturbed = random.choice(self._perturbations)
    #     name = to_be_perturbed["name"]
    #     std = to_be_perturbed["perturb_std"]
    #     vmin, vmax = to_be_perturbed["min"], to_be_perturbed["max"]
    #     value = self._current_state[name]
    #     while True:
    #         random_deviate = random.normalvariate(0, std)
    #         new_value = value + random_deviate
    #         if not (new_value < vmin or new_value > vmax)if not (new_value < vmin or new_value > vmax)::
    #             self._proposed_state[name] = new_value
    #             break
    #     return 0

    # def finalize_perturbation(self, accepted):
    #     accepted_state = self._proposed_state if accepted else self._current_state
    #     rejected_state = self._current_state if accepted else self._proposed_state
    #     for k, v in accepted_state.items():
    #         rejected_state[k] = v

    def covariance_times_vector(self, vector):
        if hasattr(self, "covariance_mat_inv"):
            if np.isscalar(self.covariance_mat_inv):
                return self.covariance_mat_inv * vector
            else:
                return self.covariance_mat_inv @ vector
        elif self.correlation is None:
            return 1 / self._proposed_state["std"] ** 2 * vector
        else:
            std = self._proposed_state["std"]
            r = self._proposed_state["correlation"]
            n = self.dobs.size
            mat = inverse_covariance(std, r, n)
            return mat @ vector

    def determinant_covariance(self):
        std = self._proposed_state["std"]
        r = self._proposed_state["correlation"]
        n = self.dobs.size
        det = std ** (2 * n) * (1 - r**2) ** (n - 1)
        self._proposed_state["determinant_covariance"] = det
        return det
