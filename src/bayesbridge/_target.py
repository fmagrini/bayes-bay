from typing import Callable, Tuple, Union
from numbers import Number
import random
import math
import numpy as np

from ._utils_1d import inverse_covariance
from ._state import State, DataNoiseState


class Target:
    """Data target that can be configured to have noise level as knowns or unknowns

    Parameters
    ----------
    name : str
        name of the data target, for display purposes only
    dobs : np.ndarray
        data observations
    covariance_mat_inv : Union[Number, np.ndarray], optional
        the inverse of the data covariance matrix, either a number or a full matrix, by
        default None
    noise_is_correlated : bool, optional
        whether the noise between data points are correlated or not, by default False
    std_min : Number, optional
        the minimum value of the standard deviation of data noise, by default 0.01
    std_max : Number, optional
        the maximum value of the standard deviation of data noise, by default 1
    std_perturb_std : Number, optional
        the perturbation standard deviation of the standard deviation of data noise, by
        default 0.1
    correlation_min : Number, optional
        the miminum value of the correlation of data noise, by default 0.01
    correlation_max : Number, optional
        the maximum value of the correlation of data noise, by default 1
    correlation_perturb_std : Number, optional
        the perturbation standard deviation of the standard deviation of data noise, by
        default 0.1
    """

    def __init__(
        self,
        name: str,
        dobs: np.ndarray,
        covariance_mat_inv: Union[Number, np.ndarray] = None,
        noise_is_correlated: bool = False,
        std_min: Number = 0.01,
        std_max: Number = 1,
        std_perturb_std: Number = 0.1,
        correlation_min: Number = 0.01,
        correlation_max: Number = 1,
        correlation_perturb_std: Number = 0.1,
    ):
        self._name = name
        self.dobs = np.array(dobs)
        self.noise_is_correlated = noise_is_correlated
        self.std_min = std_min
        self.std_max = std_max
        self.std_perturb_std = std_perturb_std
        self.correlation_min = correlation_min
        self.correlation_max = correlation_max
        self.correlation_perturb_std = correlation_perturb_std
        if covariance_mat_inv is not None:
            self._perturbation_func = None
            if np.isscalar(covariance_mat_inv):
                self.covariance_mat_inv = covariance_mat_inv
            else:
                self.covariance_mat_inv = np.array(covariance_mat_inv)
        else:
            self.covariance_mat_inv = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_hierarchical(self):
        """whether the data noise parameters are unknown (i.e. to be inversed)"""
        return self.covariance_mat_inv is None

    def initialize(self, state: State):
        """initializes the data noise parameters

        Parameters
        ----------
        state : State
            the current state where initialized DataNoiseState parameter is to be updated to
        """
        if self.is_hierarchical:
            noise_std = random.uniform(self.std_min, self.std_max)
            # state.set_param_values((self.name, "noise_std"), noise_std)
            noise_corr = random.uniform(self.correlation_min, self.correlation_max) \
                if self.noise_is_correlated else None
            state.set_param_values(self.name, DataNoiseState(std=noise_std, correlation=noise_corr))

    def inverse_covariance_times_vector(
        self, state: State, vector: np.ndarray
    ) -> np.ndarray:
        """calculates the dot product of the covariance inverse matrix with a given
        vector

        Parameters
        ----------
        state : State
            the current state state
        vector : np.ndarray
            the vector to apply the dot product on

        Returns
        -------
        np.ndarray
            the result from the dot product operation
        """
        if not self.is_hierarchical:
            if np.isscalar(self.covariance_mat_inv):
                return self.covariance_mat_inv * vector
            else:
                return self.covariance_mat_inv @ vector
        else:
            noise = state.get_param_values(self.name)
            std = noise.std
            correlation = noise.correlation
            if correlation is None:
                return 1 / std**2 * vector
            else:
                n = self.dobs.size
                mat = inverse_covariance(std, correlation, n)
                return mat @ vector

    def log_determinant_covariance(self, state: State) -> float:
        """the log determinant value of the covariance matrix

        Parameters
        ----------
        state : State
            the current state state

        Returns
        -------
        float
            the log determinant value
        """
        noise = state.get_param_values(self.name)
        std = noise.std
        r = noise.correlation
        if r is None:
            r = 0
        n = self.dobs.size
        log_det = (2 * n) * math.log(std) + (n - 1) * math.log(1 - r**2)
        return log_det
