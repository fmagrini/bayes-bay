from typing import Callable, Tuple, Union
from numbers import Number
import numpy as np

from ._utils_bayes import inverse_covariance
from .perturbations._data_noise import NoisePerturbation
from ._state import State


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
                correlation_perturb_std=correlation_perturb_std
                if noise_is_correlated
                else None,
            )
        else:
            self._perturbation_func = None
            if np.isscalar(covariance_mat_inv):
                self.covariance_mat_inv = covariance_mat_inv
            else:
                self.covariance_mat_inv = np.array(covariance_mat_inv)

    @property
    def perturbation_function(self) -> Callable[[State], Tuple[State, Number]]:
        """A list of perturbation functions generated based on whether there are 
        unknown data noise values such as data noise standard deviation and correaltion
        """
        return self._perturbation_func

    @property
    def log_prior_ratio_function(self) -> Callable[[State], Number]:
        """A list of log prior ratio functions corresponding to each of the 
        perturbation functions
        """
        return self._perturbation_func.log_prior_ratio

    @property
    def is_hierarchical(self):
        """Whether the data noise parameters are unknown (i.e. to be inversed)
        """
        return self.perturbation_function is not None

    def inverse_covariance_times_vector(self, model: State, vector: np.ndarray) -> np.ndarray:
        """Calculates the dot product of the covariance inverse matrix with a given
        vector

        Parameters
        ----------
        model : State
            the current model state
        vector : np.ndarray
            the vector to apply the dot product on

        Returns
        -------
        np.ndarray
            the result from the dot product operation
        """
        if hasattr(self, "covariance_mat_inv"):
            if np.isscalar(self.covariance_mat_inv):
                return self.covariance_mat_inv * vector
            else:
                return self.covariance_mat_inv @ vector
        elif self.correlation is None:
            return 1 / model.noise_std ** 2 * vector
        else:
            std = model.noise_std
            r = model.noise_correlation
            n = self.dobs.size
            mat = inverse_covariance(std, r, n)
            return mat @ vector

    def determinant_covariance(self, model: State) -> float:
        """The determinant value of the covariance matrix

        Parameters
        ----------
        model : State
            the current model state

        Returns
        -------
        float
            the determinant value
        """
        std = model.noise_std
        r = model.noise_correlation
        n = self.dobs.size
        det = std ** (2 * n) * (1 - r**2) ** (n - 1)
        return det
