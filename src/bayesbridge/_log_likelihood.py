import math
from typing import Any, List, Callable, Tuple
from numbers import Number
import numpy as np

from .exceptions import ForwardException
from ._state import State
from ._target import Target


class LogLikelihood:
    """High-level class that helps evaluate log likelihood ratio

    Parameters
    ----------
    targets : bayesbridge.Target
        a list of data targets
    fwd_functions : Callable[[bayesbridge.State], np.ndarray]
        a lsit of forward functions corresponding to each data targets provided above.
        Each function takes in a model and produces a numpy array of data predictions.
    """

    def __init__(
        self,
        targets: List[Target],
        fwd_functions: Callable[[State], np.ndarray],
    ):
        self.targets = targets
        self.fwd_functions = fwd_functions
        assert len(self.targets) == len(self.fwd_functions)
        self._init_perturbation_funcs()

    @property
    def perturbation_functions(self) -> List[Callable[[State], Tuple[State, Number]]]:
        """A list of perturbation functions associated with the data noise of the
        provided targets.

        Perturbation functions are included in this list only when the data noise of
        the target(s) is explicitly set to be unknown(s).
        """
        return self._perturbation_funcs

    @property
    def log_prior_ratio_functions(self) -> List[Callable[[State], Number]]:
        """A list of log prior ratio functions corresponding to the perturbations in
        :meth:`perturbation_functions`
        """
        return self._log_prior_ratio_funcs

    def log_likelihood_ratio(self, old_state, new_state):
        old_misfit, old_log_det = self._get_misfit_and_det(old_state)
        new_misfit, new_log_det = self._get_misfit_and_det(new_state)
        log_like_ratio = (old_log_det - new_log_det) + (old_misfit - new_misfit) / 2
        temperature = getattr(new_state, "temperature")
        return log_like_ratio / temperature

    def __call__(self, old_misfit, temperature) -> Any:
        return self.log_likelihood_ratio(old_misfit, temperature)

    def _init_perturbation_funcs(self):
        self._perturbation_funcs = []
        self._log_prior_ratio_funcs = []
        for target in self.targets:
            if target.is_hierarchical:
                self._perturbation_funcs.append(target.perturbation_function)
                self._log_prior_ratio_funcs.append(target.log_prior_ratio_function)

    def _get_misfit_and_det(self, state: State) -> Tuple[Number, Number]:
        misfit = 0
        log_det = 0
        for target, fwd_func in zip(self.targets, self.fwd_functions):
            try:
                dpred = fwd_func(state)
            except Exception as e:
                raise ForwardException(e)
            residual = dpred - target.dobs
            misfit += residual @ target.inverse_covariance_times_vector(state, residual)
            if target.is_hierarchical:
                log_det += target.log_determinant_covariance(state)
        return misfit, log_det
