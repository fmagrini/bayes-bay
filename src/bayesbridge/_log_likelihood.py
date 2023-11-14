import math
from typing import Any
from .exceptions import ForwardException


class LogLikelihood:
    def __init__(self, model, targets, fwd_functions):
        # def forward(proposed_model: State) -> numpy.ndarray
        self.model = model
        self.targets = targets
        self.fwd_functions = fwd_functions
        assert len(self.targets) == len(self.fwd_functions)
        self.proposed_dpred = {}

    def data_misfit(self):
        misfit = 0
        for target, fwd_func in zip(self.targets, self.fwd_functions):
            try:
                dpred = fwd_func(self.model.proposed_state)
            except Exception as e:
                raise ForwardException(e)
            dobs = target.dobs
            residual = dpred - dobs
            misfit += residual @ target.covariance_times_vector(residual)
            self.proposed_dpred[target.name] = dpred
        return misfit

    def log_likelihood_ratio(self, old_misfit, temperature):
        new_misfit = self.data_misfit()
        factor = 0
        for target in self.targets:
            if target.is_hierarchical:
                det_old = target._current_state["determinant_covariance"]
                det_new = target.determinant_covariance()
                factor += math.log(math.sqrt(det_old / det_new))
        new_log_likelihood_ratio = factor + (old_misfit - new_misfit) / (
            2 * temperature
        )
        return new_log_likelihood_ratio, new_misfit

    def __call__(self, old_misfit, temperature) -> Any:
        return self.log_likelihood_ratio(old_misfit, temperature)
