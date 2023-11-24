import math
from typing import Any, List, Callable, Tuple
from numbers import Number
from collections import OrderedDict

from .exceptions import ForwardException
from ._state import State
from ._target import Target


class LogLikelihood:
    def __init__(self, targets: List[Target], fwd_functions):
        # def forward(proposed_model: State) -> numpy.ndarray
        self.targets = targets
        self.fwd_functions = fwd_functions
        assert len(self.targets) == len(self.fwd_functions)
        self._init_perturbation_funcs()
        self._cache_misfit_det = OrderedDict()
        self._max_cache_size = 10
        
    @property
    def perturbation_functions(self) -> List[Callable[[State], Tuple[State, Number]]]:
        return self._perturbation_funcs
    
    @property
    def prior_ratio_functions(self) -> List[Callable[[State], Number]]:
        return self._prior_ratio_funcs
    
    def log_likelihood_ratio(self, old_model, new_model):
        old_misfit, old_log_det = self._get_misfit_and_det(old_model)
        new_misfit, new_log_det = self._get_misfit_and_det(new_model)
        log_like_ratio = (old_log_det - new_log_det) + (old_misfit - new_misfit) / 2
        return log_like_ratio

    def __call__(self, old_misfit, temperature) -> Any:
        return self.log_likelihood_ratio(old_misfit, temperature)
    
    def _init_perturbation_funcs(self):
        self._perturbation_funcs = []
        self._prior_ratio_funcs = []
        for target in self.targets:
            if target.is_hierarchical:
                self._perturbation_funcs.append(target.perturbation_function)
                self._prior_ratio_funcs.append(target.prior_ratio_function)

    def _get_misfit_and_det(self, model: State) -> Tuple[Number, Number]:
        model_hash = hash(model)
        if model_hash in self._cache_misfit_det:
            return self._cache_misfit_det[model_hash]
        else:
            misfit = 0
            log_det = 0
            for target, fwd_func in zip(self.targets, self.fwd_functions):
                try:
                    dpred = fwd_func(model)
                except Exception as e:
                    raise ForwardException(e)
                residual = dpred - target.dobs
                misfit += residual @ target.covariance_times_vector(residual)
                if target.is_hierarchical:
                    log_det += math.log(target.determinant_covariance())
            self._cache_misfit_det[model_hash] = misfit, log_det
            if len(self._cache_misfit_det) > self._max_cache_size:
                self._cache_misfit_det.popitem(last=False)
            return misfit, log_det
