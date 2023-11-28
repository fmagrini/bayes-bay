from numbers import Number
from typing import Tuple
import random

from .._state import State
from ..parameters._parameters import Parameter
from ._base_perturbation import Perturbation


class ParamPerturbation(Perturbation):
    def __init__(
        self,
        param_name: str,
        parameter: Parameter,
    ):
        self.param_name = param_name
        self.parameter = parameter

    def perturb(self, model: State) -> Tuple[State, Number]:
        # randomly choose a Voronoi site to perturb the value
        nsites = model.n_voronoi_cells
        isite = random.randint(0, nsites - 1)
        self._site = model.voronoi_sites[isite]
        # randomly perturb the value
        old_values = model.get_param_values(self.param_name)
        self._old_value = old_values[isite]
        self._new_value = self.parameter.perturb_value(self._site, self._old_value)
        # structure new param value into new model
        new_values = old_values.copy()
        new_values[isite] = self._new_value
        new_model = model.clone()
        new_model.set_param_values(self.param_name, new_values)
        # calculate proposal ratio
        proposal_ratio = 0
        return new_model, proposal_ratio

    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        # p(k) ratio and p(c|k) ratio both evaluate to 0
        # calculate only p(v|c) below
        return self.parameter.log_prior_ratio_perturbation_free_param(
            self._old_value, self._new_value, self._site
        )
    
    @property
    def __name__(self) -> str:
        return f"{self.type}({self.param_name})"
