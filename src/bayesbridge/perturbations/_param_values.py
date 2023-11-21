from numbers import Number
from typing import Tuple
import random

from .._state import State
from .._parameterizations import Parameterization
from ..parameters._parameters import Parameter
from ._base_perturbation import Perturbation


class ParamPerturbation(Perturbation):
    def __init__(
        self, 
        parameterization: Parameterization, 
        param_name: str, 
        parameter: Parameter, 
    ):
        super().__init__(parameterization)
        self.param_name = param_name
        self.parameter = parameter
        
    def perturb(self, model: State) -> Tuple[State, Number]:
        # randomly choose a Voronoi site to perturb the value
        nsites = model.n_voronoi_cells
        isite = random.randint(0, nsites-1)
        site = model.voronoi_sites[isite]
        # randomly perturb the value
        old_values = model.get_param_values(self.param_name)
        new_value = self.parameter.perturb_value(site, old_values[isite])
        # structure new param value into new model
        new_values = old_values.copy()
        new_values[isite] = new_value
        new_model = State(nsites, model.voronoi_sites.copy(), new_values)
        # calculate proposal ratio
        proposal_ratio = 0
        return new_model, proposal_ratio
