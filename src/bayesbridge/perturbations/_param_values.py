from numbers import Number
from typing import Tuple
import random

from .._state import State
from ..parameters._parameters import Parameter
from ._base_perturbation import Perturbation


class ParamPerturbation(Perturbation):
    """Perturbation on a chosen parameter value

    Parameters
    ----------
    param_name : str
        the name of the parameter to be perturbed
    parameter : Parameter
        the :class:`Parameter` instance to be perturbed
    """
    def __init__(
        self,
        param_name: str,
        parameter: Parameter,
    ):
        self.param_name = param_name
        self.parameter = parameter

    def perturb(self, state: State) -> Tuple[State, Number]:
        """perturb one value of the associated parameter, returning a proposed state
        after this perturbation and its associated acceptance criteria excluding log 
        likelihood ratio

        Parameters
        ----------
        state : State
            the current state to perturb from

        Returns
        -------
        Tuple[State, Number]
            proposed new state and the partial acceptance criteria excluding log
            likelihood ratio for this perturbation
        """
        # randomly choose a Voronoi site to perturb the value
        nsites = state.n_voronoi_cells
        isite = random.randint(0, nsites - 1)
        _site = state.voronoi_sites[isite]
        # randomly perturb the value
        old_values = state.get_param_values(self.param_name)
        _old_value = old_values[isite]
        _new_value = self.parameter.perturb_value(_site, _old_value)
        # structure new param value into new state
        new_values = old_values.copy()
        new_values[isite] = self._new_value
        new_state = state.copy()
        new_state.set_param_values(self.param_name, new_values)
        # calculate log prior ratio
        # in this case, log determinant of Jacobian is 0, and log proposal ratio is 0
        log_prior_ratio = self.parameter.log_prior_ratio_perturbation_free_param(
            _old_value, _new_value, _site
        )
        return new_state, log_prior_ratio

    @property
    def __name__(self) -> str:
        return f"{self.type}({self.param_name})"
