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
        after this perturbation and its log proposal ratio

        Parameters
        ----------
        state : State
            the current state to perturb from

        Returns
        -------
        Tuple[State, Number]
            the proposed state and its associated log proposal ratio
        """
        # randomly choose a Voronoi site to perturb the value
        nsites = state.n_voronoi_cells
        isite = random.randint(0, nsites - 1)
        self._site = state.voronoi_sites[isite]
        # randomly perturb the value
        old_values = state.get_param_values(self.param_name)
        self._old_value = old_values[isite]
        self._new_value = self.parameter.perturb_value(self._site, self._old_value)
        # structure new param value into new state
        new_values = old_values.copy()
        new_values[isite] = self._new_value
        new_state = state.copy()
        new_state.set_param_values(self.param_name, new_values)
        # calculate proposal ratio
        proposal_ratio = 0
        return new_state, proposal_ratio

    def log_prior_ratio(self, old_state: State, new_state: State) -> Number:
        """the log prior ratio for this parameter value perturbation
        
        Since only the value for one parameter is changed, the log prior ratio cancels
        out on all the factors that aren't changed, hence is the log prior ratio of the
        free parameter itself.

        Parameters
        ----------
        old_state : State
            the old state to perturb from
        new_state : State
            the new state to perturb into

        Returns
        -------
        Number
            the log prior ratio for a parameter value perturbation
        """
        # p(k) ratio and p(c|k) ratio (if any) both evaluate to 0
        # calculate only p(v|c) below
        return self.parameter.log_prior_ratio_perturbation_free_param(
            self._old_value, self._new_value, self._site
        )

    @property
    def __name__(self) -> str:
        return f"{self.type}({self.param_name})"
