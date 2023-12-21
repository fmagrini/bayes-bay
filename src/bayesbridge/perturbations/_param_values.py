from numbers import Number
from typing import Tuple, List
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
        param_space_name: str,
        parameters: List[Parameter],
    ):
        self.param_space_name = param_space_name
        self.parameters = parameters

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
        # randomly choose a position to perturb the value(s)
        old_ps_state = state.get_param_values(self.param_space_name)
        n_dims = old_ps_state.n_dimensions
        isite = random.randint(0, n_dims - 1)
        # randomly perturb the value(s)
        new_param_values = dict()
        log_prob_ratio = 0
        for param in self.parameters:
            old_values = old_ps_state.param_values[param.name]
            new_param_values[param.name] = old_values.copy()
            if isinstance(param, "ParameterSpace"):    # if it's a discretization
                new_value, _ratio = param.perturb_value(old_ps_state, isite)
            else:
                pos = getattr(old_ps_state, self.param_space_name, None)
                old_value = old_values[isite]
                new_value, _ratio = param.perturb_value(pos, old_value)
            log_prob_ratio += _ratio
            new_param_values[param.name][isite] = new_value
        # structure new param value(s) into new state
        new_state = state.copy()
        new_state.get_param_values(self.param_space_name).param_values.update(new_param_values)
        return new_state, log_prob_ratio

    @property
    def __name__(self) -> str:
        return f"{self.type}({self.param_name})"
