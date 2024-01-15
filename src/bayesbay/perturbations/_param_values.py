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
    parameters : List[Parameter]
        list containing the :class:`Parameter` instances to be perturbed
    """

    def __init__(
        self,
        param_space_name: str,
        parameters: List[Parameter],
    ):
        self.param_space_name = param_space_name
        self.parameters = parameters

    def perturb(self, state: State) -> Tuple[State, Number]:
        r"""perturb one value for each parameter in :attr:`self.parameters`, returning 
        a proposed state and the log of the corresponding partial acceptance probability 
        
        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} = 
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}} 
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}  
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}},

        Parameters
        ----------
        state : State
            the current state to perturb from

        Returns
        -------
        Tuple[State, Number]
            proposed new state and
            :math:`\alpha_{p} = \log(
            \frac{p({\bf m'})}{p({\bf m})}
            \frac{q\left({\bf m} 
            \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            \lvert \mathbf{J} \rvert)`
        """
        # randomly choose a position to perturb the value(s)
        old_ps_state = state[self.param_space_name]
        n_dims = old_ps_state.n_dimensions
        idx = random.randint(0, n_dims - 1)
        # randomly perturb the value(s)
        new_param_values = dict()
        log_prob_ratio = 0
        for param in self.parameters:
            if hasattr(param, "birth"):  # if it's a discretization
                new_ps_state, log_prob_ratio = param.perturb_value(old_ps_state, idx)
                new_state = state.copy()
                new_state.set_param_values(self.param_space_name, new_ps_state)
                return new_state, log_prob_ratio
            else:
                old_values = old_ps_state[param.name]
                new_param_values[param.name] = old_values.copy()
                old_value = old_values[idx]
                old_pos = old_ps_state["discretization"]
                pos = old_pos[idx] if old_pos is not None else None
                new_value, _ratio = param.perturb_value(old_value, pos)
                log_prob_ratio += _ratio
                new_param_values[param.name][idx] = new_value
        # structure new param value(s) into new state
        new_state = state.copy()
        new_state[self.param_space_name].param_values.update(new_param_values)
        return new_state, log_prob_ratio

    @property
    def __name__(self) -> str:
        param_names = [p.name for p in self.parameters]
        param_names_str = (
            str(param_names) if len(param_names) != 1 else str(param_names[0])
        )
        return f"{self.type}({param_names_str})"
