from typing import Tuple
from numbers import Number

from bayesbay._state import ParameterSpaceState

from ._base_perturbation import Perturbation, ParamSpaceMixin
from .._state import State


class BirthPerturbation(Perturbation, ParamSpaceMixin):
    """Perturbation by creating a new dimension

    Parameters
    ----------
    parameter_space : ParameterSpace
        instance of :class:`bayesbay.parameterization.ParameterSpace`
    """

    def __init__(self, parameter_space):
        self.param_space = parameter_space
        self.param_space_name = parameter_space.name

    def perturb(self, state: State) -> Tuple[State, Number]:
        r"""proposes a new state by adding a dimension to the given parameter
        space (:attr:`self.param_space`) and calculates the log of the corresponding
        partial acceptance probability

        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} =
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}}
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}},

        where :math:`{\bf d}_{obs}` denotes the observed data and
        :math:`\mathbf{J}` the Jacobian of the transformation.

        Parameters
        ----------
        state : State
            the given current state

        Returns
        -------
        Tuple[State, Number]
            the proposed new state and
            :math:`\alpha_{p} = \log(
            \frac{p({\bf m'})}{p({\bf m})}
            \frac{q\left({\bf m}
            \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            \lvert \mathbf{J} \rvert)`
        """
        old_ps_state = state[self.param_space_name]
        new_ps_state, log_prob_ratio = self.perturb_param_space_state(old_ps_state)
        new_state = state.copy()
        new_state.set_param_values(self.param_space_name, new_ps_state)
        return new_state, log_prob_ratio

    def perturb_param_space_state(
        self, ps_state: State
    ) -> Tuple[ParameterSpaceState, Number]:
        return self.param_space.birth(ps_state)

    @property
    def __name__(self) -> str:
        return f"{self.type}({self.param_space_name})"


class DeathPerturbation(Perturbation, ParamSpaceMixin):
    """Perturbation by removing an existing dimension

    Parameters
    ----------
    parameter_space : ParameterSpace
        instance of :class:`bayesbay.parameterization.ParameterSpace`
    """

    def __init__(
        self,
        parameter_space,
    ):
        self.param_space = parameter_space
        self.param_space_name = parameter_space.name

    def perturb(self, state: State) -> Tuple[State, Number]:
        r"""proposes a new state by removing a dimension from the given parameter
        space (:attr:`self.param_space`) and calculates the log of the corresponding
        partial acceptance probability

        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} =
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}}
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}}

        where :math:`{\bf d}_{obs}` denotes the observed data and
        :math:`\mathbf{J}` the Jacobian of the transformation.

        Parameters
        ----------
        state : State
            the given current state

        Returns
        -------
        Tuple[State, Number]
            the proposed new state and
            :math:`\alpha_{p} = \log(
            \frac{p({\bf m'})}{p({\bf m})}
            \frac{q\left({\bf m}
            \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            \lvert \mathbf{J} \rvert)`
        """
        old_ps_state = state[self.param_space_name]
        new_ps_state, log_prob_ratio = self.param_space.death(old_ps_state)
        new_state = state.copy()
        new_state.set_param_values(self.param_space_name, new_ps_state)
        return new_state, log_prob_ratio

    def perturb_param_space_state(
        self, ps_state: State
    ) -> Tuple[ParameterSpaceState, Number]:
        return self.param_space.death(ps_state)

    @property
    def __name__(self) -> str:
        return f"{self.type}({self.param_space_name})"
