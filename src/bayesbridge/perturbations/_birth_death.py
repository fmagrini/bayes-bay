from typing import Tuple
from numbers import Number

from bayesbridge._state import State

from ._base_perturbation import Perturbation
from .._state import State


class BirthPerturbation(Perturbation):
    """Perturbation by creating a new dimension
    
    Parameters
    ----------
    parameter_space : ParameterSpace
        instance of :class:`bayesbridge.parameterization.ParameterSpace`
    """
    def __init__(
        self,
        parameter_space: "ParameterSpace",
    ):
        self.param_space = parameter_space
        self.param_space_name = parameter_space.name

    def perturb(self, state: State) -> Tuple[State, Number]:
        """propose a new state that has a new dimension from the given state and
        calculates its associated acceptance probability excluding log likelihood ratio
        
        Parameters
        ----------
        state : State
            the given current state

        Returns
        -------
        Tuple[State, Number]
            proposed new state and the partial acceptance probability excluding log 
            likelihood ratio for this perturbation

        Raises
        ------
        DimensionalityException
            when the current state has already reached the maximum number of Voronoi
            cells
        """
        old_ps_state = state.get_param_values(self.param_space_name)
        new_ps_state, log_prob_ratio = self.param_space.birth(old_ps_state)
        new_state = state.copy()
        new_state.set_param_values(self.param_space_name, new_ps_state)
        return new_state, log_prob_ratio

    @property
    def __name__(self) -> str:
        return f"{self.type}({self.param_space_name})"


class DeathPerturbation(Perturbation):
    """Perturbation by removing an existing dimension
    
    Parameters
    ----------
    parameter_space : ParameterSpace
        instance of :class:`bayesbridge.parameterization.ParameterSpace`
    """
    def __init__(
        self,
        parameter_space: "ParameterSpace",
    ):
        self.param_space = parameter_space
        self.param_space_name = parameter_space.name

    def perturb(self, state: State) -> Tuple[State, Number]:
        """propose a new state that has an existing dimension removed from the given
        state and calculates its associated acceptance probability excluding log 
        likelihood ratio
        
        Parameters
        ----------
        state : State
            the given current state

        Returns
        -------
        Tuple[State, Number]
            proposed new state and the partial acceptance probability excluding log 
            likelihood ratio for this perturbation

        Raises
        ------
        DimensionalityException
            when the current state has already reached the minimum number of Voronoi
            cells
        """
        old_ps_state = state.get_param_values(self.param_space_name)
        new_ps_state, log_prob_ratio = self.param_space.death(old_ps_state)
        new_state = state.copy()
        new_state.set_param_values(self.param_space_name, new_ps_state)
        return new_state, log_prob_ratio
 
    @property
    def __name__(self) -> str:
        return f"{self.type}({self.param_space_name})"
