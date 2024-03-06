from typing import List, Tuple
from numbers import Number
import random

from .._state import State, ParameterSpaceState
from ._base_perturbation import Perturbation, ParamSpaceMixin
from ..exceptions._exceptions import DimensionalityException


class ParamSpacePerturbation(Perturbation, ParamSpaceMixin):
    def __init__(
        self,
        param_space_name: str,
        perturbations: List[Perturbation],
        perturbation_weights: List[Number],
    ):
        self.param_space_name = param_space_name
        assert len(perturbations) == len(perturbation_weights), (
            "The number of perturbations and perturbation weights must be equal"
        )
        self._perturbations_functions = perturbations
        self._perturbation_weights = perturbation_weights

    def perturb(self, state: State) -> Tuple[State, Number]:
        new_state = state.copy()
        states_queue = [(new_state, state)]
        log_prob_ratio = 0
        while states_queue:
            new_ps_state, old_ps_state = states_queue.pop(0)
            for k, v in old_ps_state.param_values.items():
                if k == self.param_space_name and isinstance(v, ParameterSpaceState):
                    new_v, ratio = self.perturb_param_space_state(v)
                    log_prob_ratio += ratio
                    new_ps_state.set_param_values(k, new_v)
                elif k == self.param_space_name and isinstance(v, list):
                    new_vv_list = []
                    for vv in v:
                        new_vv, ratio = self.perturb_param_space_state(vv)
                        log_prob_ratio += ratio
                        new_vv_list.append(new_vv)
                    new_ps_state.set_param_values(k, new_vv_list)
                elif isinstance(v, ParameterSpaceState):
                    states_queue.append((new_ps_state[k], v))
                elif isinstance(v, list) and isinstance(v[0], ParameterSpaceState):
                    states_queue.extend(zip(new_ps_state[k], v))
        return new_state, log_prob_ratio

    def perturb_param_space_state(
        self, ps_state: ParameterSpaceState
    ) -> Tuple[ParameterSpaceState, Number]:
        while True:
            # randomly choose a perturbation function for the current ps_state
            i_perturb = random.choices(
                range(len(self.perturbation_functions)), self.perturbation_weights
            )[0]
            perturb_func = self.perturbation_functions[i_perturb]
            # perturb and get the log of the partial acceptance probability
            try:
                new_ps_state, log_prob_ratio = perturb_func.perturb_param_space_state(
                    ps_state
                )
            except DimensionalityException:
                continue
            return new_ps_state, log_prob_ratio

    @property
    def perturbation_functions(self) -> List[Perturbation]:
        return self._perturbations_functions

    @property
    def perturbation_weights(self) -> List[Number]:
        return self._perturbation_weights

    @perturbation_weights.setter
    def perturbation_weights(self, weights: List[float]):
        self._perturbation_weights = weights

    @property
    def __name__(self) -> str:
        name = "ParamSpacePerturbation("
        name += f"param_space_name={self.param_space_name}, "
        name += f"perturbations={self.perturbation_functions}, "
        name += f"perturbation_weights={self.perturbation_weights})"
        return name
