from typing import List, Tuple
from numbers import Number
from collections import defaultdict
import random

from .._state import State, ParameterSpaceState
from ._base_perturbation import Perturbation, ParamSpaceMixin


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
        
        n_ps_states_to_perturb = 0
        states_queue = [state]
        while states_queue:
            old_ps_state = states_queue.pop(0)
            for k, v in old_ps_state.param_values.items():
                if k == self.param_space_name and isinstance(v, ParameterSpaceState):
                    n_ps_states_to_perturb += 1
                elif k == self.param_space_name and isinstance(v, list):
                    n_ps_states_to_perturb += len(v)
                elif isinstance(v, ParameterSpaceState):
                    states_queue.append(v)
                elif isinstance(v, list) and isinstance(v[0], ParameterSpaceState):
                    states_queue.extend(v)
        
        i_to_perturb = random.randint(0, n_ps_states_to_perturb - 1)
        i_ps_state = 0
        
        log_prob_ratio = 0
        stats = defaultdict(int)
        states_queue = [(new_state, state)]
        while states_queue and i_ps_state <= i_to_perturb:
            new_ps_state, old_ps_state = states_queue.pop(0)
            for k, v in old_ps_state.param_values.items():
                if k == self.param_space_name and isinstance(v, ParameterSpaceState):
                    if i_ps_state == i_to_perturb:
                        new_v, ratio, perturb_type = self.perturb_param_space_state(v)
                        new_ps_state.set_param_values(k, new_v)
                        log_prob_ratio += ratio
                        stats[perturb_type] += 1
                    i_ps_state += 1
                elif k == self.param_space_name and isinstance(v, list):
                    new_vv_list = []
                    for vv in v:
                        if i_ps_state == i_to_perturb:
                            new_vv, ratio, perturb_type = self.perturb_param_space_state(vv)
                            new_vv_list.append(new_vv)
                            log_prob_ratio += ratio
                            stats[perturb_type] += 1
                        else:
                            new_vv_list.append(vv.copy())
                        i_ps_state += 1
                    new_ps_state.set_param_values(k, new_vv_list)
                elif isinstance(v, ParameterSpaceState):
                    states_queue.append((new_ps_state[k], v))
                elif isinstance(v, list) and isinstance(v[0], ParameterSpaceState):
                    states_queue.extend(zip(new_ps_state[k], v))

        stats = {k: v / sum(stats.values()) for k, v in stats.items()}
        new_state.save_to_cache("perturb_stats", stats)
        return new_state, log_prob_ratio

    def perturb_param_space_state(
        self, ps_state: ParameterSpaceState
    ) -> Tuple[ParameterSpaceState, Number, str]:
        while True:
            # randomly choose a perturbation function for the current ps_state
            i_perturb = random.choices(
                range(len(self.perturbation_funcs)), self.perturbation_weights
            )[0]
            perturb_func = self.perturbation_funcs[i_perturb]
            # perturb and get the log of the partial acceptance probability
            new_ps_state, log_prob_ratio = perturb_func.perturb_param_space_state(
                    ps_state
                    )
            return new_ps_state, log_prob_ratio, perturb_func.__name__

    @property
    def perturbation_funcs(self) -> List[Perturbation]:
        return self._perturbations_functions
    
    @perturbation_funcs.setter
    def perturbation_funcs(self, perturbations: List[Perturbation]):
        self._perturbations_functions = perturbations

    @property
    def perturbation_weights(self) -> List[Number]:
        return self._perturbation_weights

    @perturbation_weights.setter
    def perturbation_weights(self, weights: List[float]):
        self._perturbation_weights = weights

    @property
    def __name__(self) -> str:
        name = f"ParamSpacePerturbation(param_space_name={self.param_space_name})"
        return name
