from typing import Union, List, Callable, Tuple, Any
from numbers import Number
from collections import defaultdict
from functools import partial
import random
import math
import numpy
from .exceptions import ForwardException, DimensionalityException
from ._log_likelihood import LogLikelihood
from ._target import Target
from ._parameterizations import Parameterization
from ._state import State


class BaseMarkovChain:
    """
    Parameters
    ----------
    starting_model : object
        The `starting_model` will be passed to `perturbation_funcs` and 
    """
    def __init__(
        self, 
        id: Union[int, str], 
        starting_model: Any, 
        perturbation_funcs: List[Callable[[Any], Tuple[Any, Number]]], 
        log_prior_func: Callable[[Any], Number], 
        log_likelihood_func: Callable[[Any], Number], 
        temperature: int = 1, 
    ):
        self.id = id
        self.current_model = starting_model
        self.perturbation_funcs = perturbation_funcs
        self.perturbation_types = [func.__name__ for func in perturbation_funcs]
        self._temperature = temperature
        self.log_prior_func = log_prior_func
        self.log_likelihood_func = log_likelihood_func
        self._init_statistics()
        self._init_saved_models()
    
    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
    
    @property
    def saved_models(self):
        return getattr(self, "_saved_models", None)
    
    def _init_saved_models(self):
        self._saved_models = {"model": []}

    def _init_statistics(self):
        self._current_misfit = float("inf")
        self._proposed_counts = defaultdict(int)
        self._accepted_counts = defaultdict(int)
        self._proposed_counts_total = 0
        self._accepted_counts_total = 0
        self._fwd_failure_counts_total = 0
    
    def _save_model(self):
        self.saved_models["model"].append(self.current_model)
    
    def _save_statistics(self, perturb_i, accepted):
        perturb_type = self.perturbation_types[perturb_i]
        self._proposed_counts[perturb_type] += 1
        self._accepted_counts[perturb_type] += 1 if accepted else 0
        self._proposed_counts_total += 1
        self._accepted_counts_total += 1 if accepted else 0

    def _print_statistics(self):
        head = "Chain ID: %s \nEXPLORED MODELS: %s - " % (self.id, self._proposed_counts_total)
        acceptance_rate = (
            self._accepted_counts_total / self._proposed_counts_total * 100
        )
        head += "ACCEPTANCE RATE: %d/%d (%.2f %%)" % (
            self._accepted_counts_total,
            self._proposed_counts_total,
            acceptance_rate,
        )
        print(head)
        print("PARTIAL ACCEPTANCE RATES:")
        for perturb_type in sorted(self._proposed_counts):
            proposed = self._proposed_counts[perturb_type]
            accepted = self._accepted_counts[perturb_type]
            acceptance_rate = accepted / proposed * 100
            print(
                "\t%s: %d/%d (%.2f%%)"
                % (perturb_type, accepted, proposed, acceptance_rate)
            )
        # print("CURRENT MISFIT: %.2f" % self._current_misfit)
        print("NUMBER OF FWD FAILURES: %d" % self._fwd_failure_counts_total)

    def _log_prior_ratio(self, new_model, i_perturb):
        log_prior_old = self.log_prior_func(self.current_model)
        log_prior_new = self.log_prior_func(new_model)
        return log_prior_new - log_prior_old
    
    def _log_likelihood_ratio(self, new_model, i_perturb):
        log_likelihood_old = self.log_likelihood_func(self.current_model)
        log_likelihood_new = self.log_likelihood_func(new_model)
        tempered_ratio = (log_likelihood_new - log_likelihood_old) / self.temperature
        return tempered_ratio       # TODO proofread this tempering!

    def _next_iteration(self, save_model):
        for i in range(100):
            # choose one perturbation function and type
            i_perturb = random.randint(0, len(self.perturbation_funcs) - 1)
            perturb_func = self.perturbation_funcs[i_perturb]
            
            # perturb and calculate the log proposal ratio
            try:
                new_model, log_proposal_ratio = perturb_func(self.current_model)
            except Exception as e:
                # print("LOG:", e)
                continue
            
            # calculate the log posterior ratio
            log_prior_ratio = self._log_prior_ratio(new_model, i_perturb)
            try:
                log_likelihood_ratio = self._log_likelihood_ratio(new_model, i_perturb)
            except Exception as e:
                # print("LOG:", e)
                self._fwd_failure_counts_total += 1
                continue
            log_posterior_ratio = log_prior_ratio + log_likelihood_ratio
            
            # decide whether to accept
            log_probability_ratio = log_proposal_ratio + log_posterior_ratio
            accepted = log_probability_ratio > math.log(random.random())
            if save_model and self.temperature == 1:
                self._save_model()
            
            # finalize perturbation based on whether it's accepted
            if accepted:
                self.current_model = new_model
            
            # save statistics
            self._save_statistics(i_perturb, accepted)
            return
        raise RuntimeError("Chain getting stuck")
    
    def advance_chain(
        self,
        n_iterations=1000,
        burnin_iterations=0,
        save_every=100,
        verbose=True,
        print_every=100,
    ):
        for i in range(1, n_iterations + 1):
            if i <= burnin_iterations:
                save_model = False
            else:
                save_model = not (i - burnin_iterations) % save_every

            self._next_iteration(save_model)
            if verbose and not i % print_every:
                self._print_statistics()

        return self


class MarkovChain(BaseMarkovChain):
    def __init__(
        self,
        id: Union[int, str], 
        parameterization: Parameterization, 
        targets: List[Target], 
        fwd_functions: Callable[[State], numpy.ndarray], 
        temperature: int = 1, 
    ):
        self.id = id
        self.parameterization = parameterization
        self.parameterization.initialize()
        self._log_like_ratio_func = LogLikelihood(
            targets=targets,
            fwd_functions=fwd_functions,
        )
        self._temperature = temperature
        self._init_perturbation_funcs()
        self._init_statistics()
        self._init_saved_models()
        self.initialize()
    
    def _log_prior_ratio(self, new_model, i_perturb):
        return self.log_prior_ratio_funcs[i_perturb](self.current_model, new_model)
    
    def _log_likelihood_ratio(self, new_model, i_perturb):
        # tempered by self.temperature
        log_like_ratio = self._log_like_ratio_func(self.current_model, new_model)
        return log_like_ratio / self.temperature
    
    def _init_perturbation_funcs(self):
        funcs_from_parameterization = self.parameterization.perturbation_functions
        priors_from_parameterization = self.parameterization.prior_ratio_functions
        funcs_from_log_likelihood = self._log_like_ratio_func.perturbation_functions
        priors_from_log_likelihood = self._log_like_ratio_func.prior_ratio_functions
        self.perturbation_funcs = funcs_from_parameterization + funcs_from_log_likelihood
        self.perturbation_types = [f.type for f in self.perturbation_funcs]
        self.log_prior_ratio_funcs = priors_from_parameterization + priors_from_log_likelihood

    def initialize(self):
        self.current_model = self.parameterization.initialize()

    # @property
    # def saved_targets(self):
    #     """targets that are saved in current chain; intialized everytime
    #     `advance_chain` is called
    #     """
    #     return getattr(self, "_saved_targets", None)

    # def _init_saved_models(self):
    #     trans_d = self.parameterization.trans_d
    #     saved_models = {
    #         k: []
    #         for k in self.parameterization.model.current_state
    #         if k != "n_voronoi_cells"
    #     }
    #     saved_models["n_voronoi_cells"] = (
    #         []
    #         if trans_d
    #         else self.parameterization.model.current_state["n_voronoi_cells"]
    #     )
    #     saved_models["misfits"] = []
    #     self._saved_models = saved_models

    # def _init_saved_targets(self):
    #     saved_targets = {}
    #     for target in self.log_likelihood.targets:
    #         if target.save_dpred:
    #             saved_targets[target.name] = {"dpred": []}
    #         if target.is_hierarchical:
    #             saved_targets[target.name]["sigma"] = []
    #             if target.noise_is_correlated:
    #                 saved_targets[target.name]["correlation"] = []
    #     self._saved_targets = saved_targets

    # def _save_model(self, misfit):
    #     self.saved_models["misfits"].append(misfit)
    #     for key, value in self.parameterization.model.proposed_state.items():
    #         if key == "n_voronoi_cells":
    #             if isinstance(self.saved_models["n_voronoi_cells"], int):
    #                 continue
    #             else:
    #                 self.saved_models["n_voronoi_cells"].append(value)
    #         else:
    #             self.saved_models[key].append(value)

    # def _save_target(self):
    #     for target in self.log_likelihood.targets:
    #         if target.save_dpred:
    #             self.saved_targets[target.name]["dpred"].append(
    #                 self.log_likelihood.proposed_dpred[target.name]
    #             )
    #         if target.is_hierarchical:
    #             self.saved_targets[target.name]["sigma"].append(
    #                 target._proposed_state["sigma"]
    #             )
    #             if target.noise_is_correlated:
    #                 self.saved_targets[target.name]["correlation"].append(
    #                     target._proposed_state["correlation"]
    #                 )

    # def _next_iteration(self, save_model):
    #     for i in range(500):
    #         # choose one perturbation function and type
    #         perturb_i = random.randint(0, len(self.perturbation_funcs) - 1)

    #         # propose new model and calculate probability ratios
    #         try:
    #             log_prob_ratio = self.perturbation_funcs[perturb_i]()
    #         except DimensionalityException:
    #             continue

    #         # calculate the forward and evaluate log_likelihood
    #         try:
    #             log_likelihood_ratio, misfit = self.log_likelihood(
    #                 self._current_misfit, self.temperature
    #             )
    #         except ForwardException:
    #             self.finalization_funcs[perturb_i](False)
    #             self._fwd_failure_counts_total += 1
    #             continue

    #         # decide whether to accept
    #         accepted = log_prob_ratio + log_likelihood_ratio > math.log(random.random())

    #         if save_model and self.temperature == 1:
    #             self._save_model(misfit)
    #             self._save_target()

    #         # finalize perturbation based whether it's accepted
    #         self.finalization_funcs[perturb_i](accepted)
    #         self._current_misfit = misfit if accepted else self._current_misfit

    #         # save statistics
    #         self._save_statistics(perturb_i, accepted)
    #         return
    #     raise RuntimeError("Chain getting stuck")
