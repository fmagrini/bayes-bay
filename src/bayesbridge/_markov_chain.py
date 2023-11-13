from typing import Union, List, Callable, Tuple
from numbers import Number
from collections import defaultdict
from functools import partial
import random
import math
import numpy
from ._log_likelihood import LogLikelihood
from ._target import Target
from ._parameterizations import Parameterization
from ._state import State
from ._exceptions import ForwardException, DimensionalityException


class MarkovChain:
    def __init__(
        self, 
        id: Union[int, str], 
        starting_pos: List[numpy.ndarray], 
        perturbations: Callable[[numpy.ndarray], Tuple[numpy.ndarray, Number]], 
        log_posterior_func: Callable[[numpy.ndarray], Number], 
        temperature: int = 1, 
    ):
        self.id = id
        self.starting_pos = starting_pos
        self.current_model = starting_pos
        self.perturbations = perturbations
        self.perturbation_types = [func.__name__ for func in perturbations]
        self.log_posterior_func = log_posterior_func
        self._temperature = temperature
        self._init_statistics()
        self._init_saved_models()
    
    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
    
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
        print("CURRENT MISFIT: %.2f" % self._current_misfit)
        print("NUMBER OF FWD FAILURES: %d" % self._fwd_failure_counts_total)

    def _next_iteration(self, save_model):
        for i in range(500):
            # choose one perturbation function and type
            perturb_i = random.randint(0, len(self.perturbations) - 1)
            perturb_func = self.perturbations[perturb_i]
            
            # perturb and calculate the log proposal ratio
            try:
                new_model, log_proposal_ratio = perturb_func(self.current_model)
            except RuntimeError:
                continue
            
            # calculate the log posterior ratio
            try:
                log_posterior_ratio = self.log_posterior_func(new_model)
            except Exception:
                self._fwd_failure_counts_total += 1
                continue
            
            # decide whether to accept
            log_probability_ratio = log_proposal_ratio + log_posterior_ratio
            accepted = log_probability_ratio > math.log(random.random())
            if save_model and self.temperature == 1:
                self._saved_model()
            
            # finalize perturbation based on whether it's accepted
            if accepted:
                self.current_model = new_model
            
            # save statistics
            self._save_statistics(perturb_i, accepted)
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


class MarkovChainFromParameterization(MarkovChain):
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
        self.log_likelihood = LogLikelihood(
            model=parameterization.model,
            targets=targets,
            fwd_functions=fwd_functions,
        )
        self._init_perturbation_funcs()
        self._init_statistics()
        self._temperature = temperature
        self._init_saved_models()
        self._init_saved_targets()

    @property
    def saved_models(self):
        r"""models that are saved in current chain; intialized everytime `advance_chain`
        is called

        It is a Python dict. See the following example:

        .. code-block::

           {
               'n_voronoi_cells': 5,
               'voronoi_sites': array([1.61200696, 3.69444193, 4.25564828, 4.34085936, 9.48688864]),
               'voronoi_cell_extents': array([2.65322445, 1.32182066, 0.32320872, 2.61562018, 0.        ])
           }

        """
        return getattr(self, "_saved_models", None)

    @property
    def saved_targets(self):
        """targets that are saved in current chain; intialized everytime
        `advance_chain` is called
        """
        return getattr(self, "_saved_targets", None)

    def _init_perturbation_funcs(self):
        perturb_voronoi = [self.parameterization.perturbation_voronoi_site]
        finalize_voronoi = [self.parameterization.finalize_perturbation]
        perturb_types = ["VoronoiSite"]
        if self.parameterization.trans_d:
            perturb_voronoi += [
                self.parameterization.perturbation_birth,
                self.parameterization.perturbation_death,
            ]
            finalize_voronoi += [
                self.parameterization.finalize_perturbation,
                self.parameterization.finalize_perturbation,
            ]
            perturb_types += ["Birth", "Death"]

        perturb_free_params = []
        perturb_free_params_types = []
        finalize_free_params = []
        for name in self.parameterization.free_params:
            perturb_free_params.append(
                partial(self.parameterization.perturbation_free_param, param_name=name)
            )
            perturb_free_params_types.append("Param - " + name)
            finalize_free_params.append(self.parameterization.finalize_perturbation)

        perturb_targets = []
        perturb_targets_types = []
        finalize_targets = []
        for target in self.log_likelihood.targets:
            if target.is_hierarchical:
                perturb_targets.append(target.perturb_covariance)
                perturb_targets_types.append("Target - " + target.name)
                finalize_targets.append(target.finalize_perturbation)

        self.perturbations = perturb_voronoi + perturb_free_params + perturb_targets
        self.perturbation_types = (
            perturb_types + perturb_free_params_types + perturb_targets_types
        )
        self.finalizations = finalize_voronoi + finalize_free_params + finalize_targets

        assert len(self.perturbations) == len(self.perturbation_types)
        assert len(self.perturbations) == len(self.finalizations)

    def _init_saved_models(self):
        trans_d = self.parameterization.trans_d
        saved_models = {
            k: []
            for k in self.parameterization.model.current_state
            if k != "n_voronoi_cells"
        }
        saved_models["n_voronoi_cells"] = (
            []
            if trans_d
            else self.parameterization.model.current_state["n_voronoi_cells"]
        )
        saved_models["misfits"] = []
        self._saved_models = saved_models

    def _init_saved_targets(self):
        saved_targets = {}
        for target in self.log_likelihood.targets:
            if target.save_dpred:
                saved_targets[target.name] = {"dpred": []}
            if target.is_hierarchical:
                saved_targets[target.name]["sigma"] = []
                if target.noise_is_correlated:
                    saved_targets[target.name]["correlation"] = []
        self._saved_targets = saved_targets

    def _save_model(self, misfit):
        self.saved_models["misfits"].append(misfit)
        for key, value in self.parameterization.model.proposed_state.items():
            if key == "n_voronoi_cells":
                if isinstance(self.saved_models["n_voronoi_cells"], int):
                    continue
                else:
                    self.saved_models["n_voronoi_cells"].append(value)
            else:
                self.saved_models[key].append(value)

    def _save_target(self):
        for target in self.log_likelihood.targets:
            if target.save_dpred:
                self.saved_targets[target.name]["dpred"].append(
                    self.log_likelihood.proposed_dpred[target.name]
                )
            if target.is_hierarchical:
                self.saved_targets[target.name]["sigma"].append(
                    target._proposed_state["sigma"]
                )
                if target.noise_is_correlated:
                    self.saved_targets[target.name]["correlation"].append(
                        target._proposed_state["correlation"]
                    )

    def _next_iteration(self, save_model):
        for i in range(500):
            # choose one perturbation function and type
            perturb_i = random.randint(0, len(self.perturbations) - 1)

            # propose new model and calculate probability ratios
            try:
                log_prob_ratio = self.perturbations[perturb_i]()
            except DimensionalityException:
                continue

            # calculate the forward and evaluate log_likelihood
            try:
                log_likelihood_ratio, misfit = self.log_likelihood(
                    self._current_misfit, self.temperature
                )
            except ForwardException:
                self.finalizations[perturb_i](False)
                self._fwd_failure_counts_total += 1
                continue

            # decide whether to accept
            accepted = log_prob_ratio + log_likelihood_ratio > math.log(random.random())

            if save_model and self.temperature == 1:
                self._save_model(misfit)
                self._save_target()

            # finalize perturbation based whether it's accepted
            self.finalizations[perturb_i](accepted)
            self._current_misfit = misfit if accepted else self._current_misfit

            # save statistics
            self._save_statistics(perturb_i, accepted)
            return
        raise RuntimeError("Chain getting stuck")
