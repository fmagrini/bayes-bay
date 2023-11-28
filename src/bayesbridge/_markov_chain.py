from typing import Union, List, Callable, Tuple, Any
from numbers import Number
from collections import defaultdict
import random
import math
import numpy
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
        log_prior_func: Callable[[Any], Number] = None,
        log_likelihood_func: Callable[[Any], Number] = None,
        log_prior_ratio_funcs: List[Callable[[Any, Any], Number]] = None,
        log_like_ratio_func: Callable[[Any, Any], Number] = None,
        temperature: int = 1,
    ):
        self.id = id
        self.current_model = starting_model
        self.perturbation_funcs = perturbation_funcs
        self.perturbation_types = [func.__name__ for func in perturbation_funcs]
        self._temperature = temperature
        assert not (log_prior_func is None and log_prior_ratio_funcs is None)
        assert not (log_likelihood_func is None and log_like_ratio_func is None)
        assert log_prior_ratio_funcs is None or len(log_prior_ratio_funcs) == len(
            perturbation_funcs
        )
        self.log_prior_func = log_prior_func
        self.log_likelihood_func = log_likelihood_func
        self.log_prior_ratio_funcs = log_prior_ratio_funcs
        self.log_like_ratio_func = log_like_ratio_func
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
        if isinstance(self.current_model, (State, dict)):
            self._saved_models = defaultdict(list)
            for k in self.current_model:
                self._saved_models[k] = []
        else:
            self._saved_models = []

    def _init_statistics(self):
        self._current_misfit = float("inf")
        self._proposed_counts = defaultdict(int)
        self._accepted_counts = defaultdict(int)
        self._proposed_counts_total = 0
        self._accepted_counts_total = 0
        self._fwd_failure_counts_total = 0

    def _save_model(self):
        if isinstance(self.current_model, (State, dict)):
            for k in self.current_model:
                self.saved_models[k].append(getattr(self.current_model, k))
        else:
            self.saved_models.append(self.current_model)

    def _save_statistics(self, perturb_i, accepted):
        perturb_type = self.perturbation_types[perturb_i]
        self._proposed_counts[perturb_type] += 1
        self._accepted_counts[perturb_type] += 1 if accepted else 0
        self._proposed_counts_total += 1
        self._accepted_counts_total += 1 if accepted else 0

    def _print_statistics(self):
        head = "Chain ID: %s \nEXPLORED MODELS: %s - " % (
            self.id,
            self._proposed_counts_total,
        )
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
        print("NUMBER OF FWD FAILURES: %d" % self._fwd_failure_counts_total)

    def _log_prior_ratio(self, new_model, i_perturb):
        if self.log_prior_ratio_funcs is not None:
            return self.log_prior_ratio_funcs[i_perturb](self.current_model, new_model)
        else:
            log_prior_old = self.log_prior_func(self.current_model)
            log_prior_new = self.log_prior_func(new_model)
            return log_prior_new - log_prior_old

    def _log_likelihood_ratio(self, new_model, i_perturb):
        if self.log_like_ratio_func is not None:
            return self.log_like_ratio_func(self.current_model, new_model)
        else:
            log_likelihood_old = self.log_likelihood_func(self.current_model)
            log_likelihood_new = self.log_likelihood_func(new_model)
            return log_likelihood_new - log_likelihood_old

    def _next_iteration(self, save_model):
        for i in range(500):
            # choose one perturbation function and type
            i_perturb = random.randint(0, len(self.perturbation_funcs) - 1)
            perturb_func = self.perturbation_funcs[i_perturb]

            # perturb and calculate the log proposal ratio
            try:
                new_model, log_proposal_ratio = perturb_func(self.current_model)
            except Exception:
                i -= 1  # this doesn't have to go into failure counter
                continue

            # calculate the log posterior ratio
            log_prior_ratio = self._log_prior_ratio(new_model, i_perturb)
            try:
                log_likelihood_ratio = self._log_likelihood_ratio(new_model, i_perturb)
                tempered_loglike_ratio = log_likelihood_ratio / self.temperature
            except Exception as e:
                raise e
                self._fwd_failure_counts_total += 1
                continue
            log_posterior_ratio = log_prior_ratio + tempered_loglike_ratio

            # decide whether to accept
            log_probability_ratio = log_proposal_ratio + log_posterior_ratio
            accepted = log_probability_ratio > math.log(random.random())
            if accepted:
                self.current_model = new_model

            # save statistics and current model
            self._save_statistics(i_perturb, accepted)
            if save_model and self.temperature == 1:
                self._save_model()
            return
        raise RuntimeError(
            f"Chain {self.id} failed in forward calculation for 500 times"
        )

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
        self.log_like_ratio_func = LogLikelihood(
            targets=targets,
            fwd_functions=fwd_functions,
        )
        self._temperature = temperature
        self.initialize()
        self._init_perturbation_funcs()
        self._init_statistics()
        self._init_saved_models()

    def initialize(self):
        self.current_model = self.parameterization.initialize()

    def _init_perturbation_funcs(self):
        funcs_from_parameterization = self.parameterization.perturbation_functions
        log_priors_from_parameterization = (
            self.parameterization.log_prior_ratio_functions
        )
        funcs_from_log_likelihood = self.log_like_ratio_func.perturbation_functions
        log_priors_from_log_likelihood = (
            self.log_like_ratio_func.log_prior_ratio_functions
        )
        self.perturbation_funcs = (
            funcs_from_parameterization + funcs_from_log_likelihood
        )
        self.perturbation_types = [f.type for f in self.perturbation_funcs]
        self.log_prior_ratio_funcs = (
            log_priors_from_parameterization + log_priors_from_log_likelihood
        )
