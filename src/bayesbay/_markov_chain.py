from typing import Union, List, Callable, Tuple, Any, Dict
from numbers import Number
from collections import defaultdict
import random
import math

from ._log_likelihood import LogLikelihood
from .parameterization import Parameterization
from ._state import State
from ._target import Target
from .exceptions import DimensionalityException, ForwardException, UserFunctionException


class BaseMarkovChain:
    r"""
    Low-level interface for a Markov Chain.

    Instantiation of this class is usually done by :class:`BaseBayesianInversion`.

    Parameters
    ----------
    id : Union[int, str]
        an integer or a string representing the ID of the current chain. For display
        purposes only
    starting_state : Any
        starting state of the current chain
    perturbation_funcs : List[Callable[[Any], Tuple[Any, Number]]]
        a list of perturbation functions
    perturbation_weights: List[Number]
        a list of perturbation weights
    log_likelihood : bayesbay.LogLikelihood
        instance of the ``bayesbay.LogLikelihood`` class
    temperature : int, optional
        used to temper the log likelihood, by default 1
    """

    def __init__(
        self,
        id: Union[int, str],
        starting_state: Any,
        perturbation_funcs: List[Callable[[Any], Tuple[Any, Number]]],
        perturbation_weights: List[Number], 
        log_likelihood: LogLikelihood,
        temperature: float = 1,
        save_dpred: bool = True,
    ):
        self.id = id
        self.current_state = starting_state
        self._temperature = temperature
        self.log_likelihood = log_likelihood
        self.save_dpred = save_dpred
        self.set_perturbation_funcs(perturbation_funcs, perturbation_weights)
        self._init_statistics()
        self._init_saved_states()

    @property
    def current_state(self) -> Union[State, Any]:
        return self._current_state
    
    @current_state.setter
    def current_state(self, state: Union[State, Any]):
        self._current_state = state

    @property
    def temperature(self) -> Number:
        """The current temperature used in the log likelihood ratio"""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        if isinstance(self.current_state, State):
            self.current_state.temperature = value

    @property
    def saved_states(self) -> Union[Dict[str, list], list]:
        """All the states saved so far"""
        return getattr(self, "_saved_states", None)

    @property
    def statistics(self):
        return self._statistics
    
    @property
    def ith_iteration(self):
        return self.statistics["n_proposed_models_total"]

    def set_perturbation_funcs(self, funcs: List[Callable], weights: List[Number]):
        self.perturbation_funcs = funcs
        self.perturbation_types = [func.__name__ for func in funcs]
        self.perturbation_weights = weights
    
    def update_targets(self, targets: List[Target]):
        for target in targets:
            target.initialize(self.current_state)
    
    def _init_saved_states(self):
        if isinstance(self.current_state, (State, dict)):
            self._saved_states = defaultdict(list)
        else:
            self._saved_states = []

    def _init_statistics(self):
        self._statistics = {
            "n_proposed_models": defaultdict(int),
            "n_accepted_models": defaultdict(int),
            "n_proposed_models_total": 0,
            "n_accepted_models_total": 0,
            "exceptions": defaultdict(int),
        }

    def _save_state(self):
        if isinstance(self.current_state, (State, dict)):
            for k, v in self.current_state.items():
                if (
                    "n_dimensions" not in k
                    or (
                        hasattr(self, "parameterization") and
                        self.parameterization.parameter_spaces[
                        k[: -len(".n_dimensions")]
                    ].trans_d)
                ):
                    self.saved_states[k].append(v)
            if self.save_dpred:
                for k, v in self.current_state.cache.items():
                    if "dpred" in k:
                        self.saved_states[k].append(v)
        else:
            self.saved_states.append(self.current_state)

    def _save_statistics(self, perturb_i, accepted):
        perturb_type = self.perturbation_types[perturb_i]
        self._statistics["n_proposed_models"][perturb_type] += 1
        self._statistics["n_accepted_models"][perturb_type] += 1 if accepted else 0
        self._statistics["n_proposed_models_total"] += 1
        self._statistics["n_accepted_models_total"] += 1 if accepted else 0

    def print_statistics(self):
        """print the statistics about the Markov Chain history, including the number of
        explored and accepted states for each perturbation, acceptance rates and the 
        current temperature
        """
        head = f"Chain ID: {self.id}"
        head += f"\nTEMPERATURE: {self.temperature}"
        head += f"\nEXPLORED MODELS: {self.statistics['n_proposed_models_total']}"
        _accepted_total = self.statistics["n_accepted_models_total"]
        _explored_total = self.statistics["n_proposed_models_total"]
        acceptance_rate = _accepted_total / _explored_total * 100
        head += "\nACCEPTANCE RATE: %d/%d (%.2f %%)" % (
            _accepted_total,
            _explored_total,
            acceptance_rate,
        )
        print(head)
        print("PARTIAL ACCEPTANCE RATES:")
        _accepted_all = self.statistics["n_accepted_models"]
        _explored_all = self.statistics["n_proposed_models"]
        for perturb_type in sorted(self.statistics["n_proposed_models"]):
            _explored = _explored_all[perturb_type]
            _accepted = _accepted_all[perturb_type]
            acceptance_rate = _accepted / _explored * 100
            print(
                "\t%s: %d/%d (%.2f%%)"
                % (perturb_type, _accepted, _explored, acceptance_rate)
            )

    def _log_likelihood_ratio(self, new_state):
        return self.log_likelihood.log_likelihood_ratio(
            self.current_state, new_state, self.temperature
        )

    def _next_iteration(self):
        _last_exception = None
        for i in range(500):
            # choose one perturbation function and type
            i_perturb = random.choices(
                range(len(self.perturbation_funcs)), 
                self.perturbation_weights
            )[0]
            perturb_func = self.perturbation_funcs[i_perturb]

            # perturb and get the partial acceptance probability excluding log
            # likelihood ratio
            try:
                new_state, log_prob_ratio = perturb_func(self.current_state)
            except (DimensionalityException, UserFunctionException) as e:
                _last_exception = e
                self._statistics["exceptions"][e.__class__.__name__] += 1
                continue

            # calculate the log likelihood ratio
            try:
                log_likelihood_ratio = self._log_likelihood_ratio(new_state)
            except (ForwardException, UserFunctionException) as e:
                _last_exception = e
                self._statistics["exceptions"][e.__class__.__name__] += 1
                continue

            # decide whether to accept
            acceptance_probability = log_prob_ratio + log_likelihood_ratio
            accepted = acceptance_probability > math.log(random.random())
            if accepted:
                self.current_state = new_state

            # save statistics and current state
            self._save_statistics(i_perturb, accepted)
            if self.save_current_iteration and self.temperature == 1.0:
                self._save_state()
            return
        raise RuntimeError(
            f"Chain {self.id} failed in perturb or forward calculation for 500 times. "
            "See above for the last exception."
        ) from _last_exception

    def advance_chain(
        self,
        n_iterations: int = 1000,
        burnin_iterations: int = 0,
        save_every: int = 100,
        verbose: bool = True,
        print_every: int = 100,
        begin_iteration: Callable[["BaseMarkovChain"], None] = None,
        end_iteration: Callable[["BaseMarkovChain"], None] = None,
    ):
        """advance the chain for a given number of iterations

        Parameters
        ----------
        n_iterations : int, optional
            the number of iterations to advance, by default 1000
        burnin_iterations : int, optional
            the iteration number from which we start to save samples, by default 0
        save_every : int, optional
            the frequency in which we save the samples, by default 100
        verbose : bool, optional
            whether to print the progress during sampling or not, by default True
        print_every : int, optional
            the frequency with which we print the progress and information during the
            sampling, by default 100 iterations
        begin_iteration : Callable[["BaseMarkovChain"], None], optional
            customized function that's to be run at before an iteration
        end_iteration : Callable[["BaseMarkovChain"], None], optional
            customized function that's to be run at after an iteration

        Returns
        -------
        BaseMarkovChain
            the chain itself is returned
        """
        for _ in range(n_iterations):
            i = self.ith_iteration + 1
            if i <= burnin_iterations:
                self.save_current_iteration = False
            else:
                self.save_current_iteration = not (i - burnin_iterations) % save_every

            begin_iteration(self)
            self._next_iteration()
            end_iteration(self)
            if verbose and not i % print_every:
                self.print_statistics()

        return self

    def _repr_args(self) -> dict:
        return {
            "id": self.id, 
            "temperature": self.temperature, 
            "n_proposed_models_total": self.statistics["n_proposed_models_total"], 
            "n_accepted_models_total": self.statistics["n_accepted_models_total"], 
        }
    
    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}("
        for k, v in self._repr_args().items():
            string += f"{k}={v}, "
        return f"{string[:-2]})"


class MarkovChain(BaseMarkovChain):
    """High-level interface for a Markov Chain.

    This is a subclass of :class:`BaseMarkovChain`.

    Instantiation of this class is usually done by :class:`BayesianInversion`.

    Parameters
    ----------
    id : Union[int, str]
        an integer or a string representing the ID of the current chain. For display
        purposes only
    parameterization : bayesbay.parameterization.Parameterization
        pre-configured parameterization. This includes information about the dimension,
        parameterization bounds and properties of unknown parameterizations
    log_likelihood : bayesbay.LogLikelihood
        instance of the ``bayesbay.LogLikelihood`` class
    perturbation_funcs : List[Callable]
        a list of perturbation functions (generated automatically by 
        :class:`BayesianInversion` from ``parameterization`` and ``log_likelihood``)
    perturbation_weights: List[Number]
        a list of perturbation weights
    temperature : int, optional
        used to temper the log likelihood, by default 1
    """

    def __init__(
        self,
        id: Union[int, str],
        parameterization: Parameterization,
        log_likelihood: LogLikelihood, 
        perturbation_funcs: List[Callable], 
        perturbation_weights: List[Number], 
        temperature: float = 1,
        saved_dpred: bool = True,
    ):
        self.parameterization = parameterization
        self.log_likelihood = log_likelihood
        starting_state = self._init_starting_state()
        super().__init__(
            id=id, 
            starting_state=starting_state, 
            perturbation_funcs=perturbation_funcs, 
            perturbation_weights=perturbation_weights, 
            log_likelihood=log_likelihood, 
            temperature=temperature,
            save_dpred=saved_dpred, 
        )

    def _init_starting_state(self) -> State:
        """Initialize the parameterization by defining a starting state."""
        starting_state = self.parameterization.initialize()
        self.log_likelihood.initialize(starting_state)
        return starting_state
    