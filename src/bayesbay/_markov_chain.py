from typing import Union, List, Callable, Tuple, Any, Dict
from numbers import Number
from collections import defaultdict
import random
import math

from .likelihood._log_likelihood import LogLikelihood
from .likelihood._target import Target
from .parameterization import Parameterization
from ._state import State
from .exceptions import ForwardException, UserFunctionException


_MAX_INITIAL_STATE_ATTEMPTS = 500


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
            "n_proposed_models": defaultdict(lambda: defaultdict(float)),
            "n_accepted_models": defaultdict(lambda: defaultdict(float)),
            "n_proposed_models_total": 0,
            "n_accepted_models_total": 0,
            "exceptions": defaultdict(int),
        }

    def _save_state(self):
        if isinstance(self.current_state, (State, dict)):
            for k, v in self.current_state.items():
                if (
                    ("n_dimensions" in k)
                    and (hasattr(self, "parameterization"))
                    and (
                        not self.parameterization.parameter_spaces[
                            k[: -len(".n_dimensions")]
                        ].trans_d
                    )
                ):
                    continue
                self.saved_states[k].append(v)
            if self.save_dpred:
                for k, v in self.current_state.cache.items():
                    if "dpred" in k:
                        self.saved_states[k].append(v)
        else:
            self.saved_states.append(self.current_state)

    def _save_statistics(self, perturb_i: int, accepted: bool, proposed_state: State):
        perturb_type = self.perturbation_types[perturb_i]
        if perturb_type.startswith("ParamSpacePerturbation"):
            perturb_stats = proposed_state.load_from_cache("perturb_stats")
            for k, v in perturb_stats.items():
                self._statistics["n_proposed_models"][perturb_type][k] += v
                self._statistics["n_accepted_models"][perturb_type][k] += (
                    v if accepted else 0
                )
        else:
            try:
                self._statistics["n_proposed_models"][perturb_type] += 1
                self._statistics["n_accepted_models"][perturb_type] += (
                    1 if accepted else 0
                )
            except:
                self._statistics["n_proposed_models"][perturb_type] = 1
                self._statistics["n_accepted_models"][perturb_type] = (
                    1 if accepted else 0
                )
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
            if isinstance(_explored, dict):
                print(f"\t{perturb_type}:")
                for sub_perturb_type in sorted(_explored):
                    _sub_explored = _explored[sub_perturb_type]
                    _sub_accepted = _accepted[sub_perturb_type]
                    acceptance_rate = _sub_accepted / _sub_explored * 100
                    print(
                        f"\t\t{sub_perturb_type}: "
                        f"{_sub_accepted:.2f}/{_sub_explored:.2f} ({acceptance_rate:.2f}%)"
                    )
            else:
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
                range(len(self.perturbation_funcs)), self.perturbation_weights
            )[0]
            perturb_func = self.perturbation_funcs[i_perturb]

            # perturb and get the partial acceptance probability excluding log
            # likelihood ratio
            try:
                new_state, log_prob_ratio = perturb_func(self.current_state)
            except UserFunctionException as e:
                _last_exception = e
                self._statistics["exceptions"][e.__class__.__name__] += 1
                continue

            if math.isinf(log_prob_ratio):
                log_likelihood_ratio = 0
            else:
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
            self._save_statistics(i_perturb, accepted, new_state)
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
        starting_state: State = None,
        temperature: float = 1,
        saved_dpred: bool = True,
    ):
        self.parameterization = parameterization
        self.log_likelihood = log_likelihood
        starting_state = self._init_starting_state(starting_state)
        super().__init__(
            id=id,
            starting_state=starting_state,
            perturbation_funcs=perturbation_funcs,
            perturbation_weights=perturbation_weights,
            log_likelihood=log_likelihood,
            temperature=temperature,
            save_dpred=saved_dpred,
        )

    def _init_starting_state(self, starting_state=None) -> State:
        """Initialize the parameterization by defining a starting state."""

        if starting_state is not None:
            ref_state = self.parameterization.initialize()
            self._check_starting_state(starting_state, ref_state)
            self.log_likelihood.initialize(starting_state)
            try:
                self._validate_starting_state(starting_state)
            except (ForwardException, UserFunctionException) as exc:
                raise RuntimeError(
                    "The provided starting_state caused the forward model to fail. "
                    "Please supply a feasible state or adjust the parameterization."
                ) from exc
            return starting_state

        last_exception: Exception = None
        for _ in range(_MAX_INITIAL_STATE_ATTEMPTS):
            candidate_state = self.parameterization.initialize()
            self.log_likelihood.initialize(candidate_state)
            try:
                self._validate_starting_state(candidate_state)
            except (ForwardException, UserFunctionException) as exc:
                last_exception = exc
                continue
            return candidate_state

        raise RuntimeError(
            "Unable to initialize a valid starting state after "
            f"{_MAX_INITIAL_STATE_ATTEMPTS} attempts. Consider providing a "
            "custom starting_state or updating your parameterization/log-likelihood."
        ) from last_exception

    def _validate_starting_state(self, state: State) -> None:
        """Run a forward evaluation to ensure the state is admissible."""
        forward_funcs = getattr(self.log_likelihood, "fwd_functions", None) or []
        if forward_funcs:
            for forward_func in forward_funcs:
                try:
                    forward_func(state)
                except Exception as exc:  # pragma: no cover - defensive
                    raise ForwardException(exc)
            return

        log_like_ratio_func = getattr(self.log_likelihood, "log_like_ratio_func", None)
        if log_like_ratio_func is not None:
            log_like_ratio_func(state, state)
            return

        log_like_func = getattr(self.log_likelihood, "log_like_func", None)
        if log_like_func is not None:
            log_like_func(state)
            return

        self.log_likelihood.log_likelihood_ratio(state, state)

    def _check_starting_state(self, starting_state, ref_state, level=1):
        """Ensure the user's starting state is compatible with the parameterization."""

        start = dict(starting_state.items())
        ref = dict(ref_state.items())

        # check keys and types
        keys_match = set(ref) <= set(start)
        type_match = all(type(ref[k]) == type(start[k]) for k in ref)
        if not (keys_match and type_match):
            raise ValueError(
                "The specified starting state is incompatible with the current "
                f"parameterization.\nExpected items:\n\t{ref_state.items()}"
                f"\nPassed items:\n\t{starting_state.items()}"
            )

        ps_keys = [k for k in ref if "n_dimensions" in k]
        for ps_key in ps_keys:
            base = ".".join(ps_key.split(".")[:level])
            n_dimensions = start[ps_key]
            for key, ref_value in ref.items():
                if key.startswith(f"{base}.") and key != f"{base}.n_dimensions":
                    if isinstance(ref_value, list):
                        for inner in start[key]:
                            self._check_starting_state(inner, ref_value[0], level + 1)
                    else:
                        if len(start[key]) != n_dimensions:
                            raise ValueError(
                                "The specified starting state is incompatible with the "
                                f"current parameterization.\nIssue found in the item "
                                f"`{key}`: it should be {n_dimensions}-"
                                f"dimensional but it is {len(start[key])}-dimensional"
                            )
