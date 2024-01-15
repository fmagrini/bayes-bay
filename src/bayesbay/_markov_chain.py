from typing import Union, List, Callable, Tuple, Any, Dict
from numbers import Number
from collections import defaultdict
import random
import math
import numpy
from ._log_likelihood import LogLikelihood
from ._target import Target
from .parameterization import Parameterization
from ._state import State
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
    starting_model : Any
        starting model of the current chain
    perturbation_funcs : List[Callable[[Any], Tuple[Any, Number]]]
        a list of perturbation functions
    log_like_ratio_func: Callable[[Any, Any], Number], optional
        function that calculates that the log likelihood ratio 
        :math:`\frac{p\left({{\bf d}_{obs} \mid  {\bf m'}}\right)}{p\left({{\bf d}_{obs} \mid  {\bf m}}\right)}`.
        It takes in two models (of consistent type as other arguments of this class)
        and returns a scalar corresponding to the log likelihood ratio. This is utilised in the
        inversion by default, and ``log_likelihood_func`` gets used instead only when
        this argument is None. Default to None
    temperature : int, optional
        used to temper the log likelihood, by default 1
    """

    def __init__(
        self,
        id: Union[int, str],
        starting_model: Any,
        perturbation_funcs: List[Callable[[Any], Tuple[Any, Number]]],
        log_like_ratio_func: Callable[[Any, Any], Number] = None,
        temperature: float = 1,
    ):
        self.id = id
        self.current_model = starting_model
        self.perturbation_funcs = perturbation_funcs
        self.perturbation_types = [func.__name__ for func in perturbation_funcs]
        self._temperature = temperature
        self.log_like_ratio_func = log_like_ratio_func
        self._init_statistics()
        self._init_saved_models()

    @property
    def temperature(self) -> Number:
        """The current temperature used in the log likelihood ratio"""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        self.current_model.temperature = value

    @property
    def saved_models(self) -> Union[Dict[str, list], list]:
        """All the models saved so far"""
        return getattr(self, "_saved_models", None)

    @property
    def statistics(self):
        return self._statistics

    def _init_saved_models(self):
        if isinstance(self.current_model, (State, dict)):
            self._saved_models = defaultdict(list)
        else:
            self._saved_models = []

    def _init_statistics(self):
        self._statistics = {
            "current_misfit": float("inf"),
            "n_explored_models": defaultdict(int),
            "n_accepted_models": defaultdict(int),
            "n_explored_models_total": 0,
            "n_accepted_models_total": 0,
            "exceptions": defaultdict(int),
        }

    def _save_model(self):
        if isinstance(self.current_model, (State, dict)):
            for k, v in self.current_model.items():
                self.saved_models[k].append(v)
        else:
            self.saved_models.append(self.current_model)

    def _save_statistics(self, perturb_i, accepted):
        perturb_type = self.perturbation_types[perturb_i]
        self._statistics["n_explored_models"][perturb_type] += 1
        self._statistics["n_accepted_models"][perturb_type] += 1 if accepted else 0
        self._statistics["n_explored_models_total"] += 1
        self._statistics["n_accepted_models_total"] += 1 if accepted else 0

    def _print_statistics(self):
        head = f"Chain ID: {self.id}"
        head += f"\nTEMPERATURE: {self.temperature}"
        head += f"\nEXPLORED MODELS: {self.statistics['n_explored_models_total']}"
        _accepted_total = self.statistics["n_accepted_models_total"]
        _explored_total = self.statistics["n_explored_models_total"]
        acceptance_rate = _accepted_total / _explored_total * 100
        head += "\nACCEPTANCE RATE: %d/%d (%.2f %%)" % (
            _accepted_total,
            _explored_total,
            acceptance_rate,
        )
        print(head)
        print("PARTIAL ACCEPTANCE RATES:")
        _accepted_all = self.statistics["n_accepted_models"]
        _explored_all = self.statistics["n_explored_models"]
        for perturb_type in sorted(self.statistics["n_explored_models"]):
            _explored = _explored_all[perturb_type]
            _accepted = _accepted_all[perturb_type]
            acceptance_rate = _accepted / _explored * 100
            print(
                "\t%s: %d/%d (%.2f%%)"
                % (perturb_type, _accepted, _explored, acceptance_rate)
            )
        # print("NUMBER OF FWD FAILURES: %d" % self.statistics["n_fwd_failures_total"])

    def _log_likelihood_ratio(self, new_model):
        return self.log_like_ratio_func(self.current_model, new_model)
    
    def _next_iteration(self, save_model):
        _last_exception = None
        for i in range(500):
            # choose one perturbation function and type
            i_perturb = random.randint(0, len(self.perturbation_funcs) - 1)
            perturb_func = self.perturbation_funcs[i_perturb]

            # perturb and get the partial acceptance probability excluding log 
            # likelihood ratio
            try:
                new_model, log_prob_ratio = perturb_func(self.current_model)
            except (DimensionalityException, UserFunctionException) as e:
                _last_exception = e
                self._statistics["exceptions"][e.__class__.__name__] += 1
                continue

            # calculate the log likelihood ratio
            try:
                log_likelihood_ratio = self._log_likelihood_ratio(new_model)
            except (ForwardException, UserFunctionException) as e:
                _last_exception = e
                self._statistics["exceptions"][e.__class__.__name__] += 1
                continue

            # decide whether to accept
            acceptance_probability = log_prob_ratio + log_likelihood_ratio
            accepted = acceptance_probability > math.log(random.random())
            if accepted:
                self.current_model = new_model

            # save statistics and current model
            self._save_statistics(i_perturb, accepted)
            if save_model and self.temperature == 1:
                self._save_model()
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
        on_begin_iteration: Callable[["BaseMarkovChain"], None] = None,
        on_end_iteration: Callable[["BaseMarkovChain"], None] = None,
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
        on_begin_iteration : Callable[["BaseMarkovChain"], None], optional
            customized function that's to be run at before an iteration
        on_end_iteration : Callable[["BaseMarkovChain"], None], optional
            customized function that's to be run at after an iteration

        Returns
        -------
        BaseMarkovChain
            the chain itself is returned
        """
        for i in range(1, n_iterations + 1):
            if i <= burnin_iterations:
                save_model = False
            else:
                save_model = not (i - burnin_iterations) % save_every

            on_begin_iteration(self)
            self._next_iteration(save_model)
            on_end_iteration(self)
            if verbose and not i % print_every:
                self._print_statistics()

        return self


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
    targets : List[bayesbay.Target]
        a list of data targets
    fwd_functions : Callable[[bayesbay.State], np.ndarray]
        a lsit of forward functions corresponding to each data targets provided above.
        Each function takes in a model and produces a numpy array of data predictions.
    temperature : int, optional
        used to temper the log likelihood, by default 1
    """

    def __init__(
        self,
        id: Union[int, str],
        parameterization: Parameterization,
        targets: List[Target],
        fwd_functions: Callable[[State], numpy.ndarray],
        temperature: float = 1,
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
        """Initialize the parameterization by defining a starting model.
        """
        self.current_model = self.parameterization.initialize()
        for target in self.log_like_ratio_func.targets:
            target.initialize(self.current_model)

    def _init_perturbation_funcs(self):
        funcs_from_parameterization = self.parameterization.perturbation_functions
        funcs_from_log_likelihood = self.log_like_ratio_func.perturbation_functions
        self.perturbation_funcs = (
            funcs_from_parameterization + funcs_from_log_likelihood
        )
        self.perturbation_types = [f.__name__ for f in self.perturbation_funcs]
