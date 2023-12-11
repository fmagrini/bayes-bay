from typing import Union, List, Callable, Tuple, Any, Dict
from numbers import Number
from collections import defaultdict
import random
import math
import numpy
from ._log_likelihood import LogLikelihood
from ._target import Target
from ._parameterizations import Parameterization
from ._state import State
from .exceptions import (
    DimensionalityException, 
    ForwardException, 
    UserFunctionError
)


class BaseMarkovChain:
    """
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
    log_prior_func: Callable[[Any], Number], optional
        the log prior function :math:`\\log p(m)`. It takes in a model (the type of 
        which is consistent with other arguments of this class) and returns the log of 
        the prior density function. This will be used and cannot be None when 
        ``log_prior_ratio_funcs`` is None. Default to None
    log_likelihood_func: Callable[[Any], Number], optional
        the log likelihood function :math:`\\log p(d|m)`. It takes in a model (the type 
        of which is consistent with other arguments of this class) and returns the log 
        of the likelihood function. This will be used and cannot be None when 
        ``log_like_ratio_func`` is None. Default to None
    log_prior_ratio_funcs: List[Callable[[Any, Any], Number]], optional
        a list of log prior ratio functions :math:`\\log (\\frac{p(m_2)}{p(m_1)})`. 
        Each element of this list corresponds to each of the ``perturbation_funcs``. 
        Each function takes in two models (of consistent type as other arguments of
        this class) and returns the log prior ratio as a number. This is utilised in 
        the inversion by default, and ``log_prior_func`` gets used instead only when 
        this argument is None. Default to None
    log_like_ratio_func: Callable[[Any, Any], Number], optional
        the log likelihood ratio function :math:`\\log (\\frac{p(d|m_2)}{p(d|m_1)})`.
        It takes in two models (of consistent type as other arguments of this class) 
        and returns the log likelihood ratio as a number. This is utilised in the 
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
        log_prior_func: Callable[[Any], Number] = None,
        log_likelihood_func: Callable[[Any], Number] = None,
        log_prior_ratio_funcs: List[Callable[[Any, Any], Number]] = None,
        log_like_ratio_func: Callable[[Any, Any], Number] = None,
        temperature: float = 1,
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
    def temperature(self) -> Number:
        """The current temperature used in the log likelihood ratio
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @property
    def saved_models(self) -> Union[Dict[str, list], list]:
        """All the models saved so far
        """
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
            for k in self.current_model:
                try:
                    self.saved_models[k].append(getattr(self.current_model, k))
                except:
                    self.saved_models[k].append(self.current_model.get_param_values(k))
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

    def _log_prior_ratio(self, new_model, i_perturb):
        if self.log_prior_ratio_funcs is not None:
            return self.log_prior_ratio_funcs[i_perturb](self.current_model, new_model)
        else:
            log_prior_old = self.log_prior_func(self.current_model)
            log_prior_new = self.log_prior_func(new_model)
            return log_prior_new - log_prior_old

    def _log_likelihood_ratio(self, new_model):
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
            except (DimensionalityException, UserFunctionError) as e:
                i -= 1  # this doesn't have to go into failure counter
                self._statistics["exceptions"][e.__class__.__name__] += 1
                continue

            # calculate the log posterior ratio
            log_prior_ratio = self._log_prior_ratio(new_model, i_perturb)
            try:
                log_likelihood_ratio = self._log_likelihood_ratio(new_model)
            except (ForwardException, UserFunctionError) as e:
                self._statistics["exceptions"][e.__class__.__name__] += 1
                continue
            tempered_loglike_ratio = log_likelihood_ratio / self.temperature
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
            the frequency in which we print the progress and information during the 
            sampling, by default 100
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
    """
    High-level interface for a Markov Chain.
    
    This is a subclass of :class:`BaseMarkovChain`.
    
    Instantiation of this class is usually done by :class:`BayesianInversion`.
    
    Parameters
    ----------
    id : Union[int, str]
        an integer or a string representing the ID of the current chain. For display
        purposes only
    parameterization : bayesbridge.Parameterization
        pre-configured parameterization. This includes information about the dimension,
        parameterization bounds and properties of unknown parameterizations
    targets : List[bayesbridge.Target]
        a list of data targets
    fwd_functions : Callable[[bayesbridge.State], np.ndarray]
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
        """Initialize the parameterization and the parameter values. In other words, 
        to initialize the starting points of this chain.
        """
        self.current_model = self.parameterization.initialize()
        for target in self.log_like_ratio_func.targets:
            target.initialize(self.current_model)

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
