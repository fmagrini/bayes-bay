from typing import List, Callable, Tuple, Any, Dict, Union
from numbers import Number
from copy import deepcopy
from collections import defaultdict
import numpy as np

from ._markov_chain import MarkovChain, BaseMarkovChain
from .samplers import VanillaSampler, Sampler
from ._parameterizations import Parameterization
from ._target import Target
from ._state import State


class BaseBayesianInversion:
    r"""
    A low-level class for performing Bayesian inversion using Markov Chain Monte Carlo 
    (McMC) methods.

    This class provides the basic structure for setting up and running MCMC sampling, 
    given user-provided definition of prior and likelihood functions and the 
    initialization of walkers.
    
    Parameters
    ----------
    walkers_starting_models: List[Any]
        a list of starting models for each chain. The models can be of any type so long
        as they are consistent with what is accepted as arguments in the perturbation
        functions and probability functions. The length of this list must be equal to 
        the number of chains, i.e. ``n_chains``
    perturbation_funcs: List[Callable[[Any], Tuple[Any, Number]]]
        a list of perturbation functions. Each of which takes in a model (whichever the
        allowed type is, as long as it's consistent with ``walkers_starting_models`` 
        and other probability functions), produces a new model and log of the
        corresponding proposal probability ratio.
    log_prior_func: Callable[[Any], Number], optional
        the log prior function :math:`log p(m)`. It takes in a model (the type of which 
        is consistent with other arguments of this class) and returns the log of the 
        prior density function. This will be used and cannot be None when 
        ``log_prior_ratio_funcs`` is None. Default to None
    log_likelihood_func: Callable[[Any], Number], optional
        the log likelihood function :math:`\log p(d|m)`. It takes in a model (the type 
        of which is consistent with other arguments of this class) and returns the log 
        of the likelihood function. This will be used and cannot be None when 
        ``log_like_ratio_func`` is None. Default to None
    log_prior_ratio_funcs: List[Callable[[Any, Any], Number]], optional
        a list of log prior ratio functions :math:`\log (\frac{p(m_2)}{p(m_1)})`. Each 
        element of this list corresponds to each of the ``perturbation_funcs``. Each 
        function takes in two models (of consistent type as other arguments of this 
        class) and returns the log prior ratio as a number. This is utilised in the 
        inversion by default, and ``log_prior_func`` gets used instead only when this 
        argument is None. Default to None
    log_like_ratio_func: Callable[[Any, Any], Number], optional
        the log likelihood ratio function :math:`\log (\frac{p(d|m_2)}{p(d|m_1)})`.
        It takes in two models (of consistent type as other arguments of this class) 
        and returns the log likelihood ratio as a number. This is utilised in the 
        inversion by default, and ``log_likelihood_func`` gets used instead only when
        this argument is None. Default to None
    n_chains: int, optional
        the number of chains in the McMC sampling, default to 10
    n_cpus: int, optional
        the number of CPUs available. This is usually set to be equal to the number of
        chains if there are enough CPUs, default to 10
    """
    def __init__(
        self,
        walkers_starting_models: List[Any],
        perturbation_funcs: List[Callable[[Any], Tuple[Any, Number]]],
        log_prior_func: Callable[[Any], Number] = None,
        log_likelihood_func: Callable[[Any], Number] = None,
        log_prior_ratio_funcs: List[Callable[[Any, Any], Number]] = None,
        log_like_ratio_func: Callable[[Any, Any], Number] = None,
        n_chains: int = 10,
        n_cpus: int = 10,
    ):
        self.walkers_starting_models = walkers_starting_models
        self.perturbation_funcs = [
            _preprocess_func(func) for func in perturbation_funcs
        ]
        self.log_prior_func = _preprocess_func(log_prior_func)
        self.log_likelihood_func = _preprocess_func(log_likelihood_func)
        self.log_prior_ratio_funcs = (
            [_preprocess_func(func) for func in log_prior_ratio_funcs]
            if log_prior_ratio_funcs is not None
            else None
        )
        self.log_like_ratio_func = _preprocess_func(log_like_ratio_func)
        self.n_chains = n_chains
        self.n_cpus = n_cpus
        self._chains = [
            BaseMarkovChain(
                i,
                walkers_starting_models[i],
                perturbation_funcs,
                self.log_prior_func,
                self.log_likelihood_func,
                self.log_prior_ratio_funcs,
                self.log_like_ratio_func,
            )
            for i in range(n_chains)
        ]

    @property
    def chains(self) -> List[BaseMarkovChain]:
        """The ``MarkovChain`` instances of the current Bayesian inversion
        """
        return self._chains

    def run(
        self,
        sampler: Sampler = None,
        n_iterations: int = 1000,
        burnin_iterations: int = 0,
        save_every: int = 100,
        verbose: bool = True,
        print_every: int = 100,
    ):
        """To run the inversion

        Parameters
        ----------
        sampler : bayesbridge.samplers.Sampler, optional
            a sampler instance describing how chains intereact or modifie their 
            properties during sampling, where it could be 
            :class:`bayesbridge.samplers.VanillaSampler` (default), 
            :class:`bayesbridge.samplers.ParallelTempering` 
            and so on, or a customised sampler instance, by default None
        n_iterations : int, optional
            total number of iterations to run, by default 1000
        burnin_iterations : int, optional
            the iteration number from which we start to save samples, by default 0
        save_every : int, optional
            the frequency in which we save the samples, by default 100
        verbose : bool, optional
            whether to print the progress during sampling or not, by default True
        print_every : int, optional
            the frequency in which we print the progress and information during the 
            sampling, by default 100
        """
        if sampler is None:
            sampler = VanillaSampler()
        sampler.initialize(self.chains)
        self._chains = sampler.run(
            n_iterations=n_iterations,
            n_cpus=self.n_cpus,
            burnin_iterations=burnin_iterations,
            save_every=save_every,
            verbose=verbose,
            print_every=print_every,
        )

    def get_results(self, concatenate_chains=True) -> Union[Dict[str, list], list]:
        """To get the saved models

        Parameters
        ----------
        concatenate_chains : bool, optional
            whether to aggregate samples from all the Markov chains or to keep them
            seperate, by default True

        Returns
        -------
        Union[Dict[str, list], list]
            a dictionary from name of the attribute stored to the values, or a list of
            saved models
        """
        if hasattr(self.chains[0].saved_models, "items"):
            results_model = defaultdict(list)
            for chain in self.chains:
                for key, saved_values in chain.saved_models.items():
                    if concatenate_chains and isinstance(saved_values, list):
                        results_model[key].extend(saved_values)
                    else:
                        results_model[key].append(saved_values)
        else:
            results_model = []
            for chain in self.chains:
                if concatenate_chains:
                    results_model.extend(chain.saved_models)
                else:
                    results_model.append(chain.saved_models)
        return results_model


class BayesianInversion(BaseBayesianInversion):
    """A high-level class for performing Bayesian inversion using Markov Chain Monte
    Carlo (McMC) methods.
    
    This is a subclass of :class:`BaseBayesianInversion`.

    This class provides the basic structure for setting up and running McMC sampling, 
    given user-configured parameterization settings, data targets and corresponding
    forward functions.

    Parameters
    ----------
    parameterization : bayesbridge.Parameterization
        pre-configured parameterization. This includes information about the dimension,
        parameterization bounds and properties of unknown parameterizations
    targets : List[bayesbridge.Target]
        a list of data targets
    fwd_functions : Callable[[bayesbridge.State], np.ndarray]
        a lsit of forward functions corresponding to each data targets provided above.
        Each function takes in a model and produces a numpy array of data predictions.
    n_chains: int, default to 10
        the number of chains in the McMC sampling
    n_cpus: int, default to 10
        the number of CPUs available. This is usually set to be equal to the number of
        chains if there are enough CPUs
    """
    def __init__(
        self,
        parameterization: Parameterization,
        targets: List[Target],
        fwd_functions: Callable[[State], np.ndarray],
        n_chains: int = 10,
        n_cpus: int = 10,
    ):
        self.parameterization = parameterization
        self.targets = targets
        self.fwd_functions = [_preprocess_func(func) for func in fwd_functions]
        self.n_chains = n_chains
        self.n_cpus = n_cpus
        self._chains = [
            MarkovChain(
                i,
                deepcopy(self.parameterization),
                deepcopy(self.targets),
                self.fwd_functions,
            )
            for i in range(n_chains)
        ]


def _preprocess_func(func):
    if func is None:
        return None
    f = None
    args = []
    kwargs = {}
    if isinstance(func, (tuple, list)) and len(func) > 1:
        f = func[0]
        if isinstance(func[1], (tuple, list)):
            args = func[1]
            if len(func) > 2 and isinstance(func[2], dict):
                kwargs = func[2]
        elif isinstance(func[1], dict):
            kwargs = func[1]
    elif isinstance(func, (tuple, list)):
        f = func[0]
    else:
        f = func
    return _FunctionWrapper(f, args, kwargs)


class _FunctionWrapper(object):
    """Function wrapper to make it pickleable (credit to emcee)"""

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args or []
        self.kwargs = kwargs or {}

    def __call__(self, *args):
        return self.f(*args, *self.args, **self.kwargs)
