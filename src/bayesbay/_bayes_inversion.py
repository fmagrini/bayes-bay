from typing import List, Callable, Tuple, Any, Dict, Union
from numbers import Number
from copy import deepcopy
from collections import defaultdict
import numpy as np

from ._markov_chain import MarkovChain, BaseMarkovChain
from .samplers import VanillaSampler, Sampler
from .parameterization import Parameterization
from ._target import Target
from ._state import State
from ._utils import _preprocess_func, _FunctionWrapper, _LogLikeRatioFromFunc


class BaseBayesianInversion:
    r"""
    A low-level class for Bayesian sampling based on Markov chain Monte Carlo (McMC).

    This class provides the basic structure for setting up and running McMC sampling,
    given user-provided definition of perturbations, likelihood functions and the
    initialization of walkers. At each iteration of the inference process,
    the current model :math:`\bf m` is perturbed to produce :math:`\bf m'`, and
    the new model is accepted with probability
    
    .. math::
        
        \alpha({\bf m' \mid m}) = \mbox{min} \Bigg[1, 
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}} 
            \underbrace{\frac{p\left({\bf d}_{obs} \mid {\bf m'}\right)}{p\left({\bf d}_{obs} \mid {\bf m}\right)}}_{\text{Likelihood ratio}} 
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}  
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}}
        \Bigg],
                        
    where :math:`p({\bf d}_{obs})` denotes the observed data and 
    :math:`\mathbf{J}` the Jacobian of the transformation.                                                                                                                                                                                                                                                                                                                                                   
    
    Parameters
    ----------
    walkers_starting_models: List[Any]
        a list of starting models for each chain. The models can be of any type so long
        as they are consistent with what is accepted as arguments in the perturbation
        functions and probability functions. The length of this list must be equal to
        the number of chains, i.e. ``n_chains``
    perturbation_funcs: List[Callable[[Any], Tuple[Any, Number]]]
        a list of perturbation functions. Each perturbation function should take 
        in a model :math:`\mathbf{m}` (any type is allowed, as long as it is 
        consistent with ``walkers_starting_models`` and the below functions) and 
        perturb it to produce the new model :math:`\bf m'`. Each perturbation function 
        should return :math:`\bf m'` along with :math:`\log(
        \frac{p({\bf m'})}{p({\bf m})}
        \frac{q\left({\bf m} 
        \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
        \lvert \mathbf{J} \rvert)`, which is used in the calculation of 
        the acceptance probability.
    log_likelihood_func: Callable[[Any], Number], optional
        the log likelihood function :math:`\log(p(\mathbf{d}_{obs} \mid \mathbf{m}))`. 
        It takes in a model :math:`\mathbf{m}` (any type is allowed, as long as it is 
        consistent with the other arguments of this class) and returns the log 
        of the likelihood function. This function is only used when ``log_like_ratio_func`` 
        is None. Default is None
    log_like_ratio_func: Callable[[Any, Any], Number], optional
        the log likelihood ratio function :math:`\log(\frac{p(\mathbf{d}_{obs} \mid \mathbf{m'})}
        {p(\mathbf{d}_{obs} \mid \mathbf{m})})`. It takes the current and proposed models,
        :math:`\mathbf{m}` and :math:`\mathbf{m'}`, whose type should be consistent 
        with the other arguments of this class, and returns a scalar corresponding to
        the log likelihood ratio. This is utilised in the calculation of the
        acceptance probability. If None, ``log_likelihood_func`` gets used instead. Default is None
    n_chains: int, optional
        the number of chains in the McMC sampling. Default is 10
    n_cpus: int, optional
        the number of CPUs available. This is usually set to be equal to the number of
        chains if there are enough CPUs. Default is 10
    """

    def __init__(
        self,
        walkers_starting_models: List[Any],
        perturbation_funcs: List[Callable[[Any], Tuple[Any, Number]]],
        log_likelihood_func: Callable[[Any], Number] = None,
        log_like_ratio_func: Callable[[Any, Any], Number] = None,
        n_chains: int = 10,
        n_cpus: int = 10,
    ):
        self.walkers_starting_models = walkers_starting_models
        self.perturbation_funcs = [
            _preprocess_func(func) for func in perturbation_funcs
        ]
        if log_like_ratio_func is None:
            if log_likelihood_func is None:
                raise ValueError(
                    "at least one of `log_like_ratio_func` and `log_likelihood_func` needs"
                    "to be provided"
                )
            self.log_like_ratio_func = _LogLikeRatioFromFunc(log_likelihood_func)
        else:
            self.log_like_ratio_func = _preprocess_func(log_like_ratio_func)
        self.n_chains = n_chains
        self.n_cpus = n_cpus
        self._chains = [
            BaseMarkovChain(
                i,
                self.walkers_starting_models[i],
                self.perturbation_funcs,
                self.log_like_ratio_func,
            )
            for i in range(n_chains)
        ]

    @property
    def chains(self) -> List[BaseMarkovChain]:
        """The ``MarkovChain`` instances of the current Bayesian inversion"""
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
        r"""To run the inversion

        Parameters
        ----------
        sampler : bayesbay.samplers.Sampler, optional
            a sampler instance describing how chains intereact or modify their
            properties during sampling. This could be a sampler from the 
            module :mod:`bayesbay.samplers` such as
            :class:`bayesbay.samplers.VanillaSampler` (default),
            :class:`bayesbay.samplers.ParallelTempering`, or
            :class:`bayesbay.samplers.SimulatedAnnealing`,
            or a customised sampler instance.
        n_iterations : int, optional
            total number of iterations to run, by default 1000
        burnin_iterations : int, optional
            the iteration number from which we start to save samples, by default 0
        save_every : int, optional
            the frequency with which we save the samples. By default a model is
            saved every after 100 iterations after the burn-in phase
        verbose : bool, optional
            whether to print the progress during sampling or not, by default True
        print_every : int, optional
            the frequency with which we print the progress and information during the
            sampling, by default 100 iterations
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

    def get_results(
        self,
        keys: Union[str, List[str]] = None,
        concatenate_chains: bool = True,
    ) -> Union[Dict[str, list], list]:
        """To get the saved models

        Parameters
        ----------
        keys : Union[str, List[str]]
            one or more keys to retrieve from the saved models. This will be ignored when
            models are not of type :class:`State` or dict
        concatenate_chains : bool, optional
            whether to aggregate samples from all the Markov chains or to keep them
            seperate, by default True

        Returns
        -------
        Union[Dict[str, list], list]
            a dictionary from name of the attribute stored to the values, or a list of
            saved models
        """
        if isinstance(keys, str):
            keys = [keys]
        if hasattr(self.chains[0].saved_models, "items"):
            results_model = defaultdict(list)
            for chain in self.chains:
                for key, saved_values in chain.saved_models.items():
                    if keys is None or key in keys:
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
    parameterization : bayesbay.parameterization.Parameterization
        pre-configured parameterization. This includes information about the dimension,
        parameterization bounds and properties of unknown parameterizations
    targets : List[bayesbay.Target]
        a list of data targets
    fwd_functions : Callable[[bayesbay.State], np.ndarray]
        a lsit of forward functions corresponding to each data targets provided above.
        Each function takes in a model and returns a numpy array of data predictions.
    n_chains: int, 10 by default
        the number of chains in the McMC sampling
    n_cpus: int, 10 by default
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
                self.parameterization,
                self.targets,
                self.fwd_functions,
            )
            for i in range(n_chains)
        ]
