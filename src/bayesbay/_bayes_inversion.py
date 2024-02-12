from typing import List, Callable, Tuple, Any, Dict, Union
from numbers import Number
from collections import defaultdict
from pprint import pformat
import warnings

from ._markov_chain import MarkovChain, BaseMarkovChain
from .samplers import VanillaSampler, Sampler
from .parameterization import Parameterization
from ._utils import _preprocess_func
from ._log_likelihood import LogLikelihood
from ._target import Target
from .perturbations._birth_death import BirthPerturbation, DeathPerturbation


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

    where :math:`{\bf d}_{obs}` denotes the observed data and
    :math:`\mathbf{J}` the Jacobian of the transformation.

    Parameters
    ----------
    walkers_starting_states: List[Any]
        a list of starting states for each chain. The states can be of any type so long
        as they are consistent with what is accepted as arguments in the perturbation
        functions and probability functions. The length of this list must be equal to
        the number of chains, i.e. ``n_chains``
    perturbation_funcs: List[Callable[[Any], Tuple[Any, Number]]]
        a list of perturbation functions. Each perturbation function should take
        in a model :math:`\mathbf{m}` (any type is allowed, as long as it is
        consistent with ``walkers_starting_states`` and the below functions) and
        perturb it to produce the new model :math:`\bf m'`. Each perturbation function
        should return :math:`\bf m'` along with :math:`\log(
        \frac{p({\bf m'})}{p({\bf m})}
        \frac{q\left({\bf m}
        \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
        \lvert \mathbf{J} \rvert)`, which is used in the calculation of
        the acceptance probability.
    perturbation_weights: List[Number], optional
        a list of weights corresponding to each element of ``perturbation_funcs``. If
        this is set to (the default value) ``None``, then each perturbation function
        will have equal probability of being selected on each iteration.
    log_like_ratio_func: Union[LogLikelihood, Callable[[Any, Any], Number]], optional
        the log likelihood ratio function :math:`\log(\frac{p(\mathbf{d}_{obs} \mid 
        \mathbf{m'})} {p(\mathbf{d}_{obs} \mid \mathbf{m})})`. It takes the current and 
        proposed models, :math:`\mathbf{m}` and :math:`\mathbf{m'}`, whose type should 
        be consistent with the other arguments of this class, and returns a scalar 
        corresponding to the log likelihood ratio. This is utilised in the calculation 
        of the acceptance probability. If None, ``log_like_func`` gets used 
        instead. Default to None
    log_like_func: Callable[[Any], Number], optional
        the log likelihood function :math:`\log(p(\mathbf{d}_{obs} \mid \mathbf{m}))`.
        It takes in a model :math:`\mathbf{m}` (any type is allowed, as long as it is
        consistent with the other arguments of this class) and returns the log
        of the likelihood function. This function is only used when ``log_like_ratio_func``
        is None. Default to None
    n_chains: int, optional
        the number of chains in the McMC sampling. Default is 10
    n_cpus: int, optional
        the number of CPUs available. If None (default) this is usually set to 
        the number of chains (``n_chains``)
    """

    def __init__(
        self,
        walkers_starting_states: List[Any],
        perturbation_funcs: List[Callable[[Any], Tuple[Any, Number]]],
        perturbation_weights: List[Number] = None, 
        log_like_ratio_func: Union[LogLikelihood, Callable[[Any, Any], Number]] = None,
        log_like_func: Callable[[Any], Number] = None,
        n_chains: int = 10,
        n_cpus: int = None,
        save_dpred: bool = True,
    ):
        assert len(walkers_starting_states) == n_chains, (
            "`walkers_starting_states` doesn't match the number of chains: "
            f"{len(walkers_starting_states)} != {n_chains}"
        )
        self.walkers_starting_states = walkers_starting_states
        self.set_perturbation_funcs(perturbation_funcs, perturbation_weights)
        if isinstance(log_like_ratio_func, LogLikelihood):
            self.log_likelihood = log_like_ratio_func
        else:
            self.log_likelihood = LogLikelihood(
                log_like_ratio_func=log_like_ratio_func, 
                log_like_func=log_like_func, 
            ) 
        self.n_chains = n_chains
        self.n_cpus = n_cpus if n_cpus is not None else n_chains
        self.save_dpred = save_dpred
        self._chains = [
            BaseMarkovChain(
                id=i,
                starting_state=self.walkers_starting_states[i],
                perturbation_funcs=self.perturbation_funcs,
                perturbation_weights=self.perturbation_weights, 
                log_likelihood=self.log_likelihood, 
                save_dpred=self.save_dpred,
            )
            for i in range(n_chains)
        ]
        
        self._init_repr_args()
    
    def set_perturbation_funcs(
        self, 
        perturbation_funcs: List[Callable], 
        perturbation_weights: List[Number] = None, 
    ):
        # preprocess functions (if necessary)
        perturbation_funcs = [_preprocess_func(func) for func in perturbation_funcs]
        # pad weights if it's None
        if perturbation_weights is None:
            perturbation_weights = [1] * len(perturbation_funcs)
        # check lengths
        assert len(perturbation_funcs) == len(perturbation_weights),  (
                "`perturbation_funcs` should have the same length of "
                f"`perturbation_weights`: {len(perturbation_funcs)} != "
                f"{len(perturbation_weights)}"
            )
        # validate weights of birth-death pairs to be the same
        birth_death_pairs = defaultdict(list)
        for ifunc, func in enumerate(perturbation_funcs):
            if isinstance(func, (BirthPerturbation, DeathPerturbation)):
                birth_death_pairs[func.param_space_name].append(ifunc)
        for ps_name, perturb_ifuncs in birth_death_pairs.items():
            birth_func_indices = [i for i in perturb_ifuncs if \
                isinstance(perturbation_funcs[i], BirthPerturbation)]
            death_func_indices = [i for i in perturb_ifuncs if \
                isinstance(perturbation_funcs[i], DeathPerturbation)]
            if len(birth_func_indices) != 1:
                raise ValueError(
                    "there should be exactly one birth perturbation function for each "
                    f"trans-dimensional parameter space, but {ps_name} has "
                    f"{len(birth_func_indices)} birth perturbation function instead"
                )
            if len(death_func_indices) != 1:
                raise ValueError(
                    "there should be exactly one death perturbation function for each "
                    f"trans-dimensional parameter space, but {ps_name} has "
                    f"{len(death_func_indices)} death perturbation function instead"
                )
            birth_func_idx = birth_func_indices[0]
            death_func_idx = death_func_indices[0]
            birth_func_weight = perturbation_weights[birth_func_idx]
            death_func_weight = perturbation_weights[death_func_idx]
            birth_func = perturbation_funcs[birth_func_idx]
            death_func = perturbation_funcs[death_func_idx]
            if birth_func_weight != death_func_weight:
                warnings.warn(
                    f"weights for {birth_func} and {death_func} are different: "
                    f"{birth_func_weight} != {death_func_weight}. We will default the death "
                    "perturbation weight to be the same as birth weight"
                )
                perturbation_weights[death_func_idx] = birth_func_weight
        # assign to self attributes
        self.perturbation_funcs = perturbation_funcs
        self.perturbation_weights = perturbation_weights
        if hasattr(self, "_chains"):
            for chain in self.chains:
                chain.set_perturbation_funcs(
                    self.perturbation_funcs, self.perturbation_weights
                )
    
    @property
    def chains(self) -> List[BaseMarkovChain]:
        """The ``MarkovChain`` instances of the current Bayesian inversion"""
        return self._chains
    
    @chains.setter
    def chains(self, updated_chains: List[BaseMarkovChain]):
        """setters of chains attached to this BayesianInversion instance

        Parameters
        ----------
        updated_chains : List[BaseMarkovChain]
            a list of chains to be set to current inversion

        Raises
        ------
        TypeError
            when ``updated_chains` is not a list or the elements are not instances of
            :class:`BaseMarkovChain`
        """
        if not isinstance(updated_chains, list) or \
            all([isinstance(c, BaseMarkovChain) for c in updated_chains]):
                raise TypeError("make sure the `updated_chains` is a list of chains")
        self._chains = updated_chains
        self.n_chains = len(updated_chains)

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
            the frequency with which we save the samples. By default a state is
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
        """To get the saved states from current inversion

        Parameters
        ----------
        keys : Union[str, List[str]]
            key(s) to retrieve from the saved states. This will be ignored when states 
            are not of type :class:`State` or dict
        concatenate_chains : bool, optional
            whether to aggregate samples from all the Markov chains or to keep them
            seperate, by default True

        Returns
        -------
        Union[Dict[str, list], list]
            a dictionary from name of the attribute stored to the values, or a list of
            saved states (if the base level API is used, and states are not of type 
            :class:`State` or dict)
        """
        return self.get_results_from_chains(self.chains, keys, concatenate_chains)

    @staticmethod
    def get_results_from_chains(
        chains: Union[BaseMarkovChain, List[BaseMarkovChain]], 
        keys: Union[str, List[str]] = None,
        concatenate_chains: bool = True,
    ) -> Union[Dict[str, list], list]:
        """To get the saved states from a list of given Markov chains

        Parameters
        ----------
        chains : Union[BaseMarkovChain, List[BaseMarkovChain]]
            Markov chain(s) that the results are going to be extracted from
        keys : Union[str, List[str]]
            key(s) to retrieve from the saved states. This will be ignored when states 
            are not of type :class:`State` or dict
        concatenate_chains : bool, optional
            whether to aggregate samples from all the Markov chains or to keep them
            seperate, by default True

        Returns
        -------
        Union[Dict[str, list], list]
            a dictionary from name of the attribute stored to the values, or a list of
            saved states (if the base level API is used, and states are not of type 
            :class:`State` or dict)
        """
        if not isinstance(chains, list):
            chains = [chains]
        if not all([isinstance(c, BaseMarkovChain) for c in chains]):
            raise TypeError(
                "`chains` should be a list of Markov chains (i.e. instances of "
                "BaseMarkovChain or MarkovChain)"
            )
        if isinstance(keys, str):
            keys = [keys]
        if hasattr(chains[0].saved_states, "items"):
            results = defaultdict(list)
            for chain in chains:
                for key, saved_values in chain.saved_states.items():
                    if keys is None or key in keys:
                        if concatenate_chains and isinstance(saved_values, list):
                            results[key].extend(saved_values)
                        else:
                            results[key].append(saved_values)
        else:
            results = []
            for chain in chains:
                if concatenate_chains:
                    results.extend(chain.saved_states)
                else:
                    results.append(chain.saved_states)
        return dict(results)
    
    def _init_repr_args(self):
        self._repr_args = {
            "walkers_starting_states": self.walkers_starting_states, 
            "perturbation_funcs": self.perturbation_funcs, 
            "log_likelihood": self.log_likelihood, 
            "n_chains": self.n_chains, 
            "n_cpus": self.n_cpus, 
            "save_dpred": self.save_dpred, 
            "chains": self.chains, 
        }
    
    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}("
        for k, v in self._repr_args.items():
            if hasattr(v, '__class__') and v.__class__.__repr__ is object.__repr__:
                repr_v = v.__class__.__name__
            elif k == "walkers_starting_states":
                try:
                    repr_v = f"[{len(v)} arrays with shapes {', '.join(str(arr.shape) for arr in v)}]"
                except AttributeError:
                    repr_v = f"[{len(v)} {type(v[0])} objects]"
            elif k == "chains" and len(v) > 3: 
                repr_v = f"{str(v[:3])[:-1]}, ...{len(v)} chains in total...]"
            else:
                repr_v = repr(v)
            string += f"{k}={repr_v}, "
        return f"{string[:-2]})"
    
    def __str__(self) -> str:
        repr_args_copy = self._repr_args.copy() 
        _n_chains = ""
        if 'chains' in repr_args_copy and len(repr_args_copy['chains']) > 3:
            _n_chains = f"...{len(repr_args_copy['chains'])} chains in total..."
            repr_args_copy['chains'] = repr_args_copy['chains'][:3] + [_n_chains]
        string = f"{self.__class__.__name__}("
        string += pformat(repr_args_copy)[1:-1] + ")"
        string = string.replace(repr(_n_chains), _n_chains)     # remove the quotes
        return string
    

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
        Each function takes in a state and returns a numpy array of data predictions.
    n_chains: int, 10 by default
        the number of chains in the McMC sampling
    n_cpus: int, 10 by default
        the number of CPUs available. If None (default) this is usually set to 
        the number of chains (``n_chains``)
    """

    def __init__(
        self,
        parameterization: Parameterization,
        log_likelihood: LogLikelihood, 
        n_chains: int = 10,
        n_cpus: int = None,
        save_dpred: bool = True,
    ):
        if not isinstance(log_likelihood, LogLikelihood):
            raise TypeError(
                "``log_likelihood`` should be an instance of "
                "``bayesbay.LogLikelihood``"
            )

        self.parameterization = parameterization
        self.log_likelihood = log_likelihood
        self.n_chains = n_chains
        self.n_cpus = n_cpus if n_cpus is not None else n_chains
        self.save_dpred = save_dpred
        self.perturbation_funcs, self.perturbation_weights = \
            self._init_perturbation_funcs()
        self._chains = [
            MarkovChain(
                id=i,
                parameterization=self.parameterization,
                log_likelihood=self.log_likelihood, 
                perturbation_funcs=self.perturbation_funcs, 
                perturbation_weights=self.perturbation_weights, 
                saved_dpred=self.save_dpred,
            )
            for i in range(n_chains)
        ]
        self.log_likelihood.add_targets_observer(self)
        self._init_repr_args()
    
    def _init_repr_args(self) -> dict:
        self._repr_args = {
            "parameterization": self.parameterization, 
            "log_likelihood": self.log_likelihood, 
            "n_chains": self.n_chains, 
            "n_cpus": self.n_cpus, 
            "save_dpred": self.save_dpred, 
            "chains": self.chains, 
        }
        _parameterization = dict()
        for ps_name, ps in self.parameterization.parameter_spaces.items():
            _parameterization[ps_name] = {
                k: v for k, v in ps._repr_args.items() \
                    if k not in {"parameters", "name"}
            }
            _parameterization[ps_name]["parameters"] = list(ps.parameters.values())
        self._repr_args["parameterization"] = _parameterization

    def _init_perturbation_funcs(self) -> list:
        funcs_from_parameterization = self.parameterization.perturbation_functions
        funcs_from_log_likelihood = self.log_likelihood.perturbation_functions
        weights_from_parameterization = self.parameterization.perturbation_weights
        weights_from_log_likelihood = self.log_likelihood.perturbation_weights
        return funcs_from_parameterization + funcs_from_log_likelihood, \
            weights_from_parameterization + weights_from_log_likelihood

    def update_log_likelihood_targets(self, targets: List[Target]):
        """function to be called by ``self.log_likelihood`` when there are more 
        target(s) added to the inversion. This method updates the perturbation function 
        list for the current inversion and its chains, and initializes the 
        ``current_state`` for each chain when there are hierarchical targets added.
        
        This method is called because we register ``self`` (i.e. the current 
        ``BayesianInversion``) as an observer to its log likelihood instance.

        Parameters
        ----------
        targets : List[Target]
            the list of targets added to ``self.log_likelihood``
        """
        self.perturbation_funcs, self.perturbation_weights = \
            self._init_perturbation_funcs()
        for chain in self.chains:
            chain.set_perturbation_funcs(
                self.perturbation_funcs, self.perturbation_weights
            )
            chain.update_targets(targets)
