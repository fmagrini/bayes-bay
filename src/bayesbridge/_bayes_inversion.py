from typing import List, Callable, Tuple
from numbers import Number
from copy import deepcopy
from collections import defaultdict
import numpy
from ._markov_chain import MarkovChainFromParameterization, MarkovChain
from .samplers import VanillaSampler 


class BayesianInversion:
    def __init__(
        self, 
        walkers_starting_pos: List[numpy.ndarray], 
        perturbations: List[Callable[[numpy.ndarray], Tuple[numpy.ndarray, Number]]], 
        log_posterior_func: Callable[[numpy.ndarray], Number], 
        n_chains: int = 10, 
        n_cpus: int = 10, 
    ):
        self.walkers_starting_pos = walkers_starting_pos
        self.perturbations = [_preprocess_func(func) for func in perturbations]
        self.log_posterior_func = _preprocess_func(log_posterior_func)
        self.n_chains = n_chains
        self.n_cpus = n_cpus
        self._chains = [
            MarkovChain(
                i, 
                walkers_starting_pos[i], 
                perturbations, 
                log_posterior_func, 
            )
            for i in range(n_chains)
        ]
        
    @property
    def chains(self):
        return self._chains

    def run(
        self,
        sampler=VanillaSampler(),
        n_iterations=1000,
        burnin_iterations=0,
        save_every=100,
        verbose=True,
        print_every=100,
    ):
        sampler.initialize(self.chains)
        self._chains = sampler.run(
            n_iterations=n_iterations,
            n_cpus=self.n_cpus,
            burnin_iterations=burnin_iterations,
            save_every=save_every,
            verbose=verbose,
            print_every=print_every, 
        )
        

class BayesianInversionFromParameterization(BayesianInversion):
    def __init__(
        self,
        parameterization,
        targets,
        fwd_functions,
        n_chains=10,
        n_cpus=10,
    ):
        self.parameterization = parameterization
        self.targets = targets
        self.fwd_functions = [_preprocess_func(func) for func in fwd_functions]
        self.n_chains = n_chains
        self.n_cpus = n_cpus
        self._chains = [
            MarkovChainFromParameterization(
                i, 
                deepcopy(self.parameterization),
                deepcopy(self.targets),
                self.fwd_functions,
            )
            for i in range(n_chains)
        ]

    def get_results(self, concatenate_chains=True):
        results_model = defaultdict(list)
        results_targets = {}
        for target_name in self.chains[0].saved_targets:
            results_targets[target_name] = defaultdict(list)
        for chain in self.chains:
            for key, saved_values in chain.saved_models.items():
                if concatenate_chains and isinstance(saved_values, list):
                    results_model[key].extend(saved_values)
                else:
                    results_model[key].append(saved_values)
            for target_name, target in chain.saved_targets.items():
                for key, saved_values in target.items():
                    if concatenate_chains:
                        results_targets[target_name][key].extend(saved_values)
                    else:
                        results_targets[target_name][key].append(saved_values)
        return results_model, results_targets


def _preprocess_func(func):
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

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)
