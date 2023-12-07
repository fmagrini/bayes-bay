from abc import abstractmethod, ABC
from typing import Callable, List, Tuple
from functools import partial
import multiprocessing
import math
import random
import numpy as np

from .._markov_chain import MarkovChain, BaseMarkovChain


class Sampler(ABC):
    def __init__(self):
        self._iteration = 0
        self._extra_on_initialize = []
        self._extra_on_iteration_end = []
        self._extra_on_advance_chain_end = []
        self.on_iteration_end = self.decorate(
            self.on_iteration_end, 
            self._extra_on_iteration_end
        )
        self.on_advance_chain_end = self.decorate(
            self.on_advance_chain_end, 
            self._extra_on_advance_chain_end
        )

    def initialize(self, chains: List[MarkovChain]):  # called by external
        self.on_initialize(chains)
        self._chains = chains
        for func in self._extra_on_initialize:
            func(self, chains)

    def add_on_initialize(
        self,
        func: Callable[["Sampler", List[MarkovChain]], None],
    ):
        self._extra_on_initialize.append(func)
        
    def add_on_iteration_end(self, func: Callable[["MarkovChain"], None]):
        self._extra_on_iteration_end.append(func)

    def add_on_advance_chain_end(self, func: Callable[["Sampler"], None]):
        self._extra_on_advance_chain_end.append(func)

    def decorate(self, func_to_decorate, funcs_to_add):
        def wrapper(*args, **kwargs):
            func_to_decorate(*args, **kwargs)
            for func in funcs_to_add:
                func(self)

        return wrapper

    @abstractmethod
    def on_initialize(self, chains):  # customized depending on individual sampler
        raise NotImplementedError
    
    @abstractmethod
    def on_iteration_end(self):       # customized depending on individual sampler
        raise NotImplementedError

    @abstractmethod
    def on_advance_chain_end(self):  # customized depending on individual sampler
        """
        
        In the object init stage, this method is to be overwritten by a decorated 
        function, in which all the functions in ``self._extra_on_advance_chain_end``
        are to be called immediately after this method.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self):  # customized depending on individual sampler; called by external
        raise NotImplementedError

    @property
    def chains(self) -> List[BaseMarkovChain]:
        return self._chains

    @property
    def iteration(self) -> int:
        return self._iteration

    def advance_chain(
        self,
        n_iterations,
        n_cpus=10,
        burnin_iterations=0,
        save_every=100,
        verbose=True,
        print_every=100,
    ):
        func = partial(
            MarkovChain.advance_chain,
            n_iterations=n_iterations,
            burnin_iterations=burnin_iterations,
            on_iteration_end=self.on_iteration_end,
            save_every=save_every,
            verbose=verbose,
            print_every=print_every,
        )
        if n_cpus > 1:
            pool = multiprocessing.Pool(n_cpus)
            self._chains = pool.map(func, self.chains)
            pool.close()
            pool.join()
        else:
            self._chains = [func(chain) for chain in self.chains]
        self.on_advance_chain_end()
        self._iteration += n_iterations
        return self.chains

    def _assign_temperatures_to_chains(self, temperatures, chains):
        for i, chain in enumerate(chains):
            chain.temperature = temperatures[i]
        self._chains = chains


class VanillaSampler(Sampler):
    def __init__(self):
        super().__init__()

    def on_initialize(self, chains):
        pass
    
    def on_iteration_end(self):
        pass

    def on_advance_chain_end(self):
        pass

    def run(
        self,
        n_iterations,
        n_cpus=10,
        burnin_iterations=0,
        save_every=100,
        verbose=True,
        print_every=100,
    ):
        return self.advance_chain(
            n_iterations=n_iterations,
            n_cpus=n_cpus,
            burnin_iterations=burnin_iterations,
            save_every=save_every,
            verbose=verbose,
            print_every=print_every,
        )


class ParallelTempering(Sampler):
    def __init__(
        self,
        temperature_max=5,
        chains_with_unit_temperature=0.4,
        swap_every=500,
    ):
        super().__init__()
        self._temperature_max = temperature_max
        self._chains_with_unit_tempeature = chains_with_unit_temperature
        self._swap_every = swap_every

    def on_initialize(self, chains):
        n_chains = len(chains)
        temperatures = np.ones(
            max(2, int(n_chains * self._chains_with_unit_tempeature)) - 1
        )
        if n_chains - temperatures.size > 0:
            size = n_chains - temperatures.size
        temperatures = np.concatenate(
            (temperatures, np.geomspace(1, self._temperature_max, size))
        )
        super()._assign_temperatures_to_chains(temperatures, chains)

    def on_iteration_end(self):
        pass

    def on_advance_chain_end(self):
        for i in range(len(self.chains)):
            chain1, chain2 = np.random.choice(self.chains, 2, replace=False)
            T1, T2 = chain1.temperature, chain2.temperature
            log_like_ratio = chain1._log_likelihood_ratio(chain2.current_model)
            prob = (1 / T1 - 1 / T2) * log_like_ratio
            if prob > math.log(random.random()):
                chain1.temperature = T2
                chain2.temperature = T1

    def run(
        self,
        n_iterations,
        n_cpus=10,
        burnin_iterations=0,
        save_every=100,
        verbose=True,
        print_every=100,
    ):
        while True:
            n_it = min(self._swap_every, n_iterations - self.iteration)
            burnin_it = max(
                0, min(self._swap_every, burnin_iterations - self.iteration)
            )
            self.advance_chain(
                n_iterations=n_it,
                n_cpus=n_cpus,
                burnin_iterations=burnin_it,
                save_every=save_every,
                verbose=verbose,
                print_every=print_every,
            )
            if self.iteration >= n_iterations:
                break
        return self.chains


class SimulatedAnnealing(Sampler):
    def __init__(self, temperature_start=10):
        super().__init__()
        self.temperature_start = temperature_start
        raise NotImplementedError

    
    def on_iteration_end(self, markov_chain):
        return self.temperature_start \
            * math.exp(-self.cooling_rate * markov_chain.explored_models)
    
    def run(
        self,
        n_iterations,
        n_cpus=10,
        burnin_iterations=0,
        save_every=100,
        verbose=True,
        print_every=100,
    ):
        self.cooling_rate = math.log(self.temperature_start) / burnin_iterations
        self.advance_chain(
            n_iterations=n_iterations,
            n_cpus=n_cpus,
            burnin_iterations=burnin_iterations,
            save_every=save_every,
            verbose=verbose,
            print_every=print_every
            )
        return self.chains

        
        
        
        
        
        
        
