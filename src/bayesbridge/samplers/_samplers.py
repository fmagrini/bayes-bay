from abc import abstractmethod
from functools import partial
import multiprocessing
import math
import random
import numpy as np
from .._markov_chain import MarkovChain


class Sampler:
    def __init__(self):
        self._i = 0
    
    def initialize(self, chains):           # called by external
        self.on_initialize(chains)
        self._chains = chains
    
    @abstractmethod
    def on_initialize(self, chains):        # customized depending on individual sampler
        raise NotImplementedError
    
    @abstractmethod
    def on_advance_chain_end(self, chains): # customized depending on individual sampler
        raise NotImplementedError
    
    @abstractmethod
    def run(self):     # customized depending on individual sampler; called by external
        raise NotImplementedError
    
    @property
    def chains(self):
        return self._chains
    
    @property
    def i(self):
        return self._i
    
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
        self._i += n_iterations
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
            n_it = min(self._swap_every, n_iterations - self.i)
            burnin_it = max(0, min(self._swap_every, burnin_iterations - self.i))
            self.advance_chain(
                n_iterations=n_it,
                n_cpus=n_cpus, 
                burnin_iterations=burnin_it,
                save_every=save_every,
                verbose=verbose,
                print_every=print_every,
            )
            if self.i >= n_iterations:
                break
        return self.chains
    
    def on_advance_chain_end(self):
        for i in range(len(self.chains)):
            chain1, chain2 = np.random.choice(self.chains, 2, replace=False)
            T1, T2 = chain1.temperature, chain2.temperature
            misfit1, misfit2 = chain1._current_misfit, chain2._current_misfit
            prob = (1 / T1 - 1 / T2) * (misfit1 - misfit2)
            if prob > math.log(random.random()):
                chain1.temperature = T2
                chain2.temperature = T1


class SimulatedAnnealing(Sampler):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
