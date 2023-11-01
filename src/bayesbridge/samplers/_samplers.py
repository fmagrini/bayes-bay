from functools import partial
import multiprocessing
import math
import random
import numpy as np
from .._markov_chain import MarkovChain


class Sampler:
    def __init__(self):
        pass
    
    def init_temperatures(self, chains):
        raise NotImplementedError
    
    def run(self):
        raise NotImplementedError
    
    @property
    def chains(self):
        return self._chains

    def _assign_temperatures_to_chains(self, temperatures, chains):
        for i, chain in enumerate(chains):
            chain.temperature = temperatures[i]
        self._chains = chains


class VanillaSampler(Sampler):
    def __init__(self):
        pass

    def init_temperatures(self, chains):
        temperatures = np.ones(len(chains))
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
        return self._chains
        

class ParallelTempering(Sampler):
    def __init__(
        self,
        temperature_max=5,
        chains_with_unit_temperature=0.4,
        swap_every=500,
    ):
        self._temperature_max = temperature_max
        self._chains_with_unit_tempeature = chains_with_unit_temperature
        self._swap_every = swap_every

    def init_temperatures(self, chains):
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
        func = partial(
            MarkovChain.advance_chain,
            n_iterations=self._swap_every,
            burnin_iterations=burnin_iterations,
            save_every=save_every,
            verbose=verbose,
            print_every=print_every,
        )
        i_iterations = 0
        while True:
            if n_cpus > 1:
                pool = multiprocessing.Pool(n_cpus)
                self._chains = pool.map(func, self.chains)
                pool.close()
                pool.join()
            else:
                self._chains = [func(chain) for chain in self.chains]
            i_iterations += self._swap_every
            if i_iterations >= n_iterations:
                break
            self.swap_temperatures()
            burnin_iterations = max(0, burnin_iterations - self._swap_every)
            func = partial(
                MarkovChain.advance_chain,
                n_iterations=self._swap_every,
                burnin_iterations=burnin_iterations,
                save_every=save_every,
                verbose=verbose,
                print_every=print_every,
            )
        return self._chains
    
    def swap_temperatures(self):
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
        raise NotImplementedError
