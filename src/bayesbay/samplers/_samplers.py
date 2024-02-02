from abc import abstractmethod, ABC
from typing import Callable, List, Tuple
from functools import partial
import multiprocessing
import math
import random
import numpy as np

from .._markov_chain import MarkovChain, BaseMarkovChain


class Sampler(ABC):
    """Low-level class for defining the sampling criterion of a Markov chain
    and/or modifying its attributes in-between iterations
    """
    def __init__(self):
        self._extra_on_initialize = []
        self._extra_on_begin_iteration = []
        self._extra_on_end_iteration = []
        self._extra_on_end_advance_chain = []

    def initialize(self, chains: List[BaseMarkovChain]):  # called by external
        """initializes the given Markov Chains by calling :meth:`on_initialize`
        
        Parameters
        ----------
        chains : List[BaseMarkovChain]
            List of Markov chains used to sample the posterior
        """
        self.on_initialize(chains)
        self._chains = chains
        for func in self._extra_on_initialize:
            func(self, chains)

    def begin_iteration(self, chain: BaseMarkovChain):  # called by external
        """calls :meth:`on_begin_iteration` before beginning the current Markov
        chain iteration
        
        Parameters
        ----------
        chain : bayesbay.BaseMarkovChain
            Markov chain used to sample the posterior
        """
        self.on_begin_iteration(chain)
        for func in self._extra_on_begin_iteration:
            func(self, chain)

    def end_iteration(self, chain: BaseMarkovChain):  # called by external
        """calls :meth:`on_end_iteration` before passing to the next Markov
        chain iteration.
        
        Parameters
        ----------
        chain : bayesbay.BaseMarkovChain
            Markov chain used to sample the posterior
        """
        self.on_end_iteration(chain)
        for func in self._extra_on_end_iteration:
            func(self, chain)

    def end_advance_chain(self):  # called by external
        """calls :meth:`on_end_advance_chain` before concluding the batch of
        Markov chain iterations taking place after calling :meth:`advance_chain`
        """
        self.on_end_advance_chain()
        for func in self._extra_on_end_advance_chain:
            func(self)

    def add_on_initialize(
        self,
        func: Callable[["Sampler", List[MarkovChain]], None],
    ):
        """adds a custom function that will be called at the end of the chains
        initialization (see :meth:`initialize`)
        
        Parameters
        ----------
        func : Callable
            Function that takes as arguments a Sampler and a list of MarkovChains
        """
        self._extra_on_initialize.append(func)

    def add_on_begin_iteration(
        self, func: Callable[["Sampler", BaseMarkovChain], None]
    ):
        """adds a custom function that will be called at the beginning of each 
        Markov chain iteration (see :meth:`on_begin_iteration`)
        """
        self._extra_on_begin_iteration.append(func)

    def add_on_end_iteration(self, func: Callable[["Sampler", BaseMarkovChain], None]):
        """adds a custom function that will be called at the end of each Markov 
        chain iteration (see :meth:`on_end_iteration`)
        """
        self._extra_on_end_iteration.append(func)

    def add_on_end_advance_chain(self, func: Callable[["Sampler"], None]):
        """adds a custom function that will be called at the end of the batch of
        Markov chain iterations taking place after calling :meth:`advance_chain`
        """
        self._extra_on_end_advance_chain.append(func)

    @abstractmethod
    def on_initialize(self, chains):  # customized by subclass
        """defines the behaviour of the sampler at the initialization of the 
        Markov chains
        
        Parameters
        ----------
        chains : List[BaseMarkovChain]
            List of Markov chains used to sample the posterior
        """
        raise NotImplementedError

    @abstractmethod
    def on_begin_iteration(self, chain: BaseMarkovChain):  # customized by subclass
        """defines the behaviour of the sampler at the beginning of each
        Markov chain iteration
        
        Parameters
        ----------
        chain : BaseMarkovChain
            Markov chain used to sample the posterior
        """
        raise NotImplementedError

    @abstractmethod
    def on_end_iteration(self, chain: BaseMarkovChain):  # customized by subclass
        """defines the behaviour of the sampler at the end of each
        Markov chain iteration
        
        Parameters
        ----------
        chain : BaseMarkovChain
            Markov chain used to sample the posterior
        """
        raise NotImplementedError

    @abstractmethod
    def on_end_advance_chain(self):  # customized by subclass
        """defines the behaviour of the sampler at the end of the batch of
        Markov chain iterations taking place after calling :meth:`advance_chain`
        
        Parameters
        ----------
        chain : BaseMarkovChain
            Markov chain used to sample the posterior
        """
        raise NotImplementedError

    @abstractmethod
    def run(self):  # customized by subclass; called by external
        """function that allows for running the Bayesian inference by calling 
        :meth:`advance_chain`. 
        
        .. important::
            
            To work properly, a custom Sampler has to call within this function
            the method :meth:`advance_chain`.
        """
        raise NotImplementedError

    @property
    def chains(self) -> List[BaseMarkovChain]:
        """the ``MarkovChain`` instances of the current Bayesian inference"""
        return self._chains

    def advance_chain(
        self,
        n_iterations,
        n_cpus=10,
        burnin_iterations=0,
        save_every=100,
        verbose=True,
        print_every=100,
    ) -> List[BaseMarkovChain]:
        """advances the Markov chains for a given number of iterations

        Parameters
        ----------
        n_iterations : int
            the number of iterations to advance
        n_cpus : int, optional
            the number of CPUs to use
        burnin_iterations : int, optional
            the iteration number from which we start to save samples, by default 0
        save_every : int, optional
            the frequency in which we save the samples, by default 100
        verbose : bool, optional
            whether to print the progress during sampling or not, by default True
        print_every : int, optional
            the frequency with which we print the progress and information during the
            sampling, by default 100 iterations

        Returns
        -------
        List[BaseMarkovChain]
            the Markov chains
        """
        func = partial(
            MarkovChain.advance_chain,
            n_iterations=n_iterations,
            burnin_iterations=burnin_iterations,
            save_every=save_every,
            verbose=verbose,
            print_every=print_every,
            begin_iteration=self.begin_iteration,
            end_iteration=self.end_iteration,
        )
        if n_cpus > 1:
            pool = multiprocessing.Pool(n_cpus)
            self._chains = pool.map(func, self.chains)
            pool.close()
            pool.join()
        else:
            self._chains = [func(chain) for chain in self.chains]
        self.on_end_advance_chain()
        return self.chains


class VanillaSampler(Sampler):
    """High-level class to be used to sample the posterior by means of 
    reversible-jump Markov chain Monte Carlo.
    """
    def __init__(self):
        super().__init__()

    def on_initialize(self, chains: List[BaseMarkovChain]):
        pass

    def on_begin_iteration(self, chain: BaseMarkovChain):
        pass

    def on_end_iteration(self, chain: BaseMarkovChain):
        pass

    def on_end_advance_chain(self):
        pass

    def run(
        self,
        n_iterations,
        n_cpus=10,
        burnin_iterations=0,
        save_every=100,
        verbose=True,
        print_every=100,
    ) -> List[BaseMarkovChain]:
        return self.advance_chain(
            n_iterations=n_iterations,
            n_cpus=n_cpus,
            burnin_iterations=burnin_iterations,
            save_every=save_every,
            verbose=verbose,
            print_every=print_every,
        )


class ParallelTempering(Sampler):
    r"""High-level class to be used to sample the posterior by means of 
    reversible-jump Markov chain Monte Carlo accelerated with parallel tempering.

    See references below for details on parallel tempering [1]_, [2]_.
    
    Parameters
    ----------
    temperature_max : Number
        the maximum temperature attributed to the chains
    chains_with_unit_temperature : float
        the fraction of chains having unit temperature, 0.4 by default (i.e. 40%
        of the chains)
    swap_every : int
        the frequency with which the chain temperatures are randomly chosen 
        and possibly swapped during the sampling, by default 500 iterations
    
    References
    ----------
    .. [1] Ray et al. 2013, Robust and accelerated Bayesian inversion of marine 
        controlled-source electromagnetic data using parallel tempering
    .. [2] Sambridge 2014, A parallel tempering algorithm for probabilistic 
        sampling and multimodal optimization
    """
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

    def on_initialize(self, chains: List[BaseMarkovChain]):
        n_chains = len(chains)
        temperatures = np.ones(
            max(2, int(n_chains * self._chains_with_unit_tempeature)) - 1
        )
        if n_chains - temperatures.size > 0:
            size = n_chains - temperatures.size
        temperatures = np.concatenate(
            (temperatures, np.geomspace(1, self._temperature_max, size))
        )
        for i, chain in enumerate(chains):
            chain.temperature = temperatures[i]

    def on_begin_iteration(self, chain: BaseMarkovChain):
        pass

    def on_end_iteration(self, chain: BaseMarkovChain):
        pass

    def on_end_advance_chain(self):
        for i in range(len(self.chains)):
            chain1, chain2 = np.random.choice(self.chains, 2, replace=False)
            T1, T2 = chain1.temperature, chain2.temperature
            log_like_ratio = chain1._log_likelihood_ratio(chain2.current_state)
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
    ) -> List[BaseMarkovChain]:
        while True:
            iteration = self.chains[0].statistics["n_proposed_models_total"]
            n_it = min(self._swap_every, n_iterations - iteration)
            burnin_it = max(
                0, min(self._swap_every, burnin_iterations - iteration)
            )
            self.advance_chain(
                n_iterations=n_it,
                n_cpus=n_cpus,
                burnin_iterations=burnin_it,
                save_every=save_every,
                verbose=verbose,
                print_every=print_every,
            )
            if iteration >= n_iterations:
                break
        return self.chains


class SimulatedAnnealing(Sampler):
    r"""High-level class to be used to sample the posterior by means of 
    reversible-jump Markov chain Monte Carlo accelerated with simulated annealing.

    See references below for details on simulated annealing [1]_.
    
    .. note::
        In our implementation, the temperature of each Markov chain is decreased
        exponentially with iteration, from :attr:`temperature_start` to 1, 
        during the burn-in phase. Using ``SimulatedAnnealing`` is therefore 
        incompatible with setting ``burnin_iterations`` to zero in :meth:`run`.
    
    Parameters
    ----------
    temperature_start : Number
        the starting temperature of the Markov chains
        
        
    References
    ----------
    .. [1] Kirkpatrick et al. 1983, Optimization by simulated annealing
    """
    def __init__(self, temperature_start=10):
        super().__init__()
        self.temperature_start = temperature_start

    def on_initialize(self, chains: List[BaseMarkovChain]):
        for chain in chains:
            chain.temperature = self.temperature_start

    def on_begin_iteration(self, chain: BaseMarkovChain):
        iteration = chain.statistics["n_proposed_models_total"]
        if iteration < self.burnin_iterations:
            chain.temperature = self.temperature_start * math.exp(
                -self.cooling_rate * iteration
            )
        elif iteration == self.burnin_iterations:
            chain.temperature = 1

    def on_end_iteration(self, chain: BaseMarkovChain):
        pass

    def on_end_advance_chain(self):
        pass

    def run(
        self,
        n_iterations,
        n_cpus=10,
        burnin_iterations=0,
        save_every=100,
        verbose=True,
        print_every=100,
    ) -> List[BaseMarkovChain]:
        self.burnin_iterations = burnin_iterations
        if burnin_iterations != 0:
            self.cooling_rate = math.log(self.temperature_start) / burnin_iterations
        self.advance_chain(
            n_iterations=n_iterations,
            n_cpus=n_cpus,
            burnin_iterations=burnin_iterations,
            save_every=save_every,
            verbose=verbose,
            print_every=print_every,
        )
        return self.chains
