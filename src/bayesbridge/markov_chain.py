#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:07:01 2022

@author: fabrizio

"""

from collections import defaultdict
from copy import deepcopy
from functools import partial
import multiprocessing
import random
import math
import numpy as np
from .log_likelihood import LogLikelihood
from ._utils_bayes import _get_thickness, _closest_and_final_index
from ._utils_bayes import interpolate_nearest_1d

   

class MarkovChain:
    
    def __init__(self, 
                 parameterization, 
                 targets, 
                 forward_functions,
                 temperature):
        self.parameterization = parameterization
        self.parameterization.initialize()
        self.log_likelihood = LogLikelihood(model=parameterization.model, 
                                            targets=targets, 
                                            forward_functions=forward_functions)
        self._init_perturbation_funcs()
        self._init_statistics()
        self._temperature = temperature
        
        
    @property
    def temperature(self):
        return self._temperature
    
    
    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        
        
    @property
    def saved_models(self):
        r"""models that are saved in current chain; intialized everytime `advance_chain`
        is called
        
        It is a Python dict. See the following example:
        
        .. code-block::
           
           {
               'n_voronoi_cells': 5, 
               'voronoi_sites': array([1.61200696, 3.69444193, 4.25564828, 4.34085936, 9.48688864]), 
               'voronoi_cell_extents': array([2.65322445, 1.32182066, 0.32320872, 2.61562018, 0.        ])
           }
        
        """
        return getattr(self, "_saved_models", None)


    @property
    def saved_targets(self):
        """targets that are saved in current chain; intialized everytime 
        `advance_chain` is called
        """
        return getattr(self, "_saved_targets", None)
    
    
    def _init_perturbation_funcs(self):
        perturb_voronoi = [self.parameterization.perturbation_voronoi_site]
        finalize_voronoi = [self.parameterization.finalize_perturbation]
        perturb_types = ["VoronoiSite"]
        if self.parameterization.trans_d:
            perturb_voronoi += [self.parameterization.perturbation_birth,
                                self.parameterization.perturbation_death]
            finalize_voronoi += [self.parameterization.finalize_perturbation,
                                 self.parameterization.finalize_perturbation]
            perturb_types += ["Birth", "Death"]
        
        perturb_free_params = []
        perturb_free_params_types = []
        finalize_free_params = []
        for name in self.parameterization.free_params:
            perturb_free_params.append(
                partial(self.parameterization.perturbation_free_param, 
                    param_name=name)
            )
            perturb_free_params_types.append("Param - " + name)
            finalize_free_params.append(self.parameterization.finalize_perturbation)

        perturb_targets = []
        perturb_targets_types = []
        finalize_targets = []
        for target in self.log_likelihood.targets:
            if target.is_hierarchical:
                perturb_targets.append(target.perturb_covariance)
                perturb_targets_types.append("Target - " + target.name)
                finalize_targets.append(target.finalize_perturbation)
        
        self.perturbations = perturb_voronoi + perturb_free_params + perturb_targets
        self.perturbation_types = perturb_types + perturb_free_params_types + \
                                    perturb_targets_types
        self.finalizations = finalize_voronoi + finalize_free_params + finalize_targets
        
        assert len(self.perturbations) == len(self.perturbation_types)
        assert len(self.perturbations) == len(self.finalizations)
    
    
    def _init_saved_models(self, save_models):
        nsites = getattr(self.parameterization, 
                         'n_voronoi_cells_max', 
                         len(self.parameterization.voronoi_sites))
        saved_models = {k: np.full((save_models, nsites), np.nan) \
                        for k in self.parameterization.model.current_state \
                            if k != 'n_voronoi_cells'}
        saved_models['n_voronoi_cells'] = \
            nsites if not self.parameterization.trans_d else np.full(save_models, np.nan)
        self._saved_models = saved_models

    
    def _init_saved_targets(self, save_models):
        saved_targets = {}
        for target in self.log_likelihood.targets:
            saved_targets[target.name] = {}
            saved_targets[target.name]['dpred'] = \
                np.full((save_models, target.dobs.size), np.nan)
            if target.is_hierarchical:
                saved_targets[target.name]['sigma'] = np.full(save_models, np.nan)
                if target.noise_is_correlated:
                    saved_targets[target.name]['correlation'] = \
                        np.full(save_models, np.nan)       
        self._saved_targets = saved_targets

    
    def _init_statistics(self):
        self._current_misfit = float("inf")
        self._proposed_counts = defaultdict(int)
        self._accepted_counts = defaultdict(int)
        self._proposed_counts_total = 0
        self._accepted_counts_total = 0
    
    
    def _save_model(self, save_model_idx):
        for key, value in self.parameterization.model.proposed_state.items():
            if key == 'n_voronoi_cells':
                if isinstance(self.saved_models['n_voronoi_cells'], int):
                    continue
                else:
                    self.saved_models['n_voronoi_cells'][save_model_idx] = value
            else:
                print(key, save_model_idx, value.size, self.saved_models[key].shape)
                self.saved_models[key][save_model_idx, :value.size] = value
    
    
    def _save_target(self, save_model_idx):
        for target in self.log_likelihood.targets:
            self.saved_targets[target.name]['dpred'][save_model_idx] = \
                self.log_likelihood.proposed_dpred[target.name]
            if target.is_hierarchical:
                self.saved_targets[target.name]['sigma'][save_model_idx] = \
                    target._proposed_state['sigma']
                if target.noise_is_correlated:
                    self.saved_targets[target.name]['correlation'][save_model_idx] = \
                        target._proposed_state['correlation']
                        

    def _save_statistics(self, perturb_i, accepted):
        perturb_type = self.perturbation_types[perturb_i]
        self._proposed_counts[perturb_type] += 1
        self._accepted_counts[perturb_type] += 1 if accepted else 0
        self._proposed_counts_total += 1
        self._accepted_counts_total += 1 if accepted else 0
    
    
    def _print_statistics(self):
        head = 'EXPLORED MODELS: %s - ' % self._proposed_counts_total
        acceptance_rate = self._accepted_counts_total / self._proposed_counts_total
        head += 'ACCEPTANCE RATE: %d/%d (%.2f %%)' % \
            (self._accepted_counts_total, self._proposed_counts_total, acceptance_rate)
        print(head)
        print('PARTIAL ACCEPTANCE RATES:')
        for perturb_type in self._proposed_counts:
            proposed = self._proposed_counts[perturb_type]
            accepted = self._accepted_counts[perturb_type]
            acceptance_rate = accepted / proposed * 100
            print('\t%s: %d/%d (%.2f%%)' % \
                  (perturb_type, accepted, proposed, acceptance_rate))
        print('CURRENT MISFIT: %.2f' % self._current_misfit)

    
    def _next_iteration(self, save_model):
        # choose one perturbation function and type
        perturb_i = random.randint(0, len(self.perturbations) - 1)
        
        # propose new model and calculate probability ratios
        log_prob_ratio = self.perturbations[perturb_i]()

        log_likelihood_ratio, misfit = \
            self.log_likelihood(self._current_misfit, self.temperature)
        
        # decide whether to accept
        accepted = log_prob_ratio + log_likelihood_ratio > math.log(random.random())
        
        if save_model is not None and self.temperature == 1:
            self._save_model(save_model)
            self._save_target(save_model)
        
        # finalize perturbation based whether it's accepted
        self.finalizations[perturb_i](accepted)
        self._current_misfit = misfit if accepted else self._current_misfit

        # save statistics
        self._save_statistics(perturb_i, accepted)
           
    
    def advance_chain(self, 
                      n_iterations=1000, 
                      burnin_iterations=0, 
                      save_n_models=2, 
                      verbose=True):
        self._init_saved_models(save_n_models)
        self._init_saved_targets(save_n_models)
        
        save_every = (n_iterations-burnin_iterations) // save_n_models + 1
        for i in range(1, n_iterations + 1):
            if i <= burnin_iterations:
                save_model = None
            else:
                idiff = i - burnin_iterations
                save_model = None if idiff % save_every else idiff // save_every
            self._next_iteration(save_model)
            
        if verbose:
            self._print_statistics()
        return self



class BayesianInversion:
    
    def __init__(self, 
                 parameterization, 
                 targets, 
                 fwd_functions, 
                 n_chains=10, 
                 n_cpus=10,
                 ):
        self.parameterization = parameterization
        self.targets = targets
        self.fwd_functions = fwd_functions
        self.n_chains = n_chains
        self.n_cpus = n_cpus
        self.chains = [
            MarkovChain(deepcopy(self.parameterization),
                        deepcopy(self.targets), 
                        self.fwd_functions,
                        temperature=1
                        ) for _ in range(n_chains)
            ]


    def _init_temperatures(self, 
                           parallel_tempering=False, 
                           temperature_max=5,
                           chains_with_unit_temperature=0.4):
        if parallel_tempering:
            temperatures = np.ones(
                max(2, int(self.n_chains*chains_with_unit_temperature)) - 1
                )
            if self.n_chains - temperatures.size > 0:
                size = self.n_chains - temperatures.size
            return np.concatenate((temperatures, 
                                   np.geomspace(1, temperature_max, size)))
        return np.ones(self.n_chains)

        
    def run(self, 
            n_iterations=1000, 
            burnin_iterations=0, 
            save_n_models=100,
            parallel_tempering=False,
            temperature_max=5,
            chains_with_unit_temperature=0.4,
            swap_every=500, 
            verbose=True):
        temperatures = self._init_temperatures(parallel_tempering,
                                               temperature_max,
                                               chains_with_unit_temperature)
        
        for i, chain in enumerate(self.chains):
            chain.temperature = temperatures[i]
        
        partial_iterations = swap_every if parallel_tempering else n_iterations
        func = partial(MarkovChain.advance_chain,
                       n_iterations=partial_iterations,
                       burnin_iterations=burnin_iterations,
                       save_n_models=save_n_models, 
                       verbose=verbose)
        i_iterations = 0

        while True:
            if self.n_cpus > 1:
                pool = multiprocessing.Pool(self.n_cpus)
                self.chains = pool.map(func, self.chains)
                pool.close()
                pool.join()
            else:
                self.chains = [func(chain) for chain in self.chains]
            
            i_iterations += partial_iterations
            if i_iterations >= n_iterations:
                break
            if parallel_tempering:
                self.swap_temperatures()
            burnin_iterations = max(0, burnin_iterations - partial_iterations)
            func = partial(MarkovChain.advance_chain,
                           n_iterations=partial_iterations,
                           burnin_iterations=burnin_iterations,
                           save_n_models=save_n_models, 
                           verbose=verbose)
            #TODO: save saved_models and saved_targets before next iteration
            #TODO RETURN SOMETHING


    def swap_temperatures(self):
        for i in range(len(self.chains)):
            chain1, chain2 = np.random.choice(self.chains, 2, replace=False)
            T1, T2 = chain1.temperature, chain2.temperature
            misfit1, misfit2 = chain1._current_misfit, chain2._current_misfit
            prob = (1/T1 - 1/T2) * (misfit1 - misfit2)
            if prob > math.log(random.random()):
                chain1.temperature = T2
                chain2.temperature = T1
