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
from log_likelihood import LogLikelihood
from _utils_bayes import _get_thickness, _closest_and_final_index
from _utils_bayes import interpolate_nearest_1d

   

class MarkovChain:
    
    def __init__(self, 
                 parameterization, 
                 targets, 
                 forward_functions,
                 temperature):
        self.parameterization = parameterization
        self.log_likelihood = LogLikelihood(model=parameterization.model, 
                                            targets=targets, 
                                            forward_functions=forward_functions)
        self._init_perturbations()
        self._init_statistics()
        self._temperature = temperature
        
        
    @property
    def temperature(self):
        return self._temperature
    
    
    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    
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
        self.saved_models = saved_models

    
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
        self.saved_targets = saved_targets

    
    def _init_statistics(self):
        self._current_misfit = float("inf")
        self._proposed_counts = defaultdict(int)
        self._accepted_counts = defaultdict(int)
        self._proposed_counts_total = 0
        self._accepted_counts_total = 0
    
    
    def _save_model(self, save_model_idx):
        for key, value in self.paramterization.model.proposed_state.items():
            if key == 'n_voronoi_cells':
                if isinstance(self.saved_models['n_voronoi_cells'], int):
                    continue
                else:
                    self.saved_models['n_voronoi_cells'][save_model_idx] = value
            else:
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
        perturb_i = random.randint(len(self.perturbations) - 1)
        
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
        
        save_every = (n_iterations-burnin_iterations) // save_n_models
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


    # def advance_chain(self, 
    #                   n_iterations=1000, 
    #                   burnin_iterations=0, 
    #                   save_n_models=2, 
    #                   verbose=True):
    #     self._init_saved_models(save_n_models)
    #     self._init_saved_targets(save_n_models)
                        
    #     save_every = (n_iterations - burnin_iterations) // save_n_models
    #     for i in range(burnin_iterations + 1, n_iterations + 1):
    #         save_model = None \
    #             if i % save_every or i < burnin_iterations \
    #                 else i // save_every
    #         self._next_iteration(save_model)
            
    #     if verbose:
    #         self._print_statistics()
    #     return self
    
    
class BayesianInversion:
    
    def __init__(self, 
                 parameterization, 
                 targets, 
                 fwd_functions, 
                 n_chains=10, 
                 n_cpus=10):
        self.parameterization = parameterization
        self.targets = targets
        self.fwd_functions = fwd_functions
        self.n_chains = n_chains
        self.n_cpus = n_cpus
        self.chains = [
            MarkovChain(deepcopy(self.parameterization, 
                                 self.targets, 
                                 self.fwd_functions)
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


    def swap_temperatures(self):
        for i in range(len(self.chains)):
            chain1, chain2 = np.random.choice(self.chains, 2, replace=False)
            T1, T2 = chain1.temperature, chain2.temperature
            misfit1, misfit2 = chain1._current_misfit, chain2._current_misfit
            prob = (1/T1 - 1/T2) * (misfit1 - misfit2)
            if prob > math.log(random.random()):
                chain1.temperature = T2
                chain2.temperature = T1



# class MarkovChain1D:
    
#     def __init__(self, 
#                  targets,
#                  free_params,
#                  forward,
#                  tot_iterations=500_000,
#                  burnin_iterations=100_000,
#                  temperature=1,
#                  additional_store=['vp', 'vs', 'rho', 'melt'],
#                  save_predictions=True,
#                  save_models=1000,
#                  thickness_min=1,
#                  depth_min=0.5,
#                  depth_max=60, 
#                  std_depth=0.3,
#                  no_layers_min=4,
#                  no_layers_max=20, 
#                  trans_d=True,
#                  n_layers=None,
#                  print_percentage=20,
#                  verbose=True,
#                  print_all=True):
        
#         self.targets = targets
#         self.free_params = {p.name: p for p in free_params}
#         self.forward = forward
#         self.tot_iterations = tot_iterations
#         self.burnin_iterations = burnin_iterations
#         self.temperature = temperature
#         self.save_predictions = save_predictions
#         self.save_models = save_models
#         self.save_step = self.get_save_step(tot_iterations, 
#                                             burnin_iterations, 
#                                             save_models)
#         self.print_all = print_all
#         self.verbose = verbose
#         self.depth_min = depth_min
#         self.depth_max = depth_max
#         self.std_depth = std_depth
#         self.trans_d = trans_d
#         if self.trans_d:
#             self.no_layers_max = no_layers_max
#             self.no_layers_min = no_layers_min
#         else:
#             self.n_layers = n_layers
#             self.no_layers_min = self.n_layers
#             self.no_layers_max = self.n_layers
            
#         self.thickness_min = thickness_min
        
#         self.perturbation_func = [self.perturb_depth]
#         self.perturbation_func += [p.random_perturbation \
#                                    for p in self.free_params.values()]
#         if self.trans_d:
#             self.perturbation_func += [self.add_layer, self.remove_layer]
#         self.random_changes = len(self.perturbation_func)
#         self.initialize_counters_and_booleans()
#         self.initialize_params()
#         self.store = self.initialize_store(additional_store)
#         self.birth_prob, self.death_prob = self.birth_and_death_prob()
#         self.n_percent = int(tot_iterations * print_percentage/100)    


#     def __str__(self):
#         string = 'EXPLORED MODELS: %s - '%self.explored_models
#         acceptance_rate = self.accepted_models / self.explored_models * 100
#         string += 'ACCEPTANCE RATE: %.2f %%'%acceptance_rate
#         if self.print_all:
#             string += '\nPARTIAL ACCEPTANCE RATES:\n'
#             for name, (perturbations, accepted) in self.stats.items():
#                 if perturbations:
#                     value = '%.2f %%'%(accepted/perturbations * 100)
#                 else:
#                     value = 'N.A.'
#                 string += '\t%s: %s\n'%(name, value)
#             string += 'MISFIT: %.2f'%self.misfit 
#         return string
              
#     def reset_partial_stats(self):
#         for v in self.stats.values():
#             v[:] = [0, 0]
        
        
#     def get_save_step(self, tot_iterations, burnin_iterations, save_models):
#         after_burnin_it = tot_iterations-burnin_iterations
#         step = ceil((after_burnin_it) / save_models)
#         additional_it = (step * save_models) - after_burnin_it
#         self.tot_iterations += additional_it
#         return step


#     def initialize_counters_and_booleans(self):
#         self.explored_models = 0
#         self.accepted_models = 0
#         self.saved_models = 0
#         self.accepted = False
#         self.is_burnin = True
#         if self.print_all and self.verbose:
#             self.stats = {}
#             for key in ['Birth', 'Death', 'Depth'] + list(self.free_params):
#                 self.stats[key] = [0, 0]
            
    
#     def initialize_params(self):
#         self.misfit = np.inf
#         if self.trans_d:
#             self.n_layers = random.randint(
#                     self.no_layers_min, 
#                     min(self.no_layers_min+2, self.no_layers_max)
#                                       )
#         self.depth = np.zeros((self.no_layers_max+1, 2))
#         depths = np.linspace(self.depth_min + self.thickness_min,
#                               self.depth_max - self.thickness_min,
#                               self.n_layers)
#         # depths = np.sort(np.random.choice(
#         #     np.linspace(self.depth_min + self.thickness_min*2,
#         #                 self.depth_max - self.thickness_min*2),
#         #     self.n_layers,
#         #     replace=False
#         #     ))
#         self.depth[:self.n_layers, 0] = depths
#         for param in self.free_params.values():
#             param.value = np.zeros((self.no_layers_max+1, 2))
#             param.value[:self.n_layers, 0] = param.starting_model(depths)
            
        
#     def initialize_store(self, store):
#         store = store if store is not None else []
#         store_dict = {}
#         store_dict['thickness'] = np.zeros(
#                 (self.no_layers_max+1, self.save_models)
#                 )
#         store_dict['misfit'] = np.zeros(self.save_models)
#         for name, param in self.free_params.items():
#             store_dict[name] = np.zeros((self.no_layers_max+1, self.save_models))
#         for name in store:
#             store_dict[name] = np.zeros((self.no_layers_max+1, self.save_models))
#         if self.save_predictions:
#             for target in self.targets:
#                 store_dict[target.name] = np.zeros(
#                         (target.size, self.save_models)
#                         )
#         return store_dict
                 
    
#     def update_store(self, thickness, misfit, store_params, store_predictions):
#         nlayers = thickness.size
#         self.store['thickness'][:nlayers, self.saved_models] = thickness
#         self.store['misfit'][self.saved_models] = misfit  
#         for name, param in self.free_params.items():
#             self.store[name][:nlayers, self.saved_models] = param[:nlayers, 1]
#         for name, param in store_params.items():
#             self.store[name][:nlayers, self.saved_models] = param
#         if self.save_predictions:
#             for name, prediction in store_predictions.items():
#                 self.store[name][:, self.saved_models] = prediction
        

#     def birth_and_death_prob(self):
#         birth_prob = 1
#         for param in self.free_params.values():
#             birth_prob *= param.birth_probability
#         return np.log(birth_prob), -np.log(birth_prob)

    
#     def clone_arrays(self, keep_column=0):
#         change_column = 1 if keep_column==0 else 0
#         self.depth[:, change_column] = self.depth[:, keep_column]
#         for param in self.free_params.values():
#             param.value[:, change_column] = param.value[:, keep_column]
        
        
#     def random_perturbation(self):
#         """        
#         Returns
#         -------
#         (prob, n_layers_new) :
#             Probability of random perturbation and new number of layers
#         """
#         # change_max = 4 if self.explored_models/self.tot_iterations<0.01 else 6
#         while True:
#             random_change = random.randint(0, self.random_changes-1)
#             if self.trans_d:
#                 if self.n_layers==self.no_layers_max and random_change==self.random_changes-2:
#                     continue # Don't create a new layer!
#                 if self.n_layers==self.no_layers_min and random_change==self.random_changes-1:
#                     continue # Don't remove a layer!
#             break        
        
#         return self.perturbation_func[random_change](self.n_layers)
          
                    
#     def perturb_depth(self, n_layers):
#         layer = random.randint(0, n_layers-1)
#         while True:
#             random_deviate = random.normalvariate(0, self.std_depth)
#             new_depth = self.depth[layer, 0] + random_deviate
#             if new_depth<self.depth_min or new_depth>self.depth_max: 
#                 continue
#             depth_diff = self.depth[:, 1] - new_depth
#             if np.any(np.abs(depth_diff) < self.thickness_min*0.1):
#                 continue
#             self.depth[layer, 1] = new_depth
#             isort = np.argsort(self.depth[:n_layers, 1])
#             self.depth[:n_layers, 1] = self.depth[:n_layers, 1][isort]
#             for param in self.free_params.values():
#                 param.value[:n_layers, 1] = param.value[:n_layers, 1][isort]
#             return 0, n_layers, 'Depth'
           
    
#     def add_layer(self, n_layers):
                
#         depth_old = self.depth[:n_layers, 0]
#         while True:
#             random_depth = random.random() * self.depth_max
#             depth_diff = depth_old - random_depth
#             if np.any(np.abs(depth_diff) < self.thickness_min*0.1):
#                 continue
#             break  
#         self.depth[n_layers, 1] = random_depth
#         self.fill_new_layer(n_layers)
#         iold, inew = _closest_and_final_index(depth_old, random_depth)
#         prob = 0
#         for param in self.free_params.values():
#             param.random_perturbation(n_layers, layer=inew)
#             prob += param.prob_change_dimension(param[iold, 0], param[inew, 1])
#         return self.birth_prob + prob, n_layers + 1, 'Birth'
        
    
#     def remove_layer(self, n_layers):
#         layer = random.randint(0, n_layers-1)
#         depth_old = self.depth[layer, 0]
#         for i in range(layer, n_layers):
#             self.depth[i, 1] = self.depth[i+1, 1]
#             for param in self.free_params.values():
#                 param[i, 1] = param[i+1, 1]
                
#         inew = np.argmin(np.abs(self.depth[:n_layers-1, 1] - depth_old))
#         prob = 0
#         for param in self.free_params.values():
#             prob += param.prob_change_dimension(param[layer, 0], param[inew, 1])
#         return self.death_prob - prob, n_layers - 1, 'Death'
    
    
#     def fill_new_layer(self, n_layers):
#         old_depths = self.depth[:n_layers, 0]
#         new_depths = np.sort(self.depth[:n_layers+1, 1])
#         self.depth[:n_layers+1, 1] = new_depths
#         for param in self.free_params.values():
#             vnew = interpolate_nearest_1d(xp=new_depths, 
#                                           x=old_depths, 
#                                           y=param.value[:n_layers, 0])
#             param[:n_layers+1, 1] = vnew
                    
    
#     @classmethod
#     def get_thickness(cls, depth):
#         return _get_thickness(depth)
           
    
#     def next_iteration(self, save_model=False):
#         while 1:
#             if not self.accepted:
#                 self.clone_arrays(keep_column=0)
#             prob, n_layers_new, perturb_kind = self.random_perturbation()
#             depth = self.depth[:n_layers_new, 1]
#             thickness = self.get_thickness(depth)
#             try:
#                 free_params = {k: v[:n_layers_new, 1] for k, v in self.free_params.items()}
#                 misfit_proposed, store_params, store_predictions = self.forward(
#                         targets=self.targets, 
#                         free_params=free_params, 
#                         depth=depth,
#                         thickness=thickness
#                         )
#             except KeyError as key:
#                 raise Exception('%s not found in free_params.'%key)
#             except:
#                 self.accepted = False
#                 continue
#             break
        
#         if save_model:
#             self.update_store(thickness=thickness, 
#                               misfit=misfit_proposed, 
#                               store_params=store_params, 
#                               store_predictions=store_predictions)
#             self.saved_models += 1
            
#         accept_prob = prob - (misfit_proposed-self.misfit) / (2*self.temperature)
#         self.accepted = accept_prob > log(random.random())
#         if self.accepted:
#             self.clone_arrays(keep_column=1)
#             self.n_layers = n_layers_new
#             self.misfit = misfit_proposed
#             self.accepted_models += 1
            
#         self.explored_models += 1
#         if self.print_all:
#             self.stats[perturb_kind][0] += 1
#             if self.accepted:
#                 self.stats[perturb_kind][1] += 1
    
        
#     def advance_chain(self, iterations=500):
    
#         def is_to_be_saved(explored_models, save_step, is_burnin):
#             if not is_burnin and explored_models % save_step == 0:
#                 return True
#             return False
        
#         def print_stats(explored_models, n_percent):
#             if not explored_models % n_percent:
#                 print(self)
#                 self.reset_partial_stats()
                
#         for iteration in range(iterations):
#             if self.explored_models >= self.burnin_iterations and self.is_burnin:
#                 self.is_burnin = False
#             save_model = is_to_be_saved(self.explored_models, 
#                                         self.save_step, 
#                                         self.is_burnin)
#             self.next_iteration(save_model=save_model)
#             if self.verbose and self.explored_models:
#                 print_stats(self.explored_models, self.n_percent)
                
#         return self


#%%
#
#
#
#
#
#if __name__ == '__main__':
    
#    from scipy.interpolate import interp1d
#    from pysurf96 import surf96
#    from functools import partial
#    import os
#    import matplotlib.pyplot as plt
#    from seislib.utils import load_pickle
#    import scipy.io
#    os.chdir('/home/fabrizio/PhD/bayesian_inversion/petro_inversion')
#    import seispetro
#    SRC = '/media/fabrizio/Seagate Expansion Drive/toba_extended'
#    PHASE_DIAG = 'MAGEMin_PhaseDiagram_v2_matv7.mat'
#    
#    
#    def get_phase_diag_dict(path):
#        file_dict = scipy.io.loadmat(path)
#        phase_diags = {}
#        phase_diags['dx'] = float(file_dict['Delta'][0])
#        phase_diags['dy'] = float(file_dict['Delta'][1])
#        phase_diags['dz'] = float(file_dict['Delta'][2])
#        del file_dict['Delta']
#        phase_diags['xmin'] = float(file_dict['Bound_min'][0])
#        phase_diags['ymin'] = float(file_dict['Bound_min'][1])
#        phase_diags['zmin'] = float(file_dict['Bound_min'][2])
#        del file_dict['Bound_min']
#        phase_diags['Nx'] = int(file_dict['N'][0])
#        phase_diags['Ny'] = int(file_dict['N'][1])
#        phase_diags['Nz'] = int(file_dict['N'][2])
#        del file_dict['N']
#        for k, v in file_dict.items():
#            if isinstance(v, np.ndarray):
#                phase_diags[k] = np.ascontiguousarray(v)
#        return phase_diags
#    
#    
#    def forward_seismic(targets, thickness, vs, vp, rho, save_model=False):
#        store_predictions = {}
#        misfit = 0
#        for dispcurve in targets:
#            synth_vel = surf96(thickness,
#                               vp, 
#                               vs, 
#                               rho,
#                               dispcurve.period,
#                               wave=dispcurve.name, 
#                               mode=1, 
#                               velocity=dispcurve.phase, 
#                               flat_earth=False)
#            store_predictions[dispcurve.name] = synth_vel
#            squared_diff = (dispcurve.velocity - synth_vel)**2
#            misfit += np.sum(squared_diff / dispcurve.std**2)  
#            
#        return misfit, store_predictions
#    
#    
#    def forward(targets, free_params, depth, thickness):
#        vp, vs, rho, melt = forward_petro(depth=depth, 
#                                          T=free_params['T'], 
#                                          d=free_params['d'], 
#                                          phi=free_params['phi'], 
#                                          alpha=free_params['alpha'])
#        
#        misfit, store_dispcurves = forward_seismic(targets=targets,
#                                                   thickness=thickness,
#                                                   vs=vs,
#                                                   vp=vp,
#                                                   rho=rho)
#        store_params = {'vs': vs,
#                        'vp': vp,
#                        'rho': rho,
#                        'melt': melt}
#        return misfit, store_params, store_dispcurves
#    
#    
#    phase_diags = get_phase_diag_dict(PHASE_DIAG)
#    forward_petro = partial(seispetro.forward_petro, phase_diags=phase_diags)
#    
#    rayleigh_dict = load_pickle(os.path.join(SRC, 'inversion_data_Z.pickle'))
#    love_dict = load_pickle(os.path.join(SRC, 'inversion_data_T.pickle'))
#    targets = [
#        DispersionCurve(rayleigh_dict['period'], 
#                        rayleigh_dict['vel'][:, 50],
#                        std=0.02,
#                        name='rayleigh',
#                        phase='phase'),
#        DispersionCurve(love_dict['period'], 
#                        love_dict['vel'][:, 50],
#                        std=0.025,
#                        name='love',
#                        phase='phase')
#        ]
#    T = UniformParameter(name='T', 
#                         vmin=100, 
#                         vmax=1100, 
#                         std_perturb=20, 
#                         starting_model=lambda z: 10 * z + 100)
#    d = UniformParameter(name='d', 
#                         vmin=0, 
#                         vmax=1, 
#                         std_perturb=0.05,
#                         starting_model=lambda z: interp1d([0, 50], [0.95, 0.05])(z))
#    phi = UniformParameter(name='phi', 
#                           vmin=0, 
#                           vmax=0.3, 
#                           std_perturb=0.02,
#                           starting_model=lambda z: 0.3 * 0.9 * np.exp(-z/2)
#                           )
#    alpha = UniformParameter(name='alpha', 
#                             vmin=0.15, 
#                             vmax=0.75, 
#                             std_perturb=0.05,
#                             starting_model=lambda z: 0.5
#                             )
#    chain = MarkovChain1D(
#                 targets=targets,
#                 free_params=[T, d, phi, alpha],
#                 forward=forward,
#                 tot_iterations=200000,
#                 burnin_iterations=50000,
#                 additional_store=['vs', 'vp', 'melt', 'rho'],
#                 save_predictions=True,
#                 save_models=1000,
#                 thickness_min=1,
#                 depth_min=0.5,
#                 depth_max=50, 
#                 std_depth=0.3,
#                 no_layers_min=4,
#                 no_layers_max=20, 
#                 print_percentage=5,
#                 verbose=True,
#                 print_all=True)
#    chain.advance_chain(chain.tot_iterations)



