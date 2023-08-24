#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:53:13 2022

@author: fabrizio
"""


import random
from copy import deepcopy
from math import log
import multiprocessing
from functools import partial
import time
import numpy as np
from scipy.interpolate import interp1d
from markov_chain import MarkovChain1D
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass




class BayesianInversion:

    def __init__(self, targets, free_params, forward, **kwargs):
        """
        kwargs :
            Keyword arguments passed to MarkovChain1D
        """
        self.targets = targets
        self.free_params = free_params
        self.forward = forward
        self.verbose = kwargs.pop('verbose', True)
        self.print_all = kwargs.pop('print_all', False)
        self.kwargs = kwargs
        
                  
    def swap_temperatures(self):
        for i in range(len(self.chains)):
            chain1, chain2 = np.random.choice(self.chains, 2, replace=False)
            T1, T2 = chain1.temperature, chain2.temperature
            misfit1, misfit2 = chain1.misfit, chain2.misfit
            prob = (1/T1 - 1/T2) * (misfit1 - misfit2)
            if prob > log(random.random()):
                chain1.temperature = T2
                chain2.temperature = T1


    def run(self, 
            chains=2,
            cpu=2,
            tot_iterations=500_000, 
            burnin_iterations=100_000,
            save_models_per_chain=1000,
            parallel_tempering=False,
            temperature_max=5,
            swap_every=500):
        
        def get_temperatures(chains, parallel_tempering=True, temperature_max=50):
            if parallel_tempering:
                temperatures = np.ones(max(2, int(chains*0.3)) - 1)
                if chains - temperatures.size > 0:
                    size = chains - temperatures.size
                return np.concatenate((temperatures, 
                                       np.geomspace(1, temperature_max, size)))
            return np.ones(chains)
    
        temperatures = get_temperatures(chains, 
                                parallel_tempering=parallel_tempering,
                                temperature_max=temperature_max)
        self.chains = []
        for ichain in range(chains):
            chain = MarkovChain1D(
                 targets=self.targets,
                 free_params=deepcopy(self.free_params),
                 forward=self.forward,
                 tot_iterations=tot_iterations,
                 burnin_iterations=burnin_iterations,
                 save_models=save_models_per_chain,
                 temperature=temperatures[ichain],
                 verbose=self.verbose,
                 print_all=self.print_all,
                 **self.kwargs)
            self.chains.append(chain)
        
        tot_iterations = self.chains[0].tot_iterations
        partial_iterations = swap_every if parallel_tempering else tot_iterations
        iterations = 0        
        t1 = time.time()
        func = partial(MarkovChain1D.advance_chain, 
                       iterations=partial_iterations)
        while True:
        
            if cpu > 1:
                pool = multiprocessing.Pool(cpu)
                self.chains = pool.map(func, self.chains)
                pool.close()
                pool.join()
            else:
                self.chains = [func(chain) for chain in self.chains]
            
            iterations += partial_iterations
            if iterations >= tot_iterations:
                break
            if parallel_tempering:
                self.swap_temperatures()
            
        t2 = time.time()
        if self.verbose:
            print('Inversion completed in %.2f s'%(t2 - t1))
            
        return self.get_results()


    def get_results(self):
        
        def get_no_layers():
            no_layers_max = 0
            for chain in self.chains:
                thickness = chain.store['thickness']
                nlayers = np.sum(np.where(thickness>0, 1, 0), axis=0).max() + 1
                if nlayers > no_layers_max:
                    no_layers_max = nlayers
            return no_layers_max
        
        
        def update_array(store, key, obj):
            func = np.column_stack if key!='misfit' else np.concatenate
            if key in store:
                store[key] = func((store[key], np.float32(obj)))
            else:
                store[key] = np.float32(obj)
                
        store = {}
        no_layers_max = get_no_layers()
        target_names = [target.name for target in self.targets]
        for chain in self.chains:
            for key, value in chain.store.items():
                if key in target_names or key=='misfit':
                    update_array(store, key, value)
                else:
                    update_array(store, key, value[:no_layers_max, :])
        return store
    
    
    @classmethod
    def get_step_model(cls, thickness, param):
        depths_step = []
        param_step = []
        d1 = 0
        for t, p in zip(thickness, param):
            depths_step.append(d1)
            param_step.append(p)
            if t:
                d1 += t
            else:
                d1 += 20
            depths_step.append(d1)
            param_step.append(p)
        return np.column_stack((depths_step, param_step))


    @classmethod
    def interpolate_result(cls, results, depths, param='vs', keep=None):
        if keep is not None and keep<results['misfit'].size:
            idx = np.argsort(results['misfit'])[:keep]
            param = results[param][:, idx]
            thickness = results['thickness'][:, idx]
        else:
            param = results[param]
            thickness = results['thickness']
        models = np.zeros((param.shape[1], depths.size))
        for i in range(param.shape[1]):
            idx = np.flatnonzero(thickness[:,i])
            thickness_i = thickness[idx, i]
            param_i = param[idx, i]
            model = cls.get_step_model(thickness_i, 
                                       param_i)
        
            models[i] = interp1d(*model.T, bounds_error=False)(depths)
        return models    
    
    
    @classmethod
    def plot_results(cls, 
                     results, 
                     depths, 
                     param='vs', 
                     keep=5000, 
                     median=False,
                     percentiles=(25, 75), 
                     kwargs_mean=dict(c='k', ls='-', label='Mean'),
                     kwargs_median=dict(c='r', ls='-', label='Median'),
                     kwargs_std=dict(c='k', ls='--', label='Std'), 
                     kwargs_perc=dict(c='r', ls='--', label='Perc')
                     ):
        interp_param = cls.interpolate_result(results, 
                                              depths, 
                                              param,
                                              keep=keep)
        mean = np.nanmean(interp_param, axis=0)
        std = np.nanstd(interp_param, axis=0)
        plt.plot(mean, depths, **kwargs_mean)
        label_std = kwargs_std.pop('label', None)
        plt.plot(mean + std, depths, **kwargs_std, label=label_std)
        plt.plot(mean - std, depths, **kwargs_std)
        if median:
            plt.plot(np.nanmedian(interp_param, axis=0), depths, **kwargs_median)
            perc = np.nanpercentile(interp_param, percentiles, axis=0)
            label_perc = kwargs_perc.pop('label', None)
            plt.plot(perc[0], depths, **kwargs_perc, label=label_perc)
            plt.plot(perc[1], depths, **kwargs_perc)
        plt.gca().invert_yaxis()
        plt.ylabel('Depth')
        plt.xlabel(param)
        plt.legend()


