#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:45:52 2023

@author: fabrizio

Vs model from Fu et al. 2016, https://doi.org/10.1002/2016JB013305
"""


import sys
sys.path.append('/home/fabrizio/Documents/GitHub/bayes-bridge/src')
# sys.path.append('/home/jiawen/bayes-bridge/src')
import numpy as np
import matplotlib.pyplot as plt
from bayesbridge.markov_chain import BayesianInversion
from bayesbridge.parameters import UniformParameter, Parameterization1D
from bayesbridge.target import Target
from pysurf96 import surf96
from espresso import ReceiverFunctionInversion
rf_module = ReceiverFunctionInversion().rf


RAYLEIGH_STD = 0.02
LOVE_STD = 0.02
RF_STD = 0.03
LAYERS_MIN = 3
LAYERS_MAX = 15



def forward_rayleigh(proposed_state):
    thickness = proposed_state['voronoi_cell_extents']
    vs = proposed_state['vs']
    vp = vs * 1.77
    rho = 0.32*vp + 0.77
    
    return surf96(
        thickness,
        vp,
        vs,
        rho,
        periods,
        wave="rayleigh",
        mode=1,
        velocity="phase",
        flat_earth=False
        )  
    
    
def forward_love(proposed_state):
    thickness = proposed_state['voronoi_cell_extents']
    vs = proposed_state['vs']
    vp = vs * 1.77
    rho = 0.32*vp + 0.77
    
    return surf96(
        thickness,
        vp,
        vs,
        rho,
        periods,
        wave="love",
        mode=1,
        velocity="phase",
        flat_earth=False
        )  
    

def forward_rf(proposed_state):
    vs = proposed_state['vs']
    thickness = proposed_state['voronoi_cell_extents']
    depths = np.cumsum(thickness)
    depths[-1] += 20
    ratio = np.full(vs.size, 1.77)
    # print(depths, vs)
    model = np.column_stack((depths, vs, ratio))
    _, data = rf_module.rfcalc(model)
    return data

# thickness = np.array([15, 20, 20, 0])
# vs = np.array([1.5, 3, 2.5, 4])
thickness = np.array([10, 10, 15, 20, 20, 20, 20, 20, 0])
vs = np.array([3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5])
vp = vs * 1.77
rho = 0.32*vp + 0.77

true_model = {'voronoi_cell_extents': thickness, 'vs': vs}

periods = np.linspace(4, 80, 20)

rayleigh = forward_rayleigh(true_model)
rayleigh_noisy = rayleigh + np.random.normal(0, RAYLEIGH_STD, rayleigh.size)

love = forward_love(true_model)
love_noisy = love + np.random.normal(0, LOVE_STD, love.size)

rf = forward_rf(true_model)
# rf_noisy = rf + np.random.normal(0, RF_STD, rf.size)


plt.plot(periods, rayleigh, 'r--')
plt.plot(periods, love, 'b--')

fig, ax = plt.subplots(1, 1, figsize=(8, 10))
Parameterization1D.plot_param_samples([thickness],  [vs], alpha=1, ax=ax)

targets = [Target('rayleigh', rayleigh_noisy, covariance_mat_inv=1/RAYLEIGH_STD**2),
           Target('love', love_noisy, covariance_mat_inv=1/LOVE_STD**2),
           Target('rf', rf, covariance_mat_inv=1/RF_STD**2)]

fwd_functions = [forward_rayleigh, forward_love, forward_rf]

#%%

# free_parameters = ([
#     UniformParameter('vs', 
#                      vmin=2.7, 
#                      vmax=4.8, 
#                      perturb_std=0.15, 
#                      position=None,
#                      init_sorted=True)
#     ])
free_parameters = ([
    UniformParameter('vs', 
                     vmin=[2.7, 3.2, 3.75], 
                     vmax=[4, 4.75, 5], 
                     perturb_std=0.15, 
                     position=[0, 40, 80],
                     init_sorted=True)
    ])


parameterization = Parameterization1D(voronoi_site_bounds=(0, 130), 
                                      voronoi_site_perturb_std=8,
                                      n_voronoi_cells=None,
                                      n_voronoi_cells_min=LAYERS_MIN,
                                      n_voronoi_cells_max=LAYERS_MAX,
                                      free_params=free_parameters)



inversion = BayesianInversion(parameterization, 
                              targets, 
                              fwd_functions=fwd_functions,
                              n_cpus=10,
                              n_chains=10)

inversion.run(n_iterations=750_000, 
              burnin_iterations=300_000, 
              save_every=500,
              print_every=5_000)

saved_models, saved_targets = inversion.get_results(concatenate_chains=True)
interp_depths = np.arange(130, dtype=float)
statistics = Parameterization1D.get_ensemble_statistics(
                    saved_models['voronoi_cell_extents'], 
                    saved_models['vs'], 
                    interp_depths)

ax = Parameterization1D.plot_param_samples(saved_models['voronoi_cell_extents'], 
                                           saved_models['vs'], 
                                           linewidth=0.1,
                                           color='k')
Parameterization1D.plot_param_samples([thickness], 
                                      [vs], 
                                      alpha=1, 
                                      ax=ax, 
                                      color='r',
                                      label='True')

mean = statistics['mean']
std  = statistics['std']
percentiles = statistics['percentiles']
ax.plot(mean, interp_depths, color='b', label='Mean')
ax.plot(mean-std, interp_depths, 'b--')
ax.plot(mean+std, interp_depths, 'b--')
ax.plot(statistics['median'], interp_depths, color='orange', label='Median')
ax.plot(percentiles[0], interp_depths, color='orange', ls='--', )
ax.plot(percentiles[1], interp_depths, color='orange', ls='--', )
ax.legend()




# plt.plot(periods, np.mean(r, axis=0), 'r')
# plt.plot(periods, np.mean(l, axis=0), 'b')




