#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:45:52 2023

@author: fabrizio
"""


import sys
# sys.path.append('/home/fabrizio/Documents/GitHub/bayes-bridge/src')
sys.path.append('/home/jiawen/bayes-bridge/src')
import numpy as np
from bayesbridge.markov_chain import BayesianInversion
from bayesbridge.parameters import UniformParameter, Parameterization1D
from bayesbridge.target import Target
from pysurf96 import surf96


RAYLEIGH_STD = 0.02
LOVE_STD = 0.02
LAYERS_MIN = 3
LAYERS_MAX = 7



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
    


thickness = np.array([15, 20, 20, 0])
vs = np.array([1.5, 3, 2.5, 4])
vp = vs * 1.77
rho = 0.32*vp + 0.77


periods  = np.linspace(5, 80, 20)
rayleigh = surf96(thickness,
                  vp,
                  vs,
                  rho,
                  periods,
                  wave="rayleigh",
                  mode=1,
                  velocity="phase",
                  flat_earth=False)
rayleigh_noisy = rayleigh + np.random.normal(0, RAYLEIGH_STD, rayleigh.size)

love = surf96(thickness,
              vp,
              vs,
              rho,
              periods,
              wave="love",
              mode=1,
              velocity="phase",
              flat_earth=False)

love_noisy = love + np.random.normal(0, LOVE_STD, love.size)

targets = [Target('rayleigh', rayleigh, dobs_covariance_mat=RAYLEIGH_STD),
           Target('love', love, dobs_covariance_mat=LOVE_STD)]

fwd_functions = [forward_rayleigh, forward_love]

free_parameters = ([
    UniformParameter('vs', vmin=1, vmax=5, perturb_std=0.2, position=None)
    ])


parameterization = Parameterization1D(voronoi_site_bounds=(0, 70), 
                                      voronoi_site_perturb_std=3,
                                      n_voronoi_cells=None,
                                      n_voronoi_cells_min=LAYERS_MIN,
                                      n_voronoi_cells_max=LAYERS_MAX,
                                      free_params=free_parameters)



inversion = BayesianInversion(parameterization, 
                              targets, 
                              fwd_functions=fwd_functions,
                              n_cpus=2,
                              n_chains=2)

inversion.run(n_iterations=2500, 
              burnin_iterations=500, 
              save_n_models=10,
              print_every=250)
saved_results = inversion.get_results(True)
