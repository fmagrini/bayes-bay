#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:29:47 2024

@author: fabrizio
"""


import numpy as np
import matplotlib.pyplot as plt
from bayesbay.discretization import Voronoi1D
from bayesbay.prior import UniformPrior
from bayesbay.parameterization import Parameterization

X_DATA = np.linspace(0, 10, 50)
NOISE_STD = 2


@np.vectorize
def piecewise_function(x):
    if x <= 1:
        return 1
    elif 1 < x <= 2.5:
        return 20
    elif 2.5 < x <= 3:
        return 0
    elif 3 < x <= 4:
        return -3
    elif 4 < x <= 6:
        return -10
    elif 6 < x <= 6.5:
        return -20
    elif 6.5 < x <= 8:
        return 25
    elif 8 < x <= 9:
        return 0
    elif 9 < x <= 10:
        return 10


def fwd_function(state):
    voro = state['voronoi']
    x_nuclei = voro['discretization']
    x_extents = Voronoi1D.compute_cell_extents(x_nuclei)
    x1 = x_extents[0]
    indexes = []
    indexes.extend([0] * np.flatnonzero(X_DATA <= x1).size)
    for i, extent in enumerate(x_extents[1:], 1):
        if extent == 0:
            idx_size = np.flatnonzero(X_DATA > x1).size
        else:
            x2 = x1 + x_extents[i]
            idx_size = np.flatnonzero((X_DATA > x1) & (X_DATA <= x2)).size
            x1 = x2
        indexes.extend([i] * idx_size)
        
    d_pred = voro['y'][indexes]
    return d_pred
    



d_obs = piecewise_function(X_DATA) + np.random.normal(0, NOISE_STD, X_DATA.size)

plt.step(X_DATA, piecewise_function(X_DATA), 'gray', lw=3, label='True model')
plt.plot(X_DATA, d_obs, 'ro', markeredgecolor='k', label='Obs. data')
plt.plot(X_DATA, d_pred, 'bo', markeredgecolor='k', label='Pred. data')
plt.legend()
plt.grid()



y = UniformPrior('y', vmin=-35, vmax=35, perturb_std=3.5)
voronoi = Voronoi1D(
    name="voronoi", 
    vmin=0,
    vmax=10,
    perturb_std=0.75,
    n_dimensions=None, 
    n_dimensions_min=2,
    n_dimensions_max=30,
    parameters=[y], 
    birth_from='prior'
)
parameterization = Parameterization(voronoi)






