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
from bayesbay import Target, LogLikelihood, BayesianInversion
np.random.seed(30)


X_DATA = np.linspace(0, 10, 100)
NOISE_STD = 5


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

plt.figure(figsize=(10, 6))
plt.step(X_DATA, piecewise_function(X_DATA), 'gray', lw=3, label='True model')
plt.plot(X_DATA, d_obs, 'ro', markeredgecolor='k', label='Obs. data')
plt.legend()
plt.grid()
plt.show()

#%%
y = UniformPrior('y', vmin=-35, vmax=35, perturb_std=3.5)
voronoi = Voronoi1D(
    name="voronoi", 
    vmin=0,
    vmax=10,
    perturb_std=0.75,
    n_dimensions=None, 
    n_dimensions_min=2,
    n_dimensions_max=40,
    parameters=[y], 
    birth_from='prior'
)
parameterization = Parameterization(voronoi)


target = Target("d_obs", 
                d_obs, 
                std_min=0, 
                std_max=20, 
                std_perturb_std=2)

log_likelihood = LogLikelihood(targets=target, fwd_functions=fwd_function)

inversion = BayesianInversion(
    parameterization=parameterization, 
    log_likelihood=log_likelihood,
    n_chains=10
)
#%%
inversion.run(
    sampler=None, 
    n_iterations=300_000, 
    burnin_iterations=75_000, 
    save_every=200,
    print_every=10_000,
    verbose=False
)


#%%

results = inversion.get_results(concatenate_chains=True)
sampled_nuclei = results['voronoi.discretization']
sampled_extents = [Voronoi1D.compute_cell_extents(n) for n in sampled_nuclei]
sampled_y = results['voronoi.y']
statistics = Voronoi1D.get_depth_profiles_statistics(
    sampled_extents, sampled_y, X_DATA, percentiles=(10, 90)
    )

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.step(X_DATA, piecewise_function(X_DATA), 'gray', lw=3, label='True model')
ax1.plot(X_DATA, statistics['median'], 'b', label='Inferred (median) model')
ax1.fill_between(X_DATA, 
                *statistics['percentiles'], 
                color='b', 
                alpha=0.2, 
                label='Uncertainty (10th-90th percentiles)')
ax1.plot(X_DATA, d_obs, 'ro', markeredgecolor='k', label='Obs. data')
ax1.legend(framealpha=0)
ax1.grid()

pdf, bins, _ = ax2.hist(results['d_obs.std'], density=True, bins=20, ec='w', label='Posterior')
ax2.axvline(x=NOISE_STD, color='r', lw=3, alpha=1, label='True data noise', zorder=100)
ax2.fill_between([target.std_min, target.std_max], 
                 1 / (target.std_max - target.std_min), 
                 alpha=0.2, 
                 label='Prior')
ax2.set_xlabel('Noise standard deviation')
# ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax2.legend(framealpha=0.9)
plt.tight_layout()
plt.show()



