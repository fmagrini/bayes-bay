#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:36:27 2024

@author: fabrizio
"""

import bayesbay as bb
from bayesbay.discretization import Voronoi1D
import numpy as np
import matplotlib.pyplot as plt
from disba import PhaseDispersion
np.random.seed(30)



THICKNESS = np.array([10, 10, 15, 20, 20, 20, 20, 20, 0])
VS = np.array([3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5])
VP_VS = 1.77
VP = VS * VP_VS
RHO = 0.32 * VP + 0.77
PERIODS = np.geomspace(3, 80, 20)
RAYLEIGH_STD = 0.02


ax = Voronoi1D.plot_depth_profile(THICKNESS, VS)
ax.set_xlabel('Vs [km/s]')
ax.set_ylabel('Depth [km]')
plt.show()


true_model = np.array([THICKNESS, VP, VS, RHO])
pd = PhaseDispersion(*true_model)
phase_vel = pd(PERIODS, mode=0, wave="rayleigh").velocity
d_obs = phase_vel + np.random.normal(0, RAYLEIGH_STD, phase_vel.size)

plt.plot(PERIODS, phase_vel, 'k', lw=2, label='Predicted data (true model)')
plt.plot(PERIODS, d_obs, 'ro', label='Observed data')





vs = bb.parameters.UniformParameter(name="vs", 
                                    vmin=2.5, 
                                    vmax=5, 
                                    perturb_std=0.15)

voronoi = Voronoi1D(
    name="voronoi", 
    vmin=0,
    vmax=150,
    perturb_std=10,
    n_dimensions=None, 
    n_dimensions_min=4,
    n_dimensions_max=15,
    parameters=[vs], 
)

parameterization = bb.parameterization.Parameterization(voronoi)




def forward_sw(model):
    voronoi = model["voronoi"]
    voronoi_sites = voronoi["discretization"]
    thickness = Voronoi1D.compute_cell_extents(voronoi_sites)
    vs = voronoi["vs"]
    vp = vs * VP_VS
    rho = 0.32 * vp + 0.77
    model = np.array([thickness, vp, vs, rho])
    pd = PhaseDispersion(*model)
    d_pred = pd(PERIODS, mode=0, wave="rayleigh").velocity
    return d_pred



target = bb.Target("rayleigh", 
                   d_obs, 
                   covariance_mat_inv=1/RAYLEIGH_STD**2)


inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    targets=target, 
    fwd_functions=(forward_sw), 
    n_chains=10, 
    n_cpus=10
)

inversion.run(
    sampler=None, 
    n_iterations=325_000, 
    burnin_iterations=75_000, 
    save_every=500, 
    verbose=True, 
    print_every=25_000
)




