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
np.random.seed(4)



THICKNESS = np.array([10, 10, 15, 20, 20, 20, 20, 20, 0])
VS = np.array([3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5])
VP_VS = 1.77
VP = VS * VP_VS
RHO = 0.32 * VP + 0.77
PERIODS = np.geomspace(3, 80, 20)
RAYLEIGH_STD = 0.02


true_model = np.array([THICKNESS, VP, VS, RHO])
pd = PhaseDispersion(*true_model)
phase_vel = pd(PERIODS, mode=0, wave="rayleigh").velocity
d_obs = phase_vel + np.random.normal(0, RAYLEIGH_STD, phase_vel.size)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
ax1 = Voronoi1D.plot_depth_profile(THICKNESS, VS, label='Vs', ax=ax1, color='r', lw=2)
ax1 = Voronoi1D.plot_depth_profile(THICKNESS, VP, label='Vp', ax=ax1, color='b', lw=2)
ax1 = Voronoi1D.plot_depth_profile(THICKNESS, RHO, label='Density', ax=ax1, color='green', lw=2)
ax1.set_xlabel('')
ax1.set_ylabel('Depth [km]')
ax1.set_ylim(np.cumsum(THICKNESS)[-1] + max(THICKNESS), 0)
ax1.grid()
ax1.legend(loc='lower right')
plt.show()


def initialize_vs(param, positions=None):
    vmin, vmax = param.get_vmin_vmax(positions)
    sorted_vals = np.sort(np.random.uniform(vmin, vmax, positions.size))
    return sorted_vals

vs = bb.parameters.UniformParameter(name="vs", 
                                    vmin=2.5, 
                                    vmax=5, 
                                    perturb_std=0.15)


vs.set_custom_initialize(initialize_vs)



voronoi = Voronoi1D(
    name="voronoi", 
    vmin=0,
    vmax=150,
    perturb_std=10,
    n_dimensions=None, 
    n_dimensions_min=4,
    n_dimensions_max=15,
    parameters=[vs], 
    birth_from='neighbour'
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

log_likelihood = bb.LogLikelihood(targets=target, fwd_functions=forward_sw)

inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    log_likelihood=log_likelihood,
    n_chains=9, 
    n_cpus=9
)

inversion.run(
    sampler=None, 
    n_iterations=50_000, 
    burnin_iterations=10_000, 
    save_every=500, 
    verbose=False
)
for chain in inversion.chains: chain.print_statistics()
#%%
# saving plots, models and targets
results = inversion.get_results(concatenate_chains=True)
# dpred = np.array(results["dpred"])
interp_depths = np.linspace(0, 160, 160)
# all_thicknesses = [Voronoi1D.compute_cell_extents(m) for m in results["voronoi.discretization"]]

# statistics_vs = bb.discretization.Voronoi1D.get_depth_profiles_statistics(
#     all_thicknesses, results["vs"], interp_depths
#     )


# # plot depths and velocities density profile
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
# bb.discretization.Voronoi1D.plot_depth_profiles_density(
#     all_thicknesses, results["vs"], ax=ax1
# )
# ax1.set_xlabel("Vs")
# bb.discretization.Voronoi1D.plot_interface_hist(
#     all_thicknesses, ax=ax2
# )
# # ax1.set_ylim(interp_depths.max(), interp_depths.min())
# # ax2.set_ylim(interp_depths.max(), interp_depths.min())

# ax1.plot(statistics_vs['median'], interp_depths, 'r')
# Voronoi1D.plot_depth_profile(THICKNESS, VS, color='yellow', lw=2, ax=ax1)
# plt.tight_layout()
# plt.show()


# # ax = Voronoi1D.plot_depth_profiles(
# #     all_thicknesses, results["vs"], linewidth=0.1, color="k", max_depth=150
# # )
# # Voronoi1D.plot_depth_profile(THICKNESS, VS, color='yellow', lw=2, ax=ax)
# # Voronoi1D.plot_depth_profiles_statistics(
# #     all_thicknesses, results["vs"], interp_depths, ax=ax
# # )



#%%

def get_subplot_layout(n_subplots):
    rows = int(np.sqrt(n_subplots))
    cols = int(np.ceil(n_subplots / rows))
    return rows, cols

rows, cols = get_subplot_layout(len(inversion.chains))
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
for ipanel, (ax, chain) in enumerate(zip(np.ravel(axes), inversion.chains)):
    saved_models = chain.saved_models
    saved_thickness = saved_models["voronoi.discretization"]
    saved_vs = saved_models['vs']
    
    Voronoi1D.plot_depth_profiles(
    saved_thickness, saved_vs, ax=ax, linewidth=0.1, color="k", max_depth=150
)
    Voronoi1D.plot_depth_profile(THICKNESS, VS, color='yellow', lw=2, ax=ax)
    Voronoi1D.plot_depth_profiles_statistics(
        saved_thickness, saved_vs, interp_depths, ax=ax
    )
    
    ax.set_title(f'Chain {chain.id}')
    ax.tick_params(direction='in', labelleft=False, labelbottom=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    if not ipanel % cols:
        ax.set_ylabel('Depth [km]')
        ax.tick_params(labelleft=True)
    if ipanel >= (rows-1) * cols:
        ax.set_xlabel('Vs [km/s]')
        ax.tick_params(labelbottom=True)
    
plt.tight_layout()
plt.show()










