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


inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    targets=target, 
    fwd_functions=(forward_sw), 
    n_chains=9, 
    n_cpus=9
)

inversion.run(
    sampler=None, 
    n_iterations=300_000, 
    burnin_iterations=0, 
    save_every=1, 
    verbose=False, 
    print_every=10_000
)
for chain in inversion.chains: chain._print_statistics()
#%%
# saving plots, models and targets
results = inversion.get_results(concatenate_chains=True)
dpred = np.array(results["dpred"])
interp_depths = np.linspace(0, 160, 160)
all_thicknesses = [Voronoi1D.compute_cell_extents(m) for m in results["voronoi.discretization"]]

statistics_vs = bb.discretization.Voronoi1D.get_depth_profiles_statistics(
    all_thicknesses, results["vs"], interp_depths
    )


# plot depths and velocities density profile
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
bb.discretization.Voronoi1D.plot_depth_profiles_density(
    all_thicknesses, results["vs"], ax=ax1
)
ax1.set_xlabel("Vs")
bb.discretization.Voronoi1D.plot_interface_hist(
    all_thicknesses, ax=ax2
)
# ax1.set_ylim(interp_depths.max(), interp_depths.min())
# ax2.set_ylim(interp_depths.max(), interp_depths.min())

ax1.plot(statistics_vs['median'], interp_depths, 'r')
Voronoi1D.plot_depth_profile(THICKNESS, VS, color='yellow', lw=2, ax=ax1)
plt.tight_layout()
plt.show()


# ax = Voronoi1D.plot_depth_profiles(
#     all_thicknesses, results["vs"], linewidth=0.1, color="k", max_depth=150
# )
# Voronoi1D.plot_depth_profile(THICKNESS, VS, color='yellow', lw=2, ax=ax)
# Voronoi1D.plot_depth_profiles_statistics(
#     all_thicknesses, results["vs"], interp_depths, ax=ax
# )



#%%
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, chain in zip(np.ravel(axes), inversion.chains):
    saved_models = chain.saved_models
    saved_thickness = saved_models["voronoi.discretization"]
    saved_vs = saved_models['vs']
    # statistics_vs = bb.discretization.Voronoi1D.get_depth_profiles_statistics(
    #     saved_thickness, saved_vs, interp_depths
    #     )
    
    ax.set_title(f'Chain {chain.id}')
    Voronoi1D.plot_depth_profiles(
    saved_thickness, saved_vs, ax=ax, linewidth=0.1, color="k", max_depth=150
)
    Voronoi1D.plot_depth_profile(THICKNESS, VS, color='yellow', lw=2, ax=ax)
    Voronoi1D.plot_depth_profiles_statistics(
        saved_thickness, saved_vs, interp_depths, ax=ax
    )
plt.tight_layout()
plt.savefig('/home/fabrizio/Downloads/samples_per_chains.png', dpi=200)
plt.show()


# def get_results(
#     keys=None,
#     concatenate_chains=True,
# ):
#     """To get the saved models

#     Parameters
#     ----------
#     keys : Union[str, List[str]]
#         one or more keys to retrieve from the saved models. This will be ignored when
#         models are not of type :class:`State` or dict
#     concatenate_chains : bool, optional
#         whether to aggregate samples from all the Markov chains or to keep them
#         seperate, by default True

#     Returns
#     -------
#     Union[Dict[str, list], list]
#         a dictionary from name of the attribute stored to the values, or a list of
#         saved models
#     """
#     if isinstance(keys, str):
#         keys = [keys]
#     if hasattr(chains[0].saved_models, "items"):
#         results_model = defaultdict(list)
#         for chain in chains:
#             for key, saved_values in chain.saved_models.items():
#                 if keys is None or key in keys:
#                     if concatenate_chains and isinstance(saved_values, list):
#                         results_model[key].extend(saved_values)
#                     else:
#                         results_model[key].append(saved_values)
#     else:
#         results_model = []
#         for chain in chains:
#             if concatenate_chains:
#                 results_model.extend(chain.saved_models)
#             else:
#                 results_model.append(chain.saved_models)
#     return results_model












