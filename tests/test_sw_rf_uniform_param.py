#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:45:52 2023

@author: fabrizio

Vs model from Fu et al. 2016, https://doi.org/10.1002/2016JB013305
"""

import numpy as np
import matplotlib.pyplot as plt
from bayesbridge import Parameterization1D, Target, BayesianInversionFromParameterization, State
from bayesbridge.parameters import UniformParameter
from pysurf96 import surf96
from BayHunter import SynthObs


VP_VS = 1.77
RAYLEIGH_STD = 0.02
LOVE_STD = 0.02
RF_STD = 0.03
LAYERS_MIN = 3
LAYERS_MAX = 15

def forward_sw(proposed_state, periods, wave="rayleigh", mode=1):
    thickness = proposed_state.voronoi_cell_extents
    vs = proposed_state.vs
    vp = vs * VP_VS
    rho = 0.32 * vp + 0.77
    return surf96(
        thickness,
        vp,
        vs,
        rho,
        periods,
        wave=wave,
        mode=mode,
        velocity="phase",
        flat_earth=False,
    )

def forward_rf(proposed_state):
    vs = proposed_state.vs
    h = proposed_state.voronoi_cell_extents
    data = SynthObs.return_rfdata(h, vs, vpvs=VP_VS, x=None)
    return data['srf'][1,:]


thickness = np.array([10, 10, 15, 20, 20, 20, 20, 20, 0])
vs = np.array([3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5])
vp = vs * VP_VS
rho = 0.32 * vp + 0.77

true_model = State(len(thickness), thickness, thickness)
true_model.set_param_values("vs", vs)

periods1 = np.linspace(4, 80, 20)
rayleigh1 = forward_sw(true_model, periods1, "rayleigh", 1)
rayleigh1_noisy = rayleigh1 + np.random.normal(0, RAYLEIGH_STD, rayleigh1.size)
love1 = forward_sw(true_model, periods1, "love", 1)
love1_noisy = love1 + np.random.normal(0, LOVE_STD, love1.size)

periods2 = np.linspace(0.05, 20, 20)
rayleigh2 = forward_sw(true_model, periods2, "rayleigh", 2)
rayleigh2_noisy = rayleigh2 + np.random.normal(0, RAYLEIGH_STD, rayleigh2.size)
love2 = forward_sw(true_model, periods2, "love", 2)
love2_noisy = love2 + np.random.normal(0, LOVE_STD, love2.size)

periods3 = np.linspace(0.05, 10, 20)
rayleigh3 = forward_sw(true_model, periods3, "rayleigh", 3)
rayleigh3_noisy = rayleigh3 + np.random.normal(0, RAYLEIGH_STD, rayleigh3.size)
love3 = forward_sw(true_model, periods3, "love", 3)
love3_noisy = love3 + np.random.normal(0, LOVE_STD, love3.size)

periods4 = np.linspace(0.05, 8, 20)
rayleigh4 = forward_sw(true_model, periods4, "rayleigh", 4)
rayleigh4_noisy = rayleigh4 + np.random.normal(0, RAYLEIGH_STD, rayleigh4.size)
love4 = forward_sw(true_model, periods4, "love", 4)
love4_noisy = love4 + np.random.normal(0, LOVE_STD, love4.size)

rf = forward_rf(true_model)
rf_noisy = rf + np.random.normal(0, RF_STD, rf.size)

targets = [
    Target("rayleigh1", rayleigh1_noisy, covariance_mat_inv=1 / RAYLEIGH_STD**2),
    Target("love1", love1_noisy, covariance_mat_inv=1 / LOVE_STD**2),
    Target("rayleigh2", rayleigh2_noisy, covariance_mat_inv=1 / RAYLEIGH_STD**2),
    Target("love2", love2_noisy, covariance_mat_inv=1 / LOVE_STD**2),
    Target("rayleigh3", rayleigh3_noisy, covariance_mat_inv=1 / RAYLEIGH_STD**2),
    Target("love3", love3_noisy, covariance_mat_inv=1 / LOVE_STD**2),
    Target("rayleigh4", rayleigh4_noisy, covariance_mat_inv=1 / RAYLEIGH_STD**2),
    Target("love4", love4_noisy, covariance_mat_inv=1 / LOVE_STD**2),
    Target("rf", rf, covariance_mat_inv=1 / RF_STD**2),
]

fwd_functions = [
    (forward_sw, [periods1, "rayleigh", 1]), 
    (forward_sw, [periods1, "love", 1]),
    (forward_sw, [periods2, "rayleigh", 2]), 
    (forward_sw, [periods2, "love", 2]),
    (forward_sw, [periods3, "rayleigh", 3]), 
    (forward_sw, [periods3, "love", 3]),
    (forward_sw, [periods4, "rayleigh", 4]), 
    (forward_sw, [periods4, "love", 4]),
    forward_rf
]

param_vs = UniformParameter(
    "vs", 
    vmin=[2.7, 3.2, 3.75],
    vmax=[4, 4.75, 5],
    perturb_std=0.15,
    position=[0, 40, 80],
)

def param_vs_initialize(param, positions):
    vmin, vmax = param.get_vmin_vmax(positions)
    values = np.random.uniform(vmin, vmax, positions.size)
    sorted_values = np.sort(values)
    for i in range(len(sorted_values)):
        val = sorted_values[i]
        vmin_i = vmin if np.isscalar(vmin) else vmin[i]
        vmax_i = vmax if np.isscalar(vmax) else vmax[i]
        if val < vmin_i or val > vmax_i:
            if val > vmax_i: val = vmax_i
            if val < vmin_i: val = vmin_i
            sorted_values[i] = param.perturb_value(positions[i], val)
    return sorted_values

param_vs.set_custom_initialize(param_vs_initialize)
free_parameters = [param_vs]

parameterization = Parameterization1D(
    voronoi_site_bounds=(0, 130),
    voronoi_site_perturb_std=8,
    n_voronoi_cells=None,
    n_voronoi_cells_min=LAYERS_MIN,
    n_voronoi_cells_max=LAYERS_MAX,
    free_params=free_parameters,
)

inversion = BayesianInversionFromParameterization(
    parameterization, targets, fwd_functions=fwd_functions, n_cpus=48, n_chains=48
)

inversion.run(
    n_iterations=1_000_000,
    burnin_iterations=400_000,
    save_every=1_000,
    print_every=5_000,
)

saved_models, saved_targets = inversion.get_results(concatenate_chains=True)
interp_depths = np.arange(130, dtype=float)

# plot samples, true model and statistics (mean, median, quantiles, etc.)
ax = Parameterization1D.plot_param_samples(
    saved_models["voronoi_cell_extents"], saved_models["vs"], linewidth=0.1, color="k"
)
Parameterization1D.plot_param_samples(
    [thickness], [vs], alpha=1, ax=ax, color="r", label="True"
)
Parameterization1D.plot_ensemble_statistics(
    saved_models["voronoi_cell_extents"], saved_models["vs"], interp_depths, ax=ax
)

# plot depths and velocities density profile
fig, axes = plt.subplots(1, 2, figsize=(10, 8))
Parameterization1D.plot_depth_profile(
    saved_models["voronoi_cell_extents"], saved_models["vs"], ax=axes[0], vmax=5000
)
Parameterization1D.plot_interface_distribution(
    saved_models["voronoi_cell_extents"], ax=axes[1]
)
for d in np.cumsum(thickness):
    axes[1].axhline(d, color="red", linewidth=1)

# saving plots, models and targets
# prefix = "rf_uniform_1_000_000"
# prefix = "sw_uniform_1_000_000"
prefix = "rf_sw1234_uniform_1_000_000"
ax.get_figure().savefig(f"{prefix}_samples")
fig.savefig(f"{prefix}_density")
np.save(f"{prefix}_saved_models", saved_models)
np.save(f"{prefix}_saved_targets", saved_targets)
