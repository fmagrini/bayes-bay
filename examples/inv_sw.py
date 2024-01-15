from typing import Union
from numbers import Number
import random
import numpy as np
import matplotlib.pyplot as plt

from pysurf96 import surf96
import bayesbridge as bb


# -------------- Setting up constants, fwd func, synth data
VP_VS = 1.77
RAYLEIGH_STD = 0.02
LOVE_STD = 0.02
LAYERS_MIN = 3
LAYERS_MAX = 15
LAYERS_INIT_RANGE = 0.3
VS_PERTURB_STD = 0.15
VS_UNIFORM_MIN = [2.7, 3.2, 3.75]
VS_UNIFORM_MAX = [4, 4.75, 5]
VS_UNIFORM_POS = [0, 40, 80]
VORONOI_PERTURB_STD = 8
VORONOI_POS_MIN = 0
VORONOI_POS_MAX = 150
N_CHAINS = 48


def _calc_thickness(sites: np.ndarray):
    return bb.discretization.Voronoi1D.compute_cell_extents(np.array(sites, dtype=float))

def _get_thickness(state: bb.State):
    sites = state["voronoi"]["discretization"]
    return _calc_thickness(sites)

def forward_sw(state: bb.State, periods, wave="rayleigh", mode=1):
    vs = state["voronoi"]["vs"]
    h = _get_thickness(state)
    vp = vs * VP_VS
    rho = 0.32 * vp + 0.77
    return surf96(
        h,
        vp,
        vs,
        rho,
        periods,
        wave=wave,
        mode=mode,
        velocity="phase",
        flat_earth=False,
    )

true_thickness = np.array([10, 10, 15, 20, 20, 20, 20, 20, 0])
true_voronoi_positions = np.array([5, 15, 25, 45, 65, 85, 105, 125, 145])
true_vs = np.array([3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5])
true_model = bb.State(
    {
        "voronoi": bb.ParameterSpaceState(
            len(true_vs), 
            {"discretization": true_voronoi_positions, "vs": true_vs}
        )
    }
)

periods1 = np.linspace(4, 80, 20)
rayleigh1 = forward_sw(true_model, periods1, "rayleigh", 1)
rayleigh1_dobs = rayleigh1 + np.random.normal(0, RAYLEIGH_STD, rayleigh1.size)
love1 = forward_sw(true_model, periods1, "love", 1)
love1_dobs = love1 + np.random.normal(0, LOVE_STD, love1.size)

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


# -------------- Define bayesbridge objects
targets = [
    bb.Target("rayleigh1", rayleigh1_dobs, covariance_mat_inv=1 / RAYLEIGH_STD**2),
    bb.Target("love1", love1_dobs, covariance_mat_inv=1 / LOVE_STD**2),
    # bb.Target("rayleigh2", rayleigh2_noisy, covariance_mat_inv=1 / RAYLEIGH_STD**2),
    # bb.Target("love2", love2_noisy, covariance_mat_inv=1 / LOVE_STD**2),
    # bb.Target("rayleigh3", rayleigh3_noisy, covariance_mat_inv=1 / RAYLEIGH_STD**2),
    # bb.Target("love3", love3_noisy, covariance_mat_inv=1 / LOVE_STD**2),
    # bb.Target("rayleigh4", rayleigh4_noisy, covariance_mat_inv=1 / RAYLEIGH_STD**2),
    # bb.Target("love4", love4_noisy, covariance_mat_inv=1 / LOVE_STD**2),
]

fwd_functions = [
    (forward_sw, [periods1, "rayleigh", 1]),
    (forward_sw, [periods1, "love", 1]),
    # (forward_sw, [periods2, "rayleigh", 2]),
    # (forward_sw, [periods2, "love", 2]),
    # (forward_sw, [periods3, "rayleigh", 3]),
    # (forward_sw, [periods3, "love", 3]),
    # (forward_sw, [periods4, "rayleigh", 4]),
    # (forward_sw, [periods4, "love", 4]),
]

param_vs = bb.parameters.UniformParameter(
    name="vs",
    vmin=VS_UNIFORM_MIN,
    vmax=VS_UNIFORM_MAX,
    perturb_std=VS_PERTURB_STD,
    position=VS_UNIFORM_POS, 
)


def param_vs_initialize(
    param: bb.parameters.Parameter, 
    positions: Union[np.ndarray, Number]
) -> Union[np.ndarray, Number]: 
    vmin, vmax = param.get_vmin_vmax(positions)
    if isinstance(positions, (float, int)):
        return random.uniform(vmin, vmax)
    sorted_vals = np.sort(np.random.uniform(vmin, vmax, positions.size))
    for i in range (len(sorted_vals)):
        val = sorted_vals[i]
        vmin_i = vmin if np.isscalar(vmin) else vmin[i]
        vmax_i = vmax if np.isscalar(vmax) else vmax[i]
        if val < vmin_i or val > vmax_i:
            if val > vmax_i: sorted_vals[i] = vmax_i
            if val < vmin_i: sorted_vals[i] = vmin_i
    return sorted_vals


param_vs.set_custom_initialize(param_vs_initialize)
free_parameters = [param_vs]

parameterization = bb.parameterization.Parameterization(
    bb.discretization.Voronoi1D(
        name="voronoi", 
        vmin=VORONOI_POS_MIN, 
        vmax=VORONOI_POS_MAX, 
        perturb_std=VORONOI_PERTURB_STD, 
        n_dimensions=None, 
        n_dimensions_min=LAYERS_MIN, 
        n_dimensions_max=LAYERS_MAX, 
        n_dimensions_init_range=LAYERS_INIT_RANGE, 
        parameters=free_parameters,
        birth_from="prior", 
    )
)

# -------------- Run inversion
inversion = bb.BayesianInversion(
    parameterization=parameterization,
    targets=targets,
    fwd_functions=fwd_functions,
    n_chains=N_CHAINS,
    n_cpus=N_CHAINS,
)
inversion.run(
    n_iterations=300_000,
    burnin_iterations=50_000,
    save_every=1_000,
    print_every=1_000,
)

# -------------- Plot and save results
saved_models = inversion.get_results(concatenate_chains=True)
interp_depths = np.arange(VORONOI_POS_MAX, dtype=float)
all_thicknesses = [_calc_thickness(m) for m in saved_models["voronoi.discretization"]]

# plot samples, true model and statistics (mean, median, quantiles, etc.)
ax = bb.discretization.Voronoi1D.plot_depth_profiles(
    all_thicknesses, saved_models["vs"], linewidth=0.1, color="k"
)
bb.discretization.Voronoi1D.plot_depth_profiles(
    [true_thickness], [true_vs], alpha=1, ax=ax, color="r", label="True"
)
bb.discretization.Voronoi1D.plot_depth_profiles_statistics(
    all_thicknesses, saved_models["vs"], interp_depths, ax=ax
)

# plot depths and velocities density profile
fig, axes = plt.subplots(1, 2, figsize=(10, 8))
bb.discretization.Voronoi1D.plot_depth_profiles_density(
    all_thicknesses, saved_models["vs"], ax=axes[0]
)
bb.discretization.Voronoi1D.plot_interface_hist(
    all_thicknesses, ax=axes[1]
)
for d in np.cumsum(true_thickness):
    axes[1].axhline(d, color="red", linewidth=1)

# saving plots, models and targets
prefix = "inv_sw"
ax.get_figure().savefig(f"{prefix}_samples")
fig.savefig(f"{prefix}_density")
np.save(f"{prefix}_saved_models", saved_models)
