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
RF_STD = 0.03
VS_PERTURB_STD = 0.15
VS_UNIFORM_MIN = 4
VS_UNIFORM_MAX = 5
THICKNESS = np.array([10, 10, 15, 20, 20, 20, 20, 20, 0], dtype=float)
VORONOI_SITES = np.array([5, 15, 25, 45, 65, 85, 105, 125, 145], dtype=float)
VORONOI_POS_MAX = 150
N_CHAINS = 2


def forward_sw(model, periods, wave="rayleigh", mode=1):
    vs = model["parameter_space"]["vs"]
    thickness = THICKNESS
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

true_vs = np.array([3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5])
true_model = bb.State(
    {"parameter_space": bb.ParameterSpaceState(len(true_vs), {"vs": true_vs})}
)

periods1 = np.linspace(4, 80, 20)
rayleigh1 = forward_sw(true_model, periods1, "rayleigh", 1)
rayleigh1_dobs = rayleigh1 + np.random.normal(0, RAYLEIGH_STD, rayleigh1.size)


# -------------- Define bayesbridge objects
targets = [
    bb.Target("rayleigh1", rayleigh1_dobs, covariance_mat_inv=1 / RAYLEIGH_STD**2),
]

fwd_functions = [
    (forward_sw, [periods1, "rayleigh", 1]),
]

param_vs = bb.parameters.UniformParameter(
    name="vs",
    vmin=VS_UNIFORM_MIN,
    vmax=VS_UNIFORM_MAX,
    perturb_std=VS_PERTURB_STD,
)


def param_vs_initialize(
    param: bb.parameters.Parameter, 
    positions: Union[np.ndarray, Number]
) -> Union[np.ndarray, Number]: 
    vmin, vmax = param.get_vmin_vmax(positions)
    if isinstance(positions, (float, int)):
        return random.uniform(vmin, vmax)
    sorted_vals = np.sort(np.random.uniform(vmin, vmax, positions.size))
    return sorted_vals


param_vs.set_custom_initialize(param_vs_initialize)
free_parameters = [param_vs]

parameterization = bb.parameterization.Parameterization(
    bb.parameterization.ParameterSpace(
        name="parameter_space", 
        n_dimensions=len(VORONOI_SITES), 
        parameters=free_parameters
    )
)


# -------------- Define BayesianInversion
inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    targets=targets, 
    fwd_functions=fwd_functions, 
    n_chains=N_CHAINS, 
    n_cpus=N_CHAINS, 
)
inversion.run(
    n_iterations=5_000,
    burnin_iterations=2_000,
    save_every=100,
    print_every=500,
)

# saving plots, models and targets
saved_models = inversion.get_results(concatenate_chains=True)
interp_depths = np.arange(VORONOI_POS_MAX, dtype=float)
all_thicknesses = [THICKNESS for _ in range(len(saved_models["vs"]))]

# plot samples, true model and statistics (mean, median, quantiles, etc.)
ax = bb.discretization.Voronoi1D.plot_depth_profiles(
    all_thicknesses, saved_models["vs"], linewidth=0.1, color="k"
)
bb.discretization.Voronoi1D.plot_depth_profiles(
    [THICKNESS], [true_vs], alpha=1, ax=ax, color="r", label="True"
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
for d in np.cumsum(THICKNESS):
    axes[1].axhline(d, color="red", linewidth=1)

# saving plots, models and targets
prefix = "toy_sw_fixed_sites"
ax.get_figure().savefig(f"{prefix}_samples")
fig.savefig(f"{prefix}_density")
np.save(f"{prefix}_saved_models", saved_models)
