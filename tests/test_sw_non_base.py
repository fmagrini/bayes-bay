import numpy as np
import random
import math
from pysurf96 import surf96
import bayesbridge as bb


# -------------- Setting up constants, fwd func, synth data
VP_VS = 1.77
RAYLEIGH_STD = 0.02
LOVE_STD = 0.02
RF_STD = 0.03
LAYERS_MIN = 3
LAYERS_MAX = 15
LAYERS_INIT_RANGE = 0.3
VS_PERTURB_STD = 0.15
VS_UNIFORM_MIN = 2.7
VS_UNIFORM_MAX = 5
VORONOI_PERTURB_STD = 8
VORONOI_POS_MIN = 0
VORONOI_POS_MAX = 130
N_CHAINS = 10

def forward_sw(model: bb.State, periods, wave="rayleigh", mode=1):
    k = model.n_voronoi_cells
    sites = model.voronoi_sites
    vs = model.get_param_values("vs")
    depths = (sites[:-1] + sites[1:]) / 2
    thickness = np.hstack((depths[0], depths[1:]-depths[:-1], 0))
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

true_thickness = np.array([10, 10, 15, 20, 20, 20, 20, 20, 0])
true_voronoi_positions = np.array([5, 15, 25, 45, 65, 85, 105, 125, 145])
true_vs = np.array([3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5])
true_model = bb.State(len(true_vs), true_voronoi_positions, {"vs": true_vs})

periods1 = np.linspace(4, 80, 20)
rayleigh1 = forward_sw(true_model, periods1, "rayleigh", 1)
rayleigh1_dobs = rayleigh1 + np.random.normal(0, RAYLEIGH_STD, rayleigh1.size)
love1 = forward_sw(true_model, periods1, "love", 1)
love1_dobs = love1 + np.random.normal(0, LOVE_STD, love1.size)


# -------------- Define bayesbridge objects
targets = [
    bb.Target("rayleigh1", rayleigh1_dobs, covariance_mat_inv=1 / RAYLEIGH_STD**2), 
    bb.Target("love1", love1_dobs, covariance_mat_inv=1 / LOVE_STD**2),
]

fwd_functions = [
    (forward_sw, [periods1, "rayleigh", 1]), 
    (forward_sw, [periods1, "love", 1]), 
]

param_vs = bb.parameters.UniformParameter(
    name="vs", 
    vmin=VS_UNIFORM_MIN, 
    vmax=VS_UNIFORM_MAX, 
    perturb_std=VS_PERTURB_STD, 
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

parameterization = bb.Voronoi1D(
    voronoi_site_bounds=(VORONOI_POS_MIN, VORONOI_POS_MAX), 
    voronoi_site_perturb_std=VORONOI_PERTURB_STD, 
    n_voronoi_cells=None, 
    n_voronoi_cells_min=LAYERS_MIN, 
    n_voronoi_cells_max=LAYERS_MAX, 
    birth_from="neighbour",     # or "prior"
)

# -------------- Run inversion
inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    targets=targets, 
    fwd_functions=fwd_functions, 
    n_chains=N_CHAINS, 
    n_cpus=N_CHAINS
)
inversion.run()
