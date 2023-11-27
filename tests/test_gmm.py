import numpy as np
import scipy
import matplotlib.pyplot as plt

import bayesbridge as bb


# -------------- Setting up constants, fwd func, data
DATA_FILE = "rf_sw_1_000_000_saved_models.npy"
N_DATA = 100
DATA_Y_STD = 0.005
LAYERS_MIN = 3
LAYERS_MAX = 15
VORONOI_PERTURB_STD = 3
VORONOI_POS_MIN = 0
VORONOI_POS_MAX = 130
N_CHAINS = 10


saved_models = np.load(DATA_FILE, allow_pickle=True).item()
depths = []
for sites in saved_models["voronoi_sites"]:
    depths.extend((sites[:-1] + sites[1:])/2)
h, e = np.histogram(depths, bins=N_DATA, density=True)
data_x = (e[:-1] + e[1:]) / 2.0
data_y = h

def _forward(n_mixtures, means, stds, weights):
    weights /= np.sum(weights)
    result = np.zeros_like(data_x)
    for i in range(n_mixtures):
        result += weights[i] * scipy.stats.norm.pdf(data_x, loc=means[i], scale=stds[i])
    result /= np.trapz(result, data_x)
    return result

def forward_gaussian_mixtures(model: bb.State):
    n_mixtures = model.n_voronoi_cells
    means = model.voronoi_sites
    stds = model.std
    weights = model.weight
    return _forward(n_mixtures, means, stds, weights)


# -------------- Define bayesbridge objects
targets = [bb.Target("density", data_y, covariance_mat_inv=1 / DATA_Y_STD**2)]
fwd_functions = [forward_gaussian_mixtures]
free_parameters = [
    bb.parameters.UniformParameter("std", vmin=1, vmax=20, perturb_std=0.1),
    bb.parameters.UniformParameter("weight", vmin=0, vmax=1, perturb_std=0.01),
]

parameterization = bb.Voronoi1D(
    voronoi_site_bounds=(VORONOI_POS_MIN, VORONOI_POS_MAX),
    voronoi_site_perturb_std=VORONOI_PERTURB_STD,
    n_voronoi_cells=None,
    n_voronoi_cells_min=LAYERS_MIN,
    n_voronoi_cells_max=LAYERS_MAX,
    free_params=free_parameters,
    birth_from="prior", 
)


# -------------- Run inversion
inversion = bb.BayesianInversion(
    parameterization, targets, fwd_functions, n_cpus=N_CHAINS, n_chains=N_CHAINS
)
inversion.run(
    n_iterations=1_000,
    burnin_iterations=500,
    save_every=100,
    print_every=100,
)


# -------------- Plot and save results
saved_models = inversion.get_results(concatenate_chains=True)
saved_models_dpred = [
    _forward(*m) for m in zip(
        saved_models["n_voronoi_cells"], 
        saved_models["voronoi_sites"], 
        saved_models["std"], 
        saved_models["weight"], 
    )
]
fig, ax = plt.subplots(figsize=(6, 10))
ax.barh(data_x, data_y, align="edge", label="density data")
ax.plot(
    np.mean(np.array(saved_models_dpred), axis=0),
    data_x,
    color="r",
    label="density predicted",
)

prefix = "gmm"
fig.savefig(f"{prefix}_density_fit")
np.save(f"{prefix}_saved_models", saved_models)
