import numpy as np
import scipy
import matplotlib.pyplot as plt

import bayesbridge as bb


# -------------- Setting up constants, fwd func, data
DATA_FILE = "inv_rf_sw_saved_models.npy"
N_DATA = 100
LAYERS_MIN = 3
LAYERS_MAX = 15
VORONOI_PERTURB_STD = 3
VORONOI_POS_MIN = 0
VORONOI_POS_MAX = 150
NOISE_STD = 0.005
N_CHAINS = 10


saved_models = np.load(DATA_FILE, allow_pickle=True).item()
depths = []
for sites in saved_models["voronoi"]:
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
    n_mixtures = model.get_param_values("voronoi").n_dimensions
    means = model.get_param_values("voronoi").get_param_values("voronoi")
    stds = model.get_param_values("voronoi").get_param_values("std")
    weights = model.get_param_values("voronoi").get_param_values("weight")
    return _forward(n_mixtures, means, stds, weights)


# -------------- Define bayesbridge objects
targets = [bb.Target("density", data_y, covariance_mat_inv=1 / NOISE_STD**2)]
fwd_functions = [forward_gaussian_mixtures]
free_parameters = [
    bb.parameters.UniformParameter("std", vmin=1, vmax=20, perturb_std=0.1),
    bb.parameters.UniformParameter("weight", vmin=0, vmax=1, perturb_std=0.01),
]

parameterization = bb.parameterization.Parameterization(
    bb.discretization.Voronoi1D(
        name="voronoi", 
        vmin=VORONOI_POS_MIN, 
        vmax=VORONOI_POS_MAX, 
        perturb_std=VORONOI_PERTURB_STD, 
        n_dimensions_min=LAYERS_MIN,
        n_dimensions_max=LAYERS_MAX, 
        parameters=free_parameters,
        birth_from="prior", 
    )
)


# -------------- Run inversion
inversion = bb.BayesianInversion(
    parameterization, targets, fwd_functions, n_cpus=N_CHAINS, n_chains=N_CHAINS
)
inversion.run(
    n_iterations=10_000,
    burnin_iterations=5_000,
    save_every=200,
    print_every=200,
)


# -------------- Plot and save results
saved_models = inversion.get_results(concatenate_chains=True)
saved_models_dpred = [
    _forward(*m) for m in zip(
        saved_models["voronoi.n_dimensions"], 
        saved_models["voronoi"], 
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
