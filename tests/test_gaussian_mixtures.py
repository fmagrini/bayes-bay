import numpy
import scipy
import matplotlib.pyplot as plt
import bayesbridge as bb


# constants
DATA_FILE = "rf_sw_uniform_1_000_000_saved_models.npy"
N_DATA = 100
DATA_Y_STD = 0.005
LAYERS_MIN = 3
LAYERS_MAX = 15
DEPTH_MAX = 130


# read and preprocess data
saved_models = numpy.load(DATA_FILE, allow_pickle=True).item()
depths = []
for thicknesses in saved_models["voronoi_cell_extents"]:
    depths.extend(numpy.cumsum(thicknesses))
h, e = numpy.histogram(depths, bins=N_DATA, density=True)
data_x = (e[:-1] + e[1:]) / 2.0
data_y = h

def forward_gaussian_mixtures(proposed_state: bb.State):
    n_mixtures = proposed_state.n_voronoi_cells
    means = proposed_state.voronoi_sites
    stds = proposed_state.std
    weights = proposed_state.weight
    weights /= numpy.sum(weights)
    result = numpy.zeros_like(data_x)
    for i in range(n_mixtures):
        result += weights[i] * scipy.stats.norm.pdf(data_x, loc=means[i], scale=stds[i])
    result /= numpy.trapz(result, data_x)
    return result

targets = [bb.Target("density", data_y, covariance_mat_inv=1/DATA_Y_STD**2)]
fwd_functions = [forward_gaussian_mixtures]
free_parameters = [
    bb.parameters.UniformParameter("std", vmin=1, vmax=20, perturb_std=0.1),
    bb.parameters.UniformParameter("weight", vmin=0, vmax=1, perturb_std=0.01)
]
parameterization = bb.Parameterization1D(
    voronoi_site_bounds=(0, DEPTH_MAX), 
    voronoi_site_perturb_std=3, 
    n_voronoi_cells=None,
    n_voronoi_cells_min=LAYERS_MIN,
    n_voronoi_cells_max=LAYERS_MAX, 
    free_params=free_parameters, 
)

# run
inversion = bb.BayesianInversion(
    parameterization, targets, fwd_functions, n_cpus=48, n_chains=48
)
inversion.run(
    n_iterations=1_000_000,
    burnin_iterations=400_000,
    save_every=1_000,
    print_every=5_000,
)


# plot
saved_models, saved_targets = inversion.get_results(concatenate_chains=True)
fig, ax = plt.subplots(figsize=(6,10))
ax.barh(
    data_x, 
    data_y, 
    align="edge", 
    label="density data"
)
ax.plot(
    numpy.mean(numpy.array(saved_targets["density"]["dpred"]), axis=0), 
    data_x, 
    color="r",
    label="density predicted"
)

prefix = "gmm"
fig.savefig(f"{prefix}_density_fit")
numpy.save(f"{prefix}_saved_models", saved_models)
numpy.save(f"{prefix}_saved_targets", saved_targets)
