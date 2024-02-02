import random
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

from pysurf96 import surf96
import bayesbay as bb


# -------------- Setting up constants, fwd func, synth data
VP_VS = 1.77
RAYLEIGH_STD = 0.02
RF_STD = 0.03
LAYERS_MIN = 3
LAYERS_MAX = 15
LAYERS_INIT_RANGE = 0.3
VS_PERTURB_STD = 0.15
VS_UNIFORM_MIN = 2
VS_UNIFORM_MAX = 5
VORONOI_PERTURB_STD = 8
VORONOI_POS_MIN = 0
VORONOI_POS_MAX = 150
N_CHAINS = 2


def forward_sw(model, periods, wave="rayleigh", mode=1):
    k = int(len(model) / 2)
    sites = model[:k]
    vs = model[k:]
    thickness = bb.discretization.Voronoi1D.compute_cell_extents(
        np.array(sites, dtype=float)
    )
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
true_model = np.hstack((true_thickness, true_vs))

periods1 = np.linspace(4, 80, 20)
rayleigh1 = forward_sw(true_model, periods1, "rayleigh", 1)
rayleigh1_dobs = rayleigh1 + np.random.normal(0, RAYLEIGH_STD, rayleigh1.size)


# -------------- Implement distribution functions
def log_prior(model):
    k = len(model) // 2
    # p(k) and p(c|k) are to be cancelled out in acceptance criteria
    # p(v|k) prior on param value given #layers
    log_p_v_k = -k * math.log(VS_UNIFORM_MAX - VS_UNIFORM_MIN)
    return log_p_v_k


def log_likelihood(model):
    rayleigh_dpred = forward_sw(model, periods1)
    rayleigh_residual = rayleigh1_dobs - rayleigh_dpred
    rayleigh_loglike = -0.5 * np.sum(
        (rayleigh_residual / RAYLEIGH_STD) ** 2
        + math.log(2 * np.pi * RAYLEIGH_STD**2)
    )
    return rayleigh_loglike


# -------------- Implement perturbation functions
def perturbation_vs(model):
    k = int(len(model) / 2)
    sites = model[:k]
    vs = model[k:]
    # randomly choose a Voronoi site to perturb the value
    isite = random.randint(0, k - 1)
    # randomly perturb the value
    while True:
        random_deviate = random.normalvariate(0, VS_PERTURB_STD)
        new_value = vs[isite] + random_deviate
        if new_value > VS_UNIFORM_MAX or new_value < VS_UNIFORM_MIN:
            continue
        break
    # integrate into a new model variable
    new_vs = vs.copy()
    new_vs[isite] = new_value
    new_model = np.hstack((sites, new_vs))
    log_prob_ratio = 0
    return new_model, log_prob_ratio


def perturbation_voronoi_site(model):
    k = int(len(model) / 2)
    sites = model[:k]
    vs = model[k:]
    # randomly choose a Voronoi site to perturb the position
    isite = random.randint(0, k - 1)
    old_site = sites[isite]
    # randomly perturb the position
    while True:
        random_deviate = random.normalvariate(0, VORONOI_PERTURB_STD)
        new_site = old_site + random_deviate
        if (
            new_site < VORONOI_POS_MIN
            or new_site > VORONOI_POS_MAX
            or np.any(np.abs(new_site - sites) < 1e-2)
        ):
            continue
        break
    # integrate into a new model variable
    new_sites = sites.copy()
    new_sites[isite] = new_site
    isort = np.argsort(new_sites)
    new_sites = new_sites[isort]
    new_vs = vs[isort]
    new_model = np.hstack((new_sites, new_vs))
    log_prob_ratio = 0
    return new_model, log_prob_ratio


def perturbation_birth(model):
    k = int(len(model) / 2)
    sites = model[:k]
    vs = model[k:]
    if k == LAYERS_MAX:
        raise ValueError("Maximum layers reached")
    # randomly choose a new Voronoi site position
    while True:
        new_site = random.uniform(VORONOI_POS_MIN, VORONOI_POS_MAX)
        # abort if it's too close to existing positions
        if np.any(np.abs(new_site - sites) < 1e-2):
            continue
        break
    # randomly sample the value for the new site
    new_vs_isite = random.uniform(VS_UNIFORM_MIN, VS_UNIFORM_MAX)
    # integrate into a new model variable and sort properly
    new_sites = np.append(sites, new_site)
    new_vs = np.append(vs, new_vs_isite)
    isort = np.argsort(new_sites)
    new_sites = new_sites[isort]
    new_vs = new_vs[isort]
    new_model = np.hstack((new_sites, new_vs))
    # calculate partial acceptance probability
    log_prob_ratio = 0
    return new_model, log_prob_ratio


def perturbation_death(model):
    k = int(len(model) / 2)
    sites = model[:k]
    vs = model[k:]
    if k == LAYERS_MIN:
        raise ValueError("Minimum layers reached")
    # randomly choose an existing Voronoi site to remove
    isite = random.randint(0, k - 1)
    # integrate into a new model variable
    new_sites = np.delete(sites, isite)
    new_vs = np.delete(vs, isite)
    new_model = np.hstack((new_sites, new_vs))
    # calculate partial acceptance probability
    log_prob_ratio = 0
    return new_model, log_prob_ratio


# -------------- Initialize walkers
init_max = int((LAYERS_MAX - LAYERS_MIN) * LAYERS_INIT_RANGE + LAYERS_MIN)
walkers_start = []
for i in range(N_CHAINS):
    n_sites = random.randint(LAYERS_MIN, init_max)
    sites = np.sort(np.random.uniform(VORONOI_POS_MIN, VORONOI_POS_MAX, n_sites))
    vs = np.sort(np.random.uniform(VS_UNIFORM_MIN, VS_UNIFORM_MAX, n_sites))
    model = np.hstack((sites, vs))
    walkers_start.append(model)


# -------------- Define BayesianInversion
inversion = bb.BaseBayesianInversion(
    walkers_starting_models=walkers_start,
    perturbation_funcs=[
        perturbation_vs,
        perturbation_voronoi_site,
        perturbation_birth,
        perturbation_death,
    ],
    log_like_func=log_likelihood,
    n_chains=N_CHAINS,
    n_cpus=N_CHAINS,
)
inversion.run(
    n_iterations=50_000,
    burnin_iterations=20_000,
    save_every=100,
    print_every=500,
)

# saving plots, models and targets
def _calc_thickness(model: np.ndarray):
    k = len(model) // 2
    sites = model[:k]
    thickness = bb.discretization.Voronoi1D.compute_cell_extents(
        np.array(sites, dtype=float)
    )
    return thickness

def _get_vs(model: np.ndarray):
    k = int(len(model) / 2)
    return model[k:]

saved_states = inversion.get_results(concatenate_chains=True)
interp_depths = np.arange(VORONOI_POS_MAX, dtype=float)
all_thicknesses = [_calc_thickness(m) for m in saved_states]
all_vs = [_get_vs(m) for m in saved_states]

# plot samples, true model and statistics (mean, median, quantiles, etc.)
ax = bb.discretization.Voronoi1D.plot_depth_profiles(
    all_thicknesses, all_vs, linewidth=0.1, color="k"
)
bb.discretization.Voronoi1D.plot_depth_profiles(
    [true_thickness], [true_vs], alpha=1, ax=ax, color="r", label="True"
)
bb.discretization.Voronoi1D.plot_depth_profiles_statistics(
    all_thicknesses, all_vs, interp_depths, ax=ax
)

# plot depths and velocities density profile
fig, axes = plt.subplots(1, 2, figsize=(10, 8))
bb.discretization.Voronoi1D.plot_depth_profiles_density(
    all_thicknesses, all_vs, ax=axes[0]
)
bb.discretization.Voronoi1D.plot_interface_hist(
    all_thicknesses, ax=axes[1]
)
for d in np.cumsum(true_thickness):
    axes[1].axhline(d, color="red", linewidth=1)

# saving plots, models and targets
prefix = "toy_sw_base_api"
ax.get_figure().savefig(f"{prefix}_samples")
fig.savefig(f"{prefix}_density")
with open(f"{prefix}_saved_states.pkl", "wb") as f:
    pickle.dump(saved_states, f)
