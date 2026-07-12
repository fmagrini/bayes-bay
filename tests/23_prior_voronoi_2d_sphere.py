import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb
from bayesbay.discretization import Voronoi2DSphere


# define parameter space: Voronoi2DSphere
vel = bb.prior.UniformPrior("vel", vmin=2, vmax=4, perturb_std=0.4)
voronoi = Voronoi2DSphere(
    name="my_voronoi",
    perturb_std=20,
    n_dimensions=None,
    n_dimensions_min=2,
    n_dimensions_max=10,
    parameters=[vel],
    birth_from="neighbour",
)
parameterization = bb.parameterization.Parameterization(voronoi)

# define dummy log likelihood
targets = [bb.likelihood.Target("dummy_data", np.array([1], dtype=float), 1)]
fwd_functions = [lambda _: np.array([1], dtype=float)]
log_likelihood = bb.likelihood.LogLikelihood(targets, fwd_functions)

# run the sampler
inversion = bb.BayesianInversion(
    parameterization=parameterization,
    log_likelihood=log_likelihood,
    n_chains=1,
)
inversion.run(
    n_iterations=300_000,
    burnin_iterations=2_000,
    save_every=10,
    verbose=True,
    print_every=50_000,
)

# since the likelihood is constant, the samples should follow the prior:
# Voronoi sites uniform per unit area on the sphere (i.e. uniform in
# longitude and in the sine of the latitude), the number of Voronoi cells
# uniform in the range n_dimensions_min-n_dimensions_max, and the free
# parameter `vel` uniform within its bounds
results = inversion.get_results()
sites = np.vstack(results["my_voronoi.discretization"])
vels = np.concatenate(results["my_voronoi.vel"])
n_dims = np.array(results["my_voronoi.n_dimensions"])

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes[0, 0].hist(sites[:, 0], bins=24, range=(-180, 180), density=True, ec="w")
axes[0, 0].axhline(1 / 360, color="r", label="Prior")
axes[0, 0].set_xlabel("Longitude")
axes[0, 0].legend()
axes[0, 1].hist(np.sin(np.radians(sites[:, 1])), bins=24, range=(-1, 1), density=True, ec="w")
axes[0, 1].axhline(1 / 2, color="r", label="Prior")
axes[0, 1].set_xlabel("sin(Latitude)")
axes[0, 1].legend()
axes[1, 0].hist(vels, bins=20, range=(2, 4), density=True, ec="w")
axes[1, 0].axhline(1 / 2, color="r", label="Prior")
axes[1, 0].set_xlabel("vel")
axes[1, 0].legend()
axes[1, 1].hist(
    n_dims,
    bins=np.arange(voronoi._n_dimensions_min, voronoi._n_dimensions_max + 2) - 0.5,
    density=True,
    ec="w",
)
axes[1, 1].axhline(1 / (voronoi._n_dimensions_max - voronoi._n_dimensions_min + 1), color="r", label="Prior")
axes[1, 1].set_xlabel("Number of Voronoi cells")
axes[1, 1].legend()
fig.tight_layout()
fig.savefig("23_prior_voronoi_2d_sphere")

# display one of the sampled tessellations
iplot = next(
    i
    for i in range(len(results["my_voronoi.discretization"]) - 1, -1, -1)
    if len(results["my_voronoi.discretization"][i]) >= 4
)
ax, cbar = Voronoi2DSphere.plot_tessellation(
    results["my_voronoi.discretization"][iplot],
    results["my_voronoi.vel"][iplot],
    densify_deg=1,
)
ax.figure.savefig("23_prior_voronoi_2d_sphere_tessellation")
