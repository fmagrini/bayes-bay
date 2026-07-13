import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
import bayesbay as bb
from bayesbay.discretization import Voronoi2DSphere


# region of interest: two disjoint patches (MultiPolygon)
polygon = shapely.geometry.MultiPolygon(
    [
        shapely.geometry.Polygon([(-6, 30), (12, 30), (12, 46), (-6, 46)]),
        shapely.geometry.Polygon([(20, 32), (36, 32), (36, 44), (20, 44)]),
    ]
)

vel = bb.prior.UniformPrior("vel", vmin=2, vmax=4, perturb_std=0.4)
voronoi = Voronoi2DSphere(
    name="my_voronoi",
    perturb_std=3,
    polygon=polygon,
    n_dimensions=None,
    n_dimensions_min=2,
    n_dimensions_max=10,
    parameters=[vel],
    birth_from="neighbour",
)
parameterization = bb.parameterization.Parameterization(voronoi)

# define dummy log likelihood: the samples should follow the prior
targets = [bb.likelihood.Target("dummy_data", np.array([1], dtype=float), 1)]
fwd_functions = [lambda _: np.array([1], dtype=float)]
log_likelihood = bb.likelihood.LogLikelihood(targets, fwd_functions)

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

results = inversion.get_results()
sites = np.vstack(results["my_voronoi.discretization"])
vels = np.concatenate(results["my_voronoi.vel"])
n_dims = np.array(results["my_voronoi.n_dimensions"])

# all sampled sites must lie inside the region of interest
outside = [
    s for s in sites if not polygon.contains(shapely.geometry.Point(s[0], s[1]))
]
assert not outside, f"{len(outside)} sites sampled outside the polygon!"
in_gap = (sites[:, 0] > 12) & (sites[:, 0] < 20)
assert not in_gap.any(), "sites sampled in the gap between the two patches!"
print("All sampled Voronoi sites lie within the region of interest.")

# under a constant likelihood, site positions are uniform per unit area
# within the region: the density in (lon, sin lat) space should be flat
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
h = axes[0, 0].hist2d(
    sites[:, 0], np.sin(np.radians(sites[:, 1])), bins=(42, 16), density=True, cmin=None
)
fig.colorbar(h[3], ax=axes[0, 0])
axes[0, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("sin(Latitude)")
axes[0, 0].set_title("Site density (should be flat within the region)")

axes[0, 1].hist(vels, bins=20, range=(2, 4), density=True, ec="w")
axes[0, 1].axhline(1 / 2, color="r", label="Prior")
axes[0, 1].set_xlabel("vel")
axes[0, 1].legend()

axes[1, 0].hist(
    n_dims,
    bins=np.arange(voronoi._n_dimensions_min, voronoi._n_dimensions_max + 2) - 0.5,
    density=True,
    ec="w",
)
axes[1, 0].axhline(
    1 / (voronoi._n_dimensions_max - voronoi._n_dimensions_min + 1), color="r", label="Prior"
)
axes[1, 0].set_xlabel("Number of Voronoi cells")
axes[1, 0].legend()

iplot = next(
    i
    for i in range(len(results["my_voronoi.discretization"]) - 1, -1, -1)
    if len(results["my_voronoi.discretization"][i]) >= 4
)
Voronoi2DSphere.plot_tessellation(
    results["my_voronoi.discretization"][iplot],
    results["my_voronoi.vel"][iplot],
    ax=axes[1, 1],
)
axes[1, 1].set_xlim(-10, 40)
axes[1, 1].set_ylim(25, 50)
axes[1, 1].set_title("One posterior sample")
fig.tight_layout()
fig.savefig("24_prior_voronoi_sphere_polygon")
