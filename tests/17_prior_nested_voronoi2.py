import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter space
vs = bb.prior.UniformPrior(
    name="vs", 
    # vmin=2.2, 
    # vmax=5, 
    vmin=[2.2, 2.8, 3.3, 4], 
    vmax=[3.9, 4.6, 4.8, 5], 
    position=[0, 20, 60, 150],
    perturb_std=0.1, 
    # perturb_std_birth=0.2
)
v_vertical = bb.discretization.Voronoi1D(
    name="v_vertical", 
    vmin=0,
    vmax=150,
    perturb_std=10,
    n_dimensions=None, 
    n_dimensions_min=5,
    n_dimensions_max=20,
    parameters=[vs], 
    birth_from="prior"
)
v_horizontal = bb.discretization.Voronoi2D(
    name="v_horizontal",
    vmin=0, 
    vmax=20, 
    polygon=None, 
    perturb_std=2, 
    n_dimensions_min=1, 
    n_dimensions_max=10,
    parameters=[v_vertical],
    birth_from="prior"
)
parameterization = bb.parameterization.Parameterization(v_horizontal)

# define dummy log likelihood
targets = [bb.Target("dummy_data", np.array([1], dtype=float), 1)]
fwd_functions = [lambda _: np.array([1], dtype=float)]
log_likelihood = bb.LogLikelihood(targets, fwd_functions)

# run the sampler
inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    log_likelihood=log_likelihood,  
    n_chains=10, 
)
inversion.run(
    sampler=None, 
    n_iterations=500_000, 
    burnin_iterations=0, 
    save_every=100, 
    print_every=2000, 
)

# get results and plot
results = inversion.get_results()
n_dims_v_horizontal = results["v_horizontal.n_dimensions"]
n_dims_v_vertical = []
sites_v_horizontal = np.concatenate(results["v_horizontal.discretization"])
sites_v_vertical = []
velocities = []
for v_horizontal_sample in results["v_horizontal.v_vertical"]:
    for v_vertical in v_horizontal_sample:
        n_dims_v_vertical.append(v_vertical["v_horizontal.v_vertical.n_dimensions"])
        sites_v_vertical.extend(v_vertical["v_horizontal.v_vertical.discretization"])
        velocities.extend(v_vertical["v_horizontal.v_vertical.vs"])

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].hist(n_dims_v_horizontal, bins=np.arange(0.5,10.5), ec="w")
axes[0].set_title("v_horizontal")
axes[1].hist(n_dims_v_vertical, bins=np.arange(4.5,20.5), ec="w")
axes[1].set_title("v_vertical")
axes[2].hist(velocities, bins=20, ec="w")
axes[2].set_title("velocity")
fig.tight_layout()
fig.savefig("17_nested_voronoi2")
