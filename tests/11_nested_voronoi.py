import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter space
# velocity = bb.prior.UniformPrior("velocity", 0, 100, 0.1)
# v_vertical = bb.discretization.Voronoi1D(
#     name="v_vertical",
#     vmin=0,
#     vmax=100,
#     perturb_std=5,
#     n_dimensions=None,
#     n_dimensions_min=1,
#     n_dimensions_max=10,
#     n_dimensions_init_range=0.3,
#     parameters=[velocity],
#     # birth_from="prior"
# )
# v_horizontal = bb.discretization.Voronoi2D(
#     name="v_horizontal", 
#     vmin=0, 
#     vmax=100, 
#     polygon=None, 
#     perturb_std=5, 
#     n_dimensions=None, 
#     n_dimensions_min=1, 
#     n_dimensions_max=10, 
#     n_dimensions_init_range=0.3, 
#     parameters=[v_vertical], 
#     # birth_from="prior"
# )

uniform_param = bb.prior.UniformPrior("uniform_param", -1, 1, 0.1)
voronoi1 = bb.discretization.Voronoi1D(
    name="my_voronoi1", 
    vmin=0, 
    vmax=1, 
    perturb_std=0.4, 
    n_dimensions=None, 
    n_dimensions_min=1, 
    n_dimensions_max=10, 
    parameters=[uniform_param], 
    birth_from="prior"
    # parameters=[]
)
voronoi2 = bb.discretization.Voronoi1D(
    name="my_voronoi2", 
    vmin=0, 
    vmax=1, 
    perturb_std=0.4, 
    n_dimensions=None, 
    n_dimensions_min=1, 
    n_dimensions_max=10, 
    parameters=[voronoi1], 
    birth_from="prior"
)
parameterization = bb.parameterization.Parameterization(voronoi2)

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
# perturbations = inversion.perturbation_funcs
# perturbations[0].perturbation_functions = perturbations[0].perturbation_functions[:2]
# perturbations[0].perturbation_weights = perturbations[0].perturbation_weights[:2]
# perturbations[1].perturbation_functions = perturbations[1].perturbation_functions[:2]
# perturbations[1].perturbation_weights = perturbations[1].perturbation_weights[:2]
# inversion.set_perturbation_funcs(perturbations, [1, 10])
inversion.set_perturbation_funcs(inversion.perturbation_funcs, [1, 10])
inversion.run(
    sampler=None, 
    n_iterations=100_000, 
    burnin_iterations=0, 
    save_every=1, 
    print_every=1000, 
)

# get results and plot
results = inversion.get_results()
n_dims_voronoi2 = results["my_voronoi2.n_dimensions"]
n_dims_voronoi1 = []
sites_voronoi2 = np.concatenate(results["my_voronoi2.discretization"])
sites_voronoi1 = []
uniform_param_values = []
for v_vertical_sample in results["my_voronoi2.my_voronoi1"]:
    for v_vertical in v_vertical_sample:
        n_dims_voronoi1.append(v_vertical["my_voronoi2.my_voronoi1.n_dimensions"])
        sites_voronoi1.extend(v_vertical["my_voronoi2.my_voronoi1.discretization"])
        uniform_param_values.extend(v_vertical["my_voronoi2.my_voronoi1.uniform_param"])

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].hist(n_dims_voronoi2, bins=10, ec="w")
axes[0].set_title("voronoi2")
axes[1].hist(n_dims_voronoi1, bins=10, ec="w")
axes[1].set_title("voronoi1")
axes[2].hist(uniform_param_values, bins=20, ec="w")
axes[2].set_title("uniform_param")
fig.tight_layout()
fig.savefig("11_nested_voronoi")

# n_dims_v_horizontal = results["v_horizontal.n_dimensions"]
# n_dims_v_vertical = []
# sites_v_horizontal = np.concatenate(results["v_horizontal.discretization"])
# sites_v_vertical = []
# velocities = []
# for v_horizontal_sample in results["v_horizontal.v_vertical"]:
#     for v_vertical in v_horizontal_sample:
#         n_dims_v_vertical.append(v_vertical["v_horizontal.v_vertical.n_dimensions"])
#         sites_v_vertical.extend(v_vertical["v_horizontal.v_vertical.discretization"])
#         velocities.extend(v_vertical["v_horizontal.v_vertical.velocity"])

# fig, axes = plt.subplots(1, 3, figsize=(10, 5))
# axes[0].hist(n_dims_v_horizontal, bins=10, ec="w")
# axes[0].set_title("v_horizontal")
# axes[1].hist(n_dims_v_vertical, bins=10, ec="w")
# axes[1].set_title("v_vertical")
# axes[2].hist(velocities, bins=20, ec="w")
# axes[2].set_title("velocity")
# fig.tight_layout()
# fig.savefig("11_nested_voronoi")
