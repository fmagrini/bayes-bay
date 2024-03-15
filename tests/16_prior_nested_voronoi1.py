import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter space
uniform_param = bb.prior.UniformPrior(name="uniform_param", 
                                      vmin=-1, 
                                      vmax=1, 
                                      perturb_std=0.1,
                                      perturb_std_birth=0.6)
voronoi1 = bb.discretization.Voronoi1D(
    name="my_voronoi1", 
    vmin=0, 
    vmax=1, 
    perturb_std=0.1, 
    n_dimensions=None, 
    n_dimensions_min=1, 
    n_dimensions_max=10, 
    parameters=[uniform_param], 
    # birth_from="prior"
    # parameters=[]
)
voronoi2 = bb.discretization.Voronoi1D(
    name="my_voronoi2", 
    vmin=0, 
    vmax=1, 
    perturb_std=0.1, 
    n_dimensions=None, 
    n_dimensions_min=1, 
    n_dimensions_max=10, 
    parameters=[voronoi1], 
    # birth_from="prior"
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
inversion.run(
    sampler=None, 
    n_iterations=500_000, 
    burnin_iterations=0, 
    save_every=100, 
    print_every=10000, 
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
# fig.savefig("16_nested_voronoi1")
