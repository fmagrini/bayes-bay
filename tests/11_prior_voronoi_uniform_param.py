import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter: uniform
uniform_param = bb.prior.UniformPrior("uniform_param", -1, 1, 0.1)

# define parameter space
parameterization = bb.parameterization.Parameterization(
    bb.discretization.Voronoi1D(
        name="my_voronoi", 
        vmin=0, 
        vmax=1, 
        perturb_std=10, 
        n_dimensions=None, 
        n_dimensions_min=1, 
        n_dimensions_max=10, 
        parameters=[uniform_param], 
    )
)

# define dummy log likelihood
targets = [bb.Target("dummy_data", np.array([1], dtype=float), 1)]
fwd_functions = [lambda _: np.array([1], dtype=float)]
log_likelihood = bb.LogLikelihood(targets, fwd_functions)

# run the sampler
inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    log_likelihood=log_likelihood,  
    n_chains=1, 
)
inversion.set_perturbation_funcs(inversion.perturbation_funcs[0].perturbation_funcs)
inversion.run(
    sampler=None, 
    n_iterations=500_000, 
    burnin_iterations=0, 
    save_every=200, 
    print_every=200, 
)

# get results and plot
results = inversion.get_results()
n_dims = results["my_voronoi.n_dimensions"]
sites = results["my_voronoi.discretization"]
param_values = results["my_voronoi.uniform_param"]
fig, axes = plt.subplots(1, 3)
axes[0].hist(n_dims, bins=10, ec="w")
axes[1].hist(np.concatenate(sites), bins=50, ec="w", orientation="horizontal")
axes[2].hist(np.concatenate(param_values), bins=20, ec="w")
fig.savefig("11_prior_voronoi_uniform_param")
