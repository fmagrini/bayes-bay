import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter space
parameterization = bb.parameterization.Parameterization(
    bb.parameterization.ParameterSpace(
        name="my_param_space", 
        n_dimensions=None, 
        n_dimensions_min=1, 
        n_dimensions_max=10, 
        parameters=[], 
    )
)

# define dumb log likelihood
targets = [bb.Target("dumb_data", np.array([1], dtype=float), 1)]
fwd_functions = [lambda _: np.array([1], dtype=float)]

# run the sampler
inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    targets=targets, 
    fwd_functions=fwd_functions, 
    n_chains=1, 
    n_cpus=1, 
)
inversion.run(
    sampler=None, 
    n_iterations=500_000, 
    burnin_iterations=0, 
    save_every=3, 
    print_every=200, 
)

# get results and plot
results = inversion.get_results()
n_dims = results["my_param_space.n_dimensions"]
sites = results["my_param_space.discretization"]
fig, ax = plt.subplots()
ax.hist(n_dims, bins=10, ec="w")
fig.savefig("0_prior_param_space")
