import numpy as np
import matplotlib.pyplot as plt
import bayesbridge as bb


# define parameter: uniform
uniform_param = bb.parameters.UniformParameter("uniform_param", -1, 1, 0.1)

# define parameter space
parameterization = bb.parameterization.Parameterization(
    bb.parameterization.ParameterSpace(
        name="my_param_space",  
        n_dimensions=None, 
        n_dimensions_min=1, 
        n_dimensions_max=10, 
        parameters=[uniform_param], 
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
    save_every=200, 
    print_every=200, 
)

# get results and plot
results = inversion.get_results()
n_dims = results["my_param_space.n_dimensions"]
param_values = results["uniform_param"]
fig, axes = plt.subplots(1, 2)
axes[0].hist(n_dims, bins=10, ec="w")
axes[1].hist(np.concatenate(param_values), bins=20, ec="w")
fig.savefig("2_prior_param_space_uniform_param")