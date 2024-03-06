import random
import math
import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter: customly defined uniform parameter
custom_param = bb.prior.CustomPrior(
    "custom_param",
    lambda v: - math.log(10) if 0 <= v <= 10 else float("-inf"), 
    lambda p: \
        np.random.uniform(0,10,len(p)) \
            if (not np.isscalar(p) and p is not None) \
                else random.uniform(0,10), 
    1, 
)

# define parameter space
parameterization = bb.parameterization.Parameterization(
    bb.parameterization.ParameterSpace(
        name="my_param_space", 
        n_dimensions=None, 
        n_dimensions_min=1, 
        n_dimensions_max=10, 
        parameters=[custom_param], 
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
param_values = results["my_param_space.custom_param"]
fig, axes = plt.subplots(1, 2)
axes[0].hist(n_dims, bins=10, ec="w")
axes[1].hist(np.concatenate(param_values), bins=20, ec="w")
fig.savefig("6_prior_param_space_custom_param")
