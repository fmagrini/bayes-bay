import random
import math
import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter: uniform, gaussian, custom
uniform_param = bb.prior.UniformPrior("uniform_param", -1, 1, 0.1)
gaussian_param = bb.prior.GaussianPrior("gaussian_param", 0, 1, 0.1)
custom_param = bb.prior.CustomPrior(
    "custom_param",
    lambda v: - math.log(10) if 0 <= v <= 10 else float("-inf"), 
    lambda _: random.uniform(0,10), 
    1, 
)

# define parameter space
parameterization = bb.parameterization.Parameterization(
    bb.parameterization.ParameterSpace(
        name="my_param_space", 
        n_dimensions=None, 
        n_dimensions_min=1, 
        n_dimensions_max=10, 
        parameters=[
            uniform_param, 
            gaussian_param, 
            custom_param, 
        ], 
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
uniform_param = results["my_param_space.uniform_param"]
gaussian_param = results["my_param_space.gaussian_param"]
custom_param = results["my_param_space.custom_param"]
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
axes[0].hist(n_dims, bins=10, ec="w")
axes[1].hist(np.concatenate(uniform_param), bins=20, ec="w")
axes[2].hist(np.concatenate(gaussian_param), bins=20, ec="w")
axes[3].hist(np.concatenate(custom_param), bins=20, ec="w")
fig.savefig("8_prior_param_space_multiple_params")
