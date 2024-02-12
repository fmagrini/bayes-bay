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
        parameters=[
            uniform_param, 
            gaussian_param, 
            custom_param, 
        ], 
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
uniform_param = results["uniform_param"]
gaussian_param = results["gaussian_param"]
custom_param = results["custom_param"]
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
axes[0].hist(n_dims, bins=10, ec="w")
axes[1].hist(np.concatenate(uniform_param), bins=20, ec="w")
axes[2].hist(np.concatenate(gaussian_param), bins=20, ec="w")
axes[3].hist(np.concatenate(custom_param), bins=20, ec="w")
fig.savefig("8_prior_param_space_multiple_params")
