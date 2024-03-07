import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter: Gaussian
laplace_param = bb.prior.LaplacePrior("laplace_param", 0, 1, 0.1)

# define parameter space
parameterization = bb.parameterization.Parameterization(
    bb.parameterization.ParameterSpace(
        name="my_param_space", 
        n_dimensions=None, 
        n_dimensions_min=1, 
        n_dimensions_max=10, 
        parameters=[laplace_param], 
    )
)

# define dumb log likelihood
targets = [bb.Target("dummy_data", np.array([1], dtype=float), 1)]
def forward():
    return np.array([1], dtype=float)

fwd_functions = [lambda _: np.array([1], dtype=float)]
log_likelihood = bb.LogLikelihood(targets, forward)

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
param_values = results["my_param_space.laplace_param"]
fig, axes = plt.subplots(1, 2)
axes[0].hist(n_dims, bins=10, ec="w")
axes[1].hist(np.concatenate(param_values), bins=50, ec="w")
# fig.savefig("4_prior_param_space_laplace_param")
plt.show()
