import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter space
parameterization = bb.parameterization.Parameterization(
    bb.parameterization.ParameterSpace(
        name="ps_outer", 
        n_dimensions=None, 
        n_dimensions_min=1, 
        n_dimensions_max=10, 
        parameters=[
            bb.parameterization.ParameterSpace(
                name="ps_inner", 
                n_dimensions=None,
                n_dimensions_min=1,
                n_dimensions_max=10,
                parameters=[bb.prior.UniformPrior("x", 0, 100, 1)]
            )
        ], 
    )
)

# define dumb log likelihood
targets = [bb.Target("dumb_data", np.array([1], dtype=float), 1)]
fwd_functions = [lambda _: np.array([1], dtype=float)]
log_likelihood = bb.LogLikelihood(targets, fwd_functions)

# run the sampler
inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    log_likelihood=log_likelihood,  
    n_chains=1, 
)
# inversion.run(
#     sampler=None, 
#     n_iterations=500_000, 
#     burnin_iterations=0, 
#     save_every=3, 
#     print_every=200, 
# )
