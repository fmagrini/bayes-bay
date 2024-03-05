import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter space
ps2 = bb.parameterization.ParameterSpace(
    name="ps2", 
    n_dimensions=None, 
    n_dimensions_min=1,
    n_dimensions_max=10,
    parameters=None
)
ps1 = bb.parameterization.ParameterSpace(
    name="ps1",
    n_dimensions=None,
    n_dimensions_min=1,
    n_dimensions_max=10,
    parameters=[ps2]
)
ps0 = bb.parameterization.ParameterSpace(
    name="ps0",
    n_dimensions=None,
    n_dimensions_min=1,
    n_dimensions_max=10,
    parameters=[ps1]
)
parameterization = bb.parameterization.Parameterization(ps0)

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
inversion.set_perturbation_funcs([inversion.perturbation_funcs[0], inversion.perturbation_funcs[2]])
inversion.run(
    sampler=None, 
    n_iterations=100_000, 
    burnin_iterations=0, 
    save_every=3, 
    print_every=200, 
)

# get results and plot
results = inversion.get_results()
n_dims_ps0 = results["ps0.n_dimensions"]
n_dims_ps1 = []
n_dims_ps2 = []
for ps1_sample in results["ps0.ps1"]:
    for ps1 in ps1_sample:
        n_dims_ps1.append(ps1["ps0.ps1.n_dimensions"])
        for ps2 in ps1["ps0.ps1.ps2"]:
            n_dims_ps2.append(ps2["ps0.ps1.ps2.n_dimensions"])

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].hist(n_dims_ps0, bins=10, ec="w")
axes[0].set_title("ps0")
axes[1].hist(n_dims_ps1, bins=10, ec="w")
axes[1].set_title("ps1")
axes[2].hist(n_dims_ps2, bins=10, ec="w")
axes[2].set_title("ps2")
fig.tight_layout()
fig.savefig("12_nested_ps2")
