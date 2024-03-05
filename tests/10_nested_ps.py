import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


# define parameter space
v0 = bb.prior.UniformPrior("v0", 0, 100, 1)
v1 = bb.prior.GaussianPrior("v1", 50, 10, 1)
ps3 = bb.parameterization.ParameterSpace(
    name="ps3",
    n_dimensions=None,
    n_dimensions_min=1,
    n_dimensions_max=10,
    parameters=[v0]
)
ps2 = bb.parameterization.ParameterSpace(
    name="ps2",
    n_dimensions=None,
    n_dimensions_min=1,
    n_dimensions_max=10,
    parameters=[v0, v1]
)
ps1 = bb.parameterization.ParameterSpace(
    name="ps1",
    n_dimensions=None,
    n_dimensions_min=1,
    n_dimensions_max=10,
    parameters=[ps3]
)
ps0 = bb.parameterization.ParameterSpace(
    name="ps0",
    n_dimensions=None,
    n_dimensions_min=1,
    n_dimensions_max=10,
    parameters=[ps1, ps2, v0]
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
inversion.run(
    sampler=None, 
    n_iterations=1_000_000, 
    burnin_iterations=0, 
    save_every=3, 
    print_every=200, 
)

# get results and plot
results = inversion.get_results()
n_dims_ps0 = results["ps0.n_dimensions"]
ps0_v0 = np.concatenate(results["ps0.v0"])
n_dims_ps1 = []
n_dims_ps2 = []
n_dims_ps3 = []
ps2_v0 = []
ps2_v1 = []
ps3_v0 = []
for ps1_sample in results["ps0.ps1"]:
    for ps1 in ps1_sample:
        n_dims_ps1.append(ps1["ps0.ps1.n_dimensions"])
        for ps3 in ps1["ps0.ps1.ps3"]:
            n_dims_ps3.append(ps3["ps0.ps1.ps3.n_dimensions"])
            ps3_v0.extend(ps3["ps0.ps1.ps3.v0"])
            
for ps2_sample in results["ps0.ps2"]:
    for ps2 in ps2_sample:
        n_dims_ps2.append(ps2["ps0.ps2.n_dimensions"])
        ps2_v0.extend(ps2["ps0.ps2.v0"])
        ps2_v1.extend(ps2["ps0.ps2.v1"])

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes[0, 0].hist(n_dims_ps0, bins=10, ec="w")
axes[0, 0].set_title("ps0")
axes[0, 1].hist(n_dims_ps1, bins=10, ec="w")
axes[0, 1].set_title("ps1")
axes[0, 2].hist(n_dims_ps2, bins=10, ec="w")
axes[0, 2].set_title("ps2")
axes[0, 3].hist(n_dims_ps3, bins=10, ec="w")
axes[0, 3].set_title("ps3")
axes[1, 0].hist(ps0_v0, bins=20, ec="w")
axes[1, 0].set_title("ps0.v0")
axes[1, 1].hist(ps2_v0, bins=20, ec="w")
axes[1, 1].set_title("ps2.v0")
axes[1, 2].hist(ps2_v1, bins=20, ec="w")
axes[1, 2].set_title("ps2.v1")
axes[1, 3].hist(ps3_v0, bins=20, ec="w")
axes[1, 3].set_title("ps3.v0")
fig.tight_layout()
fig.savefig("10_nested_ps")
