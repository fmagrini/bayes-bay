import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb
np.random.seed(30)


# DIMENSIONS AND TRUE COEFFICIENTS
N_DIMS = 4
M0, M1, M2, M3 = 20, -10, -3, 1

# DATA AND NOISE
N_DATA = 15
DATA_X = np.linspace(-5, 10, N_DATA)
DATA_NOISE_STD = 20

# generate synthetic data
fwd_operator = np.vander(DATA_X, N_DIMS, True)
y = fwd_operator @ [M0, M1, M2, M3]
y_noisy = y + np.random.normal(0, DATA_NOISE_STD, y.shape)

fig, ax = plt.subplots(dpi=200)
ax.set_title('Synthetic data')
ax.plot(DATA_X, y, 'k', label='Predicted data (true model)')
ax.plot(DATA_X, y_noisy, 'ro', label='Noisy observations')
ax.grid()
ax.legend()
plt.show()

# define parameters
# m0 = bb.parameters.UniformParameter(name="m0", vmin=0, vmax=40, perturb_std=2)
m0 = bb.parameters.GaussianParameter(name="m0", mean=20, std=1, perturb_std=0.5)
m1 = bb.parameters.UniformParameter(name="m1", vmin=-13, vmax=-7, perturb_std=0.4)
m2 = bb.parameters.UniformParameter(name="m2", vmin=-10, vmax=4, perturb_std=0.5)
m3 = bb.parameters.GaussianParameter(name="m3", mean=1, std=0.1, perturb_std=0.1)

# define parameterization
param_space = bb.parameterization.ParameterSpace(
    name="my_param_space", 
    n_dimensions=1, 
    parameters=[m0, m1, m2, m3], 
)
parameterization = bb.parameterization.Parameterization(param_space)

# define forward function
def fwd_function(state: bb.State) -> np.ndarray:
    m = [state["my_param_space"][f"m{i}"] for i in range(N_DIMS)]
    return np.squeeze(fwd_operator @ m)

# define data target
target = bb.Target("my_data", y_noisy, 1/DATA_NOISE_STD**2)

log_likelihood = bb.LogLikelihood(target, fwd_function)

# run the sampling
inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    log_likelihood=log_likelihood, 
    n_chains=10, 
    n_cpus=10, 
)
inversion.run(
    sampler=None, 
    n_iterations=100_000, 
    burnin_iterations=10_000, 
    save_every=500, 
    print_every=5000, 
)

results = inversion.get_results()

coefficients_samples = np.squeeze(np.array([results[f"my_param_space.m{i}"] for i in range(N_DIMS)]))

fig, ax = plt.subplots()
y_pred = np.array(results["dpred"])
ax.plot(DATA_X, y_pred[0], c='gray', lw=0.3, alpha=0.3, label="Inferred data")
ax.plot(DATA_X, y_pred[1:].T, c='gray', lw=0.3, alpha=0.3)
ax.plot(DATA_X, y, c='orange', label='Predicted data (true model)')
ax.plot(DATA_X, np.median(y_pred, axis=0), c="blue", label='Median inferred samples')
ax.scatter(DATA_X, y_noisy, c='r', label='Noisy observations', zorder=3)
ax.legend()
plt.show()

import arviz as az

fig, axes = plt.subplots(4, 4, figsize=(10,8))
results_coefficients = {k: v for k, v in results.items() if k != "dpred"}
_ = az.plot_pair(
    results_coefficients,
    marginals=True,
    reference_values={'m0': M0, 'm1': M1, 'm2': M2, 'm3': M3},
    reference_values_kwargs={'color': 'yellow',
                             'ms': 10},
    kind='kde',
    kde_kwargs={
        'hdi_probs': [0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
        'contourf_kwargs': {'cmap': 'Blues'},
        },
    ax=axes,
    textsize=10
    )
plt.tight_layout()

