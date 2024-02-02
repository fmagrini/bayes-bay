import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb


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
m0 = bb.parameters.UniformParameter("m0", -100, 100, 5)
m1 = bb.parameters.UniformParameter("m1", -50, 50, 5)
m2 = bb.parameters.UniformParameter("m2", -20, 20, 3)
m3 = bb.parameters.UniformParameter("m3", -10, 10, 2)

# define parameterization
param_space = bb.parameterization.ParameterSpace(
    name="my_param_space", 
    n_dimensions=1, 
    parameters=[m0, m1, m2, m3], 
)
parameterization = bb.parameterization.Parameterization(param_space)

# define forward function
def my_fwd(state: bb.State) -> np.ndarray:
    m = [state["my_param_space"][f"m{i}"] for i in range(N_DIMS)]
    return np.squeeze(fwd_operator @ m)
fwd_functions = [my_fwd]

# define data target
targets = [bb.Target("my_data", y_noisy, std_min=0, std_max=100, std_perturb_std=5)]

log_likelihood = bb.LogLikelihood(targets, fwd_functions)

# run the sampling
inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    log_likelihood=log_likelihood, 
    n_chains=40, 
    n_cpus=40, 
)
inversion.run(
    sampler=None, 
    n_iterations=100_000, 
    burnin_iterations=10_000, 
    save_every=500, 
    print_every=500, 
)

# get results and plot
results = inversion.get_results()
coefficients_samples = np.squeeze(np.array([results[f"my_param_space.m{i}"] for i in range(N_DIMS)]))
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
all_y_pred = np.zeros((coefficients_samples.shape[1], len(y)))
for i, coefficients in enumerate(coefficients_samples.T):
    y_pred = fwd_operator @ coefficients
    all_y_pred[i,:] = y_pred
    if i == 0:
        axes[0].plot(DATA_X, y_pred, c="gray", lw=0.05, label="Predicted data from samples")
    else:
        axes[0].plot(DATA_X, y_pred, c="gray", lw=0.05)
axes[0].plot(DATA_X, np.median(all_y_pred, axis=0), c="blue", label="Median predicted sample")
axes[0].plot(DATA_X, y, c="orange", label="Noise-free data from true model")
axes[0].scatter(DATA_X, y_noisy, c="purple", label="Noisy data used for inference", zorder=3)
axes[0].legend()
axes[1].hist(results["my_data.std"], bins=100, ec="w")
fig.savefig("linear_regression_hier_samples_pred")
