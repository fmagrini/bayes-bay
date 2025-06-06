#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 21:20:02 2025

@author: fabrizio
"""


import bayesbay as bb

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rc("text", usetex=True)
plt.rc("font", family="serif")


m_true = 10
sigma_true = 0.5
d = np.random.normal(m_true, sigma_true)

m_prior = bb.prior.UniformPrior(name="m", vmin=5, vmax=15, perturb_std=0.4)

print(f"a = {m_prior.vmin}")
print(f"b = {m_prior.vmax}")
print(f"p(m=5) = {round(np.exp(m_prior.log_prior(5)), 10)}")
print(f"p(m=10) = {round(np.exp(m_prior.log_prior(10)), 10)}")
print(f"p(m=15) = {round(np.exp(m_prior.log_prior(10)), 10)}")
print(f"p(m=20) = {round(np.exp(m_prior.log_prior(20)), 10)}")


param_space = bb.parameterization.ParameterSpace(
    name="m_space",
    n_dimensions=1,
    parameters=[m_prior],
)
print(param_space)

parameterization = bb.parameterization.Parameterization(param_space)
print(parameterization)

state = parameterization.initialize()
print(state)


def fwd_function(state: bb.State) -> np.ndarray:
    m = state["m_space"]["m"]
    return m


print(f"d_pred: {fwd_function(state)}")
print(f"d_obs - d_pred: {d - fwd_function(state)}")

# %%

from functools import partial


# Define a log-likelihood function that will be used to initialise LogLikelihood
def log_like_func(state, dobs, sigma):
    dpred = fwd_function(state)
    r = dobs - dpred
    mahalanobis_dist = (r / sigma) ** 2
    log_likelihood = -0.5 * mahalanobis_dist - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    # Store the computed value of log-likelihood in state
    state.save_to_extra_storage("log_likelihood", log_likelihood)
    return log_likelihood


wrapped_log_like_func = partial(log_like_func, dobs=d, sigma=sigma_true)

log_likelihood = bb.likelihood.LogLikelihood(log_like_func=wrapped_log_like_func)
# %%


inversion = bb.BayesianInversion(
    log_likelihood=log_likelihood, parameterization=parameterization, n_chains=10
)
inversion.run(
    n_iterations=75_000,
    burnin_iterations=25_000,
    save_every=50,
    verbose=False,
)
for chain in inversion.chains:
    chain.print_statistics()


results = inversion.get_results(concatenate_chains=True)

m_samples = np.array(results["m_space.m"])
log_likelihood_samples = np.array(results["log_likelihood"])

x = np.linspace(m_true - 4 * sigma_true, m_true + 4 * sigma_true, 1000)
pdf = norm.pdf(x, loc=m_true, scale=sigma_true)

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
ax.plot(x, pdf, label=r"$\mathcal{N}(m_{true}, \sigma^2)$", color="k")
ax.hist(m_samples, bins=50, color="dodgerblue", ec="w", density=True)
ax.axvline(x=m_true, color="r", linestyle="--", label=r"$m_{true}$", lw=2)
ax.axvline(x=d, color="k", linestyle="--", label=r"$d_{obs}$", lw=2)
ax.legend()
ax.set_xlabel("$m$")
ax.set_ylabel("PDF")
plt.show()


fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
ax.hist(
    np.exp(log_likelihood_samples), bins=50, color="dodgerblue", ec="w", density=True
)
ax.set_xlabel("$p(d | m)$")
ax.set_ylabel("PDF")
plt.show()
