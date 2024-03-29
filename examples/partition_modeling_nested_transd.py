#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:29:47 2024

@author: fabrizio
"""

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bayesbay.discretization import Voronoi1D
from bayesbay.prior import UniformPrior
from bayesbay.parameterization import Parameterization, ParameterSpace
from bayesbay import Target, LogLikelihood, BayesianInversion
np.random.seed(30)
plt.rcParams['font.family'] = 'FreeSans'
plt.rcParams['font.size'] = 12 


X_DATA = np.linspace(-10, 10, 200)
NOISE_STD = 1.5


# @np.vectorize
# def piecewise_function(x):
#     if x <= 1:
#         return 1.
#     elif 1 < x <= 2.5:
#         return 6*x
#     elif 2.5 < x <= 4:
#         return 0
#     elif 4 < x <= 6.:
#         return -5*x + 10
#     elif 6 < x <= 8:
#         return 2*x - 5
#     elif 8 < x <= 10:
#         return -5.


@np.vectorize
def piecewise_function(x):
    if x <= -7:
        return 0.
    elif -7 < x <= -3.:
        return 1.5*x + 5.
    elif -3 < x <= 3.:
        return x**2
    elif 3. < x <= 7.:
        return -1.5*x + 5.
    elif 7 < x <= 10.:
        return 0


def get_segment_index(x):
    if x <= -7:
        return 0
    elif -7 < x <= -3.:
        return 1
    elif -3 < x <= 3.:
        return 2
    elif 3. < x <= 7.:
        return 3
    elif 7 < x <= 10.:
        return 4


def get_true_segment_dim(isegment):
    if isegment == 0:
        return 0
    elif isegment == 1:
        return 1
    elif isegment == 2:
        return 2
    elif isegment == 3:
        return 1
    elif isegment == 4:
        return 0


def fwd_function(state):
    voro = state['voronoi']
    x_nuclei = voro['discretization']
    x_extents = Voronoi1D.compute_cell_extents(x_nuclei, lb=-10)
    x1 = x_extents[0] - 10
    indexes = []
    indexes.extend([0] * np.flatnonzero(X_DATA <= x1).size)
    for i, extent in enumerate(x_extents[1:], 1):
        if extent == 0:
            idx_size = np.flatnonzero(X_DATA > x1).size
        else:
            x2 = x1 + x_extents[i]
            idx_size = np.flatnonzero((X_DATA > x1) & (X_DATA <= x2)).size
            x1 = x2
        indexes.extend([i] * idx_size)
    
    indexes = np.array(indexes)
    d_pred = np.zeros(X_DATA.shape)
    poly_space = voro['poly_space']
    for ipoly in np.unique(indexes):
        poly = np.poly1d(poly_space[ipoly]['y'])
        idata = np.flatnonzero(indexes == ipoly)
        x_data = X_DATA[idata]
        d_pred[idata] = poly(x_data)

    return d_pred
    

#%%



d_obs = piecewise_function(X_DATA) + np.random.normal(0, NOISE_STD, X_DATA.size)

plt.figure(figsize=(10, 6))
plt.plot(X_DATA, piecewise_function(X_DATA), 'gray', lw=3, label='True model')
plt.plot(X_DATA, d_obs, 'ro', markeredgecolor='k', label='Obs. data')
plt.legend()
plt.grid()
plt.show()

#%%
y = UniformPrior('y', vmin=-15, vmax=15, perturb_std=2)
poly_space = ParameterSpace(name='poly_space',
                            n_dimensions=None,
                            n_dimensions_min=1,
                            n_dimensions_max=5,
                            parameters=[y],
                            n_dimensions_init_range=0.5
                            )
voronoi = Voronoi1D(
    name="voronoi", 
    vmin=-10,
    vmax=10,
    perturb_std=0.75,
    n_dimensions=None, 
    n_dimensions_min=2,
    n_dimensions_max=40,
    parameters=[poly_space], 
    birth_from='prior'
)
parameterization = Parameterization(voronoi)


target = Target("d_obs", 
                d_obs, 
                std_min=0, 
                std_max=10, 
                std_perturb_std=1)

log_likelihood = LogLikelihood(targets=target, fwd_functions=fwd_function)

inversion = BayesianInversion(
    parameterization=parameterization, 
    log_likelihood=log_likelihood,
    n_chains=10
)
#%%
inversion.run(
    sampler=None, 
    n_iterations=600_000, 
    burnin_iterations=300_000, 
    save_every=100,
    verbose=False
)


#%%

results = inversion.get_results(concatenate_chains=True)
dpred = results['d_obs.dpred']
median = np.median(dpred, axis=0)
perc_range = 10, 90
percentiles = np.percentile(dpred, perc_range, axis=0)


#%%
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])

ax1 = plt.subplot(gs[0, :]) # First row, spans all columns
ax2 = plt.subplot(gs[1, 0]) # Second row, first column
ax3 = plt.subplot(gs[1, 1]) # Second row, second column

ax1.plot(X_DATA, piecewise_function(X_DATA), 'gray', lw=3, label='True model')
ax1.plot(X_DATA, median, 'b', label='Ensemble median')
ax1.fill_between(X_DATA, 
                *percentiles, 
                color='b', 
                alpha=0.2, 
                label='Uncertainty (%s-%sth perc.)'%(perc_range))
ax1.plot(X_DATA, d_obs, 'ro', markeredgecolor='k', label='Obs. data')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend(framealpha=0.5)
ax1.grid()
ax1.set_title('Inferred model')


ndim_min, ndim_max = voronoi._n_dimensions_min, voronoi._n_dimensions_max
ax2.fill_between([ndim_min, ndim_max], 
                 1 / (ndim_max - ndim_min), 
                 alpha=0.3, 
                 color='orange',
                 label='Prior')
ax2.set_xlabel('No. partitions')
ax2.hist(results['voronoi.n_dimensions'], 
         bins=np.arange(ndim_min-0.5, ndim_max+0.5), 
         color='b',
         alpha=0.8, 
         density=True, 
         ec='w', 
         label='Posterior')
ax2.axvline(x=5, color='r', lw=2, alpha=1, label='True', zorder=100)
ax2.legend(framealpha=0)
ax2.set_title('Partitions')

ax2.legend(framealpha=0.9)

ax3.fill_between([target.std_min, target.std_max], 
                 1 / (target.std_max - target.std_min), 
                 alpha=0.3, 
                 color='orange',
                 label='Prior')
ax3.hist(results['d_obs.std'], 
         color='b',
         alpha=0.8, 
         density=True, 
         bins=10, 
         ec='w', 
         label='Posterior')
ax3.axvline(x=NOISE_STD, color='r', lw=2, alpha=1, label='True', zorder=100)
ax3.set_xlabel('Noise standard deviation')
ax3.legend(framealpha=0.9)
ax3.set_title('Data noise')
plt.tight_layout()
plt.show()


samples_sites = results['voronoi.discretization']
samples_poly_space = results['voronoi.poly_space']


#%%


dims_dict = defaultdict(list)
for sites, poly_spaces in zip(samples_sites, samples_poly_space):
    for site, poly_space in zip(sites, poly_spaces):
        isegment = get_segment_index(site)
        dims_dict[isegment].append(poly_space['voronoi.poly_space.n_dimensions'])
        


#%%

fig = plt.figure(figsize=(11, 12), constrained_layout=True)

gs_main = gridspec.GridSpec(3, 
                            1, 
                            figure=fig, 
                            height_ratios=[0.5, 1, 0.5], 
                            hspace=0.25)

# Define the GridSpec for the top 8 subplots, which occupies the top half of the figure
gs_top = gridspec.GridSpecFromSubplotSpec(1, 
                                          5, 
                                          subplot_spec=gs_main[0], 
                                          wspace=0.1, 
                                          hspace=0.05)
gs_center = gridspec.GridSpecFromSubplotSpec(2, 
                                             1, 
                                             subplot_spec=gs_main[1], 
                                             height_ratios=[1, 0.5],
                                             wspace=0.05, 
                                             hspace=0.01)

gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 
                                             2, 
                                             subplot_spec=gs_main[2], 
                                             wspace=0.15, 
                                             hspace=0.1)

for i in range(5):
    row, col = divmod(i, 5)
    ax = fig.add_subplot(gs_top[row, col])
    ax.set_xticks(np.arange(0, 6))
    if i > 0:
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel('Probability Density')

    ax.hist(np.arange(1, 6), 
            bins=np.arange(0.5, 6), 
            ec='w', 
            density=True,
            fc='orange',
            alpha=0.3,
            label='Prior')
    ax.hist(np.array(dims_dict[i])-1, 
            bins=np.arange(-0.5, 4.5), 
            ec='w', 
            density=True,
            fc='royalblue',
            label='Posterior')
    true_dim = get_true_segment_dim(i)
    ax.axvline(x=true_dim, 
               ymin=0, 
               ymax=0.2, 
               color='r', 
               lw=4, 
               alpha=1, 
               zorder=100, 
               ls='-',
               label='True\nPoly Order')
    ax.set_ylim(0, 1)
    ax.text(x=0.95, 
            y=0.97, 
            s=f'Segment {i+1}', 
            ha='right', 
            va='top', 
            transform=ax.transAxes)
    ax.set_xlabel('Poly Order')
    
ax.legend(loc='center right')


ax1 = fig.add_subplot(gs_center[0, 0])
ax1.plot(X_DATA, piecewise_function(X_DATA), 'gray', lw=3, label='True model')
ax1.plot(X_DATA, median, 'royalblue', label='Posterior median')
ax1.fill_between(X_DATA, 
                *percentiles, 
                color='royalblue', 
                alpha=0.2, 
                label='Uncertainty (%s-%sth perc.)'%(perc_range))
ax1.plot(X_DATA, d_obs, 'ro', markeredgecolor='k', label='Obs. data')
ax1.set_ylabel('y')

handles, labels = ax1.get_legend_handles_labels()
handles = [handles[0], handles[3], handles[1], handles[2]]
labels = [labels[0], labels[3], labels[1], labels[2]]

ax1.legend(handles, labels, framealpha=0.5, loc='lower center', ncols=2, columnspacing=1)
ax1.grid()
ax1.set_xlim(-10, 10)
ax1.tick_params(labelbottom=False)

ax2 = fig.add_subplot(gs_bottom[0, 0])

ndim_min, ndim_max = voronoi._n_dimensions_min, voronoi._n_dimensions_max

ax2.fill_between([ndim_min, ndim_max], 
                 1 / (ndim_max - ndim_min), 
                 alpha=0.3, 
                 color='orange',
                 label='Prior')
ax2.set_xlabel('No. partitions')
ax2.hist(results['voronoi.n_dimensions'], 
         bins=np.arange(ndim_min-0.5, ndim_max+0.5), 
         color='royalblue',
         alpha=0.8, 
         density=True, 
         ec='w', 
         label='Posterior')

ax2.axvline(x=5, 
            color='r', 
            ymin=0, 
            ymax=0.2, 
            lw=4, 
            alpha=1, 
            label='True\nNo. Partitions', 
            zorder=100)
ax2.legend(framealpha=0)
ax2.set_ylabel('Probability Density')

ax2.legend(framealpha=0.9)

ax3 = fig.add_subplot(gs_bottom[0, 1])
ax3.fill_between([target.std_min, target.std_max], 
                 1 / (target.std_max - target.std_min), 
                 alpha=0.3, 
                 color='orange',
                 label='Prior')
ax3.hist(results['d_obs.std'], 
         color='royalblue',
         alpha=0.8, 
         density=True, 
         bins=np.linspace(target.std_min, target.std_max, 35), 
         ec='w', 
         label='Posterior')

ax3.axvline(x=NOISE_STD, 
            ymin=0, 
            ymax=0.2, 
            color='r', 
            lw=4, 
            alpha=1, 
            label='True Noise\nStandard Deviation', 
            zorder=100)
ax3.set_xlabel('Noise standard deviation')
ax3.legend(framealpha=0.9)






ax4 = fig.add_subplot(gs_center[1, 0])
ax4.fill_between([voronoi.vmin, voronoi.vmax], 
                 1 / (voronoi.vmax - voronoi.vmin), 
                 alpha=0.3, 
                 color='orange',
                 label='Prior')
ax4 = Voronoi1D.plot_interface_hist(samples_sites, 
                                   ax=ax4, 
                                   swap_xy_axes=False,
                                   bins=75,
                                   fc='royalblue',
                                   ec='w',
                                   label='Posterior')
ax4.axvline(x=-7, ymax=0.2, color='r', lw=4, alpha=1, zorder=100, ls='-')
ax4.axvline(x=-3, ymax=0.2, color='r', lw=4, alpha=1, zorder=100, ls='-')
ax4.axvline(x=3, ymax=0.2, color='r', lw=4, alpha=1, zorder=100, ls='-')
ax4.axvline(x=7, ymax=0.2, color='r', lw=4, alpha=1, zorder=100, ls='-', label='True\nInterface Position')

handles, labels = ax4.get_legend_handles_labels()
handles[1], handles[2] = handles[2], handles[1]
labels[1], labels[2] = labels[2], labels[1]

ax4.set_ylabel('Probability\nDensity')
ax4.set_xlabel('x')
ax4.set_xlim(-10, 10)
ax4.grid()
ax4.legend(handles, labels)

plt.show()



