#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 23:33:37 2024

@author: fabrizio
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import cartopy.crs as ccrs
from seislib.tomography import SeismicTomography
import seislib.colormaps as scm

from bayesbay.discretization import Voronoi2D
from bayesbay.prior import UniformPrior
import bayesbay as bb

np.random.seed(10)


def get_sources_and_receivers(n_sources_per_side=10, n_receivers=50):
    ns = n_sources_per_side
    dx = (tomo.grid.lonmax - tomo.grid.lonmin) / ns
    dy = (tomo.grid.latmax - tomo.grid.latmin) / ns
    sources_x = list(np.arange(tomo.grid.lonmin + dx/2, tomo.grid.lonmax, dx))
    sources_y = list(np.arange(tomo.grid.latmin + dy/2, tomo.grid.lonmax, dy))
    sources = np.column_stack((
        sources_x + [tomo.grid.lonmax]*ns + sources_x + [tomo.grid.lonmin]*ns,
        [tomo.grid.latmin]*ns + sources_y + [tomo.grid.latmax]*ns + sources_y
        ))
    receivers = np.random.uniform(
        [tomo.grid.latmin, tomo.grid.lonmin], 
        [tomo.grid.latmax, tomo.grid.lonmax], 
        (n_receivers, 2)
        )
    return sources, receivers


def add_data_coords(tomo, sources, receivers):
    data_coords = np.zeros((sources.shape[0] * receivers.shape[0], 4))
    for icoord, (isource, ireceiver) in enumerate(
            np.ndindex((sources.shape[0], receivers.shape[0]))
            ):
        data_coords[icoord] = np.concatenate(
            (sources[isource], receivers[ireceiver])
            )        
    tomo.data_coords = data_coords
    

def compute_jacobian(tomo):
    tomo.compile_coefficients()
    jacobian = scipy.sparse.csr_matrix(tomo.A)
    del tomo.A
    return jacobian


def _forward(kdtree, vel):
    nearest_neighbors = kdtree.query(grid_points)[1]
    interp_vel = vel[nearest_neighbors]
    return interp_vel, jacobian @ (1 / interp_vel)


def forward(state):
    voronoi = state["voronoi"]
    kdtree = voronoi.load_from_cache('kdtree')
    interp_vel, d_pred = _forward(kdtree, voronoi.get_param_values('vel'))
    state.save_to_extra_storage('interp_vel', interp_vel)
    return d_pred


tomo = SeismicTomography(
    cell_size=0.005, 
    lonmin=-1, 
    lonmax=1, 
    latmin=-1, 
    latmax=1,
    regular_grid=True
    )
grid_points = np.column_stack(tomo.grid.midpoints_lon_lat())
vel_true = tomo.checkerboard(
    ref_value=3, 
    kx=3, 
    ky=3,
    lonmin=tomo.grid.lonmin,
    lonmax=tomo.grid.lonmax,
    latmin=tomo.grid.latmin,
    latmax=tomo.grid.latmax,
    anom_amp=0.3
    )(*grid_points.T)


sources, receivers = get_sources_and_receivers(n_sources_per_side=10, 
                                               n_receivers=50)
add_data_coords(tomo, sources, receivers)
jacobian = compute_jacobian(tomo)
d_obs = jacobian @ (1/vel_true)


vel = UniformPrior('vel', vmin=2, vmax=4, perturb_std=0.1)
voronoi = Voronoi2D(
    name='voronoi', 
    vmin=[tomo.grid.lonmin, tomo.grid.latmin], 
    vmax=[tomo.grid.lonmax, tomo.grid.latmax], 
    perturb_std=0.05, 
    n_dimensions_min=50, 
    n_dimensions_max=1500, 
    parameters=[vel], 
    compute_kdtree=True)
parameterization = bb.parameterization.Parameterization(voronoi)

target = bb.Target('d_obs', 
                   d_obs, 
                   std_min=0, 
                   std_max=0.01, 
                   std_perturb_std=0.001,
                   noise_is_correlated=False)
log_likelihood = bb.LogLikelihood(targets=target, fwd_functions=forward)
inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    log_likelihood=log_likelihood,
    n_chains=10
)
#%%
inversion.run(
    sampler=None, 
    n_iterations=15_000, 
    burnin_iterations=5_000, 
    save_every=200, 
    verbose=True,
    print_every=1_000
)

results = inversion.get_results()
inferred_vel = np.mean(results['interp_vel'], axis=0)

fig, ax = plt.subplots()
img = ax.tricontourf(*grid_points.T, 
                     vel_true, 
                     levels=50, 
                     cmap=scm.roma,
                     vmin=vel_true.min(),
                     vmax=vel_true.max(),
                     extend='both')
ax.plot(*sources.T, 'r*')
ax.plot(*receivers.T, 'k^')
ax.set_xlim(grid_points[:,0].min(), grid_points[:,0].max())
ax.set_ylim(grid_points[:,1].min(), grid_points[:,1].max())
cbar = fig.colorbar(img, ax=ax, aspect=35, pad=0.02)
cbar.set_label('Velocity [km/s]')
ax.set_title('True model')
plt.show()

fig, ax = plt.subplots()
img = ax.tricontourf(*grid_points.T, 
                     inferred_vel, 
                     levels=50, 
                     cmap=scm.roma,
                     vmin=vel_true.min(),
                     vmax=vel_true.max(),
                     extend='both')
ax.set_xlim(grid_points[:,0].min(), grid_points[:,0].max())
ax.set_ylim(grid_points[:,1].min(), grid_points[:,1].max())
cbar = fig.colorbar(img, ax=ax, aspect=35, pad=0.02)
ax.set_title('Inferred (average) model')
cbar.set_label('Velocity [km/s]')
plt.show()


