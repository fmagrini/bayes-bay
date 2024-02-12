#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:24:52 2024

@author: fabrizio


NOTES: 
    - KD-TREE DOES NOT GET UPDATED AFTER VORONOI PERTURB. ALWAYS REBUILD THE
    TREE
    - RIDGE_POINTS DOES NOT GET UPDATED AFTER CHANGE IN VORONOI.POINTS
"""



from collections import defaultdict
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import cKDTree
from scipy.spatial import KDTree


def perturb_voronoi(nuclei):
    irandom = np.random.randint(0, nuclei.shape[0])
    old_point = nuclei[irandom].copy()
    new_point = old_point + np.random.normal(0, 0.2)
    while np.any((new_point > 1) | (new_point < 0)):
        new_point = old_point + np.random.normal(0, 0.2)
    return irandom, old_point, new_point


def get_neighbors_dict(ridge_points):
    neighbors = defaultdict(list)
    for i, j in ridge_points:
        neighbors[i].append(j)
        neighbors[j].append(i)
    return neighbors


def get_voro_to_grid_dict(nuclei, query_points):
    tree = cKDTree(nuclei, copy_data=True)
    icell_per_point = tree.query(query_points)[1]
    voro_to_grid_dict = defaultdict(list)
    for ipoint, icell in enumerate(icell_per_point):
        voro_to_grid_dict[icell].append(ipoint)
    return voro_to_grid_dict


def map_voronoi_onto_grid(nuclei, values, grid_points, grid_shape):
    C = np.zeros(points.shape[0])
    for ivoro, ipoints in get_voro_to_grid_dict(nuclei, points).items():
        C[ipoints] = values[ivoro]
    return np.reshape(C, grid_shape)


def plot_voronoi(voronoi, ax=None, labels=True):
    if ax is None:
        fig, ax = plt.subplots()
    voronoi_plot_2d(voronoi, 
                    ax=ax, 
                    show_vertices=False, 
                    line_colors='k', 
                    line_width=0.5, 
                    line_alpha=1, 
                    show_points=False,
                    zorder=100)
    if labels:
        for i, point in enumerate(voronoi.points):
            ax.text(*point, 
                    s=i, 
                    va='center', 
                    ha='center', 
                    fontsize=10, 
                    color='r')
    return ax

#%%
# Define your Voronoi nuclei as a set of points. Example:
X, Y = np.meshgrid(np.linspace(0, 1, 250), np.linspace(0, 1, 250))
points = np.column_stack((np.ravel(X), np.ravel(Y)))

nuclei = np.random.uniform(0, 1, (15, 2))
c = np.arange(nuclei.size)  

for _ in range(20):
    irandom, old_point, new_point = perturb_voronoi(nuclei)
    nuclei[irandom] = new_point
    tree = cKDTree(nuclei, copy_data=True)
    # C1 = c[tree.query(points)[1]].reshape(X.shape)
    C = map_voronoi_onto_grid(nuclei, c, points, X.shape)
        
    vor = Voronoi(nuclei)    
    
    fig, ax = plt.subplots()
    plot_voronoi(vor, ax=ax)
    ax.pcolormesh(X, Y, C)
    ax.plot(*old_point, 'ko')
    ax.plot(*new_point, 'ks')
    # Customize the plot
    ax.set_xlim([0, 1]), ax.set_ylim([0, 1])
# plt.savefig('/home/fabrizio/Downloads/Figure 2024-02-07 164328.png', dpi=300)
    plt.show()
#%%

voro = Voronoi(np.random.uniform(0, 1, (1000, 2)))    
voro_values = np.random.normal(0, 2, voro.points.shape[0])
tree = cKDTree(voro.points, copy_data=True)

X, Y = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 1, 500))
grid_points = np.column_stack((np.ravel(X), np.ravel(Y)))
C = voro_values[tree.query(grid_points)[1]]

neighbors_dict = get_neighbors_dict(voro.ridge_points)
voro_to_grid_dict = get_voro_to_grid_dict(voro.points, grid_points)

for _ in range(10):
    iold, old_point, new_point = perturb_voronoi(voro.points)
    nuclei_new = voro.points.copy()
    nuclei_new[iold] = new_point
    voro_new = Voronoi(nuclei_new)
    assert np.all(voro_new.points[iold] == new_point)
    neighbors_dict_new = get_neighbors_dict(voro_new.ridge_points)
    neighbors_old = neighbors_dict[iold]
    neighbors_new = neighbors_dict_new[iold] + [iold]
    ineighborhood = np.unique(
        np.concatenate(
            [neighbors_dict[i] for i in neighbors_old + neighbors_new]
            )
        )
    nuclei_tmp = nuclei_new[ineighborhood]
    voro_tmp = Voronoi(nuclei_tmp)
    tree_tmp = cKDTree(voro_tmp.points, copy_data=True)
    query_points = np.concatenate([voro_to_grid_dict[i] for i in ineighborhood])
    icells_tmp = tree_tmp.query(grid_points[query_points])[1]
    icells_new = ineighborhood[icells_tmp]
    C_new = C.copy()
    C_new[query_points] = voro_values[icells_new]
    
    voro_new = Voronoi(nuclei_new)
    tree_new = cKDTree(nuclei_new, copy_data=True)
    C_new_voro = voro_values[tree_new.query(grid_points)[1]]
    assert np.all(C_new_voro == C_new)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_voronoi(voro, ax=ax)
    z = np.full(grid_points.shape[0], np.nan)
    z[query_points] = query_points
    Z = np.ma.array(z.reshape(X.shape))
    ax.pcolormesh(X, Y, Z)
    ax.plot(*old_point, 'bo')
    ax.plot(*new_point, 'ro')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.tick_params(labelleft=False, labelbottom=False)
    plt.show()

    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plot_voronoi(voro, ax=ax1, labels=False)
    ax1.pcolormesh(X, Y, C.reshape(X.shape))
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.plot(*voro.points.T, 'k.', ms=5)
    ax1.plot(*old_point, 'ro')
    ax1.tick_params(labelleft=False, labelbottom=False)
    
    plot_voronoi(voro_new, ax=ax2, labels=False)
    ax2.pcolormesh(X, Y, C_new.reshape(X.shape))
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.plot(*voro_new.points.T, 'k.', ms=5)
    ax2.plot(*new_point, 'ro')
    ax2.tick_params(labelleft=False, labelbottom=False)
    
    plot_voronoi(voro_new, ax=ax3, labels=False)
    ax3.pcolormesh(X, Y, C_new_voro.reshape(X.shape))
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.plot(*voro_new.points.T, 'k.', ms=5)
    ax3.plot(*new_point, 'ro')
    ax3.tick_params(labelleft=False, labelbottom=False)
    
    fig.suptitle(f'Pixel differences: {np.flatnonzero(C_new_voro != C_new).size}')
    plt.tight_layout()
    plt.show()
    

    






#%%

nx_grid, ny_grid = 500, 500
times_vanilla_all = []
times_optimised_all = []
n_nuclei_all = [100, 200, 400, 800, 1600, 3200, 6400, 12800]

for n_nuclei in n_nuclei_all:
    voro = Voronoi(np.random.uniform(0, 1, (n_nuclei, 2)))    
    voro_values = np.random.normal(0, 2, voro.points.shape[0])
    tree = cKDTree(voro.points, copy_data=True)
    
    X, Y = np.meshgrid(np.linspace(0, 1, nx_grid), np.linspace(0, 1, ny_grid))
    grid_points = np.column_stack((np.ravel(X), np.ravel(Y)))
    C = voro_values[tree.query(grid_points)[1]]
    
    neighbors_dict = get_neighbors_dict(voro.ridge_points)
    voro_to_grid_dict = get_voro_to_grid_dict(voro.points, grid_points)
    
    times_vanilla = []
    times_optimised = []
    
    for _ in range(10):
        iold, old_point, new_point = perturb_voronoi(voro.points)
        nuclei_new = voro.points.copy()
        nuclei_new[iold] = new_point
        
        # Vanilla
        
        t1_vanilla = time.time()
        tree_vanilla = KDTree(nuclei_new)
        C_vanilla = voro_values[tree_vanilla.query(grid_points)[1]]
        t2_vanilla = time.time()
        
        # "Optimised"
        
        t1_optimised = time.time()
        voro_new = Voronoi(nuclei_new)
        neighbors_dict_new = get_neighbors_dict(voro_new.ridge_points)
        neighbors_old = neighbors_dict[iold]
        neighbors_new = neighbors_dict_new[iold] + [iold]
        ineighborhood = np.unique(
            np.concatenate(
                [neighbors_dict[i] for i in neighbors_old + neighbors_new]
                )
            )
        nuclei_tmp = nuclei_new[ineighborhood]
        voro_tmp = Voronoi(nuclei_tmp)
        tree_tmp = cKDTree(voro_tmp.points)
        query_points = np.concatenate([voro_to_grid_dict[i] for i in ineighborhood])
        icells_tmp = tree_tmp.query(grid_points[query_points])[1]
        icells_new = ineighborhood[icells_tmp]
        C_new = C.copy()
        C_new[query_points] = voro_values[icells_new]
        t2_optimised = time.time()
        
        t_vanilla = t2_vanilla - t1_vanilla
        t_optimised = t2_optimised - t1_optimised
        times_vanilla.append(t_vanilla)
        times_optimised.append(t_optimised)
        print(f'Vanilla: {round(t_vanilla, 3)}, Opt.: {round(t_optimised, 3)}')
        
        assert np.all(C_vanilla == C_new)
    
    times_vanilla_all.append(times_vanilla)
    times_optimised_all.append(times_optimised)


plt.plot(n_nuclei_all, np.mean(times_vanilla_all, axis=1), label='Vanilla')
plt.plot(n_nuclei_all, np.mean(times_optimised_all, axis=1), label='Opt')
plt.legend()
plt.show()


    
    #%%
    
    



def test_ckdtree(random_moves, nuclei):
    nuclei = nuclei.copy()
    for move in random_moves:
        irandom = np.random.randint(0, nuclei.shape[0])
        if move == 0:
            nuclei = np.insert(nuclei, irandom, np.random.uniform(0, 1, 2), axis=0)
        elif move == 1:
            nuclei = np.delete(nuclei, irandom, axis=0)
        else:
            nuclei[irandom] += np.random.uniform(0, 0.1, 2)
        tree = cKDTree(nuclei, copy_data=True)
        neighbors = tree.query(points)[1]
    return

def test_kdtree(random_moves, nuclei):
    nuclei = nuclei.copy()
    for move in random_moves:
        irandom = np.random.randint(0, nuclei.shape[0])
        if move == 0:
            nuclei = np.insert(nuclei, irandom, np.random.uniform(0, 1, 2), axis=0)
        elif move == 1:
            nuclei = np.delete(nuclei, irandom, axis=0)
        else:
            nuclei[irandom] += np.random.uniform(0, 0.1, 2)
        tree = KDTree(nuclei, copy_data=True)
        neighbors = tree.query(points)[1]
    return

#%%
import time

moves = np.random.randint(0, 3, 100)

t1 = time.time()
test_ckdtree(moves, nuclei)
t2 = time.time()
print(t2 - t1)

t1 = time.time()
test_kdtree(moves, nuclei)
t2 = time.time()
print(t2 - t1)

