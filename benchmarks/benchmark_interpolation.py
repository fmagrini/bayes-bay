"""Benchmark: per-iteration cost of interpolating the Voronoi tessellation
onto a fixed grid, comparing

1. the `compute_kdtree` pattern: rebuild the kd-tree at every discretization
   change and query the full grid in the forward function at every iteration;
2. the incremental interpolation cache (`interpolation_positions`, available
   in both Voronoi2D and Voronoi2DSphere): assignments updated only where a
   perturbation can change them; the forward reads them with an array lookup.

Both approaches are exact; the speedup from incremental interpolation depends
on the number of sites, grid size, and hardware."""

import random
import timeit

import numpy as np
import scipy.spatial

from bayesbay.discretization import Voronoi2D, Voronoi2DSphere

random.seed(0)
np.random.seed(0)


def random_lonlat(n):
    return np.column_stack(
        (
            np.random.uniform(-180, 180, n),
            np.degrees(np.arcsin(np.random.uniform(-1, 1, n))),
        )
    )


print(
    f"{'class':>16} {'n sites':>8} {'m grid':>8} {'kdtree pattern':>16} {'incremental':>13} {'speedup':>8}"
)
for n in (100, 1500):
    for m in (10_000, 100_000):
        for sphere in (False, True):
            if sphere:
                grid = random_lonlat(m)
                voronoi = Voronoi2DSphere(
                    name="v",
                    perturb_std=5,
                    n_dimensions=n,
                    interpolation_positions=grid,
                )
            else:
                grid = np.random.uniform(0, 10, (m, 2))
                voronoi = Voronoi2D(
                    name="v",
                    vmin=[0, 0],
                    vmax=[10, 10],
                    perturb_std=0.3,
                    n_dimensions=n,
                    interpolation_positions=grid,
                )
            ps_state = voronoi._initialize()
            grid_coords = voronoi._interp_coords
            sites_coords = voronoi._interp_position_coords(ps_state["discretization"])

            # 1) kdtree pattern: tree rebuild (on site change) + full-grid
            #    query (in the forward, every iteration)
            def kdtree_pattern():
                tree = scipy.spatial.KDTree(sites_coords)
                return tree.query(grid_coords)[1]

            t_kdtree = timeit.timeit(kdtree_pattern, number=10) / 10

            # 2) incremental pattern: a site move (the most expensive
            #    update), including the proposal itself; the forward is a
            #    pure lookup
            def incremental_pattern():
                isite = random.randint(0, n - 1)
                new_state, _ = voronoi.perturb_value(ps_state, isite)
                return new_state.load_from_cache("interp_nearest")

            t_incr = timeit.timeit(incremental_pattern, number=10) / 10

            print(
                f"{type(voronoi).__name__:>16} {n:>8} {m:>8} "
                f"{t_kdtree * 1e3:>13.2f} ms {t_incr * 1e3:>10.2f} ms "
                f"{t_kdtree / t_incr:>7.0f}x"
            )
