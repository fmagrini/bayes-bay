"""Validate Voronoi cache updates against full nearest-site calculations.

The validator exercises accepted, discarded, and domain-rejected site moves,
births, and deaths for both planar and spherical tessellations. After every
proposal it checks the incremental interpolation cache and the optional
KD-tree against affinities calculated from all site-position pairs.

The final SHA-256 fingerprints also make the script useful when changing the
implementation: run it with the same arguments before and after a refactor and
compare the reported fingerprints. Identical fingerprints demonstrate that
the seeded proposal sequence, ratios, states, and cache values are unchanged.

Run from the repository root, for example::

    python benchmarks/validate_voronoi_cache_equivalence.py
    python benchmarks/validate_voronoi_cache_equivalence.py \
        --n-proposals 2000 --n-positions 5000
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayesbay.discretization import Voronoi2D, Voronoi2DSphere


def _random_lonlat(rng, n, lon_bounds=(-180.0, 180.0), sinlat_bounds=(-1.0, 1.0)):
    return np.column_stack(
        (
            rng.uniform(*lon_bounds, n),
            np.degrees(np.arcsin(rng.uniform(*sinlat_bounds, n))),
        )
    )


def _update_digest(digest, value):
    array = np.ascontiguousarray(value)
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())


def _assert_state_cache(voronoi, state):
    sites_coords = voronoi._interp_position_coords(state["discretization"])
    affinities = voronoi._interp_affinity_block(voronoi._interp_coords, sites_coords)
    best_affinity = affinities.max(axis=1)

    nearest = state.load_from_cache("interp_nearest")
    cached_affinity = state.load_from_cache("interp_affinity")
    achieved_affinity = affinities[np.arange(len(nearest)), nearest]
    np.testing.assert_allclose(achieved_affinity, cached_affinity, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cached_affinity, best_affinity, rtol=0, atol=1e-12)

    kdtree = voronoi.get_kdtree(state)
    np.testing.assert_allclose(kdtree.data, sites_coords, rtol=0, atol=1e-14)
    tree_nearest = kdtree.query(voronoi._interp_coords)[1]
    tree_affinity = affinities[np.arange(len(tree_nearest)), tree_nearest]
    np.testing.assert_allclose(tree_affinity, best_affinity, rtol=0, atol=1e-12)


def _run_scenario(name, voronoi, seed, n_proposals):
    random.seed(seed)
    np.random.seed(seed)
    state = voronoi._initialize()
    _assert_state_cache(voronoi, state)

    digest = hashlib.sha256()
    counts = {"proposed": 0, "accepted": 0, "discarded": 0, "rejected": 0}
    for _ in range(n_proposals):
        kind = random.choice(("move", "birth", "death"))
        nearest_before = state.load_from_cache("interp_nearest").copy()
        affinity_before = state.load_from_cache("interp_affinity").copy()
        kdtree_before = state.load_from_cache("kdtree")

        if kind == "move":
            isite = random.randint(0, state.n_dimensions - 1)
            proposed, ratio = voronoi.perturb_value(state, isite)
        elif kind == "birth":
            proposed, ratio = voronoi.birth(state)
        else:
            proposed, ratio = voronoi.death(state)

        digest.update(kind.encode())
        _update_digest(digest, np.asarray([ratio], dtype=np.float64))
        if proposed is state:
            counts["rejected"] += 1
            np.testing.assert_array_equal(
                state.load_from_cache("interp_nearest"), nearest_before
            )
            np.testing.assert_array_equal(
                state.load_from_cache("interp_affinity"), affinity_before
            )
            assert state.load_from_cache("kdtree") is kdtree_before
            digest.update(b"rejected")
        else:
            counts["proposed"] += 1
            _assert_state_cache(voronoi, proposed)
            _update_digest(digest, proposed["discretization"])
            _update_digest(digest, proposed.load_from_cache("interp_nearest"))
            _update_digest(digest, proposed.load_from_cache("interp_affinity"))
            if random.random() < 0.7:
                state = proposed
                counts["accepted"] += 1
                digest.update(b"accepted")
            else:
                counts["discarded"] += 1
                digest.update(b"discarded")

        _assert_state_cache(voronoi, state)
        _update_digest(digest, state["discretization"])

    assert counts["proposed"] == counts["accepted"] + counts["discarded"]
    print(
        f"{name:18s} {digest.hexdigest()} "
        f"({counts['accepted']} accepted, {counts['discarded']} discarded, "
        f"{counts['rejected']} rejected)"
    )


def _build_scenarios(n_positions):
    planar_rng = np.random.default_rng(1701)
    planar_grid = planar_rng.uniform((-2.0, -1.0), (3.0, 4.0), (n_positions, 2))
    planar = Voronoi2D(
        name="planar",
        vmin=(-2.0, -1.0),
        vmax=(3.0, 4.0),
        perturb_std=0.6,
        n_dimensions_min=2,
        n_dimensions_max=30,
        interpolation_positions=planar_grid,
        compute_kdtree=True,
    )

    sphere_rng = np.random.default_rng(2903)
    sphere_grid = _random_lonlat(sphere_rng, n_positions)
    sphere = Voronoi2DSphere(
        name="sphere",
        perturb_std=8.0,
        n_dimensions_min=2,
        n_dimensions_max=30,
        interpolation_positions=sphere_grid,
        compute_kdtree=True,
    )
    return (("Voronoi2D", planar, 101), ("Voronoi2DSphere", sphere, 202))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-proposals", type=int, default=1000)
    parser.add_argument("--n-positions", type=int, default=2000)
    args = parser.parse_args()
    if args.n_proposals <= 0 or args.n_positions <= 0:
        parser.error("--n-proposals and --n-positions should be positive")

    for name, voronoi, seed in _build_scenarios(args.n_positions):
        _run_scenario(name, voronoi, seed, args.n_proposals)


if __name__ == "__main__":
    main()
