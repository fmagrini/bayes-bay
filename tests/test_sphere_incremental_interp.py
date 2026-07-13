"""Tests for the incremental nearest-site interpolation of Voronoi2D and
Voronoi2DSphere (exact equivalence with full nearest-neighbour queries and
invariance of the default behaviour), spherical polygon domains, longitude
shift, spherical position-dependent priors, pickling and parallel chains."""

import math
import pickle
import random

import numpy as np
import pytest
import scipy.spatial
import shapely.geometry

import bayesbay as bb
from bayesbay.discretization import Voronoi1D, Voronoi2D, Voronoi2DSphere
from bayesbay.exceptions import InvalidProposalException, OutOfDomainException
from bayesbay.prior import UniformPrior


MED_POLYGON = [(-6, 30), (36, 30), (36, 46), (-6, 46)]


def _random_lonlat(n, rng, lon=(-180, 180), sinlat=(-1, 1)):
    return np.column_stack(
        (
            rng.uniform(*lon, n),
            np.degrees(np.arcsin(rng.uniform(*sinlat, n))),
        )
    )


def _assert_interp_cache_correct(voronoi, ps_state):
    """the cached nearest-site assignments must be exactly (up to floating
    point noise on the last digit) those of a full nearest-neighbour search"""
    sites_coords = voronoi._interp_position_coords(ps_state["discretization"])
    full_affinities = voronoi._interp_affinity_block(
        voronoi._interp_coords, sites_coords
    )
    full_best = full_affinities.max(axis=1)
    cached_idx = ps_state.load_from_cache("interp_nearest")
    cached_affinity = ps_state.load_from_cache("interp_affinity")
    achieved = full_affinities[np.arange(len(cached_idx)), cached_idx]
    assert np.allclose(achieved, cached_affinity, rtol=0, atol=1e-12)
    assert np.allclose(cached_affinity, full_best, rtol=0, atol=1e-12)
    assert 0 <= cached_idx.min() and cached_idx.max() < ps_state.n_dimensions


def _run_equivalence(voronoi, n_proposals=1500):
    ps_state = voronoi._initialize()
    _assert_interp_cache_correct(voronoi, ps_state)
    for _ in range(n_proposals):
        kind = random.choice(["move", "birth", "death"])
        if kind == "move":
            isite = random.randint(0, ps_state.n_dimensions - 1)
            new_state, _ = voronoi.perturb_value(ps_state, isite)
        elif kind == "birth":
            new_state, _ = voronoi.birth(ps_state)
        else:
            new_state, _ = voronoi.death(ps_state)
        if new_state is ps_state:  # rejected (-inf) proposal
            continue
        if random.random() < 0.7:  # emulate Metropolis acceptance
            ps_state = new_state
            _assert_interp_cache_correct(voronoi, ps_state)
        # the current state's cache must be untouched by discarded proposals
        _assert_interp_cache_correct(voronoi, ps_state)


@pytest.mark.parametrize("polygon", [None, MED_POLYGON])
def test_sphere_incremental_equivalence(polygon):
    np.random.seed(0)
    random.seed(0)
    rng = np.random.default_rng(0)
    if polygon is None:
        grid = _random_lonlat(5000, rng)
    else:
        grid = np.column_stack((rng.uniform(-6, 36, 5000), rng.uniform(30, 46, 5000)))
    voronoi = Voronoi2DSphere(
        name="v",
        perturb_std=10,
        polygon=polygon,
        n_dimensions_min=2,
        n_dimensions_max=40,
        parameters=[UniformPrior("p", 0, 1, 0.1)],
        interpolation_positions=grid,
    )
    _run_equivalence(voronoi)


@pytest.mark.parametrize("polygon", [None, [(0, 0), (10, 0), (10, 10), (0, 10)]])
def test_voronoi2d_incremental_equivalence(polygon):
    np.random.seed(5)
    random.seed(5)
    rng = np.random.default_rng(5)
    grid = rng.uniform(0, 10, (5000, 2))
    voronoi = Voronoi2D(
        name="v",
        vmin=[0, 0],
        vmax=[10, 10],
        polygon=polygon,
        perturb_std=0.5,
        n_dimensions_min=2,
        n_dimensions_max=40,
        parameters=[UniformPrior("p", 0, 1, 0.1)],
        interpolation_positions=grid,
    )
    _run_equivalence(voronoi)


def test_voronoi2d_default_behaviour_untouched():
    """without `interpolation_positions`, Voronoi2D states must carry no
    interpolation entries in their caches (feature strictly opt-in)"""
    np.random.seed(6)
    random.seed(6)
    voronoi = Voronoi2D(
        name="v",
        vmin=[0, 0],
        vmax=[10, 10],
        perturb_std=0.5,
        n_dimensions_min=2,
        n_dimensions_max=20,
        compute_kdtree=True,
    )
    ps_state = voronoi._initialize()
    for _ in range(200):
        kind = random.choice(["move", "birth", "death"])
        if kind == "move":
            new_state, _ = voronoi.perturb_value(
                ps_state, random.randint(0, ps_state.n_dimensions - 1)
            )
        elif kind == "birth":
            new_state, _ = voronoi.birth(ps_state)
        else:
            new_state, _ = voronoi.death(ps_state)
        if new_state is not ps_state:
            ps_state = new_state
        assert set(ps_state.cache.keys()) <= {"kdtree"}


def test_polygon_validation():
    with pytest.raises(ValueError, match="longitude"):
        Voronoi2DSphere(name="v", polygon=[(-190, 0), (0, 0), (0, 10)])
    with pytest.raises(ValueError, match="pole"):
        Voronoi2DSphere(name="v", polygon=[(0, 80), (10, 80), (5, 90)])
    with pytest.raises(ValueError, match="positive area"):
        Voronoi2DSphere(name="v", polygon=[(0, 0), (1, 0), (2, 0)])
    with pytest.raises(ValueError, match="positive area"):
        Voronoi2D(name="v", polygon=[(0, 0), (1, 0), (2, 0)])


@pytest.mark.parametrize("bad_std", [0, -1, np.nan, np.inf])
def test_site_perturb_std_validation(bad_std):
    with pytest.raises(ValueError, match="finite positive"):
        Voronoi2DSphere(name="v", perturb_std=bad_std)
    with pytest.raises(ValueError, match="finite positive"):
        Voronoi2D(name="v", vmin=[0, 0], vmax=[1, 1], perturb_std=bad_std)
    with pytest.raises(ValueError, match="finite positive"):
        Voronoi1D(name="v", vmin=0, vmax=1, perturb_std=bad_std)


def test_polygon_sampling_and_moves():
    np.random.seed(1)
    random.seed(1)
    voronoi = Voronoi2DSphere(
        name="v", perturb_std=4, polygon=MED_POLYGON, n_dimensions=3
    )
    poly = shapely.geometry.Polygon(MED_POLYGON)
    sites = np.array([voronoi.sample_site() for _ in range(2000)])
    assert all(poly.contains(shapely.geometry.Point(*s)) for s in sites)
    ps_state = voronoi._initialize()
    n_rejected = 0
    for _ in range(1000):
        new_state, ratio = voronoi.perturb_value(ps_state, random.randint(0, 2))
        if new_state is ps_state:
            n_rejected += 1
            assert math.isinf(ratio) and ratio < 0
        else:
            ps_state = new_state
            assert all(
                poly.contains(shapely.geometry.Point(*s))
                for s in ps_state["discretization"]
            )
    assert n_rejected > 0


def test_multipolygon():
    np.random.seed(2)
    random.seed(2)
    multi = shapely.geometry.MultiPolygon(
        [
            shapely.geometry.Polygon([(-6, 30), (10, 30), (10, 46), (-6, 46)]),
            shapely.geometry.Polygon([(20, 30), (36, 30), (36, 46), (20, 46)]),
        ]
    )
    voronoi = Voronoi2DSphere(name="v", perturb_std=4, polygon=multi, n_dimensions=4)
    sites = np.array([voronoi.sample_site() for _ in range(2000)])
    in_gap = (sites[:, 0] > 10) & (sites[:, 0] < 20)
    assert not in_gap.any()


def test_lon_shift_dateline():
    np.random.seed(3)
    random.seed(3)
    fiji = [(160, -25), (200, -25), (200, -10), (160, -10)]
    voronoi = Voronoi2DSphere(
        name="v", perturb_std=5, polygon=fiji, lon_shift=180, n_dimensions=4
    )
    sites = np.array([voronoi.sample_site() for _ in range(1000)])
    assert ((sites[:, 0] >= 160) & (sites[:, 0] <= 200)).all()
    ps_state = voronoi._initialize()
    for _ in range(300):
        new_state, _ = voronoi.perturb_value(ps_state, random.randint(0, 3))
        if new_state is not ps_state:
            ps_state = new_state
    lons = ps_state["discretization"][:, 0]
    assert ((lons >= 160) & (lons <= 200)).all()


@pytest.mark.parametrize("voronoi_cls", [Voronoi2D, Voronoi2DSphere])
def test_polygon_pickle_roundtrip(voronoi_cls):
    voronoi = voronoi_cls(name="v", perturb_std=4, polygon=MED_POLYGON, n_dimensions=3)
    clone = pickle.loads(pickle.dumps(voronoi))
    assert clone._prepared_polygon is not None
    assert clone.polygon.equals(voronoi.polygon)
    poly = shapely.geometry.Polygon(MED_POLYGON)
    assert poly.contains(shapely.geometry.Point(*clone.sample_site()))


def test_out_of_domain_is_invalid_proposal_exception():
    exc = OutOfDomainException("vs", np.array([0.0, 0.0]))
    assert isinstance(exc, InvalidProposalException)


def test_spherical_prior_flag_and_validation():
    # control points straddling the +/-180 meridian: the interpolation must be
    # seamless (the planar interpolation would see them ~360 degrees apart)
    position = np.array(
        [[179.0, 10.0], [-179.0, 10.0], [179.0, -10.0], [-179.0, -10.0]]
    )
    vmin = np.array([1.0, 3.0, 1.0, 3.0])
    vmax = np.array([5.0, 7.0, 5.0, 7.0])

    prior = UniformPrior(
        "vs",
        vmin=vmin,
        vmax=vmax,
        perturb_std=0.1,
        position=position,
        spherical_position=True,
    )
    lo, hi = prior.get_vmin_vmax(np.array([180.0, 0.0]))  # centre of the 4 points
    assert abs(lo - 2.0) < 1e-9 and abs(hi - 6.0) < 1e-9
    lo, hi = prior.get_vmin_vmax(np.array([179.0, 10.0]))  # exact control-point hit
    assert lo == 1.0 and hi == 5.0
    Voronoi2DSphere(name="v", perturb_std=5, n_dimensions=3, parameters=[prior])

    # a planar position-dependent prior is rejected by Voronoi2DSphere
    planar_prior = UniformPrior(
        "vs", vmin=vmin, vmax=vmax, perturb_std=0.1, position=position
    )
    assert not planar_prior.spherical_position
    with pytest.raises(ValueError, match="spherical_position"):
        Voronoi2DSphere(
            name="v", perturb_std=5, n_dimensions=3, parameters=[planar_prior]
        )

    # priors without position are accepted as they are
    scalar_prior = UniformPrior("p", 0, 1, 0.1)
    Voronoi2DSphere(name="v", perturb_std=5, n_dimensions=3, parameters=[scalar_prior])

    # the flag requires a valid (n, 2) position array
    with pytest.raises(AssertionError):
        UniformPrior("p", 0, 1, 0.1, spherical_position=True)


def test_high_level_interpolation_api():
    np.random.seed(7)
    random.seed(7)
    rng = np.random.default_rng(7)
    grid = _random_lonlat(3000, rng)
    vel = UniformPrior("vel", vmin=2, vmax=4, perturb_std=0.1)
    voronoi = Voronoi2DSphere(
        name="v",
        perturb_std=10,
        n_dimensions=25,
        parameters=[vel],
        interpolation_positions=grid,
    )
    ps_state = voronoi._initialize()

    # high-level == low-level == explicit nearest-neighbour interpolation
    indices = voronoi.get_nearest_site_indices(ps_state)
    interp_by_name = voronoi.get_interpolated_values(ps_state, "vel")
    interp_by_array = voronoi.get_interpolated_values(ps_state, ps_state["vel"])
    kdtree = scipy.spatial.KDTree(
        Voronoi2DSphere.lonlat_to_xyz(ps_state["discretization"])
    )
    explicit = ps_state["vel"][kdtree.query(Voronoi2DSphere.lonlat_to_xyz(grid))[1]]
    assert np.array_equal(interp_by_name, interp_by_array)
    assert np.array_equal(interp_by_name, ps_state["vel"][indices])
    assert np.array_equal(interp_by_name, explicit)

    # self-healing: a state lacking the cache entries (e.g. a user-provided
    # starting state) gets them computed on the fly
    fresh_state = voronoi.sample()
    assert not fresh_state.saved_in_cache("interp_nearest")
    indices_fresh = voronoi.get_nearest_site_indices(fresh_state)
    assert fresh_state.saved_in_cache("interp_nearest")
    assert len(indices_fresh) == len(grid)

    # clear error when no interpolation positions were registered
    bare = Voronoi2DSphere(name="v", perturb_std=10, n_dimensions=5)
    with pytest.raises(ValueError, match="interpolation positions"):
        bare.get_nearest_site_indices(bare._initialize())


def test_interpolation_cache_recomputed_after_reregistration():
    sites = np.array([[0.0, 0.0], [90.0, 0.0], [-90.0, 0.0]])
    voronoi = Voronoi2DSphere(
        name="v", n_dimensions=3, interpolation_positions=np.array([[0.0, 0.0]])
    )
    state = bb.ParameterSpaceState(3, {"discretization": sites})
    assert voronoi.get_nearest_site_indices(state).shape == (1,)
    old_version = state.load_from_cache("interp_version")

    positions = np.array([[0.0, 0.0], [90.0, 0.0], [-90.0, 0.0]])
    voronoi.set_interpolation_positions(positions)
    indices = voronoi.get_nearest_site_indices(state)
    assert indices.shape == (3,)
    assert np.array_equal(indices, [0, 1, 2])
    assert state.load_from_cache("interp_version") != old_version


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (Voronoi2D, {"vmin": [0, 0], "vmax": [1, 1]}),
        (Voronoi2DSphere, {}),
    ],
)
def test_kdtree_available_for_every_state_creation_path(cls, kwargs):
    voronoi = cls(name="v", n_dimensions=4, compute_kdtree=True, **kwargs)
    for state in (
        voronoi._initialize(),
        voronoi.sample(),
        voronoi.sample_discretization(),
    ):
        assert state.saved_in_cache("kdtree")
        assert voronoi.get_kdtree(state) is state.load_from_cache("kdtree")

    state = voronoi.sample()
    state.cache.pop("kdtree")
    assert voronoi.get_kdtree(state) is state.load_from_cache("kdtree")


def test_nested_birth_creates_kdtree_cache():
    inner = Voronoi2DSphere(name="inner", n_dimensions=4, compute_kdtree=True)
    outer = Voronoi1D(
        name="outer",
        vmin=0,
        vmax=1,
        perturb_std=0.1,
        n_dimensions_min=1,
        n_dimensions_max=2,
        parameters=[inner],
    )
    old_state = outer._initialize()
    born, _ = outer.birth(old_state)
    newborn_inner = born["inner"][-1]
    assert newborn_inner.saved_in_cache("kdtree")
    assert inner.get_kdtree(newborn_inner) is newborn_inner.load_from_cache("kdtree")


def test_ensemble_interpolation_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same number of samples"):
        Voronoi2DSphere._interpolate_tessellations(
            [np.array([[0.0, 0.0]])], [], np.array([[0.0, 0.0]])
        )
    with pytest.raises(ValueError, match="sample 0"):
        Voronoi2D._interpolate_tessellations(
            [np.array([[0.0, 0.0], [1.0, 1.0]])],
            [np.array([1.0])],
            np.array([[0.0, 0.0]]),
        )


def test_spherical_cell_projection_containment():
    """every probe point on the sphere must fall within the projected map
    polygon of its nearest Voronoi site -- globally, poles and longitude seam
    included. This validates the seam unwrapping and pole-cap closure of
    plot_tessellation's cell projection"""
    rng = np.random.default_rng(11)
    sites = _random_lonlat(60, rng)
    cells, lon_bounds = Voronoi2DSphere._cell_map_polygons(sites)
    assert lon_bounds == (-180.0, 180.0)
    sites_xyz = Voronoi2DSphere.lonlat_to_xyz(sites)
    probes = np.vstack(
        [
            _random_lonlat(3000, rng),
            # explicit probes at the poles and along the longitude seam
            [
                [0, 89.9],
                [0, -89.9],
                [179.9, 0],
                [-179.9, 0],
                [179.9, 55],
                [-179.9, -55],
            ],
        ]
    )
    probes_xyz = Voronoi2DSphere.lonlat_to_xyz(probes)
    assigned = np.argmax(probes_xyz @ sites_xyz.T, axis=1)
    for probe, cell_idx in zip(probes, assigned):
        assert (
            cells[cell_idx].buffer(0.05).contains(shapely.geometry.Point(probe))
        ), f"probe {probe} not in the projected polygon of its nearest site"
    # the projected cells must tile the map: their areas sum to the full frame
    total_area = sum(cell.area for cell in cells)
    assert abs(total_area - 360 * 180) / (360 * 180) < 1e-3


def test_plot_tessellation_sphere():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(12)
    sites = _random_lonlat(50, rng)
    values = rng.uniform(2, 4, 50)
    # global, values filled
    ax, cbar = Voronoi2DSphere.plot_tessellation(sites, values)
    assert cbar is not None
    # global, boundaries only
    ax, cbar = Voronoi2DSphere.plot_tessellation(sites)
    assert cbar is None
    # clipped to a region of interest
    med = shapely.geometry.Polygon(MED_POLYGON)
    ax, cbar = Voronoi2DSphere.plot_tessellation(sites, values, clip_polygon=med)
    assert ax.get_xlim()[0] < -6 and ax.get_xlim()[1] > 36
    # shifted longitude frame: sites expressed in [0, 360)
    sites_shifted = sites.copy()
    sites_shifted[:, 0] = sites_shifted[:, 0] % 360
    cells, lon_bounds = Voronoi2DSphere._cell_map_polygons(sites_shifted)
    assert lon_bounds == (0.0, 360.0)
    ax, cbar = Voronoi2DSphere.plot_tessellation(sites_shifted, values)
    plt.close("all")


def test_spherical_plot_filters_geometry_collections():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = np.array([[0, 45], [90, -20], [-90, -20], [170, 10]], dtype=float)
    cells, _ = Voronoi2DSphere._cell_map_polygons(sites)
    assert all(
        isinstance(cell, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon))
        for cell in cells
    )
    ax, cbar = Voronoi2DSphere.plot_tessellation(sites, np.arange(4))
    assert cbar is not None
    plt.close(ax.figure)


def test_spherical_plot_documents_minimum_site_count():
    with pytest.raises(ValueError, match="at least 4"):
        Voronoi2DSphere.plot_tessellation(
            np.array([[0.0, 0.0], [120.0, 0.0], [-120.0, 0.0]])
        )


def test_plot_tessellation_sphere_3d():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(14)
    sites = _random_lonlat(20, rng)
    values = rng.uniform(2, 4, len(sites))

    ax, cbar = Voronoi2DSphere.plot_tessellation_3d(
        sites,
        values,
        surface_spacing_deg=20,
        voronoi_sites_kwargs={"s": 4},
    )
    assert ax.name == "3d"
    assert cbar is not None
    assert len(ax.lines) == len(sites)
    assert ax.collections
    ax.figure.canvas.draw()

    fig = plt.figure()
    supplied_ax = fig.add_subplot(projection="3d")
    returned_ax, cbar = Voronoi2DSphere.plot_tessellation_3d(sites, ax=supplied_ax)
    assert returned_ax is supplied_ax
    assert cbar is None

    with pytest.raises(ValueError, match="one value per Voronoi site"):
        Voronoi2DSphere.plot_tessellation_3d(sites, values[:-1])
    with pytest.raises(ValueError, match="surface_spacing_deg"):
        Voronoi2DSphere.plot_tessellation_3d(sites, surface_spacing_deg=0)
    with pytest.raises(ValueError, match="3-D axes"):
        _, planar_ax = plt.subplots()
        Voronoi2DSphere.plot_tessellation_3d(sites, ax=planar_ax)
    plt.close("all")


def test_plot_tessellation_voronoi2d_clip():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(13)
    sites = rng.uniform(0, 10, (40, 2))
    values = rng.uniform(2, 4, 40)
    # default path (no clipping): unchanged behaviour, still works
    ax, cbar = Voronoi2D.plot_tessellation(sites, values)
    assert cbar is not None
    # clipped to a non-rectangular polygon
    region = shapely.geometry.Polygon([(1, 1), (9, 2), (8, 9), (4, 8), (2, 6)])
    ax, cbar = Voronoi2D.plot_tessellation(sites, values, clip_polygon=region)
    assert cbar is not None
    xmin, xmax = ax.get_xlim()
    assert xmin >= 0.5 and xmax <= 9.5  # extent follows the clip polygon
    plt.close("all")


def test_parallel_chains_smoke():
    np.random.seed(4)
    random.seed(4)
    rng = np.random.default_rng(4)
    grid = np.column_stack((rng.uniform(-6, 36, 2000), rng.uniform(30, 46, 2000)))
    d_obs = np.ones(5)

    vel = UniformPrior("vel", vmin=0.5, vmax=1.5, perturb_std=0.1)
    voronoi = Voronoi2DSphere(
        name="voronoi",
        perturb_std=3,
        polygon=MED_POLYGON,
        interpolation_positions=grid,
        n_dimensions_min=2,
        n_dimensions_max=20,
        parameters=[vel],
    )

    def forward(state):
        interp_vel = voronoi.get_interpolated_values(state["voronoi"], "vel")
        return np.full(5, interp_vel.mean())

    parameterization = bb.parameterization.Parameterization(voronoi)
    target = bb.likelihood.Target(
        "d_obs", d_obs, std_min=0.01, std_max=1, std_perturb_std=0.05
    )
    log_likelihood = bb.likelihood.LogLikelihood(targets=target, fwd_functions=forward)
    inversion = bb.BayesianInversion(
        parameterization=parameterization, log_likelihood=log_likelihood, n_chains=2
    )
    inversion.run(
        n_iterations=2000,
        burnin_iterations=500,
        save_every=50,
        verbose=False,
        parallel_config={"n_jobs": 2},
    )
    results = inversion.get_results()
    assert len(results["voronoi.n_dimensions"]) == 2 * 30
    sites = np.vstack(results["voronoi.discretization"])
    poly = shapely.geometry.Polygon(MED_POLYGON)
    assert all(poly.contains(shapely.geometry.Point(*s)) for s in sites[::20])
