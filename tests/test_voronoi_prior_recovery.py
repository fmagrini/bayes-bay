import random

import numpy as np
import scipy.stats
import shapely.geometry

from bayesbay import BaseMarkovChain, BayesianInversion, ParameterSpaceState, State
from bayesbay.discretization import Voronoi2D, Voronoi2DSphere
from bayesbay.likelihood import LogLikelihood, Target
from bayesbay.parameterization import Parameterization
from bayesbay.prior import GaussianPrior, UniformPrior


class _ConstantLogLikelihood:
    def log_likelihood_ratio(self, old_state, new_state, temperature):
        return 0.0


def _constant_high_level_likelihood():
    target = Target("dummy", np.array([1.0]), 1.0)
    return LogLikelihood([target], [lambda state: np.array([1.0])])


def test_planar_out_of_domain_moves_preserve_flat_position_prior():
    """Invalid moves must remain self-transitions, not redraws conditioned on x."""
    random.seed(123)
    np.random.seed(123)
    controls = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    prior = GaussianPrior(
        "p",
        mean=np.zeros(4),
        std=np.ones(4),
        perturb_std=0.1,
        position=controls,
    )
    voronoi = Voronoi2D(
        "v",
        vmin=[-1, -1],
        vmax=[1, 1],
        perturb_std=0.18,
        n_dimensions=1,
        parameters=[prior],
    )
    starting_state = State(
        {
            "v": ParameterSpaceState(
                1,
                {
                    "discretization": np.array([[0.0, 0.0]]),
                    "p": np.array([0.0]),
                },
            )
        }
    )

    def move_site(state):
        new_ps_state, ratio = voronoi.perturb_value(state["v"], 0)
        new_state = state.copy()
        new_state.set_param_values("v", new_ps_state)
        return new_state, ratio

    chain = BaseMarkovChain(
        0,
        starting_state,
        [move_site],
        [1],
        _ConstantLogLikelihood(),
        save_dpred=False,
    )
    chain.save_current_iteration = False
    samples = []
    for iteration in range(30_000):
        chain._next_iteration()
        if iteration >= 5_000 and iteration % 10 == 0:
            samples.append(chain.current_state["v"]["discretization"][0].copy())

    samples = np.asarray(samples)
    assert chain.statistics["n_proposed_models_total"] == 30_000
    assert chain.statistics["exceptions"]["OutOfDomainException"] > 0
    for coordinate in samples.T:
        transformed = coordinate + 0.5  # Uniform[-0.5, 0.5] -> Uniform[0, 1]
        assert scipy.stats.kstest(transformed, "uniform").pvalue > 1e-4


def test_full_sphere_constant_likelihood_recovers_joint_prior():
    random.seed(321)
    np.random.seed(321)
    voronoi = Voronoi2DSphere(
        "v",
        perturb_std=20,
        n_dimensions_min=2,
        n_dimensions_max=8,
        parameters=[UniformPrior("vel", 2, 4, 0.3)],
    )
    inversion = BayesianInversion(
        Parameterization(voronoi),
        _constant_high_level_likelihood(),
        n_chains=1,
    )
    inversion.run(
        n_iterations=60_000,
        burnin_iterations=5_000,
        save_every=60,
        verbose=False,
    )
    results = inversion.get_results()

    rng = np.random.default_rng(9)
    sites = np.array(
        [sample[rng.integers(len(sample))] for sample in results["v.discretization"]]
    )
    values = np.array(
        [sample[rng.integers(len(sample))] for sample in results["v.vel"]]
    )
    n_dimensions = np.asarray(results["v.n_dimensions"])

    transformed = (
        (sites[:, 0] + 180) / 360,
        (np.sin(np.radians(sites[:, 1])) + 1) / 2,
        (values - 2) / 2,
    )
    assert all(
        scipy.stats.kstest(sample, "uniform").pvalue > 1e-4 for sample in transformed
    )
    counts = np.bincount(n_dimensions, minlength=9)[2:9]
    assert scipy.stats.chisquare(counts).pvalue > 1e-4


def test_thin_spherical_polygon_births_match_restricted_prior():
    random.seed(42)
    np.random.seed(42)
    # A diagonal strip occupying roughly 9% of its lon/lat bounding box. Its
    # longitude width is constant at every latitude, so sin(latitude) is
    # exactly uniform under the spherical-area prior.
    polygon = shapely.geometry.Polygon([(0, 0), (10, 10), (11, 10), (1, 0)])
    voronoi = Voronoi2DSphere(
        "v",
        perturb_std=1,
        polygon=polygon,
        n_dimensions_min=2,
        n_dimensions_max=6,
    )

    sites = np.array([voronoi.sample_site() for _ in range(2_000)])
    assert all(polygon.contains(shapely.geometry.Point(site)) for site in sites)
    sinlat = np.sin(np.radians(sites[:, 1])) / np.sin(np.radians(10))
    assert scipy.stats.kstest(sinlat, "uniform").pvalue > 1e-4

    state = voronoi._initialize()
    for _ in range(200):
        born, ratio = voronoi.birth(state)
        assert born is not state
        assert np.isfinite(ratio)
        assert polygon.contains(shapely.geometry.Point(born["discretization"][-1]))

    inversion = BayesianInversion(
        Parameterization(voronoi),
        _constant_high_level_likelihood(),
        n_chains=1,
    )
    inversion.run(
        n_iterations=25_000,
        burnin_iterations=3_000,
        save_every=25,
        verbose=False,
    )
    n_dimensions = np.asarray(inversion.get_results()["v.n_dimensions"])
    counts = np.bincount(n_dimensions, minlength=7)[2:7]
    assert scipy.stats.chisquare(counts).pvalue > 1e-4
