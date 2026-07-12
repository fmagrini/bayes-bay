import numpy as np
import pytest

from bayesbay.prior import CustomPrior, GaussianPrior, LaplacePrior, UniformPrior
from bayesbay.discretization import Voronoi2D


def _geometry(name):
    if name == "1d":
        controls = np.array([0.0, 1.0, 2.0])
        queries = np.array([0.25, 0.75, 1.5])
        spherical = False
    elif name == "planar":
        controls = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        queries = np.array([[0.2, 0.2], [0.7, 0.2], [0.4, 0.6]])
        spherical = False
    else:
        controls = np.array(
            [[-170, -20], [170, -20], [-170, 20], [170, 20]], dtype=float
        )
        queries = np.array([[180, 0], [175, 10], [-175, -10]], dtype=float)
        spherical = True
    return controls, queries, spherical


def _make_prior(kind, controls, spherical):
    n = len(controls)
    position_kwargs = {
        "position": controls,
        "spherical_position": spherical,
    }
    perturb_std = np.linspace(0.05, 0.1, n)
    if kind == "uniform":
        return UniformPrior(
            "p",
            vmin=np.linspace(-2, -1, n),
            vmax=np.linspace(1, 2, n),
            perturb_std=perturb_std,
            **position_kwargs,
        )
    if kind == "gaussian":
        return GaussianPrior(
            "p",
            mean=np.linspace(-1, 1, n),
            std=np.linspace(0.5, 1, n),
            perturb_std=perturb_std,
            **position_kwargs,
        )
    if kind == "laplace":
        return LaplacePrior(
            "p",
            mean=np.linspace(-1, 1, n),
            scale=np.linspace(0.5, 1, n),
            perturb_std=perturb_std,
            **position_kwargs,
        )

    def sample(position):
        return float(np.asarray(position).reshape(-1)[0])

    return CustomPrior(
        "p",
        log_prior=lambda value, position: -0.5 * value**2,
        sample=sample,
        perturb_std=perturb_std,
        **position_kwargs,
    )


@pytest.mark.parametrize("geometry", ["1d", "planar", "sphere"])
@pytest.mark.parametrize("kind", ["uniform", "gaussian", "laplace", "custom"])
def test_position_dependent_prior_scalar_shape_and_operations(kind, geometry):
    controls, queries, spherical = _geometry(geometry)
    prior = _make_prior(kind, controls, spherical)

    values = prior.initialize(queries)
    assert values.shape == (len(queries),)
    assert np.isfinite(values).all()

    sampled = prior.sample(queries[0])
    perturbed, ratio = prior.perturb_value(sampled, queries[0])
    log_prior = prior.log_prior(perturbed, queries[0])
    assert np.ndim(sampled) == 0
    assert np.ndim(perturbed) == 0
    assert np.ndim(ratio) == 0
    assert np.ndim(log_prior) == 0


def test_custom_prior_remains_one_dimensional_through_voronoi_birth():
    controls, _, _ = _geometry("planar")
    prior = _make_prior("custom", controls, spherical=False)
    voronoi = Voronoi2D(
        "v",
        polygon=controls[[0, 1, 3, 2]],
        perturb_std=0.1,
        n_dimensions_min=1,
        n_dimensions_max=2,
        parameters=[prior],
    )
    state = voronoi._initialize()
    born, _ = voronoi.birth(state)
    assert state["p"].shape == (1,)
    assert born["p"].shape == (2,)
