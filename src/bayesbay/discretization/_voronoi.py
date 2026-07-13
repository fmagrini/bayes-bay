import math
import random
from bisect import bisect_left
from numbers import Integral, Number
from typing import Callable, List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import shapely
import shapely.geometry
import shapely.ops

from .._state import ParameterSpaceState, State
from .._utils_1d import (
    compute_voronoi1d_cell_extents,
    delete_1d,
    insert_1d,
    interpolate_depth_profile,
    interpolate_nearest_1d,
    nearest_neighbour_1d,
)
from ..parameterization._parameter_space import ParameterSpace
from ..perturbations._birth_death import BirthPerturbation, DeathPerturbation
from ..perturbations._param_space import ParamSpacePerturbation
from ..perturbations._param_values import ParamPerturbation
from ..prior import Prior
from ._discretization import Discretization

SQRT_TWO_PI = math.sqrt(2 * math.pi)
_MAX_POLYGON_SAMPLING_ATTEMPTS = 1_000_000
_SPHERICAL_BOUNDARY_SPACING_DEG = 1.0


def _validate_polygon(polygon, argument_name="polygon"):
    """Return a valid Polygon/MultiPolygon or raise a clear input error."""
    if not isinstance(
        polygon, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
    ):
        try:
            polygon = shapely.geometry.Polygon(polygon)
        except Exception as exc:
            raise ValueError(
                f"`{argument_name}` should define a valid polygonal geometry"
            ) from exc

    parts = (
        list(polygon.geoms)
        if isinstance(polygon, shapely.geometry.MultiPolygon)
        else [polygon]
    )
    bounds = np.asarray(polygon.bounds, dtype=float)
    if (
        polygon.is_empty
        or not polygon.is_valid
        or polygon.area <= 0
        or bounds.shape != (4,)
        or not np.isfinite(bounds).all()
        or any(part.is_empty or not part.is_valid or part.area <= 0 for part in parts)
    ):
        raise ValueError(
            f"`{argument_name}` should be non-empty, valid, finite, and have positive "
            "area; consider repairing invalid input with `shapely.make_valid`"
        )
    return polygon


def _validate_tessellation_samples(samples_cells, samples_values):
    if len(samples_cells) != len(samples_values):
        raise ValueError(
            "`samples_voronoi_cells` and `samples_param_values` should have "
            "the same number of samples"
        )
    for i, (cells, values) in enumerate(zip(samples_cells, samples_values)):
        if len(cells) != len(values):
            raise ValueError(
                f"Voronoi cells/sites and parameter values in sample {i} should "
                "have the same length"
            )


def _polygonal_only(geometry):
    """Return only the areal components of an arbitrary Shapely geometry."""
    if geometry.is_empty:
        return shapely.geometry.Polygon()
    if isinstance(geometry, shapely.geometry.Polygon):
        return geometry
    if isinstance(geometry, shapely.geometry.MultiPolygon):
        polygons = [geom for geom in geometry.geoms if not geom.is_empty]
    elif hasattr(geometry, "geoms"):
        polygons = []
        for geom in geometry.geoms:
            polygonal = _polygonal_only(geom)
            if isinstance(polygonal, shapely.geometry.Polygon):
                if not polygonal.is_empty:
                    polygons.append(polygonal)
            elif isinstance(polygonal, shapely.geometry.MultiPolygon):
                polygons.extend(g for g in polygonal.geoms if not g.is_empty)
    else:
        polygons = []
    if not polygons:
        return shapely.geometry.Polygon()
    union = shapely.ops.unary_union(polygons)
    if isinstance(union, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
        return union
    return _polygonal_only(union)


def _plot(x, y, ax, swap_xy_axes=False, **kwargs):
    if swap_xy_axes:
        ax.plot(y, x, **kwargs)
    else:
        ax.plot(x, y, **kwargs)


class Voronoi(Discretization):
    r"""Utility class for Voronoi tessellation

    Parameters
    ----------
    name : str
        name attributed to the Voronoi tessellation, for display and storing
        purposes
    spatial_dimensions : int
        number of dimensions of the desired Voronoi tessellation, e.g. 1D,
        2D, or 3D.
    vmin, vmax : Union[Number, np.ndarray]
        minimum/maximum value bounding each dimension
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the Voronoi
        sites in each dimension.
    n_dimensions : int, optional
        number of dimensions. None (default) results in a trans-dimensional
        discretization, with the dimensionality of the parameter space allowed
        to vary in the range ``n_dimensions_min``-``n_dimensions_max``
    n_dimensions_min, n_dimensions_max : Number, optional
        minimum and maximum number of dimensions, by default 2 and 10. These
        parameters are ignored if ``n_dimensions`` is not None, i.e. if the
        discretization is not trans-dimensional
    n_dimensions_init_range : Number, optional
        percentage of the range ``n_dimensions_min`` - ``n_dimensions_max`` used to
        initialize the number of dimensions (0.3. by default). For example, if
        ``n_dimensions_min`` = 2, ``n_dimensions_max`` = 10, and
        ``n_dimensions_init_range`` = 0.5,
        the maximum number of dimensions at the initialization is

            int((n_dimensions_max - n_dimensions_min) * n_dimensions_init_range + n_dimensions_min)

    parameters : List[Prior], optional
        a list of free parameters, by default None
    birth_from : {"prior", "neighbour"}, optional
        whether to initialize the free parameters associated with the newborn
        Voronoi cell by randomly drawing from their prior or by perturbing the
        value found in the nearest Voronoi cell (default).
    """

    def __init__(
        self,
        name: str,
        spatial_dimensions: Number,
        vmin: Union[Number, np.ndarray],
        vmax: Union[Number, np.ndarray],
        perturb_std: Union[Number, np.ndarray],
        n_dimensions: int = None,
        n_dimensions_min: int = 2,
        n_dimensions_max: int = 10,
        n_dimensions_init_range: Number = 0.3,
        parameters: List[Prior] = None,
        birth_from: str = "neighbour",  # either "neighbour" or "prior"
    ):
        if n_dimensions is not None:
            if not isinstance(n_dimensions, Integral):
                raise TypeError("`n_dimensions` should be an integer")
            n_dimensions = int(n_dimensions)
            if n_dimensions <= 0:
                raise ValueError("`n_dimensions` should be greater than zero")
        super().__init__(
            name=name,
            spatial_dimensions=spatial_dimensions,
            perturb_std=perturb_std,
            n_dimensions=n_dimensions,
            n_dimensions_min=n_dimensions_min,
            n_dimensions_max=n_dimensions_max,
            n_dimensions_init_range=n_dimensions_init_range,
            parameters=parameters,
            birth_from=birth_from,
            vmin=vmin,
            vmax=vmax,
        )
        if type(self) is Voronoi and spatial_dimensions <= 2:
            subclass = {1: "Voronoi1D", 2: "Voronoi2D"}[spatial_dimensions]
            raise ValueError(f"Use {subclass} for {spatial_dimensions}D tessellations")
        self.vmin = vmin
        self.vmax = vmax
        try:
            perturb_std_values = np.asarray(perturb_std, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("`perturb_std` should contain finite positive values") from exc
        if (
            perturb_std_values.size == 0
            or not np.isfinite(perturb_std_values).all()
            or np.any(perturb_std_values <= 0)
        ):
            raise ValueError("`perturb_std` should contain finite positive values")

    def _after_state_created(
        self, ps_state: ParameterSpaceState, eager_interp: bool
    ) -> None:
        """Populate state-local derived data after creating a state.

        ``eager_interp`` distinguishes states used to initialize a Markov chain
        from states sampled directly from the prior. Overrides should only
        update ``ps_state.cache``. In particular, they should not mutate the
        discretization object, which may be shared by concurrently running
        chains.
        """
        return None

    def _after_site_move(
        self,
        old_ps_state: ParameterSpaceState,
        new_ps_state: ParameterSpaceState,
        isite: int,
    ) -> None:
        """Update state-local derived data after an accepted site move.

        Multidimensional Voronoi sites retain their indices during a move.
        Overrides should only update the cache of ``new_ps_state`` and must
        leave ``old_ps_state`` and the discretization object unchanged.
        """
        return None

    def _after_birth(
        self,
        old_ps_state: ParameterSpaceState,
        new_ps_state: ParameterSpaceState,
    ) -> None:
        """Update state-local derived data after an accepted site birth.

        The multidimensional lifecycle appends the newborn site to the end of
        the discretization. Overrides should only update the cache of
        ``new_ps_state`` and must leave the old state and configuration
        unchanged.
        """
        return None

    def _after_death(
        self,
        old_ps_state: ParameterSpaceState,
        new_ps_state: ParameterSpaceState,
        iremove: int,
    ) -> None:
        """Update state-local derived data after an accepted site death.

        ``iremove`` is the index removed by the multidimensional lifecycle;
        subsequent site indices shift down by one. Overrides should only
        update the cache of ``new_ps_state`` and leave the old state and
        configuration unchanged.
        """
        return None

    def sample_site(self) -> np.ndarray:
        """draws a Voronoi-site position at random within the discretization domain"""
        return np.random.uniform(self.vmin, self.vmax, self.spatial_dimensions)

    def sample_discretization(self) -> ParameterSpaceState:
        # initialize number of dimensions
        if not self.trans_d:
            n_voronoi_cells = self._n_dimensions
        else:
            n_dims_min = self._n_dimensions_min
            n_dims_max = self._n_dimensions_max
            n_voronoi_cells = random.randint(n_dims_min, n_dims_max)

        # initialize Voronoi sites
        voronoi_sites = np.array([self.sample_site() for _ in range(n_voronoi_cells)])

        # initialize parameter values
        parameter_vals = {"discretization": voronoi_sites}
        ps_state = ParameterSpaceState(n_voronoi_cells, parameter_vals)
        self._after_state_created(ps_state, eager_interp=False)
        return ps_state

    def initialize(self, position: np.ndarray = None) -> Union[ParameterSpaceState, List[ParameterSpaceState]]:
        """initializes the parameter space linked to the Voronoi tessellation

        Returns
        -------
        Union[ParameterSpaceState, List[ParameterSpaceState]
            an initial parameter space state, or a list of parameter space states
        """
        if position is None:
            return self._initialize()
        else:
            return [self._initialize() for _ in position]

    def _initialize(self) -> ParameterSpaceState:
        # initialize number of dimensions
        if not self.trans_d:
            n_voronoi_cells = self._n_dimensions
        else:
            init_range = self._n_dimensions_init_range
            n_dims_min = self._n_dimensions_min
            n_dims_max = self._n_dimensions_max
            init_max = int((n_dims_max - n_dims_min) * init_range + n_dims_min)
            n_voronoi_cells = random.randint(n_dims_min, init_max)

        # initialize Voronoi sites
        voronoi_sites = np.array([self.sample_site() for _ in range(n_voronoi_cells)])

        # initialize parameter values
        parameter_vals = {"discretization": voronoi_sites}
        for name, param in self.parameters.items():
            parameter_vals[name] = param.initialize(voronoi_sites)
        ps_state = ParameterSpaceState(n_voronoi_cells, parameter_vals)
        self._after_state_created(ps_state, eager_interp=True)
        return ps_state

    def _perturb_site(self, site: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        """perturbes a Voronoi  site

        Parameters
        ----------
        site : Union[Number, np.ndarray]
            Voronoi site position.

        Returns
        -------
        Union[Number, np.ndarray, None]
            perturbed Voronoi site position, or None if the proposed position
            falls outside the discretization domain. Out-of-domain proposals
            must be rejected (rather than redrawn) to preserve the symmetry of
            the Gaussian proposal and hence detailed balance
        """
        random_deviate = np.random.normal(0, self.perturb_std, self.spatial_dimensions)
        new_site = site + random_deviate
        if all((new_site >= self.vmin) & (new_site <= self.vmax)):
            return new_site
        return None

    def perturb_value(self, old_ps_state: ParameterSpaceState, isite: int) -> Tuple[ParameterSpaceState, Number]:
        r"""perturbs the value of one Voronoi site and calculates the log of the
        partial acceptance probability

        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} =
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}}
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}}.

        Parameters
        ----------
        old_ps_state : ParameterSpaceState
            the current parameter space state
        isite : Number
            the index of the Voronoi site to be perturbed

        Returns
        -------
        Tuple[ParameterSpaceState, Number]
            the new parameter space state and its associated partial acceptance
            probability excluding log likelihood ratio
        """
        old_sites = old_ps_state["discretization"]
        old_site = old_sites[isite]
        new_site = self._perturb_site(old_sites[isite])
        if new_site is None:  # proposed site out of the domain: reject
            return old_ps_state, -math.inf
        new_sites = old_sites.copy()
        new_sites[isite] = new_site
        new_values = {"discretization": new_sites}
        log_prior_ratio = 0
        for param_name, param in self.parameters.items():
            values = old_ps_state[param_name]
            if not isinstance(param, ParameterSpace) and param.position is not None:
                log_prior_old = param.log_prior(values[isite], old_site)
                log_prior_new = param.log_prior(values[isite], new_site)
                log_prior_ratio += log_prior_new - log_prior_old
            new_values[param_name] = values

        new_ps_state = ParameterSpaceState(old_ps_state.n_dimensions, new_values)
        self._after_site_move(old_ps_state, new_ps_state, isite)
        return (
            new_ps_state,
            log_prior_ratio,
        )  # log_proposal_ratio=0 and log_det_jacobian=0

    def birth(self, old_ps_state: ParameterSpaceState) -> Tuple[ParameterSpaceState, float]:
        r"""creates a new Voronoi cell, initializes all free parameters
        associated with it, and returns the pertubed state along with the
        log of the corresponding partial acceptance probability,

        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} =
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}}
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}}.

        In this case, the prior probability of the model :math:`{\bf m}` is

        .. math::
            p({\bf m}) = p({\bf c} \mid k) p(k) \prod_i{p({\bf v}_i \mid {\bf c})} ,

        where :math:`k` denotes the number of Voronoi cells, each entry of the
        vector :math:`{\bf c}` corresponds to the position of a Voronoi site,
        and each :math:`i`\ th free parameter :math:`{\bf v}` has the same
        dimensionality as :math:`{\bf c}`.

        Following [1]_, :math:`p({\bf c} \mid k) = \frac{k! \left(N - k \right)!}{N!}`. If we then
        assume that :math:`p(k) = \frac{1}{\Delta k}`, where :math:`\Delta k = k_{max} - k_{min}`,
        the prior ratio reads

        .. math::
            \frac{p({\bf m'})}{p({\bf m})} =
            \frac{(k+1) \prod_i p(v_i^{k+1})}{(N-k)},

        where :math:`p(v_i^{k+1})` denotes the prior probability of the newly
        born :math:`i`\ th parameter, which may be dependent on :math:`{\bf c}`.
        The proposal ratio reads

        .. math::
            \frac{q({\bf m} \mid {\bf m'})}{q({\bf m'} \mid {\bf m})} =
            \frac{(N-k)}{(k+1) \prod_i q_{v_i}^{k+1}},

        where :math:`q_{v_i}^{k+1}` denotes the proposal probability for the
        newly born :math:`i`\ th parameter in the new dimension. It is easy to
        show that, in the case of a birth from neighbor [1]_ or a birth from
        prior [2]_ (see :attr:`birth_from`), :math:`\lvert \mathbf{J} \rvert = 1`
        and :math:`\alpha_{p} = \frac{p({\bf m'})}{p({\bf m})} \frac{q({\bf m} \mid {\bf m'})}{q({\bf m'} \mid {\bf m})}`.
        It follows that

        .. math::
            \alpha_{p} =
            \frac{(k+1) \prod_i p(v_i^{k+1})}{(N-k)} \frac{(N-k)}{(k+1) \prod_i q_{v_i}^{k+1}} =
            \frac{\prod_i p(v_i^{k+1})}{\prod_i{q_{v_i}^{k+1}}}.

        In the case of a birth from prior, :math:`q_{v_i}^{k+1} = p(v_i^{k+1})`
        and

        .. math::
            \alpha_{p} =
            \frac{\prod_i p(v_i^{k+1})}{\prod_i{p(v_i^{k+1})}} = 1.

        In the case of a birth from neighbor, :math:`q_{v_i}^{k+1} =
        \frac{1}{\theta \sqrt{2 \pi}} \exp \lbrace -\frac{\left( v_i^{k+1} - v_i \right)^2}{2\theta^2} \rbrace`,
        where the newly born value, :math:`v_i^{k+1}`, is generated by perturbing
        the original value, :math:`v_i`, of the :math:`i`\ th parameter. This is
        achieved through a random deviate from the normal distribution
        :math:`\mathcal{N}(v_i, \theta)`, with :math:`\theta` denoting the
        standard deviation of the Gaussian used to carry out the perturbation
        (see, for example, :attr:`bayesbay.prior.UniformPrior.perturb_std`) .
        The partial acceptance probability is then computed numerically.


        Parameters
        ----------
        old_ps_state : ParameterSpaceState
            current parameter space state

        Returns
        -------
        ParameterSpaceState
            new parameter space state
        Number
            log of the partial acceptance probability,
            :math:`log(\alpha_{p}) = \log(\frac{\prod_i p(v_i^{k+1})}{\prod_i{q_{v_i}^{k+1}}})`

        References
        ----------
        .. [1] Bodin et al. 2012, Transdimensional inversion of receiver functions
            and surface wave dispersion
        .. [2] Hawkins and Sambridge 2015, Geophysical imaging using trans-dimensional
            trees

        Notes
        -----
        Subclasses with polygonal domains draw the newborn site by rejection
        sampling from the position prior restricted to the polygon. Redrawing is
        valid here because the birth-position proposal is state independent and
        exactly equals that restricted prior; unlike a move proposal centred on
        the current site, it introduces no state-dependent normalization.
        """
        # prepare for birth perturbation
        n_cells = old_ps_state.n_dimensions
        if n_cells == self._n_dimensions_max:
            return old_ps_state, -math.inf
        # randomly choose a new Voronoi site position
        new_site = self.sample_site()
        old_sites = old_ps_state["discretization"]
        initialized_values, log_prob_ratio = self._initialize_newborn_params(new_site, old_sites, old_ps_state)
        new_values = dict()
        new_sites = np.vstack((old_sites, new_site))
        new_values["discretization"] = new_sites
        for name, value in initialized_values.items():
            old_values = old_ps_state[name]
            if isinstance(old_values, np.ndarray):
                new_values[name] = np.append(old_values, value)
            else:
                new_values[name] = old_values + [value]
        new_ps_state = ParameterSpaceState(n_cells + 1, new_values)
        self._after_birth(old_ps_state, new_ps_state)
        return new_ps_state, log_prob_ratio

    def death(self, old_ps_state: ParameterSpaceState):
        r"""removes a new Voronoi cell and returns the pertubed state along with
        the log of the corresponding partial acceptance probability,

        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} =
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}}
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}}.

        It is straightforward to show that this equals the reciprocal of
        the partial acceptance probability obtained in the case of a birth
        perturbation (see :meth:`birth`), i.e.,

        .. math::
            \alpha_{p} = \frac{\prod_i{q_{v_i}^{k+1}}}{\prod_i p(v_i^{k+1})}.

        Parameters
        ----------
        old_ps_state : ParameterSpaceState
            current parameter space state

        Returns
        -------
        ParameterSpaceState
            new parameter space state
        Number
            log of the partial acceptance probability,
            :math:`log(\alpha_{p}) = -\log(\frac{\prod_i p(v_i^{k+1})}{\prod_i{q_{v_i}^{k+1}}})`
        """
        # prepare for death perturbation
        n_cells = old_ps_state.n_dimensions
        if n_cells == self._n_dimensions_min:
            return old_ps_state, -math.inf
        # randomly choose an existing Voronoi site to kill
        iremove = random.randint(0, n_cells - 1)
        # remove parameter values for the removed site
        new_values = dict()
        for name, old_values in old_ps_state.param_values.items():
            if isinstance(old_values, np.ndarray):  # pure Prior
                new_values[name] = np.delete(old_values, iremove, axis=0)
            else:  # ParameterSpace
                new_values[name] = old_values[:iremove] + old_values[iremove + 1 :]
        new_ps_state = ParameterSpaceState(n_cells - 1, new_values)
        log_prob_ratio = self._log_prob_death_parameters(
            old_ps_state, new_ps_state, iremove
        )
        self._after_death(old_ps_state, new_ps_state, iremove)
        return new_ps_state, log_prob_ratio

    def log_prior(self, *args):
        r"""
        BayesBay implements the grid trick, which calculates the prior
        probability of a Voronoi discretization through the combinatorial
        formula :math:`{N \choose k}^{-1}`, with `k` denoting the number of
        Voronoi sites and `N` the number of possible positions allowed for the
        sites [3]_.

        References
        ----------
        .. [3] Bodin and Sambridge (2009), Seismic tomography with the reversible
            jump algorithm
        """
        raise NotImplementedError

    def _init_perturbation_funcs(self):
        self._perturbation_funcs = []
        self._perturbation_weights = []
        _ps_perturbation_funcs = []
        _ps_perturbation_weights = []
        if self.trans_d:
            _ps_perturbation_funcs.append(BirthPerturbation(self))
            _ps_perturbation_funcs.append(DeathPerturbation(self))
            _ps_perturbation_weights.append(1)
            _ps_perturbation_weights.append(1)
        if self.parameters:
            # initialize parameter values perturbation
            _params = self.parameters.values()
            _prior_pars = [p for p in _params if not isinstance(p, ParameterSpace)]
            if _prior_pars:
                _ps_perturbation_funcs.append(ParamPerturbation(self.name, _prior_pars))
                _ps_perturbation_weights.append(3)
            # initialize nested parameter space perturbations
            _ps_pars = [p for p in _params if isinstance(p, ParameterSpace)]
            for ps in _ps_pars:
                _funcs = ps.perturbation_funcs
                self._perturbation_funcs.extend(_funcs)
                self._perturbation_weights.extend(ps.perturbation_weights)
        _ps_perturbation_funcs.append(ParamPerturbation(self.name, [self]))
        _ps_perturbation_weights.append(1)
        self._perturbation_funcs.append(
            ParamSpacePerturbation(self.name, _ps_perturbation_funcs, _ps_perturbation_weights)
        )
        self._perturbation_weights.append(sum(_ps_perturbation_weights))

    @property
    def perturbation_funcs(self) -> List[Callable[[State], Tuple[State, Number]]]:
        r"""the list of perturbation functions allowed in the parameter space linked to
        the Voronoi discretization. Each function takes in a state (see :class:`State`)
        and returns a new state along with the corresponding partial acceptance
        probability,

        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} =
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}}
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}},

        """
        return self._perturbation_funcs

    @property
    def perturbation_weights(self) -> List[Number]:
        """a list of perturbation weights, corresponding to each of the
        :meth:`perturbation_funcs` that determines the probability of each of them
        to be chosen during each step

        The weights are not normalized and have the following default values:

        - Birth/Death perturbations: 1
        - Parameter values perturbation: 3
        - Voronoi site perturbation: 1
        """
        return self._perturbation_weights

    def nearest_neighbour(
        self, discretization: np.ndarray, query_point: Union[Number, np.ndarray]
    ) -> Union[Number, np.ndarray]:
        return np.argmin(np.linalg.norm(discretization - query_point, axis=1))

    def log_prob_initialize_discretization(self, ps_state: ParameterSpaceState) -> Number:
        return 0


class Voronoi1D(Voronoi):
    r"""Utility class for Voronoi tessellation in 1D

    Parameters
    ----------
    name : str
        name attributed to the Voronoi tessellation, for display and storing
        purposes
    vmin, vmax : Union[Number, np.ndarray]
        minimum/maximum value bounding each dimension
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the Voronoi
        sites in each dimension.
    n_dimensions : int, optional
        number of dimensions. None (default) results in a trans-dimensional
        discretization, with the dimensionality of the parameter space allowed
        to vary in the range ``n_dimensions_min``-``n_dimensions_max``
    n_dimensions_min, n_dimensions_max : Number, optional
        minimum and maximum number of dimensions, by default 1 and 10. These
        parameters are ignored if ``n_dimensions`` is not None, i.e. if the
        discretization is not trans-dimensional
    n_dimensions_init_range : Number, optional
        percentage of the range ``n_dimensions_min`` - ``n_dimensions_max`` used to
        initialize the number of dimensions (0.3. by default). For example, if
        ``n_dimensions_min`` = 1, ``n_dimensions_max`` = 10, and
        ``n_dimensions_init_range`` = 0.5,
        the maximum number of dimensions at the initialization is::

            int((n_dimensions_max - n_dimensions_min) * n_dimensions_init_range + n_dimensions_min)

    parameters : List[Prior], optional
        a list of free parameters, by default None
    birth_from : {"prior", "neighbour"}, optional
        whether to initialize the free parameters associated with the newborn
        Voronoi cell by randomly drawing from their prior or by perturbing the
        value found in the nearest Voronoi cell (default).
    """

    def __init__(
        self,
        name: str,
        vmin: Number,
        vmax: Number,
        perturb_std: Union[Number, np.ndarray],
        n_dimensions: int = None,
        n_dimensions_min: int = 1,
        n_dimensions_max: int = 10,
        n_dimensions_init_range: Number = 0.3,
        parameters: List[Prior] = None,
        birth_from: str = "neighbour",  # either "neighbour" or "prior"
    ):
        super().__init__(
            name=name,
            spatial_dimensions=1,
            vmin=vmin,
            vmax=vmax,
            perturb_std=perturb_std,
            n_dimensions=n_dimensions,
            n_dimensions_min=n_dimensions_min,
            n_dimensions_max=n_dimensions_max,
            n_dimensions_init_range=n_dimensions_init_range,
            parameters=parameters,
            birth_from=birth_from,
        )

    def sample_site(self) -> float:
        """draws a Voronoi-site position at random within the discretization domain"""
        return random.uniform(self.vmin, self.vmax)

    def sample_discretization(self) -> ParameterSpaceState:
        if not self.trans_d:
            n_voronoi_cells = self._n_dimensions
        else:
            n_dims_min = self._n_dimensions_min
            n_dims_max = self._n_dimensions_max
            n_voronoi_cells = random.randint(n_dims_min, n_dims_max)
        voronoi_sites = np.sort([self.sample_site() for _ in range(n_voronoi_cells)])
        parameter_vals = {"discretization": voronoi_sites}
        return ParameterSpaceState(n_voronoi_cells, parameter_vals)

    def _initialize(self) -> ParameterSpaceState:
        if not self.trans_d:
            n_voronoi_cells = self._n_dimensions
        else:
            init_range = self._n_dimensions_init_range
            n_dims_min = self._n_dimensions_min
            n_dims_max = self._n_dimensions_max
            init_max = int((n_dims_max - n_dims_min) * init_range + n_dims_min)
            n_voronoi_cells = random.randint(n_dims_min, init_max)
        voronoi_sites = np.sort([self.sample_site() for _ in range(n_voronoi_cells)])
        parameter_vals = {"discretization": voronoi_sites}
        for name, param in self.parameters.items():
            parameter_vals[name] = param.initialize(voronoi_sites)
        return ParameterSpaceState(n_voronoi_cells, parameter_vals)

    def _perturb_site(self, site: Number) -> Number:
        random_deviate = random.normalvariate(0, self.perturb_std)
        new_site = site + random_deviate
        if self.vmin <= new_site <= self.vmax:
            return new_site
        return None

    def perturb_value(self, old_ps_state: ParameterSpaceState, isite: int) -> Tuple[ParameterSpaceState, Number]:
        old_sites = old_ps_state["discretization"]
        old_site = old_sites[isite]
        new_site = self._perturb_site(old_sites[isite])
        if new_site is None:  # proposed site out of the domain: reject
            return old_ps_state, -math.inf
        new_sites = old_sites.copy()
        new_sites[isite] = new_site
        isort = np.argsort(new_sites)
        new_sites = new_sites[isort]
        new_values = {"discretization": new_sites}
        log_prior_ratio = 0
        for param_name, param in self.parameters.items():
            values = old_ps_state[param_name]
            if not isinstance(param, ParameterSpace) and param.position is not None:
                log_prior_old = param.log_prior(values[isite], old_site)
                log_prior_new = param.log_prior(values[isite], new_site)
                log_prior_ratio += log_prior_new - log_prior_old
            new_values[param_name] = values[isort] if isinstance(values, np.ndarray) else [values[i] for i in isort]
        new_ps_state = ParameterSpaceState(old_ps_state.n_dimensions, new_values)
        return new_ps_state, log_prior_ratio

    def birth(self, old_ps_state: ParameterSpaceState) -> Tuple[ParameterSpaceState, float]:
        n_cells = old_ps_state.n_dimensions
        if n_cells == self._n_dimensions_max:
            return old_ps_state, -math.inf
        new_site = self.sample_site()
        old_sites = old_ps_state["discretization"]
        initialized_values, log_prob_ratio = self._initialize_newborn_params(new_site, old_sites, old_ps_state)
        new_values = dict()
        idx_insert = bisect_left(old_sites, new_site)
        new_sites = insert_1d(old_sites, idx_insert, new_site)
        new_values["discretization"] = new_sites
        for name, value in initialized_values.items():
            old_values = old_ps_state[name]
            if isinstance(old_values, np.ndarray):
                new_values[name] = insert_1d(old_values, idx_insert, float(value))
            else:
                new_values[name] = old_values[:idx_insert] + [value] + old_values[idx_insert:]
        new_ps_state = ParameterSpaceState(n_cells + 1, new_values)
        return new_ps_state, log_prob_ratio

    def death(self, old_ps_state: ParameterSpaceState):
        n_cells = old_ps_state.n_dimensions
        if n_cells == self._n_dimensions_min:
            return old_ps_state, -math.inf
        iremove = random.randint(0, n_cells - 1)
        new_values = dict()
        for name, old_values in old_ps_state.param_values.items():
            if isinstance(old_values, np.ndarray):
                new_values[name] = delete_1d(old_values, iremove)
            else:
                new_values[name] = old_values[:iremove] + old_values[iremove + 1 :]
        new_ps_state = ParameterSpaceState(n_cells - 1, new_values)
        return new_ps_state, self._log_prob_death_parameters(old_ps_state, new_ps_state, iremove)

    def nearest_neighbour(self, discretization: np.ndarray, query_point: Number) -> int:
        return nearest_neighbour_1d(xp=float(query_point), x=discretization, xlen=int(discretization.size))

    @staticmethod
    def compute_cell_extents(voronoi_sites: np.ndarray, lb=0, ub=None, fill_value=0):
        r"""compute Voronoi cell extents from the Voronoi sites. Voronoi-cell
        boundaries are first drawn at the midpoint between consecutive Voronoi
        nuclei. The extent is then derived from the distance between consecutive
        boundaries.

        Parameters
        ----------
        voronoi_sites : np.ndarray of shape (n,)
            Voronoi-site positions. These should be greater or equal to zero

        lb, ub : float
            Lower and upper bounds used in the calculation of Voronoi-cell
            extents. `None` values for `lb` or `ub` denote an unbounded cell.
            The extent of an unbounded cell is set to `fill_value`.

        fill_value : float
            Value attributed to unbounded Voronoi cells

        Returns
        -------
        np.ndarray
            Voronoi-cell extents

        Examples
        --------
        >>> voronoi_sites = np.array([2, 5.5, 8, 10])

        >>> Voronoi1D.compute_cell_extents(voronoi_sites, lb=0, ub=None, fill_value=np.nan)
        array([3.75, 3.  , 2.25,  nan])

        >>> Voronoi1D.compute_cell_extents(voronoi_sites, lb=None, ub=None, fill_value=np.nan)
        array([ nan, 3.  , 2.25,  nan])

        >>> Voronoi1D.compute_cell_extents(voronoi_sites, lb=0, ub=15, fill_value=np.nan)
        array([3.75, 3.  , 2.25, 6.  ])
        """
        lb = lb if lb is not None else -np.inf
        ub = ub if ub is not None else np.inf
        return compute_voronoi1d_cell_extents(voronoi_sites, lb=float(lb), ub=float(ub), fill_value=float(fill_value))

    @staticmethod
    def compute_interface_positions(
        voronoi_cells: np.ndarray,
        input_type="nuclei",
        lb_tessellation=None,
    ):
        """computes the position of Voronoi-cell interfaces

        Parameters
        ----------
        samples_voronoi_cells : list
            either a list of Voronoi-cell extents or of Voronoi-site positions
            (see ``input_type``)
        input_type : str, {'nuclei', 'extents'}
            argument determining whether each entry of `voronoi_cells` should be
            interpreted as a Voronoi-site position (``'nuclei'``) or as the
            extent of the Voronoi cell (``'extents'``)
        lb_tessellation : Number
            the lower boundary of the 1D tessellation, used to calculate the
            interface positions when `input_type` is `'extents'`. Ignored otherwise.

        Returns
        -------
        np.ndarray
        """
        if input_type == "nuclei":
            return (voronoi_cells[:-1] + voronoi_cells[1:]) / 2
        elif input_type == "extents":
            assert lb_tessellation is not None, "`lb_tessellation` should not be None when `input_type` is'extents'"
            return np.cumsum(voronoi_cells)[:-1] + lb_tessellation
        else:
            raise ValueError("`input_type` should either be 'nuclei' or 'extents'")

    @staticmethod
    def interpolate_tessellation(voronoi_cells, param_values, interp_positions, input_type="nuclei"):
        """interpolates the values of a parameter associated with the given
        Voronoi tessellation onto the specified positions

        Parameters
        ----------
        voronoi_cells : np.ndarray
            either Voronoi-cell extents or Voronoi-site positions (see
            ``input_type``)
        param_values : np.ndarray
            the physical parameter value associated with each Voronoi cell
        interp_positions : np.ndarray
            the positions at which the parameter values will be returned
        input_type : str, {'nuclei', 'extents'}
            argument determining whether each entry of `voronoi_cells` should be
            interpreted as a Voronoi-site position (``'nuclei'``) or as the
            extent of the Voronoi cell (``'extents'``)
        Returns
        -------
        np.ndarray
            the physical parameter values associated with ``interp_positions``
        """
        if input_type == "nuclei":
            return interpolate_nearest_1d(interp_positions, voronoi_cells, param_values)
        elif input_type == "extents":
            return interpolate_depth_profile(np.array(voronoi_cells), np.array(param_values), interp_positions)
        raise ValueError("`input_type` should either be 'nuclei' or 'extents'")

    @staticmethod
    def _interpolate_tessellations(
        samples_voronoi_cells,
        samples_param_values,
        interp_positions,
        input_type="nuclei",
    ):
        _validate_tessellation_samples(samples_voronoi_cells, samples_param_values)
        interp_params = np.zeros((len(samples_param_values), len(interp_positions)))
        for i, (sample_cells, sample_values) in enumerate(zip(samples_voronoi_cells, samples_param_values)):
            interp_params[i, :] = Voronoi1D.interpolate_tessellation(
                np.array(sample_cells),
                np.array(sample_values),
                interp_positions,
                input_type=input_type,
            )
        return interp_params

    @staticmethod
    def get_tessellation_statistics(
        samples_voronoi_cells: list,
        samples_param_values: list,
        interp_positions: np.ndarray,
        percentiles: tuple = (10, 90),
        input_type: str = "nuclei",
    ) -> dict:
        """get the mean, median, std and percentiles of the given ensemble

        Parameters
        ----------
        samples_voronoi_cells : list
            either a list of Voronoi-cell extents or of Voronoi-site positions
            (see ``input_type``)
        samples_param_values : list
            a list of parameter values to draw statistics from
        interp_positions : np.ndarray
            points to interpolate
        percentiles : tuple, optional
            percentiles to calculate, by default (10, 90)
        input_type : str, {'nuclei', 'extents'}
            argument determining whether each entry of `voronoi_cells` should be
            interpreted as a Voronoi-site position (``'nuclei'``) or as the
            extent of the Voronoi cell (``'extents'``)

        Returns
        -------
        dict
            a dictionary with these keys: "mean", "median", "std" and
            "percentiles"
        """
        interp_params = Voronoi1D._interpolate_tessellations(
            samples_voronoi_cells,
            samples_param_values,
            interp_positions,
            input_type=input_type,
        )
        statistics = {
            "mean": np.mean(interp_params, axis=0),
            "median": np.median(interp_params, axis=0),
            "std": np.std(interp_params, axis=0),
            "percentiles": np.percentile(interp_params, percentiles, axis=0),
        }
        return statistics

    @staticmethod
    def get_tessellation_density(
        samples_voronoi_cells: np.ndarray,
        samples_param_values: np.ndarray,
        position_bins: Union[int, np.ndarray] = 100,
        param_value_bins: Union[int, np.ndarray] = 100,
        input_type="nuclei",
    ):
        """plot a 2D density histogram of the Voronoi tessellation

        Parameters
        ----------
        samples_voronoi_cells : list
            either a list of Voronoi-cell extents or of Voronoi-site positions
            (see ``input_type``)
        samples_param_values : ndarray
            a 2D numpy array where each row contains the parameter values
            associated with each Voronoi discretization found
            in ``samples_voronoi_cell_extents`` at the same row index
        position_bins: int or np.ndarray, optional
            the position bins or their number, default to 100
        param_value_bins: int or np.ndarray, optional
            the parameter value bins or their number, default to 100
        input_type : str, {'nuclei', 'extents'}
            argument determining whether each entry of `voronoi_cells` should be
            interpreted as a Voronoi-site position (``'nuclei'``) or as the
            extent of the Voronoi cell (``'extents'``)

        Returns
        -------
        density : ndarray, shape(nx, ny)
            The bi-dimensional histogram of samples x and y. Values in x are
            histogrammed along the first dimension and values in y are histogrammed
            along the second dimension
        X : ndarray, shape(nx+1,)
            The bin edges along the first dimension.
        Y : ndarray, shape(ny+1,)
            The bin edges along the second dimension.

        Examples
        --------
        .. code-block:: python

            from bayesbay.discretization import Voronoi1D

            # define and run the Bayesian inversion
            ...

            # plot
            results = inversion.get_results()
            samples_voronoi_sites = results["my_voronoi.discretization"]
            samples_param_values = results["my_voronoi.my_param_value"]
            density, X, Y = Voronoi1D.get_tessellation_density(
                samples_voronoi_sites, samples_param_values
            )
        """
        if input_type not in ["nuclei", "extents"]:
            raise ValueError("`input_type` should either be 'nuclei' or 'extents'")
        if isinstance(position_bins, int):
            lb = 0
            if input_type == "nuclei":
                ub = max([np.max(nuclei) for nuclei in samples_voronoi_cells])
            else:
                ub = 0
                for cell_extents in samples_voronoi_cells:
                    ub = max(ub, np.max(np.cumsum(np.array(cell_extents))))
            interp_positions = np.linspace(lb, ub, position_bins)
        elif isinstance(position_bins, np.ndarray):
            interp_positions = position_bins
        else:
            raise TypeError("`position_bins` should either be int or np.ndarray")
        interp_param_values = Voronoi1D._interpolate_tessellations(
            samples_voronoi_cells,
            samples_param_values,
            interp_positions,
            input_type=input_type,
        )
        density, X, Y = np.histogram2d(
            np.tile(interp_positions, interp_param_values.shape[0]),
            interp_param_values.ravel(),
            bins=(len(interp_positions), param_value_bins),
            density=True,
        )
        return density, X, Y

    @staticmethod
    def plot_tessellation_density(
        samples_voronoi_cells: np.ndarray,
        samples_param_values: np.ndarray,
        position_bins: Union[int, np.ndarray] = 100,
        param_value_bins: Union[int, np.ndarray] = 100,
        ax=None,
        colorbar=True,
        swap_xy_axes=True,
        input_type="nuclei",
        **kwargs,
    ):
        """plot a 2D density histogram of the Voronoi tessellation

        Parameters
        ----------
        samples_voronoi_cells : list
            either a list of Voronoi-cell extents or of Voronoi-site positions
            (see ``input_type``)
        samples_param_values : ndarray
            a 2D numpy array where each row contains the parameter values
            associated with each Voronoi discretization found
            in ``samples_voronoi_cell_extents`` at the same row index
        position_bins: int or np.ndarray, optional
            the position bins or their number, default to 100
        param_value_bins: int or np.ndarray, optional
            the parameter value bins or their number, default to 100
        ax : Axes, optional
            an optional Axes object to plot on
        swap_xy_axes : bool
            if True (default), the x axis is swapped with the y axis so as to display
            the parameter value associated with each Voronoi cell on the x axis
        input_type : str, {'nuclei', 'extents'}
            argument determining whether each entry of `voronoi_cells` should be
            interpreted as a Voronoi-site position (``'nuclei'``) or as the
            extent of the Voronoi cell (``'extents'``)
        kwargs : dict, optional
            Additional keyword arguments to pass to ax.hist2d

        Returns
        -------
        ax : Axes
            The Axes object containing the 2D histogram
        cbar: Colorbar
            The Colorbar object associated with the 2D histogram

        Examples
        --------
        .. code-block:: python

            from bayesbay.discretization import Voronoi1D

            # define and run the Bayesian inversion
            ...

            # plot
            results = inversion.get_results()
            samples_voronoi_sites = results["my_voronoi.discretization"]
            samples_param_values = results["my_voronoi.my_param_value"]
            ax = Voronoi1D.plot_tessellation_density(
                samples_voronoi_sites, samples_param_values
            )
        """
        density, X, Y = Voronoi1D.get_tessellation_density(
            samples_voronoi_cells, samples_param_values, position_bins, param_value_bins, input_type
        )
        if ax is None:
            _, ax = plt.subplots()
        if swap_xy_axes:
            X, Y = Y, X
            if not ax.get_xlabel():
                ax.set_xlabel("Parameter values")
        else:
            if not ax.get_ylabel():
                ax.set_ylabel("Parameter values")

        img = ax.pcolormesh(X, Y, density, **kwargs)
        cbar = plt.colorbar(img, ax=ax, aspect=35, pad=0.02)
        cbar.set_label("Probability density")
        if ax.get_ylim()[0] < ax.get_ylim()[1] and swap_xy_axes:
            ax.invert_yaxis()
        return ax, cbar

    @staticmethod
    def plot_interface_hist(
        samples_voronoi_cells: np.ndarray,
        bins=100,
        ax=None,
        swap_xy_axes=True,
        input_type="nuclei",
        lb_tessellation=None,
        **kwargs,
    ):
        """plot the 1D histogram of Voronoi-interface positions

        Parameters
        ----------
        samples_voronoi_cells : list
            either a list of Voronoi-cell extents or of Voronoi-site positions
            (see ``input_type``)
        bins : int, optional
            number of histogram bins, by default 100
        ax : matplotlib.axes.Axes, optional
            an optional user-provided ax, by default None
        input_type : str, {'nuclei', 'extents'}
            argument determining whether each entry of `voronoi_cells` should be
            interpreted as a Voronoi-site position (``'nuclei'``) or as the
            extent of the Voronoi cell (``'extents'``)
        swap_xy_axes : bool
            if True (default), the x axis is swapped with the y axis so as to display
            the parameter value associated with each Voronoi cell on the x axis
        lb_tessellation : Number
            the lower boundary of the 1D tessellation, used to calculate the
            interface positions when `input_type` is `'extents'`. Ignored otherwise.
        kwargs : dict, optional
            additional keyword arguments to pass to ax.bar

        Returns
        -------
        matplotlib.axes.Axes
        """
        positions = []
        for voronoi_cells in samples_voronoi_cells:
            positions.extend(Voronoi1D.compute_interface_positions(voronoi_cells, input_type, lb_tessellation))
        if ax is None:
            _, ax = plt.subplots()
        hist, edges = np.histogram(positions, bins=bins, density=True)
        if swap_xy_axes:
            ax.barh(edges[:-1], hist, height=np.diff(edges), align="edge", **kwargs)
            if ax.get_ylim()[0] < ax.get_ylim()[1]:
                ax.invert_yaxis()
            if not ax.get_xlabel():
                ax.set_xlabel("Probability density")
        else:
            ax.bar(edges[:-1], hist, width=np.diff(edges), align="edge", **kwargs)
            if not ax.get_ylabel():
                ax.set_ylabel("Probability density")
        return ax

    @staticmethod
    def plot_tessellation(
        voronoi_cells: list,
        param_values: list,
        ax=None,
        bounds=(0, None),
        swap_xy_axes=True,
        input_type="nuclei",
        **kwargs,
    ):
        """plot multiple 1D Earth models based on sampled parameters.

        Parameters
        ----------
        voronoi_cells : list
            either Voronoi-cell extents or Voronoi-site positions (see
            ``input_type``)
        param_values : ndarray
            parameter values associated with each Voronoi cell
        ax : Axes, optional
            an optional Axes object to plot on
        bounds : tuple, optional
            lower and upper boundaries within which the tessellation will be
            displayed. Default is (0, None). When the upper boundary is None
            (default), this is determined by the maximum value in `voronoi_cells`
        swap_xy_axes : bool
            if True (default), the x axis is swapped with the y axis so as to display
            the parameter value associated with each Voronoi cell on the x axis
        input_type : str, {'nuclei', 'extents'}
            argument determining whether each entry of `voronoi_cells` should be
            interpreted as a Voronoi-site position (``'nuclei'``) or as the
            extent of the Voronoi cell (``'extents'``)
        kwargs : dict, optional
            additional keyword arguments to pass to ax.step

        Returns
        -------
        ax : Axes
            The Axes object containing the plot
        """
        lb, ub = bounds
        interface_positions = Voronoi1D.compute_interface_positions(voronoi_cells, input_type, lb)
        if ub is not None:
            assert (
                ub > interface_positions[-1]
            ), f"`bounds[1]` should be greater than the sum of Voronoi cell extents (here, {interface_positions[-1]})"
            end_position = ub
        else:
            end_position = interface_positions[-1] + np.max(np.abs(interface_positions)) / 2

        x = np.insert(np.append(interface_positions, end_position), 0, lb)
        y = np.insert(param_values, 0, param_values[0])
        if swap_xy_axes:
            x, y = y, x

        if ax is None:
            _, ax = plt.subplots()

        # Default plotting style for samples
        sample_style = {
            "linewidth": kwargs.pop("linewidth", kwargs.pop("lw", 0.5)),
            "alpha": kwargs.pop("alpha", 1),
            "color": kwargs.pop("color", kwargs.pop("c", "blue")),  # Fixed color for the sample lines
        }
        sample_style.update(kwargs)  # Override with any provided kwargs
        ax.step(x, y, where="post", **sample_style)
        if ax.get_ylim()[0] < ax.get_ylim()[1] and swap_xy_axes:
            ax.invert_yaxis()
        if swap_xy_axes:
            if not ax.get_xlabel():
                ax.set_xlabel("Parameter values")
        else:
            if not ax.get_ylabel():
                ax.set_ylabel("Parameter values")
        return ax

    @staticmethod
    def plot_tessellations(
        samples_voronoi_cells: list,
        samples_param_values: list,
        ax=None,
        bounds=(0, None),
        swap_xy_axes=True,
        input_type="nuclei",
        **kwargs,
    ):
        """plot multiple 1D Earth models based on sampled parameters.

        Parameters
        ----------
        samples_voronoi_cells : list
            either a list of Voronoi-cell extents or of Voronoi-site positions
            (see ``input_type``)
        samples_param_values : ndarray
            a 2D numpy array where each row contains the parameter values
            associated with each Voronoi discretization found
            in ``samples_voronoi_cell_extents`` at the same row index
        ax : Axes, optional
            an optional Axes object to plot on
        bounds : tuple, optional
            lower and upper boundaries within which the tessellation will be
            displayed. Default is (0, None). When the upper boundary is None
            (default), this is determined by the maximum value in `voronoi_cells`
        swap_xy_axes : bool
            if True (default), the x axis is swapped with the y axis so as to display
            the parameter value associated with each Voronoi cell on the x axis
        input_type : str, {'nuclei', 'extents'}
            argument determining whether each entry of `voronoi_cells` should be
            interpreted as a Voronoi-site position (``'nuclei'``) or as the
            extent of the Voronoi cell (``'extents'``)
        kwargs : dict, optional
            additional keyword arguments to pass to ax.step

        Returns
        -------
        ax : Axes
            The Axes object containing the plot
        """
        lb, ub = bounds
        if input_type == "nuclei":
            samples_voronoi_cell_extents = [
                Voronoi1D.compute_cell_extents(nuclei, lb=lb) for nuclei in samples_voronoi_cells
            ]
        elif input_type == "extents":
            samples_voronoi_cell_extents = samples_voronoi_cells
        else:
            raise ValueError("`input_type` should either be 'nuclei' or 'extents'")

        if ax is None:
            _, ax = plt.subplots()
        if ub is not None:
            ax.set_ylim(0, ub)
        # Default plotting style for samples
        sample_style = {
            "linewidth": kwargs.pop("linewidth", kwargs.pop("lw", 0.5)),
            "alpha": kwargs.pop("alpha", 0.2),
            "color": kwargs.pop("color", kwargs.pop("c", "blue")),  # Fixed color for the sample lines
        }
        sample_style.update(kwargs)  # Override with any provided kwargs

        for extents, values in zip(samples_voronoi_cell_extents, samples_param_values):
            Voronoi1D.plot_tessellation(
                extents,
                values,
                **sample_style,
                ax=ax,
                bounds=bounds,
                input_type="extents",
            )

        if ax.get_ylim()[0] < ax.get_ylim()[1] and swap_xy_axes:
            ax.invert_yaxis()
        if swap_xy_axes:
            if not ax.get_xlabel():
                ax.set_xlabel("Parameter values")
        else:
            if not ax.get_ylabel():
                ax.set_ylabel("Parameter values")
        return ax

    @staticmethod
    def plot_tessellation_statistics(
        samples_voronoi_cells: list,
        samples_param_values: list,
        interp_positions: np.ndarray,
        percentiles=(10, 90),
        ax=None,
        input_type: str = "nuclei",
        swap_xy_axes: bool = True,
    ):
        """plot the mean, median, std and percentiles from the given samples

        Parameters
        ----------
        samples_voronoi_cells : list
            either a list of Voronoi-cell extents or of Voronoi-site positions
            (see ``input_type``)
        samples_param_values : list
            a list of parameter values to draw statistics from
        interp_positions : _type_
            points to interpolate
        percentiles : tuple, optional
            percentiles to calculate, by default (10, 90)
        ax : matplotlib.axes.Axes, optional
            an optional user-provided ax, by default None
        input_type : str, {'nuclei', 'extents'}
            argument determining whether each entry of `voronoi_cells` should be
            interpreted as a Voronoi-site position (``'nuclei'``) or as the
            extent of the Voronoi cell (``'extents'``)
        swap_xy_axes : bool
            if True (default), the x axis is swapped with the y axis so as to display
            the parameter value associated with each Voronoi cell on the x axis
        Returns
        -------
        matplotlib.axes.Axes
            the resulting plot that has the statistics on it
        """
        statistics = Voronoi1D.get_tessellation_statistics(
            samples_voronoi_cells,
            samples_param_values,
            interp_positions,
            percentiles,
            input_type=input_type,
        )

        if ax is None:
            _, ax = plt.subplots()
        _plot(
            interp_positions,
            statistics["mean"],
            ax,
            swap_xy_axes=swap_xy_axes,
            color="b",
            label="Mean",
        )
        _plot(
            interp_positions,
            statistics["mean"] - statistics["std"],
            ax,
            swap_xy_axes=swap_xy_axes,
            color="b",
            ls="--",
            label="STD",
        )
        _plot(
            interp_positions,
            statistics["mean"] + statistics["std"],
            ax,
            swap_xy_axes=swap_xy_axes,
            color="b",
            ls="--",
        )
        _plot(
            interp_positions,
            statistics["median"],
            ax,
            swap_xy_axes=swap_xy_axes,
            color="orange",
            label="Median",
        )
        _plot(
            interp_positions,
            statistics["percentiles"][0],
            ax,
            swap_xy_axes=swap_xy_axes,
            color="orange",
            ls="--",
            label=f"{percentiles[0]}-{percentiles[1]}th Perc.",
        )
        _plot(
            interp_positions,
            statistics["percentiles"][1],
            ax,
            swap_xy_axes=swap_xy_axes,
            color="orange",
            ls="--",
        )
        if ax.get_ylim()[0] < ax.get_ylim()[1] and swap_xy_axes:
            ax.invert_yaxis()
        ax.legend()
        return ax


class _NearestSiteCacheMixin:
    r"""Maintain state-local nearest-site data for 2-D Voronoi geometries.

    The mixin stores optional KD-trees and, for a fixed set of registered
    positions, the index of the nearest Voronoi site in each parameter-space
    state. Interpolation assignments are kept up to date through exact
    incremental updates at each perturbation of the discretization.

    Subclasses define the geometry through :meth:`_interp_position_coords`,
    which maps positions onto the coordinates in which distances are
    computed, and through the ``_interp_affinity_*`` methods, where
    'affinity' denotes a monotonically decreasing function of distance
    (higher affinity = closer): the negative squared Euclidean distance in
    :class:`Voronoi2D`, the cosine of the angular distance in
    :class:`Voronoi2DSphere`.
    """

    _interp_positions = None
    _interp_coords = None
    _interp_version = 0

    def set_interpolation_positions(self, positions: np.ndarray):
        """registers a fixed set of positions onto which the tessellation is
        interpolated during the sampling

        Once registered, every parameter space state carries in its cache:

        - ``"interp_nearest"``: the index of the Voronoi site each registered
          position belongs to, i.e. the nearest site
        - ``"interp_affinity"``: the affinity between each registered position
          and its Voronoi site, i.e. the negative squared Euclidean distance
          (:class:`Voronoi2D`) or the cosine of the angular distance
          (:class:`Voronoi2DSphere`)

        Both entries are kept up to date at every perturbation of the
        discretization through exact incremental updates (a perturbation only
        affects the assignments of the positions within the perturbed Voronoi
        cells), which is much faster than a nearest-neighbour query of all
        the registered positions at every Markov chain iteration.
        Registering a new set of positions increments an internal version;
        existing state caches are then recomputed automatically on first use.

        Configure the interpolation positions before starting the first
        sampling run. Process-based workers receive a serialized copy of the
        discretization configuration, so later changes to the original object
        do not propagate to workers that are already running.

        Parameters
        ----------
        positions : np.ndarray of shape (m, 2)
            the interpolation positions
        """
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("`positions` should be an array of shape (m, 2)")
        self._interp_positions = positions
        self._interp_coords = self._interp_position_coords(positions)
        self._interp_version = getattr(self, "_interp_version", 0) + 1

    def _after_state_created(
        self, ps_state: ParameterSpaceState, eager_interp: bool
    ) -> None:
        if getattr(self, "compute_kdtree", False):
            self._add_kdtree_to_ps_state(ps_state)
        if eager_interp and self._interp_coords is not None:
            self.initialize_interpolation(ps_state)

    def _after_site_move(
        self,
        old_ps_state: ParameterSpaceState,
        new_ps_state: ParameterSpaceState,
        isite: int,
    ) -> None:
        if getattr(self, "compute_kdtree", False):
            self._add_kdtree_to_ps_state(new_ps_state)
        if self._interp_coords is not None:
            self._update_interp_move(old_ps_state, new_ps_state, isite)

    def _after_birth(
        self,
        old_ps_state: ParameterSpaceState,
        new_ps_state: ParameterSpaceState,
    ) -> None:
        if getattr(self, "compute_kdtree", False):
            self._add_kdtree_to_ps_state(new_ps_state)
        if self._interp_coords is not None:
            self._update_interp_birth(old_ps_state, new_ps_state)

    def _after_death(
        self,
        old_ps_state: ParameterSpaceState,
        new_ps_state: ParameterSpaceState,
        iremove: int,
    ) -> None:
        if getattr(self, "compute_kdtree", False):
            self._add_kdtree_to_ps_state(new_ps_state)
        if self._interp_coords is not None:
            self._update_interp_death(old_ps_state, new_ps_state, iremove)

    def _add_kdtree_to_ps_state(
        self, ps_state: ParameterSpaceState
    ) -> ParameterSpaceState:
        sites = ps_state.get_param_values("discretization")
        sites_coords = self._interp_position_coords(sites)
        ps_state.save_to_cache("kdtree", scipy.spatial.KDTree(sites_coords))
        return ps_state

    def get_kdtree(self, ps_state: ParameterSpaceState) -> scipy.spatial.KDTree:
        """Return the state's site KD-tree, building it on demand when needed.

        The tree uses the coordinates appropriate to the tessellation geometry:
        planar coordinates for :class:`Voronoi2D` and 3-D unit vectors for
        :class:`Voronoi2DSphere`. Spherical longitude-latitude query positions
        must therefore be converted with :meth:`Voronoi2DSphere.lonlat_to_xyz`
        before calling ``query``. The accessor supports initialized, sampled,
        nested-birth, and user-created parameter-space states.
        """
        if not ps_state.saved_in_cache("kdtree"):
            self._add_kdtree_to_ps_state(ps_state)
        return ps_state.load_from_cache("kdtree")

    def _interp_cache_is_current(self, ps_state: ParameterSpaceState) -> bool:
        return (
            ps_state.saved_in_cache("interp_nearest")
            and ps_state.saved_in_cache("interp_affinity")
            and ps_state.saved_in_cache("interp_version")
            and ps_state.load_from_cache("interp_version") == self._interp_version
        )

    def _save_interp_cache(self, ps_state, nearest, affinity):
        ps_state.save_to_cache("interp_nearest", nearest)
        ps_state.save_to_cache("interp_affinity", affinity)
        ps_state.save_to_cache("interp_version", self._interp_version)
        return ps_state

    def get_nearest_site_indices(self, ps_state: ParameterSpaceState) -> np.ndarray:
        r"""returns, for each of the positions registered through
        :meth:`set_interpolation_positions`, the index of the Voronoi cell it
        belongs to (i.e., of the nearest Voronoi site), given the
        discretization found in the passed parameter space state

        The indices are read from the cache of the given state, where they
        are maintained through exact incremental updates at every
        perturbation of the discretization. If they are not present in the
        cache (e.g., for a state that was not generated by this class, such
        as a user-provided starting state), they are computed from scratch
        and stored in it.

        Parameters
        ----------
        ps_state : ParameterSpaceState
            the parameter space state holding the discretization

        Returns
        -------
        np.ndarray of shape (m,)
            the index of the Voronoi cell each registered position belongs to
        """
        if self._interp_coords is None:
            raise ValueError(
                "no interpolation positions registered: pass "
                "`interpolation_positions` to the constructor or call "
                "`set_interpolation_positions`"
            )
        if not self._interp_cache_is_current(ps_state):
            self.initialize_interpolation(ps_state)
        return ps_state.load_from_cache("interp_nearest")

    def get_interpolated_values(
        self, ps_state: ParameterSpaceState, param: Union[str, np.ndarray]
    ) -> np.ndarray:
        r"""interpolates the values that the given free parameter takes in
        each Voronoi cell onto the positions registered through
        :meth:`set_interpolation_positions` (nearest-neighbour interpolation,
        i.e., piecewise constant within each Voronoi cell), given the
        discretization found in the passed parameter space state

        This is equivalent to (but, during the sampling, much faster than)
        querying the registered positions against a nearest-neighbour
        structure built on the Voronoi sites::

            indices = self.get_nearest_site_indices(ps_state)
            interp_values = values[indices]

        Parameters
        ----------
        ps_state : ParameterSpaceState
            the parameter space state holding the discretization
        param : Union[str, np.ndarray]
            either the name of a free parameter associated with this
            discretization or an array of values, one per Voronoi cell

        Returns
        -------
        np.ndarray of shape (m,)
            the parameter values interpolated onto the registered positions
        """
        indices = self.get_nearest_site_indices(ps_state)
        if isinstance(param, str):
            values = ps_state.get_param_values(param)
        else:
            values = np.asarray(param)
        return values[indices]

    def initialize_interpolation(self, ps_state: ParameterSpaceState) -> ParameterSpaceState:
        """computes from scratch the nearest-site assignment of the positions
        registered through :meth:`set_interpolation_positions` and stores it
        in the cache of the given parameter space state

        This is done automatically at the initialization of each Markov chain
        and, incrementally, at every perturbation; calling this method is only
        needed for states created by other means (e.g. a user-provided
        starting state).

        Parameters
        ----------
        ps_state : ParameterSpaceState
            the parameter space state whose cache is to be filled

        Returns
        -------
        ParameterSpaceState
            the same instance, with ``"interp_nearest"`` and
            ``"interp_affinity"`` stored in its cache
        """
        sites_coords = self._interp_position_coords(ps_state["discretization"])
        kdtree = scipy.spatial.KDTree(sites_coords)
        nearest = kdtree.query(self._interp_coords)[1].astype(np.int32)
        affinity = self._interp_affinity_pairs(self._interp_coords, sites_coords[nearest])
        return self._save_interp_cache(ps_state, nearest, affinity)

    def _update_interp_move(
        self, old_ps_state: ParameterSpaceState, new_ps_state: ParameterSpaceState, isite: int
    ) -> ParameterSpaceState:
        """updates the nearest-site assignments after the site `isite` has moved.
        Only two groups of registered positions can change assignment: those
        currently assigned to the moved site (re-assigned against all sites)
        and those with a higher affinity to the moved site than to their
        current one. The update is exact"""
        if not self._interp_cache_is_current(old_ps_state):
            return self.initialize_interpolation(new_ps_state)
        old_nearest = old_ps_state.load_from_cache("interp_nearest")
        old_affinity = old_ps_state.load_from_cache("interp_affinity")
        sites_coords = self._interp_position_coords(new_ps_state["discretization"])
        affinity_moved = self._interp_affinity_one(sites_coords[isite])
        # copies: cache entries are shared across states and never mutated in place
        nearest = old_nearest.copy()
        affinity = old_affinity.copy()
        gained = affinity_moved > old_affinity
        nearest[gained] = isite
        affinity[gained] = affinity_moved[gained]
        in_cell = np.flatnonzero(old_nearest == isite)
        if in_cell.size:
            sub_affinities = self._interp_affinity_block(self._interp_coords[in_cell], sites_coords)
            sub_nearest = sub_affinities.argmax(axis=1)
            nearest[in_cell] = sub_nearest
            affinity[in_cell] = sub_affinities[np.arange(in_cell.size), sub_nearest]
        return self._save_interp_cache(new_ps_state, nearest, affinity)

    def _update_interp_birth(
        self, old_ps_state: ParameterSpaceState, new_ps_state: ParameterSpaceState
    ) -> ParameterSpaceState:
        """updates the nearest-site assignments after the birth of a new site
        (appended at the end of the discretization): the only positions that
        change assignment are those with a higher affinity to the newborn site
        than to their current one. The update is exact"""
        if not self._interp_cache_is_current(old_ps_state):
            return self.initialize_interpolation(new_ps_state)
        new_site_idx = new_ps_state.n_dimensions - 1
        new_site_coords = self._interp_position_coords(
            new_ps_state["discretization"][new_site_idx]
        )
        affinity_new = self._interp_affinity_one(new_site_coords)
        old_affinity = old_ps_state.load_from_cache("interp_affinity")
        nearest = old_ps_state.load_from_cache("interp_nearest").copy()
        affinity = old_affinity.copy()
        gained = affinity_new > old_affinity
        nearest[gained] = new_site_idx
        affinity[gained] = affinity_new[gained]
        return self._save_interp_cache(new_ps_state, nearest, affinity)

    def _update_interp_death(
        self,
        old_ps_state: ParameterSpaceState,
        new_ps_state: ParameterSpaceState,
        iremove: int,
    ) -> ParameterSpaceState:
        """updates the nearest-site assignments after the death of a site: the
        positions orphaned by the removed site are re-assigned against all
        remaining sites, and site indices above the removed one are shifted
        down by one. The update is exact"""
        if not self._interp_cache_is_current(old_ps_state):
            return self.initialize_interpolation(new_ps_state)
        new_sites = new_ps_state["discretization"]
        old_nearest = old_ps_state.load_from_cache("interp_nearest")
        nearest = old_nearest.copy()
        affinity = old_ps_state.load_from_cache("interp_affinity").copy()
        nearest[old_nearest > iremove] -= 1
        orphans = np.flatnonzero(old_nearest == iremove)
        if orphans.size:
            new_sites_coords = self._interp_position_coords(new_sites)
            sub_affinities = self._interp_affinity_block(
                self._interp_coords[orphans], new_sites_coords
            )
            sub_nearest = sub_affinities.argmax(axis=1)
            nearest[orphans] = sub_nearest
            affinity[orphans] = sub_affinities[np.arange(orphans.size), sub_nearest]
        return self._save_interp_cache(new_ps_state, nearest, affinity)


class Voronoi2D(_NearestSiteCacheMixin, Voronoi):
    r"""Utility class for Voronoi tessellation in 2D

    Parameters
    ----------
    name : str
        name attributed to the Voronoi tessellation, for display and storing
        purposes
    vmin, vmax : Union[Number, np.ndarray]
        minimum/maximum value bounding each dimension. Ignored when
        ``polygon`` is not ``None``
    polygon: Union[np.ndarray, shapely.geometry.Polygon, shapely.geometry.MultiPolygon], optional
        polygon defining the domain of the Voronoi tessellation; Voronoi sites
        outside this polygon are not allowed
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the Voronoi
        sites in each dimension.
    n_dimensions : int, optional
        number of dimensions. None (default) results in a trans-dimensional
        discretization, with the dimensionality of the parameter space allowed
        to vary in the range ``n_dimensions_min``-``n_dimensions_max``
    n_dimensions_min, n_dimensions_max : Number, optional
        minimum and maximum number of dimensions, by default 2 and 100. These
        parameters are ignored if ``n_dimensions`` is not None, i.e. if the
        discretization is not trans-dimensional
    n_dimensions_init_range : Number, optional
        percentage of the range ``n_dimensions_min`` - ``n_dimensions_max`` used to
        initialize the number of dimensions (0.3. by default). For example, if
        ``n_dimensions_min`` = 2, ``n_dimensions_max`` = 100, and
        ``n_dimensions_init_range`` = 0.5,
        the maximum number of dimensions at the initialization is::

            int((n_dimensions_max - n_dimensions_min) * n_dimensions_init_range + n_dimensions_min)

    parameters : List[Prior], optional
        a list of free parameters, by default None
    birth_from : {"prior", "neighbour"}, optional
        whether to initialize the free parameters associated with the newborn
        Voronoi cell by randomly drawing from their prior or by perturbing the
        value found in the nearest Voronoi cell (default)
    compute_kdtree : bool
        whether to compute a kd-tree for nearest-neighbour lookup at every
        perturbation of the discretization, stored in each state's cache
        under the key ``"kdtree"``. Use this when the forward function needs
        distances, multiple nearest neighbours, or query points that vary
        between iterations; when it only interpolates the tessellation onto
        a fixed set of points, prefer ``interpolation_positions``, which is
        much faster
    interpolation_positions : np.ndarray of shape (m, 2), optional
        fixed positions onto which the tessellation is interpolated during
        the sampling; when given, every state carries in its cache the index
        of the Voronoi cell each position falls in, kept up to date through
        exact incremental updates, so that forward functions can interpolate
        the tessellation through :meth:`get_interpolated_values` (see also
        :meth:`get_nearest_site_indices` and
        :meth:`set_interpolation_positions`). Configure these positions before
        starting the first sampling run
    """

    def __init__(
        self,
        name: str,
        vmin: Number = None,
        vmax: Number = None,
        polygon: Union[np.ndarray, shapely.geometry.Polygon, shapely.geometry.MultiPolygon] = None,
        perturb_std: Union[Number, np.ndarray] = 1,
        n_dimensions: int = None,
        n_dimensions_min: int = 2,
        n_dimensions_max: int = 100,
        n_dimensions_init_range: Number = 0.3,
        parameters: List[Prior] = None,
        birth_from: str = "neighbour",  # either "neighbour" or "prior"
        compute_kdtree: bool = False,
        interpolation_positions: np.ndarray = None,
    ):
        if (vmin is None or vmax is None) and polygon is None:
            raise ValueError(
                "Either both `vmin` and `vmax`, or `polygon`, should define "
                "the discretization domain"
            )
        if polygon is not None:
            polygon = _validate_polygon(polygon)
            vmin = polygon.bounds[:2]
            vmax = polygon.bounds[2:]
        self.polygon = polygon
        if polygon is not None:
            # Preparation builds an internal query index in place without
            # changing the polygon's coordinates or topology.
            shapely.prepare(polygon)
        super().__init__(
            name=name,
            spatial_dimensions=2,
            vmin=vmin,
            vmax=vmax,
            perturb_std=perturb_std,
            n_dimensions=n_dimensions,
            n_dimensions_min=n_dimensions_min,
            n_dimensions_max=n_dimensions_max,
            n_dimensions_init_range=n_dimensions_init_range,
            parameters=parameters,
            birth_from=birth_from,
        )
        self.compute_kdtree = compute_kdtree
        if interpolation_positions is not None:
            self.set_interpolation_positions(interpolation_positions)

    def _interp_position_coords(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(positions, dtype=float)

    def _interp_affinity_one(self, site_coords: np.ndarray) -> np.ndarray:
        deviates = self._interp_coords - site_coords
        return -np.einsum("ij,ij->i", deviates, deviates)

    def _interp_affinity_block(self, points_coords: np.ndarray, sites_coords: np.ndarray) -> np.ndarray:
        return -scipy.spatial.distance.cdist(points_coords, sites_coords, "sqeuclidean")

    def _interp_affinity_pairs(self, points_coords: np.ndarray, site_coords: np.ndarray) -> np.ndarray:
        deviates = points_coords - site_coords
        return -np.einsum("ij,ij->i", deviates, deviates)

    def sample_site(self) -> np.ndarray:
        """Draw a site uniformly from the rectangular or polygonal position prior.

        For a polygon, rejection sampling redraws from the state-independent
        bounding-box proposal until the site lies inside. The result is exactly
        the uniform prior restricted to the polygon, so it is suitable as the
        birth proposal without a Hastings correction.
        """
        if self.polygon is not None:
            for _ in range(_MAX_POLYGON_SAMPLING_ATTEMPTS):
                new_site = super().sample_site()
                if shapely.contains_xy(self.polygon, new_site[0], new_site[1]):
                    return new_site
            raise RuntimeError(
                "failed to sample a site inside `polygon` after "
                f"{_MAX_POLYGON_SAMPLING_ATTEMPTS} attempts; check that the "
                "polygon has a reasonable area relative to its bounding box"
            )
        return super().sample_site()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.polygon is not None:
            # GEOS preparation metadata is not preserved by serialization.
            shapely.prepare(self.polygon)

    def _perturb_site(self, site: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        """perturbes a Voronoi  site

        Parameters
        ----------
        site : Union[Number, np.ndarray]
            Voronoi site position.

        Returns
        -------
        Union[Number, np.ndarray, None]
            perturbed Voronoi site position, or None if the proposed position
            falls outside the discretization domain. Out-of-domain proposals
            must be rejected (rather than redrawn) to preserve the symmetry of
            the Gaussian proposal and hence detailed balance
        """
        if self.polygon is None:
            return super()._perturb_site(site)
        random_deviate = np.random.normal(0, self.perturb_std, self.spatial_dimensions)
        new_site = site + random_deviate
        if shapely.contains_xy(self.polygon, new_site[0], new_site[1]):
            return new_site
        return None

    @staticmethod
    def interpolate_tessellation(
        voronoi_sites: np.ndarray,
        param_values: np.ndarray,
        interp_positions: np.ndarray,
    ):
        r"""nearest neighbour interpolation based on Voronoi-site
        positions and values associated with them.

        Parameters
        ----------
        voronoi_sites : (n, 2) np.ndarray
            the positions of the Voronoi sites
        param_values : (n,) np.ndarray
            the parameter values associated with each Voronoi cell
        interp_positions : (m, 2) np.ndarray
            the positions where interpolation is performed

        Returns
        -------
        np.ndarray
            interpolated values
        """
        kdtree = scipy.spatial.KDTree(voronoi_sites)
        inearest = kdtree.query(interp_positions)[1]
        return param_values[inearest]

    @staticmethod
    def _interpolate_tessellations(samples_voronoi_sites, samples_param_values, interp_positions):
        _validate_tessellation_samples(samples_voronoi_sites, samples_param_values)
        interp_params = np.zeros((len(samples_param_values), len(interp_positions)))
        for i, (sample_sites, sample_values) in enumerate(zip(samples_voronoi_sites, samples_param_values)):
            interp_params[i, :] = Voronoi2D.interpolate_tessellation(
                np.array(sample_sites), np.array(sample_values), interp_positions
            )
        return interp_params

    @staticmethod
    def get_tessellation_statistics(
        samples_voronoi_cells: list,
        samples_param_values: list,
        interp_positions: np.ndarray,
        percentiles: tuple = (10, 90),
    ) -> dict:
        """get the mean, median, std and percentiles of the given ensemble

        Parameters
        ----------
        samples_voronoi_cells : list
            a list of arrays containing the Voronoi-site positions
        samples_param_values : list
            a list of parameter values to draw statistics from
        interp_positions : np.ndarray
            points to interpolate
        percentiles : tuple, optional
            percentiles to calculate, by default (10, 90)

        Returns
        -------
        dict
            a dictionary with these keys: "mean", "median", "std" and
            "percentiles"
        """
        interp_params = Voronoi2D._interpolate_tessellations(
            samples_voronoi_cells, samples_param_values, interp_positions
        )
        statistics = {
            "mean": np.mean(interp_params, axis=0),
            "median": np.median(interp_params, axis=0),
            "std": np.std(interp_params, axis=0),
            "percentiles": np.percentile(interp_params, percentiles, axis=0),
        }
        return statistics

    @staticmethod
    def plot_tessellation(
        voronoi_sites: np.ndarray,
        param_values: np.ndarray = None,
        ax=None,
        cmap="viridis",
        norm=None,
        vmin=None,
        vmax=None,
        voronoi_sites_kwargs=None,
        voronoi_plot_2d_kwargs=None,
        clip_polygon=None,
        **kwargs,
    ):
        """display the Voronoi tessellation

        Parameters
        ----------
        voronoi_sites : np.ndarray of shape (m, 2)
            2D Voronoi-site positions
        param_values: np.ndarray, optional
            parameter values associated with each Voronoi cell. These could
            represent the physical property inferred in each cell of the
            discretized medium
        ax : matplotlib.axes.Axes, optional
            an optional Axes object to plot on
        cmap : Union[str, matplotlib.colors.Colormap]
            the Colormap instance or registered colormap name used to map scalar
            data to colors
        norm : Union[str, matplotlib.colors.Normalize]
            the normalization method used to scale scalar data to the [0, 1]
            range before mapping to colors using ``cmap``. By default, a linear
            scaling is used, mapping the lowest value to 0 and the highest to 1.
        vmin, vmax : Number
            minimum and maximum values used to create the colormap
        voronoi_sites_kwargs : dict, optional
            when given, the Voronoi nuclei are displayed, styled by these
            keyword arguments (passed to ``matplotlib.pyplot.plot``). By
            default (None), the nuclei are not displayed
        voronoi_plot_2d_kwargs : dict
            keyword arguments passed to ``scipy.spatial.voronoi_plot_2d``, used to
            plot the Voronoi interfaces. When ``clip_polygon`` is given, only
            its entries ``line_colors`` and ``line_width`` are used, to style
            the edges of the clipped cells
        clip_polygon : Union[np.ndarray, shapely.geometry.Polygon, shapely.geometry.MultiPolygon], optional
            region of interest to which the displayed tessellation is clipped.
            When given, each Voronoi cell is intersected with this polygon
            before being drawn
        kwargs : dict, optional
            when ``clip_polygon`` is given, additional keyword arguments passed
            to ``matplotlib.axes.Axes.fill`` (e.g., ``transform``, ``alpha``,
            ``zorder``); ignored otherwise

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes containing the tessellation.
        cbar : Union[matplotlib.colorbar.Colorbar, None]
            The colourbar, or ``None`` when ``param_values`` is omitted.

        Notes
        -----
        ``Voronoi2D`` cell edges are straight segments in Cartesian
        coordinates, so they are drawn exactly and require no boundary-spacing
        setting. Spherical Voronoi edges are curved and therefore the plotting
        methods of :class:`Voronoi2DSphere` sample those curves internally at
        a fixed angular spacing suitable for plotting.
        """
        voronoi_plot_2d_kwargs = voronoi_plot_2d_kwargs if voronoi_plot_2d_kwargs is not None else {}
        interfaces_style = {
            "line_colors": "k",
            "show_vertices": False,
            "show_points": False,
            "line_width": 1,
        }
        interfaces_style.update(voronoi_plot_2d_kwargs)
        if voronoi_sites_kwargs is not None:  # nuclei not displayed by default
            voronoi_sites_kwargs = dict(voronoi_sites_kwargs)
            sites_style = {
                "color": voronoi_sites_kwargs.pop("color", voronoi_sites_kwargs.pop("c", "k")),
                "marker": voronoi_sites_kwargs.pop("marker", "o"),
                "ms": voronoi_sites_kwargs.pop("ms", voronoi_sites_kwargs.pop("markersize", 2)),
                "ls": "",
                "lw": 0,
            }
            sites_style.update(voronoi_sites_kwargs)
        else:
            sites_style = None

        # Distant points make the displayed cells finite. Place them relative
        # to the data rather than the origin, using one scale for both axes so
        # axis-aligned and far-from-origin site clouds remain well conditioned.
        bounds_min = voronoi_sites.min(axis=0)
        bounds_max = voronoi_sites.max(axis=0)
        center = (bounds_min + bounds_max) / 2
        coordinate_scale = max(float(np.max(np.abs(voronoi_sites))), 1.0)
        site_scale = max(
            float(np.max(bounds_max - bounds_min)), 1e-6 * coordinate_scale
        )
        dummy_offsets = 100 * site_scale * np.array(
            [[1, 1], [-1, 1], [1, -1], [-1, -1]], dtype=float
        )
        sites = np.vstack((voronoi_sites, center + dummy_offsets))

        voronoi = scipy.spatial.Voronoi(sites)
        if ax is None:
            fig, ax = plt.subplots()

        if clip_polygon is not None:
            return Voronoi2D._plot_tessellation_clipped(
                voronoi,
                voronoi_sites,
                param_values,
                clip_polygon,
                ax=ax,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                edgecolor=interfaces_style["line_colors"],
                linewidth=interfaces_style["line_width"],
                sites_style=sites_style,
                **kwargs,
            )

        if param_values is not None:
            ax, cbar = Voronoi2D._fill_tessellation(
                voronoi,
                param_values,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                cmap=cmap,
            )
        else:
            cbar = None

        scipy.spatial.voronoi_plot_2d(voronoi, ax=ax, **interfaces_style)
        if sites_style is not None:
            ax.plot(voronoi_sites[:, 0], voronoi_sites[:, 1], **sites_style)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        xlim = [voronoi_sites[:, 0].min(), voronoi_sites[:, 0].max()]
        ylim = [voronoi_sites[:, 1].min(), voronoi_sites[:, 1].max()]
        axis_padding = 0.01 * site_scale
        if xlim[0] == xlim[1]:
            xlim[0] -= axis_padding
            xlim[1] += axis_padding
        if ylim[0] == ylim[1]:
            ylim[0] -= axis_padding
            ylim[1] += axis_padding
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return ax, cbar

    @staticmethod
    def _plot_tessellation_clipped(
        voronoi: scipy.spatial.Voronoi,
        voronoi_sites: np.ndarray,
        param_values,
        clip_polygon,
        ax,
        cmap,
        norm,
        vmin,
        vmax,
        edgecolor,
        linewidth,
        sites_style,
        **kwargs,
    ):
        """draws the Voronoi cells intersected with ``clip_polygon``, filled
        by the given parameter values and edged in the given style"""
        if not isinstance(clip_polygon, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            clip_polygon = shapely.geometry.Polygon(clip_polygon)
        if param_values is not None:
            param_values = np.asarray(param_values)
            vmin = vmin if vmin is not None else param_values.min()
            vmax = vmax if vmax is not None else param_values.max()
            norm = norm if norm is not None else plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = cmap if isinstance(cmap, mpl.colors.Colormap) else mpl.colormaps[cmap]
            facecolors = cmap(norm(param_values))
        else:
            facecolors = None

        # the first len(voronoi_sites) points are the original sites, in order
        # (the remaining ones are the distant dummy corners)
        for i in range(len(voronoi_sites)):
            region = voronoi.regions[voronoi.point_region[i]]
            if not region or -1 in region:
                continue
            cell = shapely.geometry.Polygon(voronoi.vertices[region])
            if not cell.is_valid:
                cell = cell.buffer(0)
            cell = _polygonal_only(cell.intersection(clip_polygon))
            if cell.is_empty:
                continue
            geoms = cell.geoms if isinstance(cell, shapely.geometry.MultiPolygon) else [cell]
            facecolor = facecolors[i] if facecolors is not None else "none"
            for geom in geoms:
                boundary_x, boundary_y = geom.exterior.xy
                ax.fill(
                    boundary_x,
                    boundary_y,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    **kwargs,
                )

        cbar = None
        if facecolors is not None:
            cbar = plt.colorbar(
                mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, aspect=35, pad=0.02
            )
            cbar.set_label("Parameter Values")
        if sites_style is not None:
            if "transform" in kwargs:
                sites_style.setdefault("transform", kwargs["transform"])
            ax.plot(voronoi_sites[:, 0], voronoi_sites[:, 1], **sites_style)
        if "transform" not in kwargs:  # on GeoAxes, the extent is set by the user
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            clip_bounds = clip_polygon.bounds
            xmargin = (clip_bounds[2] - clip_bounds[0]) * 0.02
            ymargin = (clip_bounds[3] - clip_bounds[1]) * 0.02
            ax.set_xlim(clip_bounds[0] - xmargin, clip_bounds[2] + xmargin)
            ax.set_ylim(clip_bounds[1] - ymargin, clip_bounds[3] + ymargin)
        return ax, cbar

    @staticmethod
    def _fill_tessellation(
        voronoi: scipy.spatial.Voronoi,
        param_values: np.ndarray,
        ax=None,
        vmin=None,
        vmax=None,
        norm=None,
        cmap=None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        vmin = vmin if vmin is not None else min(param_values)
        vmax = vmax if vmax is not None else max(param_values)
        norm = norm if norm is not None else plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmap if isinstance(cmap, mpl.colors.Colormap) else mpl.colormaps[cmap]
        colors = cmap(norm(param_values))

        for ipoint, iregion in enumerate(voronoi.point_region):
            region = voronoi.regions[iregion]
            if region and -1 not in region:  # Filter out points at infinity
                polygon = [voronoi.vertices[i] for i in region if i >= 0]
                ax.fill(*zip(*polygon), color=colors[ipoint])

        # Create a colorbar to show the mapping between values and colors
        cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, aspect=35, pad=0.02)
        cbar.set_label("Parameter Values")
        return ax, cbar


class Voronoi2DSphere(_NearestSiteCacheMixin, Voronoi):
    r"""Utility class for Voronoi tessellation on the surface of a sphere

    The Voronoi sites are stored as longitude-latitude pairs, in degrees, with
    longitudes in the range [``lon_shift`` - 180, ``lon_shift`` + 180) (i.e.,
    [-180, 180) by default) and latitudes in the range [-90, 90]. The
    tessellation is defined in terms of great-circle distances: each point
    on the sphere belongs to the Voronoi cell whose site is nearest in angular
    distance. The prior probability of a site position is uniform per unit
    area on the sphere, or on the region of interest when ``polygon`` is given.

    A fixed set of positions onto which the tessellation is repeatedly
    interpolated (e.g., the spatial grid used by the forward function) can be
    registered through ``interpolation_positions``. In this case, every state
    carries in its cache the index of the Voronoi cell associated with each
    registered position, kept up to date at every perturbation of the
    discretization through exact incremental updates. Forward functions can
    then interpolate the tessellation through
    :meth:`get_interpolated_values` (or, at a lower level,
    :meth:`get_nearest_site_indices`), avoiding a nearest-neighbour query of
    the full grid at every Markov chain iteration::

        def forward(state):
            voronoi_state = state["voronoi"]
            interp_vel = voronoi.get_interpolated_values(voronoi_state, "vel")
            ...

    .. note::
        Position-dependent priors (see, e.g., the argument ``position`` of
        :class:`bayesbay.prior.UniformPrior`) directly associated with this
        discretization should be created with ``spherical_position=True``,
        so that their positions are interpreted as longitude-latitude pairs
        (in degrees) and their hyper parameters are interpolated in terms of
        great-circle distances, seamlessly across the +/-180 meridian and
        near the poles. A ``ValueError`` is raised otherwise.

    Parameters
    ----------
    name : str
        name attributed to the Voronoi tessellation, for display and storing
        purposes
    perturb_std : Number
        per-axis standard deviation of the local tangent-plane displacement,
        **expressed in degrees** like the site coordinates (the conversion to
        radians needed internally is handled by this class). Site perturbations
        follow a von Mises--Fisher distribution centered on the current site,
        with concentration :math:`\kappa = 1 / \sigma_{rad}^2`. For small
        ``perturb_std``, the total angular displacement is approximately
        Rayleigh distributed, with mean :math:`1.25\sigma` and standard
        deviation :math:`0.66\sigma`. The proposal density depends only on the
        angular distance, so it is symmetric and needs no acceptance correction
    polygon : Union[np.ndarray, shapely.geometry.Polygon, shapely.geometry.MultiPolygon], optional
        region of interest delimiting the domain of the discretization;
        Voronoi sites outside it are not allowed. The polygon is defined in
        the longitude-latitude plane, i.e. its edges are straight lines in an
        equirectangular map, consistent with typical GIS boundary data such
        as country or continent outlines; a ``MultiPolygon`` (e.g. a region
        made of disjoint patches) is also accepted. All polygon longitudes
        should lie within [``lon_shift`` - 180, ``lon_shift`` + 180), and the
        polygon should not touch or include the poles
    lon_shift : Number, optional
        finite scalar offset that shifts the longitude convention: all site
        longitudes are expressed within [``lon_shift`` - 180, ``lon_shift`` +
        180), i.e. [-180, 180) by default. Use this for regions of interest
        crossing the +/-180 meridian: for example, ``lon_shift=180`` expresses
        all longitudes within [0, 360), moving the coordinate seam to the
        Greenwich meridian
    interpolation_positions : np.ndarray of shape (m, 2), optional
        fixed positions (longitude-latitude pairs, in degrees) onto which the
        tessellation is interpolated during the sampling; see
        :meth:`set_interpolation_positions`. Configure these positions before
        starting the first sampling run
    n_dimensions : int, optional
        number of dimensions. None (default) results in a trans-dimensional
        discretization, with the dimensionality of the parameter space allowed
        to vary in the range ``n_dimensions_min``-``n_dimensions_max``
    n_dimensions_min, n_dimensions_max : Number, optional
        minimum and maximum number of dimensions, by default 2 and 100. These
        parameters are ignored if ``n_dimensions`` is not None, i.e. if the
        discretization is not trans-dimensional
    n_dimensions_init_range : Number, optional
        percentage of the range ``n_dimensions_min`` - ``n_dimensions_max`` used to
        initialize the number of dimensions (0.3. by default). For example, if
        ``n_dimensions_min`` = 2, ``n_dimensions_max`` = 100, and
        ``n_dimensions_init_range`` = 0.5,
        the maximum number of dimensions at the initialization is::

            int((n_dimensions_max - n_dimensions_min) * n_dimensions_init_range + n_dimensions_min)

    parameters : List[Prior], optional
        a list of free parameters, by default None
    birth_from : {"prior", "neighbour"}, optional
        whether to initialize the free parameters associated with the newborn
        Voronoi cell by randomly drawing from their prior or by perturbing the
        value found in the nearest Voronoi cell (default)
    compute_kdtree : bool
        whether to compute a kd-tree for nearest-neighbour lookup at every
        perturbation of the discretization, stored in each state's cache
        under the key ``"kdtree"``. The kd-tree is built on the 3-D
        Cartesian (unit-vector) representation of the Voronoi sites, for which
        nearest neighbours in Euclidean distance coincide with nearest
        neighbours in great-circle distance: use :meth:`lonlat_to_xyz` to
        convert longitude-latitude query points before calling ``kdtree.query``.
        Access the tree through :meth:`get_kdtree`, which builds it lazily for
        custom, sampled, or nested-birth states that do not yet carry the cache.
        Use this when the forward function needs distances, multiple nearest
        neighbours, or query points that vary between iterations; when it
        only interpolates the tessellation onto a fixed set of points, prefer
        ``interpolation_positions``, which is much faster
    """

    def __init__(
        self,
        name: str,
        perturb_std: Number = 1,
        polygon: Union[np.ndarray, shapely.geometry.Polygon, shapely.geometry.MultiPolygon] = None,
        lon_shift: Number = 0,
        interpolation_positions: np.ndarray = None,
        n_dimensions: int = None,
        n_dimensions_min: int = 2,
        n_dimensions_max: int = 100,
        n_dimensions_init_range: Number = 0.3,
        parameters: List[Prior] = None,
        birth_from: str = "neighbour",  # either "neighbour" or "prior"
        compute_kdtree: bool = False,
    ):
        if not np.isscalar(perturb_std):
            raise ValueError(
                "`perturb_std` should be a finite positive scalar, interpreted "
                "as the per-axis tangent-plane scale (in degrees)"
            )
        try:
            lon_shift_value = np.asarray(lon_shift, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("`lon_shift` should be a finite scalar") from exc
        if lon_shift_value.ndim != 0 or not np.isfinite(lon_shift_value):
            raise ValueError("`lon_shift` should be a finite scalar")
        lon_shift = float(lon_shift_value)
        self._lon_shift = lon_shift
        self._lon_min = lon_shift - 180.0
        self._init_polygon(polygon)
        super().__init__(
            name=name,
            spatial_dimensions=2,
            vmin=np.array([self._lon_min, -90.0]),
            vmax=np.array([self._lon_min + 360.0, 90.0]),
            perturb_std=perturb_std,
            n_dimensions=n_dimensions,
            n_dimensions_min=n_dimensions_min,
            n_dimensions_max=n_dimensions_max,
            n_dimensions_init_range=n_dimensions_init_range,
            parameters=parameters,
            birth_from=birth_from,
        )
        self.compute_kdtree = compute_kdtree
        if interpolation_positions is not None:
            self.set_interpolation_positions(interpolation_positions)
        # the position-dependent priors directly associated with this
        # discretization must interpolate their hyper parameters in terms of
        # great-circle distances (nested parameter spaces manage their own
        # parameters)
        for param in self.parameters.values():
            if (
                isinstance(param, Prior)
                and not isinstance(param, ParameterSpace)
                and param.position is not None
                and not getattr(param, "spherical_position", False)
            ):
                raise ValueError(
                    f"the position-dependent prior '{param.name}' should be "
                    "created with `spherical_position=True` when associated "
                    "with a Voronoi2DSphere, so that its hyper parameters are "
                    "interpolated in terms of great-circle distances"
                )

    def _init_polygon(self, polygon):
        self.polygon = None
        self._polygon_sampling_bounds = None
        if polygon is None:
            return
        polygon = _validate_polygon(polygon)
        lon_min, lat_min, lon_max, lat_max = polygon.bounds
        if lon_min < self._lon_min or lon_max > self._lon_min + 360:
            raise ValueError(
                f"`polygon` longitudes should lie within [{self._lon_min}, "
                f"{self._lon_min + 360}). For regions of interest crossing this "
                "seam (e.g. the +/-180 meridian when `lon_shift` is 0), provide "
                "a suitable `lon_shift` and express the polygon longitudes "
                "accordingly"
            )
        if lat_min <= -90 or lat_max >= 90:
            raise ValueError(
                "`polygon` should not touch or include the poles: such regions "
                "of interest cannot be represented as a longitude-latitude polygon"
            )
        self.polygon = polygon
        # Preparation builds an internal query index in place without changing
        # the polygon's coordinates or topology.
        shapely.prepare(polygon)
        self._polygon_sampling_bounds = (
            lon_min,
            lon_max,
            math.sin(math.radians(lat_min)),
            math.sin(math.radians(lat_max)),
        )

    def _wrap_lon(self, lon):
        """wraps longitude(s) into the interval [lon_shift - 180, lon_shift + 180)"""
        return (lon - self._lon_min) % 360.0 + self._lon_min

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.polygon is not None:
            # GEOS preparation metadata is not preserved by serialization.
            shapely.prepare(self.polygon)

    @staticmethod
    def lonlat_to_xyz(lonlat: np.ndarray) -> np.ndarray:
        """converts longitude-latitude pairs (in degrees) into 3-D Cartesian
        coordinates on the unit sphere

        Parameters
        ----------
        lonlat : np.ndarray of shape (..., 2)
            longitude-latitude pair(s), in degrees

        Returns
        -------
        np.ndarray of shape (..., 3)
            the corresponding unit vector(s)
        """
        lonlat = np.asarray(lonlat, dtype=float)
        lon = np.radians(lonlat[..., 0])
        lat = np.radians(lonlat[..., 1])
        coslat = np.cos(lat)
        return np.stack((coslat * np.cos(lon), coslat * np.sin(lon), np.sin(lat)), axis=-1)

    @staticmethod
    def xyz_to_lonlat(xyz: np.ndarray) -> np.ndarray:
        """converts 3-D Cartesian coordinates on the unit sphere into
        longitude-latitude pairs (in degrees)

        Parameters
        ----------
        xyz : np.ndarray of shape (..., 3)
            unit vector(s)

        Returns
        -------
        np.ndarray of shape (..., 2)
            the corresponding longitude-latitude pair(s), in degrees, with
            longitudes in the range [-180, 180]
        """
        xyz = np.asarray(xyz, dtype=float)
        lon = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
        lat = np.degrees(np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0)))
        return np.stack((lon, lat), axis=-1)

    def sample_site(self) -> np.ndarray:
        """draws a Voronoi-site position at random from the uniform (per unit
        area) distribution on the sphere or, when :attr:`polygon` is given,
        on the region of interest it delimits

        Rejection sampling within a polygon redraws from a state-independent
        uniform-per-area bounding-box proposal. The returned draw is therefore
        exactly distributed according to the position prior restricted to the
        polygon and can be used for birth proposals without a Hastings
        correction.
        """
        if self.polygon is None:
            lon = random.uniform(self._lon_min, self._lon_min + 360.0)
            lat = math.degrees(math.asin(random.uniform(-1, 1)))
            return np.array([lon, lat])
        lon_min, lon_max, sin_lat_min, sin_lat_max = self._polygon_sampling_bounds
        for _ in range(_MAX_POLYGON_SAMPLING_ATTEMPTS):
            # uniform per unit area within the polygon's bounding box; the
            # rejection step below restricts it to the polygon itself
            lon = random.uniform(lon_min, lon_max)
            lat = math.degrees(math.asin(random.uniform(sin_lat_min, sin_lat_max)))
            if shapely.contains_xy(self.polygon, lon, lat):
                return np.array([lon, lat])
        raise RuntimeError(
            "failed to sample a site inside `polygon` after "
            f"{_MAX_POLYGON_SAMPLING_ATTEMPTS} attempts; check that the "
            "polygon has a reasonable spherical area relative to its bounding box"
        )

    def _perturb_site(self, site: np.ndarray) -> np.ndarray:
        r"""perturbes a Voronoi site through a von Mises--Fisher proposal
        centered on it, with concentration parameter
        :math:`\kappa = 1 / \sigma_{rad}^2`, where :math:`\sigma_{rad}`
        is :attr:`perturb_std` (given by the user in degrees) converted
        to radians internally

        Parameters
        ----------
        site : np.ndarray
            Voronoi-site position, i.e. a longitude-latitude pair in degrees

        Returns
        -------
        Union[np.ndarray, None]
            perturbed Voronoi-site position, or None if the proposed position
            falls outside :attr:`polygon` (when given). The proposal density
            only depends on the angular distance from ``site``, hence it is
            symmetric; out-of-domain proposals are rejected (rather than
            redrawn) to preserve detailed balance
        """
        sigma = math.radians(self.perturb_std)
        kappa = 1 / sigma**2
        # cosine of the angular distance between the current and the proposed
        # site, drawn from the von Mises-Fisher distribution
        # random.random() may return exactly zero; using 1-u keeps the draw
        # uniform while guaranteeing a strictly positive logarithm argument
        # even when exp(-2*kappa) underflows for concentrated proposals.
        u = 1.0 - random.random()
        cos_gamma = 1 + math.log(u + (1 - u) * math.exp(-2 * kappa)) / kappa
        cos_gamma = min(1.0, max(-1.0, cos_gamma))
        sin_gamma = math.sqrt(1 - cos_gamma**2)
        xyz = self.lonlat_to_xyz(site)
        # orthonormal basis {e1, e2} of the plane tangent to the sphere at the site
        helper = np.array([0.0, 0.0, 1.0]) if abs(xyz[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(xyz, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(xyz, e1)
        azimuth = random.uniform(0, 2 * math.pi)
        new_xyz = cos_gamma * xyz + sin_gamma * (math.cos(azimuth) * e1 + math.sin(azimuth) * e2)
        new_site = self.xyz_to_lonlat(new_xyz)
        new_site[0] = self._wrap_lon(new_site[0])
        if self.polygon is not None and not shapely.contains_xy(
            self.polygon, new_site[0], new_site[1]
        ):
            return None
        return new_site

    def nearest_neighbour(self, discretization: np.ndarray, query_point: np.ndarray) -> int:
        """returns the index of the Voronoi site nearest (in great-circle
        distance) to the given query point

        Parameters
        ----------
        discretization : np.ndarray of shape (n, 2)
            the Voronoi-site positions, i.e. longitude-latitude pairs in degrees
        query_point : np.ndarray of shape (2,)
            longitude-latitude pair, in degrees

        Returns
        -------
        int
            the index of the nearest Voronoi site
        """
        sites_xyz = self.lonlat_to_xyz(discretization)
        query_xyz = self.lonlat_to_xyz(query_point)
        return int(np.argmax(sites_xyz @ query_xyz))

    def _interp_position_coords(self, positions: np.ndarray) -> np.ndarray:
        return self.lonlat_to_xyz(positions)

    def _interp_affinity_one(self, site_coords: np.ndarray) -> np.ndarray:
        return self._interp_coords @ site_coords

    def _interp_affinity_block(self, points_coords: np.ndarray, sites_coords: np.ndarray) -> np.ndarray:
        return points_coords @ sites_coords.T

    def _interp_affinity_pairs(self, points_coords: np.ndarray, site_coords: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", points_coords, site_coords)

    @staticmethod
    def interpolate_tessellation(
        voronoi_sites: np.ndarray,
        param_values: np.ndarray,
        interp_positions: np.ndarray,
    ):
        r"""nearest-neighbour interpolation, in terms of great-circle
        distance, based on the Voronoi-site positions and the values
        associated with them

        Parameters
        ----------
        voronoi_sites : (n, 2) np.ndarray
            the Voronoi-site positions, i.e. longitude-latitude pairs in degrees
        param_values : (n,) np.ndarray
            the parameter values associated with each Voronoi cell
        interp_positions : (m, 2) or (m, 3) np.ndarray
            the positions at which the interpolation is performed, either as
            longitude-latitude pairs in degrees or as unit vectors (see
            :meth:`lonlat_to_xyz`). When interpolating many tessellations onto
            the same positions, pre-converting them to unit vectors once
            avoids the conversion at each call

        Returns
        -------
        np.ndarray
            interpolated values
        """
        interp_positions = np.asarray(interp_positions, dtype=float)
        if interp_positions.shape[-1] != 3:
            interp_positions = Voronoi2DSphere.lonlat_to_xyz(interp_positions)
        kdtree = scipy.spatial.KDTree(Voronoi2DSphere.lonlat_to_xyz(voronoi_sites))
        inearest = kdtree.query(interp_positions)[1]
        return np.asarray(param_values)[inearest]

    @staticmethod
    def _interpolate_tessellations(samples_voronoi_sites, samples_param_values, interp_positions):
        _validate_tessellation_samples(samples_voronoi_sites, samples_param_values)
        interp_params = np.zeros((len(samples_param_values), len(interp_positions)))
        interp_positions = np.asarray(interp_positions, dtype=float)
        if interp_positions.shape[-1] != 3:  # convert to unit vectors only once
            interp_positions = Voronoi2DSphere.lonlat_to_xyz(interp_positions)
        for i, (sample_sites, sample_values) in enumerate(zip(samples_voronoi_sites, samples_param_values)):
            interp_params[i, :] = Voronoi2DSphere.interpolate_tessellation(
                np.array(sample_sites), np.array(sample_values), interp_positions
            )
        return interp_params

    @staticmethod
    def get_tessellation_statistics(
        samples_voronoi_cells: list,
        samples_param_values: list,
        interp_positions: np.ndarray,
        percentiles: tuple = (10, 90),
    ) -> dict:
        """get the mean, median, std and percentiles of the given ensemble

        Parameters
        ----------
        samples_voronoi_cells : list
            a list of Voronoi-site positions, i.e. arrays of longitude-latitude
            pairs in degrees
        samples_param_values : list
            a list of parameter values to draw statistics from
        interp_positions : np.ndarray
            the longitude-latitude pairs (in degrees) at which the statistics
            are calculated
        percentiles : tuple, optional
            percentiles to calculate, by default (10, 90)

        Returns
        -------
        dict
            a dictionary with these keys: "mean", "median", "std" and
            "percentiles"
        """
        interp_params = Voronoi2DSphere._interpolate_tessellations(
            samples_voronoi_cells, samples_param_values, interp_positions
        )
        statistics = {
            "mean": np.mean(interp_params, axis=0),
            "median": np.median(interp_params, axis=0),
            "std": np.std(interp_params, axis=0),
            "percentiles": np.percentile(interp_params, percentiles, axis=0),
        }
        return statistics

    @staticmethod
    def _cell_boundaries_xyz(voronoi_sites: np.ndarray) -> list:
        """computes, for each Voronoi site, the closed boundary of its cell on
        the unit sphere, as an array of unit vectors ordered along the
        boundary. Geodesic edges are sampled internally at a fixed angular
        spacing suitable for plotting.
        """
        xyz = Voronoi2DSphere.lonlat_to_xyz(voronoi_sites)
        if len(xyz) < 4:
            raise ValueError(
                "at least 4 Voronoi sites are needed to plot the spherical tessellation"
            )
        spherical_voronoi = scipy.spatial.SphericalVoronoi(xyz)
        spherical_voronoi.sort_vertices_of_regions()
        boundaries = []
        for region in spherical_voronoi.regions:
            verts = spherical_voronoi.vertices[region]
            segments = []
            for ivert in range(len(verts)):
                start = verts[ivert]
                end = verts[(ivert + 1) % len(verts)]
                angle = math.acos(min(1.0, max(-1.0, float(start @ end))))
                n_samples = max(
                    1,
                    int(
                        math.ceil(
                            math.degrees(angle) / _SPHERICAL_BOUNDARY_SPACING_DEG
                        )
                    ),
                )
                fractions = np.linspace(0, 1, n_samples, endpoint=False)[:, None]
                if angle < 1e-12:
                    segments.append(start[None, :])
                else:
                    segments.append(
                        (np.sin((1 - fractions) * angle) * start + np.sin(fractions * angle) * end)
                        / math.sin(angle)
                    )
            boundary = np.vstack(segments)
            boundary /= np.linalg.norm(boundary, axis=1, keepdims=True)
            boundaries.append(boundary)
        return boundaries

    @staticmethod
    def _cell_map_polygons(
        voronoi_sites: np.ndarray,
        clip_polygon=None,
        lon_bounds: Tuple[Number, Number] = None,
    ) -> Tuple[list, Tuple[Number, Number]]:
        """projects the spherical Voronoi cells onto the longitude-latitude
        plane, returning one shapely (Multi)Polygon per Voronoi site (empty
        when the cell does not intersect ``clip_polygon``) along with the
        longitude bounds of the map frame.

        Cells crossing the longitude seam are handled by unwrapping the
        longitudes cumulatively along each cell boundary and drawing the cell
        at every 360-degree shift that intersects the frame; cells containing
        a pole (identified by a boundary whose longitudes wind around by 360
        degrees) are closed in map space through the corresponding pole line
        """
        voronoi_sites = np.asarray(voronoi_sites, dtype=float)
        if lon_bounds is None:
            lon_bounds = (0.0, 360.0) if voronoi_sites[:, 0].max() > 180 else (-180.0, 180.0)
        frame = shapely.geometry.box(lon_bounds[0], -90.0, lon_bounds[1], 90.0)
        if clip_polygon is None:
            clip_target = frame
        elif isinstance(clip_polygon, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            clip_target = clip_polygon
        else:
            clip_target = shapely.geometry.Polygon(clip_polygon)

        boundaries = Voronoi2DSphere._cell_boundaries_xyz(voronoi_sites)
        sites_xyz = Voronoi2DSphere.lonlat_to_xyz(voronoi_sites)
        # the north (south) pole belongs to the cell of the site closest to it
        i_north = int(np.argmax(sites_xyz[:, 2]))
        i_south = int(np.argmin(sites_xyz[:, 2]))

        cell_polygons = []
        for i, boundary in enumerate(boundaries):
            lonlat = Voronoi2DSphere.xyz_to_lonlat(boundary)
            lats = lonlat[:, 1]
            # cumulative unwrap along the (closed) boundary
            closed_lons = np.radians(np.append(lonlat[:, 0], lonlat[0, 0]))
            unwrapped = np.degrees(np.unwrap(closed_lons))
            winding = unwrapped[-1] - unwrapped[0]
            lons = unwrapped[:-1]
            if abs(winding) > 180.0:  # cell contains a pole
                if i == i_north:
                    pole_lat = 90.0
                elif i == i_south:
                    pole_lat = -90.0
                else:  # numerical safeguard: decide from the cell's mean latitude
                    pole_lat = 90.0 if lats.mean() > 0 else -90.0
                ring_lons = np.concatenate((lons, [unwrapped[-1], unwrapped[-1], unwrapped[0]]))
                ring_lats = np.concatenate((lats, [lats[0], pole_lat, pole_lat]))
            else:
                ring_lons, ring_lats = lons, lats
            base_ring = np.column_stack((ring_lons, ring_lats))

            pieces = []
            for offset in (-720.0, -360.0, 0.0, 360.0, 720.0):
                lon_min, lon_max = base_ring[:, 0].min() + offset, base_ring[:, 0].max() + offset
                if lon_max < lon_bounds[0] or lon_min > lon_bounds[1]:
                    continue
                shifted = shapely.geometry.Polygon(base_ring + [offset, 0.0])
                if not shifted.is_valid:
                    shifted = shifted.buffer(0)
                piece = _polygonal_only(shifted.intersection(clip_target))
                if not piece.is_empty:
                    pieces.append(piece)
            cell_polygons.append(
                _polygonal_only(shapely.ops.unary_union(pieces))
                if pieces
                else shapely.geometry.Polygon()
            )
        return cell_polygons, lon_bounds

    @staticmethod
    def plot_tessellation(
        voronoi_sites: np.ndarray,
        param_values: np.ndarray = None,
        ax=None,
        clip_polygon=None,
        lon_bounds: Tuple[Number, Number] = None,
        cmap="viridis",
        norm=None,
        vmin=None,
        vmax=None,
        colorbar=True,
        edgecolor="k",
        linewidth=0.5,
        voronoi_sites_kwargs=None,
        **kwargs,
    ):
        """displays the Voronoi tessellation on a longitude-latitude
        (equirectangular) map, drawing the exact spherical Voronoi cells
        (computed through ``scipy.spatial.SphericalVoronoi``) as polygons.
        Cells crossing the longitude seam and cells containing a pole are
        rendered correctly, enabling global tessellations to be displayed;
        the tessellation can also be clipped to an arbitrary region of
        interest through ``clip_polygon``

        Parameters
        ----------
        voronoi_sites : np.ndarray of shape (m, 2)
            Voronoi-site positions, i.e. longitude-latitude pairs in degrees
        param_values: np.ndarray, optional
            parameter values associated with each Voronoi cell. These could
            represent the physical property inferred in each cell of the
            discretized medium. If None, only the cell boundaries are drawn
        ax : matplotlib.axes.Axes, optional
            an optional Axes object to plot on. This may be a Cartopy
            ``GeoAxes``: in that case, pass the appropriate ``transform``
            (e.g. ``transform=cartopy.crs.PlateCarree()``) through ``kwargs``
        clip_polygon : Union[np.ndarray, shapely.geometry.Polygon, shapely.geometry.MultiPolygon], optional
            region of interest to which the displayed tessellation is
            clipped, expressed in the same longitude convention as the
            Voronoi sites
        lon_bounds : Tuple[Number, Number], optional
            longitude range of the map frame. By default, (-180, 180), or
            (0, 360) when the site longitudes exceed 180 degrees (see the
            argument ``lon_shift`` of this class)
        cmap : Union[str, matplotlib.colors.Colormap]
            the Colormap instance or registered colormap name used to map scalar
            data to colors
        norm : Union[str, matplotlib.colors.Normalize]
            the normalization method used to scale scalar data to the [0, 1]
            range before mapping to colors using ``cmap``. By default, a linear
            scaling is used, mapping the lowest value to 0 and the highest to 1.
        vmin, vmax : Number
            minimum and maximum values used to create the colormap. Ignored
            when ``norm`` is given
        colorbar : bool
            whether to draw a colorbar, by default True
        edgecolor : color
            color of the Voronoi-cell edges, by default black
        linewidth : Number
            line width of the Voronoi-cell edges, by default 0.5
        voronoi_sites_kwargs : dict, optional
            when given, the Voronoi nuclei are displayed, styled by these
            keyword arguments (passed to ``matplotlib.pyplot.plot``). By
            default (None), the nuclei are not displayed
        kwargs : dict, optional
            additional keyword arguments passed to ``matplotlib.axes.Axes.fill``
            (e.g., ``transform``, ``alpha``, ``zorder``)

        Notes
        -----
        At least four non-degenerate sites are required by
        :class:`scipy.spatial.SphericalVoronoi`. Polygon holes are preserved by
        the clipping geometry but are not currently rendered by ``Axes.fill``.
        Unlike the straight Cartesian edges of :class:`Voronoi2D`, spherical
        edges must be sampled for display. This is handled internally with a
        fixed one-degree maximum angular spacing and does not affect the
        tessellation geometry.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object containing the plot
        cbar : Union[Colorbar, None]
            The Colorbar object associated with the tessellation
        """
        cell_polygons, lon_bounds = Voronoi2DSphere._cell_map_polygons(
            voronoi_sites,
            clip_polygon=clip_polygon,
            lon_bounds=lon_bounds,
        )
        if ax is None:
            _, ax = plt.subplots()

        cbar = None
        if param_values is not None:
            param_values = np.asarray(param_values)
            vmin = vmin if vmin is not None else param_values.min()
            vmax = vmax if vmax is not None else param_values.max()
            norm = norm if norm is not None else plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = cmap if isinstance(cmap, mpl.colors.Colormap) else mpl.colormaps[cmap]
            facecolors = cmap(norm(param_values))
        else:
            facecolors = None

        for i, cell in enumerate(cell_polygons):
            if cell.is_empty:
                continue
            geoms = cell.geoms if isinstance(cell, shapely.geometry.MultiPolygon) else [cell]
            facecolor = facecolors[i] if facecolors is not None else "none"
            for geom in geoms:
                boundary_x, boundary_y = geom.exterior.xy
                ax.fill(
                    boundary_x,
                    boundary_y,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    **kwargs,
                )
        if param_values is not None and colorbar:
            cbar = plt.colorbar(
                mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, aspect=35, pad=0.02
            )
            cbar.set_label("Parameter Values")

        if voronoi_sites_kwargs is not None:  # nuclei not displayed by default
            voronoi_sites_kwargs = dict(voronoi_sites_kwargs)
            sites_style = {
                "color": voronoi_sites_kwargs.pop("color", voronoi_sites_kwargs.pop("c", "k")),
                "marker": voronoi_sites_kwargs.pop("marker", "o"),
                "ms": voronoi_sites_kwargs.pop("ms", voronoi_sites_kwargs.pop("markersize", 2)),
                "ls": "",
                "lw": 0,
            }
            sites_style.update(voronoi_sites_kwargs)
            if "transform" in kwargs:
                sites_style.setdefault("transform", kwargs["transform"])
            ax.plot(voronoi_sites[:, 0], voronoi_sites[:, 1], **sites_style)
        if "transform" not in kwargs:  # on GeoAxes, the extent is set by the user
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            if clip_polygon is not None:
                clip_bounds = (
                    clip_polygon.bounds
                    if hasattr(clip_polygon, "bounds")
                    else shapely.geometry.Polygon(clip_polygon).bounds
                )
                ax.set_xlim(clip_bounds[0] - 1, clip_bounds[2] + 1)
                ax.set_ylim(clip_bounds[1] - 1, clip_bounds[3] + 1)
            else:
                ax.set_xlim(*lon_bounds)
                ax.set_ylim(-90, 90)
        return ax, cbar

    @staticmethod
    def plot_tessellation_3d(
        voronoi_sites: np.ndarray,
        param_values: np.ndarray = None,
        ax=None,
        center_lonlat: Tuple[Number, Number] = None,
        surface_spacing_deg: Number = 2.0,
        cmap="viridis",
        norm=None,
        vmin=None,
        vmax=None,
        colorbar=True,
        edgecolor="k",
        linewidth=0.5,
        alpha=1.0,
        voronoi_sites_kwargs=None,
        **kwargs,
    ):
        """Display a global Voronoi tessellation on a three-dimensional sphere.

        Cell boundaries are the exact geodesic boundaries computed through
        :class:`scipy.spatial.SphericalVoronoi` and sampled internally for
        display. When ``param_values`` are supplied, cell colours are rendered
        on a regular spherical surface mesh; decrease ``surface_spacing_deg``
        for a finer representation of the colour transitions.

        This method uses Matplotlib's built-in 3-D toolkit and introduces no
        dependency beyond those already required by BayesBay.

        Parameters
        ----------
        voronoi_sites : np.ndarray of shape (m, 2)
            Voronoi-site ``(longitude, latitude)`` pairs in degrees. Longitude
            is periodic: any finite value is accepted and interpreted modulo
            360 degrees. Latitude must lie within [-90, 90] degrees. At least
            four non-degenerate sites are required.
        param_values : np.ndarray of shape (m,), optional
            Parameter value associated with each cell. If omitted, only the
            cell boundaries are drawn.
        ax : matplotlib.axes.Axes, optional
            Matplotlib 3-D axes. A new one is created when omitted.
        center_lonlat : Tuple[Number, Number], optional
            ``(longitude, latitude)`` in degrees to place at the centre of the
            initial view. Longitude is periodic and is normalized to
            [-180, 180); latitude must lie within [-90, 90]. For example,
            ``(12.5, 42.5)`` centres the globe on Italy. This argument only
            sets the initial camera orientation; the globe can still be
            rotated interactively. When omitted, the existing orientation of
            ``ax`` (or Matplotlib's default for a new axes) is preserved.
        surface_spacing_deg : Number, optional
            Approximate angular spacing of the coloured surface mesh. Smaller
            values reduce pixelation along colour transitions but increase the
            number of surface patches approximately with the inverse square of
            the spacing. Default is 2 degrees.
        cmap, norm, vmin, vmax
            Matplotlib colour-mapping arguments.
        colorbar : bool, optional
            Whether to add a colourbar when ``param_values`` are supplied.
        edgecolor : color, optional
            Cell-boundary colour.
        linewidth : Number, optional
            Cell-boundary line width.
        alpha : Number, optional
            Opacity of the coloured sphere surface.
        voronoi_sites_kwargs : dict, optional
            Styling passed to :meth:`matplotlib.axes.Axes.scatter` when
            drawing the Voronoi sites. Sites are hidden by default.
        **kwargs
            Additional arguments passed to
            :meth:`matplotlib.axes.Axes.plot_surface`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The three-dimensional axes containing the plot.
        cbar : Union[Colorbar, None]
            The colourbar, or ``None`` when it was not requested.
        """
        voronoi_sites = np.asarray(voronoi_sites, dtype=float)
        if voronoi_sites.ndim != 2 or voronoi_sites.shape[1] != 2:
            raise ValueError("`voronoi_sites` should have shape (m, 2)")
        if not np.isfinite(voronoi_sites).all():
            raise ValueError("`voronoi_sites` should contain only finite values")
        if np.any((voronoi_sites[:, 1] < -90) | (voronoi_sites[:, 1] > 90)):
            raise ValueError(
                "Voronoi-site latitudes should lie within [-90, 90] degrees"
            )
        if center_lonlat is not None:
            try:
                center_lonlat = np.asarray(center_lonlat, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "`center_lonlat` should be a (longitude, latitude) pair "
                    "in degrees"
                ) from exc
            if center_lonlat.shape != (2,) or not np.isfinite(center_lonlat).all():
                raise ValueError(
                    "`center_lonlat` should be a pair of finite values: "
                    "(longitude, latitude) in degrees"
                )
            if not -90 <= center_lonlat[1] <= 90:
                raise ValueError(
                    "The latitude in `center_lonlat` should lie within "
                    "[-90, 90] degrees"
                )
        if (
            not np.isscalar(surface_spacing_deg)
            or not np.isfinite(surface_spacing_deg)
            or surface_spacing_deg <= 0
        ):
            raise ValueError(
                "`surface_spacing_deg` should be a finite positive scalar"
            )

        boundaries = Voronoi2DSphere._cell_boundaries_xyz(voronoi_sites)
        sites_xyz = Voronoi2DSphere.lonlat_to_xyz(voronoi_sites)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        elif not hasattr(ax, "plot_surface"):
            raise ValueError("`ax` should be a Matplotlib 3-D axes")
        if center_lonlat is not None:
            longitude, latitude = center_lonlat
            longitude = (longitude + 180.0) % 360.0 - 180.0
            ax.view_init(elev=latitude, azim=longitude)

        cbar = None
        if param_values is not None:
            param_values = np.asarray(param_values)
            if param_values.ndim != 1 or len(param_values) != len(voronoi_sites):
                raise ValueError(
                    "`param_values` should contain one value per Voronoi site"
                )
            vmin = vmin if vmin is not None else param_values.min()
            vmax = vmax if vmax is not None else param_values.max()
            norm = norm if norm is not None else plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = (
                cmap if isinstance(cmap, mpl.colors.Colormap) else mpl.colormaps[cmap]
            )

            n_lon = max(4, int(math.ceil(360 / surface_spacing_deg)) + 1)
            n_lat = max(3, int(math.ceil(180 / surface_spacing_deg)) + 1)
            lon = np.linspace(-180, 180, n_lon)
            lat = np.linspace(-90, 90, n_lat)
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            surface_xyz = Voronoi2DSphere.lonlat_to_xyz(
                np.stack((lon_grid, lat_grid), axis=-1)
            )
            nearest = scipy.spatial.KDTree(sites_xyz).query(
                surface_xyz.reshape(-1, 3)
            )[1].reshape(surface_xyz.shape[:2])
            facecolors = cmap(norm(param_values[nearest]))
            # Mark surface-mesh patches crossed by a cell interface. This
            # keeps boundaries visible under Matplotlib's 3-D depth sorting;
            # the geodesic curves plotted below provide the exact location.
            interfaces = (
                (nearest[:-1, :-1] != nearest[1:, :-1])
                | (nearest[:-1, :-1] != nearest[:-1, 1:])
            )
            facecolors[:-1, :-1][interfaces] = mpl.colors.to_rgba(edgecolor)
            surface_style = {
                "rstride": 1,
                "cstride": 1,
                "linewidth": 0,
                "antialiased": False,
                "shade": False,
            }
            surface_style.update(kwargs)
            ax.plot_surface(
                surface_xyz[..., 0],
                surface_xyz[..., 1],
                surface_xyz[..., 2],
                facecolors=facecolors,
                alpha=alpha,
                **surface_style,
            )
            if colorbar:
                cbar = plt.colorbar(
                    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                    ax=ax,
                    aspect=35,
                    pad=0.02,
                    shrink=0.7,
                )
                cbar.set_label("Parameter Values")

        # Move boundaries and sites very slightly above the surface to avoid
        # z-fighting with the coloured mesh.
        plot_radius = 1.01
        for boundary in boundaries:
            boundary = boundary * plot_radius
            closed = np.vstack((boundary, boundary[0]))
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                closed[:, 2],
                color=edgecolor,
                linewidth=linewidth,
            )

        if voronoi_sites_kwargs is not None:
            sites_style = {"color": "k", "s": 10}
            sites_style.update(voronoi_sites_kwargs)
            points = sites_xyz * (plot_radius + 0.002)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], **sites_style)

        limit = 1.05
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return ax, cbar
