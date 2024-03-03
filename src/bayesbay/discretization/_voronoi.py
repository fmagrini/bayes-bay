from bisect import bisect_left
import math
from typing import Tuple, Union, List, Dict, Callable
from numbers import Number
import random
import numpy as np
import scipy.spatial
import shapely.geometry
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..parameterization._parameter_space import ParameterSpace
from ..perturbations._param_values import ParamPerturbation
from ._discretization import Discretization
from ..exceptions import DimensionalityException
from ..prior import Prior
from .._state import State, ParameterSpaceState
from .._utils_1d import (
    interpolate_depth_profile,
    interpolate_nearest_1d, 
    compute_voronoi1d_cell_extents, 
    insert_1d,
    delete_1d,
    nearest_neighbour_1d,
    
)


SQRT_TWO_PI = math.sqrt(2 * math.pi)


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
    n_dimensions : Number, optional
        number of dimensions. None (default) results in a trans-dimensional
        discretization, with the dimensionality of the parameter space allowed
        to vary in the range ``n_dimensions_min``-``n_dimensions_max``
    n_dimensions_min, n_dimensions_max : Number, optional
        minimum and maximum number of dimensions, by default 1 and 10. These
        parameters are ignored if ``n_dimensions`` is not None, i.e. if the
        discretization is not trans-dimensional
    n_dimensions_init_range : Number, optional
        percentage of the range `n_dimensions_min`` - ``n_dimensions_max`` used to
        initialize the number of dimensions (0.3. by default). For example, if 
        ``n_dimensions_min`` = 1, ``n_dimensions_max`` = 10, and 
        ``n_dimensions_init_range`` = 0.5,
        the maximum number of dimensions at the initialization is
            
            int((n_dimensions_max - n_dimensions_min) * n_dimensions_init_range + n_dimensions_max)
            
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
            vmax=vmax
        )
        self.vmin = vmin
        self.vmax = vmax
        msg = "The %s number of Voronoi cells, "
        if n_dimensions is not None:
            assert n_dimensions > 0, msg % "minimum" + "`n_dimensions`, should be greater than zero"
            assert isinstance(n_dimensions, int), msg % "minimum" + "`n_dimensions`, should be an integer"
            assert isinstance(n_dimensions, int), msg % "maximum" + "`n_dimensions`, should be an integer"

    def sample(self):
        """draws a Voronoi-site position at random within the discretization domain"""
        return np.random.uniform(self.vmin, self.vmax, self.spatial_dimensions)
    
    def initialize(self) -> ParameterSpaceState:
        """initializes the parameter space linked to the Voronoi tessellation

        Returns
        -------
        ParameterSpaceState
            an initial parameter space state
        """
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
        voronoi_sites = np.array([self.sample() for cell in range(n_voronoi_cells)])
        if self.spatial_dimensions == 1:
            voronoi_sites = np.sort(np.ravel(voronoi_sites))

        # initialize parameter values
        parameter_vals = {"discretization": voronoi_sites}
        for name, param in self.parameters.items():
            parameter_vals[name] = param.initialize(voronoi_sites)
        return ParameterSpaceState(n_voronoi_cells, parameter_vals)  
    
    def _perturb_site(self, site: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        """perturbes a Voronoi  site
        
        Parameters
        ----------
        site : Union[Number, np.ndarray]
            Voronoi site position.

        Returns
        -------
        Union[Number, np.ndarray]
            perturbed Voronoi site position
        """        
        while True:
            if self.spatial_dimensions == 1:
                random_deviate = random.normalvariate(0, self.perturb_std)
                new_site = site + random_deviate
                if new_site >= self.vmin and new_site <= self.vmax:
                    return new_site
            else: 
                random_deviate = np.random.normal(
                    0, self.perturb_std, self.spatial_dimensions
                    )
                new_site = site + random_deviate
                if all((new_site >= self.vmin) & (new_site <= self.vmax)):
                    return new_site

    def perturb_value(self, old_ps_state: ParameterSpaceState, isite: int):
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
        new_sites = old_sites.copy()
        new_sites[isite] = new_site
        if self.spatial_dimensions == 1:
            isort = np.argsort(new_sites)
            new_sites = new_sites[isort]
            new_values = {"discretization": new_sites}
            log_prior_ratio = 0
            for param_name, param in self.parameters.items():
                values = old_ps_state[param_name]
                if param.position is not None:
                    log_prior_old = param.log_prior(values[isite], old_site)
                    log_prior_new = param.log_prior(values[isite], new_site)
                    log_prior_ratio += log_prior_new - log_prior_old
                new_values[param_name] = values[isort]
        else:
            new_values = {"discretization": new_sites}
            log_prior_ratio = 0
            for param_name, param in self.parameters.items():
                values = old_ps_state[param_name]
                if param.position is not None:
                    log_prior_old = param.log_prior(values[isite], old_site)
                    log_prior_new = param.log_prior(values[isite], new_site)
                    log_prior_ratio += log_prior_new - log_prior_old
                new_values[param_name] = values
            
        new_ps_state = ParameterSpaceState(old_ps_state.n_dimensions, new_values)
        return new_ps_state, log_prior_ratio # log_proposal_ratio=0 and log_det_jacobian=0

    def _initialize_newborn_params(
            self, 
            new_site: Union[Number, np.ndarray], 
            old_sites: np.ndarray, 
            param_space_state: ParameterSpaceState
            ):
        """initialize the parameter values in the newborn dimension
    
        Parameters
        ----------
        new_site : Union[Number, np.ndarray]
            position of the newborn Voronoi site
        old_sites : np.ndarray
            all positions of the current Voronoi sites
        param_space_state : ParameterSpaceState
            current parameter space state
    
        Returns
        -------
        Dict[str, float]
            key value pairs that map parameter names to values of the ``new_site``
        """
        if self.birth_from == 'prior':
            return self._initialize_params_from_prior(
                new_site, old_sites, param_space_state
                )
        return self._initialize_params_from_neighbour(
            new_site, old_sites, param_space_state
            )

    def _initialize_params_from_prior(
            self, 
            new_site: Union[Number, np.ndarray], 
            old_sites: np.ndarray, 
            param_space_state: ParameterSpaceState
            ) -> Tuple[Dict[str, np.ndarray], None]:
        """initialize the newborn dimension by randomly drawing parameter values
        from the prior
    
        Parameters
        ----------
        new_site : Union[Number, np.ndarray]
            position of the newborn Voronoi site
        old_sites : np.ndarray
            all positions of the current Voronoi sites
        param_space_state : ParameterSpaceState
            current parameter space state
    
        Returns
        -------
        Tuple[Dict[str, np.ndarray], None]
            key value pairs that map parameter names to values of the ``new_site``
        """
        new_born_values = dict()
        for param_name, param in self.parameters.items():
            new_value = param.sample(new_site)
            new_born_values[param_name] = new_value
        return new_born_values, None
    
    def _initialize_params_from_neighbour(
        self, 
        new_site: Union[Number, np.ndarray], 
        old_sites: np.ndarray, 
        param_space_state: ParameterSpaceState
        ) -> Tuple[Dict[str, np.ndarray], Number]:
        """initialize the newborn parameter values by perturbing the nearest 
        Voronoi cell
    
        Parameters
        ----------
        new_site : Union[Number, np.ndarray]
            position of the newborn Voronoi cell
        old_sites : np.ndarray
            all positions of the current Voronoi cells
        param_space_state : ParameterSpaceState
            current parameter space state
    
        Returns
        -------
        Tuple[Dict[str, np.ndarray], Number]
            key value pairs that map parameter names to values of the ``new_site``
            and the index of the Voronoi neighbour
        """
        if self.spatial_dimensions == 1:
            isite = nearest_neighbour_1d(
                xp=new_site, x=old_sites, xlen=old_sites.size
                )
        else:
            isite = np.argmin(np.linalg.norm(old_sites - new_site, axis=1))
        new_born_values = dict()
        for param_name, param in self.parameters.items():
            old_values = param_space_state[param_name]
            new_value, _ = param.perturb_value(old_values[isite], new_site)
            new_born_values[param_name] = new_value
        return new_born_values, isite

    def _log_probability_ratio_birth(
            self, 
            old_isite: Number, 
            old_ps_state: ParameterSpaceState, 
            new_isite: Number, 
            new_ps_state: ParameterSpaceState
            ):
        if self.birth_from == 'prior':
            return 0
        return self._log_probability_ratio_birth_from_neighbour(
            old_isite, old_ps_state, new_isite, new_ps_state
        )

    def _log_probability_ratio_birth_from_neighbour(
            self, 
            old_isite: Number, 
            old_ps_state: ParameterSpaceState, 
            new_isite: Number, 
            new_ps_state: ParameterSpaceState
    ):
        old_site = old_ps_state["discretization"][old_isite]
        new_site = new_ps_state["discretization"][new_isite]
        log_prior_ratio = 0
        log_proposal_ratio = 0
        for param_name, param in self.parameters.items():
            new_value = new_ps_state[param_name][new_isite]
            log_prior_ratio += param.log_prior(new_value, new_site)
            
            old_value = old_ps_state[param_name][old_isite]
            perturb_std = param.get_perturb_std(old_site)
            log_proposal_ratio += (
                math.log(perturb_std * SQRT_TWO_PI)
                + (new_value - old_value) ** 2 / (2 * perturb_std**2)
            )
        return log_prior_ratio + log_proposal_ratio # log_det_jacobian is 1 
    
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
        """
        # prepare for birth perturbation
        n_cells = old_ps_state.n_dimensions
        if n_cells == self._n_dimensions_max:
            raise DimensionalityException("Birth")
        # randomly choose a new Voronoi site position
        new_site = self.sample()
        old_sites = old_ps_state["discretization"]
        initialized_values, i_nearest = self._initialize_newborn_params(
            new_site, old_sites, old_ps_state
        )
        new_values = dict()
        if self.spatial_dimensions == 1:
            idx_insert = bisect_left(old_sites, new_site)
            new_sites = insert_1d(old_sites, idx_insert, new_site)
            new_values["discretization"] = new_sites
            for name, value in initialized_values.items():
                old_values = old_ps_state[name]
                new_values[name] = insert_1d(old_values, idx_insert, value)
        else:
            idx_insert = n_cells
            new_sites = np.row_stack((old_sites, new_site))
            new_values["discretization"] = new_sites
            for name, value in initialized_values.items():
                old_values = old_ps_state[name]
                new_values[name] = np.append(old_values, value)
        new_ps_state = ParameterSpaceState(n_cells + 1, new_values)
        return new_ps_state, self._log_probability_ratio_birth(
            i_nearest, old_ps_state, idx_insert, new_ps_state
        )
    
    def _log_probability_ratio_death(
            self, 
            iremove: Number, 
            old_ps_state: ParameterSpaceState, 
            new_ps_state: ParameterSpaceState
            ):
        if self.birth_from == 'prior':
            return 0
        return self._log_probability_ratio_death_from_neighbour(
            iremove, old_ps_state, new_ps_state
        )

    def _log_probability_ratio_death_from_neighbour(
            self, 
            iremove: Number, 
            old_ps_state: ParameterSpaceState, 
            new_ps_state: ParameterSpaceState
            ):
        old_sites = old_ps_state["discretization"]
        new_sites = new_ps_state["discretization"]
        if self.spatial_dimensions == 1:
            i_nearest = nearest_neighbour_1d(
                xp=old_sites[iremove], x=new_sites, xlen=new_sites.size
                )
        else:
            i_nearest = np.argmin(
                np.linalg.norm(new_sites - old_sites[iremove], axis=1)
                )

        return -self._log_probability_ratio_birth(
            i_nearest, new_ps_state, iremove, old_ps_state
        )
    
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
            raise DimensionalityException("Death")
        # randomly choose an existing Voronoi site to kill
        iremove = random.randint(0, n_cells - 1)
        # remove parameter values for the removed site
        new_values = dict()
        for name, old_values in old_ps_state.param_values.items():
            if self.spatial_dimensions == 1:
                new_values[name] = delete_1d(old_values, iremove)
            else:
                new_values[name] = np.delete(old_values, iremove, axis=0)
        new_ps_state = ParameterSpaceState(n_cells - 1, new_values) 
        return new_ps_state, self._log_probability_ratio_death(
            iremove, old_ps_state, new_ps_state
        )

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
        ParameterSpace._init_perturbation_funcs(self)
        self._perturbation_funcs.append(ParamPerturbation(self.name, [self]))
        self._perturbation_weights.append(1)

    @property
    def perturbation_functions(self) -> List[Callable[[State], Tuple[State, Number]]]:
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
        :meth:`perturbation_functions` that determines the probability of each of them
        to be chosen during each step
        
        The weights are not normalized and have the following default values:
        
        - Birth/Death perturbations: 1
        - Parameter values perturbation: 3
        - Voronoi site perturbation: 1
        """
        return self._perturbation_weights
    

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
    n_dimensions : Number, optional
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
            
            int((n_dimensions_max - n_dimensions_min) * n_dimensions_init_range + n_dimensions_max)
            
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
            birth_from: str = "neighbour"  # either "neighbour" or "prior"
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
            birth_from=birth_from
        )      
        
    @staticmethod
    def compute_cell_extents(voronoi_sites: np.ndarray, lb=0, ub=-1, fill_value=0):
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
            extents. Negative values for `lb` or `ub` denote an unbounded cell.
            The extent of an unbounded cell is set to `fill_value`

        fill_value : float
            Value attributed to unbounded Voronoi cells

        Returns
        -------
        np.ndarray
            Voronoi-cell extents

        Examples
        --------
        >>> voronoi_sites = np.array([2, 5.5, 8, 10])

        >>> Voronoi1D.compute_cell_extents(voronoi_sites, lb=0, ub=-1, fill_value=np.nan)
        array([3.75, 3.  , 2.25,  nan])

        >>> Voronoi1D.compute_cell_extents(voronoi_sites, lb=-1, ub=-1, fill_value=np.nan)
        array([ nan, 3.  , 2.25,  nan])

        >>> Voronoi1D.compute_cell_extents(voronoi_sites, lb=0, ub=15, fill_value=np.nan)
        array([3.75, 3.  , 2.25, 6.  ])
        """
        return compute_voronoi1d_cell_extents(
            voronoi_sites, lb=lb, ub=ub, fill_value=fill_value
        )
    
    @staticmethod
    def interpolate_tessellation(
        voronoi_cells, 
        param_values, 
        interp_positions, 
        input_type='nuclei'
    ):
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
        if input_type == 'nuclei':
            return interpolate_nearest_1d(interp_positions, voronoi_cells, param_values)
        elif input_type == 'extents':
            return interpolate_depth_profile(
                np.array(voronoi_cells), 
                np.array(param_values), 
                interp_positions
                )
        raise ValueError("`input_type` should either be 'nuclei' or 'extents'")

    @staticmethod
    def _interpolate_tessellations(
        samples_voronoi_cells, 
        samples_param_values, 
        interp_positions, 
        input_type='nuclei'
    ):
        interp_params = np.zeros((len(samples_param_values), len(interp_positions)))
        for i, (sample_cells, sample_values) in enumerate(
            zip(samples_voronoi_cells, samples_param_values)
        ):
            interp_params[i, :] = Voronoi1D.interpolate_tessellation(
                np.array(sample_cells), 
                np.array(sample_values), 
                interp_positions,
                input_type=input_type
            )
        return interp_params

    @staticmethod
    def get_tessellation_statistics(
        samples_voronoi_cells: list,
        samples_param_values: list,
        interp_positions: np.ndarray,
        percentiles: tuple = (10, 90),
        input_type: str = 'nuclei'
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
            a dictionary with these keys: "mean", "median", "std" and "percentile"
        """
        interp_params = Voronoi1D._interpolate_tessellations(
            samples_voronoi_cells, 
            samples_param_values, 
            interp_positions,
            input_type=input_type
        )
        statistics = {
            "mean": np.mean(interp_params, axis=0),
            "median": np.median(interp_params, axis=0),
            "std": np.std(interp_params, axis=0),
            "percentiles": np.percentile(interp_params, percentiles, axis=0),
        }
        return statistics

    @staticmethod
    def plot_tessellation_density(
        samples_voronoi_cells: np.ndarray,
        samples_param_values: np.ndarray,
        position_bins: Union[int, np.ndarray] = 100, 
        param_value_bins: Union[int, np.ndarray] = 100,
        ax=None,
        swap_xy_axes=True,
        input_type='nuclei',
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
        bounds : tuple, optional
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
        if input_type not in ['nuclei', 'extents']:
            raise ValueError("`input_type` should either be 'nuclei' or 'extents'")
        if isinstance(position_bins, int):
            lb = 0
            if input_type == 'nuclei':
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
            input_type=input_type
        )
        if ax is None:
            _, ax = plt.subplots()
            
        cax = ax.hist2d(
            interp_param_values.ravel(), 
            np.tile(interp_positions, interp_param_values.shape[0]), 
            bins=(param_value_bins, len(interp_positions)), 
            **kwargs
        )
        # colorbar (for the histogram density)
        cbar = plt.colorbar(cax[3], ax=ax, aspect=35, pad=0.02)
        cbar.set_label("Counts")
        if ax.get_ylim()[0] < ax.get_ylim()[1] and swap_xy_axes:
            ax.invert_yaxis()
        if swap_xy_axes:
            if not ax.get_xlabel():
                ax.set_xlabel("Parameter values")
        else:
            if not ax.get_ylabel():
                ax.set_ylabel("Parameter values")
        return ax, cbar

    @staticmethod
    def plot_interface_hist(
        samples_voronoi_cells: np.ndarray,
        bins=100,
        ax=None,
        swap_xy_axes=True,
        input_type='nuclei',
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

        Returns
        -------
        matplotlib.axes.Axes
        """
        positions = []
        for voronoi_cells in samples_voronoi_cells:
            if input_type == 'nuclei':
                cells_extent = Voronoi1D.compute_cell_extents(voronoi_cells)
            elif input_type == 'extents':
                cells_extent = voronoi_cells
            positions.extend(np.cumsum(cells_extent)[:-1])

        if ax is None:
            _, ax = plt.subplots()
        hist, edges = np.histogram(positions, bins=bins, density=True)
        if swap_xy_axes:
            ax.barh(edges[:-1], 
                    hist, 
                    height=np.diff(edges), 
                    align="edge",  
                    **kwargs)
            if ax.get_ylim()[0] < ax.get_ylim()[1]:
                ax.invert_yaxis()
            if not ax.get_xlabel():
                ax.set_xlabel("Probability density")
        else:
            plt.bar(edges[:-1], hist, width=np.diff(edges), align='edge')
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
        input_type='nuclei',
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
        if input_type == 'nuclei':
            voronoi_cell_extents = Voronoi1D.compute_cell_extents(voronoi_cells, lb=lb)
        elif input_type == 'extents':
            voronoi_cell_extents = voronoi_cells
        else:
            raise ValueError("`input_type` should either be 'nuclei' or 'extents'")
        interface_positions = np.cumsum(voronoi_cell_extents[:-1])
        if ub is not None:
            assert ub > interface_positions[-1], ("`bounds[1]` should be greater"
                                            " than the sum of Voronoi"
                                            f" cell extents (here, {interface_positions[-1]})")
            end_position = ub
        else:
            end_position = interface_positions[-1] + max(voronoi_cell_extents)
            
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
            "color": kwargs.pop(
                "color", kwargs.pop("c", "blue")
            ),  # Fixed color for the sample lines
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
        input_type='nuclei',
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
        if input_type == 'nuclei':
            samples_voronoi_cell_extents = [
                Voronoi1D.compute_cell_extents(nuclei, lb=lb) \
                    for nuclei in samples_voronoi_cells]
        elif input_type == 'extents':
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
            "color": kwargs.pop(
                "color", kwargs.pop("c", "blue")
            ),  # Fixed color for the sample lines
        }
        sample_style.update(kwargs)  # Override with any provided kwargs

        for extents, values in zip(
            samples_voronoi_cell_extents, samples_param_values
        ):
            Voronoi1D.plot_tessellation(extents, 
                                        values, 
                                        **sample_style, 
                                        ax=ax,
                                        bounds=bounds,
                                        input_type='extents')

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
        input_type: str = 'nuclei',
        swap_xy_axes: bool = True
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
            input_type=input_type
        ) 

        if ax is None:
            _, ax = plt.subplots()
        _plot(interp_positions, 
              statistics["mean"], 
              ax,
              swap_xy_axes=swap_xy_axes, 
              color='b', 
              label="Mean")
        _plot(interp_positions, 
              statistics["mean"]-statistics["std"], 
              ax,
              swap_xy_axes=swap_xy_axes, 
              color='b',
              ls='--',
              label='STD')
        _plot(interp_positions, 
              statistics["mean"]+statistics["std"],
              ax,
              swap_xy_axes=swap_xy_axes, 
              color='b',
              ls='--')
        _plot(interp_positions,
              statistics["median"],
              ax,
              swap_xy_axes=swap_xy_axes,
              color="orange",
              label="Median")
        _plot(interp_positions,
              statistics["percentiles"][0],
              ax,
              swap_xy_axes=swap_xy_axes,
              color="orange",
              ls='--',
              label=f'{percentiles[0]}-{percentiles[1]}th Perc.')
        _plot(interp_positions,
              statistics["percentiles"][1],
              ax,
              swap_xy_axes=swap_xy_axes,
              color="orange",
              ls='--')
        if ax.get_ylim()[0] < ax.get_ylim()[1] and swap_xy_axes:
            ax.invert_yaxis()
        ax.legend()
        return ax


class Voronoi2D(Voronoi):
    r"""Utility class for Voronoi tessellation in 1D

    Parameters
    ----------
    name : str
        name attributed to the Voronoi tessellation, for display and storing 
        purposes
    vmin, vmax : Union[Number, np.ndarray]
        minimum/maximum value bounding each dimension. Ignored when
        ``polygon`` is not ``None``
    polygon: Union[np.ndarray, shapely.geometry.Polygon], optional
        polygon defining the domain of the Voronoi tessellation; Voronoi sites
        outside this polygon are not allowed
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the Voronoi
        sites in each dimension. 
    n_dimensions : Number, optional
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
            
            int((n_dimensions_max - n_dimensions_min) * n_dimensions_init_range + n_dimensions_max)
            
    parameters : List[Parameter], optional
        a list of free parameters, by default None
    birth_from : {"prior", "neighbour"}, optional
        whether to initialize the free parameters associated with the newborn 
        Voronoi cell by randomly drawing from their prior or by perturbing the 
        value found in the nearest Voronoi cell (default)
    compute_kdtree : bool
        whether to compute a kd-tree for quick nearest-neighbor lookup at every
        perturbation of the discretization
    """
    def __init__(        
            self,
            name: str,
            vmin: Number = None,
            vmax: Number = None,
            polygon: Union[np.ndarray, shapely.geometry.Polygon] = None,
            perturb_std: Union[Number, np.ndarray] = 1,
            n_dimensions: int = None, 
            n_dimensions_min: int = 2, 
            n_dimensions_max: int = 100, 
            n_dimensions_init_range: Number = 0.3, 
            parameters: List[Prior] = None, 
            birth_from: str = "neighbour",  # either "neighbour" or "prior"
            compute_kdtree: bool = False
        ): 
        assert (vmin is not None and vmax is not None) \
            or polygon is not None, ("Either `vmin`/`vmax` or `polygon`"
            " must not be None to properly define the discretization domain.")
        if polygon is not None:
            polygon = shapely.geometry.Polygon(polygon)
            vmin = polygon.bounds[:2]
            vmax = polygon.bounds[2:]
        self.polygon = polygon
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
            birth_from=birth_from
        )    
        self.compute_kdtree = compute_kdtree

    def sample(self):
        if self.polygon is not None:
            while True:
                new_site = super().sample()
                point = shapely.geometry.Point(new_site)
                if self.polygon.contains(point):
                    return new_site
        return super().sample()
    
    def initialize(self):
        ps_state = super().initialize()
        if self.compute_kdtree:
            return self._add_kdtree_to_ps_state(ps_state)
        return ps_state
    
    def _add_kdtree_to_ps_state(self, ps_state: ParameterSpaceState):
        voronoi_sites = ps_state.get_param_values('discretization')
        kdtree = scipy.spatial.KDTree(voronoi_sites)
        ps_state.save_to_cache('kdtree', kdtree)
        return ps_state

    def _perturb_site(self, site: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        """perturbes a Voronoi  site
        
        Parameters
        ----------
        site : Union[Number, np.ndarray]
            Voronoi site position.

        Returns
        -------
        Union[Number, np.ndarray]
            perturbed Voronoi site position
        """        
        if self.polygon is None:
            return super()._perturb_site(site)
        while True:
            random_deviate = np.random.normal(
                0, self.perturb_std, self.spatial_dimensions
                )
            new_site = site + random_deviate
            point = shapely.geometry.Point(new_site)
            if self.polygon.contains(point):
                return new_site
    
    def perturb_value(self, old_ps_state: ParameterSpaceState, isite: int):
        new_ps_state, log_prior_ratio = super().perturb_value(old_ps_state, isite)
        if self.compute_kdtree:
            new_ps_state = self._add_kdtree_to_ps_state(new_ps_state)
        return new_ps_state, log_prior_ratio
    
    def birth(self, old_ps_state: ParameterSpaceState) -> Tuple[ParameterSpaceState, float]:
        new_ps_state, log_prob_ratio_birth = super().birth(old_ps_state)
        if self.compute_kdtree:
            new_ps_state = self._add_kdtree_to_ps_state(new_ps_state)
        return new_ps_state, log_prob_ratio_birth

    def death(self, old_ps_state: ParameterSpaceState):
        new_ps_state, log_prob_ratio_death = super().death(old_ps_state)
        if self.compute_kdtree:
            new_ps_state = self._add_kdtree_to_ps_state(new_ps_state)
        return new_ps_state, log_prob_ratio_death

    @staticmethod
    def interpolate_tessellation(
        voronoi_sites: np.ndarray, 
        param_values: np.ndarray, 
        interp_positions: np.ndarray
    ):
        r"""nearest neighbour interpolation based on Voronoi-site
        positions and values associated with them.

        Parameters
        ----------
        voronoi_sites : (n, 2) np.ndarray
            the positions of the Voronoi sites
        param_values : (n,) np.ndarray
            the parameter values associated with each Voronoi cell
        query_points : (m, 2) np.ndarray
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
    def _interpolate_tessellations(
        samples_voronoi_sites, 
        samples_param_values, 
        interp_positions
    ):
        interp_params = np.zeros((len(samples_param_values), len(interp_positions)))
        for i, (sample_sites, sample_values) in enumerate(
            zip(samples_voronoi_sites, samples_param_values)
        ):
            interp_params[i, :] = Voronoi2D.interpolate_tessellation(
                np.array(sample_sites), 
                np.array(sample_values), 
                interp_positions
            )
        return interp_params

    @staticmethod
    def get_tessellation_statistics(
        samples_voronoi_cells: list,
        samples_param_values: list,
        interp_positions: np.ndarray,
        percentiles: tuple = (10, 90)
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

        Returns
        -------
        dict
            a dictionary with these keys: "mean", "median", "std" and "percentile"
        """
        interp_params = Voronoi2D._interpolate_tessellations(
            samples_voronoi_cells, 
            samples_param_values, 
            interp_positions
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
        cmap='viridis',
        norm=None,
        vmin=None,
        vmax=None,
        voronoi_sites_kwargs=None,
        voronoi_plot_2d_kwargs=None
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
        voronoi_sites_kwargs : dict
            keyword arguments passed to ``matplotlib.pyplot.plot``, used to
            plot the voronoi nuclei
        voronoi_plot_2d_kwargs : dict
            keyword arguments passed to ``scipy.spatial.voronoi_plot_2d``, used to
            plot the Voronoi interfaces
        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object containing the 2D histogram
        """        
        voronoi_plot_2d_kwargs = \
            voronoi_plot_2d_kwargs if voronoi_plot_2d_kwargs is not None else {}
        interfaces_style = {
            'line_colors': 'k',
            'show_vertices': False,
            'show_points': False,
            'line_width': 1
            }
        interfaces_style.update(voronoi_plot_2d_kwargs)
        voronoi_sites_kwargs = \
            voronoi_sites_kwargs if voronoi_sites_kwargs is not None else {}
        sites_style = {
            'color': voronoi_sites_kwargs.pop('color', voronoi_sites_kwargs.pop('c', 'k')),
            'marker': voronoi_sites_kwargs.pop('marker', 'o'),
            'ms': voronoi_sites_kwargs.pop('ms', voronoi_sites_kwargs.pop('markersize', 2)),
            'ls': '',
            'lw': 0,
            }
        sites_style.update(voronoi_sites_kwargs)

        xmax = np.max(np.abs(voronoi_sites[:, 0]))
        ymax = np.max(np.abs(voronoi_sites[:, 1]))
        sites = np.append(voronoi_sites, 
                          [[xmax*100, ymax*100], 
                           [-xmax*100, ymax*100], 
                           [xmax*100, -ymax*100], 
                           [-xmax*100, -ymax*100]],
                          axis=0)
        
        voronoi = scipy.spatial.Voronoi(sites)
        if ax is None:
            fig, ax = plt.subplots()
        if param_values is not None:
            # make sure scipy.spatial.Voronoi didn't resort the original sites
            isort = [np.flatnonzero(np.all(p == voronoi.points, axis=1)).item() \
                    for i, p in enumerate(sites[:-4])]
            ax, cbar = Voronoi2D._fill_tessellation(
                voronoi, 
                param_values[isort],
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                cmap=cmap
                )
        else:
            cbar = None
        
        scipy.spatial.voronoi_plot_2d(voronoi, ax=ax, **interfaces_style)
        ax.plot(voronoi_sites[:, 0], voronoi_sites[:, 1], **sites_style) 
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(voronoi_sites[:, 0].min(), voronoi_sites[:, 0].max())
        ax.set_ylim(voronoi_sites[:, 1].min(), voronoi_sites[:, 1].max())
        return ax, cbar

    @staticmethod
    def _fill_tessellation(
        voronoi: scipy.spatial.Voronoi, 
        param_values: np.ndarray,
        ax=None,
        vmin=None,
        vmax=None,
        norm=None,
        cmap=None
    ):    
        if ax is None:
            fig, ax = plt.subplots()
        vmin = vmin if vmin is not None else min(param_values)
        vmax = vmax if vmax is not None else max(param_values)
        norm = norm if norm is not None else plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = cmap if isinstance(cmap, mpl.colors.Colormap) \
            else mpl.colormaps[cmap]
        colors = cmap(norm(param_values))
        
        for ipoint, iregion in enumerate(voronoi.point_region):
            region = voronoi.regions[iregion]
            if region and not -1 in region: # Filter out points at infinity
                polygon = [voronoi.vertices[i] for i in region if i >= 0]  
                ax.fill(*zip(*polygon), color=colors[ipoint]) 
        
        # Create a colorbar to show the mapping between values and colors
        cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                          ax=ax,
                          aspect=35, 
                          pad=0.02)
        cbar.set_label('Parameter Values')
        return ax, cbar


