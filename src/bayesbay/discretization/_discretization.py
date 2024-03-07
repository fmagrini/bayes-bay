from abc import abstractmethod
from typing import Union, List, Tuple
from numbers import Number
import math
import numpy as np

from ..prior._prior import Prior
from ..parameterization._parameter_space import ParameterSpace
from .._state import ParameterSpaceState


SQRT_TWO_PI = math.sqrt(2 * math.pi)


class Discretization(Prior, ParameterSpace):
    r"""Low-level class to define a discretization

    Parameters
    ----------
    name : str
        name of the discretization, for display and storing purposes
    spatial_dimensions : int
        number of spatial dimensions of the discretization, e.g. 1D, 2D, or 3D.
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the discretization
        in each dimension. 
    n_dimensions : Number, optional
        number of dimensions. None (default) results in a trans-dimensional
        discretization, with the dimensionality of the parameter space allowed
        to vary in the range ``n_dimensions_min`` - ``n_dimensions_max``
    n_dimensions_min, n_dimensions_max : Number, optional
        minimum and maximum number of dimensions, by default 1 and 10. These
        parameters are ignored if ``n_dimensions`` is not None, i.e. if the
        discretization is not trans-dimensional
    n_dimensions_init_range : Number, optional
        percentage of the range ``n_dimensions_min`` - ``n_dimensions_max`` used to
        initialize the number of dimensions (0.3. by default). For example, if 
        ``n_dimensions_min`` = 1, ``n_dimensions_max`` = 10, and 
        ``n_dimensions_init_range`` = 0.5,
        the maximum number of dimensions at the initialization is
        
        .. code-block:: python
            
            int((n_dimensions_max - n_dimensions_min) * n_dimensions_init_range + n_dimensions_max)
            
    parameters : List[Prior], optional
        a list of free parameters, by default None
    birth_from : {"prior", "neighbour"}, optional
        whether to initialize the newborn basis functions by randomly drawing from 
        the prior (default) or by perturbing the neighbor one.
    """
    def __init__(
        self,
        name: str,
        spatial_dimensions: Number,
        perturb_std: Union[Number, np.ndarray] = 0.1,
        n_dimensions: int = None, 
        n_dimensions_min: int = 1, 
        n_dimensions_max: int = 10, 
        n_dimensions_init_range: Number = 0.3,
        parameters: List[Prior] = None, 
        birth_from: str = "prior",
        **kwargs
    ):
        Prior.__init__(
            self, 
            name=name,
            perturb_std=perturb_std,
            spatial_dimensions=spatial_dimensions,        
        )
        ParameterSpace.__init__(
            self, 
            name=name,
            n_dimensions=n_dimensions,
            n_dimensions_min=n_dimensions_min,
            n_dimensions_max=n_dimensions_max,
            n_dimensions_init_range=n_dimensions_init_range,
            parameters=parameters
        )
        self.spatial_dimensions = spatial_dimensions
        self.perturb_std = perturb_std
        self.birth_from = birth_from
        self._update_repr_args(kwargs)
    
    def sample(self, *args) -> ParameterSpaceState:
        """sample a random ParameterSpaceState instance, including the number of
        dimensions, the Voronoi sites, and the parameter values"""
        # initialize a new ps state with number of dimensions and discretization
        ps_state = self.sample_discretization()
        
        # initialize parameter values
        for name, param in self.parameters.items():
            new_values = [param.sample(p) for p in ps_state["discretization"]]
            if not isinstance(param, ParameterSpace):
                new_values = np.array(new_values)
            ps_state.set_param_values(name, new_values)
        return ps_state
    
    @abstractmethod
    def sample_discretization(self, *args) -> ParameterSpaceState:
        """sample a parameter space state that only contains the discretization"""
        raise NotImplementedError
    
    @abstractmethod
    def initialize(self, *args) -> Union[ParameterSpaceState, List[ParameterSpaceState]]:
        """initializes the values of this discretization including its paramter values

        Returns
        -------
        Union[ParameterSpaceState, List[ParameterSpaceState]
            an initial parameter space state, or a list of paramtere space states
        """
        raise NotImplementedError

    def _init_perturbation_funcs(self):
        raise NotImplementedError
        
    @abstractmethod
    def birth(
        self, param_space_state: ParameterSpaceState
    ) -> Tuple[ParameterSpaceState, Number]:
        r"""adds a dimension to the current parameter space and returns the
        thus obtained new state along with the log of the corresponding partial 
        acceptance probability,
        
        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} = 
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}} 
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}  
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}}.
            
        Parameters
        ----------
        ParameterSpaceState
            initial parameter space state

        Returns
        -------
        ParameterSpaceState
            new parameter space state
        Number
            log of the partial acceptance probability, 
            :math:`\alpha_{p} = \log(
            \frac{p({\bf m'})}{p({\bf m})}
            \frac{q\left({\bf m} 
            \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            \lvert \mathbf{J} \rvert)`
        """
        raise NotImplementedError
    
    @abstractmethod
    def death(
        self, param_space_state: ParameterSpaceState
    ) -> Tuple[ParameterSpaceState, Number]:
        r"""removes a dimension from the given parameter space and returns the
        thus obtained new state along with the log of the corresponding partial 
        acceptance probability,
        
        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} = 
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}} 
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}  
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}}.
                    
        Parameters
        ----------
        ParameterSpaceState
            initial parameter space state

        Returns
        -------
        ParameterSpaceState
            new parameter space state
        Number
            log of the partial acceptance probability, 
            :math:`\alpha_{p} = \log(
            \frac{p({\bf m'})}{p({\bf m})}
            \frac{q\left({\bf m} 
            \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            \lvert \mathbf{J} \rvert)`
        """
        raise NotImplementedError

    @abstractmethod
    def perturb_value(
        self, param_space_state: ParameterSpaceState, idimension: int
    ) -> Tuple[ParameterSpaceState, Number]:
        r"""perturbs the parameter space inherent to the discretization and 
        calculates the log of the corresponding partial acceptance probability 
        
        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} = 
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}} 
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}  
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}}.
            
        .. note::
            All free parameters linked to the discretization will be perturbed
            at the specified index.

        Parameters
        ----------
        ParameterSpaceState
            initial parameter space state
        idimension : int
            index of the parameter-space dimension to be perturbed

        Returns
        -------
        Tuple[ParameterSpaceState, Number]
            the perturbed value and
            :math:`\alpha_{p} = \log(
            \frac{p({\bf m'})}{p({\bf m})}
            \frac{q\left({\bf m} 
            \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            \lvert \mathbf{J} \rvert)`
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_prior(self, param_space_state: ParameterSpaceState) -> Number:
        raise NotImplementedError
    
    @abstractmethod
    def nearest_neighbour(
        self, discretization: np.ndarray, query_point: Union[Number, np.ndarray]
    ) -> int:
        r"""returns the index of the nearest neighbour of a given query point in the 
        discretization
        
        Parameters
        ----------
        discretization : np.ndarray
            the discretization
        query_point : Union[Number, np.ndarray]
            the query point

        Returns
        -------
        int
            the index of the nearest neighbour point
        """
        raise NotImplementedError
    
    def sample_from_neighbour(
        self, 
        old_ps_state: ParameterSpaceState, 
        new_ps_state: ParameterSpaceState, 
    ):
        r"""samples a new parameter space state from the neighbour of the given one
        
        Parameters
        ----------
        old_ps_state : ParameterSpaceState
            the old parameter space state to sample from
        new_ps_state : ParameterSpaceState
            the new parameter space state that needs to be updated with underlying
            parameter values sampled from the neighbour of the old one
        """
        log_prob_ratio = 0
        # fill in the parameters with initial values
        for param_name, param in self.parameters.items():
            if isinstance(param, Discretization):
                init_values = [
                    param.sample_discretization() \
                        for _ in range(new_ps_state.n_dimensions)
                ]
                # TODO log_prob_ratio += ?
            elif isinstance(param, Prior):
                init_values = np.empty((new_ps_state.n_dimensions))
            else:    # ParameterSpace that is not Discretization -> birth from prior
                init_values = [
                    param.sample() for _ in range(new_ps_state.n_dimensions)
                ]
                log_prob_ratio += 0
            new_ps_state.set_param_values(param_name, init_values)
        
        # fill in the parameters with nearest neighbour values
        for i_point, point in enumerate(new_ps_state["discretization"]):
            i_nb_point = self.nearest_neighbour(old_ps_state["discretization"], point)
            nb_point = old_ps_state["discretization"][i_nb_point]
            for param_name, param in self.parameters.items():
                if isinstance(param, Discretization):
                    _log_prob = param.sample_from_neighbour(
                        old_ps_state[param_name][i_nb_point], 
                        new_ps_state[param_name][i_point], 
                    )
                    log_prob_ratio += _log_prob
                elif isinstance(param, Prior):
                    nb_value = old_ps_state[param_name][i_nb_point]
                    new_value, _ = param.perturb_value(nb_value, nb_point)
                    new_ps_state[param_name][i_point] = new_value
                    _log_prior_ratio += param.log_prior(new_value, point)
                    _perturb_std = param.get_perturb_std(point)
                    _log_proposal_ratio = (
                        math.log(_perturb_std * SQRT_TWO_PI)
                        + (new_value - nb_value) ** 2 / (2 * _perturb_std ** 2)
                    )
                    log_prob_ratio += _log_prior_ratio + _log_proposal_ratio
        return log_prob_ratio

    def get_perturb_std(self, *args) -> Number:
        """get the standard deviation of the Gaussian used to perturb the 
        discretization
        """
        return self.perturb_std

    def _update_repr_args(self, kwargs):
        self._repr_args["spatial_dimensions"] = self.spatial_dimensions
        self._repr_args["perturb_std"] = self.perturb_std
        if self.trans_d:
            self._repr_args["birth_from"] = self.birth_from
        self._repr_args.update(kwargs)

    def __repr__(self) -> str:
        return ParameterSpace.__repr__(self)
