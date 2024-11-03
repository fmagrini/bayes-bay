from abc import abstractmethod
from typing import Union, List, Tuple, Dict
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
            parameters=parameters,
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
    def initialize(
        self, *args
    ) -> Union[ParameterSpaceState, List[ParameterSpaceState]]:
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

    @abstractmethod
    def log_prob_initialize_discretization(self, ps_state: ParameterSpaceState) -> Number:
        r"""The log of the partial acceptance probability of the birth of the
        discretization. This includes only the discretization but not the parameter
        values.

        :math:`\frac{p(k')}{p(k)} \frac{p(c'|k')}{p(c|k)} \frac{q(c'|m)}{q(c|m')}`

        Parameters
        ----------
        ps_state : ParameterSpaceState
            the newly-born parameter space state

        Returns
        -------
        Number
            the log of the partial acceptance probability of the birth of the discretization
        """
        raise NotImplementedError

    def _initialize_newborn_params(
        self,
        new_position: Union[Number, np.ndarray],
        old_positions: np.ndarray,
        old_ps_state: ParameterSpaceState,
    ) -> Tuple[Dict[str, Union[float, ParameterSpaceState]], float]:
        """initialize the parameter values in the newborn dimension

        :math:`\frac{p(v'|c')}{p(v|c)} \frac{q(v|c')}{q(v'|c)}`

        Parameters
        ----------
        new_position : Union[Number, np.ndarray]
            the newborn position in the discretization
        old_positions : np.ndarray
            the current discretization
        old_ps_state : ParameterSpaceState
            the current parameter space state

        Returns
        -------
        Tuple[Dict[str, Union[float, ParameterSpaceState]], float]
            key value pairs that map parameter names to values of the ``new_position``,
            and the log of the partial acceptance probability of this birth
        """
        if self.birth_from == "prior":
            return self._initialize_params_from_prior(new_position)
        return self._initialize_params_from_neighbour(
            new_position, old_positions, old_ps_state
        )

    def _initialize_params_from_prior(
        self, new_position: Union[Number, np.ndarray]
    ) -> Dict:
        """initialize the newborn dimension by randomly drawing parameter values
        from the prior

        Parameters
        ----------
        new_position : Union[Number, np.ndarray]
            the newborn position in the discretization

        Returns
        -------
        Tuple[Dict[str, Union[float, ParameterSpaceState]], float]
            key value pairs that map parameter names to values of the ``new_position``,
            and the log of the partial acceptance probability of this birth
        """
        new_born_values = dict()
        for param_name, param in self.parameters.items():
            new_value = param.sample(new_position)
            new_born_values[param_name] = new_value
        log_prob_ratio = 0
        return new_born_values, log_prob_ratio

    def _initialize_params_from_neighbour(
        self,
        new_position: Union[Number, np.ndarray],
        old_positions: np.ndarray,
        old_ps_state: ParameterSpaceState,
    ) -> Tuple[Dict[str, Union[float, ParameterSpaceState]], float]:
        """initialize the parameter values in the newborn dimension from neighbour

        Parameters
        ----------
        new_position : Union[Number, np.ndarray]
            the newborn position in the discretization
        old_positions : np.ndarray
            the current discretization
        old_ps_state : ParameterSpaceState
            the current parameter space state

        Returns
        -------
        Tuple[Dict[str, Union[float, ParameterSpaceState]], float]
            key value pairs that map parameter names to values of the ``new_position``,
            and the log of the partial acceptance probability of this birth
        """
        i_nearest = self.nearest_neighbour(old_positions, new_position)
        log_prob_ratio = 0
        new_born_values = dict()
        for param_name, param in self.parameters.items():
            if isinstance(param, Discretization):
                neighbour_ps_state = old_ps_state[param.name][i_nearest]
                new_ps_state = param.sample_discretization()
                log_prob_ratio += param._sample_from_neighbour(
                    neighbour_ps_state, new_ps_state
                )
                new_born_values[param_name] = new_ps_state
            elif isinstance(param, Prior):
                old_values = old_ps_state[param_name]
                new_value, _ = param.perturb_value(
                    old_values[i_nearest], new_position, True
                )
                new_born_values[param_name] = new_value
                _perturb_std = param.get_perturb_std_birth(new_position)
                log_prior_ratio = param.log_prior(new_value, new_position)
                log_proposal_ratio = math.log(_perturb_std * SQRT_TWO_PI) + (
                    new_value - old_values[i_nearest]
                ) ** 2 / (2 * _perturb_std**2)
                log_prob_ratio += log_prior_ratio + log_proposal_ratio
            elif isinstance(param, ParameterSpace):     # birth from prior
                new_value, _ = param.sample()
                new_born_values[param_name] = new_value
                # log_prob_ratio += 0
            else:
                raise RuntimeError("this is not reachable")
        return new_born_values, log_prob_ratio

    def _sample_from_neighbour(
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
                    param.sample_discretization()
                    for _ in range(new_ps_state.n_dimensions)
                ]
                log_prob_ratio += sum(
                    [
                        param.log_prob_initialize_discretization(ps_state)
                        for ps_state in init_values
                    ]
                )
            elif isinstance(param, Prior):
                init_values = np.empty((new_ps_state.n_dimensions))
            else:  # ParameterSpace that is not Discretization -> birth from prior
                init_values = [param.sample() for _ in range(new_ps_state.n_dimensions)]
                # log_prob_ratio += 0
            new_ps_state.set_param_values(param_name, init_values)

        # fill in the parameters with nearest neighbour values
        for i_point, position in enumerate(new_ps_state["discretization"]):
            i_nb_point = self.nearest_neighbour(old_ps_state["discretization"], position)
            for param_name, param in self.parameters.items():
                if isinstance(param, Discretization):
                    _log_prob = param._sample_from_neighbour(
                        old_ps_state[param_name][i_nb_point],
                        new_ps_state[param_name][i_point],
                    )
                    log_prob_ratio += _log_prob
                elif isinstance(param, Prior):
                    nb_value = old_ps_state[param_name][i_nb_point]
                    new_value, _ = param.perturb_value(nb_value, position, True)
                    new_ps_state[param_name][i_point] = new_value
                    _log_prior_ratio = param.log_prior(new_value, position)
                    _perturb_std = param.get_perturb_std_birth(position)
                    _log_proposal_ratio = math.log(_perturb_std * SQRT_TWO_PI) + (
                        new_value - nb_value
                    ) ** 2 / (2 * _perturb_std**2)
                    log_prob_ratio += _log_prior_ratio + _log_proposal_ratio
        return log_prob_ratio

    def _log_prob_death_parameters(
        self, 
        old_ps_state: ParameterSpaceState,
        new_ps_state: ParameterSpaceState, 
        i_remove: int,
    ) -> float:
        log_prob_ratio = 0
        if self.birth_from == "prior":
            return log_prob_ratio
        position_to_remove = old_ps_state["discretization"][i_remove]
        i_nearest = self.nearest_neighbour(
            new_ps_state["discretization"], position_to_remove
        )
        for param_name, param in self.parameters.items():
            value_to_remove = old_ps_state[param_name][i_remove]
            nearest_value = new_ps_state[param_name][i_nearest]
            if isinstance(param, Discretization):
                log_prob_ratio -= param.log_prob_initialize_discretization(
                    old_ps_state[param_name][i_remove]
                )
                log_prob_ratio += param._log_prob_death_ps_state(value_to_remove, nearest_value)
            elif isinstance(param, Prior):
                _perturb_std = param.get_perturb_std_birth(position_to_remove)
                log_prob_ratio -= param.log_prior(value_to_remove, position_to_remove)
                log_prob_ratio -= math.log(_perturb_std * SQRT_TWO_PI) + ( 
                    value_to_remove - nearest_value
                ) ** 2 / (2 * _perturb_std**2)
            # else: # ParameterSpace that is not Discretization -> death from prior
            #     log_prob_ratio += 0
        
        return log_prob_ratio

    def _log_prob_death_ps_state(
        self,
        old_ps_state: ParameterSpaceState,
        new_ps_state: ParameterSpaceState, 
    ) -> float:
        log_prob_ratio = 0
        for i_to_remove, position in enumerate(old_ps_state["discretization"]):
            i_nearest = self.nearest_neighbour(new_ps_state["discretization"], position)
            for param_name, param in self.parameters.items():
                if isinstance(param, Discretization):
                    log_prob_ratio -= param.log_prob_initialize_discretization(
                        old_ps_state[param_name][i_to_remove]
                    )
                    log_prob_ratio += param._log_prob_death_ps_state(
                        old_ps_state[param_name][i_to_remove],
                        new_ps_state[param_name][i_nearest]
                    )
                elif isinstance(param, Prior):
                    old_value = old_ps_state[param_name][i_to_remove]
                    new_value = new_ps_state[param_name][i_nearest]
                    _perturb_std = param.get_perturb_std_birth(position)
                    log_prob_ratio -= param.log_prior(old_value, position)
                    log_prob_ratio -= math.log(_perturb_std * SQRT_TWO_PI) + (
                        old_value - new_value
                    ) ** 2 / (2 * _perturb_std**2)
                # else:  # ParameterSpace that is not Discretization -> death from prior
                #     log_prob_ratio += 0
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
