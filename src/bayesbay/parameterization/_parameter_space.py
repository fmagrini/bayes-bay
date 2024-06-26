from typing import List, Dict, Callable, Tuple, Union
from numbers import Number
import math
import random
import numpy as np

from .._state import State, ParameterSpaceState
from ..prior import Prior
from ..perturbations._param_values import ParamPerturbation
from ..perturbations._birth_death import BirthPerturbation, DeathPerturbation
from ..perturbations._param_space import ParamSpacePerturbation
from .._utils_1d import delete_1d, insert_1d


class ParameterSpace:
    r"""Utility class to parameterize the Bayesian inference problem

    Parameters
    ----------
    name : str
        name of the parameter space, for display and storing purposes
    n_dimensions : Number, optional
        number of dimensions defining the parameter space. None (default) results
        in a trans-dimensional space.
    n_dimensions_min, n_dimensions_max : Number, optional
        minimum and maximum number of dimensions, by default 1 and 10. These
        parameters are ignored if ``n_dimensions`` is not None, i.e. if the
        parameter space is not trans-dimensional
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
    """

    def __init__(
        self,
        name: str,
        n_dimensions: int = None,
        n_dimensions_min: int = 1,
        n_dimensions_max: int = 10,
        n_dimensions_init_range: Number = 0.3,
        parameters: List[Union[Prior, "ParameterSpace"]] = None,
    ):
        self._name = name
        self._trans_d = n_dimensions is None
        self._n_dimensions = n_dimensions
        self._n_dimensions_min = n_dimensions_min if self._trans_d else n_dimensions
        self._n_dimensions_max = n_dimensions_max if self._trans_d else n_dimensions
        self._n_dimensions_init_range = n_dimensions_init_range
        self._parameters = dict()
        if parameters is not None:
            for param in parameters:
                self._parameters[param.name] = param
        self._is_leaf = not any(
            isinstance(p, ParameterSpace) for p in self._parameters.values()
        )
        self._init_perturbation_funcs()
        self._init_repr_args()

    @property
    def name(self) -> str:
        """name of the parameter space"""
        return self._name

    @property
    def trans_d(self) -> bool:
        """indicates whether the current configuration allows changes in
        dimensionality
        """
        return self._trans_d
    
    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    @property
    def parameters(self) -> Dict[str, Prior]:
        """all the free parameters defined in this parameter space"""
        return self._parameters

    @property
    def perturbation_funcs(self) -> List[Callable[[State], Tuple[State, Number]]]:
        r"""a list of perturbation functions allowed in the current parameter space.
        Each function takes in a state (see :class:`State`) and returns a new
        state along with the corresponding partial acceptance probability,

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

        - Birth/Death perturbations: 3
        - Parameter values perturbation: 6
        """
        return self._perturbation_weights

    def initialize(
        self, position: np.ndarray = None
    ) -> Union[ParameterSpaceState, List[ParameterSpaceState]]:
        """initializes the parameter space including its parameter values
        
        If ``position`` is None, a single parameter space state is initialized; 
        otherwise, a list of parameter space states is initialized randomly.
        
        If ``self._n_dimensions_init_range`` is not None, the number of dimensions
        will be initialized randomly within the range defined by the init range.

        Paramters
        ---------
        position : np.ndarray, optional
            initial position of the parameter space, by default None. The type is
            np.ndarray so as to align with other instances to be initialized, such as
            the priors; here only the length of ``position`` is used.

        Returns
        -------
        ParameterSpaceState
            an initial parameter space state
        """
        if position is None:
            return self._initialize()
        else:
            return [self._initialize() for _ in position]
        
    def _initialize(self) -> ParameterSpaceState:
        # initialize number of dimensions
        if not self.trans_d:
            n_dimensions = self._n_dimensions
        else:
            n_dims_min = self._n_dimensions_min
            n_dims_max = self._n_dimensions_max
            init_range = self._n_dimensions_init_range
            init_max = int((n_dims_max - n_dims_min) * init_range + n_dims_min)
            n_dimensions = random.randint(n_dims_min, init_max)
        # initialize parameter values
        parameter_vals = dict()
        for name, param in self.parameters.items():
            parameter_vals[name] = param.initialize(np.empty(n_dimensions))
        return ParameterSpaceState(n_dimensions, parameter_vals)
    
    def sample(self, *args) -> ParameterSpaceState:
        """Randomly generate a new parameter space state
        
        The new value is sampled from (the full range of) the prior distribution of 
        each parameter, regardless of the value of ``self._n_dimensions_init_range`` 
        set during initialization.

        Returns
        -------
        ParameterSpaceState
            randomly generated parameter space state
        """
        # initialize number of dimensions
        if not self.trans_d:
            n_dimensions = self._n_dimensions
        else:
            n_dims_min = self._n_dimensions_min
            n_dims_max = self._n_dimensions_max
            n_dimensions = random.randint(n_dims_min, n_dims_max)
        # initialize parameter values
        parameter_vals = dict()
        for name, param in self.parameters.items():
            parameter_vals[name] = [param.sample() for _ in range(n_dimensions)]
            if isinstance(param, Prior):
                parameter_vals[name] = np.array(parameter_vals[name])
        return ParameterSpaceState(n_dimensions, parameter_vals)

    def log_prior(self, param_space_state: ParameterSpaceState) -> Number:
        joint_log_prior = 0
        for param_name, param in self.parameters.items():
            param_values = param_space_state[param_name]
            all_log_priors = [param.log_prior(v, None) for v in param_values]
            joint_log_prior += sum(all_log_priors)
        return joint_log_prior

    def birth(self, ps_state: ParameterSpaceState) -> Tuple[ParameterSpaceState, float]:
        r"""adds a dimension to the current parameter space and returns the
        thus obtained new state along with the log of the corresponding partial
        acceptance probability,

        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} =
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}}
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}}.

        In this case, we have

        .. math::
            \frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)} = \prod_i{p\left({m_i^{k+1}}\right)},

        because all entries of the :math:`i`\ th free parameter, :math:`\mathbf{m}_i`,
        are equal to those of :math:`\mathbf{m'}_i` except for the newly born
        :math:`m_i^{k+1}`, with :math:`k` denoting the number of dimensions in
        the parameter space prior to the birth perturbation.

        Furthermore, since :math:`m_i^{k+1}` is drawn from the prior and there are
        :math:`k+1` positions available to randomly remove a dimension or add a
        new one,

        .. math::
            \frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)} =
            \frac{1}{k+1} \frac{\prod_i{p\left({m_i^{k+1}}\right)}}{\frac{1}{(k+1)}} =
            \frac{1}{\prod_i{p\left({m_i^{k+1}}\right)}}.


        Finally, it is easy to shown that in this case :math:`\lvert \mathbf{J} \rvert = 1`.
        It follows that

        .. math::
            \alpha_{p} =
            \frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}
            \frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            = \prod_i{p\left({\mathbf{m}_i^{k+1}}\right)} \frac{1}{\prod_i{p\left({m_i^{k+1}}\right)}}
            = 1,

        and :math:`\log(\alpha_{p}) = 0`.

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
            \lvert \mathbf{J} \rvert) = 0`
        """
        n_dims = ps_state.n_dimensions
        if n_dims == self._n_dimensions_max:
            return ps_state, -math.inf
        i_insert = random.randint(0, n_dims)
        new_param_values = dict()
        for param_name, param_vals in ps_state.param_values.items():
            if isinstance(param_vals, np.ndarray):
                new_param_values[param_name] = insert_1d(
                    param_vals, i_insert, self.parameters[param_name].sample()
                )
            else:   # insert newly sampled parameter space state at i_insert
                new_param_values[param_name] = (
                    param_vals[:i_insert]
                    + [self.parameters[param_name].sample()]
                    + param_vals[i_insert:]
                )
        new_state = ParameterSpaceState(n_dims + 1, new_param_values)
        prob_ratio = 0
        return new_state, prob_ratio

    def death(self, ps_state: ParameterSpaceState) -> Tuple[ParameterSpaceState, float]:
        r"""removes a dimension from the given parameter space and returns the
        thus obtained new state along with the log of the corresponding partial
        acceptance probability,

        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} =
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}}
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}}.

        Following a reasoning similar to that explained in the documentation of
        :meth:`birth`, in this case :math:`\log(\alpha_{p}) = 0`.

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
            \lvert \mathbf{J} \rvert) = 0`
        """
        n_dims = ps_state.n_dimensions
        if n_dims == self._n_dimensions_min:
            return ps_state, -math.inf
        i_to_remove = random.randint(0, n_dims - 1)
        new_param_values = dict()
        for param_name, param_vals in ps_state.param_values.items():
            if isinstance(param_vals, np.ndarray):
                new_param_values[param_name] = delete_1d(param_vals, i_to_remove)
            else:
                new_param_values[param_name] = (
                    param_vals[:i_to_remove] + param_vals[i_to_remove + 1 :]
                )
        new_state = ParameterSpaceState(n_dims - 1, new_param_values)
        prob_ratio = 0
        return new_state, prob_ratio

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
        self._perturbation_funcs.append(
            ParamSpacePerturbation(
                self.name, _ps_perturbation_funcs, _ps_perturbation_weights
            )
        )
        self._perturbation_weights.append(sum(_ps_perturbation_weights))

    def _init_repr_args(self):
        self._repr_args = {
            "name": self.name,
            "parameters": list(self.parameters.values()),
        }
        if self.trans_d:
            self._repr_args["n_dimensions_min"] = self._n_dimensions_min
            self._repr_args["n_dimensions_max"] = self._n_dimensions_max
            self._repr_args["n_dimensions_init_range"] = self._n_dimensions_init_range
        else:
            self._repr_args["n_dimensions"] = self._n_dimensions

    def __repr__(self) -> str:
        attr_to_show = self._repr_args
        string = f"{attr_to_show['name']}("
        for k, v in attr_to_show.items():
            string += f"{k}={v}, " if k != "name" else ""
        return f"{string[:-2]})"
