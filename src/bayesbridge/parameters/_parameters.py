from abc import ABC, abstractmethod
from typing import Union, Callable, Dict
from numbers import Number
from functools import partial
import random
import math
import numpy as np

from .._utils_1d import interpolate_linear_1d


TWO_PI = 2 * math.pi
SQRT_TWO_PI = math.sqrt(TWO_PI)


class Parameter(ABC):
    """Base class for an unknown parameter"""

    def __init__(self, **kwargs):
        self._name = kwargs["name"]
        self.init_params = kwargs
        
    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def initialize(
        self, positions: Union[np.ndarray, Number]
    ) -> Union[np.ndarray, Number]:
        """initializes the values of this parameter given one or more positions

        Parameters
        ----------
        positions : Union[np.ndarray, Number]
            an array of positions or one position

        Returns
        -------
        Union[np.ndarray, Number]
            an array of values or one value corresponding to the given positions
        """
        raise NotImplementedError

    @abstractmethod
    def perturb_value(self, position: Number, value: Number) -> Number:
        """perturb the value of a given position from the given current value

        Parameters
        ----------
        position : Number
            the position of the value to be perturbed
        value : Number
            the current value to be perturbed from

        Returns
        -------
        Number
            the new value of this parameter at the given position
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self, position, value):
        """calculates the log of the prior probability density for the given position
        and value

        Parameters
        ----------
        position : Number
            the position of the value
        value : Number
            the value to calculate the probability density for
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior_ratio_perturbation_free_param(
        self,
        old_value: Number,
        new_value: Number,
        position: Number,
    ) -> Number:
        """calculates the log prior ratio when the free parameter is perturbed

        Parameters
        ----------
        old_value : Number
            the value for this parameter before perturbation
        new_value : Number
            the value for this parameter after perturbation
        position : Number
            the position of the value perturbed

        Returns
        -------
        Number
            the log prior ratio in the free parameter perturbration case
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior_ratio_perturbation_voronoi_site(
        self,
        old_position: Number,
        new_position: Number,
        value: Number,
    ) -> Number:
        """calculates the log prior ratio when the Voronoi site position is perturbed

        Parameters
        ----------
        old_position : Number
            the position before perturbation
        new_position : Number
            the position after perturbation
        position : Number
            the position of the value perturbed

        Returns
        -------
        Number
            the log prior ratio in the Voronoi site perturbation case
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior_ratio_perturbation_birth(
        self, new_position: Number, new_value: Number
    ) -> Number:
        """calculates the log prior ratio when a new cell is born

        Parameters
        ----------
        new_position : Number
            the position of the new-born cell
        new_value : Number
            the value of the new-born cell

        Returns
        -------
        Number
            the log prior ratio in the birth perturbation case
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior_ratio_perturbation_death(
        self, removed_position: Number, removed_value: Number
    ) -> Number:
        """calculates the log prior ratio when a cell is removed

        Parameters
        ----------
        removed_position : Number
            the position of the cell removed
        removed_value : Number
            the value of the cell removed

        Returns
        -------
        Number
            the log prior ratio in the death perturbation case
        """
        raise NotImplementedError

    def set_custom_initialize(
        self,
        initialize_func: Callable[
            ["Parameter", Union[np.ndarray, Number]], Union[np.ndarray, Number]
        ],
    ):
        r"""set a custom initialization function

        Parameters
        ----------
        initialize_func: Callable[[bayesbridge.parameters.Parameter, Union[np.ndarray, Number]], Union[np.ndarray, Number]]
            The function to use for initialization. This function should take no arguments.

        Examples
        --------
        .. code-block:: python

            def my_init(
                param: bb.parameters.Parameter,
                positions: Union[np.ndarray, Number]
            ) -> Union[np.ndarray, Number]:
                print("This is my custom init!")

            my_object.set_custom_initialize(my_init)
        """
        if not callable(initialize_func):
            raise ValueError("initialize_func must be a callable function.")
        self.initialize = partial(self._initializer, initialize_func)

    def _initializer(self, initialize_func, *args, **kwargs):
        return initialize_func(self, *args, **kwargs)

    def __repr__(self):
        string = f"{self.__class__.__name__}(name={self.init_params['name']}"
        for k, v in self.init_params.items():
            if k == "name":
                continue
            string += "%s=%s, " % (k, v)
        return string[:-2] + ")"


class _PriorMixin(ABC):
    @abstractmethod
    def initialize(self, position: Union[np.ndarray, Number]) -> Union[np.ndarray, Number]:
        raise NotImplementedError
    
    @abstractmethod
    def perturb_value(self, position: Union[Number, None], value: Number) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def log_prior(self, position: Union[Number, None], value: Number) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def log_prior_ratio_perturbation_free_param(
        self, 
        old_value: Number, 
        new_value: Number, 
        position: Union[Number, None]
    ) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def log_prior_ratio_perturbation_voronoi_site(
        self, 
        old_position: Number, 
        new_position: Number, 
        value: Number
    ) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def log_prior_ratio_perturbation_birth(
        self, 
        new_position: Number, 
        new_value: Number, 
    ) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def log_prior_ratio_perturbation_death(
        self,
        removed_position: Number, 
        removed_value: Number
    ) -> float:
        raise NotImplementedError


class _UniformMixin(_PriorMixin):
    def __init__(self):
        assert self.has_hyper_param("vmin")
        assert self.has_hyper_param("vmax")
        assert self.has_hyper_param("perturb_std")
        assert self.has_hyper_param("delta")
    
    def get_delta(self, position: Union[Number, np.ndarray] = None) -> Union[Number, np.ndarray]:
        """get the difference between ``vmax`` and ``vmin`` at the given position
        in the discretization domain

        Parameters
        ----------
        position: Union[Number, np.ndarray]
            the position(s) at which the uniform distribution range will be returned

        Returns
        -------
        Union[Number, np.ndarray]
            the different between ``vmax`` and ``vmin`` at the given position
        """
        return self.get_hyper_param("delta", position)

    def get_vmin_vmax(self, position: Union[Number, np.ndarray] = None) -> Union[Number, np.ndarray]:
        """get the lower and upper bounds at the given position(s) in the discretization 
        domain

        Parameters
        ----------
        position: Union[Number, np.ndarray]
            the position(s) at which the uniform distribution bounds will be returned

        Returns
        -------
        Tuple[float, float]
            the lower (``vmin``) and upper (``vmax``) bounds at the given position(s)
        """
        vmin = self.get_hyper_param("vmin", position)
        vmax = self.get_hyper_param("vmax", position)
        return vmin, vmax
    
    def get_perturb_std(self, position: Union[Number, np.ndarray] = None) -> Union[Number, np.ndarray]:
        """get the standard deviation(s) of the Gaussian(s) used to perturb
        the parameter at the given position(s) in the discretization domain
        
        Parameters
        ----------
        position: Union[Number, np.ndarray]
            position(s) at which the standard deviation(s) of the Gaussian(s)
            used to perturb the parameter will be returned

        Returns
        -------
        Union[Number, np.ndarray]
            standard deviation at the given position(s)
        """
        return self.get_hyper_param("perturb_std", position)
    
    def initialize(self, position: Union[np.ndarray, Number] = None) -> Union[np.ndarray, Number]:
        """initialize the parameter at the specified position(s) in the discretization 
        domain
        
        Parameters
        ----------
        position: Union[Number, np.ndarray]
            the position(s) at which the parameter is initialized

        Returns
        -------
        Union[Number, np.ndarray]
            random number(s) chosen uniformly in the range associated with the
            given position(s)
        """
        vmin, vmax = self.get_vmin_vmax(position)
        if position is None or isinstance(position, Number):
            return random.uniform(vmin, vmax)
        else:
            return np.random.uniform(vmin, vmax, position.size)

    def perturb_value(self, position: Union[Number, None], value: Number) -> float:
        r"""randomly perturb a given value at the specified position in the 
        discretization domain
        
        Parameters
        ----------
        position: Number
            the ``position`` at which the ``value`` is perturbed
        value: Number
            original parameter ``value``
        
        Returns
        -------
        Number
            perturbed ``value``, generated through a random deviate from a 
            normal distribution :math:`\mathcal{N}(\mu, \sigma)`, where :math:`\mu`
            denotes the original value and :math:`sigma` the standard deviation
            of the Gaussian at the specified position (:attr:`UniformParameter.perturb_std`)
        """
        # randomly perturb the value until within range
        std = self.get_perturb_std(position)
        vmin, vmax = self.get_vmin_vmax(position)
        while True:
            random_deviate = random.normalvariate(0, std)
            new_value = value + random_deviate
            if new_value >= vmin and new_value <= vmax:
                return new_value

    def log_prior(self, position: Union[Number, None], value: Number):
        """log prior probability of occurrence of a value falling at the given
        position in the discretization domain
        
        Parameters
        ----------
        position: Number
            position in the discretized domain
        value: Number
        
        Returns
        -------
        Number
            log prior probability. If `value` is outside the allowed range at
            the given position, ``-inf`` is returned
        """
        vmin, vmax = self.get_vmin_vmax(position)
        if vmin <= value <= vmax:
            return -math.log(vmax - vmin)
        else:
            return -math.inf

    def log_prior_ratio_perturbation_free_param(
        self, 
        old_value: Number, 
        new_value: Number, 
        position: Union[Number, None]
    ) -> float:
        r"""log prior probability ratio associated with the perturbation of a
        uniform parameter
        
        Parameters
        ----------
        old_value, new_value, position : Any
            
        Returns
        -------
        Number
            zero
            
        .. note::
            Since at a given position the prior probability is constant,
            the prior probability of occurrence of the perturbed parameter is always 
            equal to the prior of the original parameter. This makes the probability 
            ratio :math:`\frac{p \left( {\bf m'} \right)}{p \left( {\bf m} \right)}`
            constant and equal to one
        """
        return 0

    def log_prior_ratio_perturbation_voronoi_site(
        self, 
        old_position: Number,  
        new_position: Number, 
        value: Number
    ) -> float:
        """log prior probability ratio associated with the perturbation of a
        Voronoi site
        
        Parameters
        ----------
        old_position, new_position : Number
            Original and perturbed Voronoi site position
        value : Any
        
        Returns
        -------
        Number
        """
        old_delta = self.get_delta(old_position)
        new_delta = self.get_delta(new_position)
        return math.log(old_delta / new_delta)

    def log_prior_ratio_perturbation_birth(
        self, 
        new_position: Number, 
        new_value: Number, 
    ) -> float:
        """log prior probability ratio associated with the birth of a new
        uniformly distributed parameter
        
        Parameters
        ----------
        new_position : Number
            Position in the discretized domain associated with the new parameter
        new_value : Any
        
        Returns
        -------
        Number
        """
        return -math.log(self.get_delta(new_position))

    def log_prior_ratio_perturbation_death(
        self,
        removed_position: Number, 
        removed_value: Number
    ) -> float:
        """log prior probability ratio associated with the death of a uniformly 
        distributed parameter
        
        Parameters
        ----------
        removed_position : Number
            Position in the discretized domain associated with the removed parameter
        removed_value : Any
        
        Returns
        -------
        Number
        """
        return math.log(self.get_delta(removed_position))


class _GaussianMixin(_PriorMixin):
    pass


class _CustomPriorMixin(_PriorMixin):
    pass    


class _ParamTypeMixin:
    @abstractmethod
    def add_hyper_params(self, hyper_params: Dict[str, Union[Number, np.ndarray]]):
        raise NotImplementedError
    
    @abstractmethod
    def has_hyper_param(self, hyper_param: str) -> bool:
        raise NotImplementedError    
    
    @abstractmethod
    def get_hyper_param(self, hyper_param: str, position: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        raise NotImplementedError


class _ScalarMixin(_ParamTypeMixin):
    pass


class _VectorizedMixin(_ParamTypeMixin):    
    def __init__(self, position: np.ndarray = None, **kwargs):
        if position is not None:
            position = np.array(position, dtype=float)
        self.position = position
        self.hyper_params = kwargs
        self.add_hyper_params(self.hyper_params)
    
    def add_hyper_params(self, hyper_params: Dict[str, Union[Number, np.ndarray]]):
        msg_scalar_when_pos_none = "should be a scalar when `position` is `None`"
        msg_scalar_or_same_len = (
            "should either be a scalar or have the same length as `position`"
        )
        for name, param_val in hyper_params.items():
            if self.position is None:
                assert np.isscalar(param_val), f"`{name}` {msg_scalar_when_pos_none}"
            else:
                assert np.isscalar(param_val) or \
                    len(param_val) == len(self.position), \
                        f"`{name}` {msg_scalar_or_same_len}"
            if not np.isscalar(param_val):
                param_val = np.array(param_val, dtype=float)
            setattr(self, name, self._init_hyper_param(param_val))
    
    def has_hyper_param(self, hyper_param: str) -> bool:
        return hasattr(self, hyper_param)

    def get_hyper_param(self, hyper_param: str, position: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        hyper_param = getattr(self, hyper_param)
        return hyper_param if np.isscalar(hyper_param) else hyper_param(position)

    def _init_hyper_param(self, hyper_param):
        # to be called after self.position is assigned
        return (
            hyper_param
            if np.isscalar(hyper_param)
            else partial(interpolate_linear_1d, x=self.position, y=hyper_param)
        )


class UniformParameter(_UniformMixin, _VectorizedMixin, Parameter):
    """Class for defining a free parameter acccording to a uniform probability
    distribution

    Parameters
    ----------
    name : str
        name of the current parameter, for display and storing purposes
    vmin : Union[Number, np.ndarray]
        the lower bound for this parameter. This can either be a scalar or an array
        if the hyper parameters vary with positions
    vmax : Union[Number, np.ndarray]
        the upper bound for this parameter. This can either be a scalar or an array
        if the hyper parameters vary with positions
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the parameter. 
        This can either be a scalar or an array if the hyper parameters vary with positions
    position : np.ndarray, optional
        positions in the discretization domain corresponding to position-dependent 
        hyper parameters (``vmin``, ``vmax``, ``perturb_std``), by default None
    """
    def __init__(
        self,
        name: str,
        vmin: Union[Number, np.ndarray],
        vmax: Union[Number, np.ndarray],
        perturb_std: Union[Number, np.ndarray],
        position: np.ndarray = None,
    ):
        Parameter.__init__(
            self, 
            name=name, 
            vmin=vmin, 
            vmax=vmax, 
            perturb_std=perturb_std, 
            position=position
        )
        _VectorizedMixin.__init__(
            self, 
            vmin=vmin, 
            vmax=vmax, 
            perturb_std=perturb_std, 
            position=position
        )
        self.add_hyper_params({"delta": np.array(vmax) - np.array(vmin)})
        _UniformMixin.__init__(self)


class GaussianParameter(_GaussianMixin, _VectorizedMixin, Parameter):
    """Class for defining a free parameter acccording to a normal probability
    distribution

    Parameters
    ----------
    name : str
        name of the current parameter, for display and storing purposes
    mean : Union[Number, np.ndarray]
        mean of the Gaussian. This can either be a scalar or an array
        if the hyper parameters vary with positions
    std : Union[Number, np.ndarray]
        standard deviation of the Gaussian. This can either be a scalar or an array
        if the hyper parameters vary with positions
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the parameter. 
        This can either be a scalar or an array if the hyper parameters vary with positions
    position : np.ndarray, optional
        positions in the discretization domain corresponding to position-dependent 
        hyper parameters (``vmin``, ``vmax``, ``perturb_std``), by default None
    """
    def __init__(self, name, mean, std, perturb_std, position=None):
        Parameter.__init__(
            self, 
            name=name,
            mean=mean,
            std=std,
            perturb_std=perturb_std,
            position=position,
        )
        _VectorizedMixin.__init__(
            self, 
            mean=mean, 
            std=std, 
            perturb_std=perturb_std, 
            position=position, 
        )
        _GaussianMixin.__init__(self)


class CustomParameter(_CustomPriorMixin, _VectorizedMixin, Parameter):
    """Class enabling the definition of an arbitrary prior for a free parameter

    Parameters
    ----------
    name : str
        name of the current parameter, for display and storing purposes
    log_prior : Callable[[Number, Number], Number]
    initialize : Callable[[np.ndarray], np.ndarray]
    perturb_value : Callable[[Number, Number], Number]
    """
    pass

class UniformScalarParameter(_UniformMixin, _ScalarMixin, Parameter):
    pass

class GaussianScalarParameter(_GaussianMixin, _ScalarMixin, Parameter):
    pass

class CustomScalarParameter(_CustomPriorMixin, _ScalarMixin, Parameter):
    pass
