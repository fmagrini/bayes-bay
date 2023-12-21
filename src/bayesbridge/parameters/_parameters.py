#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:56:00 2022

@author: fabrizio
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, Union, Dict
from numbers import Number
import random
from functools import partial
import math
import numpy as np
from .._utils_1d import interpolate_linear_1d


TWO_PI = 2 * math.pi
SQRT_TWO_PI = math.sqrt(TWO_PI)


class Parameter(ABC):
    """Base class for an unknown parameter"""

    def __init__(self, **kwargs):
        self._name = kwargs["name"]
        if "position" in kwargs and kwargs["position"] is not None:
            self.position = np.array(kwargs["position"], dtype=float)
        else:
            self.position = None
        assert "perturb_std" in kwargs
        self.init_params = kwargs
        
    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def initialize(
        self, position: Union[np.ndarray, Number] = None
    ) -> Union[np.ndarray, Number]:
        """initializes the values of this parameter given one or more positions

        Parameters
        ----------
        position : Union[np.ndarray, Number]
            an array of positions or one position
        
        Returns
        -------
        Union[np.ndarray, Number]
            an array of values or one value corresponding to the given positions
        """
        raise NotImplementedError

    @abstractmethod
    def perturb_value(self, value: Number, position: Number) -> Tuple[Number, Number]:
        """perturb the value of a given position from the given current value, and
        calculates the associated acceptance criteria excluding log likelihood ratio

        Parameters
        ----------
        value : Number
            the current value to be perturbed from
        position : Number
            the position of the value to be perturbed

        Returns
        -------
        Tuple[Number, Number]
            the new value of this parameter at the given position, and its associated
            partial acceptance criteria excluding log likelihood ratio
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self, value: Number, position: Number) -> Number:
        """calculates the log of the prior probability density for the given position
        and value

        Parameters
        ----------
        value : Number
            the value to calculate the probability density for
        position : Number
            the position of the value
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
                position: Union[np.ndarray, Number]
            ) -> Union[np.ndarray, Number]:
                print("This is my custom init!")

            my_param.set_custom_initialize(my_init)
        """
        if not callable(initialize_func):
            raise ValueError("initialize_func must be a callable function.")
        self.initialize = partial(self._initializer, initialize_func)

    def _initializer(self, initialize_func, *args, **kwargs):
        return initialize_func(self, *args, **kwargs)

    def get_perturb_std(
        self, position: Union[Number, np.ndarray]
    ) -> Union[Number, np.ndarray]:
        """get the standard deviation(s) of the Gaussian(s) used to perturb
        the parameter at the given position(s) in the discretization domain
        
        Parameters
        ----------
        position: Union[Number, np.ndarray]
            the position(s) at which the standard deviation(s) of the Gaussian(s)
            used to perturb the parameter will be returned

        Returns
        -------
        Union[Number, np.ndarray]
            standard deviation at the given position(s)
        """
        if self.has_hyper_param("perturb_std"):
            return self.get_hyper_param("perturb_std", position)
        else:
            raise NotImplementedError("`get_perturb_std` needs to be implemented")

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

    def _init_hyper_param(self, hyper_param: Union[Number, np.ndarray]):
        # to be called after self.position is assigned
        return (
            hyper_param
            if np.isscalar(hyper_param)
            else partial(interpolate_linear_1d, x=self.position, y=hyper_param)
        )

    def __repr__(self) -> str:
        string = "%s(" % self.init_params["name"]
        for k, v in self.init_params.items():
            if k == "name":
                continue
            string += "%s=%s, " % (k, v)
        string = string[:-2]
        return string + ")"


class UniformParameter(Parameter):
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
        super().__init__(
            name=name,
            position=position,
            vmin=vmin,
            vmax=vmax,
            perturb_std=perturb_std,
        )
        self.add_hyper_params({
            "vmin": vmin, 
            "vmax": vmax, 
            "perturb_std": perturb_std, 
            "delta": np.array(vmax) - np.array(vmin), 
        })

    def get_delta(
        self, position: Union[Number, np.ndarray]
    ) -> Union[Number, np.ndarray]:
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

    def get_vmin_vmax(
        self, position: Union[Number, np.ndarray]
    ) -> Union[Tuple[Number, Number], Tuple[np.ndarray, np.ndarray]]:
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

    def initialize(
        self, position: Union[np.ndarray, Number]
    ) -> Union[np.ndarray, Number]:
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
        if isinstance(position, Number):
            return random.uniform(vmin, vmax)
        else:
            return np.random.uniform(vmin, vmax, position.size)

    def perturb_value(self, position, value):
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

    def log_prior(self, position, value):
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


class GaussianParameter(Parameter):
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
        This can either be a scalar or an array if the hyper parameters vary with 
        positions
    position : np.ndarray, optional
        positions in the discretization domain corresponding to position-dependent 
        hyper parameters (``mean``, ``std``, ``perturb_std``), by default None
    """

    def __init__(self, name, mean, std, perturb_std, position=None):
        super().__init__(
            name=name,
            position=position,
            mean=mean,
            std=std,
            perturb_std=perturb_std,
        )
        self.add_hyper_params({
            "mean": mean, 
            "std": std, 
            "perturb_std": perturb_std, 
        })
    
    def get_mean(
        self, position: Union[Number, np.ndarray]
    ) -> Union[Number, np.ndarray]:
        """get the prior mean of the Gaussian at the given position in the 
        discretization domain

        Parameters
        ----------
        position: Union[Number, np.ndarray]
            position(s) in the discretization domain at which the uniform 
            distribution range will be returned

        Returns
        -------
        Union[Number, np.ndarray]
            mean at the given position(s)
        """
        return self.get_hyper_param("mean", position)

    def get_std(
        self, position: Union[Number, np.ndarray]
    ) -> Union[Number, np.ndarray]:
        """get the prior standard deviation of the Gaussian at the given position

        Parameters
        ----------
        position: Union[Number, np.ndarray]
            the position(s) in the discretization domain at which the uniform 
            distribution range will be returned

        Returns
        -------
        Union[Number, np.ndarray]
            standard deviation at the given position(s)
        """
        return self.get_hyper_param("std", position)

    def initialize(self, position):
        r"""initialize the parameter at the specified position(s) in the 
        discretization domain
        
        Parameters
        ----------
        position: Union[Number, np.ndarray]
            the position(s) at which the parameter is initialized

        Returns
        -------
        Union[Number, np.ndarray]
            random number(s) chosen according to the normal distribution defined
            by :attr:`mean` and :attr:`std` at the given position(s)
        """
        mean = self.get_mean(position)
        std = self.get_std(position)
        values = np.random.normal(mean, std, position.size)
        return values

    def perturb_value(self, position, value):
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
            of the Gaussian at the specified position (:attr:`GaussianParameter.perturb_std`)
        """
        perturb_std = self.get_perturb_std(position)
        random_deviate = random.normalvariate(0, perturb_std)
        return value + random_deviate
    
    def log_prior(self, position, value):
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
            log prior probability for the Gaussian parameter
        """
        mean = self.get_mean(position)
        std = self.get_std(position)
        return -0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * ((value - mean) / std)**2


class CustomParameter(Parameter):
    """Class enabling the definition of an arbitrary prior for a free parameter

    Parameters
    ----------
    name : str
        name of the current parameter, for display and storing purposes
    log_prior : Callable[[Number, Number], Number]
    initialize : Callable[[np.ndarray], np.ndarray]
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the parameter. 
        This can either be a scalar or an array if the hyper parameters vary with 
        positions
    position : np.ndarray, optional
        positions in the discretization domain corresponding to position-dependent 
        hyper parameters (``mean``, ``std``, ``perturb_std``), by default None
    """
    def __init__(
        self,
        name: str,
        log_prior: Callable[[Number, Number], Number],
        initialize: Callable[[np.ndarray], np.ndarray],
        perturb_std: Union[Number, np.ndarray], 
        position: np.ndarray = None, 
    ):
        super().__init__(
            name=name,
            log_prior=log_prior,
            initialize=initialize,
            perturb_std=perturb_std, 
            position=position, 
        )
        self.add_hyper_params({
            "perturb_std": perturb_std
        })
        self._log_prior = log_prior
        self._initialize = initialize
        
    def initialize(self, position: np.ndarray) -> np.ndarray:
        r"""initialize the parameter at the specified position(s) in the 
        discretization domain
        
        Parameters
        ----------
        position: Union[Number, np.ndarray]
            the position(s) at which the parameter is initialized

        Returns
        -------
        Union[Number, np.ndarray]
            random number(s) chosen according to the prior probability distribution
        """
        return self._initialize(positions)
    
    def perturb_value(self, position, value):
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
            of the Gaussian at the specified position (:attr:`GaussianParameter.perturb_std`)
        """
        perturb_std = self.get_perturb_std(position)
        random_deviate = random.normalvariate(0, perturb_std)
        return value + random_deviate

    def log_prior(self, position, value):
        """log prior probability of occurrence of a value at the given position 
        in the discretization domain
        
        Parameters
        ----------
        position: Number
            Position in the discretization domain, determining the mean and
            standard deviation of the (prior) normal distribution
            
        value: Number
        
        Returns
        -------
        Number
        """
        return self._log_prior(position, value)
