#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:56:00 2022

@author: fabrizio
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, Union
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
        self.init_params = kwargs

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

    def _init_pos_dependent_hyper_param(self, hyper_param):
        # to be called after self.position is assigned
        return (
            hyper_param
            if np.isscalar(hyper_param)
            else partial(interpolate_linear_1d, x=self.position, y=hyper_param)
        )

    def _get_pos_dependent_hyper_param(self, hyper_param, position):
        return hyper_param if np.isscalar(hyper_param) else hyper_param(position)

    def __repr__(self):
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
        self.name = name
        # type standardization and validation
        self.position = (
            position if position is None else np.array(position, dtype=float)
        )
        vmin = vmin if np.isscalar(vmin) else np.array(vmin, dtype=float)
        vmax = vmax if np.isscalar(vmax) else np.array(vmax, dtype=float)
        if position is None:
            message = "should be a scalar when `position` is `None`"
            assert np.isscalar(vmin), "`vmin` " + message
            assert np.isscalar(vmax), "`vmax` " + message
            assert np.isscalar(perturb_std), "`perturb_std` " + message
        else:
            message = "should either be a scaler or have the same length as `position`"
            assert np.isscalar(vmin) or vmin.size == self.position.size, (
                "`vmin` " + message
            )
            assert np.isscalar(vmax) or vmax.size == self.position.size, (
                "`vmax` " + message
            )
            assert np.isscalar(perturb_std) or perturb_std.size == self.position.size, (
                "`perturb_std` " + message
            )
        # variables below: either a scalar or a function
        self._vmin = self._init_pos_dependent_hyper_param(vmin)
        self._vmax = self._init_pos_dependent_hyper_param(vmax)
        self._delta = self._init_pos_dependent_hyper_param(
            np.array(vmax, dtype=float) - np.array(vmin, dtype=float)
        )
        self._perturb_std = self._init_pos_dependent_hyper_param(perturb_std)

    def get_delta(self, position):
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
        return self._get_pos_dependent_hyper_param(self._delta, position)

    def get_vmin_vmax(
        self, position: Union[Number, np.ndarray]
    ) -> Tuple[Number, Number]:
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
        # It can return a scalar or an array or both
        # e.g.
        # >>> p.get_vmin_vmax(np.array([9.2, 8.7]))
        # (array([1.91111111, 1.85555556]), 3)
        vmin = self._get_pos_dependent_hyper_param(self._vmin, position)
        vmax = self._get_pos_dependent_hyper_param(self._vmax, position)
        return vmin, vmax

    def get_perturb_std(self, position):
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
        return self._get_pos_dependent_hyper_param(self._perturb_std, position)

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

    def log_prior_ratio_perturbation_free_param(self, old_value, new_value, position):
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
        self, old_position, new_position, value
    ):
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

    def log_prior_ratio_perturbation_birth(self, new_position, new_value):
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

    def log_prior_ratio_perturbation_death(self, removed_position, removed_value):
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
        This can either be a scalar or an array if the hyper parameters vary with positions
    position : np.ndarray, optional
        positions in the discretization domain corresponding to position-dependent 
        hyper parameters (``vmin``, ``vmax``, ``perturb_std``), by default None
    """

    def __init__(self, name, mean, std, perturb_std, position=None):
        super().__init__(
            name=name,
            position=position,
            mean=mean,
            std=std,
            perturb_std=perturb_std,
        )
        self.name = name

        # type standardization and validation
        self.position = (
            position if position is None else np.array(position, dtype=float)
        )
        mean = mean if np.isscalar(mean) else np.array(mean, dtype=float)
        std = std if np.isscalar(std) else np.array(std, dtype=float)
        if position is None:
            message = "should be a scalar when `position` is `None`"
            assert np.isscalar(mean), "`mean` " + message
            assert np.isscalar(std), "`std` " + message
            assert np.isscalar(perturb_std), "`perturb_std` " + message
        else:
            message = "should either be a scaler or have the same length as `position`"
            assert np.isscalar(mean) or mean.size == self.position.size, (
                "`mean` " + message
            )
            assert np.isscalar(std) or std.size == self.position.size, (
                "`std` " + message
            )
            assert np.isscalar(perturb_std) or perturb_std.size == self.position.size, (
                "`perturb_std` " + message
            )

        # variables below: either a scalar or a function
        self._mean = self._init_pos_dependent_hyper_param(mean)
        self._std = self._init_pos_dependent_hyper_param(std)
        self._perturb_std = self._init_pos_dependent_hyper_param(perturb_std)

    def get_mean(self, position):
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
        return self._get_pos_dependent_hyper_param(self._mean, position)

    def get_std(self, position):
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
        return self._get_pos_dependent_hyper_param(self._std, position)

    def get_perturb_std(self, position):
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
        return self._get_pos_dependent_hyper_param(self._perturb_std, position)

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
        mean = self.get_mean(position)
        var = self.get_std(position) ** 2
        return -0.5 * ((value - mean) ** 2 / var + math.log(2 * math.pi * var))

    def log_prior_ratio_perturbation_free_param(self, old_value, new_value, position):
        r"""log prior probability ratio associated with the perturbation of a
        Gaussian parameter at the given position in the discretization domain
        
        Parameters
        ----------
        old_value, new_value : Number
            Original and perturbed parameter values
            
        position : Number
            
        Returns
        -------
        Number
        """
        mean = self.get_mean(position)
        std = self.get_std(position)
        return (old_value - mean) ** 2 - (new_value - mean) ** 2 / (2 * std**2)

    def log_prior_ratio_perturbation_voronoi_site(
        self, old_position, new_position, value
    ):
        """log prior probability ratio associated with the perturbation of a
        Voronoi site
        
        Parameters
        ----------
        old_position, new_position : Number
            Original and perturbed Voronoi site position in the discretization
            domain
        
        value : Number
            Parameter value
        
        Returns
        -------
        Number
        """
        old_mean = self.get_mean(old_position)
        new_mean = self.get_mean(new_position)
        old_std = self.get_std(old_position)
        new_std = self.get_std(new_position)
        return math.log(old_std / new_std) + (
            new_std**2 * (value - old_mean) ** 2
            - old_std**2 * (value - new_mean) / 2 * old_std**2 * new_std**2
        )

    def log_prior_ratio_perturbation_birth(self, new_position, new_value):
        """log prior probability ratio associated with the birth of a new
        normally distributed parameter
        
        Parameters
        ----------
        new_position : Number
            Position in the discretized domain associated with the new parameter
        
        new_value : Number
            proposed numerical value for the newly born parameter
        
        Returns
        -------
        Number
        """
        mean = self.get_mean(new_position)
        std = self.get_std(new_position)
        return -math.log(std * SQRT_TWO_PI) - (new_value - mean) ** 2 / (2 * std)

    def log_prior_ratio_perturbation_death(self, removed_position, removed_value):
        """log prior probability ratio associated with the death of a normally 
        distributed parameter
        
        Parameters
        ----------
        removed_position : Number
            position in the discretized domain associated with the removed parameter
        
        removed_value : Number
            numerical value associated with the removed parameter
        
        Returns
        -------
        Number
        """
        mean = self.get_mean(removed_position)
        std = self.get_std(removed_position)
        return math.log(std * SQRT_TWO_PI) + (removed_value - mean) ** 2 / (2 * std)


class CustomParameter(Parameter):
    """Class enabling the definition of an arbitrary prior for a free parameter

    Parameters
    ----------
    name : str
        name of the current parameter, for display and storing purposes
    log_prior : Callable[[Number, Number], Number]
    initialize : Callable[[np.ndarray], np.ndarray]
    perturb_value : Callable[[Number, Number], Number]
    """

    def __init__(
        self,
        name: str,
        log_prior: Callable[[Number, Number], Number],
        initialize: Callable[[np.ndarray], np.ndarray],
        perturb_value: Callable[[Number, Number], Number],
    ):
        super().__init__(
            name=name,
            log_prior=log_prior,
            initialize=initialize,
            perturb_value=perturb_value,
        )
        self.name = name
        self._log_prior = log_prior
        self._initialize = initialize
        self._perturb_value = perturb_value

    def initialize(self, positions: np.ndarray) -> np.ndarray:
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
            perturbed ``value``, generated through the user-defined callable 
            ``perturb_value``
        """
        return self._perturb_value(position, value)

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

    def log_prior_ratio_perturbation_free_param(self, old_value, new_value, position):
        r"""log prior probability ratio associated with the perturbation of the 
        parameter at the given position in the discretization domain
        
        Parameters
        ----------
        old_value, new_value : Number
            Original and perturbed parameter values
            
        position : Number
            
        Returns
        -------
        Number
        """
        new_log_prior = self._log_prior(position, new_value)
        old_log_prior = self._log_prior(position, old_value)
        return new_log_prior - old_log_prior

    def log_prior_ratio_perturbation_voronoi_site(
        self, old_position, new_position, value
    ):
        """log prior probability ratio associated with the perturbation of a
        Voronoi site
        
        Parameters
        ----------
        old_position, new_position : Number
            Original and perturbed Voronoi site position in the discretization
            domain
        
        value : Number
            Parameter value
        
        Returns
        -------
        Number
        """
        new_log_prior = self._log_prior(new_position, value)
        old_log_prior = self._log_prior(old_position, value)
        return new_log_prior - old_log_prior

    def log_prior_ratio_perturbation_birth(self, new_position, new_value):
        """log prior probability ratio associated with the birth of a new
        normally distributed parameter
        
        Parameters
        ----------
        new_position : Number
            Position in the discretized domain associated with the new parameter
        
        new_value : Number
            proposed numerical value for the newly born parameter
        
        Returns
        -------
        Number
        """
        return self._log_prior(new_position, new_value)

    def log_prior_ratio_perturbation_death(self, removed_position, removed_value):
        """log prior probability ratio associated with the death of a parameter
        
        Parameters
        ----------
        removed_position : Number
            position in the discretized domain associated with the removed parameter
        
        removed_value : Number
            numerical value associated with the removed parameter
        
        Returns
        -------
        Number
        """
        return -self._log_prior(removed_position, removed_value)
