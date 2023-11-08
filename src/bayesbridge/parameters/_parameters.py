#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:56:00 2022

@author: fabrizio
"""

from abc import abstractmethod
from typing import Callable
from numbers import Number
import random
from functools import partial, partialmethod
import math
import numpy as np
from .._utils_bayes import interpolate_linear_1d


TWO_PI = 2 * math.pi
SQRT_TWO_PI = math.sqrt(TWO_PI)


class Parameter:
    def __init__(self, **kwargs):
        self.init_params = kwargs

    @abstractmethod
    def initialize(self, positions: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def perturb_value(self, position, value):
        raise NotImplementedError

    @abstractmethod
    def prior_ratio_perturbation_free_param(self, old_value, new_value, position):
        raise NotImplementedError
    
    @abstractmethod
    def prior_ratio_perturbation_voronoi_site(self, old_position, new_position, value):
        raise NotImplementedError
    
    @abstractmethod
    def prior_ratio_perturbation_birth(self, new_position, new_value):
        raise NotImplementedError
    
    @abstractmethod
    def prior_ratio_perturbation_death(self, removed_position, removed_value):
        raise NotImplementedError
    
    def set_custom_initialize(self, initialize_func):
        """
        Set a custom initialization function.

        Parameters:
        initialize_func (callable): The function to use for initialization. This function should take no arguments.

        Example:
        def my_init():
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
        return hyper_param if np.isscalar(hyper_param) else \
            partial(interpolate_linear_1d, x=self.position, y=hyper_param)
    
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
    def __init__(self, name, vmin, vmax, perturb_std, position=None):
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
            np.array(vmax, dtype=float) - np.array(vmin, dtype=float))
        self._perturb_std = self._init_pos_dependent_hyper_param(perturb_std)        

    def get_delta(self, position):
        return self._get_pos_dependent_hyper_param(self._delta, position)

    def get_vmin_vmax(self, position):
        # It can return a scalar or an array or both
        # e.g.
        # >>> p.get_vmin_vmax(np.array([9.2, 8.7]))
        # (array([1.91111111, 1.85555556]), 3)
        vmin = self._get_pos_dependent_hyper_param(self._vmin, position)
        vmax = self._get_pos_dependent_hyper_param(self._vmax, position)
        return vmin, vmax

    def get_perturb_std(self, position):
        return self._get_pos_dependent_hyper_param(self._perturb_std, position)
    
    def initialize(self, positions):
        vmin, vmax = self.get_vmin_vmax(positions)
        values = np.random.uniform(vmin, vmax, positions.size)
        return values

    def perturb_value(self, position, value):
        # randomly perturb the value until within range
        std = self.get_perturb_std(position)
        vmin, vmax = self.get_vmin_vmax(position)
        while True:
            random_deviate = random.normalvariate(0, std)
            new_value = value + random_deviate
            if new_value >= vmin and new_value <= vmax:
                return new_value

    def prior_ratio_perturbation_free_param(self, old_value, new_value, position):
        return 0
    
    def prior_ratio_perturbation_voronoi_site(self, old_position, new_position, value):
        old_delta = self.get_delta(old_position)
        new_delta = self.get_delta(new_position)
        return math.log(old_delta / new_delta)
    
    def prior_ratio_perturbation_birth(self, new_position, new_value):
        return 1 / self.get_delta(new_position)
    
    def prior_ratio_perturbation_death(self, removed_position, removed_value):
        return self.get_delta(removed_position)


class GaussianParameter(Parameter):
    def __init__(self, name, mean, std, perturb_std, position=None):
        super().__init__(
            name=name,
            position=position,
            mean=mean,
            std=std,
            perturb_std=perturb_std,
        )
        self.name=name
        
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
        return self._get_pos_dependent_hyper_param(self._mean, position)
    
    def get_std(self, position):
        return self._get_pos_dependent_hyper_param(self._std, position)
        
    def get_perturb_std(self, position):
        return self._get_pos_dependent_hyper_param(self._perturb_std, position)

    def initialize(self, positions):
        mean = self.get_mean(positions)
        std = self.get_std(positions)
        values = np.random.normal(mean, std, positions.size)
        return values

    def perturb_value(self, position, value):
        perturb_std = self.get_perturb_std(position)
        random_deviate = random.normalvariate(0, perturb_std)
        return value + random_deviate

    def prior_ratio_perturbation_free_param(self, old_value, new_value, position):
        mean = self.get_mean(position)
        std = self.get_std(position)
        return (old_value - mean)**2 - (new_value - mean)**2 / (2*std**2)
    
    def prior_ratio_perturbation_voronoi_site(self, old_position, new_position, value):
        old_mean = self.get_mean(old_position)
        new_mean = self.get_mean(new_position)
        old_std = self.get_std(old_position)
        new_std = self.get_std(new_position)
        return math.log(old_std / new_std) + \
            (new_std**2*(value-old_mean)**2 - old_std**2*(value-new_mean) / \
                2*old_std**2*new_std**2)
    
    def prior_ratio_perturbation_birth(self, new_position, new_value):
        mean = self.get_mean(new_position)
        std = self.get_std(new_position)
        return -math.log(std*SQRT_TWO_PI) - (new_value - mean)**2 / (2*std)
    
    def prior_ratio_perturbation_death(self, removed_position, removed_value):
        mean = self.get_mean(removed_position)
        std = self.get_std(removed_position)
        return math.log(std*SQRT_TWO_PI) + (removed_value-mean)**2 / (2*std)


class ParameterFromPrior(Parameter):
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
        return self._initialize(positions)

    def perturb_value(self, position, value):
        return self._perturb_value(position, value)
        
    def prior_ratio_perturbation_free_param(self, old_value, new_value, position):
        new_log_prior = self._log_prior(position, new_value)
        old_log_prior = self._log_prior(position, old_value)    
        return new_log_prior - old_log_prior

    def prior_ratio_perturbation_voronoi_site(self, old_position, new_position, value):
        new_log_prior = self._log_prior(new_position, value)
        old_log_prior = self._log_prior(old_position, value)
        return new_log_prior - old_log_prior
    
    def prior_ratio_perturbation_birth(self, new_position, new_value):
        return self._log_prior(new_position, new_value)
    
    def prior_ratio_perturbation_death(self, removed_position, removed_value):
        return - self._log_prior(removed_position, removed_value)
