#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:56:00 2022

@author: fabrizio
"""

import random
from math import pi, sqrt
from collections import Iterable
import numpy as np
from _utils_bayes import interpolate_linear_1d




class UniformParameter:
    
    def __init__(self, name, vmin, vmax, std_perturb, starting_model=None):
        """

        Parameters
        ----------
        name : str
        
        vmin, vmax : float
            Boundaries defining the uniform probability distribution
            
        std_perturb : float
            Standard deviation of the Gaussians used to randomly perturb the
            parameter. It affects the probability of birth/depth of a layer
            
        starting_model : func
            Function of depth, should return the values of the uniform parameter
            as at the given depths.
        """
        
        self.name = name
        self.vmin = vmin
        self.vmax = vmax
        self.std_perturb = std_perturb
        if starting_model is not None:
            self.starting_model = starting_model
    
    
    def __repr__(self):
        string = 'UniformParam(%s, vmin=%s'%(self.name, self.vmin)
        string += ', vmax=%s std=%s)'%(self.vmax, self.std_perturb)
        if getattr(self, 'value', None) is not None:
            string += '\n%s'%repr(self.value)
        return string
    
    
    def __getitem__(self, index):
        return self.value[index]
    
    
    def __setitem__(self, index, value):
        self.value[index] = value
    
    
    def starting_model(self, depths):
        n_layers = len(depths)
        return np.sort(
                np.random.choice(np.linspace(self.vmin, self.vmax), 
                                 n_layers,
                                 replace=False)
                                 )    
                
    @property
    def std_perturb(self):
        return self._std_perturb
    
    
    @std_perturb.setter
    def std_perturb(self, value):
        self._std_perturb = value
        prob_uniform = self.vmax - self.vmin
        prob_std = value * sqrt(2 * pi)
        self._birth_prob = prob_std / prob_uniform
        
    
    @property
    def birth_probability(self, *args, **kwargs):
        return self._birth_prob
    
           
    def random_perturbation(self, n_layers, layer=None):
        layer = layer if layer is not None else random.randint(0, n_layers-1)
        while True:
            random_change = random.normalvariate(0, self.std_perturb)
            new_param = self.value[layer, 1] + random_change
            if self.vmin <= new_param <= self.vmax:
                self.value[layer, 1] = new_param
                return 0, n_layers, self.name
            
    
    def prob_change_dimension(self, vold, vnew):
        return (vnew - vold)**2 / (2 * self.std_perturb**2)
        



# class UniformParameter:
    
#     error_depth = '`depth` must not be None when `vmin`, `vmax`, '
#     error_depth += 'or `std_perturb` are iterables.'
#     error_dim_v = 'Incompatible dimensions of `vmin`/`vmax` and `depth`'
#     error_dim_std = 'Incompatible dimensions of `std_perturb` and `depth`'
    
#     def __init__(self, 
#                  name, 
#                  vmin, 
#                  vmax,
#                  std_perturb,
#                  depth=None, 
#                  starting_model=None):
#         """

#         Parameters
#         ----------
#         name : str
        
#         vmin, vmax : float
#             Boundaries defining the uniform probability distribution
            
#         std_perturb : float
#             Standard deviation of the Gaussians used to randomly perturb the
#             parameter. It affects the probability of birth/depth of a layer
            
#         starting_model : func
#             Function of depth, should return the values of the uniform parameter
#             as at the given depths.
#         """
        
#         self.name = name
#         self.vmin = vmin if np.isscalar(vmin) else np.array(vmin)
#         self.vmax = vmax if np.isscalar(vmax) else np.array(vmax)
#         self.delta = self.vmax - self.vmin
#         self.is_scalar_std = np.isscalar(std_perturb)
#         self.is_scalar_delta = np.isscalar(self.delta)
#         self.std_perturb = std_perturb if self.is_scalar_std else np.array(std_perturb)
#         self.depth = depth if not isinstance(depth, Iterable) else np.array(depth)
        
#         if depth is None:
#             if not self.is_scalar_delta or not self.is_scalar_std:
#                 raise Exception(self.error_depth)
#         elif isinstance(depth, Iterable):
#             if not (self.is_scalar_delta or len(self.delta)==len(depth)):
#                 raise Exception(self.error_dim_v)
#             if not (self.is_scalar_std or len(std_perturb)==len(depth)):
#                 raise Exception(self.error_dim_std)
        
#         cond_delta = self.is_scalar_delta or np.allclose(self.delta, self.delta[0])
#         if cond_delta and not np.isscalar(self.delta):
#             self.delta = self.delta[0]
#         cond_std = self.is_scalar_std or np.allclose(std_perturb, std_perturb[0])
#         if cond_std and not np.isscalar(self.std_perturb):
#             self.std_perturb = self.std_perturb[0]
#         self.depth_dependent = False if cond_delta and cond_std else True
#         if starting_model is not None:
#             self.starting_model = starting_model
            
            
#     def __repr__(self):
#         string = 'UniformParam(%s, vmin=%s,'%(self.name, self.vmin)
#         string += ' vmax=%s, std=%s'%(self.vmax, self.std_perturb)
#         if self.depth_dependent:
#             string += ', depth=%s)'%self.depth
#         else:
#             string += ')'
#         if getattr(self, 'value', None) is not None:
#             string += '\n%s'%repr(self.value)
#         return string
    
    
#     def __getitem__(self, index):
#         return self.value[index]
    
    
#     def __setitem__(self, index, value):
#         self.value[index] = value

    
#     def get_std_perturb(self, depth):
#         if self.is_scalar_std:
#             return self.std_perturb
#         return interpolate_linear_1d(depth, self.depth, self.std_perturb)
    
    
#     def get_delta(self, depth):
#         if self.is_scalar_delta:
#             return self.delta
#         return interpolate_linear_1d(depth, self.depth, self.delta)
    
    
#     def prior_ratio(self, old_depth, new_depth):
#         if not self.depth_dependent:
#             return 1
#         delta_old = self.get_delta(old_depth)
#         delta_new = self.get_delta(new_depth)
#         return delta_old / delta_new
    
    
#     def random_perturbation(self, n_layers, layer=None):
#         layer = layer if layer is not None else random.randint(0, n_layers-1)
#         while True:
#             random_change = random.normalvariate(0, self.std_perturb)
#             new_param = self.value[layer, 1] + random_change
#             if self.vmin <= new_param <= self.vmax:
#                 self.value[layer, 1] = new_param
#                 return 0, n_layers, self.name    
    
    
    
    
class Target:

    def __init__(self, name, x, y, std, **kwargs):
        self.name = name
        self.x = x
        self.y = y
        self.std = std
        self.shape = y.shape
        self.__dict__.update(kwargs)
        self.hierarchical = True if std is None else False
    
    
