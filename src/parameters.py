#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:56:00 2022

@author: fabrizio
"""

import random
from functools import partial
from copy import copy, deepcopy
import math
from math import pi, sqrt, log
from collections import Iterable
import numpy as np
from _utils_bayes import interpolate_linear_1d, _get_thickness, _is_sorted




class Parameterization:
        
    def perturb_birth(self):
        raise NotImplementedError
        
    def perturb_death(self):
        raise NotImplementedError
        
    def perturb_position(self):
        raise NotImplementedError
        
    def perturb_value(self):
        raise NotImplementedError
        
    

class Parameterization1D(Parameterization):
    
    def __init__(self, 
                 n_voronoi_cells, 
                 voronoi_site_bounds,
                 voronoi_site_perturb_std,
                 free_params=None,
                 trans_d=False, 
                 n_voronoi_cells_min=None, 
                 n_voronoi_cells_max=None):        
        
        self.n_voronoi_cells = n_voronoi_cells
        self.voronoi_site_bounds = voronoi_site_bounds
        self._voronoi_site_perturb_std = \
            self._init_voronoi_site_perturb_std(voronoi_site_perturb_std)
            
        self.free_params = {}
        if free_params is not None:
            for param in free_params:
                self.add_free_parameter(param)
       
        self.trans_d = trans_d
        if trans_d:
            self.n_voronoi_cells_min = n_voronoi_cells_min
            self.n_voronoi_cells_max = n_voronoi_cells_max
        self.voronoi_sites = self._init_voronoi_sites()
        self.voronoi_cell_extents = \
            self._get_voronoi_cell_extents(self.voronoi_sites)
        
        self._current_state = self._init_current_state()
        self._proposed_state = deepcopy(self._current_state)
        
            
    def add_free_parameter(self, free_param):
        self.free_params[free_param.name] = free_param
        self._current_state[free_param.name] = free_param.value
        self._proposed_state[free_param.name] = free_param.value
        
        # self.update_state(key, value, keep_current=True)
        
        
    def _init_free_parameter(self, ):
        pass
        
        
    def _init_voronoi_sites(self):
        lb, ub = self.voronoi_site_bounds
        return np.sort(np.random.uniform(lb, ub, self.n_voronoi_cells))
       
    
    def _get_voronoi_cell_extents(self, voronoi_sites):
        return _get_thickness(voronoi_sites)
        
        
    def _init_voronoi_site_perturb_std(self, std):
        if np.isscalar(std):
            return std
        return partial(interpolate_linear_1d, x=std[:,0], y=std[:,1]) 

    
    def _init_current_state(self):
        current_state = {}
        current_state['n_voronoi_cells'] = self.n_voronoi_cells
        current_state['voronoi_sites'] = self.voronoi_sites
        current_state['voronoi_cell_extents'] = self.voronoi_cell_extents
        for name, param in self.free_params.items():
            current_state[name] = param.value
        return current_state
            
    
    def get_voronoi_site_perturb_std(self, site):
        if np.isscalar(self._voronoi_site_perturb_std):
            return self._voronoi_site_perturb_std
        return self._voronoi_site_perturb_std(site)
    
    
    def perturb_voronoi_site(self):
        isite = np.random.randint(0, self.n_voronoi_cells)
        old_site = self.voronoi_sites[isite]
        site_min, site_max = self.voronoi_site_bounds
        std = self.get_voronoi_site_perturb_std(old_site)
        
        while True:
            random_deviate = random.normalvariate(0, std)
            new_site = old_site + random_deviate
            if new_site<site_min or new_site>site_max: 
                continue
            if np.any(np.abs(new_site - self.voronoi_sites) < 1e-2):
                continue
            break
        self._proposed_state['voronoi_sites'][isite] = new_site
        self._sort_proposed_state()
        
        prob = 0
        for param in self.free_params:
            vmin_old, vmax_old = param.get_delta(old_site)
            vmin_new, vmax_new = param.get_delta(new_site)
            prob += np.log((vmax_old - vmin_old) / (vmax_new - vmin_new))
        
        return prob
        
    
    def _sort_proposed_state(self):
        isort = np.argsort(self._proposed_state['voronoi_sites']).astype
        if not _is_sorted(isort):
            self._proposed_state['voronoi_sites'] = \
                self._proposed_state['voronoi_sites'][isort]
            for name in self.free_params:
                self._proposed_state[name] = self._proposed_state[name][isort]
        self._proposed_state['voronoi_cell_extents'] = \
            self._get_voronoi_cell_extents(
                self._proposed_state['voronoi_sites']        
                )
            
    
    # def update_state(self, old_state, new_state):
        
    #     if keep_current:
    #         new, old = current, proposed
        
    #     else:
    #         old, new = current, proposed
            
    #     for key in old:
    #         old[key] = new[key]
            


# class UniformParameter:
    
#     def __init__(self, name, vmin, vmax, std_perturb, starting_model=None):
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
#         self.vmin = vmin
#         self.vmax = vmax
#         self.std_perturb = std_perturb
#         if starting_model is not None:
#             self.starting_model = starting_model
    
    
#     def __repr__(self):
#         string = 'UniformParam(%s, vmin=%s'%(self.name, self.vmin)
#         string += ', vmax=%s std=%s)'%(self.vmax, self.std_perturb)
#         if getattr(self, 'value', None) is not None:
#             string += '\n%s'%repr(self.value)
#         return string
    
    
#     def __getitem__(self, index):
#         return self.value[index]
    
    
#     def __setitem__(self, index, value):
#         self.value[index] = value
    
    
#     def starting_model(self, depths):
#         n_layers = len(depths)
#         return np.sort(
#                 np.random.choice(np.linspace(self.vmin, self.vmax), 
#                                  n_layers,
#                                  replace=False)
#                                  )    
                
#     @property
#     def std_perturb(self):
#         return self._std_perturb
    
    
#     @std_perturb.setter
#     def std_perturb(self, value):
#         self._std_perturb = value
#         prob_uniform = self.vmax - self.vmin
#         prob_std = value * sqrt(2 * pi)
#         self._birth_prob = prob_std / prob_uniform
        
    
#     @property
#     def birth_probability(self, *args, **kwargs):
#         return self._birth_prob
    
           
#     def random_perturbation(self, n_layers, layer=None):
#         layer = layer if layer is not None else random.randint(0, n_layers-1)
#         while True:
#             random_change = random.normalvariate(0, self.std_perturb)
#             new_param = self.value[layer, 1] + random_change
#             if self.vmin <= new_param <= self.vmax:
#                 self.value[layer, 1] = new_param
#                 return 0, n_layers, self.name
            
    
#     def prob_change_dimension(self, vold, vnew):
#         return (vnew - vold)**2 / (2 * self.std_perturb**2)
        


class Parameter:
    
    def get_perturb_std(self, position):
        raise NotImplementedError  
    
    
    def generate_random_value(self, position):
        raise NotImplementedError
    
    
    def perturb_value(self, position, value):
        raise NotImplementedError
    
    
    def prior_probability(self, old_position, new_position):
        raise NotImplementedError
    
    
    def proposal_probability(self, 
                             old_position, 
                             old_value, 
                             new_position, 
                             new_value):
        raise NotImplementedError

        
        
        
    
class PositionDependendentUniformParam(Parameter):
    
    def __init__(self, name, position, vmin, vmax, perturb_std):
        self.name = name
        self.position = position
        self._vmin, self._vmax = self._init_vmin_vmax(vmin, vmax)
        self._delta = self._init_delta() # Either a scalar or interpolator
        self._perturb_std = self._init_perturb_std(perturb_std)
        
    
    def _init_vmin_vmax(self, vmin, vmax):
        if np.isscalar(vmin) and np.isscalar(vmax):
            return vmin, vmax
        if not np.isscalar(vmin):
            vmin = np.full(self.position.size, vmin) if np.isscalar(vmin) else vmin
            vmin = partial(interpolate_linear_1d, x=self.position, y=vmin)
        if not np.isscalar(vmax):
            vmax = np.full(self.position.size, vmax) if np.isscalar(vmax) else vmax
            vmax = partial(interpolate_linear_1d, x=self.position, y=vmax)
        return vmin, vmax
        
    
    def _init_delta(self):
        delta = self.vmax - self.vmin
        if np.isscalar(delta):
            return delta
        return partial(interpolate_linear_1d, x=self.position, y=delta) 
    
    
    def _init_perturb_std(self, perturb_std):
        if np.isscalar(perturb_std):
            return perturb_std
        return partial(interpolate_linear_1d, x=self.position, y=perturb_std) 
    
    
    def get_delta(self, position):
        if np.isscalar(self._delta):
            return self._delta
        return self._delta(position)
        
    
    def get_vmin_vmax(self, position):
        vmin = self._vmin if np.isscalar(self._vmin) else self._vmin(position)
        vmax = self._vmax if np.isscalar(self._vmax) else self._vmax(position)
        return vmin, vmax
    
    
    def get_perturb_std(self, position):
        if np.isscalar(self._perturb_std):
            return self._perturb_std
        return self._perturb_std(position)    
    
    
    def generate_random_value(self, position):
        vmin, vmax = self.get_vmin_vmax(position)
        return np.random.uniform(vmin, vmax)
    
    
    def perturb_value(self, position, value):
        vmin, vmax = self.get_vmin_vmax(position)
        std = self.get_perturb_std(position)
        while True:
            random_deviate = random.normalvariate(0, std)
            new_value = value + random_deviate
            if not (new_value<vmin or new_value>vmax): 
                return new_value
    
    
    def prior_probability(self, old_position, new_position):
        delta_old = self.get_delta(old_position)
        delta_new = self.get_delta(new_position)
        return math.log(delta_old / delta_new)
    
    
    def proposal_probability(self, 
                             old_position, 
                             old_value, 
                             new_position, 
                             new_value):
        std_old = self.get_std_perturb(old_position)
        std_new = self.get_std_perturb(new_position)
        d = (old_value - new_value)**2
        term1 = math.log(std_old/std_new)
        term2 = d * (std_new**2 - std_old**2) / (2 * std_new**2 * std_old**2)

        return term1 + term2
    
    
    
        
        
    
    
    
    
# class UniformParam():
    
#     error_depth = '`depth` must not be None when `vmin`, `vmax`, '
#     error_depth += 'or `std_perturb` are iterables.'
#     error_dim_v = 'Incompatible dimensions of `vmin`/`vmax` and `depth`'
#     error_dim_std = 'Incompatible dimensions of `std_perturb` and `depth`'
    
#     def __init__(self, 
#                   name, 
#                   vmin, 
#                   vmax,
#                   std_perturb,
#                   depth=None, 
#                   starting_model=None):
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
    
    
#     def generate_random_value(self):
#         return np.random.uniform(self.vmin, self.vmax)
        
    
#     def perturb_value(self, position, value):
#         vmin, vmax = self.get_vmin_vmax(position)
#         std = self.get_perturb_std(position)
#         while True:
#             random_deviate = random.normalvariate(0, std)
#             new_value = value + random_deviate
#             if not (new_value<vmin or new_value>vmax): 
#                 return new_value       
        
    
class Target:

    def __init__(self, name, x, y, std, **kwargs):
        self.name = name
        self.x = x
        self.y = y
        self.std = std
        self.shape = y.shape
        self.__dict__.update(kwargs)
        self.hierarchical = True if std is None else False
    
    
