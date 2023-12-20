#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:41:26 2023

@author: fabrizio
"""

class Discretization(Parameter, ParameterSpace):
    
    def __init__(
        self,
        name: str,
        spatial_dimensions: Number,
        vmin: Union[Number, np.ndarray],
        vmax: Union[Number, np.ndarray],
        perturb_std: Union[Number, np.ndarray],
        n_dimensions: int = None, 
        n_dimensions_min: int = 1, 
        n_dimensions_max: int = 10, 
        n_dimensions_init_range: Number = 0.3,
        parameters: List[Parameter] = None, 
        birth_from: str = "prior",
    ):
        Parameter.__init__(
            self, 
            name=name,
            vmin=vmin,
            vmax=vmax,
            perturb_std=perturb_std,
        )
        ParameterSpace.__init__(
            self, 
            n_dimensions=n_dimensions,
            n_dimensions_min=n_dimensions_min,
            n_dimensions_max=n_dimensions_max,
            n_dimensions_init_range=n_dimensions_init_range,
            parameters=parameters
            )
        self.name = name
        self.spatial_dimensions = spatial_dimensions
        self.vmin = vmin
        self.vmax = vmax
        self.perturb_std = perturb_std
        self.birth_from = birth_from        
    
    @abstractmethod
    def birth(self, param_space_state):
        raise NotImplementedError
    
    @abstractmethod
    def death(self, param_space_state):
        raise NotImplementedError


