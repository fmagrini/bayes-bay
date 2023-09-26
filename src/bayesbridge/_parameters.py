#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:56:00 2022

@author: fabrizio
"""

import random
from functools import partial
import math
import numpy as np
from ._utils_bayes import interpolate_linear_1d


class Parameter:
    def get_perturb_std(self, position):
        raise NotImplementedError

    def generate_random_value(self, position):
        raise NotImplementedError

    def perturb_value(self, position, value):
        raise NotImplementedError

    def prior_probability(self, old_position, new_position):
        raise NotImplementedError

    def proposal_probability(self, old_position, old_value, new_position, new_value):
        raise NotImplementedError


class UniformParameter(Parameter):
    def __init__(self, name, vmin, vmax, perturb_std, position=None, init_sorted=False):
        self.init_params = {
            "name": name,
            "position": position,
            "vmin": vmin,
            "vmax": vmax,
            "perturb_std": perturb_std,
            "init_sorted": init_sorted,
        }
        self.name = name
        self.position = (
            position if position is None else np.array(position, dtype=float)
        )
        vmin = vmin if np.isscalar(vmin) else np.array(vmin, dtype=float)
        vmax = vmax if np.isscalar(vmax) else np.array(vmax, dtype=float)
        if position is None:
            message = "should be a scalar when `position is None`"
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
        self._vmin, self._vmax = self._init_vmin_vmax(vmin, vmax)
        self._delta = self._init_delta(vmin, vmax)  # Either a scalar or interpolator
        self._perturb_std = self._init_perturb_std(perturb_std)
        self.init_sorted = init_sorted

    def __repr__(self):
        string = "%s(" % self.init_params["name"]
        for k, v in self.init_params.items():
            if k == "name":
                continue
            string += "%s=%s, " % (k, v)
        string = string[:-2]
        return string + ")"

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

    def _init_delta(self, vmin, vmax):
        delta = np.array(vmax, dtype=float) - np.array(vmin, dtype=float)
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
        # It can return a scalar or an array or both
        # e.g.
        # >>> p.get_vmin_vmax(np.array([9.2, 8.7]))
        # (array([1.91111111, 1.85555556]), 3)
        vmin = self._vmin if np.isscalar(self._vmin) else self._vmin(position)
        vmax = self._vmax if np.isscalar(self._vmax) else self._vmax(position)
        return vmin, vmax

    def get_perturb_std(self, position):
        if np.isscalar(self._perturb_std):
            return self._perturb_std
        return self._perturb_std(position)

    def generate_random_values(self, positions, is_init=False):
        vmin, vmax = self.get_vmin_vmax(positions)
        values = np.random.uniform(vmin, vmax, positions.size)
        if is_init and self.init_sorted:
            return np.sort(values)
        return values

    def perturb_value(self, position, value):
        vmin, vmax = self.get_vmin_vmax(position)
        std = self.get_perturb_std(position)
        while True:
            random_deviate = random.normalvariate(0, std)
            new_value = value + random_deviate
            if not (new_value < vmin or new_value > vmax):
                return new_value

    def prior_ratio_position_perturbation(self, old_position, new_position):
        delta_old = self.get_delta(old_position)
        delta_new = self.get_delta(new_position)
        return math.log(delta_old / delta_new)

    def prior_ratio_value_perturbation(self):
        return 0

    def proposal_ratio_value_perturbation(self):
        return 0


# %%

if __name__ == "__main__":
    param = Parameterization1D(
        n_voronoi_cells=5,
        voronoi_site_bounds=(0, 10),
        voronoi_site_perturb_std=[[1.0, 10], [1.0, 10]],
    )
    p = UniformParameter("vs", position=[1, 10], vmin=[1, 2], vmax=3, perturb_std=0.1)
    param.add_free_parameter(p)
    p = UniformParameter("vp", position=[1, 10], vmin=[1, 2], vmax=3, perturb_std=0.1)
    param.add_free_parameter(p)
    # print('VS')
    # param.perturbation_free_param('vs')
    # print(param.model.current_state)
    # print(param.model.proposed_state)
    # print('-'*15)
    # param.finalize_perturbation(True)
    # print(param.model.current_state)
    # print(param.model.proposed_state)
    # print('-'*15)

    # print('SITE')
    # param.perturbation_voronoi_site()
    # print(param.model.current_state)
    # print(param.model.proposed_state)
    # print('-'*15)
    # param.finalize_perturbation(True)
    # print(param.model.current_state)
    # print(param.model.proposed_state)

    # print('BIRTH')
    # param.perturbation_birth()
    # print(param.model.current_state)
    # print(param.model.proposed_state)
    # print('-'*15)
    # param.finalize_perturbation(True)
    # print(param.model.current_state)
    # print(param.model.proposed_state)

    # print('DEATH')
    # param.perturbation_death()
    # print(param.model.current_state)
    # print(param.model.proposed_state)
    # print('-'*15)
    # param.finalize_perturbation(True)
    # print(param.model.current_state)
    # print(param.model.proposed_state)
