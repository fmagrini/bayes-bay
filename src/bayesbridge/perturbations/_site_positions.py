from numbers import Number
from typing import Tuple, Union
from functools import partial
import random
import math
import numpy as np

from .._state import State
from .._parameterizations import Parameterization
from ._base_perturbation import Perturbation
from .._utils_bayes import interpolate_linear_1d


class Voronoi1DPerturbation(Perturbation):
    def __init__(
        self, 
        parameterization: Parameterization, 
        voronoi_site_bounds: Tuple[Number, Number], 
        voronoi_site_perturb_std: Union[Number, np.ndarray], 
        position: np.ndarray = None, 
    ):
        super().__init__(parameterization)
        self.voronoi_site_bounds = voronoi_site_bounds
        self._voronoi_site_perturb_std = self._init_voronoi_site_perturb_std(
            voronoi_site_perturb_std, position
        )
    
    def perturb(self, model: State) -> Tuple[State, Number]:
        # randomly choose a Voronoi site to perturb the position
        nsites = model.n_voronoi_cells
        isite = random.randint(0, nsites - 1)
        # prepare for position perturbation
        old_sites = model.voronoi_sites
        old_site = old_sites[isite]
        site_min, site_max = self.voronoi_site_bounds
        std = self._get_voronoi_site_perturb_std(old_site)
        # perturb a valid position
        while True:
            random_deviate = random.normalvariate(0, std)
            new_site = old_site + random_deviate
            if new_site < site_min or new_site > site_max or \
                np.any(np.abs(new_site - old_sites) < 1e-2):
                continue
            break
        # structure new sites into new model
        new_sites = old_sites.copy()
        new_sites[isite] = new_site
        isort = np.argsort(new_sites)
        new_sites = new_sites[isort]
        new_values = dict()
        for name, values in model.param_values.items():
            new_values[name] = values[isort]
        new_model = State(nsites, new_sites, new_values)
        # calculate proposal ratio
        proposal_ratio = self._proposal_ratio(old_site, new_site)
        return new_model, proposal_ratio

    def _init_voronoi_site_perturb_std(self, std, position):
        if np.isscalar(std):
            return std
        assert position is not None, (
            "`position` should not be None if `voronoi_site_perturb_std` is"
            " not a scalar"
        )
        assert len(position) == len(
            std
        ), "`position` should have the same lenght as `voronoi_site_perturb_std`"
        std = np.array(std, dtype=float)
        position = np.array(position, dtype=float)
        return partial(interpolate_linear_1d, x=position, y=std)

    def _get_voronoi_site_perturb_std(self, site):
        if np.isscalar(self._voronoi_site_perturb_std):
            return self._voronoi_site_perturb_std
        return self._voronoi_site_perturb_std(site)
    
    def _proposal_ratio(self, old_site, new_site):
        std_old = self._get_voronoi_site_perturb_std(old_site)
        std_new = self._get_voronoi_site_perturb_std(new_site)
        d = (old_site - new_site) ** 2
        return math.log(std_old / std_new) + \
            d * (std_new**2 - std_old**2) / (2 * std_new**2 * std_old**2)
    