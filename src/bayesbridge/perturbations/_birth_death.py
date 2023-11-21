from abc import abstractmethod
from typing import Tuple, Dict, List
from numbers import Number
import random
import math
import numpy as np

from ._base_perturbation import Perturbation
from .._parameterizations import Parameterization
from .._state import State
from ..exceptions._exceptions import DimensionalityException
from ..parameters._parameters import SQRT_TWO_PI
from .._utils_bayes import nearest_index


class BirthPerturbation1D(Perturbation):
    def __init__(
        self,
        parameterization: Parameterization,
        n_voronoi_cells_max: int,
        n_voronoi_cells_min: int,
        voronoi_site_bounds: Tuple[int, int],
    ):
        super().__init__(parameterization)
        self.n_voronoi_cells_max = n_voronoi_cells_max
        self.n_voronoi_cells_min = n_voronoi_cells_min
        self.voronoi_site_bounds = voronoi_site_bounds

    def perturb(self, model: State) -> Tuple[State, Number]:
        # prepare for birth perturbations
        n_cells = model.n_voronoi_cells
        if n_cells == self.n_voronoi_cells_max:
            raise DimensionalityException("Birth")
        old_sites = model.voronoi_sites
        # randomly choose a new Voronoi site position
        while True:
            lb, ub = self.voronoi_site_bounds
            new_site = random.uniform(lb, ub)
            # abort if it's too close to existing positions
            if np.any(np.abs(new_site - old_sites) < 1e-2):
                continue
            break
        # intialize parameter values
        unsorted_values = self.initialize_newborn_cell(new_site, old_sites, model)
        # structure new sites and values into new model
        new_sites = np.hstack((old_sites, new_site))
        isort = np.argsort(new_sites)
        new_sites = new_sites[isort]
        new_values = dict()
        for name, values in unsorted_values:
            new_values[name] = values[isort]
        new_model = State(n_cells+1, new_sites, new_values)
        # calculate proposal ratio
        proposal_ratio = self.proposal_ratio()
        return new_model, proposal_ratio

    @abstractmethod
    def initialize_newborn_cell(
        self, new_site: Number, old_sites: List[Number], model: State
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def proposal_ratio(self, all_old_values, all_new_values, new_site):
        raise NotImplementedError


class BirthFromNeighbour1D(BirthPerturbation1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_newborn_cell(
        self, new_site: Number, old_sites: List[Number], model: State
    ) -> Dict[str, np.ndarray]:
        self._new_site = new_site
        isite = nearest_index(xp=new_site, x=old_sites)
        new_born_values = dict()
        self._all_old_values = dict()
        self._all_new_values = dict()
        for param_name, param in self.parameterization.parameters.items():
            old_values = model.get_param_values(param_name)
            new_value = param.perturb_value(new_site, old_values[isite])
            new_born_values[param_name] = np.hstack((old_values, new_value))
            self._all_old_values[param_name] = old_values[isite]
            self._all_new_values[param_name] = new_value
        return new_born_values

    def proposal_ratio(self):
        # The calculation omits the \frac{N-k}{k+1} part in the formula,
        # as this will be cancelled out with the prior ratio of the voronoi position part
        ratio = 0
        for param_name, param in self.free_params.items():
            theta = param.get_perturb_std(self._new_site)
            old_value = self._all_old_values[param_name]
            new_value = self._all_new_values[param_name]
            ratio += math.log(theta**2*SQRT_TWO_PI) + (new_value-old_value)**2 / 2*theta**2
        return ratio


class BirthFromPrior1D(BirthFromNeighbour1D):       # TODO
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_newborn_cell(self, new_site: Number, model: State):
        return super().initialize_newborn_cell(new_site, model)
    
    def proposal_ratio(self, all_old_values, all_new_values, new_site):
        return super().proposal_ratio(all_old_values, all_new_values, new_site)


class DeathPerturbation1D(Perturbation):            # TODO
    def __init__(self, parameterization: Parameterization):
        super().__init__(parameterization)
        
    def perturb(self, model: State) -> Tuple[State, Number]:
        # prepare for death perturbations
        n_cells = model.n_voronoi_cells
        if n_cells == self.n_voronoi_cells_min:
            raise DimensionalityException("Death")
        old_sites = model.voronoi_sites
        # randomly choose an existing Voronoi site to kill
        isite = random.randint(0, n_cells-1)
        removed_site = model.voronoi_sites[isite]
        new_sites = np.delete(model.voronoi_sites, isite)
        # remove parameter values for the removed site
        new_values = dict()
        for param_name, param in self.parameterization.parameters.items():
            old_values = model.get_param_values(param_name)
            new_values[param_name] = np.delete(old_values, isite)
        # structure new sites and values into new model
        new_model = State(n_cells-1, new_sites, new_values)
        # calculate proposal ratio
        proposal_ratio = self.proposal_ratio()
        return new_model, proposal_ratio
    
    def proposal_ratio(self, all_old_values, all_new_values, removed_site):
        raise NotImplementedError
