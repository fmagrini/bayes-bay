from abc import abstractmethod
from typing import Tuple, Dict, List
from numbers import Number
import random
import math
import numpy as np

from ._base_perturbation import Perturbation
from .._state import State
from ..exceptions._exceptions import DimensionalityException
from ..parameters._parameters import SQRT_TWO_PI, Parameter
from .._utils_bayes import nearest_index


class BirthPerturbation1D(Perturbation):
    def __init__(
        self,
        parameters: Dict[str, Parameter],
        n_voronoi_cells_max: int,
        voronoi_site_bounds: Tuple[int, int],
    ):
        self.parameters = parameters
        self.n_voronoi_cells_max = n_voronoi_cells_max
        self.voronoi_site_bounds = voronoi_site_bounds

    def perturb(self, model: State) -> Tuple[State, Number]:
        # prepare for birth perturbations
        n_cells = model.n_voronoi_cells
        if n_cells == self.n_voronoi_cells_max:
            raise DimensionalityException("Birth")
        old_sites = model.voronoi_sites
        # randomly choose a new Voronoi site position
        lb, ub = self.voronoi_site_bounds
        while True:
            new_site = random.uniform(lb, ub)
            self._new_site = new_site
            # abort if it's too close to existing positions
            if np.any(np.abs(new_site - old_sites) < 1e-2):
                continue
            break
        # intialize parameter values
        unsorted_values = self.initialize_newborn_cell(new_site, old_sites, model)
        # structure new sites and values into new model
        new_sites = np.append(old_sites, new_site)
        isort = np.argsort(new_sites)
        new_sites = new_sites[isort]
        new_values = dict()
        for name, values in unsorted_values.items():
            new_values[name] = values[isort]
        new_model = State(n_cells + 1, new_sites, new_values)
        # calculate proposal ratio
        proposal_ratio = self.proposal_ratio()
        return new_model, proposal_ratio

    @abstractmethod
    def initialize_newborn_cell(
        self, new_site: Number, old_sites: List[Number], model: State
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        # p(k) ratio is always 0 so omitted here
        # p(c|k) ratio = \frac{k+1}{N-k} cancels out with proposal ratio so omitted here
        # calculate only p(v|c) below
        prior_value_ratio = 0
        for param_name, param in self.parameters.items():
            new_value = new_model.get_param_values(param_name)
            prior_value_ratio += param.log_prior_ratio_perturbation_birth(
                self._new_site, new_value
            )
        return prior_value_ratio

    @abstractmethod
    def proposal_ratio(self, all_old_values, all_new_values, new_site):
        raise NotImplementedError


class BirthFromNeighbour1D(BirthPerturbation1D):
    def initialize_newborn_cell(
        self, new_site: Number, old_sites: List[Number], model: State
    ) -> Dict[str, np.ndarray]:
        isite = nearest_index(xp=new_site, x=old_sites, xlen=old_sites.size)
        new_born_values = dict()
        self._all_old_values = dict()
        self._all_new_values = dict()
        for param_name, param in self.parameters.items():
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
        for param_name, param in self.parameters.items():
            theta = param.get_perturb_std(self._new_site)
            old_value = self._all_old_values[param_name]
            new_value = self._all_new_values[param_name]
            ratio += (
                math.log(theta**2 * SQRT_TWO_PI)
                + (new_value - old_value) ** 2 / 2 * theta**2
            )
        return ratio


class BirthFromPrior1D(BirthFromNeighbour1D):
    def initialize_newborn_cell(
        self, new_site: Number, old_sites: List[Number], model: State
    ) -> Dict[str, np.ndarray]:
        self._all_new_values = dict()
        new_born_values = dict()
        for param_name, param in self.parameters.items():
            old_values = model.get_param_values(param_name)
            new_value = param.initialize(new_site)
            new_born_values[param_name] = np.append(old_values, new_value)
            self._all_new_values[param_name] = new_value
        return new_born_values

    def proposal_ratio(self):
        # The calculation omits the \frac{N-k}{k+1} part in the formula,
        # as this will be cancelled out with the prior ratio of the voronoi position part
        ratio = 0
        for param_name, param in self.parameters.items():
            new_val = self._all_new_values[param_name]
            ratio += param.log_prior(self._new_site, new_val)
        return ratio


class DeathPerturbation1D(Perturbation):
    def __init__(
        self,
        parameters: Dict[str, Parameter],
        n_voronoi_cells_min: int,
    ):
        self.parameters = parameters
        self.n_voronoi_cells_min = n_voronoi_cells_min

    def perturb(self, model: State) -> Tuple[State, Number]:
        # prepare for death perturbations
        n_cells = model.n_voronoi_cells
        if n_cells == self.n_voronoi_cells_min:
            raise DimensionalityException("Death")
        # randomly choose an existing Voronoi site to kill
        isite = random.randint(0, n_cells - 1)
        self._new_sites = np.delete(model.voronoi_sites, isite)
        # remove parameter values for the removed site
        new_values = self.remove_cell_values(isite, model)
        # structure new sites and values into new model
        new_model = State(n_cells - 1, self._new_sites, new_values)
        self._new_model = new_model
        # calculate proposal ratio
        proposal_ratio = self.proposal_ratio()
        return new_model, proposal_ratio

    def remove_cell_values(self, isite: Number, model: State) -> Dict[str, np.ndarray]:
        self._old_model = model
        self._i_removed = isite
        self._removed_site = model.voronoi_sites[isite]
        self._removed_values = dict()
        new_values = dict()
        for param_name in self.parameters:
            old_values = model.get_param_values(param_name)
            self._removed_values[param_name] = old_values[isite]
            new_values[param_name] = np.delete(old_values, isite)
        return new_values

    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        # p(k) ratio is always 0 so omitted here
        # p(c|k) ratio = \frac{k}{N-k+1} cancels out with proposal ratio so omitted here
        # calculate only p(v|c) below
        prior_value_ratio = 0
        for param_name, param in self.parameters.items():
            removed_value = old_model.get_param_values(param_name)[self._i_removed]
            prior_value_ratio += param.log_prior_ratio_perturbation_death(
                self._removed_site, removed_value
            )
        return prior_value_ratio

    @abstractmethod
    def proposal_ratio(self):
        raise NotImplementedError


class DeathFromNeighbour1D(DeathPerturbation1D):
    def proposal_ratio(self):
        # The calculation omits the \frac{N-k+1}{k} part in the formula,
        # as this will be cancelled out with the prior ratio of the voronoi position part
        ratio = 0
        i_nearest = nearest_index(
            xp=self._removed_site, x=self._new_sites, xlen=self._new_sites.size
        )
        for param_name, param in self.parameters.items():
            theta = param.get_perturb_std(self._removed_site)
            nearest_value = self._new_model.get_param_values(param_name)[i_nearest]
            removed_value = self._removed_values[param_name]
            ratio -= (
                math.log(theta**2 * SQRT_TWO_PI)
                + (removed_value - nearest_value) ** 2 / 2 * theta**2
            )
        return ratio


class DeathFromPrior1D(DeathPerturbation1D):
    def proposal_ratio(self):
        # The calculation omits the \frac{N-k+1}{k} part in the formula,
        # as this will be cancelled out with the prior ratio of the voronoi position part
        ratio = 0
        for param_name, param in self.parameters.items():
            removed_val = self._removed_values[param_name]
            ratio -= param.log_prior(self._removed_site, removed_val)
        return ratio
