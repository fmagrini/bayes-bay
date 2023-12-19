from abc import abstractmethod
from typing import Tuple, Dict, List
from numbers import Number
import random
import math
from bisect import bisect_left
import numpy as np

from ._base_perturbation import Perturbation
from .._state import State, DataNoise
from ..exceptions._exceptions import DimensionalityException
from ..parameters._parameters import SQRT_TWO_PI, Parameter
from .._utils_1d import delete, insert_scalar, nearest_index


class BirthPerturbation(Perturbation):
    """Perturbation by creating a new dimension
    
    If discretization is a part of a state, there are two different ways to initialize the parameter values for the 
    new-born cell. The following two classes are subclasses of 
    :class:`BirthPerturbation1D`.
    
    - Born from nearerst neighbour: :class:`BirthFromNeighbour1D`
    - Born from prior sampling: :class:`BirthFromPrior1D`
    
    Users can choose between ``neighbour`` and ``prior`` in the initialization of 
    :class:`bayesbridge.Voronoi1D`.
    
    Parameters
    ----------
    parameters : Dict[str, Parameter]
        list of parameters of the current inverse problem
    n_dimensions_max : int
        maximum number of dimensions, for bound check purpose
    voronoi_site_bounds : Tuple[int, int]
        minimum and maximum bounds for Voronoi site positions
    """
    def __init__(
        self,
        parameters: Dict[str, Parameter],
        n_dimensions_max: int,
        discretization_bounds: Tuple[int, int] = None,
    ):
        self.parameters = parameters
        self.n_dimensions_max = n_dimensions_max
        self.discretization_bounds = discretization_bounds

    def perturb(self, model: State) -> Tuple[State, Number]:
        """propose a new model that has a new dimension from the given model and
        calculates its associated proposal ratio
        
        Refer to :meth:`BirthFromNeighbour1D.log_proposal_ratio` and
        :meth:`BirthFromPrior1D.log_proposal_ratio` for details on how the log proposal
        ratios are calculated in different cases of this perturbation type.

        Parameters
        ----------
        model : State
            the given current model

        Returns
        -------
        Tuple[State, Number]
            proposed new model and the proposal ratio for this perturbation

        Raises
        ------
        DimensionalityException
            when the current model has already reached the maximum number of Voronoi
            cells
        """
        # prepare for birth perturbations
        n_cells = model.n_voronoi_cells
        if n_cells == self.n_dimensions_max:
            raise DimensionalityException("Birth")
        old_sites = model.voronoi_sites
        # randomly choose a new Voronoi site position
        lb, ub = self.voronoi_site_bounds
        while True:
            new_site = random.uniform(lb, ub)
            self._new_site = new_site
            break
        # intialize parameter values
        unsorted_values = self.initialize_newborn_cell(new_site, old_sites, model)
        idx_insert = bisect_left(old_sites, new_site)
        new_sites = insert_scalar(old_sites, idx_insert, new_site)
        new_values = dict()
        for name, value in unsorted_values.items():
            new_values[name] = insert_scalar(getattr(model, name), idx_insert, value)
        for name, value in model.items():
            if name not in new_values and isinstance(value, DataNoise):
                new_values[name] = value.copy()
        new_model = State(n_cells + 1, new_sites, new_values)
        # calculate proposal ratio
        log_proposal_ratio = self.log_proposal_ratio()
        return new_model, log_proposal_ratio

    @abstractmethod
    def initialize_newborn_cell(
        self, new_site: Number, old_sites: np.ndarray, model: State
    ) -> Dict[str, float]:
        """initialize the parameter values of a newborn cell
        
        The concrete implementations of this abstract method are:
        
        - Birth from nearest neighbour: 
          :meth:`BirthFromNeighbour1D.initialize_newborn_cell`
        - Birth from prior sampling: :meth:`BirthFromPrior1D.initialize_newborn_cell`

        Parameters
        ----------
        new_site : Number
            position of the newborn Voronoi cell
        old_sites : np.ndarray
            all positions of the current Voronoi cells
        model : State
            current model

        Returns
        -------
        Dict[str, float]
            key value pairs that map parameter names to values of the ``new_site``
        """
        raise NotImplementedError

    @abstractmethod
    def log_proposal_ratio(self) -> float:
        """log proposal ratio for the latest perturbation
        
        The concrete implementations of this abstract method are:
        
        - Birth from nearest neighbour:
          :meth:`BirthFromNeighbour1D.log_proposal_ratio`
        - Birth from prior sampling: :meth:`BirthFromPrior1D.log_proposal_ratio`
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        """log prior ratio for the current perturbation

        The concrete implementations of this abstract method are:
        
        - Birth from nearest neighbour:
          :meth:`BirthFromNeighbour1D.log_prior_ratio`
        - Birth from prior sampling: :meth:`BirthFromPrior1D.log_prior_ratio`
        """
        raise NotImplementedError


class BirthFromNeighbour1D(BirthPerturbation1D):
    def initialize_newborn_cell(
        self, new_site: Number, old_sites: np.ndarray, model: State
    ) -> Dict[str, float]:
        """initialize the parameter values of a newborn cell by perturbing from the
        parameter value(s) of the nearest Voronoi cell

        Parameters
        ----------
        new_site : Number
            position of the newborn Voronoi cell
        old_sites : np.ndarray
            all positions of the current Voronoi cells
        model : State
            current model

        Returns
        -------
        Dict[str, float]
            key value pairs that map parameter names to values of the ``new_site``
        """
        isite = nearest_index(xp=new_site, x=old_sites, xlen=old_sites.size)
        new_born_values = dict()
        self._all_old_values = dict()
        self._all_new_values = dict()
        for param_name, param in self.parameters.items():
            old_values = model.get_param_values(param_name)
            new_value = param.perturb_value(new_site, old_values[isite])
            new_born_values[param_name] = new_value
            self._all_old_values[param_name] = old_values[isite]
            self._all_new_values[param_name] = new_value
        return new_born_values

    def log_proposal_ratio(self) -> float:
        """log proposal ratio for the current perturbation
        
        .. math::
        
            \\frac{q(\\textbf{m}|\\textbf{m}')}{q(\\textbf{m}|\\textbf{m}')} =
            \\frac{N-k}{k+1} 
            \\prod_{i=1}^{M}
            \\theta_i'^2\\sqrt{2\\pi} 
            \\exp\\Large\\lbrace\\frac{(v_{{new}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace
        
        Derived from the following equations:
        
        .. math::
        
            \\begin{align*}
            q(\\textbf{c}|\\textbf{m}') &= \\frac{1}{k+1} \\\\
            q(\\textbf{v}|\\textbf{m}') &= 1 \\\\
            q(\\textbf{c}'|\\textbf{m}) &= \\frac{1}{N-k} \\\\
            q(\\textbf{v}'|\\textbf{m}) &= \\prod_{i=1}^{M} 
                    \\frac{1}{\\theta_i'^2\\sqrt{2\\pi}}
                    \\exp\\Large\\lbrace-\\frac{(v_{{new}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace
            \\end{align*}
        
        where :math:`M` is the number of unknown parameters.
        
        In the actual implementation, the :math:`\\frac{N-k}{k+1}` part is cancelled out
        by the prior ratio in this perturbation type (refer to 
        :meth:`BirthFromNeighbour1D.log_prior_ratio`), therefore we only calculate
        :math:`\\theta_i'^2\\sqrt{2\\pi}\\exp\\Large\\lbrace\\frac{(v_{{new}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace` 
        in this method.

        Returns
        -------
        float
            the log proposal ratio
        """
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
    
    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        """log prior ratio given two models
        
        .. math::
        
            \\frac{p(\\textbf{m}')}{p(\\textbf{m})} = 
            \\frac{k+1}{N-k}\\prod_{i=1}^{M}p(\\textbf{v}_{{new}_i})

        where :math:`M` is the number of unknown parameters.

        In the actual implementation, the :math:`\\frac{k+1}{N-k}` part is cancelled 
        out by the proposal ratio in this perturbation type (refer to
        :meth:`BirthFromNeighbour1D.log_proposal_ratio`), therefore we only calculate
        :math:`\\prod_{i=1}^{M}p(\\textbf{v}_{{new}_i})` in this method.

        Parameters
        ----------
        old_model : State
            the old model to perturb from
        new_model : State
            the new model to perturb into

        Returns
        -------
        Number
            the log prior ratio for the current perturbation
        """
        prior_value_ratio = 0
        for param_name, param in self.parameters.items():
            new_value = new_model.get_param_values(param_name)
            prior_value_ratio += param.log_prior_ratio_perturbation_birth(
                self._new_site, new_value
            )
        return prior_value_ratio


class BirthFromPrior1D(BirthFromNeighbour1D):
    def initialize_newborn_cell(
        self, new_site: Number, old_sites: List[Number], model: State
    ) -> Dict[str, float]:
        """initialize the parameter values of a newborn cell by sampling from the prior
        distribution

        Parameters
        ----------
        new_site : Number
            position of the newborn Voronoi cell
        old_sites : np.ndarray
            all positions of the current Voronoi cells
        model : State
            current model

        Returns
        -------
        Dict[str, float]
            key value pairs that map parameter names to values of the ``new_site``
        """
        self._all_new_values = dict()
        new_born_values = dict()
        for param_name, param in self.parameters.items():
            new_value = param.initialize(new_site)
            new_born_values[param_name] = new_value
            self._all_new_values[param_name] = new_value
        return new_born_values

    def log_proposal_ratio(self) -> float:
        """log proposal ratio for the current perturbation
        
        .. math::
        
            \\frac{q(\\textbf{m}|\\textbf{m}')}{q(\\textbf{m}|\\textbf{m}')} =
            \\frac{N-k}{k+1}\\prod_{i=1}^{M} \\frac{1}{p(\\textbf{v}_{{new}_{i}})}

        Derived from the following equations:
        
        .. math::
        
            \\begin{align*}
            q(\\textbf{c}|\\textbf{m}') &= \\frac{1}{k+1} \\\\
            q(\\textbf{v}|\\textbf{m}') &= 1 \\\\
            q(\\textbf{c}'|\\textbf{m}) &= \\frac{1}{N-k} \\\\
            q(\\textbf{v}'|\\textbf{m}) &= \\prod_{i=1}^{M} p(\\textbf{v}_{{new}_{i}})
            \\end{align*}
        
        where :math:`M` is the number of unknown parameters.
        
        In the actual implementation, since this formula gets cancelled out by the 
        prior ratio in this perturbation type (refer to 
        :meth:`BirthFromPrior1D.log_prior_ratio`), we return 0 directly.

        Returns
        -------
        float
            the log proposal ratio
        """
        return 0

    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        """log prior ratio given two models
        
        .. math::
        
            \\frac{p(\\textbf{m}')}{p(\\textbf{m})} = 
            \\frac{k+1}{N-k}\\prod_{i=1}^{M}p(\\textbf{v}_{{new}_i})
            
        where :math:`M` is the number of unknown parameters.

        In the actual implementation, the whole formula is cancelled out by the 
        proposal ratio in this perturbation type (refer to
        :meth:`BirthFromNeighbour1D.log_proposal_ratio`), therefore we return 0 
        directly in this method.

        Parameters
        ----------
        old_model : State
            the old model to perturb from
        new_model : State
            the new model to perturb into

        Returns
        -------
        Number
            the log prior ratio for the current perturbation
        """
        return 0


class DeathPerturbation1D(Perturbation):
    """Perturbation by removing an existing dimension in 1D Voronoi space
    
    There are two different ways to interpret such removals, based on how new cells
    are born. The following two classes are subclasses of :class:`DeathPerturbation1D`.
    
    - Death corresponding to birth from nearest neighbour: :class:`DeathFromNeighbour1D`
    - Death corresponding to birth from prior sampling: :class:`DeathFromPrior1D`
    
    Users can choose between ``neighbour`` and ``prior`` in the initialization of
    :class:`bayesbridge.Voronoi1D`.
    
    Parameters
    ----------
    parameters : Dict[str, Parameter]
        list of parameters of the current inverse problem
    n_dimensions_min : int
        minimum number of dimensions, for bound check purpose
    """
    def __init__(
        self,
        parameters: Dict[str, Parameter],
        n_dimensions_min: int,
    ):
        self.parameters = parameters
        self.n_dimensions_min = n_dimensions_min

    def perturb(self, model: State) -> Tuple[State, Number]:
        """propose a new model that has an existing dimension removed from the given
        model and calculates its associated proposal ratio
        
        Refer to :meth:`DeathFromNeighbour1D.log_proposal_ratio` and
        :meth:`DeathFromPrior1D.log_proposal_ratio` for details on how the log proposal
        ratios are calculated in different cases of this perturbation type.

        Parameters
        ----------
        model : State
            the given current model

        Returns
        -------
        Tuple[State, Number]
            proposed new model and the proposal ratio for this perturbation

        Raises
        ------
        DimensionalityException
            when the current model has already reached the minimum number of Voronoi
            cells
        """
        # prepare for death perturbations
        n_cells = model.n_voronoi_cells
        if n_cells == self.n_dimensions_min:
            raise DimensionalityException("Death")
        # randomly choose an existing Voronoi site to kill
        isite = random.randint(0, n_cells - 1)
        self._new_sites = delete(model.voronoi_sites, isite)
        # remove parameter values for the removed site
        new_values = self._remove_cell_values(isite, model)
        # structure new sites and values into new model
        for name, value in model.items():
            if name not in new_values and isinstance(value, DataNoise):
                new_values[name] = value.copy()
        new_model = State(n_cells - 1, self._new_sites, new_values)
        self._new_model = new_model
        # calculate proposal ratio
        log_proposal_ratio = self.log_proposal_ratio()
        return new_model, log_proposal_ratio

    def _remove_cell_values(self, isite: Number, model: State) -> Dict[str, np.ndarray]:
        self._old_model = model
        self._i_removed = isite
        self._removed_site = model.voronoi_sites[isite]
        self._removed_values = dict()
        new_values = dict()
        for param_name in self.parameters:
            old_values = model.get_param_values(param_name)
            self._removed_values[param_name] = old_values[isite]
            new_values[param_name] = delete(old_values, isite)
        return new_values

    @abstractmethod
    def log_proposal_ratio(self):
        """log proposal ratio for the latest perturbation
        
        The concrete implementations of this abstract method are:
        
        - Death corresponding to birth from nearerst neighbour: 
          :meth:`DeathFromNeighbour1D.log_proposal_ratio`
        - Death corresponding to birth from prior sampling:
          :meth:`DeathFromPrior1D.log_proposal_ratio`
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        """log prior ratio for the latest perturbation
        
        The concrete implementations of this abstract method are:
        
        - Death corresponding to birth from nearerst neighbour: 
          :meth:`DeathFromNeighbour1D.log_prior_ratio`
        - Death corresponding to birth from prior sampling:
          :meth:`DeathFromPrior1D.log_prior_ratio`
        """
        raise NotImplementedError


class DeathFromNeighbour1D(DeathPerturbation1D):
    def log_proposal_ratio(self) -> float:
        """log proposal ratio for the current perturbation
        
        .. math::
        
            \\frac{q(\\textbf{m}|\\textbf{m}')}{q(\\textbf{m}|\\textbf{m}')} =
            \\frac{k}{N-k+1} 
            \\prod_{i=1}^{M}
            \\frac{1}{\\theta_i'^2\\sqrt{2\\pi}} 
            \\exp\\Large\\lbrace-\\frac{(v_{{removed}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace
        
        Derived from the following equations:
        
        .. math::
        
            \\begin{align*}
            q(\\textbf{c}|\\textbf{m}') &= \\frac{1}{N-(k-1)} \\\\
            q(\\textbf{v}|\\textbf{m}') &= \\prod_{i=1}^{M}\\theta_i^2\\sqrt{2\\pi}
                \\exp\\Large\\lbrace\\frac{(v_{{removed}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace\\\\
            q(\\textbf{c}'|\\textbf{m}) &= \\frac{1}{k} \\\\
            q(\\textbf{v}'|\\textbf{m}) &= 1
            \\end{align*}
        
        where :math:`M` is the number of unknown parameters.
        
        In the actual implementation, the :math:`\\frac{k}{N-k+1}` part is cancelled out
        by the prior ratio in this perturbation type (refer to 
        :meth:`DeathFromNeighbour1D.log_prior_ratio`), therefore we only calculate
        :math:`\\prod_{i=1}^{M}\\frac{1}{\\theta_i'^2\\sqrt{2\\pi}}\\exp\\Large\\lbrace-\\frac{(v_{{removed}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace` 
        in this method.

        Returns
        -------
        float
            the log proposal ratio
        """
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
    
    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        """log prior ratio given two models

        .. math::
        
            \\frac{p(\\textbf{m}')}{p(\\textbf{m})} = 
            \\frac{N-k+1}{k}\\prod_{i=1}^M\\frac{1}{p(\\textbf{v}_{{removed}_i)}}

        In the actual implementation, the :math:`\\frac{N-k+1}{k}` part is cancelled
        out by the proposal ratio in this perturbation type (refer to
        :meth:`DeathFromNeighbour1D.log_proposal_ratio`), therefore we only calculate
        :math:`\\prod_{i=1}^M\\frac{1}{p(\\textbf{v}_{{removed}_i)}}` in this method.

        Parameters
        ----------
        old_model : State
            the old model to perturb from
        new_model : State
            the new model to perturb into

        Returns
        -------
        Number
            the log prior for the current perturbation
        """
        prior_value_ratio = 0
        for param_name, param in self.parameters.items():
            removed_value = old_model.get_param_values(param_name)[self._i_removed]
            prior_value_ratio += param.log_prior_ratio_perturbation_death(
                self._removed_site, removed_value
            )
        return prior_value_ratio


class DeathFromPrior1D(DeathPerturbation1D):
    def log_proposal_ratio(self) -> float:
        """log proposal ratio for the current perturbation
        
        .. math::
        
            \\frac{q(\\textbf{m}|\\textbf{m}')}{q(\\textbf{m}|\\textbf{m}')} =
            \\frac{k}{N-k+1} 
            \\prod_{i=1}^{M}
            \\frac{1}{\\theta_i'^2\\sqrt{2\\pi}} 
            \\exp\\Large\\lbrace-\\frac{(v_{{removed}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace
        
        Derived from the following equations:
        
        .. math::
        
            \\begin{align*}
            q(\\textbf{c}|\\textbf{m}') &= \\frac{1}{N-(k-1)} \\\\
            q(\\textbf{v}|\\textbf{m}') &= \\prod_{i=1}^{M}\\theta_i^2\\sqrt{2\\pi}
                \\exp\\Large\\lbrace\\frac{(v_{{removed}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace\\\\
            q(\\textbf{c}'|\\textbf{m}) &= \\frac{1}{k} \\\\
            q(\\textbf{v}'|\\textbf{m}) &= 1
            \\end{align*}
        
        where :math:`M` is the number of unknown parameters.
        
        In the actual implementation, since this formula gets cancelled out by the 
        prior ratio in this perturbation type (refer to 
        :meth:`DeathFromPrior1D.log_prior_ratio`), we return 0 directly.

        Returns
        -------
        float
            the log proposal ratio
        """
        ratio = 0
        for param_name, param in self.parameters.items():
            removed_val = self._removed_values[param_name]
            ratio -= param.log_prior(self._removed_site, removed_val)
        return ratio

    def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
        """log prior ratio given two models

        .. math::
        
            \\frac{p(\\textbf{m}')}{p(\\textbf{m})} = 
            \\frac{N-k+1}{k}\\prod_{i=1}^M\\frac{1}{p(\\textbf{v}_{{removed}_i)}}

        In the actual implementation, the whole formula is cancelled out by the 
        proposal ratio in this perturbation type (refer to
        :meth:`DeathFromNeighbour1D.log_proposal_ratio`), therefore we return 0 
        directly in this method.

        Parameters
        ----------
        old_model : State
            the old model to perturb from
        new_model : State
            the new model to perturb into

        Returns
        -------
        Number
            the log prior for the current perturbation
        """
        return 0
