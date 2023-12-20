from typing import Tuple
from numbers import Number

from bayesbridge._state import State

from ._base_perturbation import Perturbation
from .._state import State
from ..parameterization import ParameterSpace


class BirthPerturbation(Perturbation):
    """Perturbation by creating a new dimension
    
    Parameters
    ----------
    parameter_space : ParameterSpace
        instance of :class:`bayesbridge.parameterization.ParameterSpace`
    """
    def __init__(
        self,
        parameter_space: ParameterSpace,
    ):
        self.param_space = parameter_space
        self.param_space_name = parameter_space.name

    def perturb(self, state: State) -> Tuple[State, Number]:
        """propose a new state that has a new dimension from the given state and
        calculates its associated proposal ratio
        
        Parameters
        ----------
        state : State
            the given current state

        Returns
        -------
        Tuple[State, Number]
            proposed new state and the log proposal ratio for this perturbation

        Raises
        ------
        DimensionalityException
            when the current state has already reached the maximum number of Voronoi
            cells
        """
        old_ps_state = state.get_param_values(self.param_space_name)
        new_ps_state, log_proposal_ratio = self.param_space.birth(old_ps_state)
        new_state = state.copy()
        new_state.set_param_values(self.param_space_name, new_ps_state)
        return new_state, log_proposal_ratio
    
    def log_prior_ratio(self, old_state: State, new_state: State) -> Number:
        """log prior ratio for the current perturbation

        Parameters
        ----------
        old_state : State
            the old state to perturb from
        new_state : State
            the new state to perturb into

        Returns
        -------
        Number
            the log prior ratio for the current perturbation
        """
        return self.param_space.log_prior_ratio_birth(
            old_state.get_param_values(self.param_space_name), 
            new_state.get_param_values(self.param_space_name), 
        )

    @property
    def __name__(self) -> str:
        return f"{self.type}({self.param_space_name})"


# class BirthFromNeighbour(BirthPerturbation):
#     def initialize_newborn_params(
#         self, new_site: Number, old_sites: np.ndarray, model: State
#     ) -> Dict[str, float]:
#         """initialize the parameter values of a newborn cell by perturbing from the
#         parameter value(s) of the nearest Voronoi cell

#         Parameters
#         ----------
#         new_site : Number
#             position of the newborn Voronoi cell
#         old_sites : np.ndarray
#             all positions of the current Voronoi cells
#         model : State
#             current model

#         Returns
#         -------
#         Dict[str, float]
#             key value pairs that map parameter names to values of the ``new_site``
#         """
#         isite = nearest_index(xp=new_site, x=old_sites, xlen=old_sites.size)
#         new_born_values = dict()
#         self._all_old_values = dict()
#         self._all_new_values = dict()
#         for param_name, param in self.parameters.items():
#             old_values = model.get_param_values(param_name)
#             new_value = param.perturb_value(new_site, old_values[isite])
#             new_born_values[param_name] = new_value
#             self._all_old_values[param_name] = old_values[isite]
#             self._all_new_values[param_name] = new_value
#         return new_born_values

#     def log_proposal_ratio(self) -> float:
#         """log proposal ratio for the current perturbation
        
#         .. math::
        
#             \\frac{q(\\textbf{m}|\\textbf{m}')}{q(\\textbf{m}|\\textbf{m}')} =
#             \\frac{N-k}{k+1} 
#             \\prod_{i=1}^{M}
#             \\theta_i'^2\\sqrt{2\\pi} 
#             \\exp\\Large\\lbrace\\frac{(v_{{new}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace
        
#         Derived from the following equations:
        
#         .. math::
        
#             \\begin{align*}
#             q(\\textbf{c}|\\textbf{m}') &= \\frac{1}{k+1} \\\\
#             q(\\textbf{v}|\\textbf{m}') &= 1 \\\\
#             q(\\textbf{c}'|\\textbf{m}) &= \\frac{1}{N-k} \\\\
#             q(\\textbf{v}'|\\textbf{m}) &= \\prod_{i=1}^{M} 
#                     \\frac{1}{\\theta_i'^2\\sqrt{2\\pi}}
#                     \\exp\\Large\\lbrace-\\frac{(v_{{new}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace
#             \\end{align*}
        
#         where :math:`M` is the number of unknown parameters.
        
#         In the actual implementation, the :math:`\\frac{N-k}{k+1}` part is cancelled out
#         by the prior ratio in this perturbation type (refer to 
#         :meth:`BirthFromNeighbour.log_prior_ratio`), therefore we only calculate
#         :math:`\\theta_i'^2\\sqrt{2\\pi}\\exp\\Large\\lbrace\\frac{(v_{{new}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace` 
#         in this method.

#         Returns
#         -------
#         float
#             the log proposal ratio
#         """
#         ratio = 0
#         for param_name, param in self.parameters.items():
#             theta = param.get_perturb_std(self._new_site)
#             old_value = self._all_old_values[param_name]
#             new_value = self._all_new_values[param_name]
#             ratio += (
#                 math.log(theta**2 * SQRT_TWO_PI)
#                 + (new_value - old_value) ** 2 / 2 * theta**2
#             )
#         # return ratio
#         raise NotImplementedError
    
#     def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
#         """log prior ratio given two models
        
#         .. math::
        
#             \\frac{p(\\textbf{m}')}{p(\\textbf{m})} = 
#             \\frac{k+1}{N-k}\\prod_{i=1}^{M}p(\\textbf{v}_{{new}_i})

#         where :math:`M` is the number of unknown parameters.

#         In the actual implementation, the :math:`\\frac{k+1}{N-k}` part is cancelled 
#         out by the proposal ratio in this perturbation type (refer to
#         :meth:`BirthFromNeighbour.log_proposal_ratio`), therefore we only calculate
#         :math:`\\prod_{i=1}^{M}p(\\textbf{v}_{{new}_i})` in this method.

#         Parameters
#         ----------
#         old_model : State
#             the old model to perturb from
#         new_model : State
#             the new model to perturb into

#         Returns
#         -------
#         Number
#             the log prior ratio for the current perturbation
#         """
#         prior_value_ratio = 0
#         for param_name, param in self.parameters.items():
#             new_value = new_model.get_param_values(param_name)
#             prior_value_ratio += param.log_prior_ratio_perturbation_birth(
#                 self._new_site, new_value
#             )
#         # return prior_value_ratio
#         raise NotImplementedError


# class BirthFromPrior(BirthFromNeighbour):
#     def initialize_newborn_params(
#         self, new_site: Number, old_sites: List[Number], model: State
#     ) -> Dict[str, float]:
#         """initialize the parameter values of a newborn cell by sampling from the prior
#         distribution

#         Parameters
#         ----------
#         new_site : Number
#             position of the newborn Voronoi cell
#         old_sites : np.ndarray
#             all positions of the current Voronoi cells
#         model : State
#             current model

#         Returns
#         -------
#         Dict[str, float]
#             key value pairs that map parameter names to values of the ``new_site``
#         """
#         self._all_new_values = dict()
#         new_born_values = dict()
#         for param_name, param in self.parameters.items():
#             new_value = param.initialize(new_site)
#             new_born_values[param_name] = new_value
#             self._all_new_values[param_name] = new_value
#         return new_born_values

#     def log_proposal_ratio(self) -> float:
#         """log proposal ratio for the current perturbation
        
#         .. math::
        
#             \\frac{q(\\textbf{m}|\\textbf{m}')}{q(\\textbf{m}|\\textbf{m}')} =
#             \\frac{N-k}{k+1}\\prod_{i=1}^{M} \\frac{1}{p(\\textbf{v}_{{new}_{i}})}

#         Derived from the following equations:
        
#         .. math::
        
#             \\begin{align*}
#             q(\\textbf{c}|\\textbf{m}') &= \\frac{1}{k+1} \\\\
#             q(\\textbf{v}|\\textbf{m}') &= 1 \\\\
#             q(\\textbf{c}'|\\textbf{m}) &= \\frac{1}{N-k} \\\\
#             q(\\textbf{v}'|\\textbf{m}) &= \\prod_{i=1}^{M} p(\\textbf{v}_{{new}_{i}})
#             \\end{align*}
        
#         where :math:`M` is the number of unknown parameters.
        
#         In the actual implementation, since this formula gets cancelled out by the 
#         prior ratio in this perturbation type (refer to 
#         :meth:`BirthFromPrior.log_prior_ratio`), we return 0 directly.

#         Returns
#         -------
#         float
#             the log proposal ratio
#         """
#         # return 0
#         raise NotImplementedError

#     def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
#         """log prior ratio given two models
        
#         .. math::
        
#             \\frac{p(\\textbf{m}')}{p(\\textbf{m})} = 
#             \\frac{k+1}{N-k}\\prod_{i=1}^{M}p(\\textbf{v}_{{new}_i})
            
#         where :math:`M` is the number of unknown parameters.

#         In the actual implementation, the whole formula is cancelled out by the 
#         proposal ratio in this perturbation type (refer to
#         :meth:`BirthFromNeighbour.log_proposal_ratio`), therefore we return 0 
#         directly in this method.

#         Parameters
#         ----------
#         old_model : State
#             the old model to perturb from
#         new_model : State
#             the new model to perturb into

#         Returns
#         -------
#         Number
#             the log prior ratio for the current perturbation
#         """
#         # return 0
#         raise NotImplementedError


class DeathPerturbation(Perturbation):
    """Perturbation by removing an existing dimension
    
    Parameters
    ----------
    parameter_space : ParameterSpace
        instance of :class:`bayesbridge.parameterization.ParameterSpace`
    """
    def __init__(
        self,
        parameter_space: ParameterSpace,
    ):
        self.param_space = parameter_space
        self.param_space_name = parameter_space.name

    def perturb(self, state: State) -> Tuple[State, Number]:
        """propose a new state that has an existing dimension removed from the given
        state and calculates its associated proposal ratio
        
        Parameters
        ----------
        state : State
            the given current state

        Returns
        -------
        Tuple[State, Number]
            proposed new state and the proposal ratio for this perturbation

        Raises
        ------
        DimensionalityException
            when the current state has already reached the minimum number of Voronoi
            cells
        """
        old_ps_state = state.get_param_values(self.param_space_name)
        new_ps_state, log_proposal_ratio = self.param_space.death(old_ps_state)
        new_state = state.copy()
        new_state.set_param_values(self.param_space_name, new_ps_state)
        return new_state, log_proposal_ratio
    
    #     # prepare for death perturbations
    #     n_cells = model.n_voronoi_cells
    #     if n_cells == self.n_dimensions_min:
    #         raise DimensionalityException("Death")
    #     # randomly choose an existing Voronoi site to kill
    #     isite = random.randint(0, n_cells - 1)
    #     self._new_sites = delete(model.voronoi_sites, isite)
    #     # remove parameter values for the removed site
    #     new_values = self._remove_cell_values(isite, model)
    #     # structure new sites and values into new model
    #     for name, value in model.items():
    #         if name not in new_values and isinstance(value, DataNoise):
    #             new_values[name] = value.copy()
    #     new_model = State(n_cells - 1, self._new_sites, new_values)
    #     self._new_model = new_model
    #     # calculate proposal ratio
    #     log_proposal_ratio = self.log_proposal_ratio()
    #     return new_model, log_proposal_ratio

    # def _remove_cell_values(self, isite: Number, model: State) -> Dict[str, np.ndarray]:
    #     self._old_model = model
    #     self._i_removed = isite
    #     self._removed_site = model.voronoi_sites[isite]
    #     self._removed_values = dict()
    #     new_values = dict()
    #     for param_name in self.parameters:
    #         old_values = model.get_param_values(param_name)
    #         self._removed_values[param_name] = old_values[isite]
    #         new_values[param_name] = delete(old_values, isite)
    #     return new_values

# class DeathFromNeighbour(DeathPerturbation):
#     def log_proposal_ratio(self) -> float:
#         """log proposal ratio for the current perturbation
        
#         .. math::
        
#             \\frac{q(\\textbf{m}|\\textbf{m}')}{q(\\textbf{m}|\\textbf{m}')} =
#             \\frac{k}{N-k+1} 
#             \\prod_{i=1}^{M}
#             \\frac{1}{\\theta_i'^2\\sqrt{2\\pi}} 
#             \\exp\\Large\\lbrace-\\frac{(v_{{removed}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace
        
#         Derived from the following equations:
        
#         .. math::
        
#             \\begin{align*}
#             q(\\textbf{c}|\\textbf{m}') &= \\frac{1}{N-(k-1)} \\\\
#             q(\\textbf{v}|\\textbf{m}') &= \\prod_{i=1}^{M}\\theta_i^2\\sqrt{2\\pi}
#                 \\exp\\Large\\lbrace\\frac{(v_{{removed}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace\\\\
#             q(\\textbf{c}'|\\textbf{m}) &= \\frac{1}{k} \\\\
#             q(\\textbf{v}'|\\textbf{m}) &= 1
#             \\end{align*}
        
#         where :math:`M` is the number of unknown parameters.
        
#         In the actual implementation, the :math:`\\frac{k}{N-k+1}` part is cancelled out
#         by the prior ratio in this perturbation type (refer to 
#         :meth:`DeathFromNeighbour.log_prior_ratio`), therefore we only calculate
#         :math:`\\prod_{i=1}^{M}\\frac{1}{\\theta_i'^2\\sqrt{2\\pi}}\\exp\\Large\\lbrace-\\frac{(v_{{removed}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace` 
#         in this method.

#         Returns
#         -------
#         float
#             the log proposal ratio
#         """
#         ratio = 0
#         i_nearest = nearest_index(
#             xp=self._removed_site, x=self._new_sites, xlen=self._new_sites.size
#         )
#         for param_name, param in self.parameters.items():
#             theta = param.get_perturb_std(self._removed_site)
#             nearest_value = self._new_model.get_param_values(param_name)[i_nearest]
#             removed_value = self._removed_values[param_name]
#             ratio -= (
#                 math.log(theta**2 * SQRT_TWO_PI)
#                 + (removed_value - nearest_value) ** 2 / 2 * theta**2
#             )
#         # return ratio
#         raise NotImplementedError
    
#     def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
#         """log prior ratio given two models

#         .. math::
        
#             \\frac{p(\\textbf{m}')}{p(\\textbf{m})} = 
#             \\frac{N-k+1}{k}\\prod_{i=1}^M\\frac{1}{p(\\textbf{v}_{{removed}_i)}}

#         In the actual implementation, the :math:`\\frac{N-k+1}{k}` part is cancelled
#         out by the proposal ratio in this perturbation type (refer to
#         :meth:`DeathFromNeighbour.log_proposal_ratio`), therefore we only calculate
#         :math:`\\prod_{i=1}^M\\frac{1}{p(\\textbf{v}_{{removed}_i)}}` in this method.

#         Parameters
#         ----------
#         old_model : State
#             the old model to perturb from
#         new_model : State
#             the new model to perturb into

#         Returns
#         -------
#         Number
#             the log prior for the current perturbation
#         """
#         prior_value_ratio = 0
#         for param_name, param in self.parameters.items():
#             removed_value = old_model.get_param_values(param_name)[self._i_removed]
#             prior_value_ratio += param.log_prior_ratio_perturbation_death(
#                 self._removed_site, removed_value
#             )
#         # return prior_value_ratio
#         raise NotImplementedError


# class DeathFromPrior(DeathPerturbation):
#     def log_proposal_ratio(self) -> float:
#         """log proposal ratio for the current perturbation
        
#         .. math::
        
#             \\frac{q(\\textbf{m}|\\textbf{m}')}{q(\\textbf{m}|\\textbf{m}')} =
#             \\frac{k}{N-k+1} 
#             \\prod_{i=1}^{M}
#             \\frac{1}{\\theta_i'^2\\sqrt{2\\pi}} 
#             \\exp\\Large\\lbrace-\\frac{(v_{{removed}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace
        
#         Derived from the following equations:
        
#         .. math::
        
#             \\begin{align*}
#             q(\\textbf{c}|\\textbf{m}') &= \\frac{1}{N-(k-1)} \\\\
#             q(\\textbf{v}|\\textbf{m}') &= \\prod_{i=1}^{M}\\theta_i^2\\sqrt{2\\pi}
#                 \\exp\\Large\\lbrace\\frac{(v_{{removed}_i}-v_{{nearest}_i})^2}{2\\theta_i'^2}\\Large\\rbrace\\\\
#             q(\\textbf{c}'|\\textbf{m}) &= \\frac{1}{k} \\\\
#             q(\\textbf{v}'|\\textbf{m}) &= 1
#             \\end{align*}
        
#         where :math:`M` is the number of unknown parameters.
        
#         In the actual implementation, since this formula gets cancelled out by the 
#         prior ratio in this perturbation type (refer to 
#         :meth:`DeathFromPrior.log_prior_ratio`), we return 0 directly.

#         Returns
#         -------
#         float
#             the log proposal ratio
#         """
#         ratio = 0
#         for param_name, param in self.parameters.items():
#             removed_val = self._removed_values[param_name]
#             ratio -= param.log_prior(self._removed_site, removed_val)
#         # return ratio
#         raise NotImplementedError

#     def log_prior_ratio(self, old_model: State, new_model: State) -> Number:
#         """log prior ratio given two models

#         .. math::
        
#             \\frac{p(\\textbf{m}')}{p(\\textbf{m})} = 
#             \\frac{N-k+1}{k}\\prod_{i=1}^M\\frac{1}{p(\\textbf{v}_{{removed}_i)}}

#         In the actual implementation, the whole formula is cancelled out by the 
#         proposal ratio in this perturbation type (refer to
#         :meth:`DeathFromNeighbour.log_proposal_ratio`), therefore we return 0 
#         directly in this method.

#         Parameters
#         ----------
#         old_model : State
#             the old model to perturb from
#         new_model : State
#             the new model to perturb into

#         Returns
#         -------
#         Number
#             the log prior for the current perturbation
#         """
#         # return 0
#         raise NotImplementedError
