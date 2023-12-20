from abc import abstractmethod
from bisect import bisect_left
import math
from typing import Tuple, Union, List, Dict, Callable
from numbers import Number
import random
import numpy as np
import matplotlib.pyplot as plt

from ..exceptions import DimensionalityException
from ..parameterization import ParameterSpace
from ..parameters import Parameter
from .._state import State, ParameterSpaceState
from ..perturbations import ParamPerturbation
from .._utils_1d import (
    interpolate_result, 
    compute_voronoi1d_cell_extents, 
    insert_scalar,
    nearest_index,
    delete
)


SQRT_TWO_PI = math.sqrt(2 * math.pi)

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


class Voronoi(Discretization):
    """Utility class for Voronoi discretization

    Parameters
    ----------
    n_voronoi_cells : Number, optional
        the number of Voronoi cells. Needs to be None if this is a trans-dimensional 
        parameterization, by default None
    voronoi_sites : Union[np.ndarray, None], optional
        fixed voronoi site positions (if one would like them to be fixed). This should
        be set to None if one needs the site positions to be perturbed (including site
        position perturbation, birth and death). By default None
    voronoi_site_bounds : Tuple[Number, Number]
        the minimum and maximum values of the 1D voronoi site positions
    voronoi_site_perturb_std : Union[Number, np.ndarray]
        the perturbation standard deviation of the Voronoi site positions
    position : np.ndarray, optional
        the breaking points for varied ``voronoi_site_perturb_std``. Activated only
        when the ``voronoi_site_bounds`` and / or ``voronoi_site_perturb_std`` is
        not a scalar, by default None
    free_params : List[Parameter], optional
        a list of unknown parameters, by default None
    n_voronoi_cells_min : Number, optional
        the minimum number of Voronoi cells, by default None
    n_voronoi_cells_max : Number, optional
        the maximum number of Voronoi cells, by default None
    voronoi_cells_init_range : Number, optional
        the range in which the initialization of Voronoi cell numbers will be in
        (i.e. the number of Voronoi cells will be uniformly sampled from the range:
        :math:`(ncells_{min}, ncells_{min}+ninitrange*(ncells_{max}-ncells_{min})`,
        assuming :math:`ncells_{min} = \\text{n_voronoi_cells_min}`,
        :math:`ncells_{max} = \\text{n_voronoi_cells_max}` and
        :math:`ninitrange = \\text{n_voronoi_cells_min}`,
        by default 0.2
    birth_from : str, optional
        whether to initialize newly-born Voronoi cell parameter values from
        randomly sampling or from a perturbation based on the nearest neighbour, by
        default "neighbour"
    """

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
        birth_from: str = "neighbour",  # either "neighbour" or "prior"
    ):
        super().__init__(
            name=name,
            spatial_dimensions=spatial_dimensions,
            vmin=vmin,
            vmax=vmax,
            perturb_std=perturb_std,
            n_dimensions=n_dimensions,
            n_dimensions_min=n_dimensions_min,
            n_dimensions_max=n_dimensions_max,
            n_dimensions_init_range=n_dimensions_init_range,
            parameters=parameters,
            birth_from=birth_from
            )
        
        self._init_perturbation_funcs()
        self._init_log_prior_ratio_funcs()

    def initialize(self) -> State:
        """initializes the parameterization (if it's trans dimensional) and the
        parameter values

        Returns
        -------
        State
            an initial model state
        """
        if not self.trans_d:
            n_voronoi_cells = self._n_dimensions
        # initialize number of cells
        else:
            cells_range = self._n_dimensions_init_range
            cells_min = self._n_dimensions_min
            cells_max = self._n_dimensions_max
            init_max = int((cells_max - cells_min) * cells_range + cells_min)
            n_voronoi_cells = random.randint(cells_min, init_max)
        # initialize site positions
        lb, ub = self.voronoi_site_bounds
        voronoi_sites = np.squeeze(np.random.uniform(lb, ub, n_voronoi_cells))
        # initialize parameter values
        param_vals = dict()
        for name, param in self.free_params.items():
            param_vals[name] = param.initialize(voronoi_sites)
        return State(n_voronoi_cells, voronoi_sites, param_vals)

    def _init_perturbation_funcs(self):
        self._perturbation_funcs = [
            VoronoiPerturbation(
                parameters=self.parameters,
                voronoi_site_bounds=self.voronoi_site_bounds,
                voronoi_site_perturb_std=self.voronoi_site_perturb_std,
                position=self.position,
            )
        ] if not self.fixed_discretization else []
        for name, param in self.parameters.items():
            self._perturbation_funcs.append(ParamPerturbation(name, param))
        if self.trans_d:
            birth_perturb_params = {
                "parameters": self.parameters,
                "n_voronoi_cells_max": self.n_voronoi_cells_max,
                "voronoi_site_bounds": self.voronoi_site_bounds,
            }
            death_perturb_params = {
                "parameters": self.parameters,
                "n_voronoi_cells_min": self.n_voronoi_cells_min,
            }
            if self._birth_from == "neighbour":
                self._perturbation_funcs.append(
                    BirthFromNeighbour1D(**birth_perturb_params)
                )
                self._perturbation_funcs.append(
                    DeathFromNeighbour1D(**death_perturb_params)
                )
            else:
                self._perturbation_funcs.append(
                    BirthFromPrior1D(**birth_perturb_params)
                )
                self._perturbation_funcs.append(
                    DeathFromPrior1D(**death_perturb_params)
                )

    def _init_log_prior_ratio_funcs(self):
        self._log_prior_ratio_funcs = [
            func.log_prior_ratio for func in self._perturbation_funcs
        ]
        
    @property
    def perturbation_functions(self) -> List[Callable[[State], Tuple[State, Number]]]:
        """a list of perturbation functions allowed in the current parameterization
        configurations, each of which takes in a model :class:`State` and returns a new
        model and a log proposal ratio value
        """
        return self._perturbation_funcs

    @property
    def log_prior_ratio_functions(self) -> List[Callable[[State, State], Number]]:
        """a list of log prior ratio functions corresponding to each of the
        :meth:`perturbation_functions`
        """
        return self._log_prior_ratio_funcs

    @staticmethod
    def plot_hist_n_voronoi_cells(samples_n_voronoi_cells, ax=None, **kwargs):
        """plot a histogram of the distribution of the number of Voronoi cells.

        Parameters
        ----------
        samples_n_voronoi_cells : list or ndarray
            list or array containing the number of Voronoi cells in each sample
        ax : Axes, optional
            an optional Axes object to plot on
        kwargs : dict, optional
            additional keyword arguments to pass to ax.hist

        Returns
        -------
        ax : Axes
            The Axes object containing the plot
        """
        # create a new plot if no ax is provided
        if ax is None:
            _, ax = plt.subplots()

        # determine bins aligned to integer values
        min_val = np.min(samples_n_voronoi_cells)
        max_val = np.max(samples_n_voronoi_cells)
        bins = np.arange(min_val, max_val + 2) - 0.5  # bins between integers

        # default histogram style
        hist_style = {
            "bins": bins,
            "align": "mid",
            "rwidth": 0.8,
            "alpha": 0.0,
            "color": kwargs.pop(
                "color", kwargs.pop("c", "blue")
            ),  # Fixed color for the sample lines
        }

        # override or extend with any provided kwargs
        hist_style.update(kwargs)

        # plotting the histogram
        ax.hist(samples_n_voronoi_cells, **hist_style)

        # set plot details
        ax.set_xlabel("Number of Voronoi Cells")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Number of Voronoi Cells")
        ax.grid(True, axis="y")

        return ax






class Voronoi1D(Voronoi):
    
    def __init__(        
            self,
            name: str,
            vmin: Union[Number, np.ndarray],
            vmax: Union[Number, np.ndarray],
            perturb_std: Union[Number, np.ndarray],
            n_dimensions: int = None, 
            n_dimensions_min: int = 1, 
            n_dimensions_max: int = 10, 
            n_dimensions_init_range: Number = 0.3, 
            parameters: List[Parameter] = None, 
            birth_from: str = "neighbour"  # either "neighbour" or "prior"
            ):
        
        super().__init__(
            name=name,
            spatial_dimensions=1,
            vmin=vmin,
            vmax=vmax,
            perturb_std=perturb_std,
            n_dimensions=n_dimensions,
            n_dimensions_min=n_dimensions_min,
            n_dimensions_max=n_dimensions_max,
            n_dimensions_init_range=n_dimensions_init_range,
            parameters=parameters,
            birth_from=birth_from
            )
        
    def _initialize_params(self, 
                           new_site: Number, 
                           old_sites: np.ndarray, 
                           param_space_state: ParameterSpaceState):
        if self.birth_from == 'prior':
            return self._initialize_params_from_prior(
                new_site, old_sites, param_space_state
                )
        return self.initialize_params_from_neighbour(
            new_site, old_sites, param_space_state
            )
    
    def _initialize_params_from_prior(
            self, 
            new_site: Number, 
            old_sites: np.ndarray, 
            param_space_state: ParameterSpaceState
            ):
        """initialize the newborn parameter values by randomly drawing from the 
        prior
    
        Parameters
        ----------
        new_site : Number
            position of the newborn Voronoi cell
        old_sites : np.ndarray
            all positions of the current Voronoi cells
        param_space_state : State
            current parameter space state
    
        Returns
        -------
        Dict[str, float]
            key value pairs that map parameter names to values of the ``new_site``
        """
        new_born_values = dict()
        for param_name, param in self.parameters.items():
            new_value = param.initialize(new_site)
            new_born_values[param_name] = new_value
        return new_born_values, None
    
    def initialize_params_from_neighbour(
        self, 
        new_site: Number, 
        old_sites: np.ndarray, 
        param_space_state: ParameterSpaceState
        ) -> Dict[str, float]:
        """initialize the newborn parameter values by perturbing the nearest 
        Voronoi cell
    
        Parameters
        ----------
        new_site : Number
            position of the newborn Voronoi cell
        old_sites : np.ndarray
            all positions of the current Voronoi cells
        param_space_state : State
            current parameter space state
    
        Returns
        -------
        Dict[str, float]
            key value pairs that map parameter names to values of the ``new_site``
        """
        isite = nearest_index(xp=new_site, x=old_sites, xlen=old_sites.size)
        new_born_values = dict()
        for param_name, param in self.parameters.items():
            old_values = getattr(param_space_state, param_name)
            new_value = param.perturb_value(new_site, old_values[isite])
            new_born_values[param_name] = new_value
        return new_born_values, isite
    
    def _log_probability_ratio_birth(
            self, 
            old_isite: Number, 
            old_ps_state: ParameterSpaceState, 
            new_isite: Number, 
            new_ps_state: ParameterSpaceState
            ):
        if self.birth_from == 'prior':
            return self._log_probability_ratio_birth_from_prior(
                    old_isite, old_ps_state, new_isite, new_ps_state
                    )
        return self._log_probability_ratio_birth_from_neighbour(
                old_isite, old_ps_state, new_isite, new_ps_state
                )
    
    def _log_probability_ratio_birth_from_prior(
            self, 
            old_isite: Number, 
            old_ps_state: ParameterSpaceState, 
            new_isite: Number, 
            new_ps_state: ParameterSpaceState
            ):
        return 0
    
    def _log_probability_ratio_birth_from_neighbour(
            self, 
            old_isite: Number, 
            old_ps_state: ParameterSpaceState, 
            new_isite: Number, 
            new_ps_state: ParameterSpaceState
            ):
        new_site = getattr(new_ps_state, self.name)[new_isite]
        log_prior_ratio = 0
        log_proposal_ratio = 0
        for param_name, param in self.parameters.items():
            new_value = getattr(new_ps_state, param_name)[new_isite]
            log_prior_ratio += param.log_prior(new_site, new_value)
            
            old_value = getattr(old_ps_state, param_name)[old_isite]
            theta = param.get_perturb_std(new_site)
            log_proposal_ratio += (
                math.log(theta * SQRT_TWO_PI)
                + (new_value - old_value) ** 2 / (2 * theta**2)
            )
        return log_prior_ratio + log_proposal_ratio # log_det_jacobian is 1          
    
    def birth(self, old_ps_state: ParameterSpaceState):
        # prepare for birth perturbation
        n_cells = old_ps_state.n_dimensions
        if n_cells == self.n_dimensions_max:
            raise DimensionalityException("Birth")
        # randomly choose a new Voronoi site position
        lb, ub = self.vmin, self.vmax
        new_site = random.uniform(lb, ub)
        old_sites = getattr(old_ps_state, self.name)
        unsorted_values, i_nearest = self._initialize_params(
            new_site, old_sites, old_ps_state
            )
        new_values = dict()
        idx_insert = bisect_left(old_sites, new_site)
        new_sites = insert_scalar(old_sites, idx_insert, new_site)
        new_values[self.name] = new_sites
        for name, value in unsorted_values.items():
            old_values = getattr(old_ps_state, name)
            new_values[name] = insert_scalar(old_values, idx_insert, value)
        new_ps_state = ParameterSpaceState(self.n_dimensions + 1, new_values)
        return new_ps_state, self._log_probability_ratio_birth(
            i_nearest, old_ps_state, idx_insert, new_ps_state
            )       

    def death(self, old_ps_state: ParameterSpaceState):
        # prepare for death perturbation
        n_cells = old_ps_state.n_dimensions
        if n_cells == self.n_dimensions_min:
            raise DimensionalityException("Death")
        # randomly choose an existing Voronoi site to kill
        iremove = random.randint(0, n_cells - 1)
        # remove parameter values for the removed site
        new_values = dict()
        for name, value in old_ps_state.param_values.items():
            old_values = getattr(old_ps_state, name)
            new_values[name] = delete(old_values, iremove)
        new_ps_state = ParameterSpaceState(self.n_dimensions - 1, new_values) 
        new_sites = getattr(new_ps_state, self.name)
        old_sites = getattr(old_ps_state, self.name)
        i_nearest = nearest_index(
            xp=old_sites[iremove], x=new_sites, xlen=new_sites.size
        )
        return new_ps_state, -self._log_probability_ratio_birth(
            i_nearest, new_ps_state, iremove, old_ps_state
            )
    
    @staticmethod
    def compute_cell_extents(voronoi_sites: np.ndarray, lb=0, ub=-1, fill_value=0):
        """compute Voronoi cell extents from the Voronoi sites. Voronoi-cell
        boundaries are first drawn at the midpoint between consecutive Voronoi
        nuclei. The extent is then derived from the distance between consecutive
        boundaries.

        Parameters
        ----------
        voronoi_sites : np.ndarray of shape (n,)
            Voronoi-site positions. These should be greater or equal to zero

        lb, ub : float
            Lower and upper bounds used in the calculation of Voronoi-cell
            extents. Negative values for `lb` or `ub` denote an unbounded cell.
            The extent of an unbounded cell is set to `fill_value`

        fill_value : float
            Value attributed to unbounded Voronoi cells

        Returns
        -------
        np.ndarray
            Voronoi-cell extents

        Examples
        --------
        >>> depth = np.array([2, 5.5, 8, 10])

        >>> Voronoi1D.compute_cell_extents(depth, lb=0, ub=-1, fill_value=np.nan)
        array([3.75, 3.  , 2.25,  nan])

        >>> Voronoi1D.compute_cell_extents(depth, lb=-1, ub=-1, fill_value=np.nan)
        array([ nan, 3.  , 2.25,  nan])

        >>> Voronoi1D.compute_cell_extents(depth, lb=0, ub=15, fill_value=np.nan)
        array([3.75, 3.  , 2.25, 6.  ])
        """
        return compute_voronoi1d_cell_extents(
            voronoi_sites, lb=lb, ub=ub, fill_value=fill_value
        )

    @staticmethod
    def plot_depth_profile(
        samples_voronoi_cell_extents,
        samples_param_values,
        bins=100,
        ax=None,
        **kwargs,
    ):
        """plot a 2D histogram (depth profile) of points including the interfaces.

        Parameters
        ----------
        samples_voronoi_cell_extents : ndarray
            A 2D numpy array where each row represents a sample of thicknesses (or
            Voronoi cell extents)
        samples_param_values : ndarray
            A 2D numpy array where each row represents a sample of parameter values
            (e.g., velocities)
        bins : int or [int, int], optional
            The number of bins to use along each axis (default is 100). If you pass a
            single int, it will use that
            many bins for both axes. If you pass a list of two ints, it will use the
            first for the x-axis (velocity)
            and the second for the y-axis (depth).
        ax : Axes, optional
            An optional Axes object to plot on
        kwargs : dict, optional
            Additional keyword arguments to pass to ax.hist2d

        Returns
        -------
        ax : Axes
            The Axes object containing the plot
        """
        if ax is None:
            _, ax = plt.subplots()
        depths = []
        velocities = []
        for thicknesses, values in zip(
            samples_voronoi_cell_extents, samples_param_values
        ):
            depths.extend(np.cumsum(np.array(thicknesses)))
            velocities.extend(values)
        # plotting the 2D histogram
        cax = ax.hist2d(velocities, depths, bins=bins, **kwargs)
        # colorbar (for the histogram density)
        cbar = plt.colorbar(cax[3], ax=ax)
        cbar.set_label("Density")
        if ax.get_ylim()[0] < ax.get_ylim()[1]:
            ax.invert_yaxis()
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Depth (km)")
        return ax

    @staticmethod
    def plot_interface_distribution(
        samples_voronoi_cell_extents,
        bins=100,
        ax=None,
        **kwargs,
    ):
        """plot the 1D depth distribution

        Parameters
        ----------
        samples_voronoi_cell_extents : list
            a list of voronoi cell extents (thicknesses in the 1D case)
        bins : int, optional
            number of vertical bins, by default 100
        ax : matplotlib.axes.Axes, optional
            an optional user-provided ax, by default None

        Returns
        -------
        matplotlib.axes.Axes
            the resulting plot that has the depth distribution on it
        """
        if ax is None:
            _, ax = plt.subplots()
        depths = []
        for thicknesses in samples_voronoi_cell_extents:
            depths.extend(np.cumsum(thicknesses))
        # calculate 1D histogram
        h, e = np.histogram(depths, bins=bins, density=True)
        # plot the histogram
        ax.barh(e[:-1], h, height=np.diff(e), align="edge", label="histogram", **kwargs)
        if ax.get_ylim()[0] < ax.get_ylim()[1]:
            ax.invert_yaxis()
        ax.set_xlabel("p(discontinuity)")
        ax.set_ylabel("Depth (km)")
        return ax

    @staticmethod
    def plot_param_samples(
        samples_voronoi_cell_extents: list,
        samples_param_values: list,
        ax=None,
        **kwargs,
    ):
        """plot multiple 1D Earth models based on sampled parameters.

        Parameters
        ----------
        samples_voronoi_cell_extents : list
            a list of voronoi cell extents (thicknesses in the 1D case)
        samples_param_values : ndarray
            a 2D numpy array where each row represents a sample of parameter values
            (e.g., velocities)
        ax : Axes, optional
            an optional Axes object to plot on
        kwargs : dict, optional
            additional keyword arguments to pass to ax.step

        Returns
        -------
        ax : Axes
            The Axes object containing the plot
        """
        if ax is None:
            _, ax = plt.subplots()

        # Default plotting style for samples
        sample_style = {
            "linewidth": kwargs.pop("linewidth", kwargs.pop("lw", 0.5)),
            "alpha": 0.2,
            "color": kwargs.pop(
                "color", kwargs.pop("c", "blue")
            ),  # Fixed color for the sample lines
        }
        sample_style.update(kwargs)  # Override with any provided kwargs

        for thicknesses, values in zip(
            samples_voronoi_cell_extents, samples_param_values
        ):
            thicknesses = np.insert(thicknesses[:-1], -1, 20)
            y = np.insert(np.cumsum(thicknesses), 0, 0)
            x = np.insert(values, 0, values[0])
            ax.step(x, y, where="post", **sample_style)

        if ax.get_ylim()[0] < ax.get_ylim()[1]:
            ax.invert_yaxis()
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Depth (km)")

        return ax

    @staticmethod
    def interpolate_samples(
        samples_voronoi_cell_extents, samples_param_values, interp_positions
    ):
        """interpolates the samples given the site positions and interpolate points

        Parameters
        ----------
        samples_voronoi_cell_extents : list
            a list of voronoi cell extents (thicknesses in the 1D case)
        samples_param_values : list
            a list of physical parameter values to draw statistics from
        interp_positions : _type_
            points to interpolate

        Returns
        -------
        np.ndarray
            the resulting samples
        """
        interp_params = np.zeros((len(samples_param_values), len(interp_positions)))
        for i, (sample_extents, sample_values) in enumerate(
            zip(samples_voronoi_cell_extents, samples_param_values)
        ):
            interp_params[i, :] = interpolate_result(
                np.array(sample_extents), np.array(sample_values), interp_positions
            )
        return interp_params

    @staticmethod
    def get_ensemble_statistics(
        samples_voronoi_cell_extents: list,
        samples_param_values: list,
        interp_positions: np.ndarray,
        percentiles=(10, 90),
    ) -> dict:
        """get the mean, median, std and percentiles of the given ensemble

        Parameters
        ----------
        samples_voronoi_cell_extents : list
            a list of voronoi cell extents (thicknesses in the 1D case)
        samples_param_values : list
            a list of physical parameter values to draw statistics from
        interp_positions : _type_
            points to interpolate
        percentiles : tuple, optional
            percentiles to calculate, by default (10, 90)

        Returns
        -------
        dict
            a dictionary with these keys: "mean", "median", "std" and "percentile"
        """
        interp_params = Voronoi1D.interpolate_samples(
            samples_voronoi_cell_extents, samples_param_values, interp_positions
        )
        statistics = {
            "mean": np.mean(interp_params, axis=0),
            "median": np.median(interp_params, axis=0),
            "std": np.std(interp_params, axis=0),
            "percentiles": np.percentile(interp_params, percentiles, axis=0),
        }
        return statistics

    @staticmethod
    def plot_ensemble_statistics(
        samples_voronoi_cell_extents: list,
        samples_param_values: list,
        interp_positions: np.ndarray,
        percentiles=(10, 90),
        ax=None,
        **kwargs,
    ):
        """plot the mean, median, std and percentiles from the given samples

        Parameters
        ----------
        samples_voronoi_cell_extents : list
            a list of voronoi cell extents (thicknesses in the 1D case)
        samples_param_values : list
            a list of physical parameter values to draw statistics from
        interp_positions : _type_
            points to interpolate
        percentiles : tuple, optional
            percentiles to calculate, by default (10, 90)
        ax : matplotlib.axes.Axes, optional
            an optional user-provided ax, by default None

        Returns
        -------
        matplotlib.axes.Axes
            the resulting plot that has the statistics on it
        """
        statistics = Voronoi1D.get_ensemble_statistics(
            samples_voronoi_cell_extents,
            samples_param_values,
            interp_positions,
            percentiles,
        )
        mean = statistics["mean"]
        std = statistics["std"]
        percentiles = statistics["percentiles"]
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(mean, interp_positions, color="b", label="Mean")
        ax.plot(mean - std, interp_positions, "b--")
        ax.plot(mean + std, interp_positions, "b--")
        ax.plot(statistics["median"], interp_positions, color="orange", label="Median")
        ax.plot(
            percentiles[0],
            interp_positions,
            color="orange",
            ls="--",
        )
        ax.plot(
            percentiles[1],
            interp_positions,
            color="orange",
            ls="--",
        )
        ax.legend()
        return ax
