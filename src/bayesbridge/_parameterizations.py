from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Dict, Callable
from numbers import Number
import random
import numpy as np
import matplotlib.pyplot as plt

from .parameters import Parameter
from ._state import State
from .perturbations._birth_death import (
    BirthFromNeighbour1D,
    BirthFromPrior1D,
    DeathFromNeighbour1D,
    DeathFromPrior1D,
)
from .perturbations._param_values import ParamPerturbation
from .perturbations._site_positions import Voronoi1DPerturbation
from ._utils1d import interpolate_result, compute_voronoi1d_cell_extents


class Parameterization(ABC):
    """Parameterization class that can be configured by users to generate perturbation
    functions for easier inference setup
    """

    @property
    @abstractmethod
    def trans_d(self) -> bool:
        """indicates whether the current configuration allows changes in the
        discretization itself
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Parameter]:
        """all the unknown parameters under this parameterization setting"""
        raise NotImplementedError

    @property
    @abstractmethod
    def perturbation_functions(self) -> List[Callable[[State], Tuple[State, Number]]]:
        """a list of perturbation functions allowed in the current parameterization
        configurations, each of which takes in a model :class:`State` and returns a new
        model and a log proposal ratio value
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def log_prior_ratio_functions(self) -> List[Callable[[State, State], Number]]:
        """a list of log prior ratio functions corresponding to each of the
        :meth:`perturbation_functions`
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self) -> State:
        """initializes the parameterization (if it's trans dimensional) and the
        parameter values

        Returns
        -------
        State
            an initial model state
        """
        raise NotImplementedError


class Voronoi1D(Parameterization):
    """One dimensional Voronoi nuclei parameterization

    Parameters
    ----------
    voronoi_site_bounds : Tuple[Number, Number]
        the minimum and maximum values of the 1D voronoi site positions
    voronoi_site_perturb_std : Union[Number, np.ndarray]
        the perturbation standard deviation of the Voronoi site positions
    position : np.ndarray, optional
        the breaking points for varied ``voronoi_site_perturb_std``. Activated only
        when the ``voronoi_site_bounds`` and / or ``voronoi_site_perturb_std`` is
        not a scalar, by default None
    n_voronoi_cells : Number, optional
        the number of Voronoi cells. Needs to be None if this is a
        trans-dimensional parameterization, by default None
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
        voronoi_site_bounds: Tuple[Number, Number],
        voronoi_site_perturb_std: Union[Number, np.ndarray],
        position: np.ndarray = None,
        n_voronoi_cells: Number = None,
        free_params: List[Parameter] = None,
        n_voronoi_cells_min: Number = None,
        n_voronoi_cells_max: Number = None,
        voronoi_cells_init_range: Number = 0.2,
        birth_from: str = "neighbour",  # either "neighbour" or "prior"
    ):
        self.voronoi_site_bounds = voronoi_site_bounds
        self.voronoi_site_perturb_std = voronoi_site_perturb_std
        self.position = position
        self._trans_d = n_voronoi_cells is None
        self._n_voronoi_cells = n_voronoi_cells
        self.n_voronoi_cells_min = n_voronoi_cells_min
        self.n_voronoi_cells_max = n_voronoi_cells_max
        self._voronoi_cells_init_range = voronoi_cells_init_range
        self._birth_from = birth_from
        self.free_params = {}
        if free_params is not None:
            for param in free_params:
                self.free_params[param.name] = param
        self._init_perturbation_funcs()
        self._init_log_prior_ratio_funcs()

    @property
    def trans_d(self) -> bool:
        """indicates whether the current configuration allows changes in the
        discretization itself
        """
        return self._trans_d

    @property
    def parameters(self) -> Dict[str, Parameter]:
        """all the unknown parameters under this parameterization setting"""
        return self.free_params

    def initialize(self) -> State:
        """initializes the parameterization (if it's trans dimensional) and the
        parameter values

        Returns
        -------
        State
            an initial model state
        """
        # initialize number of cells
        if self.trans_d:
            cells_range = self._voronoi_cells_init_range
            cells_min = self.n_voronoi_cells_min
            cells_max = self.n_voronoi_cells_max
            init_max = int((cells_max - cells_min) * cells_range + cells_min)
            n_voronoi_cells = random.randint(cells_min, init_max)
        else:
            n_voronoi_cells = self._n_voronoi_cells
            del self._n_voronoi_cells
        # initialize site positions
        lb, ub = self.voronoi_site_bounds
        voronoi_sites = np.sort(np.random.uniform(lb, ub, n_voronoi_cells))
        # initialize parameter values
        param_vals = dict()
        for name, param in self.free_params.items():
            param_vals[name] = param.initialize(voronoi_sites)
        return State(n_voronoi_cells, voronoi_sites, param_vals)

    def _init_perturbation_funcs(self):
        self._perturbation_funcs = [
            Voronoi1DPerturbation(
                parameters=self.parameters,
                voronoi_site_bounds=self.voronoi_site_bounds,
                voronoi_site_perturb_std=self.voronoi_site_perturb_std,
                position=self.position,
            )
        ]
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
