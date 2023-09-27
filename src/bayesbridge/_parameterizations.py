from copy import deepcopy
from functools import partial
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from ._state import State
from ._exceptions import InitException, DimensionalityException
from ._utils_bayes import (
    _get_thickness,
    _is_sorted,
    _interpolate_result,
    nearest_index,
    interpolate_linear_1d,
)


TWO_PI = 2 * math.pi
SQRT_TWO_PI = math.sqrt(TWO_PI)


class Model:
    def __init__(self, n_voronoi_cells, voronoi_sites):
        voronoi_cell_extents = self._get_voronoi_cell_extents(voronoi_sites)
        self.current_state = self._init_current_state(
            n_voronoi_cells, voronoi_sites, voronoi_cell_extents
        )
        self.proposed_state = deepcopy(self.current_state)
        self._current_perturbation = {}
        self._finalize_dict = dict(
            site=self._finalize_site_perturbation,
            free_param=self._finalize_free_param_perturbation,
            birth=self._finalize_birth_death_perturbation,
            death=self._finalize_birth_death_perturbation,
        )

    @property
    def current_perturbation(self):
        return self._current_perturbation

    def _init_current_state(self, n_voronoi_cells, voronoi_sites, voronoi_cell_extents):
        return State(n_voronoi_cells, voronoi_sites, voronoi_cell_extents)

    def _get_voronoi_cell_extents(self, voronoi_sites):
        return _get_thickness(voronoi_sites)

    def add_free_parameter(self, name, values):
        self.current_state.set_param_values(name, values)
        self.proposed_state.set_param_values(name, values.copy())

    def propose_site_perturbation(self, isite, site):
        self.proposed_state.voronoi_sites[isite] = site
        self._sort_proposed_state()
        self._current_perturbation["type"] = "site"

    def propose_free_param_perturbation(self, name, idx, value):
        getattr(self.proposed_state, name)[idx] = value
        self._current_perturbation["type"] = name
        self._current_perturbation["idx"] = idx

    def propose_birth_perturbation(self):
        self.proposed_state.n_voronoi_cells += 1
        self._sort_proposed_state()
        self._current_perturbation["type"] = "birth"

    def propose_death_perturbation(
        self,
    ):
        self.proposed_state.n_voronoi_cells -= 1
        self.proposed_state.voronoi_cell_extents = self._get_voronoi_cell_extents(
            self.proposed_state.voronoi_sites
        )
        self._current_perturbation["type"] = "death"

    def finalize_perturbation(self, accepted):
        perturb_type = self._current_perturbation["type"]
        if perturb_type in self._finalize_dict:
            finalize = self._finalize_dict[perturb_type]
        else:
            finalize = self._finalize_dict["free_param"]
        accepted_state = self.proposed_state if accepted else self.current_state
        rejected_state = self.current_state if accepted else self.proposed_state
        finalize(accepted_state, rejected_state)

    def _finalize_site_perturbation(self, accepted_state, rejected_state):
        rejected_state.voronoi_sites = accepted_state.voronoi_sites.copy()
        rejected_state.voronoi_cell_extents = accepted_state.voronoi_cell_extents.copy()

    def _finalize_free_param_perturbation(self, accepted_state, rejected_state):
        name = self._current_perturbation["type"]
        idx = self._current_perturbation["idx"]
        getattr(rejected_state, name)[idx] = getattr(accepted_state, name)[idx]

    def _finalize_birth_death_perturbation(self, accepted_state, rejected_state):
        rejected_state.clone_from(accepted_state)

    def _sort_proposed_state(self):
        isort = np.argsort(self.proposed_state.voronoi_sites)
        if not _is_sorted(isort):
            self.proposed_state.voronoi_sites = self.proposed_state.voronoi_sites[isort]
            for name, values in self.proposed_state.param_values.items():
                self.proposed_state.set_param_values(name, values[isort])

        self.proposed_state.voronoi_cell_extents = self._get_voronoi_cell_extents(
            self.proposed_state.voronoi_sites
        )


class Parameterization:
    def perturbation_birth(self):
        raise NotImplementedError

    def perturbation_death(self):
        raise NotImplementedError

    def perturbation_voronoi_site(self):
        raise NotImplementedError

    def perturbation_free_param(self):
        raise NotImplementedError

    def finalize_perturbation(self):
        raise NotImplementedError


class Parameterization1D(Parameterization):
    def __init__(
        self,
        voronoi_site_bounds,
        voronoi_site_perturb_std,
        position=None,
        n_voronoi_cells=None,
        free_params=None,
        n_voronoi_cells_min=None,
        n_voronoi_cells_max=None,
        voronoi_cells_init_range=0.2,
    ):
        self.voronoi_site_bounds = voronoi_site_bounds
        self._voronoi_site_perturb_std = self._init_voronoi_site_perturb_std(
            voronoi_site_perturb_std, position
        )

        self._trans_d = n_voronoi_cells is None
        self._n_voronoi_cells = n_voronoi_cells
        self.n_voronoi_cells_min = n_voronoi_cells_min
        self.n_voronoi_cells_max = n_voronoi_cells_max

        self._initialized = False
        self._voronoi_cells_init_range = voronoi_cells_init_range

        self.free_params = {}
        if free_params is not None:
            for param in free_params:
                self.add_free_parameter(param)

    @property
    def trans_d(self):
        return self._trans_d

    @property
    def initialized(self):
        return self._initialized

    def initialize(self):
        if self.trans_d:
            cells_range = self._voronoi_cells_init_range
            cells_min = self.n_voronoi_cells_min
            cells_max = self.n_voronoi_cells_max
            init_max = int((cells_max - cells_min) * cells_range + cells_min)
            n_voronoi_cells = random.randint(cells_min, init_max)
        else:
            n_voronoi_cells = self._n_voronoi_cells
        voronoi_sites = self._init_voronoi_sites(n_voronoi_cells)
        self.model = Model(n_voronoi_cells, voronoi_sites)
        for free_param in self.free_params.values():
            self._init_free_parameter(free_param, voronoi_sites)
        del self._n_voronoi_cells
        self._initialized = True

    def __getattr__(self, attr):
        valid_attr = attr in ["n_voronoi_cells", "voronoi_sites", "model"]
        if valid_attr and not self.initialized:
            message = f"The {self.__class__.__name__} instance has not been"
            message += " initialized yet. Run `.initialize()` or pass it"
            message += " to `BayesianInversion`."
            raise InitException(message)
        else:
            return self.__getattribute__(attr)

    def add_free_parameter(self, free_param):
        self.free_params[free_param.name] = free_param
        if self.initialized:
            self._init_free_parameter(free_param)

    def _init_free_parameter(self, free_param, voronoi_sites):
        values = free_param.generate_random_values(voronoi_sites, is_init=True)
        self.model.add_free_parameter(free_param.name, values)

    def _init_voronoi_sites(self, n_voronoi_cells):
        lb, ub = self.voronoi_site_bounds
        return np.sort(np.random.uniform(lb, ub, n_voronoi_cells))

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

    def get_voronoi_site_perturb_std(self, site):
        if np.isscalar(self._voronoi_site_perturb_std):
            return self._voronoi_site_perturb_std
        return self._voronoi_site_perturb_std(site)

    def perturbation_birth(self):
        n_cells = self.model.current_state.n_voronoi_cells
        if n_cells == self.n_voronoi_cells_max:
            raise DimensionalityException("Birth")
        old_sites = self.model.proposed_state.voronoi_sites
        while True:
            lb, ub = self.voronoi_site_bounds
            new_site = random.uniform(lb, ub)
            if np.any(np.abs(new_site - old_sites) < 1e-2):
                continue
            break

        self.model.proposed_state.voronoi_sites = np.append(old_sites, new_site)

        isite = nearest_index(xp=new_site, x=old_sites, xlen=old_sites.size)
        prob_ratio = 0
        for param_name, param in self.free_params.items():
            old_values = getattr(self.model.current_state, param_name)
            old_value = old_values[isite]

            new_value = param.perturb_value(new_site, old_value)
            prob_ratio += self.probability_ratio_birth_death_perturbation(
                param, new_site, old_value, new_value
            )
            self.model.proposed_state.set_param_values(param_name, np.append(old_values, new_value))

        self.model.propose_birth_perturbation()
        return prob_ratio

    def perturbation_death(self):
        n_cells = self.model.current_state.n_voronoi_cells
        if n_cells == self.n_voronoi_cells_min:
            raise DimensionalityException("Death")
        isite = random.randint(0, n_cells - 1)
        site_to_remove = self.model.current_state.voronoi_sites[isite]
        old_sites = self.model.current_state.voronoi_sites
        new_sites = np.delete(old_sites, isite)
        self.model.proposed_state.voronoi_sites = new_sites

        iclosest = nearest_index(xp=site_to_remove, x=new_sites, xlen=new_sites.size)
        prob_ratio = 0
        for param_name, param in self.free_params.items():
            old_values = getattr(self.model.current_state, param_name)
            new_values = np.delete(old_values, isite)
            self.model.proposed_state.set_param_values(param_name, new_values)
            old_value = old_values[isite]
            new_value = new_values[iclosest]

            prob_ratio -= self.probability_ratio_birth_death_perturbation(
                param, site_to_remove, old_value, new_value
            )
        
        self.model.propose_death_perturbation()
        return prob_ratio

    def perturbation_voronoi_site(self):
        nsites = self.model.current_state.n_voronoi_cells
        isite = random.randint(0, nsites - 1)
        voronoi_sites = self.model.current_state.voronoi_sites
        old_site = voronoi_sites[isite]
        site_min, site_max = self.voronoi_site_bounds
        std = self.get_voronoi_site_perturb_std(old_site)

        while True:
            random_deviate = random.normalvariate(0, std)
            new_site = old_site + random_deviate
            if new_site < site_min or new_site > site_max:
                continue
            if np.any(np.abs(new_site - voronoi_sites) < 1e-2):
                continue
            break
        self.model.propose_site_perturbation(isite, new_site)
        return self.probability_ratio_site_perturbation(old_site, new_site)

    def perturbation_free_param(self, param_name):
        nsites = self.model.current_state.n_voronoi_cells
        isite = random.randint(0, nsites - 1)
        site = self.model.current_state.voronoi_sites[isite]
        old_value = getattr(self.model.current_state, param_name)[isite]
        new_value = self.free_params[param_name].perturb_value(site, old_value)
        self.model.propose_free_param_perturbation(param_name, isite, new_value)
        return self.probability_ratio_free_param_perturbation(param_name)

    def finalize_perturbation(self, accepted):
        self.model.finalize_perturbation(accepted)

    def probability_ratio_birth_death_perturbation(
        self, param, perturbed_site, value, perturbed_value
    ):
        """
        Returns probability ratio associated with a single free parameter
        """
        std_perturb = param.get_perturb_std(perturbed_site)
        delta = param.get_delta(perturbed_site)
        term1 = math.log(SQRT_TWO_PI * std_perturb / delta)
        term2 = (perturbed_value - value) ** 2 / (2 * std_perturb**2)
        return term1 + term2

    def probability_ratio_site_perturbation(self, old_site, new_site):
        proposal_ratio = self._proposal_ratio_site_perturbation(old_site, new_site)
        prior_ratio = self._prior_ratio_site_perturbation(old_site, new_site)
        return proposal_ratio + prior_ratio

    def probability_ratio_free_param_perturbation(self, param_name):
        param = self.free_params[param_name]
        prior_ratio = param.prior_ratio_value_perturbation()
        proposal_ratio = param.proposal_ratio_value_perturbation()
        return prior_ratio + proposal_ratio

    def _proposal_ratio_site_perturbation(self, old_site, new_site):
        std_old = self.get_voronoi_site_perturb_std(old_site)
        std_new = self.get_voronoi_site_perturb_std(new_site)
        d = (old_site - new_site) ** 2
        term1 = math.log(std_old / std_new)
        term2 = d * (std_new**2 - std_old**2) / (2 * std_new**2 * std_old**2)
        return term1 + term2

    def _prior_ratio_site_perturbation(self, old_site, new_site):
        prob_ratio = 0
        for param in self.free_params.values():
            prob_ratio += param.prior_ratio_position_perturbation(old_site, new_site)
        return prob_ratio

    @staticmethod
    def get_ensemble_statistics(
        samples_voronoi_cell_extents,
        samples_param_values,
        interp_positions,
        percentiles=(10, 90),
    ):
        interp_params = Parameterization1D.interpolate_samples(
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
        samples_voronoi_cell_extents,
        samples_param_values,
        interp_positions,
        percentiles=(10, 90),
        ax=None,
        **kwargs,
    ):
        statistics = Parameterization1D.get_ensemble_statistics(
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
        interp_params = np.zeros((len(samples_param_values), len(interp_positions)))
        for i, (sample_extents, sample_values) in enumerate(
            zip(samples_voronoi_cell_extents, samples_param_values)
        ):
            interp_params[i, :] = _interpolate_result(
                np.array(sample_extents), np.array(sample_values), interp_positions
            )
        return interp_params

    def plot_param_current(self, param_name, ax=None, **kwargs):
        """Plot the 1D Earth model given a param_name.

        Parameters
        ----------
        param_name: str
            name of the free parameter to plot values
        ax: matplotlib.axes.Axes, optional
            An optional Axes object to plot on
        kwargs: dict, optional
            Additional keyword arguments to pass to ax.step

        Returns
        -------
        ax
            The Axes object containing the plot
        """
        if param_name not in self.free_params:
            raise ValueError(
                "%s is not in the free parameters list: %s"
                % (param_name, list(self.free_params.keys()))
            )
        current = self.model.current_state
        values = current[param_name]
        thicknesses = np.array(current["voronoi_cell_extents"])
        thicknesses[-1] = 20
        y = np.insert(np.cumsum(thicknesses), 0, 0)
        x = np.insert(values, 0, values[0])

        # plotting
        if ax is None:
            _, ax = plt.subplots()
        ax.step(x, y, where="post", **kwargs)
        if ax.get_ylim()[0] < ax.get_ylim()[1]:
            ax.invert_yaxis()
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Depth (km)")
        return ax

    @staticmethod
    def plot_param_samples(
        samples_voronoi_cell_extents, samples_param_values, ax=None, **kwargs
    ):
        """Plot multiple 1D Earth models based on sampled parameters.

        Parameters
        ----------
        samples_voronoi_cell_extents : ndarray
            A 2D numpy array where each row represents a sample of thicknesses (or Voronoi cell extents)

        samples_param_values : ndarray
            A 2D numpy array where each row represents a sample of parameter values (e.g., velocities)

        ax : Axes, optional
            An optional Axes object to plot on

        kwargs : dict, optional
            Additional keyword arguments to pass to ax.step

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
        """
        Plot a histogram of the distribution of the number of Voronoi cells.

        Parameters
        ----------
        samples_n_voronoi_cells : list or ndarray
            List or array containing the number of Voronoi cells in each sample

        ax : Axes, optional
            An optional Axes object to plot on

        kwargs : dict, optional
            Additional keyword arguments to pass to ax.hist

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

    def plot_depth_profile(
        self,
        samples_voronoi_cell_extents,
        samples_param_values,
        bins=100,
        ax=None,
        **kwargs,
    ):
        """
        Plot a 2D histogram (depth profile) of points including the interfaces.

        Parameters
        ----------
        samples_voronoi_cell_extents : ndarray
            A 2D numpy array where each row represents a sample of thicknesses (or Voronoi cell extents)

        samples_param_values : ndarray
            A 2D numpy array where each row represents a sample of parameter values (e.g., velocities)

        bins : int or [int, int], optional
            The number of bins to use along each axis (default is 100). If you pass a single int, it will use that
            many bins for both axes. If you pass a list of two ints, it will use the first for the x-axis (velocity)
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
            thicknesses = np.array(thicknesses)
            y_depth = np.insert(np.cumsum(thicknesses), 0, 0)
            x_vel = np.insert(values, 0, values[0])
            depths.extend(y_depth)
            velocities.extend(x_vel)
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
