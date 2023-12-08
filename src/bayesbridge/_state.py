from dataclasses import dataclass, field
from typing import Dict, Any
from numbers import Number
import numpy as np


@dataclass
class State:
    """Data structure that stores a model state, including all the necessary 
    information to perform the forward operation

    Parameters
    ----------
    n_voronoi_cells : int
        number of Voronoi cells
    voronoi_sites : np.ndarray
        array containing the Voronoi site positions
    param_values : Dict[str, np.ndarray]
        dictionary containing parameter values
    noise_std : Number, optional
        standard deviation of the noise
    noise_correlation : Number, optional
        correlation of the noise
    cache : Dict[str, Any], optional
        cache for storing intermediate results

    Raises
    ------
    TypeError
        when ``n_voronoi_cells`` is not an int
    TypeError
        when ``voronoi_sites`` is not a numpy ndarray
    TypeError
        when ``param_values`` is not a dict
    AssertionError
        when the length of ``voronoi_sites`` isn't aligned with ``n_voronoi_cells``
    """

    n_voronoi_cells: int
    voronoi_sites: np.ndarray
    param_values: Dict[str, np.ndarray] = field(default_factory=dict)
    noise_std: Number = None
    noise_correlation: Number = None
    cache: Dict[str, Any] = field(default_factory=dict)
    extra_storage: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.n_voronoi_cells, int):
            raise TypeError("n_voronoi_cells should be an int")
        if not isinstance(self.voronoi_sites, np.ndarray):
            raise TypeError("voronoi_sites should be a numpy ndarray")
        if not isinstance(self.param_values, dict):
            raise TypeError("param_values should be a dict")
        assert (
            len(self.voronoi_sites) == self.n_voronoi_cells
        ), "lengths of voronoi_sites should be the same as n_voronoi_cells"
        for name, values in self.param_values.items():
            self.set_param_values(name, values)

    def set_param_values(self, param_name: str, values: np.ndarray):
        """Changes the value(s) of a parameter

        Parameters
        ----------
        param_name : str
            the parameter name (i.e. the key in the ``param_values`` dict)
        values : np.ndarray
            the value(s) to be set for the given ``param_name``
        """
        if param_name in ["n_voronoi_cells", "voronoi_sites"]:
            raise AttributeError(
                f"'{param_name}' attribute already exists on the State object."
            )
        if not isinstance(values, np.ndarray):
            raise TypeError("parameter values should be a numpy ndarray")
        self.param_values[param_name] = values
        setattr(self, param_name, values)

    def get_param_values(self, param_name: str) -> np.ndarray:
        """Get the value(s) of a parameter

        Parameters
        ----------
        param_name : str
            the parameter name (i.e. the key in the ``param_values`` dict)

        Returns
        -------
        np.ndarray
            the value(s) of the given ``param_name``
        """
        return getattr(self, param_name, None)

    def has_cache(self, name: str) -> bool:
        """Indicates whether there is cache value stored for the given ``name``

        Parameters
        ----------
        name : str
            the cache name to look up

        Returns
        -------
        bool
            whether there is cache stored for the given ``name``
        """
        return name in self.cache

    def load_cache(self, name: str) -> Any:
        """Load the cached value for the given ``name``

        Parameters
        ----------
        name : str
            the cache name to look up

        Returns
        -------
        Any
            the cache stored for the given ``name``
        """
        return self.cache[name]

    def store_cache(self, name: str, value: Any):
        """Store the given value to cache

        Parameters
        ----------
        name : str
            the cache name to store
        value : Any
            the cache value to store
        """
        self.cache[name] = value

    def _vars(self):
        return {
            k: v
            for k, v in vars(self).items()
            if not (k == "noise_std" and v is None)
            and not (k == "noise_correlation" and v is None)
            and k != "param_values"
            and k != "cache"
        }

    def __iter__(self):
        return iter(self._vars())

    def items(self):
        """Key-value pairs of all the values in the current model, expanding all
        parameter values, excluding cache

        Returns
        -------
        dict_items
            the key-value dict pairs of all the attributes
        """
        return self._vars().items()

    def clone(self) -> "State":
        """Creates a clone of the current State itself

        Returns
        -------
        State
            the clone of self
        """
        _n_voronoi_cells = self.n_voronoi_cells
        _voronoi_sites = self.voronoi_sites.copy()
        _noise_std = self.noise_std
        _noise_corr = self.noise_correlation
        _param_values = dict()
        for k, v in self.param_values.items():
            _param_values[k] = v.copy()
        return State(
            _n_voronoi_cells,
            _voronoi_sites,
            _param_values,
            _noise_std,
            _noise_corr,
        )

    def __hash__(self):
        voronoi_sites_sum = np.sum(self.voronoi_sites)
        voronoi_sites_min = np.min(self.voronoi_sites)
        voronoi_sites_max = np.max(self.voronoi_sites)
        voronoi_sites_hash = hash(
            (voronoi_sites_sum, voronoi_sites_min, voronoi_sites_max)
        )
        param_values_hash = hash(
            (hash((k, np.sum(v), np.min(v), np.max(v))) for k, v in self.param_values)
        )
        return hash(
            (
                self.n_voronoi_cells,
                voronoi_sites_hash,
                param_values_hash,
                self.noise_std,
                self.noise_correlation,
            )
        )
