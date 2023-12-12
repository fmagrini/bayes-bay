from dataclasses import dataclass, field
from collections import namedtuple
from typing import Dict, Any, Union
import numpy as np


_DataNoise = namedtuple("DataNoise", ["std", "correlation"])

class DataNoise(_DataNoise):
    def copy(self) -> "DataNoise":
        return self._replace()


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
    param_values : Dict[str, Union[np.ndarray, DataNoise]]
        dictionary containing parameter values, e.g. 
        ``{"vs": np.ndarray([4,5]), "rayleigh": DataNoise(std=0.1, correlation=0.01)}``
    cache : Dict[str, Any], optional
        cache for storing intermediate results
    extra_storage: Dict[str, Any], optional
        extra storage that will be saved into results (e.g. when one calls
        :meth:`BayesianInversion.get_results`)

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
    param_values: Dict[str, Union[np.ndarray, DataNoise]] = field(default_factory=dict)
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

    def set_param_values(
        self, param_name: str, values: Union[np.ndarray, DataNoise]
    ):
        """Changes the value(s) of a parameter

        Parameters
        ----------
        param_name : str
            the parameter name (i.e. the key in the ``param_values``)
        values : Union[np.ndarray, DataNoise]
            the value(s) to be set for the given ``param_name``
        """
        if param_name in ["n_voronoi_cells", "voronoi_sites"]:
            raise AttributeError(
                f"'{param_name}' attribute already exists on the State object."
            )
        if isinstance(param_name, str):
            if not isinstance(values, (np.ndarray, DataNoise)):
                raise TypeError("parameter values should either be a numpy ndarray or a `DataNoise` instance")
            self.param_values[param_name] = values
            setattr(self, param_name, values)
        else:
            raise ValueError("`param_name` should be a string")

    def get_param_values(self, param_name: str) -> Union[np.ndarray, DataNoise]:
        """Get the value(s) of a parameter

        Parameters
        ----------
        param_name : str
            the parameter name (i.e. the key in the ``param_values`` dict)

        Returns
        -------
        Union[np.ndarray, DataNoise]
            the value(s) of the given ``param_name``
        """
        if isinstance(param_name, str):
            return self.param_values.get(param_name, None)
        else:
            raise ValueError("`param_name` should be a string")

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
        all_vars = {
            k: v
            for k, v in vars(self).items()
            if not (k == "noise_std" and v is None)
            and not (k == "noise_correlation" and v is None)
            and k != "param_values"
            and k != "cache"
            and k != "extra_storage"
        }
        all_vars.update(self.extra_storage)
        return all_vars

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
        """Creates a clone of the current State itself, in which the following will be
        (deep-)copied over:

        - :attr:`n_voronoi_cells`
        - :attr:`voronoi_sites`
        - :attr:`param_values`

        And the following won't be copied at all:

        - :attr:`cache`
        - :attr:`extra_storage`

        Returns
        -------
        State
            the clone of self
        """
        _n_voronoi_cells = self.n_voronoi_cells
        _voronoi_sites = self.voronoi_sites.copy()
        _param_values = dict()
        for k, v in self.param_values.items():
            _param_values[k] = v.copy()
        return State(
            n_voronoi_cells=_n_voronoi_cells,
            voronoi_sites=_voronoi_sites,
            param_values=_param_values,
        )
