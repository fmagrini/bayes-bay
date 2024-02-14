from dataclasses import dataclass, field
from collections import namedtuple
from typing import Dict, Any, Union
from numbers import Number
import numpy as np


_DataNoiseState = namedtuple("DataNoiseState", ["std", "correlation"])


class DataNoiseState(_DataNoiseState):
    """Data structure that stores the state of the data noise parameters during
    the inference.
    """

    def copy(self) -> "DataNoiseState":
        """Returns a deep-copy of the current DataNoiseState

        Returns
        -------
        DataNoiseState
            the clone of self
        """
        return self._replace()

    def todict(self, name: str) -> dict:
        """Returns a dictionary containing the noise properties

        Parameters
        ----------
        name : str
            identifier for the ``DataNoiseState`` instance

        Returns
        -------
        dict
            dictionary object with keys

            - :attr:`name`.std
            - :attr:`name`.correlation (only returned if set as a free parameter,
              see :class:`Target`)
        """
        res = {f"{name}.std": self.std}
        if self.correlation is not None:
            res[f"{name}.correlation"] = self.correlation
        return res


@dataclass
class ParameterSpaceState:
    """Data structure that stores the state of a parameter space.

    Parameters
    ----------
    n_dimensions : int
        number of dimensions characterizing the parameter space
    param_values : Dict[str, Union[ParameterSpaceState, DataNoiseState]]
        dictionary containing parameter values, e.g.
        ``{"voronoi": ParameterSpaceState(3, {"voronoi": np.array([1,2,3]), "vs":
        np.array([4,5,6])}), "rayleigh": DataNoiseState(std=0.01,
        correlation=None)}``

    Raises
    ------
    TypeError
        when ``param_values`` is not a dict
    """

    n_dimensions: int
    param_values: Dict[str, np.ndarray] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.n_dimensions, int):
            raise TypeError("n_dimensions should be an int")
        if not isinstance(self.param_values, dict):
            raise TypeError("param_values should be a dict")
        for name, values in self.param_values.items():
            if len(values) != self.n_dimensions:
                raise ValueError(
                    f"parameter {name} should have the same length as `n_dimensions` "
                    f"({self.n_dimensions}) but have {len(values)} instead"
                )
            self.set_param_values(name, values)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.get_param_values(name)

    def set_param_values(self, param_name: str, values: np.ndarray):
        """Changes the numerical value(s) of a parameter

        Parameters
        ----------
        param_name : str
            the parameter name (i.e. the key in the ``param_values``)
        values : np.ndarray
            the value(s) to be set for the given ``param_name``
        """
        if isinstance(param_name, str):
            if not isinstance(values, np.ndarray):
                raise TypeError("parameter values should be a numpy ndarray instance")
            self.param_values[param_name] = values
            setattr(self, param_name, values)
        else:
            raise ValueError("`param_name` should be a string")

    def get_param_values(self, param_name: str) -> np.ndarray:
        """Get the value(s) of a parameter

        Parameters
        ----------
        param_name : str
            the parameter name (i.e. the key in the ``param_values`` dict)

        Returns
        -------
        Union[np.ndarray, None]
            the value(s) of the given ``param_name``, if present in
            :attr:`param_values`
        """
        if isinstance(param_name, str):
            return self.param_values.get(param_name, None)
        else:
            raise ValueError("`param_name` should be a string")

    def saved_in_cache(self, name: str) -> bool:
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

    def load_from_cache(self, name: str) -> Any:
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

    def save_to_cache(self, name: str, value: Any):
        """Store the given value to cache

        Parameters
        ----------
        name : str
            the cache name to store
        value : Any
            the cache value to store
        """
        self.cache[name] = value

    def copy(self) -> "ParameterSpaceState":
        """Returns a clone of self

        Returns
        -------
        ParameterSpaceState
        """
        new_param_values = dict()
        for name, param_vals in self.param_values.items():
            new_param_values[name] = param_vals.copy()
        new_ps_state = ParameterSpaceState(self.n_dimensions, new_param_values)
        new_ps_state.cache = self.cache.copy()
        return new_ps_state

    def todict(self, name: str) -> dict:
        """Returns a dictionary containing the numerical values defining the
        parameter space

        Parameters
        ----------
        name : str
            identifier for the ``ParameterSpaceState`` instance

        Returns
        -------
        dict
            dictionary object with keys

            - :attr:`name`.std
            - :attr:`name`.correlation (only returned if set as a free parameter,
              see :class:`Target`)
        """
        _discretization = "discretization"
        res = {f"{name}.n_dimensions": self.n_dimensions}
        res.update({f"{name}.{k}": v for k, v in self.param_values.items()})
        return res


@dataclass
class State:
    """Data structure that stores a Markov chain state, including all the necessary
    information to perform the forward operation

    Parameters
    ----------
    param_values : Dict[str, Union[ParameterSpaceState, DataNoiseState]]
        dictionary containing parameter values, e.g.
        ``{"voronoi": ParameterSpaceState(3, {"voronoi": np.array([1,2,3]), "vs":
        np.array([4,5,6])}), "rayleigh": DataNoiseState(std=0.01,
        correlation=None)}``
    temperature : float
        the temperature of the Markov chain associated with this state
    cache : Dict[str, Any], optional
        cache for storing intermediate results
    extra_storage: Dict[str, Any], optional
        extra storage that will be saved into results (e.g. when one calls
        :meth:`BayesianInversion.get_results`)

    Raises
    ------
    TypeError
        when ``param_values`` is not a dict
    """

    param_values: Dict[str, Union[ParameterSpaceState, DataNoiseState]] = field(
        default_factory=dict
    )
    temperature: float = 1
    cache: Dict[str, Any] = field(default_factory=dict)
    extra_storage: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.temperature, Number):
            raise TypeError("temperature should be a number")
        if not isinstance(self.param_values, dict):
            raise TypeError("param_values should be a dict")
        for name, values in self.param_values.items():
            self.set_param_values(name, values)

    def __getitem__(self, name: str) -> Union[ParameterSpaceState, DataNoiseState]:
        return self.get_param_values(name)

    def set_param_values(
        self, param_name: str, values: Union[ParameterSpaceState, DataNoiseState]
    ):
        """Changes the value(s) of a parameter

        Parameters
        ----------
        param_name : str
            the parameter name (i.e. the key in the ``param_values``)
        values : Union[ParameterSpaceState, DataNoiseState]
            the value(s) to be set for the given ``param_name``
        """
        if isinstance(param_name, str):
            if not isinstance(values, (ParameterSpaceState, DataNoiseState)):
                raise TypeError(
                    "parameter values should either be a ParameterSpaceState or a "
                    "`DataNoiseState` instance"
                )
            self.param_values[param_name] = values
            setattr(self, param_name, values)
        else:
            raise ValueError("`param_name` should be a string")

    def get_param_values(
        self, param_name: str
    ) -> Union[ParameterSpaceState, DataNoiseState]:
        """Get the value(s) of a parameter

        Parameters
        ----------
        param_name : str
            the parameter name (i.e. the key in the ``param_values`` dict)

        Returns
        -------
        Union[ParameterSpaceState, DataNoiseState]
            the value(s) of the given ``param_name``
        """
        if isinstance(param_name, str):
            return self.param_values.get(param_name, None)
        else:
            raise ValueError("`param_name` should be a string")

    def saved_in_cache(self, name: str) -> bool:
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

    def load_from_cache(self, name: str) -> Any:
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

    def save_to_cache(self, name: str, value: Any):
        """Store the given value to cache

        Parameters
        ----------
        name : str
            the cache name to store
        value : Any
            the cache value to store
        """
        self.cache[name] = value
    
    def saved_in_extra_storage(self, name: str) -> bool:
        """Indicates whether there is an extra_storage value stored for the given 
        ``name``

        Parameters
        ----------
        name : str
            the extra_storage name to look up

        Returns
        -------
        bool
            whether there is extra_storage stored for the given ``name``
        """
        return name in self.extra_storage

    def load_from_extra_storage(self, name: str) -> Any:
        """Load the extra_storage value for the given ``name``

        Parameters
        ----------
        name : str
            the extra_storage name to look up

        Returns
        -------
        Any
            the extra_storage stored for the given ``name``
        """
        return self.extra_storage[name]

    def save_to_extra_storage(self, name: str, value: Any):
        """Store the given value to extra_storage

        Parameters
        ----------
        name : str
            the extra_storage name to store
        value : Any
            the extra_storage value to store
        """
        self.extra_storage[name] = value

    def _vars(self):
        all_vars = dict()
        for k, v in self.param_values.items():
            all_vars.update(v.todict(k))
        all_vars.update(self.extra_storage)
        return all_vars

    def __iter__(self):
        return iter(self._vars())

    def items(self):
        """Key-value pairs of all the values in the current state, expanding all
        parameter values, excluding cache

        Returns
        -------
        dict_items
            the key-value dict pairs of all the attributes
        """
        return self._vars().items()

    def copy(self, keep_dpred: bool = False) -> "State":
        """Creates a clone of the current State itself

        The following will be (deep-)copied over:

        - :attr:`param_values`
        - :attr:`temperature`

        And the following won't be copied at all:

        - :attr:`cache`
        - :attr:`extra_storage`

        If ``keep_dpred`` is ``True``, then the ``dpred`` in ``cache`` will be referred
        to in the new instance. Note that this is not a deep copy, since we assume no
        changes will be performed on predicted data for a certain state.

        Parameters
        ----------
        keep_dpred : bool, optional
            whether to copy over the predicted data (stored in cache)

        Returns
        -------
        State
            the clone of self
        """
        _param_values = dict()
        for k, v in self.param_values.items():
            _param_values[k] = v.copy()
        _state_args = {"param_values": _param_values, "temperature": self.temperature}
        if keep_dpred:
            _state_args["cache"] = {k: v for k, v in self.cache.items() if k == "dpred"}
        return State(**_state_args)
