from dataclasses import dataclass, field
from typing import Dict
from numbers import Number
import numpy as np


@dataclass
class State:
    """Data structure that stores the necessary information to perform the forward
    operation

    Raises
    ------
    TypeError
        when ``n_voronoi_cells`` is not an int
    TypeError
        when ``voronoi_sites`` is not a numpy ndarray
    AssertionError
        when the length of ``voronoi_sites`` isn't aligned with ``n_voronoi_cells``
    """
    n_voronoi_cells: int
    voronoi_sites: np.ndarray
    param_values: Dict[str, np.ndarray] = field(default_factory=dict)
    noise_std: Number = None
    noise_correlation: Number = None
    
    def __post_init__(self):
        if not isinstance(self.n_voronoi_cells, int):
            raise TypeError("n_voronoi_cells should be an int")
        if not isinstance(self.voronoi_sites, np.ndarray):
            raise TypeError("voronoi_sites should be a numpy ndarray")
        assert len(self.voronoi_sites) == self.n_voronoi_cells, \
            "lengths of voronoi_sites should be the same as n_voronoi_cells"
        for name, values in self.param_values.items():
            self.set_param_values(name, values)
        
    def set_param_values(self, param_name, values):
        if param_name in ["n_voronoi_cells", "voronoi_sites"]:
            raise AttributeError(f"'{param_name}' attribute already exists on the State object.")
        if not isinstance(values, np.ndarray):
            raise TypeError("parameter values should be a numpy ndarray")
        self.param_values[param_name] = values
        setattr(self, param_name, values)

    def get_param_values(self, name):
        return getattr(self, name, None)

    def __iter__(self):
        return iter(vars(self))

    def items(self):
        return vars(self).items()
    
    def clone(self) -> "State":
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
