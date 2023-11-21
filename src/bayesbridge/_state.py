from dataclasses import dataclass, field
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
    param_values: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.n_voronoi_cells, int):
            raise TypeError("n_voronoi_cells should be an int")
        if not isinstance(self.voronoi_sites, np.ndarray):
            raise TypeError("voronoi_sites should be a numpy ndarray")
        assert len(self.voronoi_sites) == self.n_voronoi_cells, \
            "lengths of voronoi_sites should be the same as n_voronoi_cells"
        
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

    def clone_from(self, other):
        self.n_voronoi_cells = other.n_voronoi_cells
        self.voronoi_sites = other.voronoi_sites.copy()
        self.param_values = {}
        for k, v in other.param_values.items():
            self.set_param_values(k, v.copy())
