from abc import ABC, abstractmethod
from typing import Union
from numbers import Number
import numpy as np

from .._state import State


class Parameter(ABC):
    def __init__(self, **kwargs):
        self.init_params = kwargs
    
    @abstractmethod
    def random_sample(self, model: State):
        raise NotImplementedError
    
    @abstractmethod
    def perturb_value(self, value: Number):
        raise NotImplementedError
    
    @abstractmethod
    def log_prior(self, value: Number):
        raise NotImplementedError
