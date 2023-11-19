from abc import abstractmethod
from numbers import Number

class Perturbation:
    def __init__(self):
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def __run__(self) -> Number:
        raise NotImplementedError
