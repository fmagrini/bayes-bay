from .exceptions import UserFunctionException
from .perturbations._base_perturbation import Perturbation


def _preprocess_func(func):
    if func is None:
        return None
    elif isinstance(func, Perturbation):
        return func
    f = None
    args = []
    kwargs = {}
    if isinstance(func, (tuple, list)) and len(func) > 1:
        f = func[0]
        if isinstance(func[1], (tuple, list)):
            args = func[1]
            if len(func) > 2 and isinstance(func[2], dict):
                kwargs = func[2]
        elif isinstance(func[1], dict):
            kwargs = func[1]
        else:
            raise TypeError(
                "additional arguments should be a list (i.e. args) or a dict (i.e. "
                f"kwargs), but got {func[1]} for the function instead"
            )
    elif isinstance(func, (tuple, list)):
        f = func[0]
    else:
        f = func
    return _FunctionWrapper(f, args, kwargs)


class _FunctionWrapper:
    """Function wrapper to make it pickleable (credit to emcee)"""

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args or []
        self.kwargs = kwargs or {}

    def __call__(self, *args):
        try:
            return self.f(*args, *self.args, **self.kwargs)
        except Exception as e:
            raise UserFunctionException(e)

    @property
    def __name__(self) -> str:
        return self.f.__name__

    def __repr__(self) -> str:
        return self.__name__
