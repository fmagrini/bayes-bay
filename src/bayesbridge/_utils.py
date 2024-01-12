from .exceptions import UserFunctionException
from ._state import State


def _preprocess_func(func):
    if func is None:
        return None
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


class _LogLikeRatioFromFunc:
    """Log likelihood ratio function from log likelihood function"""

    def __init__(self, log_likelihood_func):
        self.f = log_likelihood_func
    
    def __call__(self, old_state: State, new_state: State):
        try:
            old_loglike = self.f(old_state)
            new_loglike = self.f(new_state)
        except Exception as e:
            raise UserFunctionException(e)
        return new_loglike - old_loglike

    @property
    def __name__(self) -> str:
        return self.f.__name__
