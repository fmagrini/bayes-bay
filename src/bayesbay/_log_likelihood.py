import math
from typing import Any, List, Callable, Tuple, Union
from numbers import Number
import numpy as np

from .exceptions import ForwardException
from ._state import State
from ._target import Target
from .perturbations._data_noise import NoisePerturbation
from ._utils import _preprocess_func


# class BaseLogLikelihood:
#     """Base API class to evaluate the log likelihood ratio
#     """
#     pass


class LogLikelihood:
    """Helper class to evaluate the log likelihood ratio

    One out of the following three sets of input needs to be supplied:

    1. ``target`` and ``fwd_functions``
    2. ``log_like_ratio_func``
    3. ``log_like_func``

    The input will be taken in precedence as above. For example, if both ``target``
    and ``fwd_functions`` are supplied, then ``log_like_func`` and
    ``log_like_ratio_func`` are ignored regardless of their values, and so on.

    This class then will wrap the input functions properly so that the
    :meth:`log_likelihood_ratio` is ready for use by the Markov chains.

    If ``targets`` are supplied and if any of them have unknown data noise, and if the
    high level API is used (i.e. ``BayesianInversion`` instead of
    ``BaseBayesianInversion`` is used), then the perturbations of the data noises will
    be taken into account regardless of whether ``fwd_functions`` is given, and the
    state for each chain will be initialized accordingly for the unknown data noise.

    Parameters
    ----------
    targets : bayesbay.Target, optional
        a list of data targets, default to None
    fwd_functions : List[Callable[[bayesbay.State], np.ndarray]], optional
        a list of forward functions corresponding to each data targets provided above.
        Each function takes in a model and produces a numpy array of data predictions.
        Default to None.
    log_like_ratio_func: Callable[[Any, Any], Number], optional
        the log likelihood ratio function :math:`\log(\frac{p(\mathbf{d}_{obs} \mid
        \mathbf{m'})} {p(\mathbf{d}_{obs} \mid \mathbf{m})})`. It takes the current and
        proposed models, :math:`\mathbf{m}` and :math:`\mathbf{m'}`, whose type should
        be consistent with the other arguments of this class, and returns a scalar
        corresponding to the log likelihood ratio. This is utilised in the calculation
        of the acceptance probability. If None, ``log_like_func`` gets used
        instead. Default to None
    log_like_func: Callable[[Any], Number], optional
        the log likelihood function :math:`\log(p(\mathbf{d}_{obs} \mid \mathbf{m}))`.
        It takes in a model :math:`\mathbf{m}` (any type is allowed, as long as it is
        consistent with the other arguments of this class) and returns the log
        of the likelihood function. This function is only used when ``log_like_ratio_func``
        is None. Default to None
    """

    def __init__(
        self,
        targets: List[Target] = None,
        fwd_functions: List[Callable[[State], np.ndarray]] = None,
        log_like_ratio_func: Callable[[Any, Any], Number] = None,
        log_like_func: Callable[[Any], Number] = None,
    ):
        self.targets = targets
        self.fwd_functions = fwd_functions
        self.log_like_ratio_func = log_like_ratio_func
        self.log_like_func = log_like_func
        self._check_duplicate_target_names()
        self._init_log_likelihood_ratio()
        self._init_perturbation_funcs()

    @property
    def perturbation_functions(self) -> List[Callable[[State], Tuple[State, Number]]]:
        """A list of perturbation functions associated with the data noise of the
        provided targets.

        Perturbation functions are included in this list only when the data noise of
        the target(s) is explicitly set to be unknown(s).
        """
        return self._perturbation_funcs

    def initialize(self, state: State):
        if self.targets is not None:
            for target in self.targets:
                target.initialize(state)

    def log_likelihood_ratio(
        self,
        old_state: Union[State, Any],
        new_state: Union[State, Any],
        temperature: Number = 1,
    ) -> Number:
        r"""Returns the (possibly tempered) log of the likelihood ratio

        .. math::
            \left[
                \frac{p\left(\mathbf{d}_{obs} \mid \mathbf{m'}\right)}{p\left(\mathbf{d}_{obs} \mid \mathbf{m}\right)}
            \right]^{\frac{1}{T}}
            =
            \left[
            \frac{\lvert \mathbf{C}_e \rvert}{\lvert \mathbf{C}^{\prime}_e \rvert}
            \exp\left(- \frac{\Phi(\mathbf{m'}) - \Phi(\mathbf{m})}{2}\right)
            \right]^{\frac{1}{T}},


        where :math:`\mathbf{C}_e` denotes the data covariance matrix,
        :math:`\Phi(\mathbf{m})` the data misfit associated with the model
        :math:`\mathbf{m}`, :math:`T` the chain temperature, and the prime
        superscript indicates that the model has been perturbed.


        Parameters
        ----------
        old_state : bayesbay.State
            the state of the Bayesian inference prior to the model perturbation
        new_state : bayesbay.State
            the state of the Bayesian inference after the model perturbation
        temperature : Number
            the temperature associated with current chain and iteration

        Returns
        -------
        Number
            log likelihood ratio
        """
        return self._log_likelihood_ratio(old_state, new_state) / temperature

    def _log_likelihood_ratio_from_targets(
        self, old_state: State, new_state: State
    ) -> Number:
        old_misfit, old_log_det = self._get_misfit_and_det(old_state)
        new_misfit, new_log_det = self._get_misfit_and_det(new_state)
        log_like_ratio = (old_log_det - new_log_det) + (old_misfit - new_misfit) / 2
        return log_like_ratio

    def _log_likelihood_ratio_from_loglike(
        self, old_state: Union[State, Any], new_state: Union[State, Any]
    ) -> Number:
        old_loglike = self.log_like_func(old_state)
        new_loglike = self.log_like_func(new_state)
        return new_loglike - old_loglike

    def _log_likelihood_ratio_from_loglike_ratio(
        self, old_state: Union[State, Any], new_state: Union[State, Any]
    ) -> Number:
        return self.log_like_ratio_func(old_state, new_state)

    def _check_duplicate_target_names(self):
        if self.targets is not None:
            all_target_names = [t.name for t in self.targets]
            if len(all_target_names) != len(set(all_target_names)):
                raise ValueError("duplicate target names found")

    def _init_log_likelihood_ratio(self):
        if self.targets is not None and self.fwd_functions is not None:
            _fwd_functions = self.fwd_functions
            _targets = self.targets
            if not isinstance(_fwd_functions, list):
                _fwd_functions = [_fwd_functions]
            if not isinstance(_targets, list):
                _targets = [_targets]
            assert len(_fwd_functions) == len(_targets)
            self.targets = _targets
            self.fwd_functions = [_preprocess_func(func) for func in _fwd_functions]
            self._log_likelihood_ratio = self._log_likelihood_ratio_from_targets
        elif self.log_like_ratio_func is not None:
            self.log_like_ratio_func = _preprocess_func(self.log_like_ratio_func)
            self._log_likelihood_ratio = self._log_likelihood_ratio_from_loglike_ratio
        elif self.log_like_func is not None:
            self.log_like_func = _preprocess_func(self.log_like_func)
            self._log_likelihood_ratio = self._log_likelihood_ratio_from_loglike
        else:
            raise ValueError(
                "please provide one out of the following three sets of input: \n"
                "\t1. ``targets`` and ``fwd_functions``\n"
                "\t2. ``log_like_ratio_func``\n"
                "\t3. ``log_like_func``"
            )

    def _init_perturbation_funcs(self):
        if self.targets is not None:
            hier_targets = (
                [t for t in self.targets if t.is_hierarchical]
                if self.targets is not None
                else []
            )
            self._perturbation_funcs = (
                [NoisePerturbation(self.targets)] if hier_targets else []
            )
        else:
            self._perturbation_funcs = []

    def _get_misfit_and_det(self, state: State) -> Tuple[Number, Number]:
        misfit = 0
        log_det = 0
        for target, fwd_func in zip(self.targets, self.fwd_functions):
            if state.saved_in_cache("dpred"):
                dpred = state.load_from_cache("dpred")
            else:
                try:
                    dpred = fwd_func(state)
                except Exception as e:
                    raise ForwardException(e)
                state.save_to_cache("dpred", dpred)
            residual = dpred - target.dobs
            misfit += residual @ target.inverse_covariance_times_vector(state, residual)
            if target.is_hierarchical:
                log_det += target.log_determinant_covariance(state)
        if not isinstance(misfit, Number):
            raise TypeError(
                f"misfit is expected to be of a Number type, but is {type(misfit)} "
                "instead. This might be due to predicted data not being 1D. Try using "
                "numpy.squeeze() function on predicted data before returning it from "
                "your forward function"
            )
        return misfit, log_det
