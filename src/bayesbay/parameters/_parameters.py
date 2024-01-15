#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:56:00 2022

@author: fabrizio
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, Union, Dict
from numbers import Number
import random
from functools import partial
import math
import numpy as np
from .._utils_1d import interpolate_linear_1d


TWO_PI = 2 * math.pi
SQRT_TWO_PI = math.sqrt(TWO_PI)


class Parameter(ABC):
    """Base class for an unknown parameter"""

    def __init__(self, **kwargs):
        self._name = kwargs["name"]
        if "position" in kwargs and kwargs["position"] is not None:
            self.position = np.array(kwargs["position"], dtype=float)
        else:
            self.position = None
        assert "perturb_std" in kwargs
        self.init_params = kwargs
        
    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def initialize(
        self, position: Union[np.ndarray, Number] = None
    ) -> Union[np.ndarray, Number]:
        r"""initializes the values of this (possibly position-dependent) 
        parameter

        Parameters
        ----------
        position : Union[np.ndarray, Number], optional
            the position (in the discretization domain) associated with the value, 
            None by default

        Returns
        -------
        Union[np.ndarray, Number]
            an array of values or one value corresponding to the given positions
        """
        raise NotImplementedError

    @abstractmethod
    def perturb_value(self, value: Number, position: Number = None) -> Tuple[Number, Number]:
        r"""perturbs the given value, in a way that may depend on the position 
        associated with such a parameter value, and calculates the log of the 
        corresponding partial acceptance probability 
        
        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} = 
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}} 
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}  
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}},

        Parameters
        ----------
        value : Number
            the current value to be perturbed from
        position : Number, optional
            the position (in the discretization domain) associated with the value, 
            None by default

        Returns
        -------
        Tuple[Number, Number]
            the perturbed value and
            :math:`\alpha_{p} = \log(
            \frac{p({\bf m'})}{p({\bf m})}
            \frac{q\left({\bf m} 
            \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            \lvert \mathbf{J} \rvert)`
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self, value: Number, position: Number = None) -> Number:
        r"""calculates the log of the prior probability of occurrence of the
        given value, which may depend on the considered position

        Parameters
        ----------
        value : Number
            the value to calculate the probability density for
        position : Number, optional
            the position (in the discretization domain) associated with the value, 
            None by default
        
        Returns
        -------
        Number
            the log prior probability density
        """
        raise NotImplementedError

    def set_custom_initialize(
        self,
        initialize_func: Callable[
            ["Parameter", Union[np.ndarray, Number]], Union[np.ndarray, Number]
        ],
    ):
        r"""sets a custom initialization function

        Parameters
        ----------
        initialize_func: Callable[[bayesbay.parameters.Parameter, Union[np.ndarray, Number]], Union[np.ndarray, Number]]
            The function to use for initialization. This function should take no arguments.

        Examples
        --------
        .. code-block:: python

            def my_init(
                param: bb.parameters.Parameter,
                position: Union[np.ndarray, Number]
            ) -> Union[np.ndarray, Number]:
                print("This is my custom init!")

            my_param.set_custom_initialize(my_init)
        """
        if not callable(initialize_func):
            raise ValueError("initialize_func must be a callable function.")
        self.initialize = partial(self._initializer, initialize_func)

    def _initializer(self, initialize_func, *args, **kwargs):
        return initialize_func(self, *args, **kwargs)

    def get_perturb_std(
        self, position: Union[Number, np.ndarray] = None
    ) -> Union[Number, np.ndarray]:
        r"""get the standard deviation of the Gaussian used to perturb
        the parameter, which may be dependent on the position in the
        discretization domain
        
        Parameters
        ----------
        position: Union[Number, np.ndarray], optional
            the position in the discretization domain at which the standard 
            deviation of the Gaussian used to perturb the parameter is returned.
            Default is None

        Returns
        -------
        Union[Number, np.ndarray]
            standard deviation of the Gaussian used to perturb the parameter,
            possibly at the specified position(s)
        """
        if self.has_hyper_param("perturb_std"):
            return self.get_hyper_param("perturb_std", position)
        else:
            raise NotImplementedError("`get_perturb_std` needs to be implemented")

    def add_hyper_params(self, hyper_params: Dict[str, Union[Number, np.ndarray]]):
        r"""Sets the attributes from the given dict and checks for errors
        
        Parameters
        ----------
        hyper_params : Dict[str, Union[Number, np.ndarray]]
            dictionary of attributes to be set
        """
        msg_scalar_when_pos_none = "should be a scalar when `position` is `None`"
        msg_scalar_or_same_len = (
            "should either be a scalar or have the same length as `position`"
        )
        for name, param_val in hyper_params.items():
            if self.position is None:
                assert np.isscalar(param_val), f"`{name}` {msg_scalar_when_pos_none}"
            else:
                assert np.isscalar(param_val) or \
                    len(param_val) == len(self.position), \
                        f"`{name}` {msg_scalar_or_same_len}"
            if not np.isscalar(param_val):
                param_val = np.array(param_val, dtype=float)
            setattr(self, name, self._init_hyper_param(param_val))
    
    def has_hyper_param(self, hyper_param: str) -> bool:
        r"""Whether or not the :class:`Parameter` instance has the specified
        attribute
        
        Parameters
        ----------
        hyper_param : str
            the name of the attribute
        """
        return hasattr(self, hyper_param)

    def get_hyper_param(
            self, 
            hyper_param: str, 
            position: Union[Number, np.ndarray] = None
            ) -> Union[Number, np.ndarray]:
        r"""Retrieves the value corresponding to the specified attribute, which 
        may be a function of position
        
        Parameters
        ----------
        hyper_param : str
            the name of the attribute
        position : Union[np.ndarray, Number], optional
            the position (in the discretization domain) associated with the value, 
            None by default
            
        Returns
        -------
        Union[Number, np.ndarray]
            value corresponding to the specified attribute
        """
        _hyper_param = getattr(self, hyper_param)
        if np.isscalar(_hyper_param):
            return _hyper_param
        elif position is not None:
            return _hyper_param(position)
        else:
            raise ValueError(
                f"`{hyper_param}` is dependent on position but got None for it"
            )

    def _init_hyper_param(self, hyper_param: Union[Number, np.ndarray]):
        # to be called after self.position is assigned
        return (
            hyper_param
            if np.isscalar(hyper_param)
            else partial(interpolate_linear_1d, x=self.position, y=hyper_param)
        )

    def __repr__(self) -> str:
        string = "%s(" % self.init_params["name"]
        for k, v in self.init_params.items():
            if k == "name":
                continue
            string += "%s=%s, " % (k, v)
        string = string[:-2]
        return string + ")"


class UniformParameter(Parameter):
    r"""Class for defining a free parameter according to a uniform probability
    distribution

    Parameters
    ----------
    name : str
        name of the parameter, for display and storing purposes
    vmin : Union[Number, np.ndarray]
        the lower bound for this parameter. This can either be a scalar or an array
        if the defined probability distribution is a function of ``position``
        in the discretization domain
    vmax : Union[Number, np.ndarray]
        the upper bound for this parameter. This can either be a scalar or an array
        if the defined probability distribution is a function of ``position``
        in the discretization domain
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the parameter. 
        This can either be a scalar or an array if the defined probability 
        distribution is a function of ``position`` in the discretization domain
    position : np.ndarray, optional
        position in the discretization domain, used to define a position-dependent
        probability distribution. None by default
    """

    def __init__(
        self,
        name: str,
        vmin: Union[Number, np.ndarray],
        vmax: Union[Number, np.ndarray],
        perturb_std: Union[Number, np.ndarray],
        position: np.ndarray = None,
    ):
        super().__init__(
            name=name,
            position=position,
            vmin=vmin,
            vmax=vmax,
            perturb_std=perturb_std,
        )
        self.add_hyper_params({
            "vmin": vmin, 
            "vmax": vmax, 
            "perturb_std": perturb_std, 
            "delta": np.array(vmax) - np.array(vmin), 
        })

    def get_delta(
        self, position: Union[Number, np.ndarray] = None
    ) -> Union[Number, np.ndarray]:
        r"""get the range :math:`\Delta v = v_{max} - v_{min}`, which may be
        dependent on the specified position in the discretization domain

        Parameters
        ----------
        position: Union[Number, np.ndarray], optional
            the position in the discretization domain at which the uniform 
            distribution range will be returned. None by default

        Returns
        -------
        Union[Number, np.ndarray]
            the range :math:`\Delta v = v_{max} - v_{min}`
        """
        return self.get_hyper_param("delta", position)

    def get_vmin_vmax(
        self, position: Union[Number, np.ndarray] = None
    ) -> Union[Tuple[Number, Number], Tuple[np.ndarray, np.ndarray]]:
        r"""get the lower and upper bounds of the parameter, which may be 
        dependent on position in the discretization domain

        Parameters
        ----------
        position: Union[Number, np.ndarray], optional
            the position in the discretization domain at which the the lower 
            and upper bounds of the parameter will be returned. None by default

        Returns
        -------
        Tuple[float, float]
            the lower (``vmin``) and upper (``vmax``)
        """
        vmin = self.get_hyper_param("vmin", position)
        vmax = self.get_hyper_param("vmax", position)
        return vmin, vmax

    def initialize(
        self, position: Union[np.ndarray, Number] = None
    ) -> Union[np.ndarray, Number]:
        r"""initialize the parameter, possibly at a specific position in the 
        discretization domain
        
        Parameters
        ----------
        position: Union[Number, np.ndarray], optional
            the position in the discretization domain at which the parameter is 
            initialized. None by default

        Returns
        -------
        Union[Number, np.ndarray]
            random number(s) chosen uniformly in the range associated with the
            given position(s)
        """
        vmin, vmax = self.get_vmin_vmax(position)
        if isinstance(position, Number):
            return random.uniform(vmin, vmax)
        else:
            n_vals = len(position) if position is not None else 1
            return np.random.uniform(vmin, vmax, n_vals)

    def perturb_value(self, value: Number, position: Number = None) -> Tuple[Number, Number]:
        r"""perturbs the given value, in a way that may depend on the position 
        in the discretization domain and calculates the log of the corresponding 
        partial acceptance probability,
        
        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} = 
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}} 
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}  
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}},

        which in this case equals zero. In the above equation, :math:`\bf m'` denotes 
        the perturbed model as obtained through a random deviate from a normal 
        distribution :math:`\mathcal{N}(v, \sigma)`, where :math:`v` denotes 
        the original ``value`` and :math:`\sigma` the standard deviation of the 
        Gaussian used for the perturbation. :math:`\sigma` may be dependent
        on the specified position (:attr:`UniformParameter.perturb_std`).

        Parameters
        ----------
        value : Number
            the current value to be perturbed from
        position : Number, optional
            the position in the discretization domain at which the parameter 
            ``value`` is to be perturbed

        Returns
        -------
        Tuple[Number, Number]
            the perturbed value and
            :math:`\alpha_{p} = \log(
            \frac{p({\bf m'})}{p({\bf m})}
            \frac{q\left({\bf m} 
            \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            \lvert \mathbf{J} \rvert) = 0`
        """
        # randomly perturb the value until within range
        std = self.get_perturb_std(position)
        vmin, vmax = self.get_vmin_vmax(position)
        while True:
            random_deviate = random.normalvariate(0, std)
            new_value = value + random_deviate
            if new_value >= vmin and new_value <= vmax:
                return new_value, 0

    def log_prior(self, 
                  value: Number, 
                  position: Union[Number, np.ndarray] = None) -> Number:
        r"""calculates the log of the prior probability density for the given 
        value, which may be dependent on the position in the discretization
        domain

        Parameters
        ----------
        value : Number
            the value to calculate the probability density for
        position : Number, optional
            the position in the discretization domain at which the prior
            probability of the parameter ``value`` is to be retrieved

        Returns
        -------
        Number
            the log of the prior probability :math:`p(v) = \frac{1}{\Delta v}`,
            where :math:`v` denotes the ``value`` passed by the user and 
            :math:`\Delta v = v_{max} - v_{min}` the range within which the
            uniform parameter is allowed to vary. This may be
            dependent on position in the discretization domain
        """
        vmin, vmax = self.get_vmin_vmax(position)
        if vmin <= value <= vmax:
            return -math.log(vmax - vmin)
        else:
            return -math.inf


class GaussianParameter(Parameter):
    """Class for defining a free parameter using a Gaussian probability density
    function :math:`\mathcal{N}(\mu, \sigma)`, where :math:`\mu` denotes 
    the mean and :math:`\sigma` the standard deviation of the Gaussian

    Parameters
    ----------
    name : str
        name of the parameter, for display and storing purposes
    mean : Union[Number, np.ndarray]
        mean of the Gaussian. This can either be a scalar or an array
        if the defined probability distribution is a function of ``position``
        in the discretization domain
    std : Union[Number, np.ndarray]
        standard deviation of the Gaussian. This can either be a scalar or an array
        if the defined probability distribution is a function of ``position``
        in the discretization domain
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the parameter. 
        This can either be a scalar or an array if the defined probability distribution 
        is a function of ``position`` in the discretization domain
    position : np.ndarray, optional
        position in the discretization domain, used to define a position-dependent
        probability distribution. None by default
    """

    def __init__(self, name, mean, std, perturb_std, position=None):
        super().__init__(
            name=name,
            position=position,
            mean=mean,
            std=std,
            perturb_std=perturb_std,
        )
        self.add_hyper_params({
            "mean": mean, 
            "std": std, 
            "perturb_std": perturb_std, 
        })
    
    def get_mean(
        self, position: Union[Number, np.ndarray] = None
    ) -> Union[Number, np.ndarray]:
        """get the mean of the Gaussian parameter, which may be dependent on
        position in the discretization domain

        Parameters
        ----------
        position: Union[Number, np.ndarray], optional
            position(s) in the discretization domain at which the mean of the 
            Gaussian will be returned. None by default

        Returns
        -------
        Union[Number, np.ndarray]
            mean at the given position(s)
        """
        return self.get_hyper_param("mean", position)

    def get_std(
        self, position: Union[Number, np.ndarray] = None
    ) -> Union[Number, np.ndarray]:
        """get the standard deviation of the Gaussian parameter, which may be 
        dependent on position in the discretization domain

        Parameters
        ----------
        position: Union[Number, np.ndarray], optional
            position(s) in the discretization domain at which the standard 
            deviation of the Gaussian will be returned. None by default

        Returns
        -------
        Union[Number, np.ndarray]
            standard deviation at the given position(s)
        """
        return self.get_hyper_param("std", position)

    def initialize(
        self, position: Union[np.ndarray, Number] = None
    ) -> Union[np.ndarray, Number]:
        r"""initialize the parameter, possibly at a specific position in the 
        discretization domain
        
        Parameters
        ----------
        position: Union[Number, np.ndarray], optional
            the position in the discretization domain at which the parameter is 
            initialized. None by default

        Returns
        -------
        Union[np.ndarray, Number]
            an array of values or one value corresponding to the given position(s),
            chosen according to the normal distribution defined
            by :attr:`mean` and :attr:`std` at the given position(s)
        """
        mean = self.get_mean(position)
        std = self.get_std(position)
        n_vals = len(position) if position is not None else 1
        values = np.random.normal(mean, std, n_vals)
        return values

    def perturb_value(self, 
                      value: Number, 
                      position: Union[Number, np.ndarray] = None) -> Tuple[Number, Number]:
        r"""perturbs the given value, in a way that may depend on the position 
        in the discretization, and calculates the log of the corresponding 
        partial acceptance probability,
        
        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} = 
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}} 
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}  
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}},

        where :math:`\bf m'` denotes the perturbed model as obtained through a 
        random deviate from a normal distribution :math:`\mathcal{N}(v, \theta)`, 
        where :math:`v` denotes the original ``value`` and :math:`\theta` the 
        standard deviation of the Gaussian used for the perturbation. 
        :math:`\theta` may be dependent on the specified position 
        (:attr:`GaussianParameter.perturb_std`).
        
        Parameters
        ----------
        value : Number
            the current value to be perturbed from
        position : Number
            the position of the value to be perturbed
        
        Returns
        -------
        Tuple[Number, Number]
            the perturbed value and
            :math:`\alpha_{p} = \log(
            \frac{p({\bf m'})}{p({\bf m})}
            \frac{q\left({\bf m} 
            \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            \lvert \mathbf{J} \rvert)`
        """
        perturb_std = self.get_perturb_std(position)
        random_deviate = random.normalvariate(0, perturb_std)
        new_value = value + random_deviate
        mean = self.get_mean(position)
        std = self.get_std(position)
        ratio = (value - new_value) * (value + new_value - 2 * mean) / (2 * std**2)
        return new_value, ratio
    
    def log_prior(self, 
                  value: Number, 
                  position: Union[Number, np.ndarray] = None) -> Number:
        r"""calculates the log of the prior probability density for the given 
        value, which may be dependent on the position in the discretization
        domain

        Parameters
        ----------
        value : Number
            the value to calculate the probability density for
        position : Number, optional
            the position in the discretization domain at which the prior
            probability of the parameter ``value`` is to be retrieved
        
        Returns
        -------
        Number
            the log of the prior probability :math:`p(v) = \frac{1}{\sigma \sqrt{2 \pi}} 
            \exp \Big \lbrace -\frac{\left( v - \mu \right)^2}
            {2\sigma^2} \Big \rbrace`, where :math:`\mu` and :math:`\sigma`
            denote the (prior) mean and standard deviation of the 
            Gaussian parameter and :math:`v` the ``value`` at which the prior
            probability is to be retrieved. :math:`\mu` and :math:`\sigma` may
            be dependent on position in the discretization domain.
        """
        mean = self.get_mean(position)
        std = self.get_std(position)
        return -0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * ((value - mean) / std)**2


class CustomParameter(Parameter):
    """Class enabling the definition of an arbitrary prior for a free parameter

    Parameters
    ----------
    name : str
        name of the current parameter, for display and storing purposes
    log_prior : Callable[[Number, Number], Number]
    initialize : Callable[[np.ndarray], np.ndarray]
    perturb_std : Union[Number, np.ndarray]
        standard deviation of the Gaussians used to randomly perturb the parameter. 
        This can either be a scalar or an array if the defined probability distribution 
        is a function of ``position`` in the discretization domain
    position : np.ndarray, optional
        position in the discretization domain, used to define a position-dependent
        probability distribution. None by default
    """
    def __init__(
        self,
        name: str,
        log_prior: Callable[[Number, Number], Number],
        initialize: Callable[[np.ndarray], np.ndarray],
        perturb_std: Union[Number, np.ndarray], 
        position: np.ndarray = None, 
    ):
        super().__init__(
            name=name,
            log_prior=log_prior,
            initialize=initialize,
            perturb_std=perturb_std, 
            position=position, 
        )
        self.add_hyper_params({
            "perturb_std": perturb_std
        })
        self._log_prior = log_prior
        self._initialize = initialize
        
    def initialize(
        self, position: Union[np.ndarray, Number] = None
    ) -> Union[np.ndarray, Number]:
        r"""initialize the parameter, possibly at a specific position in the 
        discretization domain
        
        Parameters
        ----------
        position: Union[Number, np.ndarray], optional
            the position in the discretization domain at which the parameter is 
            initialized. None by default
        
        Returns
        -------
        Union[np.ndarray, Number]
            an array of values or one value corresponding to the given positions
        """
        return self._initialize(position)
    
    def perturb_value(self, 
                      value: Number, 
                      position: Union[Number, np.ndarray] = None
                      ) -> Tuple[Number, Number]:
        r"""perturbs the given value, in a way that may depend on the position 
        in the discretization domain, and calculates the log of the corresponding 
        partial acceptance probability,
        
        .. math::
            \underbrace{\alpha_{p}}_{\begin{array}{c} \text{Partial} \\ \text{acceptance} \\ \text{probability} \end{array}} = 
            \underbrace{\frac{p\left({\bf m'}\right)}{p\left({\bf m}\right)}}_{\text{Prior ratio}} 
            \underbrace{\frac{q\left({\bf m} \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}}_{\text{Proposal ratio}}  
            \underbrace{\lvert \mathbf{J} \rvert}_{\begin{array}{c} \text{Jacobian} \\ \text{determinant} \end{array}},

        where :math:`\bf m'` denotes the perturbed model as obtained through a 
        random deviate from a normal distribution :math:`\mathcal{N}(v, \theta)`, 
        where :math:`v` denotes the original ``value`` and :math:`\theta` the 
        standard deviation of the Gaussian used for the perturbation. 
        :math:`\theta` may be dependent on the specified position in the 
        discretization domain (:attr:`CustomParameter.perturb_std`).
        
        Parameters
        ----------
        value : Number
            the current value to be perturbed from
        position : Number, optional
            the position of the value to be perturbed

        Returns
        -------
        Tuple[Number, Number]
            the perturbed value and
            :math:`\alpha_{p} = \log(
            \frac{p({\bf m'})}{p({\bf m})}
            \frac{q\left({\bf m} 
            \mid {\bf m'}\right)}{q\left({\bf m'} \mid {\bf m}\right)}
            \lvert \mathbf{J} \rvert)`
        """
        perturb_std = self.get_perturb_std(position)
        random_deviate = random.normalvariate(0, perturb_std)
        new_value = value + random_deviate
        ratio = self.log_prior(new_value, position) - self.log_prior(value, position)
        return new_value, ratio

    def log_prior(self, 
                  value: Number, 
                  position: Union[Number, np.ndarray] = None) -> Number:
        r"""calculates the log of the prior probability density for the given 
        value, which may be dependent on the position in the discretization
        domain

        Parameters
        ----------
        value : Number
            the value to calculate the probability density for
        position : Number, optional
            the position in the discretization domain at which the prior
            probability of the parameter ``value`` is to be retrieved
        
        Returns
        -------
        Number
            the log prior probability density
        """
        try:
            return self._log_prior(value, position)
        except:
            return self._log_prior(value)
