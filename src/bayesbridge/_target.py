import random
from copy import deepcopy
import numpy as np
from ._utils_bayes import inverse_covariance


class Target:
    def __init__(
        self,
        name,
        dobs,
        covariance_mat_inv=None,
        noise_is_correlated=False,
        sigma_min=0.01,
        sigma_max=1,
        sigma_perturb_std=0.1,
        correlation_min=0.01,
        correlation_max=1,
        correlation_perturb_std=0.1,
        save_dpred=True,
    ):
        self.name = name
        self.dobs = np.array(dobs)
        self.noise_is_correlated = noise_is_correlated
        self.sigma = None
        self.correlation = None
        self._save_dpred = save_dpred
        if covariance_mat_inv is None:
            self.sigma = {
                "name": "sigma",
                "min": sigma_min,
                "max": sigma_max,
                "perturb_std": sigma_perturb_std,
            }
            self._current_state = {"sigma": random.uniform(sigma_min, sigma_max)}
            self._perturbations = [self.sigma]
            if noise_is_correlated:
                self.correlation = {
                    "name": "correlation",
                    "min": correlation_min,
                    "max": correlation_max,
                    "perturb_std": correlation_perturb_std,
                }
                self._current_state.update(
                    {"correlation": random.uniform(correlation_min, correlation_max)}
                )
                self._perturbations.append(self.correlation)
            else:
                self._current_state["correlation"] = 0
            self._proposed_state = deepcopy(self._current_state)
            self._current_state[
                "determinant_covariance"
            ] = self.determinant_covariance()
        else:
            if np.isscalar(covariance_mat_inv):
                self.covariance_mat_inv = covariance_mat_inv
            else:
                self.covariance_mat_inv = np.array(covariance_mat_inv)

    @property
    def is_hierarchical(self):
        return not hasattr(self, "covariance_mat_inv")

    @property
    def save_dpred(self):
        return self._save_dpred

    def __repr__(self):
        string = "%s(" % self.name
        string += "dobs=%s, " % self.dobs
        if self.sigma is not None:
            string += "sigma=%s, " % self.sigma
            if self.noise_is_correlated:
                string += "correlation=%s" % self.correlation
        else:
            string += "covariance=%s" % self.covariance_mat_inv
        return string + ")"

    def perturb_covariance(self):
        to_be_perturbed = random.choice(self._perturbations)
        name = to_be_perturbed["name"]
        std = to_be_perturbed["perturb_std"]
        vmin, vmax = to_be_perturbed["min"], to_be_perturbed["max"]
        value = self._current_state[name]
        while True:
            random_deviate = random.normalvariate(0, std)
            new_value = value + random_deviate
            if not (new_value < vmin or new_value > vmax):
                self._proposed_state[name] = new_value
                break
        return 0

    def finalize_perturbation(self, accepted):
        accepted_state = self._proposed_state if accepted else self._current_state
        rejected_state = self._current_state if accepted else self._proposed_state
        for k, v in accepted_state.items():
            rejected_state[k] = v

    def covariance_times_vector(self, vector):
        if hasattr(self, "covariance_mat_inv"):
            if np.isscalar(self.covariance_mat_inv):
                return self.covariance_mat_inv * vector
            else:
                return self.covariance_mat_inv @ vector
        elif self.correlation is None:
            return 1 / self._proposed_state["sigma"] ** 2 * vector
        else:
            sigma = self._proposed_state["sigma"]
            r = self._proposed_state["correlation"]
            n = self.dobs.size
            mat = inverse_covariance(sigma, r, n)
            return mat @ vector

    def determinant_covariance(self):
        sigma = self._proposed_state["sigma"]
        r = self._proposed_state["correlation"]
        n = self.dobs.size
        det = sigma ** (2 * n) * (1 - r**2) ** (n - 1)
        self._proposed_state["determinant_covariance"] = det
        return det
