import math
import random
import numpy as np
import bayesbay as bb


uniform_param = bb.parameters.UniformParameter("uniform_param", -1, 1, 0.1)
gaussian_param = bb.parameters.GaussianParameter("gaussian_param", 0, 1, 0.1)
custom_param = bb.parameters.CustomParameter(
    name="custom_param",
    log_prior=lambda v: 0 if 0 <= v <= 1 else float("-inf"), 
    initialize=(lambda p: \
        np.random.uniform(0,1,len(p)) \
            if (not np.isscalar(p) and p is not None) \
                else random.uniform(0,1)), 
    perturb_std=1, 
)

my_ps = bb.parameterization.ParameterSpace(
    name="test", 
    n_dimensions=1, 
    parameters=[uniform_param, gaussian_param, custom_param], 
)

my_test_pss = bb.ParameterSpaceState(1, {
    "uniform_param": np.array([0,]), 
    "gaussian_param": np.array([0,]), 
    "custom_param": np.array([0.5,]), 
})

result = my_ps.log_prior(my_test_pss)
expected = math.log(1/2) + math.log(1/(math.sqrt(2*math.pi)*math.exp(0))) + 0
assert result == expected
