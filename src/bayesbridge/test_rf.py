import sys
# sys.path.append('/home/fabrizio/Documents/GitHub/bayes-bridge/src')
sys.path.append('/home/jiawen/bayes-bridge/src')

import numpy as np
from espresso import ReceiverFunctionInversion

from bayesbridge.target import Target
from bayesbridge.parameters import UniformParameter, Parameterization1D
from bayesbridge.markov_chain import BayesianInversion


LAYERS_MIN = 3
LAYERS_MAX = 7
VS_VP_RATIO = 1.77
RF_DATA_STD = 0.01

rf_module = ReceiverFunctionInversion().rf

thickness = np.array([15, 20, 20, 20])
depths = np.cumsum(thickness)

vs = np.array([1.5, 3, 2.5, 4])
vp = vs * VS_VP_RATIO

model_setup = np.zeros((vs.shape[0], 3))
model_setup[:,0] = depths
model_setup[:,1] = vp
model_setup[:,2] = VS_VP_RATIO


t, rf_data = rf_module.rfcalc(model_setup)
rf_data += np.random.normal(0, RF_DATA_STD, rf_data.size)

def forward_rf(proposed_state):
    vp = proposed_state['vp']
    thickness = proposed_state['voronoi_cell_extents']
    depths = np.cumsum(thickness)
    model = np.array(model_setup)
    model[:,0] = depths
    model[:,1] = vp
    _, data = rf_module.rfcalc(model)
    return data

targets = [Target('vp', rf_data, dobs_covariance_mat=RF_DATA_STD)]
fwd_functions = [forward_rf]
free_parameters = [UniformParameter('vp', vmin=1, vmax=4.5, perturb_std=0.3, position=None)]

parameterization = Parameterization1D(voronoi_site_bounds=(0, 80),
                                      voronoi_site_perturb_std=3,
                                      n_voronoi_cells=None,
                                      n_voronoi_cells_min=LAYERS_MIN,
                                      n_voronoi_cells_max=LAYERS_MAX,
                                      free_params=free_parameters)

inversion = BayesianInversion(parameterization,
                              targets,
                              fwd_functions,
                              n_cpus=2,
                              n_chains=2)

inversion.run(n_iterations=2500, 
              burnin_iterations=500, 
              save_n_models=10,
              print_every=250)
saved_results = inversion.get_results(True)

